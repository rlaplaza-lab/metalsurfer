"""Sequential and multi-molecule saturation workflow entry points."""

import copy
import logging
import time
from dataclasses import replace
from typing import Any, NamedTuple

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..filters import adsorbate_connected_components
from ..ml.dataset import DatasetLogger
from ..models import (
    BOStepMemory,
    MultiMolSaturationRunResult,
    MultiMolSaturationStepResult,
    ReferenceEnergies,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
    merge_bo_step_memories,
    windowed_bo_step_memories,
)
from ..optimization import clear_autobatcher_cache
from ..placement.generators import (
    distribute_placement_budget,
    estimate_molecule_complexity,
)
from ..surface_prep import apply_material_pbc
from ..surfaces import SlabContainer, accept_substrate_for_api
from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from .bayesian import process_molecule_bayesian
from .core import process_molecule
from .screening import _setup_screening_run
from .shared import (
    _build_surface_reference_slab,
    _compute_slab_energy,
    resolve_saturation_step_workload_config,
)

logger = logging.getLogger(__name__)


def _slab_after_saturation_step(
    atoms: Atoms, config: AdsorptionConfig
) -> SlabContainer:
    """Build the next-step slab from a relaxed placement, restoring prep-time PBC."""
    slab_atoms = atoms.copy()
    apply_material_pbc(slab_atoms, config.material_type)
    return SlabContainer(slab_atoms)


def _saturation_symmetry_broken_vs_reference(
    current_atoms: Atoms,
    reference_atoms: Atoms,
    *,
    symmetry_tolerance: float,
    log_context: str,
) -> bool:
    """True if symmetry vs *reference_atoms* is broken or analysis fails (treat as C1)."""
    analyzer = SymmetryAnalyzer(current_atoms, symmetry_tolerance=symmetry_tolerance)
    try:
        broken = analyzer.detect_symmetry_breaking(reference_atoms)
    except SymmetryAnalysisError as exc:
        logger.warning(
            "%s: symmetry analysis unavailable (%s); assuming C1",
            log_context,
            exc,
        )
        return True
    if broken:
        logger.debug("%s: symmetry broken; using full site sampling", log_context)
    return broken


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    try:
        numeric = value.__float__  # type: ignore[attr-defined]
        return float(numeric())
    except (TypeError, ValueError, AttributeError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    try:
        integer = value.__int__  # type: ignore[attr-defined]
        return int(integer())
    except (TypeError, ValueError, AttributeError):
        return default


class _BoTransferStepFields(NamedTuple):
    bo_transfer_used: bool
    bo_transfer_disabled_reason: str | None
    bo_transfer_weight_share: float
    bo_transfer_bad_rounds: int
    bo_transfer_last_mae_delta: float | None


def _bo_transfer_fields_from_info(
    transfer_info: dict[str, object],
) -> _BoTransferStepFields:
    """Map BO transfer_info dict to step result fields (single- and multi-molecule paths)."""
    return _BoTransferStepFields(
        bo_transfer_used=bool(transfer_info.get("transfer_used", False)),
        bo_transfer_disabled_reason=(
            str(transfer_info["transfer_disabled_reason"])
            if transfer_info.get("transfer_disabled_reason") is not None
            else None
        ),
        bo_transfer_weight_share=float(
            _as_float(transfer_info.get("transfer_weight_share", 0.0))
        ),
        bo_transfer_bad_rounds=int(
            _as_int(transfer_info.get("transfer_bad_rounds", 0))
        ),
        bo_transfer_last_mae_delta=(
            _as_float(transfer_info["transfer_last_mae_delta"])
            if transfer_info.get("transfer_last_mae_delta") is not None
            else None
        ),
    )


def _bo_transfer_memory_in(
    config: AdsorptionConfig,
    *,
    prior_step_memories: list[BOStepMemory],
    prior_cumulative_memory: BOStepMemory | None,
) -> BOStepMemory | None:
    if not config.bo_transfer_enabled:
        return None
    if config.bo_transfer_mode == "cumulative_refit":
        return prior_cumulative_memory
    # Weighted transfer: windowed prior memory with recency and placement decay.
    return windowed_bo_step_memories(
        prior_step_memories,
        window=config.bo_transfer_prior_step_window,
    )


def _validate_distinct_bo_memories(
    bo_memories: dict[str, BOStepMemory | None],
    *,
    stage: str,
) -> None:
    """Fail fast if competing saturation tries to share BO state across adsorbates."""
    seen_by_id: dict[int, str] = {}
    for molecule, memory in bo_memories.items():
        if memory is None:
            continue
        other_molecule = seen_by_id.get(id(memory))
        if other_molecule is not None:
            raise RuntimeError(
                "Competing saturation BO state must remain independent per adsorbate "
                f"during {stage}; molecules {other_molecule!r} and {molecule!r} "
                "received the same BOStepMemory object"
            )
        seen_by_id[id(memory)] = molecule


def _n_at_saturation_from_steps(steps: list[Any]) -> int:
    if not steps:
        return 0
    last_step = steps[-1]
    return last_step.n_molecules_on_slab + (
        1 if last_step.best_result.energy_adsorption < 0 else 0
    )


def _reference_smiles_units_single_molecule(
    reference_smiles: str, step: int
) -> list[str]:
    """One SMILES per adsorbate unit expected on the slab after this step."""
    return [reference_smiles] * step


def _reference_smiles_units_multi_molecule(
    active_molecules: list[str],
    active_smiles: dict[str, str],
    molecule_counts: dict[str, int],
    placing_molecule: str,
) -> list[str]:
    """SMILES list for all adsorbate units after placing *placing_molecule* this step."""
    units: list[str] = []
    for mol in active_molecules:
        n = molecule_counts.get(mol, 0) + (1 if mol == placing_molecule else 0)
        units.extend([active_smiles[mol]] * n)
    return units


def _saturation_adsorbate_topology_ok(
    atoms: Atoms,
    *,
    base_slab_len: int,
    reference_unit_smiles: list[str],
    config: AdsorptionConfig,
) -> tuple[bool, str]:
    """Return whether the full adsorbate pool has the expected unit count.

    This guard is intentionally connectivity-only: it blocks adsorbate coupling
    (merged fragments) or unexpected splits while allowing strong
    adsorbate-material interactions that do not change adsorbate connectivity.
    """
    if config.skip_topology_check:
        return True, "topology checks disabled"

    components = adsorbate_connected_components(
        atoms,
        base_slab_len,
        config.connectivity_multipliers,
    )
    if len(components) != len(reference_unit_smiles):
        return (
            False,
            f"expected {len(reference_unit_smiles)} adsorbate units, "
            f"found {len(components)} connected fragments",
        )

    return True, "adsorbate connectivity intact"


def _filter_saturation_topology_results(
    results: list[ScreeningResult],
    *,
    base_slab_len: int,
    reference_unit_smiles: list[str],
    config: AdsorptionConfig,
) -> list[ScreeningResult]:
    """Drop candidates with adsorbate rearrangement before best-slab selection."""
    if not config.saturation_discard_topology_rearrangements:
        return results
    if not results:
        return results

    kept: list[ScreeningResult] = []
    discarded = 0
    for entry in results:
        ok, reason = _saturation_adsorbate_topology_ok(
            entry.atoms,
            base_slab_len=base_slab_len,
            reference_unit_smiles=reference_unit_smiles,
            config=config,
        )
        if ok:
            kept.append(entry)
        else:
            discarded += 1
            logger.debug(
                "Saturation topology guard (pid=%s): %s",
                entry.placement_id,
                reason,
            )

    if discarded:
        logger.info(
            "Saturation topology guard: kept %d/%d candidates (%d rearranged)",
            len(kept),
            len(results),
            discarded,
        )
    return kept


def _run_multi_molecule_saturation(
    smiles_list: list[str],
    molecules: list[str],
    base_slab: Atoms,
    calculator: object,
    ts_model: object,
    ref: ReferenceEnergies,
    config: AdsorptionConfig,
    surface_type: str,
    failure_summary_out: dict[str, object] | None,
    ds_logger: DatasetLogger,
) -> MultiMolSaturationRunResult:
    """Run a competitive multi-molecule saturation loop."""
    conformer_cache: dict[str, tuple[list[Atoms], list[float]]] = {}
    complexities: dict[str, float] = {}

    for smi, mol in zip(smiles_list, molecules, strict=True):
        result = create_conformers_from_smiles(
            smi,
            calculator=calculator,
            config=config,
            ts_model=ts_model,
        )
        if result is None:
            logger.warning(
                "Multi-mol saturation: could not generate conformers for %s; skipping this molecule",
                mol,
            )
            continue
        conformers, conformer_energies = result
        conformer_cache[mol] = (conformers, conformer_energies)

        complexities[mol] = estimate_molecule_complexity(
            conformers, base_slab, config, smi
        )

    active_smiles = {
        mol: smi
        for smi, mol in zip(smiles_list, molecules, strict=True)
        if mol in conformer_cache
    }
    active_molecules = list(active_smiles)

    if not active_molecules:
        logger.error(
            "Multi-mol saturation: no molecules with valid conformers; aborting"
        )
        return MultiMolSaturationRunResult(
            molecules=molecules,
            steps=[],
            n_molecules_at_saturation=0,
            final_slab_atoms=base_slab.copy(),
            molecule_counts={},
        )

    logger.info(
        "Multi-mol saturation: %d active molecules %s | complexities: %s",
        len(active_molecules),
        active_molecules,
        {m: round(c) for m, c in complexities.items()},
    )

    largest_mol = max(
        active_molecules,
        key=lambda m: len(conformer_cache[m][0][0]) if conformer_cache[m][0] else 0,
    )

    current_slab = SlabContainer(base_slab.copy())
    reference_slab_for_symmetry = base_slab.copy()
    symmetry_broken = False
    step = 0
    steps: list[MultiMolSaturationStepResult] = []
    molecule_counts: dict[str, int] = {mol: 0 for mol in active_molecules}
    bo_memory_per_mol: dict[str, BOStepMemory | None] = {
        mol: None for mol in active_molecules
    }
    bo_step_memories_per_mol: dict[str, list[BOStepMemory]] = {
        mol: [] for mol in active_molecules
    }
    bo_cumulative_per_mol: dict[str, BOStepMemory | None] = {
        mol: None for mol in active_molecules
    }

    while True:
        step += 1

        if config.bo_enabled:
            _validate_distinct_bo_memories(
                bo_memory_per_mol,
                stage=f"step {step} input",
            )

        if step > 1 and not config.saturation_autobatcher_reuse:
            clear_autobatcher_cache()

        n_on_slab = step - 1
        logger.info(
            "Multi-mol saturation step %d (molecules on slab: %d)",
            step,
            n_on_slab,
        )

        if step > 1 and not symmetry_broken:
            symmetry_broken = _saturation_symmetry_broken_vs_reference(
                current_slab.atoms,
                reference_slab_for_symmetry,
                symmetry_tolerance=config.symmetry_tolerance,
                log_context=f"Multi-mol saturation step {step}",
            )

        E_slab = _compute_slab_energy(
            current_slab.atoms,
            calculator,
            label=f"multi-mol saturation step {step} slab",
        )
        ref_step = ReferenceEnergies(
            slab_energy=E_slab,
            molecule_energies=ref.molecule_energies,
        )

        needs_workload_autotune = config.num_placements is None or (
            config.bo_enabled
            and (config.bo_initial_random is None or config.bo_batch_size is None)
        )
        if needs_workload_autotune:
            largest_conformers, _ = conformer_cache[largest_mol]
            slab_for_sites = _build_surface_reference_slab(
                current_slab.atoms, base_slab
            )
            step_config = resolve_saturation_step_workload_config(
                config,
                ts_model=ts_model,
                conformers=largest_conformers,
                slab_atoms=current_slab.atoms,
                slab_for_sites=slab_for_sites,
                smiles=active_smiles[largest_mol],
                base_slab_for_frozen=base_slab,
                symmetry_broken=symmetry_broken,
                bo_enabled=config.bo_enabled,
            )
        else:
            step_config = config

        assert step_config.num_placements is not None
        budgets = distribute_placement_budget(
            {m: complexities[m] for m in active_molecules},
            step_config.num_placements,
        )
        logger.info("Step %d placement budgets: %s", step, budgets)

        per_molecule_results: dict[str, list[ScreeningResult]] = {}
        per_molecule_bo_transfer: dict[str, dict[str, object]] = {}
        new_bo_memory_raw: dict[str, BOStepMemory | None] = {}

        for mol in active_molecules:
            smi = active_smiles[mol]
            mol_budget = budgets[mol]
            mol_config = replace(step_config, num_placements=mol_budget)

            transfer_info: dict[str, object] = {}
            if config.bo_enabled:
                bo_out: dict[str, BOStepMemory] = {}
                mol_results = process_molecule_bayesian(
                    smi,
                    mol,
                    current_slab,
                    calculator,
                    ref_step,
                    ts_model=ts_model,
                    config=mol_config,
                    surface_type=surface_type,
                    reference_smiles=smi,
                    base_slab_for_frozen=base_slab,
                    slab_energy_override=E_slab,
                    failure_summary_out=failure_summary_out,
                    symmetry_broken=symmetry_broken,
                    bo_step_memory_in=_bo_transfer_memory_in(
                        config,
                        prior_step_memories=bo_step_memories_per_mol[mol],
                        prior_cumulative_memory=bo_cumulative_per_mol[mol],
                    ),
                    bo_prior_step_memory=bo_memory_per_mol[mol],
                    bo_step_memory_out=bo_out,
                    bo_transfer_info_out=transfer_info,
                )
                new_bo_memory_raw[mol] = bo_out.get("memory")
            else:
                mol_results = process_molecule(
                    smi,
                    mol,
                    current_slab,
                    calculator,
                    ref_step,
                    ts_model=ts_model,
                    config=mol_config,
                    surface_type=surface_type,
                    reference_smiles=smi,
                    base_slab_for_frozen=base_slab,
                    slab_energy_override=E_slab,
                    failure_summary_out=failure_summary_out,
                    saturation_reuse=True,
                    symmetry_broken=symmetry_broken,
                )
                new_bo_memory_raw[mol] = None

            per_molecule_bo_transfer[mol] = transfer_info

            resolved = list(mol_results) if mol_results else []
            resolved = _filter_saturation_topology_results(
                resolved,
                base_slab_len=len(base_slab),
                reference_unit_smiles=_reference_smiles_units_multi_molecule(
                    active_molecules,
                    active_smiles,
                    molecule_counts,
                    mol,
                ),
                config=config,
            )
            per_molecule_results[mol] = resolved
            if resolved:
                ds_logger.add_results(resolved, smiles=smi, surface_id=surface_type)
                best_mol = min(resolved, key=lambda r: r.energy_adsorption)
                logger.info(
                    "Step %d | %s: best E_ads = %.4f eV (%d results)",
                    step,
                    mol,
                    best_mol.energy_adsorption,
                    len(resolved),
                )
            else:
                logger.warning("Step %d | %s: no valid placements", step, mol)

        if config.bo_enabled:
            _validate_distinct_bo_memories(
                new_bo_memory_raw,
                stage=f"step {step} output",
            )

        bo_memory_per_mol = {
            mol: copy.deepcopy(memory) for mol, memory in new_bo_memory_raw.items()
        }
        for mol in active_molecules:
            step_memory = bo_memory_per_mol.get(mol)
            if step_memory is not None:
                bo_step_memories_per_mol[mol].append(copy.deepcopy(step_memory))
        if config.bo_transfer_enabled:
            bo_cumulative_per_mol = {
                mol: merge_bo_step_memories(
                    [bo_cumulative_per_mol[mol], bo_memory_per_mol[mol]]
                )
                for mol in active_molecules
            }

        if not any(per_molecule_results.values()):
            logger.warning(
                "Multi-mol saturation step %d: no valid placements for any molecule "
                "(including after topology rearrangement guard); stopping",
                step,
            )
            break

        all_results_flat = [
            r for results in per_molecule_results.values() for r in results
        ]
        best_overall = min(all_results_flat, key=lambda r: r.energy_adsorption)
        winning_molecule = best_overall.molecule

        bo_transfer_used: dict[str, bool] = {}
        bo_transfer_disabled_reason: dict[str, str | None] = {}
        bo_transfer_weight_share: dict[str, float] = {}
        bo_transfer_bad_rounds: dict[str, int] = {}
        bo_transfer_last_mae_delta: dict[str, float | None] = {}
        for mol in active_molecules:
            f = _bo_transfer_fields_from_info(per_molecule_bo_transfer[mol])
            bo_transfer_used[mol] = f.bo_transfer_used
            bo_transfer_disabled_reason[mol] = f.bo_transfer_disabled_reason
            bo_transfer_weight_share[mol] = f.bo_transfer_weight_share
            bo_transfer_bad_rounds[mol] = f.bo_transfer_bad_rounds
            bo_transfer_last_mae_delta[mol] = f.bo_transfer_last_mae_delta

        steps.append(
            MultiMolSaturationStepResult(
                step=step,
                winning_molecule=winning_molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=best_overall,
                per_molecule_results=per_molecule_results,
                per_molecule_budgets=dict(budgets),
                bo_transfer_enabled=bool(config.bo_transfer_enabled),
                bo_transfer_used=bo_transfer_used,
                bo_transfer_disabled_reason=bo_transfer_disabled_reason,
                bo_transfer_weight_share=bo_transfer_weight_share,
                bo_transfer_bad_rounds=bo_transfer_bad_rounds,
                bo_transfer_last_mae_delta=bo_transfer_last_mae_delta,
            )
        )
        molecule_counts[winning_molecule] += 1

        logger.info(
            "Step %d: winner = %s, E_ads = %.4f eV",
            step,
            winning_molecule,
            best_overall.energy_adsorption,
        )

        if best_overall.energy_adsorption >= 0:
            logger.info(
                "Multi-mol saturation: slab saturated at step %d (E_ads >= 0)",
                step,
            )
            break

        if (
            config.saturation_max_steps is not None
            and step >= config.saturation_max_steps
        ):
            logger.info(
                "Multi-mol saturation: reached max steps (%d)",
                config.saturation_max_steps,
            )
            break

        current_slab = _slab_after_saturation_step(best_overall.atoms, config)

    return MultiMolSaturationRunResult(
        molecules=molecules,
        steps=steps,
        n_molecules_at_saturation=_n_at_saturation_from_steps(steps),
        final_slab_atoms=current_slab.atoms.copy(),
        molecule_counts=molecule_counts,
    )


def run_saturation_screening(
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | tuple[str, str] | str = "smiles.csv",
    smiles_file: str | None = None,
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    failure_summary_out: dict[str, object] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[SaturationRunResult] | list[MultiMolSaturationRunResult]:
    """Sequential saturation: add molecules until best E_ads >= 0."""
    if config is None:
        config = AdsorptionConfig()

    slab = accept_substrate_for_api(slab, config=config)
    molecules_input = smiles_file if smiles_file is not None else molecules

    t_run_start = time.perf_counter()

    with log_context(surface_type=surface_type, seed=config.seed):
        setup = _setup_screening_run(
            slab,
            molecules_input,
            config,
            surface_type,
            skip_existing,
            skip_saturation_file=skip_existing,
        )
        if setup is None:
            return []

        calculator, ts_model, molecule_names, smiles_list, ref, t_ref_s = setup
        base_slab = slab.atoms.copy()
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)

        if config.multi_molecule_saturation and len(molecule_names) > 1:
            logger.info(
                "Multi-molecule saturation enabled: %d molecules competing per step",
                len(molecule_names),
            )
            multi_result = _run_multi_molecule_saturation(
                smiles_list=smiles_list,
                molecules=molecule_names,
                base_slab=base_slab,
                calculator=calculator,
                ts_model=ts_model,
                ref=ref,
                config=config,
                surface_type=surface_type,
                failure_summary_out=failure_summary_out,
                ds_logger=ds_logger,
            )
            ds_logger.flush()
            t_run_total = time.perf_counter() - t_run_start
            total_steps = len(multi_result.steps)
            total_configs = sum(
                len(r)
                for s in multi_result.steps
                for r in s.per_molecule_results.values()
            )
            logger.info(
                "Multi-mol saturation complete: %d molecules, %d steps, %.1fs",
                len(molecule_names),
                total_steps,
                t_run_total,
            )
            if run_metadata_out is not None:
                run_metadata_out.update(
                    n_molecules=len(molecule_names),
                    total_configs=total_configs,
                    t_ref_s=t_ref_s,
                    t_total_s=t_run_total,
                )
            return [multi_result]

        if config.multi_molecule_saturation and len(molecule_names) == 1:
            logger.warning(
                "multi_molecule_saturation=True but only one molecule provided; falling back to standard single-molecule saturation"
            )

        all_saturation_results: list[SaturationRunResult] = []
        for smi, mol in zip(smiles_list, molecule_names, strict=True):
            E_mol = ref.get_molecule_energy(mol)
            if E_mol is None:
                if config.fail_on_missing_reference:
                    raise ValueError(
                        f"No reference energy for {mol}; cannot continue with fail_on_missing_reference=True"
                    )
                logger.warning("Skipping %s: no reference energy", mol)
                continue

            current_slab = SlabContainer(base_slab.copy())
            steps: list[SaturationStepResult] = []
            step = 0
            prior_step_memory: BOStepMemory | None = None
            prior_step_memories: list[BOStepMemory] = []
            prior_cumulative_memory: BOStepMemory | None = None

            reference_slab_for_symmetry = base_slab.copy()
            symmetry_broken = False

            while True:
                step += 1
                if step > 1 and not config.saturation_autobatcher_reuse:
                    clear_autobatcher_cache()
                n_on_slab = 0 if step == 1 else step - 1
                logger.info(
                    "Saturation step %d for %s (n_molecules on slab: %d)",
                    step,
                    mol,
                    n_on_slab,
                )

                if step > 1 and not symmetry_broken:
                    symmetry_broken = _saturation_symmetry_broken_vs_reference(
                        current_slab.atoms,
                        reference_slab_for_symmetry,
                        symmetry_tolerance=config.symmetry_tolerance,
                        log_context=f"Saturation step {step} for {mol}",
                    )

                E_slab = _compute_slab_energy(
                    current_slab.atoms,
                    calculator,
                    label=f"saturation step {step} slab",
                )

                ref_step = ReferenceEnergies(
                    slab_energy=E_slab,
                    molecule_energies=ref.molecule_energies,
                )

                transfer_info: dict[str, object] = {}
                if config.bo_enabled:
                    bo_memory_out: dict[str, BOStepMemory] = {}
                    mol_results = process_molecule_bayesian(
                        smi,
                        mol,
                        current_slab,
                        calculator,
                        ref_step,
                        ts_model=ts_model,
                        config=config,
                        surface_type=surface_type,
                        reference_smiles=smi,
                        base_slab_for_frozen=base_slab,
                        slab_energy_override=E_slab,
                        failure_summary_out=failure_summary_out,
                        symmetry_broken=symmetry_broken,
                        bo_step_memory_in=_bo_transfer_memory_in(
                            config,
                            prior_step_memories=prior_step_memories,
                            prior_cumulative_memory=prior_cumulative_memory,
                        ),
                        bo_prior_step_memory=prior_step_memory,
                        bo_step_memory_out=bo_memory_out,
                        bo_transfer_info_out=transfer_info,
                    )
                    prior_step_memory = copy.deepcopy(bo_memory_out.get("memory"))
                    if prior_step_memory is not None:
                        prior_step_memories.append(copy.deepcopy(prior_step_memory))
                    if config.bo_transfer_enabled:
                        prior_cumulative_memory = merge_bo_step_memories(
                            [prior_cumulative_memory, prior_step_memory]
                        )
                else:
                    mol_results = process_molecule(
                        smi,
                        mol,
                        current_slab,
                        calculator,
                        ref_step,
                        ts_model=ts_model,
                        config=config,
                        surface_type=surface_type,
                        reference_smiles=smi,
                        base_slab_for_frozen=base_slab,
                        slab_energy_override=E_slab,
                        failure_summary_out=failure_summary_out,
                        saturation_reuse=True,
                        symmetry_broken=symmetry_broken,
                    )

                mol_results = _filter_saturation_topology_results(
                    list(mol_results) if mol_results else [],
                    base_slab_len=len(base_slab),
                    reference_unit_smiles=_reference_smiles_units_single_molecule(
                        smi, step
                    ),
                    config=config,
                )

                if not mol_results:
                    logger.warning(
                        "Step %d: no valid placements for %s "
                        "(including after topology rearrangement guard); stopping saturation",
                        step,
                        mol,
                    )
                    break

                best = min(mol_results, key=lambda r: r.energy_adsorption)
                bt = _bo_transfer_fields_from_info(transfer_info)
                steps.append(
                    SaturationStepResult(
                        step=step,
                        molecule=mol,
                        n_molecules_on_slab=n_on_slab,
                        best_result=best,
                        all_results=mol_results,
                        bo_transfer_enabled=bool(config.bo_transfer_enabled),
                        bo_transfer_used=bt.bo_transfer_used,
                        bo_transfer_disabled_reason=bt.bo_transfer_disabled_reason,
                        bo_transfer_weight_share=bt.bo_transfer_weight_share,
                        bo_transfer_bad_rounds=bt.bo_transfer_bad_rounds,
                        bo_transfer_last_mae_delta=bt.bo_transfer_last_mae_delta,
                    )
                )
                ds_logger.add_results(mol_results, smiles=smi, surface_id=surface_type)

                logger.info(
                    "Step %d: best E_ads = %.4f eV (placement %d)",
                    step,
                    best.energy_adsorption,
                    best.placement_id,
                )

                if best.energy_adsorption >= 0:
                    logger.info(
                        "Slab saturated for %s at step %d (E_ads >= 0)",
                        mol,
                        step,
                    )
                    break

                if (
                    config.saturation_max_steps is not None
                    and step >= config.saturation_max_steps
                ):
                    logger.info(
                        "Saturation for %s: reached max steps (%d)",
                        mol,
                        config.saturation_max_steps,
                    )
                    break

                current_slab = _slab_after_saturation_step(best.atoms, config)

            if steps:
                all_saturation_results.append(
                    SaturationRunResult(
                        molecule=mol,
                        steps=steps,
                        n_molecules_at_saturation=_n_at_saturation_from_steps(steps),
                        final_slab_atoms=current_slab.atoms.copy(),
                    )
                )

        ds_logger.flush()

    t_run_total = time.perf_counter() - t_run_start
    total_steps = sum(len(sr.steps) for sr in all_saturation_results)
    total_configs = sum(
        len(s.all_results) for sr in all_saturation_results for s in sr.steps
    )
    logger.info(
        "Saturation screening complete: %d molecules, %d total steps, %.1fs",
        len(molecule_names),
        total_steps,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecule_names),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_saturation_results

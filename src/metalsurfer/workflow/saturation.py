"""Sequential and multi-molecule saturation workflow entry points."""

import copy
import dataclasses as _dc
import logging
import time
from typing import Any

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..ml.dataset import DatasetLogger
from ..models import (
    BOStepMemory,
    MultiMolSaturationRunResult,
    MultiMolSaturationStepResult,
    ReferenceEnergies,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
)
from ..optimization import clear_autobatcher_cache
from ..placement import distribute_placement_budget
from ..surfaces import SlabContainer, coerce_slab_container
from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from .bayesian import process_molecule_bayesian
from .core import process_molecule
from .screening import _setup_screening_run
from .shared import _compute_slab_energy

logger = logging.getLogger(__name__)


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

        from ..placement import generators as _gen

        complexities[mol] = _gen.estimate_molecule_complexity(
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
            symmetry_analyzer = SymmetryAnalyzer(
                current_slab.atoms,
                symmetry_tolerance=config.symmetry_tolerance,
            )
            try:
                symmetry_broken = symmetry_analyzer.detect_symmetry_breaking(
                    reference_slab_for_symmetry
                )
            except SymmetryAnalysisError as exc:
                logger.warning(
                    "Multi-mol saturation step %d: symmetry analysis unavailable (%s); assuming C1",
                    step,
                    exc,
                )
                symmetry_broken = True
            if symmetry_broken:
                logger.info(
                    "Multi-mol saturation step %d: symmetry broken, switching to comprehensive site sampling",
                    step,
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

        budgets = distribute_placement_budget(
            {m: complexities[m] for m in active_molecules},
            config.num_placements,
        )
        logger.info("Step %d placement budgets: %s", step, budgets)

        per_molecule_results: dict[str, list[ScreeningResult]] = {}
        per_molecule_bo_transfer: dict[str, dict[str, object]] = {}
        new_bo_memory_raw: dict[str, BOStepMemory | None] = {}

        for mol in active_molecules:
            smi = active_smiles[mol]
            mol_budget = budgets[mol]
            mol_config = _dc.replace(config, num_placements=mol_budget)

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
                    bo_step_memory_in=bo_memory_per_mol[mol],
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
                    allow_auto_resize=(step == 1 and mol == largest_mol),
                )
                new_bo_memory_raw[mol] = None

            per_molecule_bo_transfer[mol] = transfer_info

            resolved = list(mol_results) if mol_results else []
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

        if not any(per_molecule_results.values()):
            logger.warning(
                "Multi-mol saturation step %d: no valid placements for any molecule; stopping",
                step,
            )
            break

        all_results_flat = [
            r for results in per_molecule_results.values() for r in results
        ]
        best_overall = min(all_results_flat, key=lambda r: r.energy_adsorption)
        winning_molecule = best_overall.molecule

        bo_transfer_used = {
            mol: bool(per_molecule_bo_transfer[mol].get("transfer_used", False))
            for mol in active_molecules
        }
        bo_transfer_disabled_reason: dict[str, str | None] = {
            mol: (
                str(per_molecule_bo_transfer[mol]["transfer_disabled_reason"])
                if per_molecule_bo_transfer[mol].get("transfer_disabled_reason")
                is not None
                else None
            )
            for mol in active_molecules
        }
        bo_transfer_weight_share = {
            mol: _as_float(
                per_molecule_bo_transfer[mol].get("transfer_weight_share", 0.0)
            )
            for mol in active_molecules
        }
        bo_transfer_bad_rounds = {
            mol: _as_int(per_molecule_bo_transfer[mol].get("transfer_bad_rounds", 0))
            for mol in active_molecules
        }
        bo_transfer_last_mae_delta: dict[str, float | None] = {
            mol: (
                _as_float(per_molecule_bo_transfer[mol]["transfer_last_mae_delta"])
                if per_molecule_bo_transfer[mol].get("transfer_last_mae_delta")
                is not None
                else None
            )
            for mol in active_molecules
        }

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

        current_slab = SlabContainer(best_overall.atoms.copy())

    if steps:
        last_step = steps[-1]
        n_at_saturation = last_step.n_molecules_on_slab + (
            1 if last_step.best_result.energy_adsorption < 0 else 0
        )
    else:
        n_at_saturation = 0

    return MultiMolSaturationRunResult(
        molecules=molecules,
        steps=steps,
        n_molecules_at_saturation=n_at_saturation,
        final_slab_atoms=current_slab.atoms.copy(),
        molecule_counts=molecule_counts,
    )


def run_saturation_screening(
    slab: SlabContainer | Atoms,
    smiles_file: str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    failure_summary_out: dict[str, object] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[SaturationRunResult] | list[MultiMolSaturationRunResult]:
    """Sequential saturation: add molecules until best E_ads >= 0."""
    if config is None:
        config = AdsorptionConfig()

    slab = coerce_slab_container(slab)

    t_run_start = time.perf_counter()

    with log_context(surface_type=surface_type, seed=config.seed):
        setup = _setup_screening_run(
            slab,
            smiles_file,
            config,
            surface_type,
            skip_existing,
            skip_saturation_file=skip_existing,
        )
        if setup is None:
            return []

        calculator, ts_model, molecules, smiles_list, ref, t_ref_s = setup
        base_slab = slab.atoms.copy()
        base_slab.set_pbc([True, True, True])
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)

        if config.multi_molecule_saturation and len(molecules) > 1:
            logger.info(
                "Multi-molecule saturation enabled: %d molecules competing per step",
                len(molecules),
            )
            multi_result = _run_multi_molecule_saturation(
                smiles_list=smiles_list,
                molecules=molecules,
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
                len(molecules),
                total_steps,
                t_run_total,
            )
            if run_metadata_out is not None:
                run_metadata_out.update(
                    n_molecules=len(molecules),
                    total_configs=total_configs,
                    t_ref_s=t_ref_s,
                    t_total_s=t_run_total,
                )
            return [multi_result]

        elif config.multi_molecule_saturation and len(molecules) == 1:
            logger.warning(
                "multi_molecule_saturation=True but only one molecule provided; falling back to standard single-molecule saturation"
            )

        all_saturation_results: list[SaturationRunResult] = []
        for smi, mol in zip(smiles_list, molecules, strict=True):
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
                    symmetry_analyzer = SymmetryAnalyzer(
                        current_slab.atoms,
                        symmetry_tolerance=config.symmetry_tolerance,
                    )
                    try:
                        symmetry_broken = symmetry_analyzer.detect_symmetry_breaking(
                            reference_slab_for_symmetry
                        )
                    except SymmetryAnalysisError as exc:
                        logger.warning(
                            "Saturation step %d for %s: symmetry analysis unavailable (%s); assuming C1 and switching to comprehensive site sampling",
                            step,
                            mol,
                            exc,
                        )
                        symmetry_broken = True
                    if symmetry_broken:
                        logger.info(
                            "Saturation step %d: Detected symmetry breaking, switching to comprehensive site sampling",
                            step,
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
                        bo_step_memory_in=prior_step_memory,
                        bo_step_memory_out=bo_memory_out,
                        bo_transfer_info_out=transfer_info,
                    )
                    prior_step_memory = copy.deepcopy(bo_memory_out.get("memory"))
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
                        allow_auto_resize=(step == 1),
                    )

                if not mol_results:
                    logger.warning(
                        "Step %d: no valid placements for %s; stopping saturation",
                        step,
                        mol,
                    )
                    break

                best = min(mol_results, key=lambda r: r.energy_adsorption)
                steps.append(
                    SaturationStepResult(
                        step=step,
                        molecule=mol,
                        n_molecules_on_slab=n_on_slab,
                        best_result=best,
                        all_results=mol_results,
                        bo_transfer_enabled=bool(config.bo_transfer_enabled),
                        bo_transfer_used=bool(
                            transfer_info.get("transfer_used", False)
                        ),
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

                current_slab = SlabContainer(best.atoms.copy())

            if steps:
                last_step = steps[-1]
                n_at_saturation = last_step.n_molecules_on_slab + (
                    1 if last_step.best_result.energy_adsorption < 0 else 0
                )
                final_atoms = current_slab.atoms.copy()
                all_saturation_results.append(
                    SaturationRunResult(
                        molecule=mol,
                        steps=steps,
                        n_molecules_at_saturation=n_at_saturation,
                        final_slab_atoms=final_atoms,
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
        len(molecules),
        total_steps,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecules),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_saturation_results

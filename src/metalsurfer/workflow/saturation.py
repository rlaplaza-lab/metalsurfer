"""Sequential and multi-molecule saturation workflow entry points."""

import copy
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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
from ..surface_prep import SlabContainer, apply_material_pbc
from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from .bayesian import BOTransferInfo, process_molecule_bayesian
from .core import process_molecule
from .shared import (
    MoleculeScreenOutcome,
    _bootstrap_screening_run,
    _build_surface_reference_slab,
    _compute_slab_energy,
    _normalize_molecules_input,
    needs_workload_autotune,
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
) -> bool:
    """True if symmetry vs *reference_atoms* is broken or analysis fails (treat as C1)."""
    analyzer = SymmetryAnalyzer(current_atoms, symmetry_tolerance=symmetry_tolerance)
    try:
        broken = analyzer.detect_symmetry_breaking(reference_atoms)
    except SymmetryAnalysisError as exc:
        logger.warning(
            "Symmetry analysis unavailable (%s); assuming C1",
            exc,
        )
        return True
    if broken:
        logger.debug("Symmetry broken; using full site sampling")
    return broken


@dataclass
class _BoMemoryState:
    """Per-adsorbate BO memory carried across saturation steps."""

    prior_step_memory: BOStepMemory | None = None
    prior_step_memories: list[BOStepMemory] = field(default_factory=list)
    prior_cumulative_memory: BOStepMemory | None = None


def _bo_transfer_memory_in(
    config: AdsorptionConfig,
    state: _BoMemoryState,
) -> BOStepMemory | None:
    if not config.bo.transfer.enabled:
        return None
    if config.bo.transfer.mode == "cumulative_refit":
        return state.prior_cumulative_memory
    return windowed_bo_step_memories(
        state.prior_step_memories,
        window=config.bo.transfer.prior_step_window,
    )


def _commit_bo_memory_state(
    state: _BoMemoryState,
    new_memory: BOStepMemory | None,
    *,
    config: AdsorptionConfig,
) -> None:
    """Advance per-adsorbate BO memory after a screening call."""
    if new_memory is None:
        state.prior_step_memory = None
    else:
        committed = copy.deepcopy(new_memory)
        state.prior_step_memory = committed
        state.prior_step_memories.append(committed)
    if config.bo.transfer.enabled:
        state.prior_cumulative_memory = merge_bo_step_memories(
            [state.prior_cumulative_memory, state.prior_step_memory]
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


class _SaturationStepPreamble(NamedTuple):
    symmetry_broken: bool
    E_slab: float
    ref_step: ReferenceEnergies


def _saturation_step_preamble(
    *,
    step: int,
    current_slab: SlabContainer,
    reference_slab_for_symmetry: Atoms,
    symmetry_broken: bool,
    calculator: object,
    ref: ReferenceEnergies,
    config: AdsorptionConfig,
    log_label: str,
) -> _SaturationStepPreamble:
    """Shared per-step setup before molecule screening."""
    if step > 1 and not config.saturation_autobatcher_reuse:
        clear_autobatcher_cache()

    if step > 1 and not symmetry_broken:
        symmetry_broken = _saturation_symmetry_broken_vs_reference(
            current_slab.atoms,
            reference_slab_for_symmetry,
            symmetry_tolerance=config.symmetry_tolerance,
        )

    E_slab = _compute_slab_energy(
        current_slab.atoms,
        calculator,
        label=f"{log_label} slab",
    )
    ref_step = ReferenceEnergies(
        slab_energy=E_slab,
        molecule_energies=ref.molecule_energies,
    )
    return _SaturationStepPreamble(symmetry_broken, E_slab, ref_step)


def _screen_saturation_molecule(
    *,
    smiles: str,
    molecule_name: str,
    current_slab: SlabContainer,
    calculator: object,
    ref_step: ReferenceEnergies,
    ts_model: object,
    config: AdsorptionConfig,
    surface_type: str,
    base_slab: Atoms,
    E_slab: float,
    failure_summary_out: dict[str, object] | None,
    symmetry_broken: bool,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_state: _BoMemoryState | None,
    reference_unit_smiles: list[str],
    conformers: list[Atoms] | None = None,
    skip_workload_autotune: bool = False,
) -> tuple[list[ScreeningResult], BOTransferInfo, BOStepMemory | None]:
    """Run one molecule's place/opt/filter for a saturation step."""
    common_kwargs: dict[str, Any] = {
        "ts_model": ts_model,
        "config": config,
        "surface_type": surface_type,
        "reference_smiles": smiles,
        "base_slab_for_frozen": base_slab,
        "slab_energy_override": E_slab,
        "symmetry_broken": symmetry_broken,
        "conformers": conformers,
        "skip_workload_autotune": skip_workload_autotune,
    }
    if process_fn is process_molecule_bayesian:
        if bo_state is None:
            raise ValueError("bo_state must be provided for Bayesian processing")
        outcome = process_fn(
            smiles,
            molecule_name,
            current_slab,
            calculator,
            ref_step,
            bo_step_memory_in=_bo_transfer_memory_in(config, bo_state),
            bo_prior_step_memory=bo_state.prior_step_memory,
            **common_kwargs,
        )
        transfer_info = outcome.transfer_info or BOTransferInfo()
        new_memory = outcome.bo_memory
    else:
        outcome = process_fn(
            smiles,
            molecule_name,
            current_slab,
            calculator,
            ref_step,
            saturation_reuse=True,
            **common_kwargs,
        )
        transfer_info = BOTransferInfo()
        new_memory = None

    if failure_summary_out is not None and outcome.failure_summary:
        failure_summary_out.clear()
        failure_summary_out.update(outcome.failure_summary)

    filtered = _filter_saturation_topology_results(
        list(outcome.results) if outcome.results else [],
        base_slab_len=len(base_slab),
        reference_unit_smiles=reference_unit_smiles,
        config=config,
    )
    return filtered, transfer_info, new_memory


def _saturation_should_stop(
    *,
    best_energy: float,
    step: int,
    config: AdsorptionConfig,
    log_prefix: str,
) -> bool:
    if best_energy >= 0:
        logger.info("%s: slab saturated at step %d (E_ads >= 0)", log_prefix, step)
        return True
    if config.saturation_max_steps is not None and step >= config.saturation_max_steps:
        logger.info(
            "%s: reached max steps (%d)",
            log_prefix,
            config.saturation_max_steps,
        )
        return True
    return False


@dataclass(frozen=True)
class _StepScreenOutcome:
    """Result of one saturation step's screening phase."""

    best: ScreeningResult
    payload: Any = None


def _run_saturation_steps(
    *,
    config: AdsorptionConfig,
    current_slab: SlabContainer,
    reference_slab_for_symmetry: Atoms,
    calculator: object,
    ref: ReferenceEnergies,
    log_prefix: str,
    log_step_start: Callable[[int, int], None],
    make_log_label: Callable[[int], str],
    screen_step: Callable[
        [int, _SaturationStepPreamble, bool, SlabContainer],
        _StepScreenOutcome | None,
    ],
    record_step: Callable[[int, int, _StepScreenOutcome], None],
) -> Atoms:
    """Shared coverage loop; single/multi differ only via screen/record callbacks."""
    symmetry_broken = False
    step = 0
    while True:
        step += 1
        n_on_slab = step - 1
        log_step_start(step, n_on_slab)

        preamble = _saturation_step_preamble(
            step=step,
            current_slab=current_slab,
            reference_slab_for_symmetry=reference_slab_for_symmetry,
            symmetry_broken=symmetry_broken,
            calculator=calculator,
            ref=ref,
            config=config,
            log_label=make_log_label(step),
        )
        symmetry_broken = preamble.symmetry_broken

        outcome = screen_step(step, preamble, symmetry_broken, current_slab)
        if outcome is None:
            break

        record_step(step, n_on_slab, outcome)

        if _saturation_should_stop(
            best_energy=outcome.best.energy_adsorption,
            step=step,
            config=config,
            log_prefix=log_prefix,
        ):
            break

        current_slab = _slab_after_saturation_step(outcome.best.atoms, config)

    return current_slab.atoms.copy()


def _run_single_molecule_saturation(
    *,
    smiles: str,
    molecule: str,
    base_slab: Atoms,
    calculator: object,
    ts_model: object,
    ref: ReferenceEnergies,
    config: AdsorptionConfig,
    surface_type: str,
    failure_summary_out: dict[str, object] | None,
    ds_logger: DatasetLogger,
    process_fn: Callable[..., MoleculeScreenOutcome],
) -> SaturationRunResult | None:
    """Coverage loop for one adsorbate until unbound or max steps."""
    bo_enabled = process_fn is process_molecule_bayesian
    cached_conformers: list[Atoms] | None = None
    conformer_pack = create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )
    if conformer_pack is not None:
        cached_conformers, _ = conformer_pack

    current_slab = SlabContainer(base_slab.copy())
    steps: list[SaturationStepResult] = []
    bo_state = _BoMemoryState()
    skip_autotune = not needs_workload_autotune(config, bo=bo_enabled)

    def screen_step(
        step: int,
        preamble: _SaturationStepPreamble,
        symmetry_broken: bool,
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        mol_results, transfer_info, new_memory = _screen_saturation_molecule(
            smiles=smiles,
            molecule_name=molecule,
            current_slab=slab,
            calculator=calculator,
            ref_step=preamble.ref_step,
            ts_model=ts_model,
            config=config,
            surface_type=surface_type,
            base_slab=base_slab,
            E_slab=preamble.E_slab,
            failure_summary_out=failure_summary_out,
            symmetry_broken=symmetry_broken,
            process_fn=process_fn,
            bo_state=bo_state if bo_enabled else None,
            reference_unit_smiles=[smiles] * step,
            conformers=cached_conformers,
            skip_workload_autotune=skip_autotune,
        )
        if bo_enabled:
            _commit_bo_memory_state(bo_state, new_memory, config=config)

        if not mol_results:
            logger.warning(
                "Step %d: no valid placements for %s "
                "(including after topology rearrangement guard); stopping saturation",
                step,
                molecule,
            )
            return None

        best = min(mol_results, key=lambda r: r.energy_adsorption)
        return _StepScreenOutcome(
            best=best,
            payload=(mol_results, transfer_info),
        )

    def record_step(step: int, n_on_slab: int, outcome: _StepScreenOutcome) -> None:
        mol_results, transfer_info = outcome.payload
        steps.append(
            SaturationStepResult(
                step=step,
                molecule=molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=outcome.best,
                all_results=mol_results,
                bo_transfer_enabled=bool(config.bo.transfer.enabled),
                transfer=transfer_info,
            )
        )
        ds_logger.add_results(mol_results, smiles=smiles, surface_id=surface_type)
        logger.info(
            "Step %d: best E_ads = %.4f eV (placement %d)",
            step,
            outcome.best.energy_adsorption,
            outcome.best.placement_id,
        )

    final_atoms = _run_saturation_steps(
        config=config,
        current_slab=current_slab,
        reference_slab_for_symmetry=base_slab.copy(),
        calculator=calculator,
        ref=ref,
        log_prefix=f"Saturation for {molecule}",
        log_step_start=lambda step, n_on_slab: logger.info(
            "Saturation step %d for %s (n_molecules on slab: %d)",
            step,
            molecule,
            n_on_slab,
        ),
        make_log_label=lambda step: f"Saturation step {step} for {molecule}",
        screen_step=screen_step,
        record_step=record_step,
    )

    if not steps:
        return None
    return SaturationRunResult(
        molecule=molecule,
        steps=steps,
        n_molecules_at_saturation=_n_at_saturation_from_steps(steps),
        final_slab_atoms=final_atoms,
    )


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
    *,
    process_fn: Callable[..., MoleculeScreenOutcome],
) -> MultiMolSaturationRunResult:
    """Run a competitive multi-molecule saturation loop."""
    bo_enabled = process_fn is process_molecule_bayesian
    conformer_cache: dict[str, tuple[list[Atoms], list[float]]] = {}

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
        "Multi-mol saturation: %d active molecules %s",
        len(active_molecules),
        active_molecules,
    )

    largest_mol = max(
        active_molecules,
        key=lambda m: len(conformer_cache[m][0][0]) if conformer_cache[m][0] else 0,
    )

    current_slab = SlabContainer(base_slab.copy())
    steps: list[MultiMolSaturationStepResult] = []
    molecule_counts: dict[str, int] = {mol: 0 for mol in active_molecules}
    bo_states: dict[str, _BoMemoryState] = {
        mol: _BoMemoryState() for mol in active_molecules
    }

    def screen_step(
        step: int,
        preamble: _SaturationStepPreamble,
        symmetry_broken: bool,
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        if bo_enabled:
            _validate_distinct_bo_memories(
                {m: s.prior_step_memory for m, s in bo_states.items()},
                stage=f"step {step} input",
            )

        E_slab = preamble.E_slab
        ref_step = preamble.ref_step

        if needs_workload_autotune(config, bo=bo_enabled):
            largest_conformers, _ = conformer_cache[largest_mol]
            slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab)
            step_config = resolve_saturation_step_workload_config(
                config,
                ts_model=ts_model,
                conformers=largest_conformers,
                slab_atoms=slab.atoms,
                slab_for_sites=slab_for_sites,
                smiles=active_smiles[largest_mol],
                base_slab_for_frozen=base_slab,
                symmetry_broken=symmetry_broken,
                bo_enabled=bo_enabled,
            )
        else:
            step_config = config

        if step_config.num_placements is None:
            raise ValueError("config.num_placements must be set for saturation")

        slab_for_sites_budget = _build_surface_reference_slab(slab.atoms, base_slab)
        step_complexities: dict[str, float] = {}
        for mol in active_molecules:
            confs, _ = conformer_cache[mol]
            step_complexities[mol] = estimate_molecule_complexity(
                confs,
                slab_for_sites_budget,
                step_config,
                active_smiles[mol],
                full_slab=slab.atoms,
            )
        budget_inputs = {m: c for m, c in step_complexities.items() if c > 0.0}
        if not budget_inputs:
            logger.warning(
                "Step %d: no molecules with available sites under coverage; stopping",
                step,
            )
            return None
        budgets = distribute_placement_budget(
            budget_inputs,
            step_config.num_placements,
        )
        logger.info(
            "Step %d placement budgets: %s (complexities: %s)",
            step,
            budgets,
            {m: round(c) for m, c in step_complexities.items()},
        )

        per_molecule_results: dict[str, list[ScreeningResult]] = {}
        per_molecule_bo_transfer: dict[str, BOTransferInfo] = {}
        new_bo_memory_raw: dict[str, BOStepMemory | None] = {}

        for mol in active_molecules:
            if mol not in budgets:
                per_molecule_results[mol] = []
                per_molecule_bo_transfer[mol] = BOTransferInfo()
                new_bo_memory_raw[mol] = None
                logger.warning(
                    "Step %d | %s: zero site capacity under coverage; skipping",
                    step,
                    mol,
                )
                continue
            smi = active_smiles[mol]
            mol_config = replace(step_config, num_placements=budgets[mol])

            resolved, transfer_info, new_memory = _screen_saturation_molecule(
                smiles=smi,
                molecule_name=mol,
                current_slab=slab,
                calculator=calculator,
                ref_step=ref_step,
                ts_model=ts_model,
                config=mol_config,
                surface_type=surface_type,
                base_slab=base_slab,
                E_slab=E_slab,
                failure_summary_out=failure_summary_out,
                symmetry_broken=symmetry_broken,
                process_fn=process_fn,
                bo_state=bo_states[mol] if bo_enabled else None,
                reference_unit_smiles=_reference_smiles_units_multi_molecule(
                    active_molecules,
                    active_smiles,
                    molecule_counts,
                    mol,
                ),
                conformers=conformer_cache[mol][0],
                skip_workload_autotune=True,
            )
            per_molecule_bo_transfer[mol] = transfer_info
            new_bo_memory_raw[mol] = new_memory
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

        if bo_enabled:
            _validate_distinct_bo_memories(
                new_bo_memory_raw,
                stage=f"step {step} output",
            )
            for mol in active_molecules:
                _commit_bo_memory_state(
                    bo_states[mol], new_bo_memory_raw.get(mol), config=config
                )

        if not any(per_molecule_results.values()):
            logger.warning(
                "Multi-mol saturation step %d: no valid placements for any molecule "
                "(including after topology rearrangement guard); stopping",
                step,
            )
            return None

        all_results_flat = [
            r for results in per_molecule_results.values() for r in results
        ]
        best_overall = min(all_results_flat, key=lambda r: r.energy_adsorption)
        return _StepScreenOutcome(
            best=best_overall,
            payload=(
                best_overall.molecule,
                per_molecule_results,
                dict(budgets),
                per_molecule_bo_transfer,
            ),
        )

    def record_step(step: int, n_on_slab: int, outcome: _StepScreenOutcome) -> None:
        (
            winning_molecule,
            per_molecule_results,
            budgets,
            per_molecule_bo_transfer,
        ) = outcome.payload

        steps.append(
            MultiMolSaturationStepResult(
                step=step,
                winning_molecule=winning_molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=outcome.best,
                per_molecule_results=per_molecule_results,
                per_molecule_budgets=budgets,
                bo_transfer_enabled=bool(config.bo.transfer.enabled),
                transfer_by_molecule=dict(per_molecule_bo_transfer),
            )
        )
        molecule_counts[winning_molecule] += 1

        logger.info(
            "Step %d: winner = %s, E_ads = %.4f eV",
            step,
            winning_molecule,
            outcome.best.energy_adsorption,
        )

    final_atoms = _run_saturation_steps(
        config=config,
        current_slab=current_slab,
        reference_slab_for_symmetry=base_slab.copy(),
        calculator=calculator,
        ref=ref,
        log_prefix="Multi-mol saturation",
        log_step_start=lambda step, n_on_slab: logger.info(
            "Multi-mol saturation step %d (molecules on slab: %d)",
            step,
            n_on_slab,
        ),
        make_log_label=lambda step: f"Multi-mol saturation step {step}",
        screen_step=screen_step,
        record_step=record_step,
    )

    return MultiMolSaturationRunResult(
        molecules=molecules,
        steps=steps,
        n_molecules_at_saturation=_n_at_saturation_from_steps(steps),
        final_slab_atoms=final_atoms,
        molecule_counts=molecule_counts,
    )


def run_saturation_screening(
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | tuple[str, str] | str,
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    failure_summary_out: dict[str, object] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
    *,
    bo_enabled: bool = False,
) -> list[SaturationRunResult] | list[MultiMolSaturationRunResult]:
    """Sequential saturation: add molecules until best E_ads >= 0.

    Parameters
    ----------
    molecules:
        In-memory ``(smiles, name)`` list/tuple or path to a two-column CSV.
    bo_enabled:
        When True, each step uses Bayesian placement selection (and optional
        cross-step transfer). Prefer :func:`~metalsurfer.run_saturation_bo` at
        the campaign layer.
    """
    if config is None:
        config = AdsorptionConfig()

    t_run_start = time.perf_counter()

    with log_context(surface_type=surface_type, seed=config.seed):
        molecule_pairs, load_status, _molecules_source = _normalize_molecules_input(
            molecules,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_existing,
        )
        if not molecule_pairs:
            if load_status == "all_skipped":
                summary_csv = f"results_{surface_type}/saturation_summary.csv"
                logger.warning(
                    "No molecules to process: all already listed in %s. "
                    "Set skip_existing=False or remove that CSV to rerun",
                    summary_csv,
                )
            elif load_status == "empty_file":
                logger.warning("No molecules to process: file empty or no valid rows")
            else:
                logger.warning("No molecules to process")
            return []

        bootstrap = _bootstrap_screening_run(slab, molecule_pairs, config)
        calculator = bootstrap.calculator
        ts_model = bootstrap.ts_model
        ref = bootstrap.ref
        t_ref_s = bootstrap.t_ref_s
        slab = bootstrap.slab
        smiles_list = [smiles for smiles, _ in molecule_pairs]
        molecule_names = [name for _, name in molecule_pairs]
        base_slab = slab.atoms.copy()
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)
        process_fn = process_molecule_bayesian if bo_enabled else process_molecule

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
                process_fn=process_fn,
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
                "Multi_molecule_saturation=True but only one molecule provided; falling back to standard single-molecule saturation"
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

            run_result = _run_single_molecule_saturation(
                smiles=smi,
                molecule=mol,
                base_slab=base_slab,
                calculator=calculator,
                ts_model=ts_model,
                ref=ref,
                config=config,
                surface_type=surface_type,
                failure_summary_out=failure_summary_out,
                ds_logger=ds_logger,
                process_fn=process_fn,
            )
            if run_result is not None:
                all_saturation_results.append(run_result)

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

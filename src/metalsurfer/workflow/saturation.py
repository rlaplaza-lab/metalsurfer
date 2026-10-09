"""Sequential and multi-molecule saturation workflow entry points."""

import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, NamedTuple

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..filters import adsorbate_connected_components
from ..models import (
    BOStepMemory,
    BOTransferInfo,
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
    estimate_conformer_count,
)
from ..placement.site_context import (
    resolve_site_context_for_sampling,
    skip_symmetry_for_sampling,
)
from ..reporting import FailureSummary
from ..result_paths import results_dir_for
from ..surface_prep import SlabContainer, apply_material_pbc
from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from .bayesian import process_molecule_bayesian
from .core import process_molecule
from .joint_tuplet import (
    JointTupletScreenOutcome,
    commit_best_joint_config,
    joint_winning_molecule_label,
    process_joint_tuplet_bayesian,
    screen_joint_tuplet_homogeneous,
    screen_joint_tuplet_multi,
)
from .shared import (
    MoleculeScreenOutcome,
    _bootstrap_screening_run,
    _build_surface_reference_slab,
    _compute_slab_energy,
    _dump_debug_sites_if_enabled,
    _missing_molecule_reference,
    _normalize_molecules_input,
    adsorption_ranking_energy,
    empty_molecule_input_message,
    joint_config_ranking_energy,
    needs_workload_autotune,
    require_saturation_activity,
    resolve_saturation_activities,
    resolve_saturation_step_workload_config,
)

logger = logging.getLogger(__name__)


def _omega(
    result: ScreeningResult,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> float:
    activity = require_saturation_activity(result.molecule, activity_by_molecule)
    return adsorption_ranking_energy(
        result.energy_adsorption,
        activity,
        temperature,
        pressure,
    )


def _omega_sort_key(
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> Callable[[ScreeningResult], tuple[float, int, str]]:
    def key(result: ScreeningResult) -> tuple[float, int, str]:
        return (
            _omega(result, activity_by_molecule, temperature, pressure),
            result.placement_id,
            result.molecule,
        )

    return key


def _step_ranking_snapshot(
    *,
    committed: Sequence[ScreeningResult],
    pool_best: ScreeningResult,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[float, float, str]:
    """Return ``(Ω, E_ads, label)`` matching the stop-condition ranking.

    Committed steps use :func:`joint_config_ranking_energy` (``Ω`` or
    ``Ω_tuplet``). Empty commits report the pool-best single-unit ``Ω``.
    """
    if committed:
        omega = joint_config_ranking_energy(
            committed,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        )
        e_ads = float(committed[0].energy_adsorption)
        label = "Ω_tuplet" if len(committed) > 1 else "Ω"
        return omega, e_ads, label
    return (
        _omega(pool_best, activity_by_molecule, temperature, pressure),
        float(pool_best.energy_adsorption),
        "Ω",
    )


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
    reference_analyzer: SymmetryAnalyzer | None = None,
) -> bool:
    """Check whether symmetry vs *reference_atoms* is broken or analysis fails (treat as C1)."""
    analyzer = SymmetryAnalyzer(current_atoms, symmetry_tolerance=symmetry_tolerance)
    try:
        broken = analyzer.detect_symmetry_breaking(
            reference_atoms, reference_analyzer=reference_analyzer
        )
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
    if new_memory is not None:
        state.prior_step_memories.append(new_memory)
    if config.bo.transfer.enabled and config.bo.transfer.mode == "cumulative_refit":
        state.prior_cumulative_memory = merge_bo_step_memories(
            [state.prior_cumulative_memory, new_memory]
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


def _n_at_saturation_from_steps(
    steps: Sequence[SaturationStepResult | MultiMolSaturationStepResult],
) -> int:
    """Total adsorbates folded onto the slab: sum of per-step ``n_added``.

    Bound steps contribute one placement each in one-molecule-per-step mode
    (n-tuplet steps contribute several); an unbound final step contributes zero,
    so the total always equals the number of adsorbates on the returned final
    slab.
    """
    return sum(step.n_added for step in steps)


def _reference_smiles_units_multi_molecule(
    active_molecules: list[str],
    active_smiles: dict[str, str],
    molecule_counts: dict[str, int],
    placing_molecule: str = "",
    pending_additions: Mapping[str, int] | None = None,
) -> list[str]:
    """SMILES for every adsorbate unit present in the *screened* structure.

    ``molecule_counts`` is read before ``record_step`` commits the step's
    winners, so it holds the units already on the slab. By default the
    candidate being screened contains exactly one pending unit of
    *placing_molecule*. n-tuplet flows pass ``pending_additions`` (molecule ->
    count committed this step) instead; the topology guard counts CONNECTED
    COMPONENTS of the screened structure, so the reference length must equal
    units-on-slab plus all pending units. ``molecule_counts`` stays the single
    source of truth.

    Parameters
    ----------
    active_molecules
        Molecule names competing in this run.
    active_smiles
        Molecule name -> SMILES mapping.
    molecule_counts
        Units already folded onto the slab per molecule.
    placing_molecule
        Molecule whose candidates are being screened.
    pending_additions
        Explicit pending units for this step; ``None`` means one unit of
        *placing_molecule* (legacy behavior).
    """
    pending: Mapping[str, int] = (
        {placing_molecule: 1} if pending_additions is None else pending_additions
    )
    units: list[str] = []
    for mol in active_molecules:
        n = molecule_counts.get(mol, 0) + pending.get(mol, 0)
        units.extend([active_smiles[mol]] * n)
    return units


def _scale_budget_for_tuplet(config: AdsorptionConfig) -> AdsorptionConfig:
    """Divide autotuned workload capacity across tuplet members (conservative).

    The workload probe measures parallel relaxation capacity with a
    representative ONE-molecule geometry; a composite candidate carries ~n x
    atoms, so the probed pool size is floor-divided by the tuplet size before
    budget distribution. Applied once, right after resolution (the scaled
    config is written back, so repeat calls are never compounded).

    ``num_placements`` counts joint configs; ``bo.initial_random`` and
    ``bo.batch_size`` are scaled the same way because each eval relaxes *n*
    adsorbates together.
    """
    n_per_step = config.saturation_molecules_per_step
    num_placements = config.num_placements
    if n_per_step <= 1 or num_placements is None:
        return config
    scaled = max(1, num_placements // n_per_step)
    initial_random = config.bo.initial_random
    batch_size = config.bo.batch_size
    scaled_bo = config.bo
    if initial_random is not None or batch_size is not None:
        scaled_bo = replace(
            config.bo,
            initial_random=(
                max(1, initial_random // n_per_step)
                if initial_random is not None
                else None
            ),
            batch_size=(
                max(1, batch_size // n_per_step) if batch_size is not None else None
            ),
        )
    if scaled == num_placements and scaled_bo is config.bo:
        return config
    logger.info(
        "n-tuplet mode: dividing probed workload capacity %d by %d -> "
        "num_placements=%d (joint configs)",
        num_placements,
        n_per_step,
        scaled,
    )
    return replace(config, num_placements=scaled, bo=scaled_bo)


def _resolve_step_workload_config(
    config: AdsorptionConfig,
    *,
    bo_enabled: bool,
    ts_model: object,
    conformers: list[Atoms] | None,
    slab_atoms: Atoms,
    slab_for_sites: Atoms,
    smiles: str,
    base_slab_for_frozen: Atoms,
    symmetry_broken: bool,
) -> AdsorptionConfig:
    """Autotune and scale the step workload, or return *config* unchanged."""
    if not needs_workload_autotune(config, bo=bo_enabled):
        return config
    if conformers is None:
        raise ValueError("conformers required to resolve saturation workload config")
    return _scale_budget_for_tuplet(
        resolve_saturation_step_workload_config(
            config,
            ts_model=ts_model,
            conformers=conformers,
            slab_atoms=slab_atoms,
            slab_for_sites=slab_for_sites,
            smiles=smiles,
            base_slab_for_frozen=base_slab_for_frozen,
            symmetry_broken=symmetry_broken,
            bo_enabled=bo_enabled,
        )
    )


def _commit_joint_step(
    joint_out: JointTupletScreenOutcome,
    *,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[ScreeningResult, list[ScreeningResult]] | None:
    """Return ``(pool_best, committed)`` for a joint screen, or ``None`` if empty."""
    if not joint_out.valid_configs:
        return None
    best, committed = commit_best_joint_config(
        joint_out.valid_configs,
        activity_by_molecule=activity_by_molecule,
        temperature=temperature,
        pressure=pressure,
    )
    assert best is not None
    return best, committed


def _commit_sequential_step(
    results: Sequence[ScreeningResult],
    *,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[ScreeningResult, list[ScreeningResult]]:
    """Pick the Ω-best sequential row and commit it when it binds."""
    best = min(
        results,
        key=_omega_sort_key(activity_by_molecule, temperature, pressure),
    )
    return _resolve_step_commit(
        pool_best=best,
        activity_by_molecule=activity_by_molecule,
        temperature=temperature,
        pressure=pressure,
    )


def _log_step_ranking(
    *,
    step: int,
    committed: Sequence[ScreeningResult],
    pool_best: ScreeningResult,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
    style: Literal["single", "multi"],
    winning_molecule: str | None = None,
) -> None:
    """Log the stop-condition Ω / Ω_tuplet snapshot for one saturation step."""
    omega, e_ads, label = _step_ranking_snapshot(
        committed=committed,
        pool_best=pool_best,
        activity_by_molecule=activity_by_molecule,
        temperature=temperature,
        pressure=pressure,
    )
    if style == "single":
        if len(committed) > 1:
            logger.info(
                "Step %d: %s = %.4f eV (E_ads = %.4f eV, %d units)",
                step,
                label,
                omega,
                e_ads,
                len(committed),
            )
        elif committed:
            logger.info(
                "Step %d: %s = %.4f eV (E_ads = %.4f eV, placement %d)",
                step,
                label,
                omega,
                e_ads,
                committed[0].placement_id,
            )
        else:
            logger.info(
                "Step %d: no commit (%s = %.4f eV, E_ads = %.4f eV, placement %d)",
                step,
                label,
                omega,
                e_ads,
                pool_best.placement_id,
            )
    elif style == "multi":
        if winning_molecule is None:
            raise ValueError("winning_molecule is required for multi ranking logs")
        if len(committed) > 1:
            logger.info(
                "Step %d: winners = %s, %s = %.4f eV (E_ads = %.4f eV, %d units)",
                step,
                ",".join(placement.molecule for placement in committed),
                label,
                omega,
                e_ads,
                len(committed),
            )
        elif committed:
            logger.info(
                "Step %d: winner = %s, %s = %.4f eV (E_ads = %.4f eV)",
                step,
                winning_molecule,
                label,
                omega,
                e_ads,
            )
        else:
            logger.info(
                "Step %d: no commit (best = %s, %s = %.4f eV, E_ads = %.4f eV)",
                step,
                winning_molecule,
                label,
                omega,
                e_ads,
            )
    else:
        raise ValueError(f"unknown ranking log style: {style!r}")


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
        return True, ""

    components = adsorbate_connected_components(
        atoms,
        base_slab_len,
        config.connectivity_multiplier,
        material_type=config.material_type,
    )
    if len(components) != len(reference_unit_smiles):
        return (
            False,
            f"expected {len(reference_unit_smiles)} adsorbate units, "
            f"found {len(components)} connected fragments",
        )

    return True, ""


def _joint_topology_check(
    *,
    base_slab: Atoms,
    reference_unit_smiles: list[str],
    smiles_by_molecule: Mapping[str, str],
    config: AdsorptionConfig,
) -> Callable[[Atoms, list[str]], tuple[bool, str]] | None:
    """Return a connectivity guard for joint composite validation, or ``None``."""
    if not config.saturation_discard_topology_rearrangements:
        return None

    def topology_check(opt_atoms: Atoms, pending_names: list[str]) -> tuple[bool, str]:
        reference_units = [
            *reference_unit_smiles,
            *(smiles_by_molecule[name] for name in pending_names),
        ]
        return _saturation_adsorbate_topology_ok(
            opt_atoms,
            base_slab_len=len(base_slab),
            reference_unit_smiles=reference_units,
            config=config,
        )

    return topology_check


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
    reference_analyzer: SymmetryAnalyzer,
) -> _SaturationStepPreamble:
    """Shared per-step setup before molecule screening."""
    if step > 1 and not config.saturation_autobatcher_reuse:
        clear_autobatcher_cache()

    if step > 1 and not symmetry_broken:
        # Substrate only: adsorbates would always flip the space-group fingerprint.
        substrate_for_symmetry = _build_surface_reference_slab(
            current_slab.atoms,
            reference_slab_for_symmetry,
        )
        symmetry_broken = _saturation_symmetry_broken_vs_reference(
            substrate_for_symmetry,
            reference_slab_for_symmetry,
            symmetry_tolerance=config.symmetry_tolerance,
            reference_analyzer=reference_analyzer,
        )

    E_slab = (
        ref.slab_energy
        if step == 1
        else _compute_slab_energy(
            current_slab.atoms,
            calculator,
            label=f"{log_label} slab",
        )
    )
    ref_step = ReferenceEnergies(
        slab_energy=E_slab,
        molecule_energies=ref.molecule_energies,
        conformer_packs=ref.conformer_packs,
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
    failure_summary_out: dict[str, FailureSummary] | None,
    symmetry_broken: bool,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_enabled: bool,
    bo_state: _BoMemoryState | None,
    reference_unit_smiles: list[str],
    conformers: list[Atoms] | None = None,
    conformer_energies: list[float] | None = None,
    skip_workload_autotune: bool = False,
    site_context: object | None = None,
    debug_sites_step: int | None = None,
) -> tuple[list[ScreeningResult], BOTransferInfo | None, BOStepMemory | None]:
    """Run one molecule's place/opt/filter for a saturation step."""
    kwargs: dict[str, Any] = {
        "ts_model": ts_model,
        "config": config,
        "surface_type": surface_type,
        "reference_smiles": smiles,
        "base_slab_for_frozen": base_slab,
        "slab_energy_override": E_slab,
        "symmetry_broken": symmetry_broken,
        "conformers": conformers,
        "conformer_energies": conformer_energies,
        "skip_workload_autotune": skip_workload_autotune,
        "saturation_reuse": True,
        "site_context": site_context,
        "debug_sites_step": debug_sites_step,
    }
    if bo_enabled:
        kwargs["bo_step_memory_in"] = (
            _bo_transfer_memory_in(config, bo_state) if bo_state is not None else None
        )

    outcome = process_fn(
        smiles,
        molecule_name,
        current_slab,
        calculator,
        ref_step,
        **kwargs,
    )
    if bo_enabled:
        if outcome.transfer_info is None:
            raise RuntimeError(
                f"Bayesian screening for {molecule_name!r} returned no transfer_info"
            )
        transfer_info = outcome.transfer_info
        new_memory = outcome.bo_memory
    else:
        transfer_info = None
        new_memory = None

    if failure_summary_out is not None and outcome.failure_summary:
        failure_summary_out[molecule_name] = outcome.failure_summary

    filtered = _filter_saturation_topology_results(
        list(outcome.results),
        base_slab_len=len(base_slab),
        reference_unit_smiles=reference_unit_smiles,
        config=config,
    )
    return filtered, transfer_info, new_memory


def _saturation_should_stop(
    *,
    best_energy: float,
    n_committed: int,
    step: int,
    config: AdsorptionConfig,
    log_prefix: str,
) -> bool:
    """Stop when the step commits nothing, Ω is non-negative, or max steps.

    Empty commit is always terminal. For bound steps, ``best_energy`` is the
    committed winner's (or tuplet's) Ω.
    """
    if n_committed == 0:
        logger.info(
            "%s: slab saturated at step %d (no placements committed)",
            log_prefix,
            step,
        )
        return True
    if best_energy >= 0:
        logger.info(
            "%s: slab saturated at step %d (Ω >= 0)",
            log_prefix,
            step,
        )
        return True
    if config.saturation_max_steps is not None and step >= config.saturation_max_steps:
        logger.info(
            "%s: reached max steps (%d)",
            log_prefix,
            config.saturation_max_steps,
        )
        return True
    return False


def _resolve_step_commit(
    *,
    pool_best: ScreeningResult,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[ScreeningResult, list[ScreeningResult]]:
    """Commit the pool best when it binds (Ω < 0); sequential n=1 only."""
    committed = (
        [pool_best]
        if _omega(pool_best, activity_by_molecule, temperature, pressure) < 0
        else []
    )
    return pool_best, committed


def _resolve_conformer_pack(
    *,
    smiles: str,
    molecule: str,
    ref: ReferenceEnergies,
    calculator: object,
    ts_model: object,
    config: AdsorptionConfig,
) -> tuple[list[Atoms], list[float]] | None:
    """Conformers+energies for *molecule*: reference cache first, else generate."""
    cached_pack = ref.get_conformer_pack(molecule)
    if cached_pack is not None:
        return cached_pack
    return create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )


@dataclass(frozen=True)
class _SingleStepPayload:
    """Per-step bookkeeping for single-molecule saturation."""

    mol_results: list[ScreeningResult]
    transfer_info: BOTransferInfo | None


@dataclass(frozen=True)
class _MultiStepPayload:
    """Per-step bookkeeping for competitive multi-molecule saturation."""

    winning_molecule: str
    per_molecule_results: dict[str, list[ScreeningResult]]
    budgets: dict[str, int]
    transfer_by_molecule: dict[str, BOTransferInfo | None]


@dataclass(frozen=True)
class _StepScreenOutcome:
    """Result of one saturation step's screening phase.

    ``committed`` lists the placements folded into the coverage slab this step
    (one element in one-molecule-per-step mode; several for n-tuplet steps).
    """

    best: ScreeningResult
    committed: list[ScreeningResult]
    payload: _SingleStepPayload | _MultiStepPayload


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
        [int, _SaturationStepPreamble, SlabContainer],
        _StepScreenOutcome | None,
    ],
    record_step: Callable[[int, int, _StepScreenOutcome], None],
    activity_by_molecule: Mapping[str, float],
) -> Atoms:
    """Shared coverage loop; single/multi differ only via screen/record callbacks.

    ``record_step`` receives the explicit number of adsorbate units already on
    the slab (not ``step - 1``), keeping the loop correct for n-tuplet steps
    that commit several placements at once.
    """
    symmetry_broken = False
    # Clean reference is fixed for the run; build its analyzer once.
    reference_analyzer = SymmetryAnalyzer(
        reference_slab_for_symmetry,
        symmetry_tolerance=config.symmetry_tolerance,
    )
    step = 0
    n_on_slab = 0
    temperature = config.saturation_temperature
    pressure = config.saturation_pressure
    while True:
        step += 1
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
            reference_analyzer=reference_analyzer,
        )
        symmetry_broken = preamble.symmetry_broken

        outcome = screen_step(step, preamble, current_slab)
        if outcome is None:
            break

        record_step(step, n_on_slab, outcome)

        # Only fold a bound step's committed placements into the coverage slab.
        # An unbound final step is recorded for the record but not incorporated,
        # and this also fixes the max-steps path so the final slab holds exactly
        # ``n_molecules_at_saturation`` adsorbates. Sequential steps commit at
        # most one placement; n-tuplet steps may commit several here.
        for placement in outcome.committed:
            current_slab = _slab_after_saturation_step(placement.atoms, config)
        n_on_slab += len(outcome.committed)

        ranking_energy = joint_config_ranking_energy(
            outcome.committed,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        )
        if _saturation_should_stop(
            best_energy=ranking_energy,
            n_committed=len(outcome.committed),
            step=step,
            config=config,
            log_prefix=log_prefix,
        ):
            break

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
    failure_summary_out: dict[str, FailureSummary] | None,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_enabled: bool,
    activity_by_molecule: Mapping[str, float],
) -> SaturationRunResult | None:
    """Coverage loop for one adsorbate until unbound or max steps."""
    temperature = config.saturation_temperature
    pressure = config.saturation_pressure
    pack = _resolve_conformer_pack(
        smiles=smiles,
        molecule=molecule,
        ref=ref,
        calculator=calculator,
        ts_model=ts_model,
        config=config,
    )
    cached_conformers, cached_conformer_energies = (
        pack
        if pack is not None
        else (
            None,
            None,
        )
    )

    current_slab = SlabContainer(base_slab.copy())
    steps: list[SaturationStepResult] = []
    bo_state = _BoMemoryState()
    # SMILES per committed adsorbate unit already on the slab.
    units_on_slab: list[str] = []

    def screen_step(
        step: int,
        preamble: _SaturationStepPreamble,
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        """Screen placements for one saturation step."""
        nonlocal config
        symmetry_broken = preamble.symmetry_broken
        if needs_workload_autotune(config, bo=bo_enabled):
            slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab)
            config = _resolve_step_workload_config(
                config,
                bo_enabled=bo_enabled,
                ts_model=ts_model,
                conformers=cached_conformers,
                slab_atoms=slab.atoms,
                slab_for_sites=slab_for_sites,
                smiles=smiles,
                base_slab_for_frozen=base_slab,
                symmetry_broken=symmetry_broken,
            )
        n_tuplet = config.saturation_molecules_per_step > 1
        transfer_info: BOTransferInfo | None = None
        if n_tuplet:
            ref_units = list(units_on_slab)
            topology_check = _joint_topology_check(
                base_slab=base_slab,
                reference_unit_smiles=ref_units,
                smiles_by_molecule={molecule: smiles},
                config=config,
            )
            if bo_enabled:
                joint_out = process_joint_tuplet_bayesian(
                    smiles,
                    molecule,
                    slab,
                    calculator,
                    preamble.ref_step,
                    ts_model=ts_model,
                    config=config,
                    surface_type=surface_type,
                    base_slab_for_frozen=base_slab,
                    slab_energy_override=preamble.E_slab,
                    symmetry_broken=symmetry_broken,
                    bo_step_memory_in=_bo_transfer_memory_in(config, bo_state)
                    if bo_state is not None
                    else None,
                    conformers=cached_conformers,
                    conformer_energies=cached_conformer_energies,
                    skip_workload_autotune=True,
                    debug_sites_step=step,
                    topology_check=topology_check,
                    activity_by_molecule=activity_by_molecule,
                )
                transfer_info = joint_out.transfer_info
                _commit_bo_memory_state(bo_state, joint_out.bo_memory, config=config)
            else:
                if cached_conformers is None:
                    raise ValueError("conformers required for joint tuplet screening")
                joint_out = screen_joint_tuplet_homogeneous(
                    smiles=smiles,
                    molecule_name=molecule,
                    current_slab=slab,
                    calculator=calculator,
                    ref_step=preamble.ref_step,
                    ts_model=ts_model,
                    config=config,
                    base_slab=base_slab,
                    E_slab=preamble.E_slab,
                    symmetry_broken=symmetry_broken,
                    conformers=cached_conformers,
                    conformer_energies=cached_conformer_energies,
                    site_context=None,
                    topology_check=topology_check,
                    debug_sites_step=step,
                    log_prefix=f"Saturation for {molecule} | step {step} | ",
                )
            committed_pair = _commit_joint_step(
                joint_out,
                activity_by_molecule=activity_by_molecule,
                temperature=temperature,
                pressure=pressure,
            )
            if committed_pair is None:
                logger.warning(
                    "Step %d: no valid joint configs for %s; stopping saturation",
                    step,
                    molecule,
                )
                return None
            best, committed = committed_pair
            mol_results = joint_out.flat_results
        else:
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
                bo_enabled=bo_enabled,
                bo_state=bo_state if bo_enabled else None,
                reference_unit_smiles=[*units_on_slab, smiles],
                conformers=cached_conformers,
                conformer_energies=cached_conformer_energies,
                skip_workload_autotune=True,
                debug_sites_step=step,
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

            best, committed = _commit_sequential_step(
                mol_results,
                activity_by_molecule=activity_by_molecule,
                temperature=temperature,
                pressure=pressure,
            )
        return _StepScreenOutcome(
            best=best,
            committed=committed,
            payload=_SingleStepPayload(
                mol_results=mol_results,
                transfer_info=transfer_info,
            ),
        )

    def record_step(step: int, n_on_slab: int, outcome: _StepScreenOutcome) -> None:
        """Record the results of one saturation step."""
        payload = outcome.payload
        assert isinstance(payload, _SingleStepPayload)
        steps.append(
            SaturationStepResult(
                step=step,
                molecule=molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=outcome.best,
                all_results=payload.mol_results,
                bo_transfer_enabled=bool(bo_enabled and config.bo.transfer.enabled),
                transfer=payload.transfer_info,
                n_added=len(outcome.committed),
                committed_results=outcome.committed,
            )
        )
        for _placement in outcome.committed:
            units_on_slab.append(smiles)
        _log_step_ranking(
            step=step,
            committed=outcome.committed,
            pool_best=outcome.best,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
            style="single",
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
        activity_by_molecule=activity_by_molecule,
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
    failure_summary_out: dict[str, FailureSummary] | None,
    *,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_enabled: bool,
    activity_by_molecule: Mapping[str, float],
) -> MultiMolSaturationRunResult:
    """Run a competitive multi-molecule saturation loop."""
    if bo_enabled and int(config.saturation_molecules_per_step) > 1:
        raise ValueError(
            "Joint n-tuplet Bayesian screening is single-species only; "
            "multi_molecule_saturation with saturation_molecules_per_step > 1 "
            "cannot use bo_enabled=True"
        )
    temperature = config.saturation_temperature
    pressure = config.saturation_pressure
    conformer_cache: dict[str, tuple[list[Atoms], list[float]]] = {}

    for smi, mol in zip(smiles_list, molecules, strict=True):
        pack = _resolve_conformer_pack(
            smiles=smi,
            molecule=mol,
            ref=ref,
            calculator=calculator,
            ts_model=ts_model,
            config=config,
        )
        if pack is None:
            logger.warning(
                "Multi-mol saturation: could not generate conformers for %s; skipping this molecule",
                mol,
            )
            continue
        conformer_cache[mol] = pack

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
        key=lambda m: len(conformer_cache[m][0][0]),
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
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        """Screen placements for one multi-molecule saturation step."""
        nonlocal config
        symmetry_broken = preamble.symmetry_broken

        E_slab = preamble.E_slab
        ref_step = preamble.ref_step

        slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab)
        largest_conformers, _ = conformer_cache[largest_mol]
        step_config = _resolve_step_workload_config(
            config,
            bo_enabled=bo_enabled,
            ts_model=ts_model,
            conformers=largest_conformers,
            slab_atoms=slab.atoms,
            slab_for_sites=slab_for_sites,
            smiles=active_smiles[largest_mol],
            base_slab_for_frozen=base_slab,
            symmetry_broken=symmetry_broken,
        )
        config = step_config

        step_complexities: dict[str, float] = {}
        for mol in active_molecules:
            confs, _ = conformer_cache[mol]
            step_complexities[mol] = estimate_conformer_count(confs)
        num_placements = step_config.num_placements
        if num_placements is None:
            raise ValueError("num_placements must be resolved before saturation steps")
        budgets = distribute_placement_budget(
            step_complexities,
            num_placements,
        )
        logger.info(
            "Step %d placement budgets: %s (complexities: %s)",
            step,
            budgets,
            {m: round(c) for m, c in step_complexities.items()},
        )

        per_molecule_results: dict[str, list[ScreeningResult]] = {
            mol: [] for mol in active_molecules
        }
        per_molecule_bo_transfer: dict[str, BOTransferInfo | None] = {
            mol: None for mol in active_molecules
        }
        new_bo_memory_raw: dict[str, BOStepMemory | None] = {
            mol: None for mol in active_molecules
        }

        shared_site_context = resolve_site_context_for_sampling(
            slab_for_sites,
            step_config,
            symmetry_broken=skip_symmetry_for_sampling(
                symmetry_broken=symmetry_broken,
                slab_for_sites=slab_for_sites,
                full_slab=slab.atoms,
                config=step_config,
            ),
        )
        _dump_debug_sites_if_enabled(
            slab_for_sites,
            shared_site_context,
            step_config,
            surface_type,
            step=step,
        )

        if step_config.saturation_molecules_per_step > 1:
            ref_units = _reference_smiles_units_multi_molecule(
                active_molecules,
                active_smiles,
                molecule_counts,
                pending_additions={},
            )
            joint_out = screen_joint_tuplet_multi(
                active_molecules=active_molecules,
                active_smiles=active_smiles,
                conformer_cache=conformer_cache,
                current_slab=slab,
                calculator=calculator,
                ref_step=ref_step,
                ts_model=ts_model,
                config=step_config,
                base_slab=base_slab,
                E_slab=E_slab,
                site_context=shared_site_context,
                topology_check=_joint_topology_check(
                    base_slab=base_slab,
                    reference_unit_smiles=ref_units,
                    smiles_by_molecule=active_smiles,
                    config=step_config,
                ),
                log_prefix=f"Multi-mol saturation | step {step} | ",
            )
            for row in joint_out.flat_results:
                per_molecule_results[row.molecule].append(row)
            committed_pair = _commit_joint_step(
                joint_out,
                activity_by_molecule=activity_by_molecule,
                temperature=temperature,
                pressure=pressure,
            )
            if committed_pair is None:
                logger.warning(
                    "Multi-mol saturation step %d: no valid joint configs; stopping",
                    step,
                )
                return None
            best_overall, committed = committed_pair
            winning_label = (
                joint_winning_molecule_label(committed)
                if committed
                else best_overall.molecule
            )
            return _StepScreenOutcome(
                best=best_overall,
                committed=committed,
                payload=_MultiStepPayload(
                    winning_molecule=winning_label,
                    per_molecule_results=per_molecule_results,
                    budgets=dict(budgets),
                    transfer_by_molecule=per_molecule_bo_transfer,
                ),
            )

        for mol in active_molecules:
            if mol not in budgets:
                per_molecule_results[mol] = []
                per_molecule_bo_transfer[mol] = None
                new_bo_memory_raw[mol] = None
                logger.warning(
                    "Step %d | %s: omitted from placement budget "
                    "(budget smaller than molecule count); skipping",
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
                bo_enabled=bo_enabled,
                bo_state=bo_states[mol] if bo_enabled else None,
                reference_unit_smiles=_reference_smiles_units_multi_molecule(
                    active_molecules,
                    active_smiles,
                    molecule_counts,
                    mol,
                ),
                conformers=conformer_cache[mol][0],
                conformer_energies=conformer_cache[mol][1],
                skip_workload_autotune=True,
                site_context=shared_site_context,
            )
            per_molecule_bo_transfer[mol] = transfer_info
            new_bo_memory_raw[mol] = new_memory
            per_molecule_results[mol] = resolved
            if resolved:
                best_mol = min(
                    resolved,
                    key=_omega_sort_key(activity_by_molecule, temperature, pressure),
                )
                logger.info(
                    "Step %d | %s: best Ω = %.4f eV (E_ads = %.4f eV, %d results)",
                    step,
                    mol,
                    _omega(best_mol, activity_by_molecule, temperature, pressure),
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
        best_overall, committed = _commit_sequential_step(
            all_results_flat,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        )
        return _StepScreenOutcome(
            best=best_overall,
            committed=committed,
            payload=_MultiStepPayload(
                winning_molecule=best_overall.molecule,
                per_molecule_results=per_molecule_results,
                budgets=dict(budgets),
                transfer_by_molecule=per_molecule_bo_transfer,
            ),
        )

    def record_step(step: int, n_on_slab: int, outcome: _StepScreenOutcome) -> None:
        """Record the results of one multi-molecule saturation step."""
        payload = outcome.payload
        assert isinstance(payload, _MultiStepPayload)
        winning_molecule = payload.winning_molecule
        committed = outcome.committed

        steps.append(
            MultiMolSaturationStepResult(
                step=step,
                winning_molecule=winning_molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=outcome.best,
                per_molecule_results=payload.per_molecule_results,
                per_molecule_budgets=payload.budgets,
                bo_transfer_enabled=bool(bo_enabled and config.bo.transfer.enabled),
                transfer_by_molecule=dict(payload.transfer_by_molecule),
                n_added=len(committed),
                committed_results=committed,
            )
        )

        for molecule_name, count in Counter(
            placement.molecule for placement in committed
        ).items():
            molecule_counts[molecule_name] += count

        _log_step_ranking(
            step=step,
            committed=committed,
            pool_best=outcome.best,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
            style="multi",
            winning_molecule=winning_molecule,
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
        activity_by_molecule=activity_by_molecule,
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
    config: AdsorptionConfig,
    surface_type: str = "manual",
    skip_existing: bool = True,
    failure_summary_out: dict[str, FailureSummary] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
    *,
    bo_enabled: bool = False,
) -> list[SaturationRunResult] | list[MultiMolSaturationRunResult]:
    """Sequential saturation until ranking energy Ω ≥ 0 (E_ads at SATP).

    Parameters
    ----------
    slab
        Substrate structure.
    molecules
        In-memory ``(smiles, name)`` list/tuple or path to a two-column CSV.
    config
        Adsorption configuration. Optional ``saturation_temperature`` /
        ``saturation_pressure`` / ``saturation_activities`` /
        ``saturation_omega_shift`` shift ranking via Ω.
    surface_type
        Surface type label.
    skip_existing
        Whether to skip molecules with existing results.
    failure_summary_out
        Optional per-molecule failure summaries
        (``{molecule_name: summary_dict}``).
    run_metadata_out
        Optional dict to populate with run metadata.
    bo_enabled
        When True, each step uses Bayesian placement selection. Prefer
        :func:`~metalsurfer.run_saturation_bo` at the campaign layer.

    Notes
    -----
    With ``saturation_molecules_per_step > 1``, each step screens joint
    configs of exactly that many adsorbates relaxed together (see
    ``workflow/joint_tuplet.py``).
    """
    t_run_start = time.perf_counter()

    with log_context(surface_type=surface_type, seed=config.seed):
        molecule_pairs, load_status, _molecules_source = _normalize_molecules_input(
            molecules,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_existing,
        )
        if not molecule_pairs:
            listed_csv = (
                results_dir_for(surface_type) / "saturation_summary.csv"
            ).as_posix()
            msg = empty_molecule_input_message(load_status, listed_csv=listed_csv)
            if msg is not None:
                logger.warning(msg)
            return []

        bootstrap = _bootstrap_screening_run(slab, molecule_pairs, config)
        calculator = bootstrap.calculator
        ts_model = bootstrap.ts_model
        ref = bootstrap.ref
        t_ref_s = bootstrap.t_ref_s
        slab = bootstrap.slab
        smiles_list = [smiles for smiles, _ in molecule_pairs]
        molecule_names = [name for _, name in molecule_pairs]
        activity_by_molecule = resolve_saturation_activities(
            molecule_names,
            config.saturation_activities,
            omega_shifts=config.saturation_omega_shift,
            temperature=config.saturation_temperature,
        )
        base_slab = slab.atoms.copy()
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
                process_fn=process_fn,
                bo_enabled=bo_enabled,
                activity_by_molecule=activity_by_molecule,
            )
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

        all_saturation_results: list[SaturationRunResult] = []
        for smi, mol in zip(smiles_list, molecule_names, strict=True):
            E_mol = ref.get_molecule_energy(mol)
            if E_mol is None:
                _missing_molecule_reference(mol, config)
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
                process_fn=process_fn,
                bo_enabled=bo_enabled,
                activity_by_molecule=activity_by_molecule,
            )
            if run_result is not None:
                all_saturation_results.append(run_result)

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

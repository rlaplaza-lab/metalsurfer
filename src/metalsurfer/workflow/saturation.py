"""Sequential and multi-molecule saturation workflow entry points."""

import logging
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..filters import adsorbate_connected_components
from ..ml.dataset import DatasetLogger
from ..ml.features import extract_features
from ..ml.schema import PlacementRecord
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
from ..placement._material import material_aware_pbc
from ..placement.generators import (
    distribute_placement_budget,
    estimate_conformer_count,
)
from ..reporting import FailureSummary
from ..result_paths import results_dir_for
from ..surface_prep import SlabContainer, apply_material_pbc
from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from .bayesian import process_molecule_bayesian
from .composite import (
    evaluate_composite_commit,
    pack_tuplet_adsorbates,
    select_tuplet_winners,
)
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
        state.prior_step_memory = new_memory
        state.prior_step_memories.append(new_memory)
    if config.bo.transfer.enabled and config.bo.transfer.mode == "cumulative_refit":
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


def _n_at_saturation_from_steps(
    steps: Sequence[SaturationStepResult | MultiMolSaturationStepResult],
) -> int:
    """Total adsorbates folded onto the slab: sum of per-step ``n_added``.

    Bound steps contribute one placement each in legacy mode (n-tuplet steps
    contribute several); an unbound final step contributes zero, so the total
    always equals the number of adsorbates on the returned final slab.
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
    """
    n_per_step = config.saturation_molecules_per_step
    num_placements = config.num_placements
    if n_per_step <= 1 or num_placements is None:
        return config
    scaled = max(1, num_placements // n_per_step)
    if scaled == num_placements:
        return config
    logger.info(
        "n-tuplet mode: dividing probed workload capacity %d by %d -> "
        "num_placements=%d",
        num_placements,
        n_per_step,
        scaled,
    )
    return replace(config, num_placements=scaled)


def _commit_n_tuplet(
    *,
    step: int,
    candidates: Sequence[ScreeningResult],
    slab_atoms: Atoms,
    base_slab: Atoms,
    ts_model: object,
    config: AdsorptionConfig,
    E_slab: float,
    reference_unit_smiles: list[str],
    smiles_by_molecule: Mapping[str, str],
    log_prefix: str,
) -> tuple[list[ScreeningResult], str]:
    """Select and commit up to ``saturation_molecules_per_step`` winners at once.

    Greedy mutual-clearance selection over *candidates* (already screened on
    the current coverage slab), followed by composite relaxation/validation
    via :func:`workflow.composite.evaluate_composite_commit`. On composite
    failure with more than one winner, retries with the best winner alone
    before giving up, so a broken tuplet never loses a step that a single
    placement could have bound.

    Returns
    -------
    tuple[list[ScreeningResult], str]
        ``(rewritten_results, "committed")`` on success, ``([], "no_binders")``
        when no candidate binds or is mutually clear, or
        ``([], "failed")`` when every composite attempt failed validation.
        Callers map these to the legacy outcome conventions upstream.
    """
    winners = select_tuplet_winners(
        candidates,
        cell=slab_atoms.get_cell(),
        pbc=material_aware_pbc(config.material_type),
        min_separation=config.min_adsorbate_separation,
        max_winners=config.saturation_molecules_per_step,
        config=config,
        slab_atoms=slab_atoms,
    )
    if not winners:
        logger.info(
            "%sstep %d: no mutually clear binders; committing nothing this step",
            log_prefix,
            step,
        )
        return [], "no_binders"

    original_best = winners[0]
    packed = pack_tuplet_adsorbates(winners, slab_atoms, config)
    if not packed:
        logger.info(
            "%sstep %d: n-tuplet pack emptied the winner list; committing nothing",
            log_prefix,
            step,
        )
        return [], "no_binders"

    topology_check = None
    if config.saturation_discard_topology_rearrangements:

        def topology_check(
            opt_atoms: Atoms, pending_names: list[str]
        ) -> tuple[bool, str]:
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

    step_log_prefix = f"{log_prefix}step {step} | "
    rewritten, failure = evaluate_composite_commit(
        winners=packed,
        slab_atoms=slab_atoms,
        base_slab=base_slab,
        ts_model=ts_model,
        config=config,
        E_slab=E_slab,
        topology_check=topology_check,
        log_prefix=step_log_prefix,
    )
    if rewritten:
        return rewritten, "committed"
    if len(packed) == 1 and packed[0].placement_id == original_best.placement_id:
        logger.warning(
            "%scomposite validation failed (%s); committing nothing",
            step_log_prefix,
            failure,
        )
        return [], "failed"
    logger.warning(
        "%scomposite validation failed (%s); retrying with best winner alone",
        step_log_prefix,
        failure,
    )
    rewritten, failure = evaluate_composite_commit(
        winners=[original_best],
        slab_atoms=slab_atoms,
        base_slab=base_slab,
        ts_model=ts_model,
        config=config,
        E_slab=E_slab,
        topology_check=topology_check,
        log_prefix=step_log_prefix,
    )
    if not rewritten:
        logger.warning(
            "%ssingle-winner retry failed (%s); committing nothing",
            step_log_prefix,
            failure,
        )
        return [], "failed"
    return rewritten, "committed"


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
    )
    if len(components) != len(reference_unit_smiles):
        return (
            False,
            f"expected {len(reference_unit_smiles)} adsorbate units, "
            f"found {len(components)} connected fragments",
        )

    return True, ""


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


def _committed_placement_features(
    result: ScreeningResult,
    *,
    smiles: str,
    surface_id: str,
    config: AdsorptionConfig,
) -> dict[str, float]:
    """Feature row for a placement that was committed onto the slab."""
    record = PlacementRecord.from_screening_result(
        result,
        smiles=smiles,
        surface_id=surface_id,
        config=config,
    )
    return extract_features(record)


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
    occupancy_placement_X: list[dict[str, float]] | None = None,
) -> tuple[
    list[ScreeningResult], BOTransferInfo, BOStepMemory | None, list[PlacementRecord]
]:
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
        "conformer_energies": conformer_energies,
        "skip_workload_autotune": skip_workload_autotune,
    }
    if bo_enabled:
        outcome = process_fn(
            smiles,
            molecule_name,
            current_slab,
            calculator,
            ref_step,
            bo_step_memory_in=(
                _bo_transfer_memory_in(config, bo_state)
                if bo_state is not None
                else None
            ),
            occupancy_placement_X=occupancy_placement_X,
            saturation_reuse=True,
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
        failure_summary_out[molecule_name] = outcome.failure_summary

    filtered = _filter_saturation_topology_results(
        list(outcome.results),
        base_slab_len=len(base_slab),
        reference_unit_smiles=reference_unit_smiles,
        config=config,
    )
    return filtered, transfer_info, new_memory, list(outcome.ml_records)


def _saturation_should_stop(
    *,
    best_energy: float,
    step: int,
    config: AdsorptionConfig,
    log_prefix: str,
) -> bool:
    """Stop when the step's best E_ads is non-negative or max steps is reached.

    n-tuplet semantics: ``best_energy`` is the committed tuplet's shared total
    E_ads (negative whenever the composite binds), so a tuplet step keeps the
    run going exactly when it committed at least one binder. A step that
    commits nothing records an unbound final step (``n_added == 0``) upstream
    and stops, preserving the max-steps guarantee that the final slab holds
    exactly ``n_molecules_at_saturation`` adsorbates.
    """
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
    transfer_info: BOTransferInfo


@dataclass(frozen=True)
class _MultiStepPayload:
    """Per-step bookkeeping for competitive multi-molecule saturation."""

    winning_molecule: str
    per_molecule_results: dict[str, list[ScreeningResult]]
    budgets: dict[str, int]
    transfer_by_molecule: dict[str, BOTransferInfo]


@dataclass(frozen=True)
class _StepScreenOutcome:
    """Result of one saturation step's screening phase.

    ``committed`` lists the placements folded into the coverage slab this step
    (one element in legacy sequential mode; several for n-tuplet steps).
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
        # ``n_molecules_at_saturation`` adsorbates. Legacy steps commit exactly
        # one placement; n-tuplet steps iterate several here.
        for placement in outcome.committed:
            current_slab = _slab_after_saturation_step(placement.atoms, config)
        n_on_slab += len(outcome.committed)

        if _saturation_should_stop(
            best_energy=outcome.best.energy_adsorption,
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
    ds_logger: DatasetLogger,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_enabled: bool,
) -> SaturationRunResult | None:
    """Coverage loop for one adsorbate until unbound or max steps."""
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
    committed_placement_X: list[dict[str, float]] = []
    # SMILES of every adsorbate unit folded onto the slab so far (one per
    # committed placement); the screened candidate adds *smiles* on top.
    units_on_slab: list[str] = []

    def screen_step(
        step: int,
        preamble: _SaturationStepPreamble,
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        """Screen placements for one saturation step.

        Parameters
        ----------
        step
            Current step number (1-based).
        preamble
            Precomputed slab energy and symmetry state.
        slab
            Current slab container.
        """
        nonlocal config
        symmetry_broken = preamble.symmetry_broken
        if needs_workload_autotune(config, bo=bo_enabled):
            if cached_conformers is None:
                raise ValueError(
                    "conformers required to resolve saturation workload config"
                )
            slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab)
            config = _scale_budget_for_tuplet(
                resolve_saturation_step_workload_config(
                    config,
                    ts_model=ts_model,
                    conformers=cached_conformers,
                    slab_atoms=slab.atoms,
                    slab_for_sites=slab_for_sites,
                    smiles=smiles,
                    base_slab_for_frozen=base_slab,
                    symmetry_broken=symmetry_broken,
                    bo_enabled=bo_enabled,
                )
            )
        mol_results, transfer_info, new_memory, ml_records = (
            _screen_saturation_molecule(
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
                occupancy_placement_X=committed_placement_X or None,
            )
        )
        for record in ml_records:
            ds_logger.add_record(record)
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
        committed = [best] if best.energy_adsorption < 0 else []
        if config.saturation_molecules_per_step > 1:
            # n-tuplet step: greedy mutual-clearance selection + one composite
            # relaxation covering all winners (legacy pools, new commit path).
            committed, commit_status = _commit_n_tuplet(
                step=step,
                candidates=mol_results,
                slab_atoms=slab.atoms,
                base_slab=base_slab,
                ts_model=ts_model,
                config=config,
                E_slab=preamble.E_slab,
                reference_unit_smiles=list(units_on_slab),
                smiles_by_molecule={molecule: smiles},
                log_prefix=f"Saturation for {molecule} | ",
            )
            if commit_status == "failed":
                logger.warning(
                    "Step %d: n-tuplet composite validation failed for %s; "
                    "stopping saturation",
                    step,
                    molecule,
                )
                return None
        if committed:
            best = min(
                committed,
                key=lambda r: (r.energy_adsorption, r.placement_id, r.molecule),
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
        """Record the results of one saturation step.

        Parameters
        ----------
        step
            Current step number (1-based).
        n_on_slab
            Adsorbate units already folded onto the slab before this step.
        outcome
            Screening outcome to record.
        """
        payload = outcome.payload
        assert isinstance(payload, _SingleStepPayload)
        steps.append(
            SaturationStepResult(
                step=step,
                molecule=molecule,
                n_molecules_on_slab=n_on_slab,
                best_result=outcome.best,
                all_results=payload.mol_results,
                bo_transfer_enabled=bool(config.bo.transfer.enabled),
                transfer=payload.transfer_info,
                n_added=len(outcome.committed),
                committed_results=outcome.committed,
            )
        )
        ds_logger.add_results(
            payload.mol_results, smiles=smiles, surface_id=surface_type
        )
        for placement in outcome.committed:
            units_on_slab.append(smiles)
            committed_placement_X.append(
                _committed_placement_features(
                    placement,
                    smiles=smiles,
                    surface_id=surface_type,
                    config=config,
                )
            )
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
    failure_summary_out: dict[str, FailureSummary] | None,
    ds_logger: DatasetLogger,
    *,
    process_fn: Callable[..., MoleculeScreenOutcome],
    bo_enabled: bool,
) -> MultiMolSaturationRunResult:
    """Run a competitive multi-molecule saturation loop."""
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
    committed_placement_X: list[dict[str, float]] = []

    def screen_step(
        step: int,
        preamble: _SaturationStepPreamble,
        slab: SlabContainer,
    ) -> _StepScreenOutcome | None:
        """Screen placements for one multi-molecule saturation step.

        Parameters
        ----------
        step
            Current step number (1-based).
        preamble
            Precomputed slab energy and symmetry state.
        slab
            Current slab container.
        """
        nonlocal config
        symmetry_broken = preamble.symmetry_broken

        E_slab = preamble.E_slab
        ref_step = preamble.ref_step

        slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab)
        if needs_workload_autotune(config, bo=bo_enabled):
            largest_conformers, _ = conformer_cache[largest_mol]
            step_config = _scale_budget_for_tuplet(
                resolve_saturation_step_workload_config(
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
            )
            config = step_config
        else:
            step_config = config

        step_complexities: dict[str, float] = {}
        for mol in active_molecules:
            confs, _ = conformer_cache[mol]
            step_complexities[mol] = estimate_conformer_count(confs)
        budget_inputs = {m: c for m, c in step_complexities.items() if c > 0.0}
        if not budget_inputs:
            logger.warning(
                "Step %d: no molecules with available sites under coverage; stopping",
                step,
            )
            return None
        num_placements = step_config.num_placements
        if num_placements is None:
            raise ValueError("num_placements must be resolved before saturation steps")
        budgets = distribute_placement_budget(
            budget_inputs,
            num_placements,
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

            resolved, transfer_info, new_memory, ml_records = (
                _screen_saturation_molecule(
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
                    occupancy_placement_X=committed_placement_X or None,
                )
            )
            for record in ml_records:
                ds_logger.add_record(record)
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
        # Legacy competitive steps commit exactly one winner per step;
        # n-tuplet steps commit up to ``saturation_molecules_per_step``
        # mutually clear winners via one composite relaxation.
        if step_config.saturation_molecules_per_step > 1:
            committed, commit_status = _commit_n_tuplet(
                step=step,
                candidates=all_results_flat,
                slab_atoms=slab.atoms,
                base_slab=base_slab,
                ts_model=ts_model,
                config=step_config,
                E_slab=E_slab,
                reference_unit_smiles=_reference_smiles_units_multi_molecule(
                    active_molecules,
                    active_smiles,
                    molecule_counts,
                    pending_additions={},
                ),
                smiles_by_molecule=active_smiles,
                log_prefix="Multi-mol saturation | ",
            )
            if commit_status == "failed":
                logger.warning(
                    "Multi-mol saturation step %d: n-tuplet composite "
                    "validation failed for every candidate; stopping",
                    step,
                )
                return None
        else:
            committed = [best_overall] if best_overall.energy_adsorption < 0 else []
        if committed:
            best_overall = min(
                committed,
                key=lambda r: (r.energy_adsorption, r.placement_id, r.molecule),
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
        """Record the results of one multi-molecule saturation step.

        Parameters
        ----------
        step
            Current step number (1-based).
        n_on_slab
            Adsorbate units already folded onto the slab before this step.
        outcome
            Screening outcome to record.
        """
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
                bo_transfer_enabled=bool(config.bo.transfer.enabled),
                transfer_by_molecule=dict(payload.transfer_by_molecule),
                n_added=len(committed),
                committed_results=committed,
            )
        )
        # Fold every committed winner (one per legacy step; several per future
        # n-tuplet step) into the coverage bookkeeping.
        for molecule_name, count in Counter(
            placement.molecule for placement in committed
        ).items():
            molecule_counts[molecule_name] += count
        for placement in committed:
            winning_smiles = active_smiles[placement.molecule]
            committed_placement_X.append(
                _committed_placement_features(
                    placement,
                    smiles=winning_smiles,
                    surface_id=surface_type,
                    config=config,
                )
            )

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
    failure_summary_out: dict[str, FailureSummary] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
    *,
    bo_enabled: bool = False,
) -> list[SaturationRunResult] | list[MultiMolSaturationRunResult]:
    """Sequential saturation: add molecules until best E_ads >= 0.

    Parameters
    ----------
    slab
        Substrate structure.
    molecules
        In-memory ``(smiles, name)`` list/tuple or path to a two-column CSV.
    config
        Adsorption configuration.
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
        When True, each step uses Bayesian placement selection (and optional
        cross-step transfer). Prefer :func:`~metalsurfer.run_saturation_bo` at
        the campaign layer.

    Notes
    -----
    With ``config.saturation_molecules_per_step > 1`` (n-tuplet mode), each
    step greedily commits up to that many mutually compatible winners
    simultaneously: per-molecule pools are screened as usual, winners are
    selected by ascending E_ads subject to pairwise
    ``min_adsorbate_separation`` clearance, and ONE composite candidate is
    relaxed per step. Each committed row carries the full tuplet E_ads (see
    ``workflow/composite.py`` for the agreed representation).
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
                summary_csv = (
                    results_dir_for(surface_type) / "saturation_summary.csv"
                ).as_posix()
                logger.warning(
                    "No molecules to process: all already listed in %s. "
                    "Set skip_existing=False or remove that CSV to rerun",
                    summary_csv,
                )
            elif load_status == "empty_file":
                logger.warning("No molecules to process: file empty or no valid rows")
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
        results_dir = results_dir_for(surface_type).as_posix()
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
                bo_enabled=bo_enabled,
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
                bo_enabled=bo_enabled,
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

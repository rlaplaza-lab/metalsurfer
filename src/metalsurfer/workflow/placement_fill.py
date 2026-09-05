"""Yield-aware placement fill shared by non-BO and BO screening."""

import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement._constants import _RETRY_BLOCK_SITE_AFTER, RECOVERABLE_DISTANCE_REASONS
from ..placement.generators import (
    enumerate_placement_specs,
    estimate_placement_capacity,
)
from ..placement.site_context import SiteContext
from .shared import PlacementFailureEvent, _materialize_spec_placements

logger = logging.getLogger(__name__)

# Prior success rate before the first attempt observes real yield.
_YIELD_EST_PRIOR = 0.5


def placement_spec_key(
    spec: PlacementSpec,
) -> tuple[
    int,
    str,
    int,
    float,
    float,
    float,
    float,
    bool,
    int | None,
]:
    """Hashable identity for retry diversity (exclude known-bad specs).

    Parameters
    ----------
    spec
        Placement specification.
    """
    return (
        spec.conformer_index,
        spec.orientation_type,
        spec.site_index,
        float(spec.z_fraction),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        bool(spec.face_flip),
        spec.en_atom_index,
    )


def placement_cell_key(
    spec: PlacementSpec,
) -> tuple[int, str, int, bool, int | None]:
    """Discrete placement neighborhood (excludes continuous pose params).

    Two specs in the same cell share conformer/orientation/site/face/EN-slot;
    they differ only in continuous z/tilt/azimuth, which is why re-seeding the
    retry sampler reproduces the same failing neighborhood. Used to avoid
    re-relaxing already-explored cells.

    Parameters
    ----------
    spec
        Placement specification.
    """
    return (
        spec.conformer_index,
        spec.orientation_type,
        spec.site_index,
        bool(spec.face_flip),
        spec.en_atom_index,
    )


def _clamp_target_to_capacity(
    *,
    n_target: int,
    conformers: list[Atoms],
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_atoms: Atoms,
    log_label: str = "",
) -> int:
    """R1: clamp a fill target to the enumerable placement-spec capacity.

    Returns ``n_target`` unchanged when the clamp is disabled or capacity is
    already sufficient; otherwise returns the (non-negative) capacity so the
    retry loop cannot spin until ``max_attempts`` on an unreachable target.
    """
    if not config.placement_fill_clamp_to_capacity:
        return n_target

    capacity = estimate_placement_capacity(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
        full_slab=slab_atoms,
    )
    capacity_int = max(int(math.floor(capacity)), 0)
    if capacity_int >= n_target:
        return n_target

    logger.warning(
        "Placement fill target%s clamped from %d to %d: enumerable spec "
        "capacity exhausted (material_type=%s)",
        log_label,
        n_target,
        capacity_int,
        config.material_type,
    )
    return capacity_int


def _request_count(remaining: int, yield_est: float, oversample_max: float) -> int:
    """How many specs to enumerate given deficit and estimated materialization yield."""
    if remaining <= 0:
        return 0
    yield_floor = _yield_floor(oversample_max)
    effective_yield = max(float(yield_est), yield_floor)
    by_yield = int(math.ceil(remaining / effective_yield))
    by_cap = int(math.ceil(remaining * float(oversample_max)))
    return max(remaining, min(by_yield, by_cap))


def _yield_floor(oversample_max: float) -> float:
    return 1.0 / max(float(oversample_max), 1.0)


def _absorb_chunk(
    *,
    combined: list[Atoms],
    placement_ids: list[int],
    descriptors: list[PlacementDescriptor],
    failures: list[PlacementFailureEvent],
    new_combined: list[Atoms],
    new_ids: list[int],
    new_descriptors: list[PlacementDescriptor],
    new_failures: list[PlacementFailureEvent],
    n_target: int,
    n_tried: int,
    yield_est: float,
    yield_floor: float,
) -> tuple[float, int]:
    """Record failures, update yield estimate, and trim successes into room.

    Returns ``(updated_yield_est, n_taken)``.
    """
    failures.extend(new_failures)
    n_ok = len(new_combined)
    if n_tried > 0:
        yield_est = max(n_ok / n_tried, yield_floor)
    room = n_target - len(combined)
    take = min(room, n_ok)
    if take:
        combined.extend(new_combined[:take])
        placement_ids.extend(new_ids[:take])
        descriptors.extend(new_descriptors[:take])
    return yield_est, take


@dataclass
class MaterializeFillResult:
    """Outcome of materializing specs up to a target count.

    ``n_attempts`` counts enumeration/materialization rounds: always ``1``
    for :func:`materialize_specs_filling_target` (single primary pass plus
    yield-sized backfill chunks), and the real retry-round count for
    :func:`fill_materialized_placements`.
    """

    combined: list[Atoms]
    placement_ids: list[int]
    descriptors: list[PlacementDescriptor]
    failures: list[PlacementFailureEvent]
    n_attempts: int = 0


def materialize_specs_filling_target(
    *,
    primary_specs: Sequence[PlacementSpec],
    backfill_specs: Sequence[PlacementSpec],
    n_target: int,
    conformers: list[Atoms],
    slab_atoms: Atoms,
    calculator,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
) -> MaterializeFillResult:
    """Materialize ``primary_specs``, then backfill until ``n_target`` successes.

    Stops early when the target is met or both primary and backfill are exhausted.
    Extra successes beyond ``n_target`` are discarded (oversampling trim).
    Backfill chunks oversample by estimated materialization yield (capped by
    ``placement_retry_oversample_max``).

    Parameters
    ----------
    primary_specs
        Primary placement specs to materialize.
    backfill_specs
        Backfill specs used when primary is insufficient.
    n_target
        Target number of valid placements.
    conformers
        List of conformer geometries.
    slab_atoms
        Full slab atoms.
    calculator
        ASE calculator instance.
    config
        Adsorption configuration.
    smiles
        SMILES string of the molecule.
    site_context
        Resolved site context for sampling.
    slab_for_sites
        Substrate reference for site enumeration.
    materialization_cache
        Cache for spec materialization.
    """
    if n_target <= 0:
        return MaterializeFillResult([], [], [], [], n_attempts=0)

    n_target = _clamp_target_to_capacity(
        n_target=n_target,
        conformers=conformers,
        slab_for_sites=slab_for_sites if slab_for_sites is not None else slab_atoms,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
        log_label=" (BO)",
    )

    combined: list[Atoms] = []
    placement_ids: list[int] = []
    descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []
    oversample_max = float(config.placement_retry_oversample_max)
    yield_floor = _yield_floor(oversample_max)
    yield_est = _YIELD_EST_PRIOR

    def _materialize_and_absorb(
        specs: Sequence[PlacementSpec],
    ) -> tuple[int, int]:
        """Materialize ``specs``; append until target. Returns (n_tried, n_taken)."""
        nonlocal yield_est
        if not specs or len(combined) >= n_target:
            return 0, 0
        new_combined, new_ids, new_descs, new_failures = _materialize_spec_placements(
            specs=list(specs),
            conformers=conformers,
            slab_atoms=slab_atoms,
            calculator=calculator,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
            materialization_cache=materialization_cache,
        )
        yield_est, take = _absorb_chunk(
            combined=combined,
            placement_ids=placement_ids,
            descriptors=descriptors,
            failures=failures,
            new_combined=new_combined,
            new_ids=new_ids,
            new_descriptors=new_descs,
            new_failures=new_failures,
            n_target=n_target,
            n_tried=len(specs),
            yield_est=yield_est,
            yield_floor=yield_floor,
        )
        return len(specs), take

    _materialize_and_absorb(primary_specs)

    if len(combined) < n_target and backfill_specs:
        offset = 0
        while len(combined) < n_target and offset < len(backfill_specs):
            remaining = n_target - len(combined)
            n_request = _request_count(remaining, yield_est, oversample_max)
            chunk = list(backfill_specs[offset : offset + n_request])
            if not chunk:
                break
            # A fully failed chunk just advances *offset* so the next
            # iteration draws fresh backfill specs.
            offset += _materialize_and_absorb(chunk)[0]

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_attempts=1,
    )


def fill_materialized_placements(
    *,
    conformers: list[Atoms],
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_atoms: Atoms,
    calculator,
    conformer_energies: list[float] | None = None,
) -> MaterializeFillResult:
    """Enumerate and materialize until ``n_target`` valid placements or retries end.

    Each deficit round oversamples by estimated materialization yield (capped by
    ``placement_retry_oversample_max``), excludes failed spec keys, and blocks
    sites that repeatedly clash. Early-exits on fill or empty enumeration.

    Parameters
    ----------
    conformers
        List of conformer geometries.
    slab_for_sites
        Substrate reference for site enumeration.
    config
        Adsorption configuration.
    smiles
        SMILES string of the molecule.
    site_context
        Resolved site context for sampling.
    slab_atoms
        Full slab atoms.
    calculator
        ASE calculator instance.
    conformer_energies
        Energies aligned with conformers (optional).
    """
    n_target = config.num_placements
    if n_target is None:
        raise ValueError("num_placements must be set before materializing placements")

    # R1: clamp the success goal to the enumerable spec capacity so the retry
    # loop cannot spin until max_attempts on a target that is unreachable.
    # `n_target` keeps the original request (used for placement-index offsets);
    # `effective_target` is the clamped success goal.
    effective_target = _clamp_target_to_capacity(
        n_target=n_target,
        conformers=conformers,
        slab_for_sites=slab_for_sites,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
    )

    combined: list[Atoms] = []
    placement_ids: list[int] = []
    descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []
    failed_keys: set[tuple] = set()
    # R3: discrete placement neighborhoods already relaxed (failures only).
    tried_cells: set[tuple] = set()
    site_fail_counts: Counter[int] = Counter()
    blocked_sites: set[int] = set()
    last_spec_by_index: dict[int, PlacementSpec] = {}

    max_attempts = (
        config.placement_retry_max_attempts if config.placement_retry_enabled else 1
    )
    seed_increment = config.placement_retry_diversity_seed_increment
    oversample_max = float(config.placement_retry_oversample_max)
    yield_floor = _yield_floor(oversample_max)
    yield_est = _YIELD_EST_PRIOR
    attempts_used = 0
    next_placement_index = 0
    # R2: consecutive zero-yield attempts (no new placements absorbed).
    consecutive_zero_yield = 0

    def _make_spec_filter(*, check_failed: bool, check_cells: bool = False):
        def _filter(
            spec,
            *,
            _failed=failed_keys,
            _blocked=blocked_sites,
            _cells=tried_cells,
        ):
            if check_failed and placement_spec_key(spec) in _failed:
                return False
            if check_cells and placement_cell_key(spec) in _cells:
                return False
            if int(spec.site_index) in _blocked:
                return False
            if config.placement_filter is not None:
                return bool(config.placement_filter(spec))
            return True

        return _filter

    for attempt in range(max_attempts):
        if len(combined) >= effective_target:
            break

        remaining = effective_target - len(combined)
        if remaining <= 0:
            break

        n_request = _request_count(remaining, yield_est, oversample_max)
        attempt_seed = config.seed + (seed_increment * attempt)
        attempts_used = attempt + 1

        # R3: on retry attempts also exclude already-tried discrete cells so a
        # new seed explores fresh neighborhoods instead of re-relaxing failures.
        specs = enumerate_placement_specs(
            conformers,
            slab_for_sites,
            config,
            smiles,
            n_request,
            filter_spec=_make_spec_filter(check_failed=True, check_cells=(attempt > 0)),
            site_context=site_context,
            seed=attempt_seed,
            full_slab=slab_atoms,
            conformer_energies=conformer_energies,
        )

        # R4: if the failed-key/cell filter emptied the pool, relax the block
        # partially (unblock the least-clashing sites) before the ultimate
        # unfiltered fallback, which the plan keeps as a final safety net.
        if not specs and failed_keys and remaining > 0:
            if blocked_sites:
                ranked = sorted(blocked_sites, key=lambda s: (site_fail_counts[s], s))
                unblock_k = max(1, len(ranked) // 2)
                for site_idx in ranked[:unblock_k]:
                    blocked_sites.discard(site_idx)
                logger.info(
                    "Placement retry attempt %d: failed-key filter emptied the pool; "
                    "partially unblocking %d least-clashing site(s) (kept %d blocked)",
                    attempt + 1,
                    unblock_k,
                    len(blocked_sites),
                )
                specs = enumerate_placement_specs(
                    conformers,
                    slab_for_sites,
                    config,
                    smiles,
                    n_request,
                    filter_spec=_make_spec_filter(check_failed=True, check_cells=False),
                    site_context=site_context,
                    seed=attempt_seed + 1,
                    full_slab=slab_atoms,
                    conformer_energies=conformer_energies,
                )

            if not specs:
                logger.warning(
                    "Placement retry attempt %d: pool still empty after partial "
                    "unblock; falling back to unfiltered enumeration once "
                    "(blocked sites kept)",
                    attempt + 1,
                )
                specs = enumerate_placement_specs(
                    conformers,
                    slab_for_sites,
                    config,
                    smiles,
                    n_request,
                    filter_spec=_make_spec_filter(
                        check_failed=False, check_cells=False
                    ),
                    site_context=site_context,
                    seed=attempt_seed + 2,
                    full_slab=slab_atoms,
                    conformer_energies=conformer_energies,
                )

        if not specs:
            break

        for spec in specs:
            spec.placement_index = next_placement_index
            next_placement_index += 1
            last_spec_by_index[spec.placement_index] = spec

        (
            new_combined,
            new_ids,
            new_descriptors,
            new_failures,
        ) = _materialize_spec_placements(
            specs=specs,
            conformers=conformers,
            slab_atoms=slab_atoms,
            calculator=calculator,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
        )

        for fail in new_failures:
            failed_spec = last_spec_by_index.get(fail.placement_id)
            if failed_spec is not None:
                failed_keys.add(placement_spec_key(failed_spec))
                tried_cells.add(placement_cell_key(failed_spec))
                site_idx = int(failed_spec.site_index)
                if site_idx >= 0 and fail.reason in RECOVERABLE_DISTANCE_REASONS:
                    site_fail_counts[site_idx] += 1
                    if site_fail_counts[site_idx] >= _RETRY_BLOCK_SITE_AFTER:
                        blocked_sites.add(site_idx)

        yield_est, take = _absorb_chunk(
            combined=combined,
            placement_ids=placement_ids,
            descriptors=descriptors,
            failures=failures,
            new_combined=new_combined,
            new_ids=new_ids,
            new_descriptors=new_descriptors,
            new_failures=new_failures,
            n_target=effective_target,
            n_tried=len(specs),
            yield_est=yield_est,
            yield_floor=yield_floor,
        )

        if attempt > 0 and take:
            logger.debug(
                "Retry attempt %d/%d: generated %d new placements (total: %d/%d)",
                attempt + 1,
                max_attempts,
                take,
                len(combined),
                effective_target,
            )

        # R2: a zero-yield attempt (no new placements absorbed) is a plateau
        # signal; give up early after `patience` such attempts. max_attempts
        # remains the absolute hard cap.
        if take == 0:
            consecutive_zero_yield += 1
        else:
            consecutive_zero_yield = 0
        if (
            consecutive_zero_yield >= config.placement_retry_early_stop_patience
            and len(combined) < effective_target
        ):
            logger.info(
                "Placement fill early-stop after %d consecutive zero-yield attempts "
                "(capacity likely exhausted; got %d/%d)",
                consecutive_zero_yield,
                len(combined),
                effective_target,
            )
            break

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_attempts=attempts_used if attempts_used > 0 else (1 if max_attempts else 0),
    )

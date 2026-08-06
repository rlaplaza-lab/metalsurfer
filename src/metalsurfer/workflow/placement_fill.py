"""Yield-aware placement fill shared by non-BO and BO screening."""


import logging
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement._constants import _RETRY_BLOCK_SITE_AFTER
from ..placement.generators import enumerate_placement_specs
from ..placement.site_context import SiteContext
from .shared import PlacementFailureEvent, _materialize_spec_placements

logger = logging.getLogger(__name__)

# Prior success rate before the first attempt observes real yield.
_YIELD_EST_PRIOR = 0.5
_CLASH_REASONS = frozenset({"adsorbate_overlap", "too_close"})


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
    """Hashable identity for retry diversity (exclude known-bad specs)."""
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


def _request_count(remaining: int, yield_est: float, oversample_max: float) -> int:
    """How many specs to enumerate given deficit and estimated materialization yield."""
    if remaining <= 0:
        return 0
    # Floor so ceil(remaining / floor) never exceeds remaining * oversample_max.
    yield_floor = 1.0 / max(float(oversample_max), 1.0)
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
    """Outcome of materializing specs up to a target count."""

    combined: list[Atoms]
    placement_ids: list[int]
    descriptors: list[PlacementDescriptor]
    failures: list[PlacementFailureEvent]
    n_backfill_used: int = 0
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
    """
    if n_target <= 0:
        return MaterializeFillResult([], [], [], [], n_backfill_used=0, n_attempts=0)

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

    n_backfill_used = 0
    if len(combined) < n_target and backfill_specs:
        offset = 0
        while len(combined) < n_target and offset < len(backfill_specs):
            remaining = n_target - len(combined)
            n_request = _request_count(remaining, yield_est, oversample_max)
            chunk = list(backfill_specs[offset : offset + n_request])
            if not chunk:
                break
            tried, ok = _materialize_and_absorb(chunk)
            n_backfill_used += tried
            offset += tried
            if tried > 0 and ok == 0:
                # Entire chunk failed; advance and try further backfill.
                continue

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_backfill_used=n_backfill_used,
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
    n_target: int | None = None,
) -> MaterializeFillResult:
    """Enumerate and materialize until ``n_target`` valid placements or retries end.

    Each deficit round oversamples by estimated materialization yield (capped by
    ``placement_retry_oversample_max``), excludes failed spec keys, and blocks
    sites that repeatedly clash. Early-exits on fill or empty enumeration.
    """
    if n_target is None:
        assert config.num_placements is not None
        n_target = config.num_placements

    combined: list[Atoms] = []
    placement_ids: list[int] = []
    descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []
    failed_keys: set[tuple] = set()
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

    for attempt in range(max_attempts):
        if len(combined) >= n_target:
            break

        remaining = n_target - len(combined)
        if remaining <= 0:
            break

        n_request = _request_count(remaining, yield_est, oversample_max)
        attempt_seed = config.seed + (seed_increment * attempt)
        attempts_used = attempt + 1

        def _composed_filter(
            spec,
            *,
            _failed=failed_keys,
            _blocked=blocked_sites,
        ):
            if placement_spec_key(spec) in _failed:
                return False
            if int(spec.site_index) in _blocked:
                return False
            if config.placement_filter is not None:
                return bool(config.placement_filter(spec))
            return True

        specs = enumerate_placement_specs(
            conformers,
            slab_for_sites,
            config,
            smiles,
            n_request,
            filter_spec=_composed_filter,
            site_context=site_context,
            seed=attempt_seed,
            full_slab=slab_atoms,
        )
        if not specs and failed_keys and remaining > 0:
            logger.warning(
                "Placement retry attempt %d: failed-key filter emptied the pool; "
                "falling back to unfiltered enumeration once (blocked sites kept)",
                attempt + 1,
            )

            def _fallback_filter(spec, *, _blocked=blocked_sites):
                if int(spec.site_index) in _blocked:
                    return False
                if config.placement_filter is not None:
                    return bool(config.placement_filter(spec))
                return True

            specs = enumerate_placement_specs(
                conformers,
                slab_for_sites,
                config,
                smiles,
                n_request,
                filter_spec=_fallback_filter,
                site_context=site_context,
                seed=attempt_seed + 1,
                full_slab=slab_atoms,
            )

        if not specs:
            break

        id_offset = attempt * n_target
        for spec in specs:
            spec.placement_index = int(spec.placement_index) + id_offset
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
                site_idx = int(failed_spec.site_index)
                if site_idx >= 0 and fail.reason in _CLASH_REASONS:
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
            n_target=n_target,
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
                n_target,
            )

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_backfill_used=0,
        n_attempts=attempts_used if attempts_used > 0 else (1 if max_attempts else 0),
    )

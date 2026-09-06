"""One-shot placement fill shared by non-BO and BO screening."""

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement.generators import (
    enumerate_placement_specs,
    estimate_placement_capacity,
)
from ..placement.site_context import SiteContext
from .shared import PlacementFailureEvent, _materialize_spec_placements

logger = logging.getLogger(__name__)


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
    """Hashable identity for excluding known-bad specs on the diversity retry."""
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


def _estimate_capacity_int(
    *,
    conformers: list[Atoms],
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_atoms: Atoms,
) -> int:
    return max(
        int(
            math.floor(
                estimate_placement_capacity(
                    conformers,
                    slab_for_sites,
                    config,
                    smiles,
                    site_context=site_context,
                    full_slab=slab_atoms,
                )
            )
        ),
        0,
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
    capacity: int | None = None,
    log_label: str = "",
) -> int:
    """Clamp *n_target* to enumerable capacity when clamping is enabled."""
    if not config.placement_fill_clamp_to_capacity:
        return n_target
    if capacity is None:
        capacity = _estimate_capacity_int(
            conformers=conformers,
            slab_for_sites=slab_for_sites,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_atoms=slab_atoms,
        )
    if capacity >= n_target:
        return n_target
    logger.warning(
        "Placement fill target%s clamped from %d to %d: enumerable spec "
        "capacity exhausted (material_type=%s)",
        log_label,
        n_target,
        capacity,
        config.material_type,
    )
    return capacity


def _pool_request_count(
    n_target: int,
    oversample_max: float,
    *,
    capacity: int | None = None,
) -> int:
    """Specs to enumerate: ``n_target * oversample``, optionally capped by capacity."""
    if n_target <= 0:
        return 0
    requested = max(n_target, int(math.ceil(n_target * float(oversample_max))))
    if capacity is None:
        return requested
    return max(0, min(requested, capacity))


@dataclass
class MaterializeFillResult:
    """Outcome of materializing specs up to a target count."""

    combined: list[Atoms]
    placement_ids: list[int]
    descriptors: list[PlacementDescriptor]
    failures: list[PlacementFailureEvent]
    n_attempts: int = 0


def materialize_specs(
    *,
    specs: Sequence[PlacementSpec],
    n_target: int,
    conformers: list[Atoms],
    slab_atoms: Atoms,
    calculator,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
    clamp_log_label: str = "",
) -> MaterializeFillResult:
    """Materialize *specs* once and keep up to *n_target* successes."""
    if n_target <= 0 or not specs:
        return MaterializeFillResult([], [], [], [], n_attempts=0)

    n_target = _clamp_target_to_capacity(
        n_target=n_target,
        conformers=conformers,
        slab_for_sites=slab_for_sites if slab_for_sites is not None else slab_atoms,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
        log_label=clamp_log_label,
    )
    if n_target <= 0:
        return MaterializeFillResult([], [], [], [], n_attempts=0)

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
    take = min(n_target, len(new_combined))
    return MaterializeFillResult(
        combined=new_combined[:take],
        placement_ids=new_ids[:take],
        descriptors=new_descs[:take],
        failures=list(new_failures),
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
    """Enumerate an oversized pool once, materialize, and take up to ``n_target``.

    Pool size is ``min(capacity, n_target * placement_retry_oversample_max)``.
    When ``placement_retry_enabled`` and the first pass is short, one diversity
    round re-enumerates excluding exact failed-spec keys.
    """
    n_target = config.num_placements
    if n_target is None:
        raise ValueError("num_placements must be set before materializing placements")

    capacity_int = _estimate_capacity_int(
        conformers=conformers,
        slab_for_sites=slab_for_sites,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
    )
    effective_target = _clamp_target_to_capacity(
        n_target=n_target,
        conformers=conformers,
        slab_for_sites=slab_for_sites,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
        capacity=capacity_int,
    )
    if effective_target <= 0:
        return MaterializeFillResult([], [], [], [], n_attempts=0)

    oversample_max = float(config.placement_retry_oversample_max)
    pool_capacity = capacity_int if config.placement_fill_clamp_to_capacity else None

    combined: list[Atoms] = []
    placement_ids: list[int] = []
    descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []
    failed_keys: set[tuple] = set()
    last_spec_by_index: dict[int, PlacementSpec] = {}
    next_placement_index = 0
    attempts_used = 0

    def _filter_failed(spec: PlacementSpec) -> bool:
        if placement_spec_key(spec) in failed_keys:
            return False
        if config.placement_filter is not None:
            return bool(config.placement_filter(spec))
        return True

    def _run_round(*, n_request: int, seed: int, exclude_failed: bool) -> None:
        nonlocal next_placement_index, attempts_used
        if n_request <= 0 or len(combined) >= effective_target:
            return

        specs = enumerate_placement_specs(
            conformers,
            slab_for_sites,
            config,
            smiles,
            n_request,
            filter_spec=_filter_failed if exclude_failed else config.placement_filter,
            site_context=site_context,
            seed=seed,
            full_slab=slab_atoms,
            conformer_energies=conformer_energies,
        )
        if not specs:
            return

        for spec in specs:
            spec.placement_index = next_placement_index
            next_placement_index += 1
            last_spec_by_index[spec.placement_index] = spec

        attempts_used += 1
        new_combined, new_ids, new_descriptors, new_failures = (
            _materialize_spec_placements(
                specs=specs,
                conformers=conformers,
                slab_atoms=slab_atoms,
                calculator=calculator,
                config=config,
                smiles=smiles,
                site_context=site_context,
                slab_for_sites=slab_for_sites,
            )
        )
        for fail in new_failures:
            failed_spec = last_spec_by_index.get(fail.placement_id)
            if failed_spec is not None:
                failed_keys.add(placement_spec_key(failed_spec))
        failures.extend(new_failures)

        take = min(effective_target - len(combined), len(new_combined))
        if take:
            combined.extend(new_combined[:take])
            placement_ids.extend(new_ids[:take])
            descriptors.extend(new_descriptors[:take])

    _run_round(
        n_request=_pool_request_count(
            effective_target, oversample_max, capacity=pool_capacity
        ),
        seed=config.seed,
        exclude_failed=False,
    )

    remaining = effective_target - len(combined)
    if remaining > 0 and config.placement_retry_enabled and failed_keys:
        retry_capacity = (
            max(0, capacity_int - len(combined))
            if config.placement_fill_clamp_to_capacity
            else None
        )
        _run_round(
            n_request=_pool_request_count(
                remaining, oversample_max, capacity=retry_capacity
            ),
            seed=config.seed + 1,
            exclude_failed=True,
        )

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_attempts=attempts_used,
    )

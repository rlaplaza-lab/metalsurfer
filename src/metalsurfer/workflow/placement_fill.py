"""One-shot placement fill shared by non-BO and BO screening."""

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement.generators import (
    _spec_grid_info,
    _SpecGridInfo,
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
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
    str | None,
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
        spec.site_type,
        float(spec.z_fraction),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        bool(spec.face_flip),
        spec.en_atom_index,
    )


def _pose_family_key(
    spec: PlacementSpec,
) -> tuple[int, str, int, float, float, float, bool, int | None]:
    """Rigid pose family matching the height-interval cache key (azimuth included)."""
    return (
        int(spec.conformer_index),
        str(spec.orientation_type),
        int(spec.site_index),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        bool(spec.face_flip),
        spec.en_atom_index,
    )


def _same_height_key(
    spec: PlacementSpec,
    *,
    is_void: bool | None = None,
) -> tuple:
    """Identity of specs that place at the same COM.

    On wall sites every ``z_fraction <= 0.5`` is the contact height.
    Fractions above 0.5, and every void or dissociative fraction, stay distinct.
    """
    family = (
        int(spec.conformer_index),
        str(spec.orientation_type),
        int(spec.site_index),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        bool(spec.face_flip),
        spec.en_atom_index,
    )
    zf = float(spec.z_fraction)
    void = bool(is_void) if is_void is not None else False
    if is_void is None and spec.site_type == "pore":
        void = True
    if spec.orientation_type == "dissociative" or void:
        return (*family, zf)
    if zf <= 0.5:
        return (*family, "contact")
    return (*family, zf)


def _dedupe_same_height_specs(
    specs: list[PlacementSpec],
    *,
    site_is_void: Callable[[int], bool] | None = None,
) -> list[PlacementSpec]:
    """Keep one spec per collapsed height, preferring ``z_fraction`` nearest 0.5."""
    best: dict[tuple, int] = {}
    order: list[tuple] = []
    for i, spec in enumerate(specs):
        is_void = (
            None if site_is_void is None else bool(site_is_void(int(spec.site_index)))
        )
        key = _same_height_key(spec, is_void=is_void)
        prev = best.get(key)
        if prev is None:
            best[key] = i
            order.append(key)
            continue
        prev_spec = specs[prev]
        if abs(float(spec.z_fraction) - 0.5) < abs(float(prev_spec.z_fraction) - 0.5):
            best[key] = i
    return [specs[best[key]] for key in order]


@dataclass
class _RetryBans:
    """Reason-aware exclusions for one fill call (never written to SiteContext)."""

    failed_keys: set[tuple] = field(default_factory=set)
    # (conformer_index, site_index) after adsorbate_overlap
    crowded_conformer_sites: set[tuple[int, int]] = field(default_factory=set)
    # orientation/tilt families banned when the pose cannot contact at any height
    contact_families: set[tuple] = field(default_factory=set)
    # wall contact height (z_fraction <= 0.5) and other identical COMs
    same_height_keys: set[tuple] = field(default_factory=set)
    site_is_void: Callable[[int], bool] | None = None

    def _void_for(self, spec: PlacementSpec) -> bool | None:
        if self.site_is_void is None:
            return None
        return bool(self.site_is_void(int(spec.site_index)))

    def record(self, spec: PlacementSpec, reason: str) -> None:
        self.failed_keys.add(placement_spec_key(spec))
        if reason == "adsorbate_overlap":
            self.crowded_conformer_sites.add(
                (int(spec.conformer_index), int(spec.site_index))
            )
            return
        if reason.startswith("insufficient_contact") or reason == "infeasible_pose":
            self.contact_families.add(_pose_family_key(spec))
            return
        if reason in ("too_close", "vdw_overlap"):
            self.same_height_keys.add(
                _same_height_key(spec, is_void=self._void_for(spec))
            )

    def allows(self, spec: PlacementSpec) -> bool:
        if placement_spec_key(spec) in self.failed_keys:
            return False
        if (
            int(spec.conformer_index),
            int(spec.site_index),
        ) in self.crowded_conformer_sites:
            return False
        if _pose_family_key(spec) in self.contact_families:
            return False
        return (
            _same_height_key(spec, is_void=self._void_for(spec))
            not in self.same_height_keys
        )


def _filter_pool_by_bans(
    specs: list[PlacementSpec], bans: _RetryBans
) -> list[PlacementSpec]:
    """Drop specs already banned (applied between chunks of the same pool)."""
    if (
        not bans.failed_keys
        and not bans.crowded_conformer_sites
        and not bans.contact_families
        and not bans.same_height_keys
    ):
        return specs
    return [s for s in specs if bans.allows(s)]


def _estimate_capacity_int(
    *,
    conformers: list[Atoms],
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_atoms: Atoms,
    grid_info: _SpecGridInfo | None = None,
) -> int:
    return max(
        0,
        estimate_placement_spec_capacity(
            conformers,
            slab_for_sites,
            config,
            smiles,
            site_context=site_context,
            full_slab=slab_atoms,
            grid_info=grid_info,
        ),
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
    capacity: int | None = None,
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
        capacity=capacity,
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


def _materialize_pool_in_chunks(
    *,
    specs: list[PlacementSpec],
    n_target: int,
    conformers: list[Atoms],
    slab_atoms: Atoms,
    calculator,
    config: AdsorptionConfig,
    smiles: str,
    site_context: SiteContext | None,
    slab_for_sites: Atoms,
    combined: list[Atoms],
    placement_ids: list[int],
    descriptors: list[PlacementDescriptor],
    failures: list[PlacementFailureEvent],
    bans: _RetryBans,
    last_spec_by_index: dict[int, PlacementSpec],
) -> None:
    """Materialize *specs* in chunks of ~*n_target*; stop once the target is met.

    After each chunk, remaining specs are filtered by *bans* so orientation and
    overlap failures skip siblings in the same pool without a second enumerate.
    """
    if n_target <= 0 or not specs:
        return
    remaining = list(specs)
    chunk_size = max(1, int(n_target))
    while remaining and len(combined) < n_target:
        chunk = remaining[:chunk_size]
        remaining = remaining[chunk_size:]
        new_combined, new_ids, new_descriptors, new_failures = (
            _materialize_spec_placements(
                specs=chunk,
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
                bans.record(failed_spec, str(fail.reason or ""))
        failures.extend(new_failures)

        take = min(n_target - len(combined), len(new_combined))
        if take:
            combined.extend(new_combined[:take])
            placement_ids.extend(new_ids[:take])
            descriptors.extend(new_descriptors[:take])

        remaining = _filter_pool_by_bans(remaining, bans)


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
    """Enumerate an oversized pool once, materialize in chunks, take ``n_target``.

    Pool size is ``min(capacity, n_target * placement_retry_oversample_max)``.
    Specs are materialized in chunks of about ``n_target`` and stop early once
    enough successes exist. Family / overlap bans apply within the pool between
    chunks. When ``placement_retry_enabled``, the first pass is short, and at
    least one spec failed materialization, one diversity round re-enumerates
    excluding those exact failed-spec keys, ``(conformer, site)`` pairs that
    failed with ``adsorbate_overlap``, orientation families after
    ``insufficient_contact_*`` / ``infeasible_pose``, and same-height
    ``z_fraction`` siblings after ``too_close`` / ``vdw_overlap`` (wall-near
    fractions at or below 0.5 share the contact COM; higher fractions stay).
    """
    n_target = config.num_placements
    if n_target is None:
        raise ValueError("num_placements must be set before materializing placements")

    # Occupancy/shape/dissociative inputs are unchanged across estimate + enumerate
    # rounds within one fill; compute once and share.
    grid_info = _spec_grid_info(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context,
        full_slab=slab_atoms,
    )
    capacity_int = _estimate_capacity_int(
        conformers=conformers,
        slab_for_sites=slab_for_sites,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab_atoms,
        grid_info=grid_info,
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

    def _site_is_void(site_idx: int) -> bool:
        if site_context is None or not site_context.sites:
            return False
        if site_idx < 0 or site_idx >= len(site_context.sites):
            return False
        return site_context.sites[site_idx].kind == "void"

    bans = _RetryBans(site_is_void=_site_is_void)
    last_spec_by_index: dict[int, PlacementSpec] = {}
    next_placement_index = 0
    attempts_used = 0

    def _filter_failed(spec: PlacementSpec) -> bool:
        if not bans.allows(spec):
            return False
        if config.placement_filter is not None:
            return bool(config.placement_filter(spec))
        return True

    def _enumerate_and_index(
        *, n_request: int, seed: int, exclude_failed: bool
    ) -> list[PlacementSpec]:
        nonlocal next_placement_index
        if n_request <= 0:
            return []
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
            grid_info=grid_info,
        )
        specs = _dedupe_same_height_specs(specs, site_is_void=_site_is_void)
        for spec in specs:
            spec.placement_index = next_placement_index
            next_placement_index += 1
            last_spec_by_index[spec.placement_index] = spec
        return specs

    pool = _enumerate_and_index(
        n_request=_pool_request_count(
            effective_target, oversample_max, capacity=pool_capacity
        ),
        seed=config.seed,
        exclude_failed=False,
    )

    def _run_chunks(specs: list[PlacementSpec]) -> None:
        nonlocal attempts_used
        if not specs:
            return
        attempts_used += 1
        _materialize_pool_in_chunks(
            specs=specs,
            n_target=effective_target,
            conformers=conformers,
            slab_atoms=slab_atoms,
            calculator=calculator,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
            combined=combined,
            placement_ids=placement_ids,
            descriptors=descriptors,
            failures=failures,
            bans=bans,
            last_spec_by_index=last_spec_by_index,
        )

    _run_chunks(pool)

    remaining = effective_target - len(combined)
    if remaining > 0 and config.placement_retry_enabled and bans.failed_keys:
        retry_capacity = (
            max(0, capacity_int - len(combined))
            if config.placement_fill_clamp_to_capacity
            else None
        )
        _run_chunks(
            _enumerate_and_index(
                n_request=_pool_request_count(
                    remaining, oversample_max, capacity=retry_capacity
                ),
                seed=config.seed + 1,
                exclude_failed=True,
            )
        )

    return MaterializeFillResult(
        combined=combined,
        placement_ids=placement_ids,
        descriptors=descriptors,
        failures=failures,
        n_attempts=attempts_used,
    )

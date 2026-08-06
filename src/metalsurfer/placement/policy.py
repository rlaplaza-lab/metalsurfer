"""Sampler policy for batch-based BO placement spec proposals."""

import itertools
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from ..models import PlacementSpec
from ._constants import (
    _AZIMUTH,
    _AZIMUTH_IN_PLANE,
    _EARLY_CAP_WORKING_SET_MULTIPLIER,
    _GRID_BUILD_CAP,
    _PLACEMENT_GRID_COUNT_SEED,
    _POLICY_PRIOR_TILT_WEIGHT_PER_DEG,
    _POLICY_PRIOR_Z_FRACTION_TARGET,
    _POLICY_PRIOR_Z_FRACTION_WEIGHT,
    _TILT_FULL,
    _TILT_PARALLEL,
    _Z_FRACTIONS,
)


def max_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    shape: str,
    n_binders: int,
    flat_aromatic: bool,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
) -> int:
    """Closed-form count of policy-grid specs (per-branch clamp at ``_GRID_BUILD_CAP``)."""
    n_sites = max(len(site_indices), 1)

    if dissociative:
        if n_hollow_pairs <= 0:
            return 0
        return min(n_hollow_pairs * len(_Z_FRACTIONS), _GRID_BUILD_CAP)

    if flat_aromatic:
        parallel = (
            n_conformers
            * 2
            * len(_TILT_PARALLEL)
            * len(_AZIMUTH)
            * len(_Z_FRACTIONS)
            * len(_AZIMUTH_IN_PLANE)
            * n_sites
        )
        en_down = (
            n_conformers
            * max(n_binders, 1)
            * len(_TILT_FULL)
            * len(_AZIMUTH)
            * len(_Z_FRACTIONS)
            * n_sites
        )
        return min(parallel, _GRID_BUILD_CAP) + min(en_down, _GRID_BUILD_CAP)

    # Shape only selects orientation label ("vertical" vs "round"); grid is the same size for both.
    _ = shape
    return min(
        n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * len(_Z_FRACTIONS) * n_sites,
        _GRID_BUILD_CAP,
    )


def _spec_prior_key(
    spec: PlacementSpec,
    tie: float,
    *,
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
) -> tuple[float, float]:
    """Sort key for soft physical priors: lower score is preferred.

    Prefers milder absolute tilt and ``z_fraction`` near *z_fraction_target*.
    When *site_index_weight* > 0 (porous open-pore lists), lower ``site_index``
    is preferred so quality-sorted site lists stay near the front.
    *tie* breaks remaining ties deterministically (seeded shuffle rank).
    """
    tilt_pen = abs(float(spec.tilt_deg)) * _POLICY_PRIOR_TILT_WEIGHT_PER_DEG
    z_pen = (
        abs(float(spec.z_fraction) - float(z_fraction_target))
        * _POLICY_PRIOR_Z_FRACTION_WEIGHT
    )
    site_pen = float(spec.site_index) * float(site_index_weight)
    return (tilt_pen + z_pen + site_pen, tie)


def _stratified_sample(
    specs: list[PlacementSpec],
    n_desired: int,
    seed: int,
    *,
    preferred_site_types: tuple[str, ...] = (),
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
) -> list[PlacementSpec]:
    """Sample up to *n_desired* specs stratified by ``site_type`` (seeded, deterministic).

    Within each site-type bucket, specs are ordered by a soft prior (milder tilt,
    ``z_fraction`` near *z_fraction_target*) with a seeded tie-break so draws
    remain deterministic. When *preferred_site_types* is set (e.g. ``("pore",)``
    for porous frameworks), those buckets are drawn first each round-robin pass.
    When *site_index_weight* > 0, draws also round-robin across ``site_index``
    values (quality-sorted) so open-pore lists keep multi-site coverage.
    """
    if len(specs) <= n_desired:
        return list(specs)

    buckets: dict[str, list[PlacementSpec]] = defaultdict(list)
    for spec in specs:
        key = str(spec.site_type) if spec.site_type is not None else "none"
        buckets[key].append(spec)

    # Preferred buckets first, then remaining keys sorted for determinism.
    preferred = [k for k in preferred_site_types if k in buckets]
    keys = preferred + [k for k in sorted(buckets) if k not in preferred]
    rng = random.Random(seed)
    for key in keys:
        bucket = buckets[key]
        ranks = list(range(len(bucket)))
        rng.shuffle(ranks)
        if site_index_weight > 0.0:
            # Round-robin across site indices so quality bias does not collapse
            # the draw onto a single open pore.
            by_site: dict[int, list[PlacementSpec]] = defaultdict(list)
            for spec, _rank in zip(bucket, ranks, strict=True):
                by_site[int(spec.site_index)].append(spec)
            ranked_by_site: dict[int, list[PlacementSpec]] = {}
            for si, site_specs in by_site.items():
                site_ranks = list(range(len(site_specs)))
                rng.shuffle(site_ranks)
                ordered = sorted(
                    zip(site_specs, site_ranks, strict=True),
                    key=lambda item: _spec_prior_key(
                        item[0],
                        float(item[1]),
                        z_fraction_target=z_fraction_target,
                        site_index_weight=0.0,
                    ),
                )
                # Best first for round-robin.
                ranked_by_site[si] = [spec for spec, _ in ordered]
            site_order = sorted(ranked_by_site.keys())
            interleaved: list[PlacementSpec] = []
            while True:
                progressed = False
                for si in site_order:
                    site_bucket = ranked_by_site[si]
                    if site_bucket:
                        interleaved.append(site_bucket.pop(0))
                        progressed = True
                if not progressed:
                    break
            # Pop from end → reverse so preferred (earlier interleaved) come off first.
            buckets[key] = list(reversed(interleaved))
        else:
            ordered = sorted(
                zip(bucket, ranks, strict=True),
                key=lambda item: _spec_prior_key(
                    item[0],
                    float(item[1]),
                    z_fraction_target=z_fraction_target,
                    site_index_weight=0.0,
                ),
            )
            # Pop from end → reverse so preferred specs come off first.
            buckets[key] = [spec for spec, _ in reversed(ordered)]

    selected: list[PlacementSpec] = []
    # Round-robin across buckets until n_desired (preferred keys first each pass).
    while len(selected) < n_desired:
        progressed = False
        for key in keys:
            if len(selected) >= n_desired:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    site_type_for_index: Callable[[int], str | None],
    shape: str,
    n_binders: int,
    flat_aromatic: bool,
    parallel_fraction: float,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
    seed: int = _PLACEMENT_GRID_COUNT_SEED,
    preferred_site_types: tuple[str, ...] = (),
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
) -> list[PlacementSpec]:
    """BO candidate ``PlacementSpec`` list: full Cartesian grid (capped), then stratified subsample to *n_desired* (*seed*)."""
    normalized_sites = site_indices if site_indices else [-1]
    base_fields = {
        "face_flip": False,
        "en_atom_index": None,
        "azimuth_in_plane_deg": 0.0,
    }

    def _fields(**overrides: Any) -> dict[str, Any]:
        return {**base_fields, **overrides}

    def _collect(
        items: Iterable[dict[str, Any]],
        cap: int,
    ) -> list[PlacementSpec]:
        """Collect specs from *items* until *cap*, applying ``filter_spec`` when set."""
        out: list[PlacementSpec] = []
        for fields in items:
            if len(out) >= cap:
                break
            spec = PlacementSpec(
                **fields,
                site_type=site_type_for_index(int(fields["site_index"])),
                placement_index=0,
            )
            if filter_spec is None or filter_spec(spec):
                out.append(spec)
        return out

    def _subsample(
        specs: list[PlacementSpec], n: int, sub_seed: int
    ) -> list[PlacementSpec]:
        return _stratified_sample(
            specs,
            n,
            sub_seed,
            preferred_site_types=preferred_site_types,
            z_fraction_target=z_fraction_target,
            site_index_weight=site_index_weight,
        )

    if dissociative:
        if n_hollow_pairs <= 0:
            specs = []
        else:
            # Shuffle pair indices before expanding z so early-cap is not
            # biased toward the first pairs in enumeration order.
            pair_indices = list(range(n_hollow_pairs))
            random.Random(seed).shuffle(pair_indices)
            items = (
                _fields(
                    conformer_index=0,
                    orientation_type="dissociative",
                    site_index=pair_idx,
                    tilt_deg=0.0,
                    azimuth_deg=0.0,
                    z_fraction=zfv,
                )
                for pair_idx, zfv in itertools.product(pair_indices, _Z_FRACTIONS)
            )
            working_cap = min(
                _GRID_BUILD_CAP,
                max(n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER, n_desired),
            )
            specs = _collect(items, cap=working_cap)
            if len(specs) > n_desired:
                specs = _subsample(specs, n_desired, seed)
    elif flat_aromatic:
        n_par = int(round(n_desired * parallel_fraction))
        n_par = max(0, min(n_par, n_desired))
        n_en = n_desired - n_par

        parallel_items = (
            _fields(
                conformer_index=ci,
                orientation_type="parallel",
                face_flip=ff,
                site_index=si,
                tilt_deg=tl,
                azimuth_deg=azv,
                azimuth_in_plane_deg=aip,
                z_fraction=zfv,
            )
            for ci, ff, tl, azv, zfv, aip, si in itertools.product(
                range(n_conformers),
                [False, True],
                _TILT_PARALLEL,
                _AZIMUTH,
                _Z_FRACTIONS,
                _AZIMUTH_IN_PLANE,
                normalized_sites,
            )
        )
        par_specs = _collect(parallel_items, cap=_GRID_BUILD_CAP)

        en_down_items = (
            _fields(
                conformer_index=ci,
                orientation_type="EN-down",
                en_atom_index=ei if n_binders > 1 else None,
                site_index=si,
                tilt_deg=tl,
                azimuth_deg=azv,
                z_fraction=zfv,
            )
            for ci, ei, tl, azv, zfv, si in itertools.product(
                range(n_conformers),
                range(max(n_binders, 1)),
                _TILT_FULL,
                _AZIMUTH,
                _Z_FRACTIONS,
                normalized_sites,
            )
        )
        en_specs = _collect(en_down_items, cap=_GRID_BUILD_CAP)

        if len(par_specs) > n_par:
            par_specs = _subsample(par_specs, n_par, seed)
        if len(en_specs) > n_en:
            en_specs = _subsample(en_specs, n_en, seed + 1)
        specs = par_specs + en_specs
    else:
        orient = "vertical" if shape == "linear" else "round"
        items = (
            _fields(
                conformer_index=ci,
                orientation_type=orient,
                site_index=si,
                tilt_deg=tl,
                azimuth_deg=azv,
                z_fraction=zfv,
            )
            for ci, tl, azv, zfv, si in itertools.product(
                range(n_conformers),
                _TILT_FULL,
                _AZIMUTH,
                _Z_FRACTIONS,
                normalized_sites,
            )
        )
        specs = _collect(items, cap=_GRID_BUILD_CAP)
        if len(specs) > n_desired:
            specs = _subsample(specs, n_desired, seed)

    for i, spec in enumerate(specs):
        spec.placement_index = i

    return specs[:n_desired]

"""Sampler policy for batch-based BO placement spec proposals."""

import itertools
import random
from collections.abc import Callable, Iterable
from typing import Any

from ..models import PlacementSpec
from ._constants import (
    _AZIMUTH,
    _AZIMUTH_IN_PLANE,
    _GRID_BUILD_CAP,
    _PLACEMENT_GRID_COUNT_SEED,
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
        return min(max(n_hollow_pairs, 1) * len(_Z_FRACTIONS), _GRID_BUILD_CAP)

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
) -> list[PlacementSpec]:
    """BO candidate ``PlacementSpec`` list: full Cartesian grid (capped), then uniform subsample to *n_desired* (*seed*); dissociative branch is small and fully enumerated."""
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

    if dissociative:
        pair_indices = range(max(n_hollow_pairs, 1))
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
        specs = _collect(items, cap=n_desired)
    elif flat_aromatic:
        n_par = max(1, int(n_desired * parallel_fraction))
        n_en = max(1, n_desired - n_par)

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

        rng = random.Random(seed)
        if len(par_specs) > n_par:
            par_specs = rng.sample(par_specs, n_par)
        if len(en_specs) > n_en:
            en_specs = rng.sample(en_specs, n_en)
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
            specs = random.Random(seed).sample(specs, n_desired)

    for i, spec in enumerate(specs):
        spec.placement_index = i

    return specs[:n_desired]

# Public alias for backward compatibility
Z_FRACTIONS = _Z_FRACTIONS

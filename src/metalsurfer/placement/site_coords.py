"""Coordinate frame, PBC images, deduplication, and radii helpers for site detection.

The cell-frame primitives (fractional conversion, wrapping, minimum image, slab
frame) live in :mod:`metalsurfer._geom_pbc` so :mod:`metalsurfer.symmetry` can use
them without importing the ``placement`` package. They are re-exported here under
their historical underscore names for existing call sites.
"""

import numpy as np
from scipy.spatial import KDTree

from .._geom_pbc import (
    cart_to_frac as _cart_to_frac,
)
from .._geom_pbc import (
    frac_to_cart as _frac_to_cart,
)
from .._geom_pbc import (
    height_along_slab_normal as _height_along_slab_normal,
)
from .._geom_pbc import (
    minimum_image_fractional_delta as _minimum_image_fractional_delta,
)
from .._geom_pbc import (
    project_to_slab_plane as _project_to_slab_plane,
)
from .._geom_pbc import (
    reciprocal_plane_spacings as _reciprocal_plane_spacings,
)
from .._geom_pbc import (
    shift_along_slab_normal as _shift_along_slab_normal,
)
from .._geom_pbc import (
    slab_normal as _slab_normal,
)
from .._geom_pbc import (
    slab_plane_projectors as _slab_plane_projectors,
)
from .._geom_pbc import (
    wrap_cartesian as _wrap_cartesian,
)
from .._geom_pbc import (
    wrap_fractional as _wrap_fractional,
)
from .._utils import union_find_cluster as _union_find_cluster
from ._constants import (
    _PORE_THRESHOLD_COVALENT_SCALE,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _STEP_TERRACE_MAX_GAP_ANGSTROM,
    _TOP_LAYER_DEPTH_COVALENT_SCALE,
    _TOP_LAYER_DEPTH_MAX_ANGSTROM,
    _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
)
from .geometry import _get_covalent_radius

__all__ = [
    # Re-exported cell-frame geometry (see metalsurfer._geom_pbc).
    "_cart_to_frac",
    "_frac_to_cart",
    "_height_along_slab_normal",
    "_minimum_image_fractional_delta",
    "_project_to_slab_plane",
    "_reciprocal_plane_spacings",
    "_shift_along_slab_normal",
    "_slab_normal",
    "_slab_plane_projectors",
    "_wrap_cartesian",
    "_wrap_fractional",
    # Defined here.
    "_build_periodic_images",
    "_deduplicate_points",
    "_derive_top_layer_tolerance",
    "_derive_voronoi_distance_window",
    "_filter_non_duplicate_candidates",
    "_mean_covalent_radius",
    "_periodic_image_offsets",
    "_union_find_cluster",
    "derive_pore_threshold",
    "top_layer_mask_by_normal",
]


def _primary_height_band_mask(
    heights: np.ndarray,
    tolerance: float,
    *,
    h_max: float | None = None,
) -> np.ndarray:
    """Boolean mask for atoms within *tolerance* of the maximum slab-normal height."""
    if h_max is None:
        h_max = float(np.max(heights))
    return heights >= (h_max - float(tolerance))


def top_layer_mask_by_normal(
    positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float,
    *,
    include_terrace: bool = True,
) -> np.ndarray:
    """Return mask of atoms belonging to exposed surface layers of a slab.

    For flat top layers (including standard multi-layer bulk slabs) this reduces
    to the topmost layer (``h_max - tolerance``).  When *include_terrace* is
    true and a discrete terrace sits immediately below the primary band
    (stepped/reconstructed surfaces), that terrace is included once if its gap
    from ``h_max`` is at most ``_STEP_TERRACE_MAX_GAP_ANGSTROM``.  The mask
    never walks deeper into the bulk by lowering the floor by another full
    tolerance.

    Parameters
    ----------
    positions
        Atomic Cartesian positions.
    cell
        Unit-cell matrix.
    tolerance
        Height tolerance for the top layer.
    include_terrace
        When false, return only the primary height band (no stepped-surface
        terrace expansion). Used by freeze policy and alloy top-layer selection.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros(0, dtype=bool)

    heights = _height_along_slab_normal(positions, cell)
    tol = float(tolerance)
    h_max = float(np.max(heights))
    primary = _primary_height_band_mask(heights, tol, h_max=h_max)
    if not include_terrace:
        return primary

    # Steps already inside the primary band (e.g. Δh ≤ tol) stay in primary.
    below = ~primary
    if not np.any(below):
        return primary

    terrace_top = float(np.max(heights[below]))
    gap = h_max - terrace_top
    if gap <= _STEP_TERRACE_MAX_GAP_ANGSTROM:
        terrace = (heights >= terrace_top - tol) & (heights <= terrace_top + tol)
        return primary | terrace
    return primary


def _filter_non_duplicate_candidates(
    candidates: np.ndarray,
    existing: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Boolean mask: True where *candidates* are not within *tolerance* of *existing*."""
    if len(existing) == 0:
        return np.ones(len(candidates), dtype=bool)
    if len(candidates) == 0:
        return np.ones(0, dtype=bool)
    tree = KDTree(existing)
    dists, _ = tree.query(np.asarray(candidates, dtype=float), k=1)
    return np.asarray(dists, dtype=float).ravel() >= tolerance


# ---------------------------------------------------------------------------
# Periodic image generation
# ---------------------------------------------------------------------------


def _periodic_image_offsets(
    cell: np.ndarray,
    pbc: np.ndarray,
    margin: float,
) -> list[np.ndarray]:
    """Return enough image offsets to cover a Cartesian margin around the cell."""
    if not np.any(pbc):
        return [np.zeros(3, dtype=float)]

    spacings = _reciprocal_plane_spacings(cell)
    ranges: list[list[int]] = []
    for dim in range(3):
        if bool(pbc[dim]):
            spacing = float(spacings[dim])
            if not np.isfinite(spacing) or spacing <= 0.0:
                n_img = 1
            else:
                n_img = max(
                    1, int(np.ceil((margin + _VORONOI_DEDUP_TOLERANCE) / spacing))
                )
            ranges.append(list(range(-n_img, n_img + 1)))
        else:
            ranges.append([0])

    offsets: list[np.ndarray] = []
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                offsets.append(i * cell[0] + j * cell[1] + k * cell[2])
    return offsets


def _build_periodic_images(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    margin: float = 0.0,
) -> np.ndarray:
    """Return extended positions including enough periodic images for *margin*."""
    offsets = _periodic_image_offsets(cell, pbc, margin)
    return np.vstack([positions + off for off in offsets])


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _deduplicate_points(
    points: np.ndarray,
    tolerance: float,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return a boolean keep-mask that removes near-duplicate points.

    When *cell* and *pbc* are provided, periodic duplicates across the unit-cell
    boundary are also merged.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return np.ones(0, dtype=bool)

    if cell is None or pbc is None or not np.any(pbc):
        ded_tree = KDTree(pts)
        pairs = ded_tree.query_pairs(r=tolerance, output_type="ndarray")
        merge_set: set[tuple[int, int]] = set()
        for i, j in pairs:
            a, b = int(i), int(j)
            if a == b:
                continue
            merge_set.add((min(a, b), max(a, b)))
        components = _union_find_cluster(n, sorted(merge_set))
        keep = np.zeros(n, dtype=bool)
        for comp in components:
            keep[min(comp)] = True
        return keep

    offsets = _periodic_image_offsets(
        np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool), tolerance
    )
    expanded = np.vstack([pts + off for off in offsets])
    tree = KDTree(expanded)
    raw_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")
    periodic_merge: set[tuple[int, int]] = set()
    for a_exp, b_exp in raw_pairs:
        a = int(a_exp) % n
        b = int(b_exp) % n
        if a == b:
            continue
        periodic_merge.add((min(a, b), max(a, b)))
    components = _union_find_cluster(n, sorted(periodic_merge))
    keep = np.zeros(n, dtype=bool)
    for comp in components:
        keep[min(comp)] = True
    return keep


# ---------------------------------------------------------------------------
# Parameter derivation helpers
# ---------------------------------------------------------------------------


def _mean_covalent_radius(symbols: list[str]) -> float:
    radii = [_get_covalent_radius(s) for s in symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        raise ValueError(
            f"No positive covalent radii for symbols {symbols!r}; "
            "cannot derive Voronoi distance window"
        )
    return float(np.mean(valid))


def _derive_voronoi_distance_window(
    positions: np.ndarray,
    symbols: list[str],
    pbc: np.ndarray,
    cell: np.ndarray | None = None,
) -> tuple[float, float]:
    if len(positions) == 0:
        raise ValueError("Cannot derive Voronoi distance window from empty positions")
    if bool(pbc[0]) and bool(pbc[1]) and not bool(pbc[2]) and cell is not None:
        # Slab: characterise the exposed top layer along the slab normal
        # (orientation-aware), not a Cartesian-z slice or the bulk average.
        heights = _height_along_slab_normal(positions, cell)
        top_depth = _derive_top_layer_tolerance(symbols)
        top_mask = _primary_height_band_mask(heights, top_depth)
        top_idx = np.nonzero(top_mask)[0]
        top_symbols = [symbols[int(i)] for i in top_idx] if len(top_idx) else symbols
        base_radius = _mean_covalent_radius(top_symbols)
    else:
        # Nanoparticle, porous, and non-periodic: mean over all framework atoms.
        base_radius = _mean_covalent_radius(symbols)

    probe_radius = _VORONOI_PROBE_RADIUS_COVALENT_SCALE * base_radius
    max_distance = _VORONOI_MAX_DISTANCE_COVALENT_SCALE * base_radius
    return float(probe_radius), float(max(max_distance, probe_radius))


def _derive_top_layer_tolerance(symbols: list[str]) -> float:
    """Covalent-radius-derived top-layer depth, capped for FCC-like slabs."""
    mean_radius = _mean_covalent_radius(symbols)
    return float(
        min(
            max(
                _TOP_LAYER_DEPTH_MIN_ANGSTROM,
                _TOP_LAYER_DEPTH_COVALENT_SCALE * mean_radius,
            ),
            _TOP_LAYER_DEPTH_MAX_ANGSTROM,
        )
    )


def derive_pore_threshold(symbols: list[str]) -> float:
    """Return pore classification threshold from mean covalent radius.

    Parameters
    ----------
    symbols
        List of element symbols.
    """
    mean_radius = _mean_covalent_radius(symbols)
    return max(
        _PORE_THRESHOLD_MIN_ANGSTROM, _PORE_THRESHOLD_COVALENT_SCALE * mean_radius
    )

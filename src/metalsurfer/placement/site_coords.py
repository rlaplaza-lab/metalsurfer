"""Coordinate frame, PBC images, deduplication, and radii helpers for site detection."""

import logging

import numpy as np
from scipy.spatial import KDTree

from ._constants import (
    _PORE_THRESHOLD_COVALENT_SCALE,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _STEP_TERRACE_MAX_GAP_ANGSTROM,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _TOP_LAYER_DEPTH_COVALENT_SCALE,
    _TOP_LAYER_DEPTH_MAX_ANGSTROM,
    _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
    _VORONOI_RADIUS_FALLBACK_ANGSTROM,
)
from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)


def _cart_to_frac(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert Cartesian row-vectors to fractional coordinates for ASE cells."""
    arr = np.asarray(points, dtype=float)
    inv_cell = np.linalg.inv(cell)
    return arr @ inv_cell


def _frac_to_cart(points_frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert fractional row-vectors to Cartesian coordinates."""
    return np.asarray(points_frac, dtype=float) @ cell


def _wrap_fractional(frac: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates to [0, 1) on periodic axes only."""
    wrapped = np.asarray(frac, dtype=float).copy()
    for dim in range(3):
        if bool(pbc[dim]):
            wrapped[..., dim] -= np.floor(wrapped[..., dim])
    return wrapped


def _wrap_cartesian(
    points: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Wrap Cartesian points into the reference cell along periodic axes."""
    if not np.any(pbc):
        return np.asarray(points, dtype=float).copy()
    frac = _cart_to_frac(points, cell)
    return _frac_to_cart(_wrap_fractional(frac, pbc), cell)


def _minimum_image_fractional_delta(
    delta_frac: np.ndarray, pbc: np.ndarray
) -> np.ndarray:
    """Apply the minimum-image convention to fractional coordinate differences."""
    delta = np.asarray(delta_frac, dtype=float).copy()
    for dim in range(3):
        if bool(pbc[dim]):
            delta[..., dim] -= np.round(delta[..., dim])
    return delta


def _reciprocal_plane_spacings(cell: np.ndarray) -> np.ndarray:
    """Distance between adjacent lattice planes normal to each cell vector."""
    inv_cell = np.linalg.inv(cell)
    spacings = np.empty(3, dtype=float)
    for dim in range(3):
        g = inv_cell[:, dim]
        norm_g = float(np.linalg.norm(g))
        spacings[dim] = 1.0 / norm_g if norm_g > 0.0 else np.inf
    return spacings


def _slab_plane_projectors(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return projectors for slab-plane coordinates.

    Returns
    -------
    (pinv_ab_T, ortho_basis)
        pinv_ab_T : (3, 2) array
            Right-multiplier for least-squares coordinates in the span of a/b.
            For Cartesian row vectors r, in-plane coordinates are r @ pinv_ab_T.
        ortho_basis : (2, 3) array
            Two orthonormal basis vectors spanning the same plane.
    """
    a = np.asarray(cell[0], dtype=float)
    b = np.asarray(cell[1], dtype=float)

    ab = np.column_stack([a, b])
    pinv_ab = np.linalg.pinv(ab)
    pinv_ab_T = pinv_ab.T

    norm_a = float(np.linalg.norm(a))
    if norm_a < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = a / norm_a

    b_perp = b - np.dot(b, e1) * e1
    norm_b_perp = float(np.linalg.norm(b_perp))
    if norm_b_perp < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        trial = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(trial, e1)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        b_perp = trial - np.dot(trial, e1) * e1
        norm_b_perp = float(np.linalg.norm(b_perp))
    e2 = b_perp / max(norm_b_perp, _SURFACE_NORMAL_FALLBACK_NORM_EPS)
    ortho_basis = np.vstack([e1, e2])
    return pinv_ab_T, ortho_basis


def _slab_normal(cell: np.ndarray) -> np.ndarray:
    """Unit normal to the slab plane spanned by cell a and b."""
    a = np.asarray(cell[0], dtype=float)
    b = np.asarray(cell[1], dtype=float)
    n = np.cross(a, b)
    norm_n = float(np.linalg.norm(n))
    if norm_n < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.array([0.0, 0.0, 1.0])
    return n / norm_n


def _height_along_slab_normal(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Signed coordinate of points along the slab normal."""
    n = _slab_normal(cell)
    arr = np.asarray(points, dtype=float)
    return arr @ n


def _shift_along_slab_normal(
    points: np.ndarray, cell: np.ndarray, distance: float
) -> np.ndarray:
    """Translate points by *distance* along the slab normal."""
    n = _slab_normal(cell)
    arr = np.asarray(points, dtype=float)
    return arr + float(distance) * n


def _project_to_slab_plane(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Project Cartesian points to a 2D orthonormal basis spanning a/b."""
    _, ortho_basis = _slab_plane_projectors(cell)
    arr = np.asarray(points, dtype=float)
    return arr @ ortho_basis.T


def top_layer_mask_by_normal(
    positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return mask of atoms belonging to exposed surface layers of a slab.

    For flat top layers (including standard multi-layer bulk slabs) this reduces
    to the topmost layer (``h_max - tolerance``).  When a discrete terrace sits
    immediately below the primary band (stepped/reconstructed surfaces), that
    terrace is included once if its gap from ``h_max`` is at most
    ``_STEP_TERRACE_MAX_GAP_ANGSTROM``.  The mask never walks deeper into the
    bulk by lowering the floor by another full tolerance.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros(0, dtype=bool)

    heights = _height_along_slab_normal(positions, cell)
    h_max = float(np.max(heights))
    tol = float(tolerance)
    primary = heights >= (h_max - tol)

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


def _union_find_cluster(
    n: int,
    merge_pairs: list[tuple[int, int]],
) -> list[list[int]]:
    """Union-find with path compression and union-by-rank."""
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in merge_pairs:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


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
        return _VORONOI_RADIUS_FALLBACK_ANGSTROM
    return float(np.mean(valid))


def _derive_voronoi_distance_window(
    positions: np.ndarray,
    symbols: list[str],
    pbc: np.ndarray,
    cell: np.ndarray | None = None,
) -> tuple[float, float]:
    if len(positions) == 0:
        base_radius = _VORONOI_RADIUS_FALLBACK_ANGSTROM
    elif bool(pbc[0]) and bool(pbc[1]) and not bool(pbc[2]) and cell is not None:
        # Slab: characterise the exposed top layer along the slab normal
        # (orientation-aware), not a Cartesian-z slice or the bulk average.
        heights = _height_along_slab_normal(positions, cell)
        h_max = float(np.max(heights))
        top_depth = _derive_top_layer_tolerance(symbols)
        top_mask = heights >= (h_max - top_depth)
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
    """Return pore classification threshold from mean covalent radius."""
    mean_radius = _mean_covalent_radius(symbols)
    return max(
        _PORE_THRESHOLD_MIN_ANGSTROM, _PORE_THRESHOLD_COVALENT_SCALE * mean_radius
    )

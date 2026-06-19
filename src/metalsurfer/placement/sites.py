"""Hybrid topology/Voronoi site generation, clustering, and optional spglib-based symmetry reduction.

Key improvements over the original implementation
-----------------------------------------------
1. Slab handling is now orientation-aware:
   - top-layer detection is based on the slab normal (from a x b), not on the
     Cartesian z axis
   - slab filtering and atop injection also use the slab normal, so rotated
     slabs behave correctly
2. Slabs now use a hybrid default site generator:
   - explicit topology-derived atop/bridge/hollow candidates from the top layer
   - Voronoi-derived candidates are still used for enrichment and porous/rugged
     features
3. Point deduplication is periodic-aware for skewed cells and boundary-adjacent
   duplicates, reducing overcounting near cell edges.
4. Existing public API and external behaviour are preserved as closely as
   possible, including symmetry reduction helpers and z-base utilities.

The module keeps relative imports unchanged so it can directly replace the
existing package file.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree, QhullError, Voronoi

from ..symmetry import SymmetryAnalyzer
from ._constants import (
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _ATOP_RATIO,
    _BOUNDING_BOX_CELL_PAD_ANGSTROM,
    _BRIDGE_EQ_TOL,
    _BRIDGE_FAR_RATIO,
    _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    _DEFAULT_SYMMETRY_TOLERANCE,
    _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM,
    _DISTANCE_RATIO_FLOOR_EPS,
    _DISTANCE_ZERO_EPS,
    _ENRICHMENT_MAX_SUBDIVISIONS,
    _ENRICHMENT_SPACING_BETA,
    _HOLLOW_EQ_TOL,
    _KD_RADIUS_SEARCH_PADDING,
    _MOL_COVALENT_RADIUS_FALLBACK,
    _NON_SLAB_Z_HI_FROM_NN_SCALE,
    _NON_SLAB_Z_LO_FROM_NN_SCALE,
    _NORMAL_K_NEIGHBOURS,
    _PORE_THRESHOLD_COVALENT_SCALE,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _SITE_Z_RADIUS_REFERENCE_ANGSTROM,
    _SITE_Z_RADIUS_SHIFT_SCALE,
    _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _TOP_LAYER_DEPTH_COVALENT_SCALE,
    _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_FRACTIONAL_CELL_MARGIN,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
    _VORONOI_RADIUS_FALLBACK_ANGSTROM,
)
from ._material import material_type_for_placement
from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)

DEFAULT_SYMMETRY_TOLERANCE = _DEFAULT_SYMMETRY_TOLERANCE
DEFAULT_SITE_EQUIVALENCE_TOLERANCE = _DEFAULT_SITE_EQUIVALENCE_TOLERANCE
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE = _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


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


def _top_layer_mask_by_normal(
    positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return mask of atoms belonging to exposed surface layers of a slab.

    For flat top layers (including standard multi-layer bulk slabs) this reduces
    to the topmost layer (``h_max - tolerance``).  When multiple height levels
    fall within the top tolerance band (stepped/reconstructed surfaces), the
    mask also includes the terrace one step below the top band.
    """
    heights = _height_along_slab_normal(positions, cell)
    h_max = float(np.max(heights))
    tol = float(tolerance)
    primary = heights >= (h_max - tol)

    top_heights = heights[primary]
    if len(top_heights) == 0:
        return primary

    unique_top = np.sort(np.unique(top_heights))
    if len(unique_top) < 2:
        return primary

    gaps = np.diff(unique_top)
    if not np.any(gaps > tol * 0.5):
        return primary

    # Multiple height levels within the top band: include the full top band.
    top_floor = float(unique_top[0]) - tol
    mask = heights >= top_floor

    # Step-edge terrace immediately below the top band.
    below = heights < top_floor
    if np.any(below):
        terrace_top = float(np.max(heights[below]))
        if h_max - terrace_top > tol:
            mask |= (heights >= terrace_top - tol) & (heights <= terrace_top + tol)
    return mask


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
        top_mask = heights >= (h_max - _TOP_LAYER_DEPTH_MIN_ANGSTROM)
        top_idx = np.nonzero(top_mask)[0]
        top_symbols = [symbols[int(i)] for i in top_idx] if len(top_idx) else symbols
        base_radius = _mean_covalent_radius(top_symbols)
    else:
        # Nanoparticle, porous, and non-periodic: mean over all framework atoms.
        base_radius = _mean_covalent_radius(symbols)

    probe_radius = _VORONOI_PROBE_RADIUS_COVALENT_SCALE * base_radius
    max_distance = _VORONOI_MAX_DISTANCE_COVALENT_SCALE * base_radius
    return float(probe_radius), float(max(max_distance, probe_radius))


def _derive_top_layer_tolerance(
    positions: np.ndarray,
    symbols: list[str],
) -> float:
    mean_radius = _mean_covalent_radius(symbols)
    return max(
        _TOP_LAYER_DEPTH_MIN_ANGSTROM, _TOP_LAYER_DEPTH_COVALENT_SCALE * mean_radius
    )


def _derive_pore_threshold(symbols: list[str]) -> float:
    """Return pore classification threshold from mean covalent radius."""
    mean_radius = _mean_covalent_radius(symbols)
    return max(
        _PORE_THRESHOLD_MIN_ANGSTROM, _PORE_THRESHOLD_COVALENT_SCALE * mean_radius
    )


# ---------------------------------------------------------------------------
# Voronoi site generation
# ---------------------------------------------------------------------------


def _voronoi_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    probe_radius: float | None = None,
    max_distance: float | None = None,
    enrich: bool = True,
    symbols: list[str] | None = None,
    nn_reference_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Voronoi vertices accessible for adsorption, optionally enriched."""
    if len(positions) < 4:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    if symbols is None:
        symbols = ["C"] * len(positions)

    if probe_radius is None or max_distance is None:
        derived_probe, derived_max = _derive_voronoi_distance_window(
            positions, symbols, pbc, cell
        )
        probe_radius = derived_probe if probe_radius is None else probe_radius
        max_distance = derived_max if max_distance is None else max_distance

    if np.linalg.det(cell) <= 0.0:
        logger.debug(
            "Degenerate cell for Voronoi generation; falling back to no-PBC enumeration"
        )
        pbc = np.zeros(3, dtype=bool)

    extension_margin = float(max_distance) + _VORONOI_DEDUP_TOLERANCE
    extended = _build_periodic_images(positions, cell, pbc, margin=extension_margin)

    nn_positions = (
        nn_reference_positions if nn_reference_positions is not None else positions
    )
    nn_extension_margin = extension_margin
    nn_extended = _build_periodic_images(
        nn_positions, cell, pbc, margin=nn_extension_margin
    )

    try:
        vor = Voronoi(extended)
    except (QhullError, ValueError, RuntimeError) as exc:
        logger.debug("Voronoi computation failed (%s); returning no vertices", exc)
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    raw_vertices = np.asarray(vor.vertices, dtype=float)
    if len(raw_vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    wrapped_vertices = _wrap_cartesian(raw_vertices, cell, pbc)

    tree = KDTree(nn_extended)
    nn_dists, _ = tree.query(raw_vertices, k=1)
    nn_dists = np.asarray(nn_dists, dtype=float).ravel()

    accessible = (nn_dists >= probe_radius) & (nn_dists <= max_distance)
    wrapped_vertices = wrapped_vertices[accessible]
    nn_dists = nn_dists[accessible]
    raw_accessible_indices = np.nonzero(accessible)[0]

    if len(wrapped_vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    if np.any(pbc):
        frac = _cart_to_frac(raw_vertices[accessible], cell)
        inside = np.ones(len(frac), dtype=bool)
        for dim in range(3):
            if bool(pbc[dim]):
                inside &= (frac[:, dim] >= -_VORONOI_FRACTIONAL_CELL_MARGIN) & (
                    frac[:, dim] < 1.0 + _VORONOI_FRACTIONAL_CELL_MARGIN
                )
        wrapped_vertices = wrapped_vertices[inside]
        nn_dists = nn_dists[inside]
        raw_accessible_indices = raw_accessible_indices[inside]

    if len(wrapped_vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    keep = _deduplicate_points(
        wrapped_vertices, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
    )
    vertices = wrapped_vertices[keep]
    nn_dists = nn_dists[keep]

    if len(vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    if not enrich or len(vertices) < 2:
        return vertices, nn_dists

    kept_tree = KDTree(vertices)
    raw_to_kept: dict[int, int] = {}
    accessible_wrapped = wrapped_vertices
    dist_to_kept, idx_to_kept = kept_tree.query(accessible_wrapped, k=1)
    for raw_idx, d, kept_idx in zip(
        raw_accessible_indices, dist_to_kept, idx_to_kept, strict=False
    ):
        if float(d) <= _VORONOI_DEDUP_TOLERANCE:
            raw_to_kept[int(raw_idx)] = int(kept_idx)

    enriched_verts, enriched_dists = _enrich_along_ridges(
        vertices,
        nn_dists,
        vor.ridge_vertices,
        raw_to_kept,
        nn_extended,
        tree,
        probe_radius,
        max_distance,
        cell=cell,
        pbc=pbc,
    )
    return enriched_verts, enriched_dists


# ---------------------------------------------------------------------------
# Topology-derived slab sites
# ---------------------------------------------------------------------------


def _generate_slab_topology_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    top_atom_indices: np.ndarray,
    local_tree: KDTree,
    site_height: float,
    probe_radius: float,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate slab atop/bridge/hollow candidates from the top layer.

    Candidates are created in an orientation-aware way and wrapped back into the
    reference cell on periodic axes.
    """
    if len(top_atom_indices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), []

    n_hat = _slab_normal(cell)
    top_positions = positions[np.asarray(top_atom_indices, dtype=int)]

    candidates: list[np.ndarray] = []
    candidate_dists: list[float] = []
    candidate_sources: list[str] = []

    def _add_candidate(point: np.ndarray, source: str) -> None:
        p = np.asarray(point, dtype=float)
        if np.any(pbc):
            p = _wrap_cartesian(p.reshape(1, 3), cell, pbc)[0]
        d_nn = float(local_tree.query(p.reshape(1, 3), k=1)[0].ravel()[0])
        if probe_radius <= d_nn <= max_distance:
            candidates.append(p)
            candidate_dists.append(d_nn)
            candidate_sources.append(source)

    # Atop candidates: always useful and cheap.
    atop_positions = top_positions + float(site_height) * n_hat
    for p in atop_positions:
        _add_candidate(p, "topology_atop")

    # Need 2D triangulation for bridge/hollow candidates.
    if len(top_positions) < 2:
        if not candidates:
            return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), []
        cand_arr = np.asarray(candidates, dtype=float)
        keep = _deduplicate_points(
            cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
        )
        return (
            cand_arr[keep],
            np.asarray(candidate_dists, dtype=float)[keep],
            [candidate_sources[i] for i in np.nonzero(keep)[0]],
        )

    ranges_a = (-1, 0, 1) if bool(pbc[0]) else (0,)
    ranges_b = (-1, 0, 1) if bool(pbc[1]) else (0,)

    expanded_points_2d: list[np.ndarray] = []
    expanded_points_3d: list[np.ndarray] = []
    expanded_origin_local_index: list[int] = []
    top_positions_2d = _project_to_slab_plane(top_positions, cell)
    for ia in ranges_a:
        for ib in ranges_b:
            offset = ia * cell[0] + ib * cell[1]
            pts3d = top_positions + offset
            pts2d = (
                top_positions_2d + _project_to_slab_plane(offset.reshape(1, 3), cell)[0]
            )
            for li in range(len(top_positions)):
                expanded_points_2d.append(pts2d[li])
                expanded_points_3d.append(pts3d[li])
                expanded_origin_local_index.append(int(li))

    exp2d = np.asarray(expanded_points_2d, dtype=float)
    exp3d = np.asarray(expanded_points_3d, dtype=float)
    tri: Delaunay | None = None
    if len(exp2d) >= 3:
        try:
            tri = Delaunay(exp2d)
        except (QhullError, ValueError, RuntimeError):
            tri = None

    if tri is not None:
        seen_edges: set[tuple[int, int]] = set()
        for simplex in np.asarray(tri.simplices, dtype=int):
            for e0, e1 in ((0, 1), (1, 2), (0, 2)):
                i_exp, j_exp = int(simplex[e0]), int(simplex[e1])
                li = expanded_origin_local_index[i_exp]
                lj = expanded_origin_local_index[j_exp]
                if li == lj:
                    continue
                edge_key = (min(li, lj), max(li, lj))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                midpoint = (
                    0.5 * (exp3d[i_exp] + exp3d[j_exp]) + float(site_height) * n_hat
                )
                _add_candidate(midpoint, "topology_bridge")

        seen_tris: set[tuple[int, int, int]] = set()
        for simplex in np.asarray(tri.simplices, dtype=int):
            local_ids = tuple(
                sorted({expanded_origin_local_index[int(k)] for k in simplex})
            )
            if len(local_ids) != 3:
                continue
            if local_ids in seen_tris:
                continue
            seen_tris.add(local_ids)
            centroid = np.mean(exp3d[simplex], axis=0) + float(site_height) * n_hat
            _add_candidate(centroid, "topology_hollow")

    if not candidates:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), []

    cand_arr = np.asarray(candidates, dtype=float)
    cand_dist = np.asarray(candidate_dists, dtype=float)
    keep = _deduplicate_points(cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    kept_idx = np.nonzero(keep)[0]
    return cand_arr[keep], cand_dist[keep], [candidate_sources[i] for i in kept_idx]


# ---------------------------------------------------------------------------
# Ridge-based geodesic enrichment
# ---------------------------------------------------------------------------


def _enrich_along_ridges(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    ridge_vertices: list[list[int]],
    raw_to_kept: dict[int, int],
    extended_positions: np.ndarray,
    framework_tree: KDTree,
    probe_radius: float,
    max_distance: float,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide long admissible Voronoi edges and re-check accessibility."""
    n_kept = len(vertices)
    if n_kept < 2:
        return vertices, nn_dists

    k_support = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(extended_positions))
    _, support_indices = framework_tree.query(vertices, k=k_support)
    if np.ndim(support_indices) == 1:
        support_indices = np.asarray(support_indices).reshape(-1, 1)
    support_sets = [set(int(j) for j in row) for row in np.asarray(support_indices)]

    median_nn = float(np.median(nn_dists)) if len(nn_dists) else float(probe_radius)
    target_spacing = _ENRICHMENT_SPACING_BETA * median_nn

    edges: set[tuple[int, int]] = set()
    for ridge in ridge_vertices:
        if len(ridge) != 2:
            continue
        r0, r1 = int(ridge[0]), int(ridge[1])
        if r0 < 0 or r1 < 0:
            continue
        if r0 in raw_to_kept and r1 in raw_to_kept:
            k0, k1 = raw_to_kept[r0], raw_to_kept[r1]
            if k0 != k1:
                edges.add((min(k0, k1), max(k0, k1)))

    if not edges:
        return vertices, nn_dists

    new_verts: list[np.ndarray] = []
    new_dists: list[float] = []

    for k0, k1 in sorted(edges):
        if not support_sets[k0] & support_sets[k1]:
            continue

        v0, v1 = vertices[k0], vertices[k1]
        if np.any(pbc):
            f0 = _cart_to_frac(v0.reshape(1, 3), cell)[0]
            f1 = _cart_to_frac(v1.reshape(1, 3), cell)[0]
            df = _minimum_image_fractional_delta((f1 - f0).reshape(1, 3), pbc)[0]
            edge_vec = _frac_to_cart(df.reshape(1, 3), cell)[0]
        else:
            edge_vec = v1 - v0

        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len <= target_spacing:
            continue

        n_subdivisions = min(
            int(edge_len / target_spacing), _ENRICHMENT_MAX_SUBDIVISIONS
        )
        if n_subdivisions < 1:
            continue

        for s in range(1, n_subdivisions + 1):
            t = s / (n_subdivisions + 1)
            candidate = v0 + t * edge_vec
            if np.any(pbc):
                candidate = _wrap_cartesian(candidate.reshape(1, 3), cell, pbc)[0]
            d_nn = float(
                framework_tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0]
            )
            if probe_radius <= d_nn <= max_distance:
                new_verts.append(candidate)
                new_dists.append(d_nn)

    if not new_verts:
        return vertices, nn_dists

    all_verts = np.vstack([vertices, np.asarray(new_verts, dtype=float)])
    all_dists = np.concatenate([nn_dists, np.asarray(new_dists, dtype=float)])
    keep = _deduplicate_points(all_verts, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    return all_verts[keep], all_dists[keep]


# ---------------------------------------------------------------------------
# Site classification
# ---------------------------------------------------------------------------


def _classify_voronoi_site_from_neighbors(
    dists: np.ndarray,
    idx: np.ndarray,
    pore_threshold: float = _PORE_THRESHOLD_MIN_ANGSTROM,
) -> tuple[str, tuple[int, ...]]:
    """Classify a site from precomputed nearest-neighbour distances and indices."""
    dists = np.asarray(dists, dtype=float).ravel()
    idx = np.asarray(idx, dtype=int).ravel()
    if len(dists) == 0:
        return "atop", ()
    d1 = float(dists[0])
    if d1 < _DISTANCE_ZERO_EPS:
        return "atop", (int(idx[0]),)
    if d1 > pore_threshold:
        return "pore", tuple(int(i) for i in idx)
    if len(dists) >= 2 and dists[1] / d1 > _ATOP_RATIO:
        return "atop", (int(idx[0]),)
    if len(dists) >= 3 and all(
        abs(float(dists[i]) - d1) / max(d1, _DISTANCE_RATIO_FLOOR_EPS) < _HOLLOW_EQ_TOL
        for i in range(1, 3)
    ):
        return "hollow", tuple(int(i) for i in idx[:3])
    if (
        len(dists) >= 2
        and abs(float(dists[1]) - d1) / max(d1, _DISTANCE_RATIO_FLOOR_EPS)
        < _BRIDGE_EQ_TOL
    ):
        far3 = len(dists) < 3 or float(dists[2]) / d1 > _BRIDGE_FAR_RATIO
        if far3:
            return "bridge", tuple(int(i) for i in idx[:2])
    return "hollow", tuple(int(i) for i in idx[:3])


def _hollow_coordination_order(dists: np.ndarray) -> int | None:
    """Return 3 or 4 when *dists* indicate equidistant hollow coordination."""
    dists_arr = np.asarray(dists, dtype=float).ravel()
    if len(dists_arr) < 3:
        return None
    d1 = float(dists_arr[0])
    if d1 < _DISTANCE_ZERO_EPS:
        return None
    floor = max(d1, _DISTANCE_RATIO_FLOOR_EPS)
    count = 1
    for i in range(1, len(dists_arr)):
        if abs(float(dists_arr[i]) - d1) / floor < _HOLLOW_EQ_TOL:
            count += 1
        else:
            break
    return count if count >= 3 else None


def _classify_voronoi_site(
    vertex: np.ndarray,
    positions: np.ndarray,
    tree: KDTree | None = None,
    pore_threshold: float = _PORE_THRESHOLD_MIN_ANGSTROM,
    k: int = _SITE_CLASSIFICATION_NEIGHBOURS,
) -> tuple[str, tuple[int, ...]]:
    """Classify vertex as atop/bridge/hollow/pore."""
    k = min(k, len(positions))
    if tree is None:
        tree = KDTree(positions)
    dists, idx = tree.query(vertex.reshape(1, 3), k=k)
    return _classify_voronoi_site_from_neighbors(
        np.asarray(dists, dtype=float).ravel(),
        np.asarray(idx, dtype=int).ravel(),
        pore_threshold=pore_threshold,
    )


def _delaunay_site_classification(
    vertex: np.ndarray,
    top_positions_2d: np.ndarray,
    top_atom_indices: np.ndarray,
    triangulation: Delaunay,
    positions: np.ndarray,
    *,
    vertex_2d: np.ndarray,
    bridge_threshold: float = _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    char_len: float | None = None,
    positions_tree: KDTree | None = None,
) -> tuple[str, tuple[int, ...]]:
    """Classify a slab site using Delaunay triangulation in the slab plane."""
    xy = np.asarray(vertex_2d, dtype=float)
    simplices = triangulation.simplices
    top_xy = np.asarray(top_positions_2d, dtype=float)

    best_type = "hollow"
    best_dist = float("inf")
    best_indices: tuple[int, ...] = ()

    for li, gi in enumerate(top_atom_indices):
        d = float(np.linalg.norm(xy - top_xy[li]))
        if d < best_dist:
            best_dist = d
            best_type = "atop"
            best_indices = (int(gi),)

    if char_len is None:
        if len(top_xy) >= 2:
            _top_tree = KDTree(top_xy)
            _nn_d, _ = _top_tree.query(top_xy, k=2)
            char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))
        else:
            char_len = (
                best_dist
                if best_dist < float("inf")
                else _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM
            )

    seen_edges: set[tuple[int, int]] = set()
    for simplex in simplices:
        for e0, e1 in ((0, 1), (1, 2), (0, 2)):
            li0, li1 = int(simplex[e0]), int(simplex[e1])
            edge_key = (min(li0, li1), max(li0, li1))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            mid = (top_xy[li0] + top_xy[li1]) / 2.0
            d = float(np.linalg.norm(xy - mid))
            if d < best_dist:
                best_dist = d
                best_type = "bridge"
                best_indices = (int(top_atom_indices[li0]), int(top_atom_indices[li1]))

    for simplex in simplices:
        li0, li1, li2 = int(simplex[0]), int(simplex[1]), int(simplex[2])
        centroid = (top_xy[li0] + top_xy[li1] + top_xy[li2]) / 3.0
        d = float(np.linalg.norm(xy - centroid))
        if d < best_dist:
            best_dist = d
            best_type = "hollow"
            best_indices = (
                int(top_atom_indices[li0]),
                int(top_atom_indices[li1]),
                int(top_atom_indices[li2]),
            )

    if best_type == "bridge" and best_dist > bridge_threshold * char_len:
        _tree = positions_tree if positions_tree is not None else KDTree(positions)
        _, idx = _tree.query(vertex.reshape(1, 3), k=min(3, len(positions)))
        idx = np.asarray(idx, dtype=int).ravel()
        best_type = "hollow"
        best_indices = tuple(int(i) for i in idx[:3])

    return best_type, best_indices


# ---------------------------------------------------------------------------
# Local surface normal
# ---------------------------------------------------------------------------


def _compute_local_normal(
    vertex: np.ndarray,
    positions: np.ndarray,
    tree: KDTree | None = None,
    k: int = _NORMAL_K_NEIGHBOURS,
) -> np.ndarray:
    """Outward unit normal at *vertex* from centroid of k nearest atoms."""
    k = min(k, len(positions))
    if tree is None:
        tree = KDTree(positions)
    _, idx = tree.query(vertex.reshape(1, 3), k=k)
    centroid = np.mean(positions[np.asarray(idx).ravel()], axis=0)
    vec = np.asarray(vertex, dtype=float) - centroid
    norm = float(np.linalg.norm(vec))
    if norm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def _compute_local_normals_batch(
    vertices: np.ndarray,
    positions: np.ndarray,
    support_indices: np.ndarray,
) -> np.ndarray:
    """Outward unit normals for each vertex from batched neighbour centroids."""
    n = len(vertices)
    if n == 0:
        return np.empty((0, 3), dtype=float)
    idx = np.asarray(support_indices, dtype=int)
    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
    centroids = np.mean(positions[idx], axis=1)
    vecs = np.asarray(vertices, dtype=float) - centroids
    norms = np.linalg.norm(vecs, axis=1)
    fallback = np.array([0.0, 0.0, 1.0], dtype=float)
    out = np.empty((n, 3), dtype=float)
    for i in range(n):
        if norms[i] < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            out[i] = fallback
        else:
            out[i] = vecs[i] / norms[i]
    return out


def _build_site_records(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    positions: np.ndarray,
    symbols: list[str],
    local_tree: KDTree,
    material_type: str,
    pore_threshold: float,
    *,
    use_delaunay: bool,
    delaunay_tri: Delaunay | None,
    top_positions_2d: np.ndarray | None,
    top_atom_indices: np.ndarray | None,
    cell: np.ndarray,
    source_hints: list[str] | None = None,
) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = []
    n_verts = len(vertices)
    vertex_2d = _project_to_slab_plane(vertices, cell) if n_verts else np.empty((0, 2))

    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
    k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
    class_dists: np.ndarray | None = None
    class_idx: np.ndarray | None = None
    normals: np.ndarray | None = None
    delaunay_char_len: float | None = None

    if n_verts > 0 and not (
        use_delaunay
        and delaunay_tri is not None
        and top_positions_2d is not None
        and top_atom_indices is not None
    ):
        dists_raw, idx_raw = local_tree.query(vertices, k=k_class)
        class_dists = np.asarray(dists_raw, dtype=float)
        class_idx = np.asarray(idx_raw, dtype=int)
        if class_dists.ndim == 1:
            class_dists = class_dists.reshape(-1, 1)
            class_idx = class_idx.reshape(-1, 1)
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)
    elif n_verts > 0:
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)
        if top_positions_2d is not None and len(top_positions_2d) >= 2:
            _top_tree = KDTree(top_positions_2d)
            _nn_d, _ = _top_tree.query(top_positions_2d, k=2)
            delaunay_char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))

    for i, vertex in enumerate(vertices):
        if (
            use_delaunay
            and delaunay_tri is not None
            and top_positions_2d is not None
            and top_atom_indices is not None
        ):
            site_type, nearest_idx = _delaunay_site_classification(
                vertex,
                top_positions_2d,
                top_atom_indices,
                delaunay_tri,
                positions,
                vertex_2d=vertex_2d[i],
                char_len=delaunay_char_len,
                positions_tree=local_tree,
            )
        elif class_dists is not None and class_idx is not None:
            site_type, nearest_idx = _classify_voronoi_site_from_neighbors(
                class_dists[i],
                class_idx[i],
                pore_threshold=pore_threshold,
            )
        else:
            raise ValueError(
                "Voronoi neighbour arrays required when Delaunay classification "
                "is not used"
            )

        env_fingerprint = (
            tuple(sorted(symbols[j] for j in nearest_idx if j < len(symbols))),
            site_type,
        )
        hollow_order: int | None = None
        if site_type == "hollow":
            if (
                use_delaunay
                and delaunay_tri is not None
                and top_positions_2d is not None
                and top_atom_indices is not None
            ):
                hollow_order = 3
            elif class_dists is not None:
                hollow_order = _hollow_coordination_order(class_dists[i])
            if hollow_order is None and nearest_idx:
                hollow_order = len(nearest_idx)
        normal = (
            normals[i]
            if normals is not None
            else _compute_local_normal(vertex, positions, tree=local_tree)
        )
        sites.append(
            {
                "xy": vertex[:2].copy(),
                "z": float(vertex[2]),
                "xyz": vertex.copy(),
                "site_type": site_type,
                "slab_indices": nearest_idx,
                "normal": normal,
                "nn_distance": float(nn_dists[i]) if i < len(nn_dists) else None,
                "site_source": (
                    source_hints[i]
                    if source_hints is not None and i < len(source_hints)
                    else "voronoi"
                ),
                "material_type": material_type,
                "env_fingerprint": env_fingerprint,
                "hollow_order": hollow_order,
            }
        )
    return sites


# ---------------------------------------------------------------------------
# Unified site dict builder
# ---------------------------------------------------------------------------


def get_unified_sites(
    atoms: Atoms,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    top_layer_tolerance: float | None = None,
    material_type: str | None = None,
    pore_threshold: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "distance_ratio",
) -> list[dict[str, object]]:
    """Return adsorption/placement site dicts for *atoms*.

    Improved default behaviour
    --------------------------
    - slabs use a hybrid site generator: topology-derived surface sites plus
      Voronoi enrichment
    - rotated slabs are handled using the slab normal rather than Cartesian z
    """
    if material_type is None:
        raise ValueError(
            "material_type must be explicitly specified: 'slab', 'nanoparticle', or 'porous'"
        )
    if material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
        )

    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)

    pbc_for_voronoi = pbc.copy()
    if material_type == "nanoparticle":
        pbc_for_voronoi[:] = False

    symbols = atoms.get_chemical_symbols()
    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(positions, symbols)
    if pore_threshold is None:
        pore_threshold = _derive_pore_threshold(symbols)

    if np.linalg.det(cell) <= 0:
        cell = _bounding_box_cell(positions)
        if np.any(pbc_for_voronoi):
            logger.warning(
                "Input cell is degenerate while PBC is enabled; using a padded bounding-box cell for site enumeration"
            )

    if probe_radius is None or max_site_distance is None:
        derived_probe, derived_max = _derive_voronoi_distance_window(
            positions, symbols, pbc_for_voronoi, cell
        )
        probe_radius = derived_probe if probe_radius is None else probe_radius
        max_site_distance = (
            derived_max if max_site_distance is None else max_site_distance
        )

    voronoi_positions = positions
    if material_type == "slab":
        top_only_mask = _top_layer_mask_by_normal(
            positions, cell, float(top_layer_tolerance)
        )
        top_only = positions[top_only_mask]
        if len(top_only) >= 4:
            voronoi_positions = top_only

    vertices, nn_dists = _voronoi_sites(
        voronoi_positions,
        cell,
        pbc_for_voronoi,
        probe_radius=probe_radius,
        max_distance=max_site_distance,
        enrich=enrich,
        symbols=symbols,
        nn_reference_positions=positions,
    )
    source_hints = ["voronoi"] * len(vertices)

    local_tree = KDTree(positions)

    slab_top_mask: np.ndarray | None = None
    slab_top_atom_indices: np.ndarray | None = None
    slab_has_topology_atop = False

    # Slab-specific topology enrichment becomes part of the default generator.
    if material_type == "slab":
        slab_top_mask = _top_layer_mask_by_normal(
            positions, cell, float(top_layer_tolerance)
        )
        slab_top_atom_indices = np.nonzero(slab_top_mask)[0]
        median_nn = (
            float(np.median(nn_dists))
            if len(nn_dists) > 0
            else _VORONOI_MAX_DISTANCE_COVALENT_SCALE
            * _VORONOI_RADIUS_FALLBACK_ANGSTROM
        )
        site_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
        topo_vertices, topo_dists, topo_sources = _generate_slab_topology_sites(
            positions,
            cell,
            pbc_for_voronoi,
            slab_top_atom_indices,
            local_tree,
            site_height,
            float(probe_radius),
            float(max_site_distance),
        )
        slab_has_topology_atop = any(s == "topology_atop" for s in topo_sources)
        if len(topo_vertices) > 0:
            if len(vertices) == 0:
                vertices = topo_vertices
                nn_dists = topo_dists
                source_hints = topo_sources
            else:
                vertices = np.vstack([vertices, topo_vertices])
                nn_dists = np.concatenate([nn_dists, topo_dists])
                source_hints = source_hints + topo_sources
                keep = _deduplicate_points(
                    vertices, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc_for_voronoi
                )
                kept_idx = np.nonzero(keep)[0]
                vertices = vertices[keep]
                nn_dists = nn_dists[keep]
                source_hints = [source_hints[i] for i in kept_idx]

    if len(vertices) == 0:
        logger.warning(
            "No accessible sites for %d-atom structure (probe_radius=%s, max_distance=%s, material_type=%r)",
            len(atoms),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_distance:.2f}" if max_site_distance is not None else "auto",
            material_type,
        )
        return []

    if material_type == "slab":
        heights = _height_along_slab_normal(positions, cell)
        h_surface = float(np.max(heights))
        nn_margin = (
            float(np.median(nn_dists))
            if len(nn_dists) > 0
            else float(top_layer_tolerance)
        )
        h_min = h_surface - max(float(top_layer_tolerance), nn_margin)
        keep_mask = _height_along_slab_normal(vertices, cell) >= h_min
        vertices = vertices[keep_mask]
        nn_dists = nn_dists[keep_mask]
        source_hints = [source_hints[i] for i in np.nonzero(keep_mask)[0]]

    if material_type == "nanoparticle" and len(vertices) > 0:
        com = np.mean(positions, axis=0)
        k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)
        outward = np.einsum("ij,ij->i", normals, vertices - com) > 0.0
        vertices = vertices[outward]
        nn_dists = nn_dists[outward]
        source_hints = [source_hints[i] for i in np.nonzero(outward)[0]]

    # Atop injection safety net for nanoparticles; for slabs only when topology
    # did not already produce atop candidates.
    skip_slab_atop_injection = material_type == "slab" and slab_has_topology_atop
    if material_type in ("slab", "nanoparticle") and len(vertices) > 0:
        if material_type == "slab" and skip_slab_atop_injection:
            pass
        else:
            median_nn = (
                float(np.median(nn_dists))
                if len(nn_dists) > 0
                else _VORONOI_MAX_DISTANCE_COVALENT_SCALE
                * _VORONOI_RADIUS_FALLBACK_ANGSTROM
            )
            atop_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn

            if material_type == "slab":
                if slab_top_atom_indices is None:
                    slab_top_mask = _top_layer_mask_by_normal(
                        positions, cell, float(top_layer_tolerance)
                    )
                    slab_top_atom_indices = np.nonzero(slab_top_mask)[0]
                top_atom_indices = slab_top_atom_indices
                atom_normals: np.ndarray | None = None
            else:
                com = np.mean(positions, axis=0)
                k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
                _, norm_idx_all = local_tree.query(positions, k=k_norm)
                if np.ndim(norm_idx_all) == 1:
                    norm_idx_all = np.asarray(norm_idx_all, dtype=int).reshape(-1, 1)
                atom_normals = _compute_local_normals_batch(
                    positions, positions, norm_idx_all
                )
                outward_dots = np.einsum("ij,ij->i", atom_normals, positions - com)
                top_atom_indices = np.nonzero(outward_dots > 0.0)[0].astype(int)

            candidate_verts: list[np.ndarray] = []
            candidate_dists: list[float] = []
            candidate_sources: list[str] = []
            for ai in top_atom_indices:
                atom_pos = positions[int(ai)]
                if material_type == "slab":
                    candidate = _shift_along_slab_normal(
                        atom_pos.reshape(1, 3), cell, atop_height
                    )[0]
                    if np.any(pbc_for_voronoi):
                        candidate = _wrap_cartesian(
                            candidate.reshape(1, 3), cell, pbc_for_voronoi
                        )[0]
                else:
                    assert atom_normals is not None
                    candidate = atom_pos + atop_height * atom_normals[int(ai)]

                d_nn = float(
                    local_tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0]
                )
                if d_nn < float(probe_radius) or d_nn > float(max_site_distance):
                    continue
                candidate_verts.append(candidate)
                candidate_dists.append(d_nn)
                candidate_sources.append("atop_injected")

            if candidate_verts:
                candidate_arr = np.asarray(candidate_verts, dtype=float)
                keep_new = _filter_non_duplicate_candidates(
                    candidate_arr, vertices, _VORONOI_DEDUP_TOLERANCE
                )
                candidate_arr = candidate_arr[keep_new]
                candidate_dist_arr = np.asarray(candidate_dists, dtype=float)[keep_new]
                candidate_sources = [
                    candidate_sources[i] for i in np.nonzero(keep_new)[0]
                ]
                if len(candidate_arr) > 0:
                    combined = np.vstack([vertices, candidate_arr])
                    combined_dists = np.concatenate([nn_dists, candidate_dist_arr])
                    combined_sources = source_hints + candidate_sources
                    keep = _deduplicate_points(
                        combined,
                        _VORONOI_DEDUP_TOLERANCE,
                        cell=cell,
                        pbc=pbc_for_voronoi,
                    )
                    kept_idx = np.nonzero(keep)[0]
                    n_existing = len(vertices)
                    n_injected = int(np.count_nonzero(keep[n_existing:]))
                    vertices = combined[keep]
                    nn_dists = combined_dists[keep]
                    source_hints = [combined_sources[i] for i in kept_idx]
                    logger.debug(
                        "Injected %d atop candidate sites (%d total sites)",
                        n_injected,
                        len(vertices),
                    )

    if len(vertices) == 0:
        return []

    # Use Delaunay automatically for slabs when possible, even if the legacy
    # default argument is still 'distance_ratio'. This improves default labeling.
    _use_delaunay = material_type == "slab" and site_classification_method in (
        "delaunay",
        "distance_ratio",
        "auto",
    )
    _delaunay_tri = None
    _top_positions_2d: np.ndarray | None = None
    _top_atom_indices: np.ndarray | None = None
    if _use_delaunay:
        if material_type == "slab" and slab_top_atom_indices is not None:
            _top_atom_indices = slab_top_atom_indices
        else:
            top_mask = _top_layer_mask_by_normal(
                positions, cell, float(top_layer_tolerance)
            )
            _top_atom_indices = np.nonzero(top_mask)[0]
        if len(_top_atom_indices) >= 3:
            top_positions = positions[_top_atom_indices]
            _top_positions_2d = _project_to_slab_plane(top_positions, cell)
            try:
                _delaunay_tri = Delaunay(_top_positions_2d)
            except (QhullError, ValueError, RuntimeError) as exc:
                logger.debug("Delaunay classification disabled (%s)", exc)
                _use_delaunay = False
        else:
            _use_delaunay = False

    sites = _build_site_records(
        vertices,
        nn_dists,
        positions,
        symbols,
        local_tree,
        material_type,
        pore_threshold,
        use_delaunay=_use_delaunay,
        delaunay_tri=_delaunay_tri,
        top_positions_2d=_top_positions_2d,
        top_atom_indices=_top_atom_indices,
        cell=cell,
        source_hints=source_hints,
    )

    if np.linalg.det(cell) > 0:

        def _site_frac_key(site: dict[str, object]) -> tuple:
            frac = _wrap_fractional(
                _cart_to_frac(np.asarray(site["xyz"], dtype=float).reshape(1, 3), cell),
                pbc_for_voronoi,
            )[0]
            return (
                float(frac[0]),
                float(frac[1]),
                float(frac[2]),
                str(site["site_type"]),
            )

        sites.sort(key=_site_frac_key)

    return sites


# ---------------------------------------------------------------------------
# Hollow sites for adatom / dissociative placement
# ---------------------------------------------------------------------------


def get_hollow_sites_for_adatoms(
    slab: Atoms,
    top_layer_tolerance: float = _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    dedup_tolerance: float = DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
) -> list[np.ndarray]:
    """Hollow/pore site xy positions for adatom placement, deduplicated."""
    raw = get_unified_sites(
        slab,
        top_layer_tolerance=top_layer_tolerance,
        material_type="slab",
    )
    hollow_sites = [s for s in raw if s.get("site_type") in ("hollow", "pore")]
    if not hollow_sites:
        return []
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.array([bool(slab.get_pbc()[0]), bool(slab.get_pbc()[1]), False])
    hollow_xyz = np.array(
        [np.asarray(s["xyz"], dtype=float) for s in hollow_sites], dtype=float
    )
    keep = _deduplicate_points(hollow_xyz, dedup_tolerance, cell=cell, pbc=pbc)
    return [np.asarray(hollow_sites[i]["xy"]) for i in np.nonzero(keep)[0]]


# ---------------------------------------------------------------------------
# Environment-aware site clustering
# ---------------------------------------------------------------------------


def _env_fingerprint(site: dict[str, object]) -> tuple:
    """Return the local-environment fingerprint of *site*."""
    fp = site.get("env_fingerprint")
    if fp is not None:
        return cast(tuple, fp)
    return (str(site.get("site_type", "")),)


def _cluster_with_metric(
    n: int,
    coords: np.ndarray,
    fps: list[tuple],
    *,
    image_offsets: list[np.ndarray] | None,
    kdtree_radius: float,
    pair_filter: Callable[[int, int, np.ndarray], bool] | None = None,
) -> list[int]:
    """KDTree query_pairs + union-find; one representative index per cluster."""
    if image_offsets:
        all_coords = np.vstack([coords + off for off in image_offsets])
    else:
        all_coords = coords

    tree = KDTree(all_coords)
    raw_pairs = tree.query_pairs(r=kdtree_radius, output_type="ndarray")

    merge_set: set[tuple[int, int]] = set()
    for a_exp, b_exp in raw_pairs:
        a = int(a_exp) % n
        b = int(b_exp) % n
        if a == b:
            continue
        if fps[a] != fps[b]:
            continue
        if pair_filter is not None and not pair_filter(a, b, coords):
            continue
        merge_set.add((min(a, b), max(a, b)))

    components = _union_find_cluster(n, list(merge_set))
    return sorted(min(comp) for comp in components)


def _cluster_equivalent_sites(
    sites: list[dict[str, object]],
    cell: np.ndarray,
    tolerance: float = DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    z_abs_tolerance: float | None = None,
) -> list[dict[str, object]]:
    """Group equivalent sites; return unique representatives."""
    if not sites:
        return []

    n = len(sites)
    mat_type = material_type_for_placement(sites[0], when_no_site="slab")

    def _get_xyz(s: dict[str, object]) -> np.ndarray:
        if "xyz" in s:
            return np.asarray(s["xyz"], dtype=float)
        return np.array([*np.asarray(s["xy"], dtype=float), float(s.get("z", 0.0))])

    def _sort_key(s: dict[str, object]) -> tuple:
        xyz = _get_xyz(s)
        return (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            str(s.get("site_type", "")),
        )

    order = sorted(range(n), key=lambda i: _sort_key(sites[i]))
    sorted_sites = [sites[i] for i in order]
    fps = [_env_fingerprint(s) for s in sorted_sites]

    if mat_type == "nanoparticle" or np.linalg.det(cell) <= 0:
        coords = np.array([_get_xyz(s) for s in sorted_sites])
        reps = _cluster_with_metric(
            n,
            coords,
            fps,
            image_offsets=None,
            kdtree_radius=tolerance,
        )
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    if mat_type == "porous":
        coords = np.array([_get_xyz(s) for s in sorted_sites])
        pbc_full = np.array([True, True, True])
        image_offsets = _periodic_image_offsets(cell, pbc_full, tolerance)
        reps = _cluster_with_metric(
            n,
            coords,
            fps,
            image_offsets=image_offsets,
            kdtree_radius=tolerance,
        )
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    z_tol = (
        z_abs_tolerance
        if z_abs_tolerance is not None
        else _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM
    )
    pinv_ab_T, _ = _slab_plane_projectors(cell)
    coords = np.array([_get_xyz(s) for s in sorted_sites])
    heights = _height_along_slab_normal(coords, cell)
    pbc_slab = np.array([True, True, False])
    image_offsets = _periodic_image_offsets(cell, pbc_slab, tolerance)

    def _slab_pair_filter(a: int, b: int, _coords_arr: np.ndarray) -> bool:
        xyz_a = coords[a]
        xyz_b = coords[b]
        delta_frac = _minimum_image_fractional_delta(
            _cart_to_frac((xyz_b - xyz_a).reshape(1, 3), cell),
            pbc_slab,
        )[0]
        delta_cart = _frac_to_cart(delta_frac.reshape(1, 3), cell)[0]
        dxy = float(np.linalg.norm(delta_cart[:2]))
        dz = abs(float(heights[a]) - float(heights[b]))
        return dxy < tolerance and dz < z_tol

    r_search = tolerance * _KD_RADIUS_SEARCH_PADDING
    reps = _cluster_with_metric(
        n,
        coords,
        fps,
        image_offsets=image_offsets,
        kdtree_radius=r_search,
        pair_filter=_slab_pair_filter,
    )
    result = [sorted_sites[i] for i in reps]

    def _slab_coord(s: dict[str, object]) -> np.ndarray:
        xyz = _get_xyz(s)
        frac2 = xyz @ pinv_ab_T
        frac2 = frac2 - np.floor(frac2)
        z = float(cast(float, s.get("z", xyz[2])))
        return np.array([float(frac2[0]), float(frac2[1]), z])

    def _slab_key(s: dict[str, object]) -> tuple:
        c = _slab_coord(s)
        return (float(c[0]), float(c[1]), float(c[2]), str(s.get("site_type", "")))

    return sorted(result, key=_slab_key)


# ---------------------------------------------------------------------------
# Symmetry-aware site reduction
# ---------------------------------------------------------------------------


def get_symmetry_aware_sites(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    material_type: str = "slab",
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "distance_ratio",
    raw_sites: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Symmetry-reduced adsorption sites using spglib."""
    if material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
        )

    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(
            slab.get_positions(),
            slab.get_chemical_symbols(),
        )

    if raw_sites is not None:
        site_list = raw_sites
    else:
        site_list = get_unified_sites(
            slab,
            probe_radius=probe_radius,
            max_site_distance=max_site_distance,
            top_layer_tolerance=top_layer_tolerance,
            material_type=material_type,
            enrich=enrich,
            site_classification_method=site_classification_method,
        )
    if not site_list:
        return []

    sym_mode = "cluster" if material_type == "nanoparticle" else "auto"
    planar_for_symmetry = (material_type == "slab") and _is_top_layer_planar(
        slab, top_layer_tolerance
    )

    symmetry_analyzer = SymmetryAnalyzer(
        slab,
        symmetry_tolerance=symmetry_tolerance,
        mode=sym_mode,
    )
    return symmetry_analyzer.analyze_site_symmetry(
        site_list,
        planar=planar_for_symmetry,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_top_layer_planar(
    slab: Atoms,
    top_layer_tolerance: float = _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
) -> bool:
    """True if the topmost atomic layer is approximately flat.

    The fit is done in an orientation-aware slab coordinate system rather than
    assuming the slab normal is Cartesian z.
    """
    positions = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    top_mask = _top_layer_mask_by_normal(positions, cell, float(top_layer_tolerance))
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return False
    top_pos = positions[top_indices]
    xy = _project_to_slab_plane(top_pos, cell)
    h = _height_along_slab_normal(top_pos, cell)
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    coeffs, residuals, rank, _ = np.linalg.lstsq(A, h, rcond=None)
    if rank < 3 or len(residuals) == 0:
        return False
    h_pred = A @ coeffs
    return float(np.var(h - h_pred)) < z_variance_threshold


def _bounding_box_cell(
    positions: np.ndarray,
    pad: float = _BOUNDING_BOX_CELL_PAD_ANGSTROM,
) -> np.ndarray:
    """Orthorhombic cell spanning atomic positions plus padding."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo + pad, pad)
    return np.diag(span)


# ---------------------------------------------------------------------------
# Z-base range computation (used by generators.py)
# ---------------------------------------------------------------------------


def _get_site_surface_radii(
    slab: Atoms,
    site: dict[str, object] | None = None,
) -> float | None:
    """Mean covalent radius of framework atoms nearest to the placement site."""
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    cell = np.asarray(slab.get_cell(), dtype=float)

    if site is not None and "slab_indices" in site:
        indices = site["slab_indices"]
        if not indices:
            indices = None
    else:
        indices = None

    if indices is None:
        top_depth = _derive_top_layer_tolerance(positions, symbols)
        top_mask = _top_layer_mask_by_normal(positions, cell, float(top_depth))
        indices = tuple(int(i) for i in np.nonzero(top_mask)[0])

    radii = [_get_covalent_radius(symbols[int(i)]) for i in indices]
    radii = [r for r in radii if r is not None]
    if not radii:
        return None
    return float(np.mean(radii))


def _compute_site_z_base(
    config,
    slab: Atoms,
    site: dict[str, object] | None,
    mol_symbols: list[str],
) -> tuple[float, float]:
    """Compute z-offset range for placement above *site*."""
    z_lo, z_hi = config.placement_z_range

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    if (
        mat_type != "slab"
        and site is not None
        and site.get("nn_distance") is not None
        and str(site.get("site_type", "")) != "pore"
    ):
        nn = float(site["nn_distance"])
        nn_lo = nn * _NON_SLAB_Z_LO_FROM_NN_SCALE
        nn_hi = nn * _NON_SLAB_Z_HI_FROM_NN_SCALE
        if nn_hi - nn_lo < z_hi - z_lo:
            z_lo, z_hi = nn_lo, nn_hi

    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi
    if site is not None and str(site.get("site_type", "")) == "pore":
        return z_lo, z_hi

    r_surface = _get_site_surface_radii(slab, site)
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else _MOL_COVALENT_RADIUS_FALLBACK

    if r_surface is None:
        return z_lo, z_hi

    delta = _SITE_Z_RADIUS_SHIFT_SCALE * (
        r_mol + r_surface - _SITE_Z_RADIUS_REFERENCE_ANGSTROM
    )
    return z_lo + delta, z_hi + delta


# ---------------------------------------------------------------------------
# Public API (for external imports)
# ---------------------------------------------------------------------------


def get_symmetry_info(
    slab: Atoms,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, object]:
    """Symmetry metadata including spglib space group."""
    symmetry_analyzer = SymmetryAnalyzer(slab, symmetry_tolerance=symmetry_tolerance)
    return symmetry_analyzer.get_symmetry_info()

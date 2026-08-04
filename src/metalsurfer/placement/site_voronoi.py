"""Voronoi site generation, topology candidates, ridge enrichment, and classification."""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial import Delaunay, KDTree, QhullError, Voronoi

from ._constants import (
    _ATOP_RATIO,
    _BRIDGE_EQ_TOL,
    _BRIDGE_FAR_RATIO,
    _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM,
    _DISTANCE_RATIO_FLOOR_EPS,
    _DISTANCE_ZERO_EPS,
    _ENRICHMENT_MAX_SUBDIVISIONS,
    _ENRICHMENT_SPACING_BETA,
    _HOLLOW_EQ_TOL,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_FRACTIONAL_CELL_MARGIN,
)
from .site_coords import (
    _build_periodic_images,
    _cart_to_frac,
    _deduplicate_points,
    _derive_voronoi_distance_window,
    _frac_to_cart,
    _minimum_image_fractional_delta,
    _project_to_slab_plane,
    _slab_normal,
    _wrap_cartesian,
)

logger = logging.getLogger(__name__)


def _expand_top_layer_ab_images(
    top_xy: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    top_positions_3d: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int], list[tuple[int, int]], np.ndarray | None]:
    """Expand top-layer 2D points by ±1 images along periodic a/b.

    Returns ``(exp_xy, origin_local, image_id, exp_3d_or_None)``.
    """
    ranges_a = (-1, 0, 1) if bool(pbc[0]) else (0,)
    ranges_b = (-1, 0, 1) if bool(pbc[1]) else (0,)
    exp_xy: list[np.ndarray] = []
    origin_local: list[int] = []
    image_id: list[tuple[int, int]] = []
    exp_3d: list[np.ndarray] | None = [] if top_positions_3d is not None else None
    for ia in ranges_a:
        for ib in ranges_b:
            offset = ia * cell[0] + ib * cell[1]
            off_2d = _project_to_slab_plane(offset.reshape(1, 3), cell)[0]
            for li in range(len(top_xy)):
                exp_xy.append(top_xy[li] + off_2d)
                origin_local.append(int(li))
                image_id.append((int(ia), int(ib)))
                if exp_3d is not None and top_positions_3d is not None:
                    exp_3d.append(top_positions_3d[li] + offset)
    exp3d_arr = np.asarray(exp_3d, dtype=float) if exp_3d is not None else None
    return np.asarray(exp_xy, dtype=float), origin_local, image_id, exp3d_arr


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
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    Delaunay | None,
]:
    """Generate slab atop/bridge/hollow candidates from the top layer.

    Candidates are created in an orientation-aware way and wrapped back into the
    reference cell on periodic axes.

    Returns ``(vertices, nn_dists, sources, primary_delaunay)`` where
    *primary_delaunay* is the top-layer triangulation in the slab plane (for
    classification reuse), or ``None`` when unavailable.
    """
    if len(top_atom_indices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float), [], None

    n_hat = _slab_normal(cell)
    top_atom_indices = np.asarray(top_atom_indices, dtype=int)
    top_positions = positions[top_atom_indices]

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

    top_positions_2d = _project_to_slab_plane(top_positions, cell)
    primary_delaunay: Delaunay | None = None
    if len(top_positions_2d) >= 3:
        try:
            primary_delaunay = Delaunay(top_positions_2d)
        except (QhullError, ValueError, RuntimeError):
            primary_delaunay = None

    # Need 2D triangulation for bridge/hollow candidates.
    if len(top_positions) < 2:
        if not candidates:
            return (
                np.empty((0, 3), dtype=float),
                np.empty(0, dtype=float),
                [],
                primary_delaunay,
            )
        cand_arr = np.asarray(candidates, dtype=float)
        keep = _deduplicate_points(
            cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
        )
        return (
            cand_arr[keep],
            np.asarray(candidate_dists, dtype=float)[keep],
            [candidate_sources[i] for i in np.nonzero(keep)[0]],
            primary_delaunay,
        )

    exp2d, expanded_origin_local_index, _image_id, exp3d = _expand_top_layer_ab_images(
        top_positions_2d,
        cell=cell,
        pbc=pbc,
        top_positions_3d=top_positions,
    )
    assert exp3d is not None
    tri: Delaunay | None = None
    if len(exp2d) >= 3:
        try:
            tri = Delaunay(exp2d)
        except (QhullError, ValueError, RuntimeError):
            tri = None

    if tri is not None:
        seen_edges: set[tuple[int, int, float, float, float]] = set()
        for simplex in np.asarray(tri.simplices, dtype=int):
            for e0, e1 in ((0, 1), (1, 2), (0, 2)):
                i_exp, j_exp = int(simplex[e0]), int(simplex[e1])
                li = expanded_origin_local_index[i_exp]
                lj = expanded_origin_local_index[j_exp]
                if li == lj:
                    continue
                midpoint = (
                    0.5 * (exp3d[i_exp] + exp3d[j_exp]) + float(site_height) * n_hat
                )
                edge_key = (
                    min(li, lj),
                    max(li, lj),
                    round(float(midpoint[0]), 5),
                    round(float(midpoint[1]), 5),
                    round(float(midpoint[2]), 5),
                )
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                _add_candidate(midpoint, "topology_bridge")

        seen_tris: set[tuple[int, int, int, float, float, float]] = set()
        for simplex in np.asarray(tri.simplices, dtype=int):
            local_ids = tuple(
                sorted({expanded_origin_local_index[int(k)] for k in simplex})
            )
            if len(local_ids) != 3:
                continue
            centroid = np.mean(exp3d[simplex], axis=0) + float(site_height) * n_hat
            tri_key = (
                local_ids[0],
                local_ids[1],
                local_ids[2],
                round(float(centroid[0]), 5),
                round(float(centroid[1]), 5),
                round(float(centroid[2]), 5),
            )
            if tri_key in seen_tris:
                continue
            seen_tris.add(tri_key)
            _add_candidate(centroid, "topology_hollow")

    if not candidates:
        return (
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            [],
            primary_delaunay,
        )

    cand_arr = np.asarray(candidates, dtype=float)
    cand_dist = np.asarray(candidate_dists, dtype=float)
    keep = _deduplicate_points(cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    kept_idx = np.nonzero(keep)[0]
    return (
        cand_arr[keep],
        cand_dist[keep],
        [candidate_sources[i] for i in kept_idx],
        primary_delaunay,
    )


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
        verts = [int(v) for v in ridge]
        if len(verts) < 2:
            continue
        # 3D Voronoi ridges are polygonal faces; walk consecutive finite
        # vertex pairs (and close the loop when the ridge is bounded).
        pair_indices: list[tuple[int, int]] = []
        for i in range(len(verts) - 1):
            pair_indices.append((verts[i], verts[i + 1]))
        if len(verts) >= 3 and all(v >= 0 for v in verts):
            pair_indices.append((verts[-1], verts[0]))
        for r0, r1 in pair_indices:
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


def _build_delaunay_classification_index(
    top_positions_2d: np.ndarray,
    top_atom_indices: np.ndarray,
    triangulation: Delaunay | None = None,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, ...]]]:
    """Precompute (atop, bridge midpoint, hollow centroid) XY candidates for KDTree classify.

    When *cell* and *pbc* are provided, also append bridge candidates from ±1
    a/b images (cross-boundary edges only). Callers should classify with the
    primary index first and only consult the PBC-augmented index to upgrade
    near-boundary atops, so interior hollow labels stay stable.
    """
    top_xy = np.asarray(top_positions_2d, dtype=float)
    top_atom_indices = np.asarray(top_atom_indices, dtype=int)
    cand_xy: list[np.ndarray] = []
    cand_types: list[str] = []
    cand_indices: list[tuple[int, ...]] = []

    for li, gi in enumerate(top_atom_indices):
        cand_xy.append(top_xy[li])
        cand_types.append("atop")
        cand_indices.append((int(gi),))

    def _add_from_triangulation(
        work_xy: np.ndarray,
        work_tri: Delaunay,
        origin_local: list[int],
        *,
        image_id: list[tuple[int, int]] | None = None,
        only_cross_boundary: bool = False,
    ) -> None:
        seen: set[tuple] = set()
        for simplex in np.asarray(work_tri.simplices, dtype=int):
            if only_cross_boundary and image_id is not None:
                imgs = {image_id[int(k)] for k in simplex}
                if imgs == {(0, 0)}:
                    continue
            for e0, e1 in ((0, 1), (1, 2), (0, 2)):
                i_exp, j_exp = int(simplex[e0]), int(simplex[e1])
                li0 = origin_local[i_exp]
                li1 = origin_local[j_exp]
                if li0 == li1:
                    continue
                if (
                    only_cross_boundary
                    and image_id is not None
                    and image_id[i_exp] == (0, 0)
                    and image_id[j_exp] == (0, 0)
                ):
                    continue
                mid = (work_xy[i_exp] + work_xy[j_exp]) / 2.0
                mid_key = (
                    "bridge",
                    min(li0, li1),
                    max(li0, li1),
                    round(float(mid[0]), 5),
                    round(float(mid[1]), 5),
                )
                if mid_key in seen:
                    continue
                seen.add(mid_key)
                cand_xy.append(mid)
                cand_types.append("bridge")
                cand_indices.append(
                    (int(top_atom_indices[li0]), int(top_atom_indices[li1]))
                )

            local_ids = tuple(sorted({origin_local[int(k)] for k in simplex}))
            if len(local_ids) != 3:
                continue
            # Cross-boundary pass only adds bridges; primary hollows stay authoritative
            # so interior labels are not pulled toward image centroids.
            if only_cross_boundary:
                continue
            centroid = np.mean(work_xy[list(simplex)], axis=0)
            hollow_key = (
                "hollow",
                local_ids,
                round(float(centroid[0]), 5),
                round(float(centroid[1]), 5),
            )
            if hollow_key in seen:
                continue
            seen.add(hollow_key)
            cand_xy.append(centroid)
            cand_types.append("hollow")
            cand_indices.append(
                (
                    int(top_atom_indices[local_ids[0]]),
                    int(top_atom_indices[local_ids[1]]),
                    int(top_atom_indices[local_ids[2]]),
                )
            )

    if triangulation is not None:
        _add_from_triangulation(
            top_xy,
            triangulation,
            list(range(len(top_xy))),
            only_cross_boundary=False,
        )

    if cell is not None and pbc is not None and (bool(pbc[0]) or bool(pbc[1])):
        work_xy, origin_local, image_id, _ = _expand_top_layer_ab_images(
            top_xy, cell=cell, pbc=pbc
        )
        if len(work_xy) >= 3:
            try:
                work_tri = Delaunay(work_xy)
            except (QhullError, ValueError, RuntimeError):
                work_tri = None
            if work_tri is not None:
                _add_from_triangulation(
                    work_xy,
                    work_tri,
                    origin_local,
                    image_id=image_id,
                    only_cross_boundary=True,
                )

    if not cand_xy:
        return np.empty((0, 2), dtype=float), [], []
    return np.asarray(cand_xy, dtype=float), cand_types, cand_indices


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
    cand_xy: np.ndarray | None = None,
    cand_types: list[str] | None = None,
    cand_indices: list[tuple[int, ...]] | None = None,
    cand_tree: KDTree | None = None,
) -> tuple[str, tuple[int, ...]]:
    """Classify a slab site using Delaunay triangulation in the slab plane.

    When *cand_xy* / *cand_tree* are provided (preferred), classification is a
    nearest-neighbour query over precomputed atop/bridge/hollow candidates.
    """
    xy = np.asarray(vertex_2d, dtype=float)
    top_xy = np.asarray(top_positions_2d, dtype=float)

    if cand_xy is None or cand_types is None or cand_indices is None:
        cand_xy, cand_types, cand_indices = _build_delaunay_classification_index(
            top_positions_2d, top_atom_indices, triangulation
        )
    if cand_tree is None and len(cand_xy) > 0:
        cand_tree = KDTree(cand_xy)

    if char_len is None:
        if len(top_xy) >= 2:
            _top_tree = KDTree(top_xy)
            _nn_d, _ = _top_tree.query(top_xy, k=2)
            char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))
        else:
            char_len = _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM

    if cand_tree is None or len(cand_xy) == 0:
        return "hollow", tuple(int(i) for i in np.asarray(top_atom_indices)[:3])

    _dist, idx = cand_tree.query(xy.reshape(1, 2), k=1)
    nearest = int(np.asarray(idx, dtype=int).ravel()[0])
    best_type = cand_types[nearest]
    best_dist = float(np.asarray(_dist, dtype=float).ravel()[0])
    best_indices = cand_indices[nearest]

    if best_type == "bridge" and best_dist > bridge_threshold * float(char_len):
        _tree = positions_tree if positions_tree is not None else KDTree(positions)
        _, idx3 = _tree.query(vertex.reshape(1, 3), k=min(3, len(positions)))
        idx3 = np.asarray(idx3, dtype=int).ravel()
        best_type = "hollow"
        best_indices = tuple(int(i) for i in idx3[:3])

    return best_type, best_indices


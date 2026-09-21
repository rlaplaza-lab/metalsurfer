"""Voronoi site generation, topology candidates, ridge enrichment, and classification."""

import logging
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import numpy as np
from scipy.spatial import Delaunay, KDTree, QhullError, Voronoi

from .._utils import cell_has_volume
from ._constants import (
    _ATOP_RATIO,
    _BRIDGE_EQ_TOL,
    _BRIDGE_FAR_RATIO,
    _DISTANCE_RATIO_FLOOR_EPS,
    _DISTANCE_ZERO_EPS,
    _ENRICHMENT_MAX_SUBDIVISIONS,
    _ENRICHMENT_SPACING_BETA,
    _HOLLOW_EQ_TOL,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _VORONOI_DEDUP_TOLERANCE,
)
from ._parallel import resolve_materialize_workers
from .site_coords import (
    _build_periodic_images,
    _cart_to_frac,
    _deduplicate_points,
    _derive_voronoi_distance_window,
    _frac_to_cart,
    _minimum_image_fractional_delta,
    _periodic_image_offsets,
    _project_to_slab_plane,
    _slab_normal,
    _slab_plane_projectors,
    _wrap_cartesian,
    derive_pore_threshold,
)

logger = logging.getLogger(__name__)


def _expand_top_layer_ab_images(
    top_xy: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    top_positions_3d: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray | None]:
    """Expand top-layer 2D points by ±1 images along periodic a/b.

    Returns ``(exp_xy, origin_local, exp_3d_or_None)``.
    """
    ranges_a = (-1, 0, 1) if bool(pbc[0]) else (0,)
    ranges_b = (-1, 0, 1) if bool(pbc[1]) else (0,)
    exp_xy: list[np.ndarray] = []
    origin_local: list[int] = []
    exp_3d: list[np.ndarray] | None = [] if top_positions_3d is not None else None
    # Hoist the slab-plane basis once: every (ia, ib) image shares the same
    # ortho_basis, so projecting each offset via the repeated pinv is wasted.
    _, ortho_basis = _slab_plane_projectors(cell)
    for ia in ranges_a:
        for ib in ranges_b:
            offset = ia * cell[0] + ib * cell[1]
            off_2d = offset @ ortho_basis.T
            for li in range(len(top_xy)):
                exp_xy.append(top_xy[li] + off_2d)
                origin_local.append(int(li))
                if exp_3d is not None and top_positions_3d is not None:
                    exp_3d.append(top_positions_3d[li] + offset)
    exp3d_arr = np.asarray(exp_3d, dtype=float) if exp_3d is not None else None
    return np.asarray(exp_xy, dtype=float), origin_local, exp3d_arr


def _iter_unique_simplex_sites(
    simplices: np.ndarray,
    origin_local: list[int],
    base_points: np.ndarray,
) -> Iterator[tuple[str, tuple[int, ...], np.ndarray]]:
    """Yield unique bridge midpoints and hollow centroids over Delaunay simplices.

    Yields ``(kind, local_ids, point)`` where *kind* is ``"bridge"`` or
    ``"hollow"``, *local_ids* are the origin-cell indices contributing to the
    candidate, and *point* is computed from *base_points* (callers apply any
    height offsets themselves — a constant shift cannot change which candidates
    collide). Edges whose endpoints map to the same origin atom are skipped;
    repeats across adjacent simplices and periodic-image copies are emitted once.
    """
    seen_edges: set[tuple] = set()
    seen_tris: set[tuple] = set()
    for simplex in np.asarray(simplices, dtype=int):
        for e0, e1 in ((0, 1), (1, 2), (0, 2)):
            i_exp, j_exp = int(simplex[e0]), int(simplex[e1])
            li = origin_local[i_exp]
            lj = origin_local[j_exp]
            if li == lj:
                continue
            pair = (min(li, lj), max(li, lj))
            mid = 0.5 * (base_points[i_exp] + base_points[j_exp])
            edge_key = (pair, tuple(round(float(v), 5) for v in mid))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            yield "bridge", pair, mid

        local_ids = tuple(sorted({origin_local[int(k)] for k in simplex}))
        if len(local_ids) != 3:
            continue
        centroid = np.mean(base_points[list(simplex)], axis=0)
        tri_key = (local_ids, tuple(round(float(v), 5) for v in centroid))
        if tri_key in seen_tris:
            continue
        seen_tris.add(tri_key)
        yield "hollow", local_ids, centroid


def _empty_voronoi_result() -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Empty vertices / nn distances / support-atom tuples."""
    return (
        np.empty((0, 3), dtype=float),
        np.empty(0, dtype=float),
        [],
    )


def _voronoi_primary_supports(
    vertices: np.ndarray,
    framework_tree: KDTree,
    *,
    n_origin: int,
    pore_threshold: float,
) -> list[tuple[int, ...]]:
    """Primary-cell support atoms per vertex (empty for pore centres)."""
    n = len(vertices)
    if n == 0 or n_origin <= 0:
        return []
    k = min(_SITE_CLASSIFICATION_NEIGHBOURS, n_origin)
    dists, idx = framework_tree.query(vertices, k=k)
    dists_arr = np.asarray(dists, dtype=float)
    idx_arr = np.asarray(idx, dtype=int)
    if dists_arr.ndim == 1:
        dists_arr = dists_arr.reshape(-1, 1)
        idx_arr = idx_arr.reshape(-1, 1)
    out: list[tuple[int, ...]] = []
    for i in range(n):
        primary = np.asarray(idx_arr[i], dtype=int) % int(n_origin)
        _site_type, support = _classify_voronoi_site_from_neighbors(
            dists_arr[i],
            primary,
            pore_threshold=pore_threshold,
        )
        out.append(tuple(int(j) for j in support))
    return out


def _voronoi_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    probe_radius: float | None = None,
    max_distance: float | None = None,
    enrich: bool = True,
    *,
    symbols: list[str],
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, ...]]]:
    """Voronoi vertices accessible for adsorption, optionally enriched.

    Returns ``(vertices, nn_dists, atom_indices)``. Wall-near vertices carry
    primary-cell support atoms for fingerprints; pore centres keep empty
    supports so distance-ratio typing can still label them ``pore``.
    """
    if len(positions) < 4:
        return _empty_voronoi_result()

    if probe_radius is None or max_distance is None:
        derived_probe, derived_max = _derive_voronoi_distance_window(
            positions, symbols, pbc, cell
        )
        probe_radius = derived_probe if probe_radius is None else probe_radius
        max_distance = derived_max if max_distance is None else max_distance

    if not cell_has_volume(cell):
        logger.debug(
            "Degenerate cell for Voronoi generation; falling back to no-PBC enumeration"
        )
        pbc = np.zeros(3, dtype=bool)

    extension_margin = float(max_distance) + _VORONOI_DEDUP_TOLERANCE
    image_offsets = _periodic_image_offsets(cell, pbc, extension_margin)
    extended = _build_periodic_images(
        positions, cell, pbc, margin=extension_margin, offsets=image_offsets
    )

    try:
        vor = Voronoi(extended)
    except (QhullError, ValueError, RuntimeError) as exc:
        logger.debug("Voronoi computation failed (%s); returning no vertices", exc)
        return _empty_voronoi_result()

    raw_vertices = np.asarray(vor.vertices, dtype=float)
    if len(raw_vertices) == 0:
        return _empty_voronoi_result()

    inv_cell = np.linalg.inv(cell) if np.any(pbc) else None
    wrapped_vertices = _wrap_cartesian(raw_vertices, cell, pbc, inv_cell=inv_cell)

    # Accessibility and returned nn distances must use wrapped (in-cell) sites:
    # raw vertices just outside the cell can have different framework distances.
    tree = KDTree(extended)
    nn_dists, _ = tree.query(wrapped_vertices, k=1)
    nn_dists = np.asarray(nn_dists, dtype=float).ravel()

    accessible = (nn_dists >= probe_radius) & (nn_dists <= max_distance)
    wrapped_vertices = wrapped_vertices[accessible]
    nn_dists = nn_dists[accessible]
    raw_accessible_indices = np.nonzero(accessible)[0]

    if len(wrapped_vertices) == 0:
        return _empty_voronoi_result()

    # Keep wrapped vertices; PBC dedup merges images. Do not filter on unwrapped
    # fractional coords (values just outside [0, 1) still wrap to valid sites).
    dedup_offsets = (
        _periodic_image_offsets(cell, pbc, _VORONOI_DEDUP_TOLERANCE)
        if np.any(pbc)
        else None
    )
    keep = _deduplicate_points(
        wrapped_vertices,
        _VORONOI_DEDUP_TOLERANCE,
        cell=cell,
        pbc=pbc,
        image_offsets=dedup_offsets,
    )
    vertices = wrapped_vertices[keep]
    nn_dists = nn_dists[keep]

    if len(vertices) == 0:
        return _empty_voronoi_result()

    n_origin = len(positions)
    if enrich and len(vertices) >= 2:
        kept_tree = KDTree(vertices)
        raw_to_kept: dict[int, int] = {}
        accessible_wrapped = wrapped_vertices
        dist_to_kept, idx_to_kept = kept_tree.query(accessible_wrapped, k=1)
        for raw_idx, d, kept_idx in zip(
            raw_accessible_indices, dist_to_kept, idx_to_kept, strict=False
        ):
            if float(d) <= _VORONOI_DEDUP_TOLERANCE:
                raw_to_kept[int(raw_idx)] = int(kept_idx)

        vertices, nn_dists = _enrich_along_ridges(
            vertices,
            nn_dists,
            vor.ridge_vertices,
            raw_to_kept,
            extended,
            tree,
            probe_radius,
            max_distance,
            cell=cell,
            pbc=pbc,
            n_origin=n_origin,
            inv_cell=inv_cell,
            dedup_offsets=dedup_offsets,
            n_jobs=n_jobs,
        )

    atom_indices = _voronoi_primary_supports(
        vertices,
        tree,
        n_origin=n_origin,
        pore_threshold=derive_pore_threshold(list(symbols)),
    )
    return vertices, nn_dists, atom_indices


# ---------------------------------------------------------------------------
# Topology-derived slab sites
# ---------------------------------------------------------------------------


class _SlabTopologyResult(NamedTuple):
    """Output of :func:`_generate_slab_topology_sites`.

    *atom_indices* are primary-cell support atoms for each kept vertex.
    *exp_xy* / *exp_origin* / *exp_tri* are the ±1 a/b expanded scratchpad
    (for classification reuse), or ``None`` when unavailable.
    """

    vertices: np.ndarray
    nn_dists: np.ndarray
    sources: list[str]
    atom_indices: list[tuple[int, ...]]
    primary_delaunay: Delaunay | None
    exp_xy: np.ndarray | None
    exp_origin: list[int] | None
    exp_tri: Delaunay | None


def _generate_slab_topology_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    top_atom_indices: np.ndarray,
    accessibility_tree: KDTree,
    site_height: float,
    probe_radius: float,
    max_distance: float,
    *,
    primary_delaunay: Delaunay | None = None,
    exp2d: np.ndarray | None = None,
    expanded_origin_local_index: list[int] | None = None,
    exp_tri: Delaunay | None = None,
    reuse_delaunay: bool = False,
) -> _SlabTopologyResult:
    """Generate slab atop/bridge/hollow candidates from the top layer.

    Candidates are created in an orientation-aware way and wrapped back into the
    reference cell on periodic axes. *accessibility_tree* must be MIC-aware
    under PBC (see :func:`metalsurfer.placement.site_plugins.helpers.periodic_accessibility_tree`).

    When *reuse_delaunay* is True, the provided primary/expanded Delaunay objects
    are reused (planar auto-widen) instead of rebuilding Qhull.
    """
    empty_vertices = np.empty((0, 3), dtype=float)
    empty_dists = np.empty(0, dtype=float)
    empty_atoms: list[tuple[int, ...]] = []
    if len(top_atom_indices) == 0:
        return _SlabTopologyResult(
            empty_vertices, empty_dists, [], empty_atoms, None, None, None, None
        )

    n_hat = _slab_normal(cell)
    top_atom_indices = np.asarray(top_atom_indices, dtype=int)
    top_positions = positions[top_atom_indices]

    candidates: list[np.ndarray] = []
    candidate_dists: list[np.ndarray] = []
    candidate_sources: list[str] = []
    candidate_atoms: list[tuple[int, ...]] = []

    def _add_candidates_batch(
        anchors: np.ndarray,
        source: str,
        atom_ids: list[tuple[int, ...]],
        *,
        probe_points: np.ndarray | None = None,
    ) -> None:
        """Gate accessibility on *probe_points*; store *anchors* as catalog xyz.

        *site_height* lift is probe-only. Catalog identity is the support-plane
        (or simplex) anchor so placement lifts once in the contact solve.
        """
        if len(anchors) == 0:
            return
        pts = np.asarray(anchors, dtype=float)
        if np.any(pbc):
            pts = _wrap_cartesian(pts, cell, pbc)
        probe = pts if probe_points is None else np.asarray(probe_points, dtype=float)
        if probe_points is not None and np.any(pbc):
            probe = _wrap_cartesian(probe, cell, pbc)
        dists, _ = accessibility_tree.query(probe, k=1)
        dists = np.asarray(dists, dtype=float).ravel()
        keep = (probe_radius <= dists) & (dists <= max_distance)
        if not np.any(keep):
            return
        kept = np.nonzero(keep)[0]
        candidates.append(pts[keep])
        candidate_dists.append(dists[keep])
        candidate_sources.extend([source] * len(kept))
        candidate_atoms.extend(atom_ids[i] for i in kept)

    # Single exit: assemble the (deduplicated) result from everything gathered.
    def _finalize() -> _SlabTopologyResult:
        if not candidates:
            return _SlabTopologyResult(
                empty_vertices,
                empty_dists,
                [],
                empty_atoms,
                primary_delaunay,
                exp2d,
                expanded_origin_local_index,
                exp_tri,
            )
        cand_arr = np.vstack(candidates)
        cand_dist = np.concatenate(candidate_dists)
        keep = _deduplicate_points(
            cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
        )
        kept_idx = np.nonzero(keep)[0]
        return _SlabTopologyResult(
            cand_arr[keep],
            cand_dist[keep],
            [candidate_sources[i] for i in kept_idx],
            [candidate_atoms[i] for i in kept_idx],
            primary_delaunay,
            exp2d,
            expanded_origin_local_index,
            exp_tri,
        )

    # Atop: catalog = atom positions; probe at site_height along the slab normal.
    atop_atoms = [(int(ai),) for ai in top_atom_indices]
    atop_probe = top_positions + float(site_height) * n_hat
    _add_candidates_batch(
        top_positions, "topology_atop", atop_atoms, probe_points=atop_probe
    )

    top_positions_2d = _project_to_slab_plane(top_positions, cell)
    if len(top_positions) < 2:
        if not reuse_delaunay:
            primary_delaunay = None
            exp2d = None
            expanded_origin_local_index = None
            exp_tri = None
        return _finalize()

    if not reuse_delaunay:
        primary_delaunay = None
        if len(top_positions_2d) >= 3:
            try:
                primary_delaunay = Delaunay(top_positions_2d)
            except (QhullError, ValueError, RuntimeError):
                primary_delaunay = None

        exp2d, expanded_origin_local_index, exp3d = _expand_top_layer_ab_images(
            top_positions_2d,
            cell=cell,
            pbc=pbc,
            top_positions_3d=top_positions,
        )
        if exp3d is None:
            raise ValueError("3D image expansion failed for top-layer sites")
        exp_tri = None
        if len(exp2d) >= 3:
            try:
                exp_tri = Delaunay(exp2d)
            except (QhullError, ValueError, RuntimeError):
                exp_tri = None
    else:
        # Reuse Qhull; only rebuild image expansion for 3D lift coordinates.
        if exp2d is None or expanded_origin_local_index is None:
            raise ValueError("reuse_delaunay requires exp2d and origin indices")
        _, _, exp3d = _expand_top_layer_ab_images(
            top_positions_2d,
            cell=cell,
            pbc=pbc,
            top_positions_3d=top_positions,
        )
        if exp3d is None:
            raise ValueError("3D image expansion failed for top-layer sites")

    if exp_tri is not None:
        bridge_points: list[np.ndarray] = []
        bridge_atoms: list[tuple[int, ...]] = []
        hollow_points: list[np.ndarray] = []
        hollow_atoms: list[tuple[int, ...]] = []
        if expanded_origin_local_index is None:
            raise ValueError("expanded origin indices required for bridge/hollow")
        for kind, ids, pt in _iter_unique_simplex_sites(
            exp_tri.simplices, expanded_origin_local_index, exp3d
        ):
            # Catalog = simplex point on the support plane; lift is probe-only.
            global_ids = tuple(int(top_atom_indices[i]) for i in ids)
            if kind == "bridge":
                bridge_points.append(np.asarray(pt, dtype=float))
                bridge_atoms.append(global_ids)
            else:
                hollow_points.append(np.asarray(pt, dtype=float))
                hollow_atoms.append(global_ids)
        if bridge_points:
            bridge_arr = np.asarray(bridge_points, dtype=float)
            _add_candidates_batch(
                bridge_arr,
                "topology_bridge",
                bridge_atoms,
                probe_points=bridge_arr + float(site_height) * n_hat,
            )
        if hollow_points:
            hollow_arr = np.asarray(hollow_points, dtype=float)
            _add_candidates_batch(
                hollow_arr,
                "topology_hollow",
                hollow_atoms,
                probe_points=hollow_arr + float(site_height) * n_hat,
            )

    return _finalize()


# ---------------------------------------------------------------------------
# Ridge-based geodesic enrichment
# ---------------------------------------------------------------------------


def _enrich_edge_candidates(
    edge_chunk: Sequence[tuple[int, int]],
    *,
    vertices: np.ndarray,
    support_sets: list[set[int]],
    target_spacing: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    inv_cell: np.ndarray | None,
) -> list[np.ndarray]:
    """Subdivide a chunk of Voronoi edges into enrichment sample points."""
    candidate_pts: list[np.ndarray] = []
    for k0, k1 in edge_chunk:
        if not support_sets[k0] & support_sets[k1]:
            continue

        v0, v1 = vertices[k0], vertices[k1]
        if np.any(pbc):
            f0 = _cart_to_frac(v0.reshape(1, 3), cell, inv_cell=inv_cell)[0]
            f1 = _cart_to_frac(v1.reshape(1, 3), cell, inv_cell=inv_cell)[0]
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
                candidate = _wrap_cartesian(
                    candidate.reshape(1, 3), cell, pbc, inv_cell=inv_cell
                )[0]
            candidate_pts.append(candidate)
    return candidate_pts


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
    n_origin: int | None = None,
    inv_cell: np.ndarray | None = None,
    dedup_offsets: list[np.ndarray] | None = None,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide long admissible Voronoi edges and re-check accessibility."""
    n_kept = len(vertices)
    if n_kept < 2:
        return vertices, nn_dists

    n_ext = len(extended_positions)
    if n_origin is None:
        n_origin = n_ext

    k_support = min(_SITE_CLASSIFICATION_NEIGHBOURS, n_ext)
    _, support_indices = framework_tree.query(vertices, k=k_support)
    if np.ndim(support_indices) == 1:
        support_indices = np.asarray(support_indices).reshape(-1, 1)
    support_sets = [
        {int(j) % n_origin for j in row} for row in np.asarray(support_indices)
    ]

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

    if np.any(pbc) and inv_cell is None:
        inv_cell = np.linalg.inv(cell)

    edge_list = sorted(edges)
    n_workers = resolve_materialize_workers(n_jobs, n_tasks=len(edge_list))
    if n_workers <= 1 or len(edge_list) < 32:
        candidate_pts = _enrich_edge_candidates(
            edge_list,
            vertices=vertices,
            support_sets=support_sets,
            target_spacing=target_spacing,
            cell=cell,
            pbc=pbc,
            inv_cell=inv_cell,
        )
    else:
        chunk = max(1, (len(edge_list) + n_workers - 1) // n_workers)
        chunks = [edge_list[i : i + chunk] for i in range(0, len(edge_list), chunk)]
        candidate_pts = []
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [
                pool.submit(
                    _enrich_edge_candidates,
                    ch,
                    vertices=vertices,
                    support_sets=support_sets,
                    target_spacing=target_spacing,
                    cell=cell,
                    pbc=pbc,
                    inv_cell=inv_cell,
                )
                for ch in chunks
            ]
            for fut in futures:
                candidate_pts.extend(fut.result())

    if not candidate_pts:
        return vertices, nn_dists

    cand_arr = np.asarray(candidate_pts, dtype=float)
    d_nn_all = np.asarray(framework_tree.query(cand_arr, k=1)[0], dtype=float).ravel()
    keep_acc = (d_nn_all >= probe_radius) & (d_nn_all <= max_distance)
    if not np.any(keep_acc):
        return vertices, nn_dists

    new_verts = cand_arr[keep_acc]
    new_dists = d_nn_all[keep_acc]

    all_verts = np.vstack([vertices, new_verts])
    all_dists = np.concatenate([nn_dists, new_dists])
    keep = _deduplicate_points(
        all_verts,
        _VORONOI_DEDUP_TOLERANCE,
        cell=cell,
        pbc=pbc,
        image_offsets=dedup_offsets,
    )
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
        return "pore", ()
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


def _build_delaunay_classification_index(
    top_positions_2d: np.ndarray,
    top_atom_indices: np.ndarray,
    triangulation: Delaunay | None = None,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
    expanded_xy: np.ndarray | None = None,
    expanded_origin: list[int] | None = None,
    expanded_tri: Delaunay | None = None,
) -> tuple[np.ndarray, list[str], list[tuple[int, ...]]]:
    """Precompute (atop, bridge midpoint, hollow centroid) XY candidates for KDTree classify.

    When *cell* and *pbc* are provided and a or b is periodic, the whole index is
    built from the ±1 a/b expanded top-layer point set: atop, bridge **and**
    hollow candidates are emitted for every simplex of the expanded
    triangulation. This is what makes cross-boundary bridges and hollows
    classifiable at all — the primary-cell triangulation has no simplices
    spanning the periodic cut, so sites there would otherwise fall back to the
    nearest interior candidate (usually an atop).

    When *expanded_xy* / *expanded_origin* / *expanded_tri* are provided (from
    topology generation), the expensive expansion + Delaunay rebuild is skipped.

    Candidate atom indices are always mapped back to primary-cell indices
    through the image origin, so the returned ``cand_indices`` reference
    ``top_atom_indices`` entries regardless of which image produced them.
    """
    top_xy = np.asarray(top_positions_2d, dtype=float)
    top_atom_indices = np.asarray(top_atom_indices, dtype=int)
    cand_xy: list[np.ndarray] = []
    cand_types: list[str] = []
    cand_indices: list[tuple[int, ...]] = []

    work_xy = top_xy
    origin_local = list(range(len(top_xy)))
    work_tri = triangulation

    use_pbc = cell is not None and pbc is not None and (bool(pbc[0]) or bool(pbc[1]))
    if use_pbc:
        if (
            expanded_xy is not None
            and expanded_origin is not None
            and expanded_tri is not None
        ):
            work_xy = np.asarray(expanded_xy, dtype=float)
            origin_local = list(expanded_origin)
            work_tri = expanded_tri
        else:
            exp_xy, exp_origin, _ = _expand_top_layer_ab_images(
                top_xy, cell=cell, pbc=pbc
            )
            exp_tri: Delaunay | None = None
            if len(exp_xy) >= 3:
                try:
                    exp_tri = Delaunay(exp_xy)
                except (QhullError, ValueError, RuntimeError):
                    exp_tri = None
            if exp_tri is not None:
                work_xy = exp_xy
                origin_local = exp_origin
                work_tri = exp_tri

    for wi in range(len(work_xy)):
        cand_xy.append(work_xy[wi])
        cand_types.append("atop")
        cand_indices.append((int(top_atom_indices[origin_local[wi]]),))

    if work_tri is not None:
        # Shared candidate extraction with the topology generator (same dedup
        # keys and ordering); 2D points need no height offset.
        for kind, ids, pt in _iter_unique_simplex_sites(
            work_tri.simplices, origin_local, work_xy
        ):
            cand_xy.append(pt)
            cand_types.append(kind)
            cand_indices.append(tuple(int(top_atom_indices[i]) for i in ids))

    if not cand_xy:
        return np.empty((0, 2), dtype=float), [], []
    return np.asarray(cand_xy, dtype=float), cand_types, cand_indices

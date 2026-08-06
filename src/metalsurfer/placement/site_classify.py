"""Local surface normals and site record construction."""


import numpy as np
from scipy.spatial import Delaunay, KDTree

from ._constants import (
    _NORMAL_K_NEIGHBOURS,
    _PBC_DELAUNAY_BOUNDARY_FRAC,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
)
from .site_coords import _cart_to_frac, _project_to_slab_plane, _wrap_fractional
from .site_types import Site
from .site_voronoi import (
    _build_delaunay_classification_index,
    _classify_voronoi_site_from_neighbors,
    _delaunay_site_classification,
    _hollow_coordination_order,
)


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
    delaunay_class_index: (
        tuple[np.ndarray, list[str], list[tuple[int, ...]]] | None
    ) = None,
    delaunay_class_index_pbc: (
        tuple[np.ndarray, list[str], list[tuple[int, ...]]] | None
    ) = None,
    pbc: np.ndarray | None = None,
) -> list[Site]:
    sites: list[Site] = []
    n_verts = len(vertices)
    vertex_2d = _project_to_slab_plane(vertices, cell) if n_verts else np.empty((0, 2))

    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
    k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
    class_dists: np.ndarray | None = None
    class_idx: np.ndarray | None = None
    normals: np.ndarray | None = None
    delaunay_char_len: float | None = None

    has_delaunay = use_delaunay and (
        delaunay_class_index is not None
        or (
            delaunay_tri is not None
            and top_positions_2d is not None
            and top_atom_indices is not None
        )
    )

    if n_verts > 0:
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)

    if n_verts > 0 and not has_delaunay:
        dists_raw, idx_raw = local_tree.query(vertices, k=k_class)
        class_dists = np.asarray(dists_raw, dtype=float)
        class_idx = np.asarray(idx_raw, dtype=int)
        if class_dists.ndim == 1:
            class_dists = class_dists.reshape(-1, 1)
            class_idx = class_idx.reshape(-1, 1)
    elif n_verts > 0 and has_delaunay:
        if top_positions_2d is not None and len(top_positions_2d) >= 2:
            _top_tree = KDTree(top_positions_2d)
            _nn_d, _ = _top_tree.query(top_positions_2d, k=2)
            delaunay_char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))

    delaunay_cand_xy: np.ndarray | None = None
    delaunay_cand_types: list[str] | None = None
    delaunay_cand_indices: list[tuple[int, ...]] | None = None
    delaunay_cand_tree: KDTree | None = None
    delaunay_cand_xy_pbc: np.ndarray | None = None
    delaunay_cand_types_pbc: list[str] | None = None
    delaunay_cand_indices_pbc: list[tuple[int, ...]] | None = None
    delaunay_cand_tree_pbc: KDTree | None = None

    # Callers should pass material-aware PBC; default is non-periodic (safe).
    pbc_arr = (
        np.asarray(pbc, dtype=bool)
        if pbc is not None
        else np.array([False, False, False], dtype=bool)
    )

    if has_delaunay and n_verts > 0:
        if delaunay_class_index is not None:
            (
                delaunay_cand_xy,
                delaunay_cand_types,
                delaunay_cand_indices,
            ) = delaunay_class_index
        elif (
            delaunay_tri is not None
            and top_positions_2d is not None
            and top_atom_indices is not None
        ):
            (
                delaunay_cand_xy,
                delaunay_cand_types,
                delaunay_cand_indices,
            ) = _build_delaunay_classification_index(
                top_positions_2d,
                top_atom_indices,
                delaunay_tri,
            )
        else:
            raise ValueError(
                "Delaunay classification requested but neither delaunay_class_index "
                "nor (delaunay_tri, top_positions_2d, top_atom_indices) were provided"
            )
        if delaunay_cand_xy is not None and len(delaunay_cand_xy) > 0:
            delaunay_cand_tree = KDTree(delaunay_cand_xy)

        if delaunay_class_index_pbc is not None:
            (
                delaunay_cand_xy_pbc,
                delaunay_cand_types_pbc,
                delaunay_cand_indices_pbc,
            ) = delaunay_class_index_pbc
        elif (
            delaunay_tri is not None
            and top_positions_2d is not None
            and top_atom_indices is not None
            and (bool(pbc_arr[0]) or bool(pbc_arr[1]))
        ):
            (
                delaunay_cand_xy_pbc,
                delaunay_cand_types_pbc,
                delaunay_cand_indices_pbc,
            ) = _build_delaunay_classification_index(
                top_positions_2d,
                top_atom_indices,
                delaunay_tri,
                cell=cell,
                pbc=pbc_arr,
            )
        if delaunay_cand_xy_pbc is not None and len(delaunay_cand_xy_pbc) > 0:
            delaunay_cand_tree_pbc = KDTree(delaunay_cand_xy_pbc)

    vertex_frac = (
        _wrap_fractional(_cart_to_frac(vertices, cell), pbc_arr)
        if n_verts and has_delaunay
        else np.empty((0, 3))
    )

    for i, vertex in enumerate(vertices):
        if (
            has_delaunay
            and delaunay_cand_xy is not None
            and delaunay_cand_types is not None
            and delaunay_cand_indices is not None
            and top_positions_2d is not None
            and top_atom_indices is not None
        ):
            if delaunay_tri is None and delaunay_cand_tree is None:
                raise ValueError(
                    "Delaunay classification requires delaunay_tri or a prebuilt "
                    "candidate index with cand_tree"
                )
            # Prefer candidate NN path; triangulation is only a rebuild fallback.
            tri_for_classify = delaunay_tri
            if tri_for_classify is None:
                # Minimal placeholder when only prebuilt candidates are supplied.
                tri_for_classify = Delaunay(top_positions_2d)
            site_type, nearest_idx = _delaunay_site_classification(
                vertex,
                top_positions_2d,
                top_atom_indices,
                tri_for_classify,
                positions,
                vertex_2d=vertex_2d[i],
                char_len=delaunay_char_len,
                positions_tree=local_tree,
                cand_xy=delaunay_cand_xy,
                cand_types=delaunay_cand_types,
                cand_indices=delaunay_cand_indices,
                cand_tree=delaunay_cand_tree,
            )
            # Primary-cell Delaunay misses bridges across a/b. Upgrade near-boundary
            # atops using PBC-expanded candidates without touching interior hollows.
            if (
                site_type == "atop"
                and delaunay_cand_xy_pbc is not None
                and delaunay_cand_types_pbc is not None
                and delaunay_cand_indices_pbc is not None
            ):
                frac = vertex_frac[i]
                near_boundary = bool(
                    (
                        bool(pbc_arr[0])
                        and (
                            frac[0] < _PBC_DELAUNAY_BOUNDARY_FRAC
                            or frac[0] > 1.0 - _PBC_DELAUNAY_BOUNDARY_FRAC
                        )
                    )
                    or (
                        bool(pbc_arr[1])
                        and (
                            frac[1] < _PBC_DELAUNAY_BOUNDARY_FRAC
                            or frac[1] > 1.0 - _PBC_DELAUNAY_BOUNDARY_FRAC
                        )
                    )
                )
                if near_boundary:
                    site_type_pbc, nearest_idx_pbc = _delaunay_site_classification(
                        vertex,
                        top_positions_2d,
                        top_atom_indices,
                        tri_for_classify,
                        positions,
                        vertex_2d=vertex_2d[i],
                        char_len=delaunay_char_len,
                        positions_tree=local_tree,
                        cand_xy=delaunay_cand_xy_pbc,
                        cand_types=delaunay_cand_types_pbc,
                        cand_indices=delaunay_cand_indices_pbc,
                        cand_tree=delaunay_cand_tree_pbc,
                    )
                    if site_type_pbc == "bridge":
                        site_type = site_type_pbc
                        nearest_idx = nearest_idx_pbc
        elif class_dists is not None and class_idx is not None:
            site_type, nearest_idx = _classify_voronoi_site_from_neighbors(
                class_dists[i],
                class_idx[i],
                pore_threshold=pore_threshold,
            )
        else:
            raise ValueError(
                "Incomplete classification inputs: provide Voronoi neighbour "
                "arrays, or a complete Delaunay candidate set"
            )

        env_fingerprint = (
            tuple(sorted(symbols[j] for j in nearest_idx if j < len(symbols))),
            site_type,
        )
        hollow_order: int | None = None
        if site_type == "hollow":
            if nearest_idx:
                hollow_order = len(nearest_idx)
            elif class_dists is not None:
                hollow_order = _hollow_coordination_order(class_dists[i])
        assert normals is not None
        normal = normals[i]
        sites.append(
            Site(
                xyz=vertex.copy(),
                normal=normal,
                site_type=site_type,
                slab_indices=tuple(int(j) for j in nearest_idx),
                material_type=material_type,
                site_source=(
                    source_hints[i]
                    if source_hints is not None and i < len(source_hints)
                    else "voronoi"
                ),
                env_fingerprint=env_fingerprint,
                nn_distance=float(nn_dists[i]) if i < len(nn_dists) else None,
                hollow_order=hollow_order,
            )
        )
    return sites

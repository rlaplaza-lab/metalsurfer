"""Local surface normals and site record construction."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.spatial import KDTree

from .._utils import cell_has_volume
from ._constants import (
    _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM,
    _NORMAL_K_NEIGHBOURS,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
)
from .site_coords import (
    _build_periodic_images,
    _project_to_slab_plane,
    _slab_normal,
)
from .site_types import Site
from .site_voronoi import (
    _classify_voronoi_site_from_neighbors,
)


class _DelaunayClassifyInputs(NamedTuple):
    """Prebuilt Delaunay classification inputs. All fields are required.

    ``class_index`` is a single atop/bridge/hollow candidate index. On periodic
    slabs it is built from the ±1 a/b expanded top layer, so cross-boundary
    bridges and hollows are represented directly and no separate PBC "upgrade"
    pass is needed.
    """

    top_positions_2d: np.ndarray
    top_atom_indices: np.ndarray
    class_index: tuple[np.ndarray, list[str], list[tuple[int, ...]]]


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
    small = norms < _SURFACE_NORMAL_FALLBACK_NORM_EPS
    out = np.empty_like(vecs)
    out[small] = fallback
    out[~small] = vecs[~small] / norms[~small, np.newaxis]
    return out


def _slab_site_normals(n_verts: int, cell: np.ndarray) -> np.ndarray:
    """Exact surface normals for slab sites.

    Every slab site shares the a×b surface normal. This is the same ``n_hat``
    that :func:`_generate_slab_topology_sites` offsets candidates along, so the
    site positions and their normals are consistent by construction. A k-nearest
    centroid estimate is *not* used here: with only 3-4 neighbours it tilts by
    tens of degrees on bridge/hollow sites and near cell boundaries.
    """
    if n_verts == 0:
        return np.empty((0, 3), dtype=float)
    return np.tile(_slab_normal(cell).reshape(1, 3), (n_verts, 1))


def _periodic_local_normals(
    vertices: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    k: int,
) -> np.ndarray:
    """Local normals from a k-nearest centroid taken over periodic images.

    The centroid is computed on *image* coordinates rather than wrapped primary
    positions, so 3D-periodic frameworks stop tilting at cell faces.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=float)
    d_knn, _ = local_tree.query(vertices, k=k)
    d_arr = np.asarray(d_knn, dtype=float)
    if d_arr.ndim == 1:
        d_arr = d_arr.reshape(-1, 1)
    # The non-periodic k-th neighbour distance upper-bounds the periodic one,
    # so it is a safe image margin.
    margin = float(np.max(d_arr[:, -1])) if d_arr.size else 0.0
    images = _build_periodic_images(positions, cell, pbc, margin=margin)
    image_tree = KDTree(images)
    _, idx = image_tree.query(vertices, k=min(k, len(images)))
    idx_arr = np.asarray(idx, dtype=int)
    if idx_arr.ndim == 1:
        idx_arr = idx_arr.reshape(-1, 1)
    return _compute_local_normals_batch(vertices, images, idx_arr)


def _site_normals_for_material(
    vertices: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
    *,
    material_type: str,
    cell: np.ndarray,
    pbc: np.ndarray,
    k: int,
) -> np.ndarray:
    """Per-material-type surface normal dispatch."""
    n_verts = len(vertices)
    if n_verts == 0:
        return np.empty((0, 3), dtype=float)
    if material_type == "slab":
        return _slab_site_normals(n_verts, cell)
    if material_type == "porous" and bool(np.any(pbc)) and cell_has_volume(cell):
        return _periodic_local_normals(
            vertices, positions, local_tree, cell=cell, pbc=pbc, k=k
        )
    # Nanoparticles (and degenerate/non-periodic frameworks): the plain
    # non-periodic k-nearest centroid is the correct outward estimate.
    _, norm_idx = local_tree.query(vertices, k=k)
    norm_idx_arr = np.asarray(norm_idx, dtype=int)
    if norm_idx_arr.ndim == 1:
        norm_idx_arr = norm_idx_arr.reshape(-1, 1)
    return _compute_local_normals_batch(vertices, positions, norm_idx_arr)


@dataclass(frozen=True)
class _ClassificationContext:
    vertex_2d: np.ndarray
    normals: np.ndarray
    pbc: np.ndarray
    class_dists: np.ndarray | None  # Voronoi path
    class_idx: np.ndarray | None  # Voronoi path
    delaunay: _DelaunayClassifyInputs | None
    char_len: float | None
    cand_tree: KDTree | None


def _build_classification_context(
    vertices: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
    *,
    material_type: str,
    cell: np.ndarray,
    pbc: np.ndarray,
    delaunay: _DelaunayClassifyInputs | None,
) -> _ClassificationContext:
    n_verts = len(vertices)
    vertex_2d = _project_to_slab_plane(vertices, cell) if n_verts else np.empty((0, 2))

    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
    k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))

    pbc_arr = np.asarray(pbc, dtype=bool)

    normals = _site_normals_for_material(
        vertices,
        positions,
        local_tree,
        material_type=material_type,
        cell=cell,
        pbc=pbc_arr,
        k=k_norm,
    )

    if n_verts > 0 and delaunay is None:
        # MIC neighbours when PBC is on (porous / slab distance_ratio).
        use_periodic = bool(np.any(pbc_arr)) and cell_has_volume(cell)
        if use_periodic:
            d_knn, _ = local_tree.query(vertices, k=k_class)
            d_arr = np.asarray(d_knn, dtype=float)
            if d_arr.ndim == 1:
                d_arr = d_arr.reshape(-1, 1)
            margin = float(np.max(d_arr[:, -1])) if d_arr.size else 0.0
            images = _build_periodic_images(positions, cell, pbc_arr, margin=margin)
            image_tree = KDTree(images)
            dists_raw, idx_raw = image_tree.query(vertices, k=min(k_class, len(images)))
            class_dists = np.asarray(dists_raw, dtype=float)
            class_idx = np.asarray(idx_raw, dtype=int) % len(positions)
        else:
            dists_raw, idx_raw = local_tree.query(vertices, k=k_class)
            class_dists = np.asarray(dists_raw, dtype=float)
            class_idx = np.asarray(idx_raw, dtype=int)
        if class_dists.ndim == 1:
            class_dists = class_dists.reshape(-1, 1)
            class_idx = class_idx.reshape(-1, 1)
    else:
        class_dists = None
        class_idx = None

    char_len: float | None = None
    cand_tree: KDTree | None = None
    if n_verts > 0 and delaunay is not None:
        top_positions_2d = delaunay.top_positions_2d
        if len(top_positions_2d) >= 2:
            _top_tree = KDTree(top_positions_2d)
            _nn_d, _ = _top_tree.query(top_positions_2d, k=2)
            char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))
        cand_xy, _cand_types, _cand_indices = delaunay.class_index
        cand_tree = KDTree(cand_xy) if len(cand_xy) > 0 else None

    return _ClassificationContext(
        vertex_2d=vertex_2d,
        normals=normals,
        pbc=pbc_arr,
        class_dists=class_dists,
        class_idx=class_idx,
        delaunay=delaunay,
        char_len=char_len,
        cand_tree=cand_tree,
    )


def _classify_delaunay_vertices_batch(
    ctx: _ClassificationContext,
    vertices: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
) -> list[tuple[str, tuple[int, ...]]]:
    """Classify all vertices with one ``(M, 2)`` cand_tree query."""
    delaunay = ctx.delaunay
    if delaunay is None:
        raise ValueError("ctx.delaunay must be set for Delaunay classification")
    n = len(vertices)
    if n == 0:
        return []
    cand_xy, cand_types, cand_indices = delaunay.class_index
    cand_tree = ctx.cand_tree
    if cand_tree is None or len(cand_xy) == 0:
        fallback = tuple(int(i) for i in np.asarray(delaunay.top_atom_indices)[:3])
        return [("hollow", fallback) for _ in range(n)]

    char_len = (
        float(ctx.char_len)
        if ctx.char_len is not None
        else _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM
    )
    bridge_cut = _DELAUNAY_BRIDGE_THRESHOLD_FRACTION * char_len

    dists, idxs = cand_tree.query(np.asarray(ctx.vertex_2d, dtype=float), k=1)
    dists = np.asarray(dists, dtype=float).ravel()
    idxs = np.asarray(idxs, dtype=int).ravel()

    site_types: list[str] = []
    site_indices: list[tuple[int, ...]] = []
    fallback_i: list[int] = []
    for i in range(n):
        nearest = int(idxs[i])
        best_type = cand_types[nearest]
        best_dist = float(dists[i])
        best_indices = cand_indices[nearest]
        site_types.append(best_type)
        site_indices.append(best_indices)
        if best_type == "bridge" and best_dist > bridge_cut:
            fallback_i.append(i)

    if fallback_i:
        k3 = min(3, len(positions))
        fb_verts = np.asarray(vertices[fallback_i], dtype=float)
        _, idx3 = local_tree.query(fb_verts, k=k3)
        idx3 = np.asarray(idx3, dtype=int)
        # k=1 yields a 1-D index vector; reshape to (n_fb, k).
        if idx3.ndim == 1:
            idx3 = idx3.reshape(-1, 1)
        for row, vi in enumerate(fallback_i):
            site_types[vi] = "hollow"
            site_indices[vi] = tuple(int(j) for j in idx3[row, :3])

    return list(zip(site_types, site_indices, strict=True))


def _classify_vertices(
    ctx: _ClassificationContext,
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    positions: np.ndarray,
    symbols: list[str],
    local_tree: KDTree,
    material_type: str,
    pore_threshold: float,
    source_hints: list[str] | None,
) -> list[Site]:
    if ctx.delaunay is not None:
        classifications = _classify_delaunay_vertices_batch(
            ctx, vertices, positions, local_tree
        )
    else:
        if ctx.class_dists is None or ctx.class_idx is None:
            raise ValueError(
                "ctx.class_dists and ctx.class_idx must be set for Voronoi classification"
            )
        classifications = [
            _classify_voronoi_site_from_neighbors(
                ctx.class_dists[i],
                ctx.class_idx[i],
                pore_threshold=pore_threshold,
            )
            for i in range(len(vertices))
        ]

    sites: list[Site] = []
    for i, (site_type, nearest_idx) in enumerate(classifications):
        env_fingerprint = (
            tuple(sorted(symbols[j] for j in nearest_idx if j < len(symbols))),
            site_type,
        )
        hollow_order: int | None = None
        if site_type == "hollow" and nearest_idx:
            hollow_order = len(nearest_idx)
        normal = ctx.normals[i]
        sites.append(
            Site(
                xyz=vertices[i].copy(),
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
                nn_distance=float(nn_dists[i]),
                hollow_order=hollow_order,
            )
        )
    return sites


def _build_site_records(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    positions: np.ndarray,
    symbols: list[str],
    local_tree: KDTree,
    material_type: str,
    pore_threshold: float,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    source_hints: list[str] | None = None,
    delaunay: _DelaunayClassifyInputs | None = None,
) -> list[Site]:
    ctx = _build_classification_context(
        vertices,
        positions,
        local_tree,
        material_type=material_type,
        cell=cell,
        pbc=pbc,
        delaunay=delaunay,
    )
    return _classify_vertices(
        ctx,
        vertices,
        nn_dists,
        positions,
        symbols,
        local_tree,
        material_type,
        pore_threshold,
        source_hints,
    )

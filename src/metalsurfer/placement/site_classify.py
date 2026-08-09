"""Local surface normals and site record construction."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from scipy.spatial import Delaunay, KDTree

from .._utils import cell_has_volume
from ._constants import (
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
    _delaunay_site_classification,
    _hollow_coordination_order,
)


class _DelaunayClassifyInputs(NamedTuple):
    """Prebuilt Delaunay classification inputs. All fields are required.

    ``class_index`` is a single atop/bridge/hollow candidate index. On periodic
    slabs it is built from the ±1 a/b expanded top layer, so cross-boundary
    bridges and hollows are represented directly and no separate PBC "upgrade"
    pass is needed.
    """

    tri: Delaunay
    top_positions_2d: np.ndarray
    top_atom_indices: np.ndarray
    class_index: tuple[np.ndarray, list[str], list[tuple[int, ...]]]


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
    pbc: np.ndarray | None,
    delaunay: _DelaunayClassifyInputs | None,
) -> _ClassificationContext:
    n_verts = len(vertices)
    vertex_2d = _project_to_slab_plane(vertices, cell) if n_verts else np.empty((0, 2))

    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
    k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))

    # Callers should pass material-aware PBC; default is non-periodic (safe).
    pbc_arr = (
        np.asarray(pbc, dtype=bool)
        if pbc is not None
        else np.array([False, False, False], dtype=bool)
    )

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


def _classify_delaunay_vertex(
    ctx: _ClassificationContext,
    i: int,
    vertex: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
) -> tuple[str, tuple[int, ...]]:
    delaunay = ctx.delaunay
    if delaunay is None:
        raise ValueError("ctx.delaunay must be set for Delaunay classification")
    cand_xy, cand_types, cand_indices = delaunay.class_index

    return _delaunay_site_classification(
        vertex,
        delaunay.top_positions_2d,
        delaunay.top_atom_indices,
        delaunay.tri,
        positions,
        vertex_2d=ctx.vertex_2d[i],
        char_len=ctx.char_len,
        positions_tree=local_tree,
        cand_xy=cand_xy,
        cand_types=cand_types,
        cand_indices=cand_indices,
        cand_tree=ctx.cand_tree,
    )


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
    sites: list[Site] = []
    for i, vertex in enumerate(vertices):
        if ctx.delaunay is not None:
            site_type, nearest_idx = _classify_delaunay_vertex(
                ctx, i, vertex, positions, local_tree
            )
        else:
            if ctx.class_dists is None or ctx.class_idx is None:
                raise ValueError(
                    "ctx.class_dists and ctx.class_idx must be set for Voronoi classification"
                )
            site_type, nearest_idx = _classify_voronoi_site_from_neighbors(
                ctx.class_dists[i],
                ctx.class_idx[i],
                pore_threshold=pore_threshold,
            )

        env_fingerprint = (
            tuple(sorted(symbols[j] for j in nearest_idx if j < len(symbols))),
            site_type,
        )
        hollow_order: int | None = None
        if site_type == "hollow":
            if nearest_idx:
                hollow_order = len(nearest_idx)
            elif ctx.class_dists is not None:
                hollow_order = _hollow_coordination_order(ctx.class_dists[i])
        normal = ctx.normals[i]
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

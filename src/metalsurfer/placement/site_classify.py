"""Local surface normals and site record construction."""

from dataclasses import dataclass
from typing import NamedTuple

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
    _classify_voronoi_site_from_neighbors,
    _delaunay_site_classification,
    _hollow_coordination_order,
)


class _DelaunayClassifyInputs(NamedTuple):
    """Prebuilt Delaunay classification inputs. All fields are required."""

    tri: Delaunay
    top_positions_2d: np.ndarray
    top_atom_indices: np.ndarray
    class_index: tuple[np.ndarray, list[str], list[tuple[int, ...]]]
    class_index_pbc: tuple[np.ndarray, list[str], list[tuple[int, ...]]] | None = None


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


@dataclass(frozen=True)
class _ClassificationContext:
    vertex_2d: np.ndarray
    normals: np.ndarray
    pbc: np.ndarray
    vertex_frac: np.ndarray
    class_dists: np.ndarray | None  # Voronoi path
    class_idx: np.ndarray | None  # Voronoi path
    delaunay: _DelaunayClassifyInputs | None
    char_len: float | None
    cand_tree: KDTree | None
    cand_tree_pbc: KDTree | None


def _build_classification_context(
    vertices: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
    *,
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

    if n_verts > 0:
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)
    else:
        normals = np.empty((0, 3))

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
    cand_tree_pbc: KDTree | None = None
    if n_verts > 0 and delaunay is not None:
        top_positions_2d = delaunay.top_positions_2d
        if len(top_positions_2d) >= 2:
            _top_tree = KDTree(top_positions_2d)
            _nn_d, _ = _top_tree.query(top_positions_2d, k=2)
            char_len = float(np.mean(np.asarray(_nn_d, dtype=float)[:, 1]))
        cand_xy, _cand_types, _cand_indices = delaunay.class_index
        cand_tree = KDTree(cand_xy) if len(cand_xy) > 0 else None
        if delaunay.class_index_pbc is not None:
            cand_xy_pbc, _cand_types_pbc, _cand_indices_pbc = delaunay.class_index_pbc
            cand_tree_pbc = KDTree(cand_xy_pbc) if len(cand_xy_pbc) > 0 else None

    vertex_frac = (
        _wrap_fractional(_cart_to_frac(vertices, cell), pbc_arr)
        if n_verts and delaunay is not None
        else np.empty((0, 3))
    )

    return _ClassificationContext(
        vertex_2d=vertex_2d,
        normals=normals,
        pbc=pbc_arr,
        vertex_frac=vertex_frac,
        class_dists=class_dists,
        class_idx=class_idx,
        delaunay=delaunay,
        char_len=char_len,
        cand_tree=cand_tree,
        cand_tree_pbc=cand_tree_pbc,
    )


def _near_ab_boundary(frac: np.ndarray, pbc: np.ndarray) -> bool:
    """True when *frac* lies near a periodic a/b boundary."""
    return bool(
        (
            bool(pbc[0])
            and (
                frac[0] < _PBC_DELAUNAY_BOUNDARY_FRAC
                or frac[0] > 1.0 - _PBC_DELAUNAY_BOUNDARY_FRAC
            )
        )
        or (
            bool(pbc[1])
            and (
                frac[1] < _PBC_DELAUNAY_BOUNDARY_FRAC
                or frac[1] > 1.0 - _PBC_DELAUNAY_BOUNDARY_FRAC
            )
        )
    )


def _classify_delaunay_vertex(
    ctx: _ClassificationContext,
    i: int,
    vertex: np.ndarray,
    positions: np.ndarray,
    local_tree: KDTree,
) -> tuple[str, tuple[int, ...]]:
    delaunay = ctx.delaunay
    assert delaunay is not None
    tri = delaunay.tri
    top_positions_2d = delaunay.top_positions_2d
    top_atom_indices = delaunay.top_atom_indices
    cand_xy, cand_types, cand_indices = delaunay.class_index

    site_type, nearest_idx = _delaunay_site_classification(
        vertex,
        top_positions_2d,
        top_atom_indices,
        tri,
        positions,
        vertex_2d=ctx.vertex_2d[i],
        char_len=ctx.char_len,
        positions_tree=local_tree,
        cand_xy=cand_xy,
        cand_types=cand_types,
        cand_indices=cand_indices,
        cand_tree=ctx.cand_tree,
    )

    # Primary-cell Delaunay misses bridges across a/b. Upgrade near-boundary
    # atops using PBC-expanded candidates without touching interior hollows.
    if site_type == "atop" and delaunay.class_index_pbc is not None:
        cand_xy_pbc, cand_types_pbc, cand_indices_pbc = delaunay.class_index_pbc
        if _near_ab_boundary(ctx.vertex_frac[i], ctx.pbc):
            site_type_pbc, nearest_idx_pbc = _delaunay_site_classification(
                vertex,
                top_positions_2d,
                top_atom_indices,
                tri,
                positions,
                vertex_2d=ctx.vertex_2d[i],
                char_len=ctx.char_len,
                positions_tree=local_tree,
                cand_xy=cand_xy_pbc,
                cand_types=cand_types_pbc,
                cand_indices=cand_indices_pbc,
                cand_tree=ctx.cand_tree_pbc,
            )
            if site_type_pbc == "bridge":
                site_type = site_type_pbc
                nearest_idx = nearest_idx_pbc

    return site_type, nearest_idx


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
            assert ctx.class_dists is not None and ctx.class_idx is not None
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
        vertices, positions, local_tree, cell=cell, pbc=pbc, delaunay=delaunay
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

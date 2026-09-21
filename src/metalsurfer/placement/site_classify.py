"""Local surface normals and site record construction."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
from scipy.spatial import KDTree

from .._utils import cell_has_volume
from ._constants import (
    _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    _KD_RADIUS_SEARCH_PADDING,
    _NORMAL_K_NEIGHBOURS,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _SITE_ENV_FP_DIST_BIN,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
)
from .geometry import tangent_basis_from_normal
from .site_coords import (
    _build_periodic_images,
    _minimum_image_cartesian_delta,
    _project_to_slab_plane,
    _slab_normal,
    project_anchor_to_support_plane,
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
    images: np.ndarray | None = None,
    image_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Local normals from a k-nearest centroid taken over periodic images.

    The centroid is computed on *image* coordinates rather than wrapped primary
    positions, so 3D-periodic frameworks stop tilting at cell faces.

    *images* / *image_idx* may be supplied from a shared build (so the normals
    path reuses the same periodic-image KDTree as the classifier). When omitted,
    they are built here via :func:`_build_periodic_images`. Production callers
    (``_build_classification_context``) always supply both when PBC is active.
    """
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=float)
    if images is None or image_idx is None:
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
    else:
        idx_arr = np.asarray(image_idx, dtype=int)
    if idx_arr.ndim == 1:
        idx_arr = idx_arr.reshape(-1, 1)
    # A shared build may have queried a larger k; slice to the requested k.
    if idx_arr.shape[1] > k:
        idx_arr = idx_arr[:, :k]
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
    images: np.ndarray | None = None,
    image_idx: np.ndarray | None = None,
) -> np.ndarray:
    """Per-material-type surface normal dispatch."""
    n_verts = len(vertices)
    if n_verts == 0:
        return np.empty((0, 3), dtype=float)
    if material_type == "slab":
        return _slab_site_normals(n_verts, cell)
    if material_type == "porous" and bool(np.any(pbc)) and cell_has_volume(cell):
        return _periodic_local_normals(
            vertices,
            positions,
            local_tree,
            cell=cell,
            pbc=pbc,
            k=k,
            images=images,
            image_idx=image_idx,
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
    k_max = max(k_norm, k_class)

    pbc_arr = np.asarray(pbc, dtype=bool)
    use_periodic = n_verts > 0 and bool(np.any(pbc_arr)) and cell_has_volume(cell)
    # Slab+Delaunay ignores images for normals/classification; skip the build.
    need_periodic_images = use_periodic and not (
        material_type == "slab" and delaunay is not None
    )

    # One periodic image KDTree for normals and the Voronoi classifier
    # (k_class <= k_max so nearest-k over the shared set stays correct).
    images = None
    image_tree = None
    idx_img = None
    dists_img = None
    if need_periodic_images:
        d0, _ = local_tree.query(vertices, k=k_max)
        d0_arr = np.asarray(d0, dtype=float)
        if d0_arr.ndim == 1:
            d0_arr = d0_arr.reshape(-1, 1)
        margin = float(np.max(d0_arr[:, -1])) + _KD_RADIUS_SEARCH_PADDING
        images = _build_periodic_images(positions, cell, pbc_arr, margin=margin)
        image_tree = KDTree(images)
        # Shared query for normals + classifier:
        # so the classifier's slice is a prefix of these results.
        d_img, idx_raw = image_tree.query(vertices, k=min(k_max, len(images)))
        idx_img = np.asarray(idx_raw, dtype=int)
        dists_img = np.asarray(d_img, dtype=float)
        if idx_img.ndim == 1:
            idx_img = idx_img.reshape(-1, 1)
            dists_img = dists_img.reshape(-1, 1)

    normals = _site_normals_for_material(
        vertices,
        positions,
        local_tree,
        material_type=material_type,
        cell=cell,
        pbc=pbc_arr,
        k=k_norm,
        images=images,
        image_idx=idx_img,
    )

    if n_verts > 0 and delaunay is None:
        # MIC neighbours when PBC is on (porous / slab distance_ratio).
        if use_periodic and idx_img is not None and dists_img is not None:
            k_slice = min(k_class, idx_img.shape[1])
            class_dists = dists_img[:, :k_slice]
            class_idx = idx_img[:, :k_slice] % len(positions)
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
    *,
    pore_threshold: float,
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
        # Empty Delaunay candidate index: classify by neighbor distance ratios
        # instead of inventing hollow labels with a shared dummy slab_indices.
        k = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
        dists_raw, idx_raw = local_tree.query(vertices, k=k)
        class_dists = np.asarray(dists_raw, dtype=float)
        class_idx = np.asarray(idx_raw, dtype=int)
        if class_dists.ndim == 1:
            class_dists = class_dists.reshape(-1, 1)
            class_idx = class_idx.reshape(-1, 1)
        return [
            _classify_voronoi_site_from_neighbors(
                class_dists[i],
                class_idx[i],
                pore_threshold=pore_threshold,
            )
            for i in range(n)
        ]

    char_len = (
        float(ctx.char_len)
        if ctx.char_len is not None
        else _SURFACE_COVALENT_RADIUS_FALLBACK
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
        # Reclassify overgrown bridge sites as hollow using the *top-layer*
        # atoms, not the full (bulk-inclusive) ``local_tree``.
        top_idx = np.asarray(delaunay.top_atom_indices)
        if len(top_idx) > 0:
            top_tree = KDTree(positions[top_idx])
            k3 = min(3, len(top_idx))
            fb_verts = np.asarray(vertices[fallback_i], dtype=float)
            _, idx3 = top_tree.query(fb_verts, k=k3)
            idx3 = np.asarray(idx3, dtype=int)
            # k=1 yields a 1-D index vector; reshape to (n_fb, k).
            if idx3.ndim == 1:
                idx3 = idx3.reshape(-1, 1)
            idx3 = top_idx[idx3]
        else:
            k3 = min(3, len(positions))
            fb_verts = np.asarray(vertices[fallback_i], dtype=float)
            _, idx3 = local_tree.query(fb_verts, k=k3)
            idx3 = np.asarray(idx3, dtype=int)
            if idx3.ndim == 1:
                idx3 = idx3.reshape(-1, 1)
        for row, vi in enumerate(fallback_i):
            site_types[vi] = "hollow"
            site_indices[vi] = tuple(int(j) for j in idx3[row, :3])

    return list(zip(site_types, site_indices, strict=True))


_TOPOLOGY_SOURCE_TO_TYPE = {
    "topology_atop": "atop",
    "topology_bridge": "bridge",
    "topology_hollow": "hollow",
    "atop_injected": "atop",
}
# Only these sources may derive site_type from support-atom count.
_SUPPORT_COUNT_TYPED_SOURCES = frozenset({"adaptive_grid", "rolling_probe"})


def site_env_fingerprint(
    support_indices: Sequence[int],
    symbols: Sequence[str],
    support_distances: Sequence[float] | None = None,
    *,
    side_label: int = 0,
    dist_bin: float = _SITE_ENV_FP_DIST_BIN,
) -> tuple[tuple[str, ...], tuple[int, ...], int]:
    """Shared environment fingerprint: chemistry, distance bins, side.

    ``site_type`` is intentionally excluded — identity is the local support
    environment, not the classified label.
    """
    numbers = tuple(
        sorted(
            str(symbols[int(i)]) for i in support_indices if 0 <= int(i) < len(symbols)
        )
    )
    if support_distances is None:
        dist_bins: tuple[int, ...] = ()
    else:
        dist_bins = tuple(
            int(round(float(d) / float(dist_bin)))
            for d in sorted(float(x) for x in support_distances)
        )
    return (numbers, dist_bins, int(side_label))


def _side_label_from_normal(
    normal: np.ndarray,
    *,
    material_type: str,
    cell: np.ndarray,
) -> int:
    if material_type == "slab" and cell_has_volume(cell):
        return 1 if float(np.dot(normal, _slab_normal(cell))) >= 0.0 else -1
    return 0


def _support_mic_distances(
    vertex: np.ndarray,
    positions: np.ndarray,
    support: Sequence[int],
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[float, ...]:
    """Centre-to-centre MIC distances from *vertex* to each support atom."""
    if not support:
        return ()
    vert = np.asarray(vertex, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    use_mic = bool(np.any(pbc_arr)) and cell_has_volume(cell)
    out: list[float] = []
    for j in support:
        delta = np.asarray(positions[int(j)], dtype=float) - vert
        if use_mic:
            delta = _minimum_image_cartesian_delta(delta, cell, pbc_arr)
        out.append(float(np.linalg.norm(delta)))
    return tuple(out)


def _normal_from_support(
    vertex: np.ndarray,
    positions: np.ndarray,
    support: Sequence[int],
    cell: np.ndarray,
    pbc: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    """Return the unit normal from vertex minus MIC-aware support centroid."""
    if not support:
        return np.asarray(fallback, dtype=float)
    vert = np.asarray(vertex, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    use_mic = bool(np.any(pbc_arr)) and cell_has_volume(cell)
    pts = []
    for j in support:
        delta = np.asarray(positions[int(j)], dtype=float) - vert
        if use_mic:
            delta = _minimum_image_cartesian_delta(delta, cell, pbc_arr)
        pts.append(vert + delta)
    centroid = np.mean(np.asarray(pts, dtype=float), axis=0)
    lift = vert - centroid
    nrm = float(np.linalg.norm(lift))
    if nrm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.asarray(fallback, dtype=float)
    return lift / nrm


def _site_type_from_support_count(n_support: int) -> str:
    if n_support <= 1:
        return "atop"
    if n_support == 2:
        return "bridge"
    return "hollow"


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
    atom_indices: list[tuple[int, ...]] | None = None,
    *,
    cell: np.ndarray,
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
) -> list[Site]:
    n = len(vertices)
    hints = list(source_hints) if source_hints is not None else ["voronoi"] * n
    has_plugin_atoms = atom_indices is not None
    provided_atoms = list(atom_indices) if has_plugin_atoms else [() for _ in range(n)]
    pbc = np.asarray(ctx.pbc, dtype=bool)

    classifications: list[tuple[str, tuple[int, ...]] | None] = [None] * n
    for i, hint in enumerate(hints):
        if hint in _TOPOLOGY_SOURCE_TO_TYPE:
            # Defer to Delaunay when supports are empty (Voronoi enrich on slabs).
            if (
                ctx.delaunay is not None
                and hint != "atop_injected"
                and not provided_atoms[i]
            ):
                continue
            site_type = _TOPOLOGY_SOURCE_TO_TYPE[hint]
            atoms_i = tuple(int(j) for j in provided_atoms[i])
            if not atoms_i:
                n_keep = {"atop": 1, "bridge": 2, "hollow": 3}[site_type]
                n_keep = min(n_keep, len(positions))
                if n_keep and ctx.class_idx is not None:
                    atoms_i = tuple(
                        int(j) for j in np.asarray(ctx.class_idx[i]).ravel()[:n_keep]
                    )
                elif n_keep:
                    _, idx = local_tree.query(vertices[i].reshape(1, 3), k=n_keep)
                    atoms_i = tuple(int(j) for j in np.atleast_1d(idx).ravel())
            classifications[i] = (site_type, atoms_i)
            continue
        if hint in _SUPPORT_COUNT_TYPED_SOURCES and provided_atoms[i]:
            atoms_i = tuple(int(j) for j in provided_atoms[i])
            classifications[i] = (
                _site_type_from_support_count(len(atoms_i)),
                atoms_i,
            )

    need = [i for i, c in enumerate(classifications) if c is None]
    if need:
        if ctx.delaunay is not None:
            delaunay_all = _classify_delaunay_vertices_batch(
                ctx, vertices, positions, local_tree, pore_threshold=pore_threshold
            )
            for i in need:
                classifications[i] = delaunay_all[i]
        else:
            if ctx.class_dists is None or ctx.class_idx is None:
                raise ValueError(
                    "ctx.class_dists and ctx.class_idx must be set for Voronoi classification"
                )
            for i in need:
                classifications[i] = _classify_voronoi_site_from_neighbors(
                    ctx.class_dists[i],
                    ctx.class_idx[i],
                    pore_threshold=pore_threshold,
                )

    sites: list[Site] = []
    for i, classified in enumerate(classifications):
        assert classified is not None
        site_type, nearest_idx = classified
        support: tuple[int, ...]
        if site_type == "pore":
            support = ()
        elif has_plugin_atoms:
            support = tuple(int(j) for j in provided_atoms[i])
        else:
            support = tuple(int(j) for j in nearest_idx)

        if normals is not None:
            normal = np.asarray(normals[i], dtype=float)
        elif material_type == "slab":
            normal = np.asarray(ctx.normals[i], dtype=float)
        elif support:
            normal = _normal_from_support(
                vertices[i],
                positions,
                support,
                cell,
                pbc,
                fallback=ctx.normals[i],
            )
        else:
            normal = np.asarray(ctx.normals[i], dtype=float)

        xyz = np.asarray(vertices[i], dtype=float).copy()
        if site_type != "pore" and support:
            idx = np.asarray(support, dtype=int)
            n_pos = len(positions)
            idx = idx[(idx >= 0) & (idx < n_pos)]
            if idx.size > 0:
                xyz = project_anchor_to_support_plane(xyz, normal, positions[idx])

        dists = _support_mic_distances(xyz, positions, support, cell, pbc)
        side = _side_label_from_normal(normal, material_type=material_type, cell=cell)
        env_fingerprint = site_env_fingerprint(support, symbols, dists, side_label=side)
        nn_distance = float(nn_dists[i])
        clearance = nn_distance if clearances is None else float(clearances[i])
        sites.append(
            Site(
                xyz=xyz,
                normal=normal,
                site_type=site_type,
                slab_indices=support,
                material_type=material_type,
                site_source=hints[i],
                env_fingerprint=env_fingerprint,
                nn_distance=nn_distance,
                hollow_order=(
                    len(support) if site_type == "hollow" and support else None
                ),
                clearance=clearance,
                tangent_basis=tangent_basis_from_normal(normal),
            )
        )
    return sites


def project_sites_to_support_plane(
    sites: Sequence[Site],
    positions: np.ndarray,
    symbols: Sequence[str],
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> list[Site]:
    """Project wall-near site vertices onto the coordinating-atom plane.

    Prefer the in-classify projection (fingerprints already use unlifted xyz).
    This helper remains for tests and callers that build :class:`Site` records
    without going through :func:`_classify_vertices`. Pore / empty-support
    sites keep their free-volume vertex. Normals and tangent frames stay as
    classified; ``clearance`` / ``nn_distance`` stay probe metadata.
    """
    pos = np.asarray(positions, dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    n_pos = len(pos)
    out: list[Site] = []
    for site in sites:
        if site.site_type == "pore" or not site.slab_indices:
            out.append(site)
            continue
        idx = np.asarray(site.slab_indices, dtype=int)
        idx = idx[(idx >= 0) & (idx < n_pos)]
        if idx.size == 0:
            out.append(site)
            continue
        n_hat = np.asarray(site.normal, dtype=float)
        if float(np.linalg.norm(n_hat)) <= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            out.append(site)
            continue
        new_xyz = project_anchor_to_support_plane(site.xyz, n_hat, pos[idx])
        dists = _support_mic_distances(
            new_xyz, pos, site.slab_indices, cell_arr, pbc_arr
        )
        fp = site_env_fingerprint(
            site.slab_indices,
            symbols,
            dists,
            side_label=int(site.env_fingerprint[2]),
        )
        out.append(replace(site, xyz=new_xyz, env_fingerprint=fp))
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
    cell: np.ndarray,
    pbc: np.ndarray,
    source_hints: list[str] | None = None,
    delaunay: _DelaunayClassifyInputs | None = None,
    atom_indices: list[tuple[int, ...]] | None = None,
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
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
        atom_indices=atom_indices,
        cell=cell,
        normals=normals,
        clearances=clearances,
    )

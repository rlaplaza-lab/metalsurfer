"""Atom-centred adaptive Cartesian grid for adsorption-site candidates.

One PBC-aware pipeline for every material: shells around all atoms, atom-aware
clearance, exposure/side policy, one representative per support key snapped to
a target clearance above the support centroid, and a ``merge_radius`` NMS so
the catalog does not oversample. This is wall-near sampling — not free-volume
/ pore-centre enumeration.

Spacing is the exposed absolute ``adaptive_grid_spacing`` (Å) with optional
refine halvings; ``merge_radius`` tracks spacing and floors on framework
median NN. Site identity comes from support atoms, local normals, and
clearance — adsorbate size must not redefine the substrate surface.

Classify / cluster / symmetry / placement use the shared enumerator path —
this module returns candidate vertices plus support indices / clearances /
normals for :class:`~metalsurfer.placement.site_types.Site` construction.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.spatial import KDTree

from .._utils import cell_has_volume
from ._constants import (
    _ADAPTIVE_GRID_BIN_PRETHIN,
    _ADAPTIVE_GRID_DEFAULT_REFINE_LEVELS,
    _ADAPTIVE_GRID_DEFAULT_SPACING,
    _ADAPTIVE_GRID_EXPOSURE_N_STEPS,
    _ADAPTIVE_GRID_EXPOSURE_STEP,
    _ADAPTIVE_GRID_FINE_SCALE,
    _ADAPTIVE_GRID_H_MIN,
    _ADAPTIVE_GRID_MAX_SUPPORT,
    _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE,
    _ADAPTIVE_GRID_NMS_LENGTH_SCALE,
    _ADAPTIVE_GRID_NMS_SCALE,
    _ADAPTIVE_GRID_PRETHIN_BIN,
    _ADAPTIVE_GRID_REFINE_POS_TOL,
    _ADAPTIVE_GRID_REFINE_SCORE_TOL,
    _ADAPTIVE_GRID_SCORE_W_BALANCE,
    _ADAPTIVE_GRID_SCORE_W_CLEARANCE,
    _ADAPTIVE_GRID_SCORE_W_GRADIENT,
    _ADAPTIVE_GRID_STATIONARITY_STEP,
    _ADAPTIVE_GRID_SUPPORT_DELTA,
    _ADAPTIVE_GRID_WORK_BUDGET,
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _VECTOR_NORM_EPS,
    _VORONOI_DEDUP_TOLERANCE,
)
from ._parallel import resolve_materialize_workers
from .geometry import (
    _get_covalent_radius,
    tangent_basis_from_normal,
)
from .site_coords import (
    _cart_to_frac,
    _deduplicate_points,
    _minimum_image_cartesian_delta,
    _periodic_image_offsets,
    _slab_normal,
    _wrap_cartesian,
    _wrap_fractional,
)

_SOURCE_HINT = "adaptive_grid"
_DEFAULT_N_JOBS = -2

SidePolicy = Literal["all", "positive", "negative", "external"]


@dataclass(frozen=True)
class CandidateSite:
    """Internal adaptive-grid candidate before shared Site construction."""

    position: np.ndarray
    clearance: float
    support_indices: tuple[int, ...]
    support_image_shifts: tuple[tuple[int, int, int], ...]
    support_distances: tuple[float, ...]
    normal: np.ndarray
    score: float
    basin_id: int = -1


@dataclass(frozen=True)
class AdaptiveGridSpacing:
    """Absolute shell spacing and derived refine / merge radii."""

    characteristic_length: float
    initial_spacing: float
    fine_spacing: float
    max_levels: int
    merge_radius: float


@dataclass(frozen=True)
class AdaptiveGridResult:
    """Vertices plus per-candidate metadata for the enumerator/plugin."""

    vertices: np.ndarray
    nn_dists: np.ndarray
    clearances: np.ndarray
    atom_indices: list[tuple[int, ...]]
    normals: np.ndarray
    spacing: AdaptiveGridSpacing
    accessibility_tree: KDTree


def _framework_median_nn(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> float:
    """MIC median nearest-neighbour spacing of framework atoms."""
    # Deferred: site_plugins.__init__ imports AdaptiveGridGenerator → this module.
    from .site_plugins.helpers import median_nn_or_fallback

    pts = np.asarray(positions, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(
        median_nn_or_fallback(
            np.empty(0, dtype=float),
            reference_positions=pts,
            cell=cell,
            pbc=pbc,
        )
    )


def _framework_radii_from_symbols(symbols: Sequence[str]) -> np.ndarray:
    """Per-atom covalent radii with a shared fallback for missing tables."""
    radii = []
    for sym in symbols:
        r = _get_covalent_radius(str(sym))
        radii.append(
            float(r)
            if r is not None and r > 0.0
            else float(_SURFACE_COVALENT_RADIUS_FALLBACK)
        )
    return np.asarray(radii, dtype=float)


def adaptive_grid_spacing(
    *,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
    framework_median_nn: float | None = None,
    nms_framework_scale: float | None = None,
) -> AdaptiveGridSpacing:
    """Return coarse/fine spacing and refine depth for the adaptive grid.

    *initial_spacing* is the absolute shell increment in Å
    (``AdsorptionConfig.adaptive_grid_spacing``). When omitted, the module
    default is used.
    """
    nn = float(framework_median_nn) if framework_median_nn is not None else 0.0
    levels = (
        max(0, int(max_levels))
        if max_levels is not None
        else int(_ADAPTIVE_GRID_DEFAULT_REFINE_LEVELS)
    )
    nms_scale = (
        float(nms_framework_scale)
        if nms_framework_scale is not None
        else float(_ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE)
    )
    if not np.isfinite(nms_scale) or nms_scale <= 0.0:
        raise ValueError(
            f"nms_framework_scale must be finite and > 0, got {nms_scale!r}"
        )

    h0 = (
        float(initial_spacing)
        if initial_spacing is not None
        else float(_ADAPTIVE_GRID_DEFAULT_SPACING)
    )
    if not np.isfinite(h0) or h0 <= 0.0:
        raise ValueError(
            f"adaptive_grid initial_spacing must be finite and > 0, got {h0!r}"
        )
    L = h0

    h_target = max(_ADAPTIVE_GRID_FINE_SCALE * L, _ADAPTIVE_GRID_H_MIN * 0.5)
    h_fine = h0
    for _ in range(levels):
        if h_fine <= h_target:
            break
        h_fine *= 0.5
    merge = max(
        _ADAPTIVE_GRID_NMS_SCALE * h_fine,
        _ADAPTIVE_GRID_NMS_LENGTH_SCALE * L,
        _VORONOI_DEDUP_TOLERANCE,
    )
    if np.isfinite(nn) and nn > 0.0:
        merge = max(merge, nms_scale * nn)
    return AdaptiveGridSpacing(
        characteristic_length=float(L),
        initial_spacing=float(h0),
        fine_spacing=float(h_fine),
        max_levels=levels,
        merge_radius=float(merge),
    )


def _fractional_bin_keys(
    frac: np.ndarray,
    dfrac: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Equal-width fractional bin keys (periodic axes wrap with ``n_bins``)."""
    keys = np.empty_like(frac, dtype=np.int64)
    pbc_arr = np.asarray(pbc, dtype=bool)
    for dim in range(3):
        f = frac[:, dim]
        if bool(pbc_arr[dim]):
            f = np.mod(f, 1.0)
            n_bins = max(1, int(np.ceil(1.0 / dfrac[dim])))
            keys[:, dim] = np.floor(f * n_bins).astype(np.int64) % n_bins
        else:
            keys[:, dim] = np.floor(f / dfrac[dim]).astype(np.int64)
    return keys


def _wrap_dedup(points: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    inv = np.linalg.inv(cell) if np.any(pbc) else None
    wrapped = _wrap_cartesian(points, cell, pbc, inv_cell=inv)
    return wrapped[
        _deduplicate_points(wrapped, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    ]


def _exact_pbc_union(
    a: np.ndarray,
    b: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Union of two point sets with numerical PBC deduplication."""
    if len(a) == 0:
        return _wrap_dedup(b, cell, pbc) if len(b) else a
    if len(b) == 0:
        return _wrap_dedup(a, cell, pbc)
    return _wrap_dedup(np.vstack([a, b]), cell, pbc)


def _shell_offsets(radius: float, spacing: float, *, r_min: float = 0.0) -> np.ndarray:
    """Lab-frame Cartesian shell offsets; skip the inaccessible inner ball."""
    h = float(spacing)
    if h <= 0.0 or radius <= 0.0:
        return np.empty((0, 3), dtype=float)
    n = int(np.ceil(radius / h))
    coords = np.arange(-n, n + 1, dtype=float) * h
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    local = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    r = np.linalg.norm(local, axis=1)
    keep = (r > max(float(r_min), 1e-12)) & (r <= float(radius) + 1e-9)
    return local[keep]


def _clearance_query(
    vertices: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Atom-aware clearance ``c = min(||x-R_i|| - r_i)`` and nearest image xyz."""
    if len(vertices) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, np.empty(0, dtype=float), empty
    k = min(max(1, _ADAPTIVE_GRID_MAX_SUPPORT), len(np.asarray(tree.data)))
    dists, idxs = tree.query(vertices, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    idxs = np.atleast_2d(np.asarray(idxs, dtype=int))
    atom_idx = idxs % n_atoms
    effective = dists - framework_radii[atom_idx]
    clearance = np.min(effective, axis=1)
    nearest_col = np.argmin(effective, axis=1)
    nearest_img = idxs[np.arange(len(vertices)), nearest_col]
    data = np.asarray(tree.data, dtype=float)
    return vertices, clearance, data[nearest_img]


def _filter_accessible_clearance(
    vertices: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    min_clearance: float,
    max_clearance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep points in the clearance window; return verts, clearance, nearest xyz."""
    verts, clearance, nearest = _clearance_query(
        vertices, tree, framework_radii, n_atoms
    )
    if len(verts) == 0:
        return verts, clearance, nearest
    keep = (clearance >= float(min_clearance)) & (clearance <= float(max_clearance))
    return verts[keep], clearance[keep], nearest[keep]


def _image_shift_for_index(
    image_index: int,
    n_atoms: int,
    positions: np.ndarray,
    tree_data: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[int, tuple[int, int, int]]:
    """Map a KDTree image index to (base_atom, lattice shift)."""
    base = int(image_index) % n_atoms
    if not np.any(pbc) or not cell_has_volume(cell):
        return base, (0, 0, 0)
    img_pos = np.asarray(tree_data[int(image_index)], dtype=float)
    primary = np.asarray(positions[base], dtype=float)
    delta = img_pos - primary
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    shift = (
        int(np.rint(frac[0])) if bool(pbc[0]) else 0,
        int(np.rint(frac[1])) if bool(pbc[1]) else 0,
        int(np.rint(frac[2])) if bool(pbc[2]) else 0,
    )
    return base, shift


def _support_environment(
    vertices: np.ndarray,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    support_delta: float = _ADAPTIVE_GRID_SUPPORT_DELTA,
    max_support: int = _ADAPTIVE_GRID_MAX_SUPPORT,
) -> tuple[
    list[tuple[int, ...]],
    list[tuple[tuple[int, int, int], ...]],
    list[np.ndarray],
    np.ndarray,
]:
    """Adaptive support sets with periodic image shifts retained."""
    n_atoms = len(positions)
    if len(vertices) == 0 or n_atoms == 0:
        return [], [], [], np.empty(0, dtype=float)
    k = min(max_support, len(np.asarray(tree.data)))
    dists, image_indices = tree.query(vertices, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    image_indices = np.atleast_2d(np.asarray(image_indices, dtype=int))
    atom_indices = image_indices % n_atoms
    effective = dists - framework_radii[atom_indices]
    effective_min = np.min(effective, axis=1)
    tree_data = np.asarray(tree.data, dtype=float)

    support_sets: list[tuple[int, ...]] = []
    support_shifts: list[tuple[tuple[int, int, int], ...]] = []
    support_dists: list[np.ndarray] = []

    for row in range(len(vertices)):
        mask = effective[row] <= effective_min[row] + float(support_delta)
        ids = atom_indices[row, mask]
        imgs = image_indices[row, mask]
        ds = effective[row, mask]
        order = np.lexsort((ids, ds))
        ids = ids[order]
        imgs = imgs[order]
        ds = ds[order]

        seen: dict[tuple[int, tuple[int, int, int]], float] = {}
        for _atom_i, img_i, d_i in zip(ids, imgs, ds, strict=True):
            base, shift = _image_shift_for_index(
                int(img_i), n_atoms, positions, tree_data, cell, pbc
            )
            key = (base, shift)
            if key not in seen or d_i < seen[key]:
                seen[key] = float(d_i)
        items = sorted(seen.items(), key=lambda kv: (kv[1], kv[0][0], kv[0][1]))
        items = items[:max_support]
        support_sets.append(tuple(int(k[0]) for k, _ in items))
        support_shifts.append(tuple(k[1] for k, _ in items))
        support_dists.append(np.asarray([v for _, v in items], dtype=float))

    return support_sets, support_shifts, support_dists, effective_min


def _local_surface_normal(
    vertex: np.ndarray,
    support_positions: np.ndarray,
    support_distances: np.ndarray,
) -> np.ndarray:
    """Weighted PCA normal from support atoms; radial fallback for one atom."""
    if len(support_positions) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    if len(support_positions) == 1:
        vector = np.asarray(vertex, dtype=float) - support_positions[0]
        norm = float(np.linalg.norm(vector))
        if norm < _VECTOR_NORM_EPS:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vector / norm

    weights = 1.0 / np.maximum(np.asarray(support_distances, dtype=float), 1e-6) ** 2
    weights = weights / weights.sum()
    centroid = np.sum(weights[:, None] * support_positions, axis=0)
    centered = support_positions - centroid
    covariance = (centered * weights[:, None]).T @ centered
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    outward = np.asarray(vertex, dtype=float) - centroid
    if float(np.dot(normal, outward)) < 0.0:
        normal = -normal
    nrm = float(np.linalg.norm(normal))
    if nrm < _VECTOR_NORM_EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normal / nrm


def _support_positions_for_candidate(
    support_indices: tuple[int, ...],
    support_shifts: tuple[tuple[int, int, int], ...],
    positions: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    if not support_indices:
        return np.empty((0, 3), dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    out = []
    for idx, shift in zip(support_indices, support_shifts, strict=True):
        offset = (
            float(shift[0]) * cell_arr[0]
            + float(shift[1]) * cell_arr[1]
            + float(shift[2]) * cell_arr[2]
        )
        out.append(positions[int(idx)] + offset)
    return np.asarray(out, dtype=float)


def _batch_clearances(
    points: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Atom-aware clearances for a batch of points (shape ``(N,)``)."""
    if len(points) == 0:
        return np.empty(0, dtype=float)
    _, clearances, _ = _clearance_query(points, tree, framework_radii, n_atoms)
    return clearances


def _batch_tangential_stationarity(
    vertices: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    step: float = _ADAPTIVE_GRID_STATIONARITY_STEP,
) -> np.ndarray:
    """Batched finite-difference ||grad_parallel clearance|| for each vertex."""
    n = len(vertices)
    if n == 0:
        return np.empty(0, dtype=float)
    # Two tangent directions × ±step → 4 sample points per vertex.
    samples = np.empty((n, 4, 3), dtype=float)
    for i in range(n):
        basis = tangent_basis_from_normal(normals[i])
        samples[i, 0] = vertices[i] + float(step) * basis[0]
        samples[i, 1] = vertices[i] - float(step) * basis[0]
        samples[i, 2] = vertices[i] + float(step) * basis[1]
        samples[i, 3] = vertices[i] - float(step) * basis[1]
    clear = _batch_clearances(
        samples.reshape(-1, 3), tree, framework_radii, n_atoms
    ).reshape(n, 4)
    inv_2h = 1.0 / (2.0 * float(step))
    g0 = (clear[:, 0] - clear[:, 1]) * inv_2h
    g1 = (clear[:, 2] - clear[:, 3]) * inv_2h
    return np.sqrt(g0 * g0 + g1 * g1)


def _support_balance(support_distances: np.ndarray) -> float:
    """Higher when support effective distances are balanced (bridge/hollow)."""
    if len(support_distances) <= 1:
        return 0.0
    ds = np.asarray(support_distances, dtype=float)
    spread = float(np.std(ds))
    return float(1.0 / (1.0 + spread))


def _shell_target_clearance(
    min_clearance: float,
    max_clearance: float,
    median_nn: float,
) -> float:
    """Target clearance inside the accessibility window."""
    if median_nn > 0.0:
        target = 0.5 * float(median_nn) * _ATOP_INJECTION_HEIGHT_FACTOR
    else:
        target = 0.5 * (float(min_clearance) + float(max_clearance))
    return float(np.clip(target, min_clearance, max_clearance))


def _ray_exposure_mask(
    vertices: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    *,
    step: float = _ADAPTIVE_GRID_EXPOSURE_STEP,
    n_steps: int = _ADAPTIVE_GRID_EXPOSURE_N_STEPS,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Vectorized ray exposure: clearance non-decreasing along each local normal."""
    n = len(vertices)
    if n == 0:
        return np.ones(0, dtype=bool)
    n_steps_i = int(n_steps)
    steps = np.arange(0, n_steps_i + 1, dtype=float)
    # (n, n_steps+1, 3): vertex + k * step * normal
    pts = (
        np.asarray(vertices, dtype=float)[:, None, :]
        + steps[None, :, None]
        * float(step)
        * np.asarray(normals, dtype=float)[:, None, :]
    )
    clear = _batch_clearances(
        pts.reshape(-1, 3), tree, framework_radii, n_atoms
    ).reshape(n, n_steps_i + 1)
    diffs = clear[:, 1:] - clear[:, :-1]
    return np.all(diffs >= -float(tolerance), axis=1)


def _side_policy_mask(
    normals: np.ndarray,
    *,
    material_type: str,
    cell: np.ndarray,
    side_policy: SidePolicy,
    vertices: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Apply slab face / external exposure policy to candidate normals."""
    n = len(normals)
    if n == 0:
        return np.ones(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    if material_type == "slab" and cell_has_volume(cell):
        n_hat = _slab_normal(cell)
        dots = np.einsum("ij,j->i", normals, n_hat)
        if side_policy == "positive":
            keep &= dots >= -1e-12
        elif side_policy == "negative":
            keep &= dots <= 1e-12
        elif side_policy in ("all", "external"):
            pass
        else:
            raise ValueError(f"Unknown side_policy {side_policy!r}")
    elif side_policy == "external" and vertices is not None and positions is not None:
        com = np.mean(np.asarray(positions, dtype=float), axis=0)
        outward = np.asarray(vertices, dtype=float) - com
        norms = np.linalg.norm(outward, axis=1)
        valid = norms > _VECTOR_NORM_EPS
        uhat = np.zeros_like(outward)
        uhat[valid] = outward[valid] / norms[valid, None]
        keep &= np.einsum("ij,ij->i", normals, uhat) >= -1e-12
    return keep


def _exposure_mask(
    vertices: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    *,
    material_type: str,
    cell: np.ndarray,
    side_policy: SidePolicy = "positive",
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Ray exposure along local normals plus side-policy half-space filter."""
    n = len(vertices)
    if n == 0:
        return np.ones(0, dtype=bool)
    keep = _ray_exposure_mask(
        vertices,
        normals,
        tree,
        framework_radii,
        n_atoms,
    )
    keep &= _side_policy_mask(
        normals,
        material_type=material_type,
        cell=cell,
        side_policy=side_policy,
        vertices=vertices,
        positions=positions,
    )
    return keep


def _pair_within_radius(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> bool:
    """Return True if *a* and *b* are within *radius* under triclinic-safe MIC."""
    return _mic_distance(a, b, cell, pbc) <= float(radius) + 1e-12


def _mic_distance(
    a: np.ndarray,
    b: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> float:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    if np.any(pbc_arr) and cell_has_volume(cell):
        delta = _minimum_image_cartesian_delta(delta, cell, pbc_arr)
    return float(np.linalg.norm(delta))


def _nms_rank_order(
    vertices: np.ndarray,
    scores: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Deterministic lex order: higher score, then wrapped fractional xyz."""
    n = len(vertices)
    if n == 0:
        return np.empty(0, dtype=int)
    if cell_has_volume(cell):
        frac = _wrap_fractional(_cart_to_frac(vertices, cell), pbc)
    else:
        frac = np.asarray(vertices, dtype=float)
    return np.lexsort(
        (
            frac[:, 2],
            frac[:, 1],
            frac[:, 0],
            -np.asarray(scores, dtype=float),
        )
    )


def _image_offsets_for_radius(
    cell: np.ndarray, pbc: np.ndarray, radius: float
) -> list[np.ndarray] | None:
    if not np.any(pbc) or not cell_has_volume(cell):
        return None
    return _periodic_image_offsets(
        np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool), float(radius)
    )


def _nms(
    vertices: np.ndarray,
    nn: np.ndarray,
    scores: np.ndarray,
    merge_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    radii: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS with deterministic ranking and min(r_i, r_j) suppression."""
    if len(vertices) == 0:
        return vertices, nn
    if radii is None:
        radii = np.full(len(vertices), float(merge_r), dtype=float)
    max_r = float(max(float(merge_r), float(np.max(radii))))
    offsets = _image_offsets_for_radius(cell, pbc, max_r)
    if offsets is None:
        tree = KDTree(vertices)

        def query_ball(pt: np.ndarray, r: float) -> list[int]:
            return tree.query_ball_point(pt, r=float(r))

    else:
        n = len(vertices)
        expanded = np.vstack([vertices + off for off in offsets])
        tree = KDTree(expanded)

        def query_ball(pt: np.ndarray, r: float) -> list[int]:
            hits = tree.query_ball_point(pt, r=float(r))
            return list({int(h) % n for h in hits})

    suppressed = np.zeros(len(vertices), dtype=bool)
    accepted: list[int] = []
    for i in _nms_rank_order(vertices, scores, cell, pbc):
        ii = int(i)
        if suppressed[ii]:
            continue
        accepted.append(ii)
        r_i = float(radii[ii])
        for j in query_ball(vertices[ii], max_r):
            if j == ii or suppressed[j]:
                continue
            r_ij = float(min(r_i, float(radii[j])))
            if _pair_within_radius(vertices[ii], vertices[j], r_ij, cell, pbc):
                suppressed[j] = True
    idx = np.asarray(accepted, dtype=int)
    return vertices[idx], nn[idx]


def _bin_prethin(
    vertices: np.ndarray,
    scores: np.ndarray,
    *,
    bin_size: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """O(N) spatial hash: keep the best-scoring point per equal-width bin."""
    if len(vertices) == 0:
        return vertices, scores
    h = max(float(bin_size), 1e-6)
    sc = np.asarray(scores, dtype=float)
    if np.any(pbc) and cell_has_volume(cell):
        frac = _cart_to_frac(vertices, cell)
        spacings = np.linalg.norm(cell, axis=1)
        dfrac = np.maximum(h / np.maximum(spacings, 1e-12), 1e-9)
        keys = _fractional_bin_keys(frac, dfrac, pbc)
    else:
        keys = np.floor(vertices / h).astype(np.int64)
    order = _nms_rank_order(vertices, sc, cell, pbc)
    keys_sorted = keys[order]
    _, first = np.unique(keys_sorted, axis=0, return_index=True)
    keep = order[first]
    keep.sort()
    return vertices[keep], sc[keep]


def _support_key(
    indices: tuple[int, ...],
    shifts: tuple[tuple[int, int, int], ...],
) -> frozenset[tuple[int, tuple[int, int, int]]]:
    return frozenset(zip(indices, shifts, strict=True))


def _merge_by_radius(
    candidates: list[CandidateSite],
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    merge_radius: float,
) -> list[CandidateSite]:
    """Isotropic NMS at *merge_radius*; assign basin_id in score order."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [replace(candidates[0], basin_id=0)]

    verts = np.asarray([c.position for c in candidates], dtype=float)
    scores = np.asarray([c.score for c in candidates], dtype=float)
    kept_verts, _ = _nms(
        verts,
        np.zeros(len(verts), dtype=float),
        scores,
        merge_r=float(merge_radius),
        cell=cell,
        pbc=pbc,
    )
    tree_v = KDTree(verts)
    kept_idx = list(dict.fromkeys(int(tree_v.query(pt, k=1)[1]) for pt in kept_verts))
    out = [candidates[i] for i in kept_idx]
    rep_verts = np.asarray([c.position for c in out], dtype=float)
    rep_scores = np.asarray([c.score for c in out], dtype=float)
    rep_order = _nms_rank_order(rep_verts, rep_scores, cell, pbc)
    return [replace(out[int(i)], basin_id=b) for b, i in enumerate(rep_order)]


def _best_per_support_key(
    candidates: list[CandidateSite],
) -> list[CandidateSite]:
    """Keep the highest-score candidate for each support environment."""
    best: dict[frozenset[tuple[int, tuple[int, int, int]]], CandidateSite] = {}
    for c in candidates:
        key = _support_key(c.support_indices, c.support_image_shifts)
        prev = best.get(key)
        if prev is None or float(c.score) > float(prev.score):
            best[key] = c
    return list(best.values())


def _height_for_target_clearance(
    support_positions: np.ndarray,
    support_radii: np.ndarray,
    normal: np.ndarray,
    target_clearance: float,
) -> float:
    """Height along *normal* so atom-aware clearance ≈ *target_clearance*."""
    supp = np.asarray(support_positions, dtype=float)
    if len(supp) == 0:
        return float(target_clearance)
    n_hat = np.asarray(normal, dtype=float)
    nrm = float(np.linalg.norm(n_hat))
    if nrm < _VECTOR_NORM_EPS:
        return float(target_clearance)
    n_hat = n_hat / nrm
    centroid = np.mean(supp, axis=0)
    radii = np.asarray(support_radii, dtype=float)
    r_eff = float(np.mean(radii)) if len(radii) else 0.0
    lat2 = 0.0
    for p in supp:
        d = p - centroid
        d_perp = d - float(np.dot(d, n_hat)) * n_hat
        lat2 = max(lat2, float(np.dot(d_perp, d_perp)))
    need = float(target_clearance) + r_eff
    need2 = need * need
    if need2 > lat2:
        return float(np.sqrt(need2 - lat2))
    return max(need, float(target_clearance), _VECTOR_NORM_EPS)


def _lateral_snap_candidates(
    candidates: list[CandidateSite],
    *,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    target_clearance: float,
) -> list[CandidateSite]:
    """Snap each candidate to support-centroid at atom-aware *target_clearance*."""
    if not candidates:
        return []
    n_atoms = len(positions)
    out: list[CandidateSite] = []
    for c in candidates:
        supp = _support_positions_for_candidate(
            c.support_indices, c.support_image_shifts, positions, cell
        )
        if len(supp) == 0:
            continue
        normal = np.asarray(c.normal, dtype=float)
        nrm = float(np.linalg.norm(normal))
        if nrm < _VECTOR_NORM_EPS:
            continue
        n_hat = normal / nrm
        idxs = list(c.support_indices)
        radii = np.asarray(framework_radii[idxs], dtype=float)
        height = _height_for_target_clearance(
            supp, radii, n_hat, float(target_clearance)
        )
        centroid = np.mean(supp, axis=0)
        snapped = centroid + height * n_hat
        if np.any(pbc) and cell_has_volume(cell):
            snapped = _wrap_cartesian(snapped.reshape(1, 3), cell, pbc)[0]
        _, clearance, _ = _clearance_query(
            snapped.reshape(1, 3), tree, framework_radii, n_atoms
        )
        if float(clearance[0]) < 0.0:
            continue
        out.append(
            replace(
                c,
                position=np.asarray(snapped, dtype=float).copy(),
                clearance=float(clearance[0]),
            )
        )
    return out


def _finalize_candidates(
    candidates: list[CandidateSite],
    *,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    merge_radius: float,
    target_clearance: float,
) -> list[CandidateSite]:
    """One site per support key, snap to target clearance, then merge_radius NMS."""
    unique = _best_per_support_key(candidates)
    snapped = _lateral_snap_candidates(
        unique,
        positions=positions,
        framework_radii=framework_radii,
        tree=tree,
        cell=cell,
        pbc=pbc,
        target_clearance=target_clearance,
    )
    return _merge_by_radius(snapped, cell=cell, pbc=pbc, merge_radius=merge_radius)


def _work_chunk_bounds(n_seeds: int, n_offsets: int) -> list[tuple[int, int]]:
    """Split seeds so each chunk's cartesian product stays under the work budget."""
    if n_seeds <= 0:
        return []
    max_seeds = max(1, _ADAPTIVE_GRID_WORK_BUDGET // max(1, int(n_offsets)))
    return [
        (start, min(start + max_seeds, n_seeds))
        for start in range(0, n_seeds, max_seeds)
    ]


def _annotate_candidates(
    vertices: np.ndarray,
    clearances: np.ndarray,
    *,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    target_clearance: float,
) -> list[CandidateSite]:
    """Attach support / normal / score metadata to each vertex."""
    n_atoms = len(positions)
    if len(vertices) == 0:
        return []
    supports, shifts, dists, _ = _support_environment(
        vertices, positions, framework_radii, tree, cell, pbc
    )
    normals = np.empty((len(vertices), 3), dtype=float)
    for i, vert in enumerate(vertices):
        supp_pos = _support_positions_for_candidate(
            supports[i], shifts[i], positions, cell
        )
        normals[i] = _local_surface_normal(vert, supp_pos, dists[i])
    grads = _batch_tangential_stationarity(
        vertices, normals, tree, framework_radii, n_atoms
    )
    w_c = float(_ADAPTIVE_GRID_SCORE_W_CLEARANCE)
    w_g = float(_ADAPTIVE_GRID_SCORE_W_GRADIENT)
    w_b = float(_ADAPTIVE_GRID_SCORE_W_BALANCE)
    out: list[CandidateSite] = []
    for i, vert in enumerate(vertices):
        balance = _support_balance(dists[i])
        score = (
            -w_c * abs(float(clearances[i]) - float(target_clearance))
            - w_g * float(grads[i])
            + w_b * balance
        )
        out.append(
            CandidateSite(
                position=np.asarray(vert, dtype=float).copy(),
                clearance=float(clearances[i]),
                support_indices=supports[i],
                support_image_shifts=shifts[i],
                support_distances=tuple(float(d) for d in dists[i]),
                normal=normals[i].copy(),
                score=float(score),
            )
        )
    return out


def _candidates_to_arrays(
    candidates: list[CandidateSite],
    *,
    tree: KDTree,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[int, ...]],
    np.ndarray,
]:
    """Return verts, centre-to-centre nn, clearances, atoms, normals."""
    if not candidates:
        empty = np.empty((0, 3), dtype=float)
        return (
            empty,
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
            [],
            empty,
        )
    verts = np.asarray([c.position for c in candidates], dtype=float)
    clear = np.asarray([c.clearance for c in candidates], dtype=float)
    atoms = [c.support_indices for c in candidates]
    normals = np.asarray([c.normal for c in candidates], dtype=float)
    nn = np.asarray(tree.query(verts, k=1)[0], dtype=float).ravel()
    return verts, nn, clear, atoms, normals


def _seed_chunk_candidates(
    seeds: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    min_clearance: float,
    max_clearance: float,
    *,
    material_type: str,
    cell: np.ndarray,
    side_policy: SidePolicy,
    positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(seeds) == 0 or len(offsets) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    pts = (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    verts, clearance, nearest = _filter_accessible_clearance(
        pts, tree, framework_radii, n_atoms, min_clearance, max_clearance
    )
    if len(verts) == 0:
        return verts, clearance
    # Provisional normals from nearest atom for exposure gating.
    delta = verts - nearest
    norms = np.linalg.norm(delta, axis=1)
    normals = np.zeros_like(delta)
    ok = norms > _VECTOR_NORM_EPS
    normals[ok] = delta[ok] / norms[ok, None]
    normals[~ok, 2] = 1.0
    keep = _exposure_mask(
        verts,
        normals,
        tree,
        framework_radii,
        n_atoms,
        material_type=material_type,
        cell=cell,
        side_policy=side_policy,
        positions=positions,
    )
    return verts[keep], clearance[keep]


def _parallel_shell_filter(
    centres: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    min_clearance: float,
    max_clearance: float,
    *,
    material_type: str,
    cell: np.ndarray,
    side_policy: SidePolicy,
    positions: np.ndarray,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centre+offset candidates and filter; chunk by work budget."""
    n = len(centres)
    if n == 0 or len(offsets) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    bounds = _work_chunk_bounds(n, len(offsets))
    if not bounds:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    def _one(bound: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = bound
        return _seed_chunk_candidates(
            centres[lo:hi],
            offsets,
            tree,
            framework_radii,
            n_atoms,
            min_clearance,
            max_clearance,
            material_type=material_type,
            cell=cell,
            side_policy=side_policy,
            positions=positions,
        )

    n_workers = resolve_materialize_workers(n_jobs, n_tasks=len(bounds))
    if n_workers == 1 or len(bounds) == 1:
        parts = [_one(b) for b in bounds]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_one, bounds))
    verts = [p[0] for p in parts if len(p[0])]
    dists = [p[1] for p in parts if len(p[1])]
    if not verts:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.vstack(verts), np.concatenate(dists)


def _basin_converged(
    old: CandidateSite | None,
    new: CandidateSite,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> bool:
    if old is None:
        return False
    if old.support_indices != new.support_indices:
        return False
    if old.support_image_shifts != new.support_image_shifts:
        return False
    if abs(old.score - new.score) >= _ADAPTIVE_GRID_REFINE_SCORE_TOL:
        return False
    return (
        _mic_distance(old.position, new.position, cell, pbc)
        < _ADAPTIVE_GRID_REFINE_POS_TOL
    )


def generate_adaptive_grid_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    material_type: str,
    probe_radius: float,
    max_site_distance: float,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
    framework_radii: np.ndarray | None = None,
    symbols: Sequence[str] | None = None,
    side_policy: SidePolicy = "positive",
    nms_framework_scale: float | None = None,
) -> AdaptiveGridResult:
    """Enumerate adaptive-grid candidates with support / clearance metadata.

    *probe_radius* / *max_site_distance* map onto the clearance window
    (element-dependent surface). *initial_spacing* is the absolute shell
    increment in Å.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    n_atoms = len(positions)
    if framework_radii is None:
        if symbols is not None and len(symbols) == n_atoms:
            framework_radii = _framework_radii_from_symbols(symbols)
        else:
            framework_radii = np.full(
                n_atoms, float(_SURFACE_COVALENT_RADIUS_FALLBACK), dtype=float
            )
    else:
        framework_radii = np.asarray(framework_radii, dtype=float)
        if len(framework_radii) != n_atoms:
            raise ValueError(
                f"framework_radii length {len(framework_radii)} != n_atoms {n_atoms}"
            )

    # Accessibility window in clearance space.
    mean_r = float(np.mean(framework_radii)) if n_atoms else 0.0
    min_clearance = float(probe_radius) - mean_r
    max_clearance = float(max_site_distance) - mean_r
    if max_clearance < min_clearance:
        max_clearance = min_clearance

    median_nn = _framework_median_nn(positions, cell, pbc)
    spacing = adaptive_grid_spacing(
        initial_spacing=initial_spacing,
        max_levels=max_levels,
        framework_median_nn=median_nn,
        nms_framework_scale=nms_framework_scale,
    )
    # Deferred: site_plugins.__init__ imports AdaptiveGridGenerator → this module.
    from .site_plugins.helpers import periodic_accessibility_tree

    tree = periodic_accessibility_tree(
        positions, cell, pbc, float(max_site_distance) + float(np.max(framework_radii))
    )
    empty = AdaptiveGridResult(
        vertices=np.empty((0, 3), dtype=float),
        nn_dists=np.empty(0, dtype=float),
        clearances=np.empty(0, dtype=float),
        atom_indices=[],
        normals=np.empty((0, 3), dtype=float),
        spacing=spacing,
        accessibility_tree=tree,
    )
    if n_atoms == 0:
        return empty

    shell_outer = float(max_clearance) + float(np.max(framework_radii))
    shell_inner = max(0.0, float(min_clearance) + float(np.min(framework_radii)))
    offsets = _shell_offsets(shell_outer, spacing.initial_spacing, r_min=shell_inner)
    if len(offsets) == 0:
        return empty

    raw_verts, _ = _parallel_shell_filter(
        positions,
        offsets,
        tree,
        framework_radii,
        n_atoms,
        min_clearance,
        max_clearance,
        material_type=material_type,
        cell=cell,
        side_policy=side_policy,
        positions=positions,
        n_jobs=n_jobs,
    )
    vertices = _wrap_dedup(raw_verts, cell, pbc)
    if len(vertices) == 0:
        return empty
    vertices, clearances, _ = _filter_accessible_clearance(
        vertices, tree, framework_radii, n_atoms, min_clearance, max_clearance
    )
    if len(vertices) == 0:
        return empty

    # Optional bin prethin for huge clouds, then annotate.
    if len(vertices) > _ADAPTIVE_GRID_BIN_PRETHIN:
        vertices, clearances = _bin_prethin(
            vertices,
            clearances,
            bin_size=float(_ADAPTIVE_GRID_PRETHIN_BIN),
            cell=cell,
            pbc=pbc,
        )

    target_c = _shell_target_clearance(min_clearance, max_clearance, median_nn)
    candidates = _annotate_candidates(
        vertices,
        clearances,
        positions=positions,
        framework_radii=framework_radii,
        tree=tree,
        cell=cell,
        pbc=pbc,
        target_clearance=target_c,
    )
    verts, _nn, _clears, _, normals = _candidates_to_arrays(candidates, tree=tree)
    keep = _exposure_mask(
        verts,
        normals,
        tree,
        framework_radii,
        n_atoms,
        material_type=material_type,
        cell=cell,
        side_policy=side_policy,
        positions=positions,
    )
    candidates = [c for c, k in zip(candidates, keep, strict=True) if k]
    if not candidates:
        return empty

    h = spacing.initial_spacing
    h_target = max(
        _ADAPTIVE_GRID_FINE_SCALE * spacing.characteristic_length,
        _ADAPTIVE_GRID_H_MIN * 0.5,
    )
    all_candidates = list(candidates)
    frontier = list(candidates)
    prev_by_support: dict[
        frozenset[tuple[int, tuple[int, int, int]]], CandidateSite
    ] = {_support_key(c.support_indices, c.support_image_shifts): c for c in frontier}

    for _ in range(spacing.max_levels):
        if not frontier or h <= h_target:
            break
        frontier_reps = _merge_by_radius(
            frontier,
            cell=cell,
            pbc=pbc,
            merge_radius=spacing.merge_radius,
        )
        f_verts = np.asarray([c.position for c in frontier_reps], dtype=float)
        f_scores = np.asarray([c.score for c in frontier_reps], dtype=float)
        if len(f_verts) > _ADAPTIVE_GRID_BIN_PRETHIN:
            f_verts, f_scores = _bin_prethin(
                f_verts,
                f_scores,
                bin_size=float(_ADAPTIVE_GRID_PRETHIN_BIN),
                cell=cell,
                pbc=pbc,
            )
        if len(f_verts) == 0:
            break

        h2 = 0.5 * h
        stencil = np.array(
            [
                [dx, dy, dz]
                for dx in (-h2, 0.0, h2)
                for dy in (-h2, 0.0, h2)
                for dz in (-h2, 0.0, h2)
            ],
            dtype=float,
        )
        raw_verts, _ = _parallel_shell_filter(
            f_verts,
            stencil,
            tree,
            framework_radii,
            n_atoms,
            min_clearance,
            max_clearance,
            material_type=material_type,
            cell=cell,
            side_policy=side_policy,
            positions=positions,
            n_jobs=n_jobs,
        )
        child_verts = _wrap_dedup(raw_verts, cell, pbc)
        child_verts, child_clear, _ = _filter_accessible_clearance(
            child_verts,
            tree,
            framework_radii,
            n_atoms,
            min_clearance,
            max_clearance,
        )
        if len(child_verts) == 0:
            break
        child_cands = _annotate_candidates(
            child_verts,
            child_clear,
            positions=positions,
            framework_radii=framework_radii,
            tree=tree,
            cell=cell,
            pbc=pbc,
            target_clearance=target_c,
        )
        cv, _, _, _, cn = _candidates_to_arrays(child_cands, tree=tree)
        keep_c = _exposure_mask(
            cv,
            cn,
            tree,
            framework_radii,
            n_atoms,
            material_type=material_type,
            cell=cell,
            side_policy=side_policy,
            positions=positions,
        )
        child_cands = [c for c, k in zip(child_cands, keep_c, strict=True) if k]

        union_verts = _exact_pbc_union(
            np.asarray([c.position for c in all_candidates], dtype=float),
            np.asarray([c.position for c in child_cands], dtype=float)
            if child_cands
            else np.empty((0, 3), dtype=float),
            cell,
            pbc,
        )
        union_verts, union_clear, _ = _filter_accessible_clearance(
            union_verts,
            tree,
            framework_radii,
            n_atoms,
            min_clearance,
            max_clearance,
        )
        all_candidates = _annotate_candidates(
            union_verts,
            union_clear,
            positions=positions,
            framework_radii=framework_radii,
            tree=tree,
            cell=cell,
            pbc=pbc,
            target_clearance=target_c,
        )
        all_candidates = _merge_by_radius(
            all_candidates,
            cell=cell,
            pbc=pbc,
            merge_radius=spacing.merge_radius,
        )

        new_frontier: list[CandidateSite] = []
        for c in child_cands:
            key = _support_key(c.support_indices, c.support_image_shifts)
            old = prev_by_support.get(key)
            if not _basin_converged(old, c, cell=cell, pbc=pbc):
                new_frontier.append(c)
            prev_by_support[key] = c
        frontier = new_frontier
        h = h2

    if not all_candidates:
        return empty

    clustered = _finalize_candidates(
        all_candidates,
        positions=positions,
        framework_radii=framework_radii,
        tree=tree,
        cell=cell,
        pbc=pbc,
        merge_radius=spacing.merge_radius,
        target_clearance=target_c,
    )
    if not clustered:
        return empty
    verts, nn_dists, clears, atoms, normals = _candidates_to_arrays(
        clustered, tree=tree
    )
    return AdaptiveGridResult(
        vertices=verts,
        nn_dists=nn_dists,
        clearances=clears,
        atom_indices=atoms,
        normals=normals,
        spacing=spacing,
        accessibility_tree=tree,
    )

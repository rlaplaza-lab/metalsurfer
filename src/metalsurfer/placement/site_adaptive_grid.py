"""Atom-centred adaptive Cartesian grid for adsorption-site candidates.

All materials use the same pipeline: shells around seed atoms, accessibility
filtering, iterative refinement, and thinning toward a near-atom adsorption
shell (not free-volume / pore centres).

Spacing may be driven by a shared ``grid_spacing_scale`` (typically the
**minimum** adsorbate characteristic length across competing molecules) so one
catalog serves every adsorbate. Per-adsorbate clearance and ranking happen at
placement time, not by regenerating sites.

CPU-parallel stages (seed shells and refine stencils) honour joblib-style
``n_jobs`` via :func:`~metalsurfer.placement._parallel.resolve_materialize_workers`
and a thread pool; greedy PBC NMS stays serial.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.spatial import KDTree

from .._utils import cell_has_volume
from ._constants import (
    _ADAPTIVE_GRID_EXTENT_EPS,
    _ADAPTIVE_GRID_FINE_SCALE,
    _ADAPTIVE_GRID_H_MAX,
    _ADAPTIVE_GRID_H_MIN,
    _ADAPTIVE_GRID_MAX_CANDIDATES,
    _ADAPTIVE_GRID_MAX_LEVELS,
    _ADAPTIVE_GRID_NMS_LENGTH_SCALE,
    _ADAPTIVE_GRID_NMS_SCALE,
    _ADAPTIVE_GRID_SPACING_SCALE,
    _ADSORBATE_COVALENT_RADIUS_FALLBACK,
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _VORONOI_DEDUP_TOLERANCE,
)
from ._parallel import resolve_materialize_workers
from .geometry import _classify_molecule_shape
from .occupancy import incoming_inplane_radius
from .site_coords import (
    _build_periodic_images,
    _cart_to_frac,
    _deduplicate_points,
    _mean_covalent_radius,
    _pbc_merge_pair_set,
    _periodic_image_offsets,
    _slab_normal,
    _wrap_cartesian,
    top_layer_mask_by_normal,
)
from .site_np import (
    _convex_hull_surface_mask,
    _outside_convex_hull_mask,
    _try_convex_hull,
)

_SOURCE_HINT = "adaptive_grid"
# Default matches AdsorptionConfig.n_jobs (all CPUs but one).
_DEFAULT_N_JOBS = -2


def _accessibility_tree(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    max_distance: float,
) -> KDTree:
    """KDTree over periodic images for candidate-to-framework distance gating."""
    if not np.any(pbc) or not cell_has_volume(cell):
        return KDTree(positions)
    margin = float(max_distance) + _VORONOI_DEDUP_TOLERANCE
    return KDTree(_build_periodic_images(positions, cell, pbc, margin=margin))


def _framework_median_nn(
    positions: np.ndarray, cell: np.ndarray, pbc: np.ndarray
) -> float:
    """MIC median nearest-neighbour spacing of framework atoms."""
    pts = np.asarray(positions, dtype=float)
    if len(pts) < 2:
        return 0.0
    pbc_arr = np.asarray(pbc, dtype=bool)
    cell_arr = np.asarray(cell, dtype=float)
    if cell_has_volume(cell_arr) and np.any(pbc_arr):
        margin = float(np.max(np.linalg.norm(cell_arr[pbc_arr], axis=1)))
        offsets = _periodic_image_offsets(cell_arr, pbc_arr, margin)
        ext = np.vstack([pts + off for off in offsets])
        tree = KDTree(ext)
        k = min(len(ext), len(offsets) + 1)
        dists, idxs = tree.query(pts, k=k)
        dists = np.atleast_2d(np.asarray(dists, dtype=float))
        idxs = np.atleast_2d(np.asarray(idxs))
        n = len(pts)
        valid = (idxs % n) != np.arange(n)[:, None]
        nn = np.where(valid, dists, np.inf).min(axis=1)
        finite = nn[np.isfinite(nn)]
        if len(finite) > 0:
            return float(np.median(finite))
    tree = KDTree(pts)
    nn_d, _ = tree.query(pts, k=2)
    return float(np.median(np.asarray(nn_d, dtype=float)[:, 1]))


@dataclass(frozen=True)
class AdaptiveGridSpacing:
    """Adsorbate- or probe-derived grid increments."""

    characteristic_length: float
    initial_spacing: float
    fine_spacing: float
    max_levels: int
    merge_radius: float


def adaptive_grid_characteristic_length(
    probe_radius: float,
    adsorbate: Atoms | None = None,
) -> float:
    """Molecule size scale for grid density.

    Without an adsorbate, returns *probe_radius*. With an adsorbate, uses
    in-plane footprint / thickness / covalent radius (not capped by probe).
    """
    probe = float(probe_radius)
    if adsorbate is None or len(adsorbate) == 0:
        return probe
    eps = _ADAPTIVE_GRID_EXTENT_EPS
    footprint = float(incoming_inplane_radius(adsorbate, footprint_scale=1.0))
    pos = np.asarray(adsorbate.get_positions(), dtype=float)
    thickness = 0.0
    if len(pos) >= 2:
        centered = pos - np.mean(pos, axis=0)
        shape, _, eigenvecs = _classify_molecule_shape(centered)
        axis = eigenvecs[:, 2] if shape == "flat" else eigenvecs[:, 0]
        along = centered @ axis
        thickness = float(np.max(along) - np.min(along))
    r_cov = float(
        _mean_covalent_radius(
            list(adsorbate.get_chemical_symbols()),
            fallback=_ADSORBATE_COVALENT_RADIUS_FALLBACK,
        )
    )
    if footprint > eps:
        length = footprint
    elif thickness > eps:
        length = thickness
    elif r_cov > eps:
        length = r_cov
    else:
        length = probe
    return float(max(length, eps))


def min_adsorbate_grid_scale(
    probe_radius: float,
    adsorbates: Sequence[Atoms | None],
) -> float:
    """Return the minimum characteristic length across *adsorbates*.

    Used to size one shared adaptive-grid catalog for competing molecules:
    the smallest footprint drives the densest sampling that still covers all
    species. Empty / missing adsorbates are skipped; falls back to *probe_radius*.
    """
    scales = [
        adaptive_grid_characteristic_length(probe_radius, ads)
        for ads in adsorbates
        if ads is not None and len(ads) > 0
    ]
    return float(min(scales)) if scales else float(probe_radius)


def adsorbate_contact_distance(
    adsorbate: Atoms | None,
    surface_symbols: Sequence[str] | None = None,
) -> float | None:
    """Mean covalent contact distance (surface + adsorbate), or None.

    Used at placement time to rank shared adaptive-grid sites toward each
    molecule's preferred clearance; not applied during shared catalog generation.
    """
    if adsorbate is None or len(adsorbate) == 0:
        return None
    r_ads = float(
        _mean_covalent_radius(
            list(adsorbate.get_chemical_symbols()),
            fallback=_ADSORBATE_COVALENT_RADIUS_FALLBACK,
        )
    )
    if surface_symbols:
        r_surf = float(
            _mean_covalent_radius(
                list(surface_symbols),
                fallback=_SURFACE_COVALENT_RADIUS_FALLBACK,
            )
        )
    else:
        r_surf = float(_SURFACE_COVALENT_RADIUS_FALLBACK)
    return float(r_surf + r_ads)


def resolve_adaptive_grid_spacing_scale(
    probe_radius: float | None,
    adsorbates: Sequence[Atoms | None],
) -> float:
    """Min characteristic length across *adsorbates* (shared catalog scale).

    *probe_radius* is the fallback when no adsorbates are provided.
    """
    probe = float(probe_radius) if probe_radius is not None else 1.0
    return min_adsorbate_grid_scale(probe, adsorbates)


def adaptive_grid_spacing(
    probe_radius: float,
    adsorbate: Atoms | None = None,
    *,
    grid_spacing_scale: float | None = None,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
) -> AdaptiveGridSpacing:
    """Return coarse/fine spacing and refine depth for the adaptive grid."""
    if grid_spacing_scale is not None:
        L = float(grid_spacing_scale)
    else:
        L = adaptive_grid_characteristic_length(probe_radius, adsorbate)
    h0 = (
        float(initial_spacing)
        if initial_spacing is not None
        else float(
            np.clip(
                _ADAPTIVE_GRID_SPACING_SCALE * L,
                _ADAPTIVE_GRID_H_MIN,
                _ADAPTIVE_GRID_H_MAX,
            )
        )
    )
    h_target = max(_ADAPTIVE_GRID_FINE_SCALE * L, _ADAPTIVE_GRID_H_MIN * 0.5)
    levels = max(
        0, int(max_levels) if max_levels is not None else _ADAPTIVE_GRID_MAX_LEVELS
    )
    h_fine = h0
    for _ in range(levels):
        if h_fine <= h_target:
            break
        h_fine *= 0.5
    return AdaptiveGridSpacing(
        characteristic_length=float(L),
        initial_spacing=float(h0),
        fine_spacing=float(h_fine),
        max_levels=levels,
        merge_radius=float(
            max(
                _ADAPTIVE_GRID_NMS_SCALE * h_fine,
                _ADAPTIVE_GRID_NMS_LENGTH_SCALE * L,
                _VORONOI_DEDUP_TOLERANCE,
            )
        ),
    )


def _orthonormal_frame(cell: np.ndarray, material_type: str) -> np.ndarray:
    """Return 3x3 orthonormal basis (rows) for shell offsets.

    Slab: in-plane â, n̂×â, surface normal. Porous: QR of the cell. Nanoparticle:
    lab Cartesian identity (no preferred lattice frame).
    """
    if material_type == "nanoparticle":
        return np.eye(3, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if material_type == "slab":
        n_hat = _slab_normal(cell)
        a = np.asarray(cell[0], dtype=float)
        a_proj = a - np.dot(a, n_hat) * n_hat
        norm_a = float(np.linalg.norm(a_proj))
        if norm_a < 1e-12:
            trial = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(trial, n_hat)) > 0.9:
                trial = np.array([0.0, 1.0, 0.0])
            a_proj = trial - np.dot(trial, n_hat) * n_hat
            norm_a = float(np.linalg.norm(a_proj))
        a_hat = a_proj / max(norm_a, 1e-12)
        b_hat = np.cross(n_hat, a_hat)
        return np.vstack([a_hat, b_hat, n_hat])
    # Porous / other: orthonormalize cell rows via QR.
    q, _ = np.linalg.qr(cell.T)
    basis = q.T
    # Ensure right-handed orientation.
    if np.linalg.det(basis) < 0.0:
        basis = basis.copy()
        basis[2] *= -1.0
    return basis


def _seed_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    material_type: str,
    top_layer_tolerance: float,
    pbc: np.ndarray,
    *,
    seed_voxel: float | None = None,
) -> np.ndarray:
    """Seed atoms: slab top layer, NP hull skin, else all (optionally voxelled)."""
    n = len(positions)
    if material_type == "slab":
        idx = np.nonzero(
            top_layer_mask_by_normal(positions, cell, float(top_layer_tolerance))
        )[0]
        return idx if len(idx) else np.arange(n, dtype=int)
    if material_type == "nanoparticle":
        hull = _try_convex_hull(positions)
        if hull is None:
            return np.arange(n, dtype=int)
        idx = np.nonzero(_convex_hull_surface_mask(positions, hull=hull))[0]
        return idx if len(idx) else np.arange(n, dtype=int)
    idx = np.arange(n, dtype=int)
    if seed_voxel is not None and seed_voxel > 0.0 and n > 1:
        idx = _fractional_voxel_seeds(positions, cell, pbc, float(seed_voxel), idx)
    return idx


def _fractional_voxel_seeds(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    seed_voxel: float,
    idx: np.ndarray,
) -> np.ndarray:
    """One seed per fractional voxel; wrap periodic axes into ``[0, 1)``."""
    pts = positions[idx]
    if not cell_has_volume(cell):
        keys = np.floor(pts / float(seed_voxel)).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        return np.sort(idx[keep])
    frac = _cart_to_frac(pts, cell)
    spacings = np.linalg.norm(cell, axis=1)
    # Voxel size in fractional units along each lattice vector.
    dfrac = np.maximum(float(seed_voxel) / np.maximum(spacings, 1e-12), 1e-9)
    keys = np.empty_like(frac, dtype=np.int64)
    pbc_arr = np.asarray(pbc, dtype=bool)
    for dim in range(3):
        f = frac[:, dim]
        if bool(pbc_arr[dim]):
            f = np.mod(f, 1.0)
            n_bins = max(1, int(np.ceil(1.0 / dfrac[dim] - 1e-12)))
            keys[:, dim] = np.floor(f / dfrac[dim]).astype(np.int64) % n_bins
        else:
            keys[:, dim] = np.floor(f / dfrac[dim]).astype(np.int64)
    _, keep = np.unique(keys, axis=0, return_index=True)
    return np.sort(idx[keep])


def _wrap_dedup(points: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    inv = np.linalg.inv(cell) if np.any(pbc) else None
    wrapped = _wrap_cartesian(points, cell, pbc, inv_cell=inv)
    return wrapped[
        _deduplicate_points(wrapped, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    ]


def _shell_offsets(
    radius: float,
    spacing: float,
    frame: np.ndarray,
    *,
    r_min: float = 0.0,
) -> np.ndarray:
    """Cartesian shell offsets in *frame*; skip the inaccessible inner ball."""
    h = float(spacing)
    if h <= 0.0 or radius <= 0.0:
        return np.empty((0, 3), dtype=float)
    n = int(np.ceil(radius / h))
    coords = np.arange(-n, n + 1, dtype=float) * h
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    local = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    r = np.linalg.norm(local, axis=1)
    keep = (r > max(float(r_min), 1e-12)) & (r <= float(radius) + 1e-9)
    local = local[keep]
    if len(local) == 0:
        return np.empty((0, 3), dtype=float)
    # frame rows are orthonormal basis vectors → local @ frame.
    return local @ np.asarray(frame, dtype=float)


def _filter_accessible(
    vertices: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
    hull=None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    nn, _ = tree.query(vertices, k=1)
    nn = np.asarray(nn, dtype=float).ravel()
    keep = (nn >= probe) & (nn <= max_d)
    if hull is not None:
        keep &= _outside_convex_hull_mask(vertices, hull)
    return vertices[keep], nn[keep]


def _shell_target(
    probe: float,
    max_d: float,
    median_nn: float,
) -> float:
    """Preferred framework distance: near-atom shell inside the probe/max window."""
    if median_nn > 0.0:
        target = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
    else:
        target = 0.5 * (probe + max_d)
    return float(np.clip(target, probe, max_d))


def _scores(
    nn: np.ndarray,
    *,
    probe: float,
    max_d: float,
    median_nn: float,
) -> np.ndarray:
    """Peak at the preferred near-atom shell for every material type."""
    target = _shell_target(probe, max_d, median_nn)
    return -np.abs(nn - target)


def _image_offsets_for_radius(
    cell: np.ndarray, pbc: np.ndarray, radius: float
) -> list[np.ndarray] | None:
    if not np.any(pbc) or not cell_has_volume(cell):
        return None
    return _periodic_image_offsets(
        np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool), float(radius)
    )


def _local_max_mask(
    points: np.ndarray,
    scores: np.ndarray,
    radius: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    n = len(points)
    if n <= 1:
        return np.ones(n, dtype=bool)
    offsets = _image_offsets_for_radius(cell, pbc, radius)
    pairs = _pbc_merge_pair_set(points, float(radius), image_offsets=offsets)
    keep = np.ones(n, dtype=bool)
    for i, j in pairs:
        if scores[i] < scores[j] - 1e-12:
            keep[i] = False
        elif scores[j] < scores[i] - 1e-12:
            keep[j] = False
    return keep


def _nms(
    vertices: np.ndarray,
    nn: np.ndarray,
    scores: np.ndarray,
    merge_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return vertices, nn
    offsets = _image_offsets_for_radius(cell, pbc, merge_r)
    if offsets is None:
        tree = KDTree(vertices)

        def query_ball(pt: np.ndarray) -> list[int]:
            return tree.query_ball_point(pt, r=float(merge_r))

    else:
        n = len(vertices)
        expanded = np.vstack([vertices + off for off in offsets])
        tree = KDTree(expanded)

        def query_ball(pt: np.ndarray) -> list[int]:
            hits = tree.query_ball_point(pt, r=float(merge_r))
            return list({int(h) % n for h in hits})

    suppressed = np.zeros(len(vertices), dtype=bool)
    accepted: list[int] = []
    for i in np.argsort(-scores):
        ii = int(i)
        if suppressed[ii]:
            continue
        accepted.append(ii)
        for j in query_ball(vertices[ii]):
            if j != ii:
                suppressed[j] = True
    idx = np.asarray(accepted, dtype=int)
    return vertices[idx], nn[idx]


def _bin_prethin(
    vertices: np.ndarray,
    nn: np.ndarray,
    *,
    probe: float,
    max_d: float,
    median_nn: float,
    bin_size: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """O(N) spatial hash: keep the best-scoring point per bin (PBC-aware)."""
    if len(vertices) == 0:
        return vertices, nn
    h = max(float(bin_size), 1e-6)
    sc = _scores(
        nn,
        probe=probe,
        max_d=max_d,
        median_nn=median_nn,
    )
    keys: np.ndarray
    if np.any(pbc) and cell_has_volume(cell):
        frac = _cart_to_frac(vertices, cell)
        spacings = np.linalg.norm(cell, axis=1)
        dfrac = np.maximum(h / np.maximum(spacings, 1e-12), 1e-9)
        keys = np.empty((len(vertices), 3), dtype=np.int64)
        pbc_arr = np.asarray(pbc, dtype=bool)
        for dim in range(3):
            f = frac[:, dim]
            if bool(pbc_arr[dim]):
                f = np.mod(f, 1.0)
                n_bins = max(1, int(np.ceil(1.0 / dfrac[dim] - 1e-12)))
                keys[:, dim] = np.floor(f / dfrac[dim]).astype(np.int64) % n_bins
            else:
                keys[:, dim] = np.floor(f / dfrac[dim]).astype(np.int64)
    else:
        keys = np.floor(vertices / h).astype(np.int64)
    # Stable: first occurrence of each unique key after sorting by score desc.
    order = np.argsort(-sc, kind="mergesort")
    keys_sorted = keys[order]
    _, first = np.unique(keys_sorted, axis=0, return_index=True)
    keep = order[first]
    keep.sort()
    return vertices[keep], nn[keep]


def _thin(
    vertices: np.ndarray,
    nn: np.ndarray,
    *,
    probe: float,
    max_d: float,
    median_nn: float,
    neighbour_r: float,
    merge_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertices) == 0:
        return vertices, nn
    # Dense frameworks can leave 1e5+ accessible points; hash-bin before NMS.
    if len(vertices) > 2 * _ADAPTIVE_GRID_MAX_CANDIDATES:
        vertices, nn = _bin_prethin(
            vertices,
            nn,
            probe=probe,
            max_d=max_d,
            median_nn=median_nn,
            bin_size=max(merge_r, neighbour_r),
            cell=cell,
            pbc=pbc,
        )
    sc = _scores(
        nn,
        probe=probe,
        max_d=max_d,
        median_nn=median_nn,
    )
    mask = _local_max_mask(vertices, sc, neighbour_r, cell, pbc)
    if not np.any(mask):
        mask = np.ones(len(vertices), dtype=bool)
    vertices, nn = vertices[mask], nn[mask]
    sc = _scores(
        nn,
        probe=probe,
        max_d=max_d,
        median_nn=median_nn,
    )
    vertices, nn = _nms(vertices, nn, sc, merge_r, cell, pbc)
    if cap is not None and len(vertices) > cap:
        sc = _scores(
            nn,
            probe=probe,
            max_d=max_d,
            median_nn=median_nn,
        )
        order = np.argsort(-sc)[:cap]
        order.sort()
        vertices, nn = vertices[order], nn[order]
    return vertices, nn


def _chunk_bounds(n: int, n_chunks: int) -> list[tuple[int, int]]:
    n_chunks = max(1, min(int(n_chunks), max(1, n)))
    sizes = [n // n_chunks] * n_chunks
    for i in range(n % n_chunks):
        sizes[i] += 1
    bounds: list[tuple[int, int]] = []
    start = 0
    for size in sizes:
        if size <= 0:
            continue
        bounds.append((start, start + size))
        start += size
    return bounds or [(0, n)]


def _seed_chunk_candidates(
    seeds: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
    hull,
) -> tuple[np.ndarray, np.ndarray]:
    if len(seeds) == 0 or len(offsets) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    pts = (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    return _filter_accessible(pts, tree, probe, max_d, hull=hull)


def _parallel_shell_filter(
    centres: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
    hull,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centre+offset candidates and filter; thread over centre chunks."""
    n = len(centres)
    if n == 0 or len(offsets) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    n_workers = resolve_materialize_workers(n_jobs, n_tasks=n)
    if n_workers == 1 or n == 1:
        return _seed_chunk_candidates(centres, offsets, tree, probe, max_d, hull)

    bounds = _chunk_bounds(n, n_workers)
    if len(bounds) == 1:
        return _seed_chunk_candidates(centres, offsets, tree, probe, max_d, hull)

    def _one(bound: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = bound
        return _seed_chunk_candidates(centres[lo:hi], offsets, tree, probe, max_d, hull)

    with ThreadPoolExecutor(max_workers=len(bounds)) as pool:
        parts = list(pool.map(_one, bounds))
    verts = [p[0] for p in parts if len(p[0])]
    dists = [p[1] for p in parts if len(p[1])]
    if not verts:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    return np.vstack(verts), np.concatenate(dists)


def generate_adaptive_grid_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    material_type: str,
    probe_radius: float,
    max_site_distance: float,
    top_layer_tolerance: float,
    adsorbate: Atoms | None = None,
    grid_spacing_scale: float | None = None,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> tuple[np.ndarray, np.ndarray, AdaptiveGridSpacing, KDTree]:
    """Enumerate adaptive-grid candidates and the accessibility tree used.

    *grid_spacing_scale* sizes the shared catalog (min adsorbate scale across
    competing molecules). *adsorbate* is only used when *grid_spacing_scale*
    is omitted (direct/API tests).
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    probe, max_d = float(probe_radius), float(max_site_distance)
    spacing = adaptive_grid_spacing(
        probe,
        adsorbate,
        grid_spacing_scale=grid_spacing_scale,
        initial_spacing=initial_spacing,
        max_levels=max_levels,
    )
    tree = _accessibility_tree(positions, cell, pbc, max_d)
    empty = (
        np.empty((0, 3), dtype=float),
        np.empty(0, dtype=float),
        spacing,
        tree,
    )
    if len(positions) == 0:
        return empty

    # Dense frameworks: one seed per voxel keeps shells near atoms without
    # exploding candidate count (same atom-centred recipe as slab/NP).
    seed_voxel = None
    if material_type == "porous":
        # Coarser than slab/NP: MOF walls radiate many overlapping shells.
        seed_voxel = max(
            spacing.initial_spacing,
            spacing.characteristic_length,
            float(max_d),
        )
    seeds = positions[
        _seed_indices(
            positions,
            cell,
            material_type,
            float(top_layer_tolerance),
            pbc,
            seed_voxel=seed_voxel,
        )
    ]
    frame = _orthonormal_frame(cell, material_type)
    offsets = _shell_offsets(max_d, spacing.initial_spacing, frame, r_min=probe)
    if len(offsets) == 0 or len(seeds) == 0:
        return empty

    hull = _try_convex_hull(positions) if material_type == "nanoparticle" else None
    raw_verts, raw_nn = _parallel_shell_filter(
        seeds, offsets, tree, probe, max_d, hull, n_jobs
    )
    vertices = _wrap_dedup(raw_verts, cell, pbc)
    if len(vertices) == 0:
        return empty
    # Re-query nn after wrap so distances stay consistent with wrapped coords.
    vertices, nn = _filter_accessible(vertices, tree, probe, max_d, hull=hull)
    if len(vertices) == 0:
        return empty

    median_nn = _framework_median_nn(positions, cell, pbc)

    h = spacing.initial_spacing
    h_target = max(
        _ADAPTIVE_GRID_FINE_SCALE * spacing.characteristic_length,
        _ADAPTIVE_GRID_H_MIN * 0.5,
    )
    for _ in range(spacing.max_levels):
        if len(vertices) == 0 or h <= h_target:
            break
        parents, _ = _thin(
            vertices,
            nn,
            probe=probe,
            max_d=max_d,
            median_nn=median_nn,
            neighbour_r=max(spacing.merge_radius, 0.5 * h),
            merge_r=max(0.5 * h, spacing.merge_radius),
            cell=cell,
            pbc=pbc,
        )
        h2 = 0.5 * h
        # Half-spacing stencil in the same local frame (27 neighbours).
        stencil_local = np.array(
            [
                [dx, dy, dz]
                for dx in (-h2, 0.0, h2)
                for dy in (-h2, 0.0, h2)
                for dz in (-h2, 0.0, h2)
            ],
            dtype=float,
        )
        stencil = stencil_local @ frame
        raw_verts, _ = _parallel_shell_filter(
            parents, stencil, tree, probe, max_d, hull, n_jobs
        )
        vertices = _wrap_dedup(raw_verts, cell, pbc)
        vertices, nn = _filter_accessible(vertices, tree, probe, max_d, hull=hull)
        h = h2

    if len(vertices) == 0:
        return empty

    vertices, nn = _thin(
        vertices,
        nn,
        probe=probe,
        max_d=max_d,
        median_nn=median_nn,
        neighbour_r=max(spacing.merge_radius, spacing.fine_spacing),
        merge_r=spacing.merge_radius,
        cell=cell,
        pbc=pbc,
        cap=_ADAPTIVE_GRID_MAX_CANDIDATES,
    )
    return vertices, nn, spacing, tree

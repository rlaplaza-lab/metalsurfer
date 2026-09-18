"""Atom-centred adaptive Cartesian grid for adsorption-site candidates.

One PBC-aware pipeline for every material: shells around all atoms, accessibility
window, exposure filter (plus slab half-space), iterative refinement, and NMS
toward the near-atom adsorption shell — not free-volume / pore centres.

Spacing may be driven by a shared ``grid_spacing_scale`` (typically the
**minimum** adsorbate characteristic length across competing molecules) so one
catalog serves every adsorbate. Framework median NN floors that scale and the
NMS merge radius so tiny adsorbates cannot pack denser than the surface lattice
resolves.

Classify / cluster / symmetry / placement use the shared enumerator path —
this module only produces candidate vertices.

CPU-parallel stages (seed shells and refine stencils) honour joblib-style
``n_jobs`` via :func:`~metalsurfer.placement._parallel.resolve_materialize_workers`
and a thread pool; greedy PBC NMS stays serial. Work is streamed in chunks so
``n_seeds × n_offsets`` never exceeds ``_ADAPTIVE_GRID_WORK_BUDGET`` per chunk.
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
    _ADAPTIVE_GRID_BIN_PRETHIN,
    _ADAPTIVE_GRID_COORD_NN_FACTOR,
    _ADAPTIVE_GRID_EXPOSURE_STEP,
    _ADAPTIVE_GRID_EXTENT_EPS,
    _ADAPTIVE_GRID_FINE_SCALE,
    _ADAPTIVE_GRID_H_MAX,
    _ADAPTIVE_GRID_H_MIN,
    _ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE,
    _ADAPTIVE_GRID_MAX_LEVELS,
    _ADAPTIVE_GRID_NMS_ATOP_NN_SCALE,
    _ADAPTIVE_GRID_NMS_BRIDGE_NN_SCALE,
    _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE,
    _ADAPTIVE_GRID_NMS_HARD_FLOOR,
    _ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE,
    _ADAPTIVE_GRID_NMS_LENGTH_SCALE,
    _ADAPTIVE_GRID_NMS_SCALE,
    _ADAPTIVE_GRID_SPACING_SCALE,
    _ADAPTIVE_GRID_WORK_BUDGET,
    _ADSORBATE_COVALENT_RADIUS_FALLBACK,
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _VECTOR_NORM_EPS,
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
)

_SOURCE_HINT = "adaptive_grid"
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
    probe_radius: float | None,
    adsorbates: Sequence[Atoms | None],
) -> float:
    """Return the minimum characteristic length across *adsorbates*.

    Used to size one shared adaptive-grid catalog for competing molecules:
    the smallest footprint drives the densest sampling that still covers all
    species. Empty / missing adsorbates are skipped; falls back to *probe_radius*
    (or ``1.0`` when *probe_radius* is ``None``).
    """
    probe = float(probe_radius) if probe_radius is not None else 1.0
    scales = [
        adaptive_grid_characteristic_length(probe, ads)
        for ads in adsorbates
        if ads is not None and len(ads) > 0
    ]
    return float(min(scales)) if scales else probe


def adaptive_grid_spacing(
    probe_radius: float,
    adsorbate: Atoms | None = None,
    *,
    grid_spacing_scale: float | None = None,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
    framework_median_nn: float | None = None,
) -> AdaptiveGridSpacing:
    """Return coarse/fine spacing and refine depth for the adaptive grid.

    Characteristic length is ``max(adsorbate_or_probe_L, c * framework_median_nn)``
    when a framework NN is available, so tiny adsorbates cannot request a denser
    shell than the surface lattice resolves. The same floor applies to the NMS
    merge radius.
    """
    if grid_spacing_scale is not None:
        L = float(grid_spacing_scale)
    else:
        L = adaptive_grid_characteristic_length(probe_radius, adsorbate)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError(
            f"adaptive_grid characteristic length must be finite and > 0, got {L!r}"
        )
    nn = float(framework_median_nn) if framework_median_nn is not None else 0.0
    if np.isfinite(nn) and nn > 0.0:
        L = max(L, _ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE * nn)
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
    merge = max(
        _ADAPTIVE_GRID_NMS_SCALE * h_fine,
        _ADAPTIVE_GRID_NMS_LENGTH_SCALE * L,
        _VORONOI_DEDUP_TOLERANCE,
    )
    if np.isfinite(nn) and nn > 0.0:
        merge = max(merge, _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE * nn)
    return AdaptiveGridSpacing(
        characteristic_length=float(L),
        initial_spacing=float(h0),
        fine_spacing=float(h_fine),
        max_levels=levels,
        merge_radius=float(merge),
    )


def _fractional_voxel_seeds(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    seed_voxel: float,
    idx: np.ndarray,
) -> np.ndarray:
    """One seed per fractional (or Cartesian) voxel."""
    pts = positions[idx]
    if not cell_has_volume(cell):
        keys = np.floor(pts / float(seed_voxel)).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        return np.sort(idx[keep])
    frac = _cart_to_frac(pts, cell)
    spacings = np.linalg.norm(cell, axis=1)
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


def _filter_accessible(
    vertices: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep points in the probe/max window; return vertices, nn, nearest-image xyz."""
    if len(vertices) == 0:
        empty = np.empty((0, 3), dtype=float)
        return empty, np.empty(0, dtype=float), empty
    nn, idx = tree.query(vertices, k=1)
    nn = np.asarray(nn, dtype=float).ravel()
    idx = np.asarray(idx, dtype=int).ravel()
    keep = (nn >= probe) & (nn <= max_d)
    data = np.asarray(tree.data, dtype=float)
    return vertices[keep], nn[keep], data[idx[keep]]


def _exposure_mask(
    vertices: np.ndarray,
    nn: np.ndarray,
    nearest: np.ndarray,
    tree: KDTree,
    *,
    material_type: str,
    cell: np.ndarray,
    step: float = _ADAPTIVE_GRID_EXPOSURE_STEP,
) -> np.ndarray:
    """Keep points that walk into void (and, for slabs, the top half-space)."""
    n = len(vertices)
    if n == 0:
        return np.ones(0, dtype=bool)
    delta = vertices - nearest
    norms = np.linalg.norm(delta, axis=1)
    keep = norms > _VECTOR_NORM_EPS
    uhat = np.zeros_like(delta)
    uhat[keep] = delta[keep] / norms[keep, None]
    stepped = vertices + float(step) * uhat
    nn_step, _ = tree.query(stepped, k=1)
    nn_step = np.asarray(nn_step, dtype=float).ravel()
    keep &= nn_step >= nn - 1e-9
    if material_type == "slab" and cell_has_volume(cell):
        n_hat = _slab_normal(cell)
        keep &= np.einsum("ij,j->i", uhat, n_hat) >= -1e-12
    return keep


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
    groups: np.ndarray | None = None,
    radii: np.ndarray | None = None,
    hard_floor: float = 0.0,
) -> np.ndarray:
    """Keep local score maxima; same-class compete, plus hard-floor cross-class."""
    n = len(points)
    if n <= 1:
        return np.ones(n, dtype=bool)
    pair_r = float(radius)
    if radii is not None and len(radii):
        pair_r = max(pair_r, float(np.max(radii)))
    pair_r = max(pair_r, float(hard_floor))
    offsets = _image_offsets_for_radius(cell, pbc, pair_r)
    pairs = _pbc_merge_pair_set(points, float(pair_r), image_offsets=offsets)
    keep = np.ones(n, dtype=bool)
    hard = float(hard_floor)
    for i, j in pairs:
        same = groups is None or int(groups[i]) == int(groups[j])
        if same:
            r_ij = (
                float(min(radii[i], radii[j]))
                if radii is not None
                else float(radius)
            )
            if not _pair_within_radius(points[i], points[j], r_ij, cell, pbc):
                continue
        elif hard <= 0.0 or not _pair_within_radius(
            points[i], points[j], hard, cell, pbc
        ):
            continue
        if scores[i] < scores[j] - 1e-12:
            keep[i] = False
        elif scores[j] < scores[i] - 1e-12:
            keep[j] = False
    return keep


def _pair_within_radius(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> bool:
    """Return True if *a* and *b* are within *radius* under material PBC."""
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    if np.any(pbc_arr) and cell_has_volume(cell):
        frac = delta @ np.linalg.inv(cell)
        frac = frac - np.round(frac) * pbc_arr.astype(float)
        delta = frac @ cell
    return float(np.linalg.norm(delta)) <= float(radius) + 1e-12


def _class_merge_radii(
    groups: np.ndarray,
    *,
    base_merge: float,
    median_nn: float,
) -> np.ndarray:
    """Per-point same-class merge radii (atop / bridge / hollow / pore scales)."""
    # Index 0 unused; 1=atop, 2=bridge, 3=hollow, 4=pore (pore uses hollow scale).
    scales = (
        0.0,
        float(_ADAPTIVE_GRID_NMS_ATOP_NN_SCALE),
        float(_ADAPTIVE_GRID_NMS_BRIDGE_NN_SCALE),
        float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE),
        float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE),
    )
    nn = float(median_nn) if np.isfinite(median_nn) and median_nn > 0.0 else 0.0
    out = np.empty(len(groups), dtype=float)
    base = float(base_merge)
    for i, g in enumerate(groups):
        gi = int(g)
        scale = scales[gi] if 1 <= gi < len(scales) else scales[3]
        out[i] = max(base, scale * nn) if nn > 0.0 else base
    return out


def _nms(
    vertices: np.ndarray,
    nn: np.ndarray,
    scores: np.ndarray,
    merge_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    groups: np.ndarray | None = None,
    radii: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy NMS; same-class merge plus absolute hard-floor cross-class dedup."""
    if len(vertices) == 0:
        return vertices, nn
    if radii is None:
        radii = np.full(len(vertices), float(merge_r), dtype=float)
    hard = float(_ADAPTIVE_GRID_NMS_HARD_FLOOR)
    max_r = float(max(float(merge_r), float(np.max(radii)), hard))
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
    for i in np.argsort(-scores):
        ii = int(i)
        if suppressed[ii]:
            continue
        accepted.append(ii)
        g_i = int(groups[ii]) if groups is not None else None
        r_i = float(radii[ii])
        for j in query_ball(vertices[ii], max(r_i, hard)):
            if j == ii:
                continue
            same = g_i is None or int(groups[j]) == g_i
            if same:
                if _pair_within_radius(vertices[ii], vertices[j], r_i, cell, pbc):
                    suppressed[j] = True
            elif hard > 0.0 and _pair_within_radius(
                vertices[ii], vertices[j], hard, cell, pbc
            ):
                suppressed[j] = True
    idx = np.asarray(accepted, dtype=int)
    return vertices[idx], nn[idx]


def _provisional_coordination_count(
    vertices: np.ndarray,
    tree: KDTree,
    nn: np.ndarray,
) -> np.ndarray:
    """1 / 2 / 3+ coordinating neighbours within ``COORD_NN_FACTOR * nn``."""
    n = len(vertices)
    if n == 0:
        return np.empty(0, dtype=int)
    k = min(4, len(np.asarray(tree.data)))
    if k <= 1:
        return np.ones(n, dtype=int)
    dists, _ = tree.query(vertices, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    thresh = np.asarray(nn, dtype=float).reshape(-1, 1) * float(
        _ADAPTIVE_GRID_COORD_NN_FACTOR
    )
    counts = np.sum(dists <= thresh, axis=1).astype(int)
    return np.clip(counts, 1, 3)


def _nms_pass(
    vertices: np.ndarray,
    nn: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    merge_r: float,
    median_nn: float,
    neighbour_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """One local-max + same-class NMS pass."""
    radii = _class_merge_radii(groups, base_merge=merge_r, median_nn=median_nn)
    local_r = np.maximum(radii, float(_ADAPTIVE_GRID_NMS_HARD_FLOOR))
    mask = _local_max_mask(
        vertices,
        scores,
        max(float(neighbour_r), float(_ADAPTIVE_GRID_NMS_HARD_FLOOR)),
        cell,
        pbc,
        groups=groups,
        radii=local_r,
        hard_floor=float(_ADAPTIVE_GRID_NMS_HARD_FLOOR),
    )
    if not np.any(mask):
        mask = np.ones(len(vertices), dtype=bool)
    vertices, nn, scores, groups, radii = (
        vertices[mask],
        nn[mask],
        scores[mask],
        groups[mask],
        radii[mask],
    )
    return _nms(
        vertices, nn, scores, merge_r, cell, pbc, groups=groups, radii=radii
    )


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
    tree: KDTree,
) -> tuple[np.ndarray, np.ndarray]:
    """Same-class NMS by provisional coordination (1/2/3+).

    Cross-type peaks may sit close; :func:`dedupe_adaptive_sites_within_type`
    removes same-label near-duplicates after shared classification.
    """
    if len(vertices) == 0:
        return vertices, nn
    if len(vertices) > _ADAPTIVE_GRID_BIN_PRETHIN:
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
    sc = _scores(nn, probe=probe, max_d=max_d, median_nn=median_nn)
    groups = _provisional_coordination_count(vertices, tree, nn)
    return _nms_pass(
        vertices,
        nn,
        sc,
        groups,
        merge_r=merge_r,
        median_nn=median_nn,
        neighbour_r=neighbour_r,
        cell=cell,
        pbc=pbc,
    )


def dedupe_adaptive_sites_within_type(
    sites: list,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    median_nn: float,
    probe_radius: float,
    max_site_distance: float,
) -> list:
    """Greedy within-type NMS after shared classification (adaptive_grid only).

    Bridge and hollow peaks often sit ~0.5 Å apart in the adaptive cloud; once
    both label as hollow they look like duplicates. Deduping *after* typing
    with class-specific radii collapses those without killing true bridges.
    """
    if len(sites) < 2:
        return list(sites)
    scales = {
        "atop": float(_ADAPTIVE_GRID_NMS_ATOP_NN_SCALE),
        "bridge": float(_ADAPTIVE_GRID_NMS_BRIDGE_NN_SCALE),
        "hollow": float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE),
        "pore": float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE),
        "envelope": float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE),
    }
    nn_med = float(median_nn) if np.isfinite(median_nn) and median_nn > 0.0 else 0.0
    target = _shell_target(float(probe_radius), float(max_site_distance), nn_med)
    by_type: dict[str, list[int]] = {}
    for i, s in enumerate(sites):
        by_type.setdefault(str(s.site_type), []).append(i)
    keep_mask = np.zeros(len(sites), dtype=bool)
    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    for stype, idxs in by_type.items():
        if len(idxs) == 1:
            keep_mask[idxs[0]] = True
            continue
        scale = scales.get(stype, float(_ADAPTIVE_GRID_NMS_HOLLOW_NN_SCALE))
        merge_r = max(0.35, scale * nn_med) if nn_med > 0.0 else 0.5
        xyz = np.asarray([sites[i].xyz for i in idxs], dtype=float)
        nn_vals = np.asarray(
            [
                float(sites[i].nn_distance)
                if sites[i].nn_distance is not None
                else target
                for i in idxs
            ],
            dtype=float,
        )
        scores = -np.abs(nn_vals - target)
        groups = np.ones(len(idxs), dtype=int)
        radii = np.full(len(idxs), float(merge_r), dtype=float)
        v_keep, _ = _nms(
            xyz,
            nn_vals,
            scores,
            float(merge_r),
            cell_arr,
            pbc_arr,
            groups=groups,
            radii=radii,
        )
        tree = KDTree(xyz)
        for pt in v_keep:
            j = int(tree.query(pt, k=1)[1])
            keep_mask[idxs[j]] = True
    return [s for i, s in enumerate(sites) if keep_mask[i]]


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
    sc = _scores(nn, probe=probe, max_d=max_d, median_nn=median_nn)
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
    order = np.argsort(-sc, kind="mergesort")
    keys_sorted = keys[order]
    _, first = np.unique(keys_sorted, axis=0, return_index=True)
    keep = order[first]
    keep.sort()
    return vertices[keep], nn[keep]


def _work_chunk_bounds(n_seeds: int, n_offsets: int) -> list[tuple[int, int]]:
    """Split seeds so each chunk's cartesian product stays under the work budget."""
    if n_seeds <= 0:
        return []
    max_seeds = max(1, _ADAPTIVE_GRID_WORK_BUDGET // max(1, int(n_offsets)))
    return [
        (start, min(start + max_seeds, n_seeds))
        for start in range(0, n_seeds, max_seeds)
    ]


def _seed_chunk_candidates(
    seeds: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
    *,
    material_type: str,
    cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(seeds) == 0 or len(offsets) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    pts = (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    verts, nn, nearest = _filter_accessible(pts, tree, probe, max_d)
    if len(verts) == 0:
        return verts, nn
    keep = _exposure_mask(
        verts, nn, nearest, tree, material_type=material_type, cell=cell
    )
    return verts[keep], nn[keep]


def _parallel_shell_filter(
    centres: np.ndarray,
    offsets: np.ndarray,
    tree: KDTree,
    probe: float,
    max_d: float,
    *,
    material_type: str,
    cell: np.ndarray,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build centre+offset candidates and filter; chunk by work budget, thread over chunks."""
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
            probe,
            max_d,
            material_type=material_type,
            cell=cell,
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


def generate_adaptive_grid_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    material_type: str,
    probe_radius: float,
    max_site_distance: float,
    adsorbate: Atoms | None = None,
    grid_spacing_scale: float | None = None,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
    n_jobs: int = _DEFAULT_N_JOBS,
) -> tuple[np.ndarray, np.ndarray, AdaptiveGridSpacing, KDTree]:
    """Enumerate adaptive-grid candidates and the accessibility tree used.

    *grid_spacing_scale* sizes the shared catalog (min adsorbate scale across
    competing molecules). *adsorbate* is only used when *grid_spacing_scale*
    is omitted (direct/API tests). Every framework atom is a seed; shell work
    is streamed in RAM-capped chunks.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    probe, max_d = float(probe_radius), float(max_site_distance)
    median_nn = _framework_median_nn(positions, cell, pbc)
    spacing = adaptive_grid_spacing(
        probe,
        adsorbate,
        grid_spacing_scale=grid_spacing_scale,
        initial_spacing=initial_spacing,
        max_levels=max_levels,
        framework_median_nn=median_nn,
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

    offsets = _shell_offsets(max_d, spacing.initial_spacing, r_min=probe)
    if len(offsets) == 0:
        return empty

    raw_verts, raw_nn = _parallel_shell_filter(
        positions,
        offsets,
        tree,
        probe,
        max_d,
        material_type=material_type,
        cell=cell,
        n_jobs=n_jobs,
    )
    vertices = _wrap_dedup(raw_verts, cell, pbc)
    if len(vertices) == 0:
        return empty
    vertices, nn, nearest = _filter_accessible(vertices, tree, probe, max_d)
    if len(vertices) == 0:
        return empty
    keep = _exposure_mask(
        vertices, nn, nearest, tree, material_type=material_type, cell=cell
    )
    vertices, nn = vertices[keep], nn[keep]
    if len(vertices) == 0:
        return empty

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
            tree=tree,
        )
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
            parents,
            stencil,
            tree,
            probe,
            max_d,
            material_type=material_type,
            cell=cell,
            n_jobs=n_jobs,
        )
        vertices = _wrap_dedup(raw_verts, cell, pbc)
        vertices, nn, nearest = _filter_accessible(vertices, tree, probe, max_d)
        if len(vertices) == 0:
            break
        keep = _exposure_mask(
            vertices, nn, nearest, tree, material_type=material_type, cell=cell
        )
        vertices, nn = vertices[keep], nn[keep]
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
        tree=tree,
    )
    return vertices, nn, spacing, tree

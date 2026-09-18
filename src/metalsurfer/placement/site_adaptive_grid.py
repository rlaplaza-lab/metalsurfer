"""Atom-centred adaptive Cartesian grid for adsorption-site candidates.

All materials use the same pipeline: shells around seed atoms, accessibility
filtering, iterative refinement, and thinning toward the near-atom adsorption
shell (not free-volume / pore centres). Optional adsorbate geometry scales
sampling density only; the probe/max window stays substrate-derived.
"""

from __future__ import annotations

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
    _VORONOI_DEDUP_TOLERANCE,
)
from .geometry import _classify_molecule_shape
from .occupancy import incoming_inplane_radius
from .site_coords import (
    _build_periodic_images,
    _deduplicate_points,
    _mean_covalent_radius,
    _wrap_cartesian,
    top_layer_mask_by_normal,
)
from .site_np import (
    _convex_hull_surface_mask,
    _outside_convex_hull_mask,
    _try_convex_hull,
)

_SOURCE_HINT = "adaptive_grid"


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
    """Molecule size scale for grid density, capped by *probe_radius*."""
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
    return float(min(max(length, eps), probe))


def adaptive_grid_spacing(
    probe_radius: float,
    adsorbate: Atoms | None = None,
    *,
    initial_spacing: float | None = None,
    max_levels: int | None = None,
) -> AdaptiveGridSpacing:
    """Return coarse/fine spacing and refine depth for the adaptive grid."""
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


def _seed_indices(
    positions: np.ndarray,
    cell: np.ndarray,
    material_type: str,
    top_layer_tolerance: float,
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
        keys = np.floor(positions / float(seed_voxel)).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        idx = np.sort(idx[keep])
    return idx


def _wrap_dedup(points: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points
    inv = np.linalg.inv(cell) if np.any(pbc) else None
    wrapped = _wrap_cartesian(points, cell, pbc, inv_cell=inv)
    return wrapped[
        _deduplicate_points(wrapped, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    ]


def _atom_grid(
    seeds: np.ndarray,
    radius: float,
    spacing: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    h = float(spacing)
    if h <= 0.0 or radius <= 0.0 or len(seeds) == 0:
        return np.empty((0, 3), dtype=float)
    n = int(np.ceil(radius / h))
    coords = np.arange(-n, n + 1, dtype=float) * h
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    r = np.linalg.norm(offsets, axis=1)
    offsets = offsets[(r > 1e-12) & (r <= float(radius) + 1e-9)]
    if len(offsets) == 0:
        return np.empty((0, 3), dtype=float)
    return _wrap_dedup(
        (seeds[:, None, :] + offsets[None, :, :]).reshape(-1, 3), cell, pbc
    )


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


def _shell_target(probe: float, max_d: float, median_nn: float) -> float:
    """Preferred framework distance: near-atom shell, inside the probe/max window."""
    if median_nn > 0.0:
        target = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
    else:
        target = 0.5 * (probe + max_d)
    return float(np.clip(target, probe, max_d))


def _scores(
    nn: np.ndarray, *, probe: float, max_d: float, median_nn: float
) -> np.ndarray:
    """Peak at the near-atom shell for every material type."""
    target = _shell_target(probe, max_d, median_nn)
    return -np.abs(nn - target)


def _local_max_mask(
    points: np.ndarray, scores: np.ndarray, radius: float
) -> np.ndarray:
    n = len(points)
    if n <= 1:
        return np.ones(n, dtype=bool)
    keep = np.ones(n, dtype=bool)
    for i, j in KDTree(points).query_pairs(r=float(radius)):
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
    tree = KDTree(vertices)
    suppressed = np.zeros(len(vertices), dtype=bool)
    accepted: list[int] = []
    for i in np.argsort(-scores):
        ii = int(i)
        if suppressed[ii]:
            continue
        accepted.append(ii)
        for j in tree.query_ball_point(vertices[ii], r=float(merge_r)):
            if j != ii:
                suppressed[j] = True
    idx = np.asarray(accepted, dtype=int)
    verts, dists = vertices[idx], nn[idx]
    if np.any(pbc) and len(verts) > 1:
        keep = _deduplicate_points(verts, float(merge_r), cell=cell, pbc=pbc)
        verts, dists = verts[keep], dists[keep]
    return verts, dists


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
    sc = _scores(nn, probe=probe, max_d=max_d, median_nn=median_nn)
    mask = _local_max_mask(vertices, sc, neighbour_r)
    if not np.any(mask):
        mask = np.ones(len(vertices), dtype=bool)
    vertices, nn = vertices[mask], nn[mask]
    sc = _scores(nn, probe=probe, max_d=max_d, median_nn=median_nn)
    vertices, nn = _nms(vertices, nn, sc, merge_r, cell, pbc)
    if cap is not None and len(vertices) > cap:
        sc = _scores(nn, probe=probe, max_d=max_d, median_nn=median_nn)
        order = np.argsort(-sc)[:cap]
        order.sort()
        vertices, nn = vertices[order], nn[order]
    return vertices, nn


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
    initial_spacing: float | None = None,
    max_levels: int | None = None,
) -> tuple[np.ndarray, np.ndarray, AdaptiveGridSpacing, KDTree]:
    """Enumerate adaptive-grid candidates and the accessibility tree used."""
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    probe, max_d = float(probe_radius), float(max_site_distance)
    spacing = adaptive_grid_spacing(
        probe, adsorbate, initial_spacing=initial_spacing, max_levels=max_levels
    )
    tree = (
        KDTree(positions)
        if not np.any(pbc) or not cell_has_volume(cell)
        else KDTree(
            _build_periodic_images(
                positions, cell, pbc, margin=max_d + _VORONOI_DEDUP_TOLERANCE
            )
        )
    )
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
        seed_voxel = max(
            spacing.initial_spacing,
            spacing.characteristic_length,
            0.5 * max_d,
        )
    seeds = positions[
        _seed_indices(
            positions,
            cell,
            material_type,
            float(top_layer_tolerance),
            seed_voxel=seed_voxel,
        )
    ]
    candidates = _atom_grid(seeds, max_d, spacing.initial_spacing, cell, pbc)
    if len(candidates) == 0:
        return empty

    hull = _try_convex_hull(positions) if material_type == "nanoparticle" else None
    vertices, nn = _filter_accessible(candidates, tree, probe, max_d, hull=hull)

    local = KDTree(positions)
    if len(positions) >= 2:
        d_nn, _ = local.query(positions, k=2)
        median_nn = float(np.median(np.asarray(d_nn, dtype=float)[:, 1]))
    else:
        median_nn = probe

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
        coords = np.array([-h2, 0.0, h2])
        xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
        offsets = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        expanded = _wrap_dedup(
            (parents[:, None, :] + offsets[None, :, :]).reshape(-1, 3), cell, pbc
        )
        vertices, nn = _filter_accessible(expanded, tree, probe, max_d, hull=hull)
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

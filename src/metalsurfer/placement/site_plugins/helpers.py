"""Shared candidate-pipeline helpers used by site generator plugins."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree

from ..._utils import cell_has_volume
from .._constants import (
    _BOUNDING_BOX_CELL_PAD_ANGSTROM,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
)
from ..site_coords import (
    _build_periodic_images,
    _deduplicate_points,
    _height_along_slab_normal,
    _pbc_merge_pair_set,
    _periodic_image_offsets,
    _project_to_slab_plane,
    top_layer_mask_by_normal,
)

_EMPTY_ATOM_INDICES: tuple[int, ...] = ()


@dataclass
class PlanarWidenScratch:
    """Topology Qhull objects retained for planar-slab auto-widen reuse."""

    planar_skip_voronoi: bool = False
    primary_delaunay: Delaunay | None = None
    exp_xy: np.ndarray | None = None
    exp_origin: list[int] | None = None
    exp_tri: Delaunay | None = None


def merge_dedup_site_arrays(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    new_vertices: np.ndarray,
    new_dists: np.ndarray,
    new_sources: list[str],
    *,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
    atom_indices: list[tuple[int, ...]] | None = None,
    new_atom_indices: list[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, ...]]]:
    """Append *new_* sites without collapsing already-unique existing sites.

    The existing unique set is frozen: new points are first deduplicated among
    themselves, then dropped when within ``_VORONOI_DEDUP_TOLERANCE`` of any
    existing point (PBC-aware). A new midpoint near two old sites therefore
    cannot merge those old representatives into one.
    """
    n_old = len(vertices)
    n_new = len(new_vertices)
    old_atoms = (
        list(atom_indices)
        if atom_indices is not None
        else [_EMPTY_ATOM_INDICES for _ in range(n_old)]
    )
    new_atoms = (
        list(new_atom_indices)
        if new_atom_indices is not None
        else [_EMPTY_ATOM_INDICES for _ in range(n_new)]
    )
    if n_new == 0:
        return vertices, nn_dists, source_hints, old_atoms
    pbc_arr = np.asarray(pbc, dtype=bool)
    if n_old == 0:
        keep_new = _deduplicate_points(
            new_vertices, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc_arr
        )
        kept = np.nonzero(keep_new)[0]
        return (
            new_vertices[keep_new],
            new_dists[keep_new],
            [new_sources[i] for i in kept],
            [new_atoms[i] for i in kept],
        )

    keep_new = _deduplicate_points(
        new_vertices, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc_arr
    )
    cand_verts = new_vertices[keep_new]
    cand_dists = new_dists[keep_new]
    kept_new_idx = np.nonzero(keep_new)[0]
    cand_sources = [new_sources[i] for i in kept_new_idx]
    cand_atoms = [new_atoms[i] for i in kept_new_idx]
    if len(cand_verts) == 0:
        return vertices, nn_dists, source_hints, old_atoms

    cell_arr = np.asarray(cell, dtype=float)
    image_offsets = None
    if np.any(pbc_arr):
        image_offsets = _periodic_image_offsets(
            cell_arr, pbc_arr, _VORONOI_DEDUP_TOLERANCE
        )
    # Pair set across old+new; drop a candidate if it merges with any old site.
    combined = np.vstack([vertices, cand_verts])
    merge_set = _pbc_merge_pair_set(
        combined, _VORONOI_DEDUP_TOLERANCE, image_offsets=image_offsets
    )
    collide = np.zeros(len(cand_verts), dtype=bool)
    for a, b in merge_set:
        if a < n_old <= b:
            collide[b - n_old] = True
        elif b < n_old <= a:
            collide[a - n_old] = True
    accept = ~collide
    if not np.any(accept):
        return vertices, nn_dists, source_hints, old_atoms

    accepted_idx = np.nonzero(accept)[0]
    return (
        np.vstack([vertices, cand_verts[accept]]),
        np.concatenate([nn_dists, cand_dists[accept]]),
        source_hints + [cand_sources[i] for i in accepted_idx],
        old_atoms + [cand_atoms[i] for i in accepted_idx],
    )


def median_nn_or_fallback(
    nn_dists: np.ndarray,
    *,
    reference_positions: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> float:
    """Median nearest-neighbour distance, or top-layer / covalent fallback.

    When *nn_dists* is empty (e.g. planar slabs that skip Voronoi), prefer the
    median MIC nearest-neighbour spacing of *reference_positions* (typically the
    top layer) over a fixed covalent-scale constant.
    """
    if len(nn_dists) > 0:
        return float(np.median(nn_dists))
    if (
        reference_positions is not None
        and len(reference_positions) >= 2
        and cell is not None
        and pbc is not None
    ):
        pts = np.asarray(reference_positions, dtype=float)
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
        else:
            tree = KDTree(pts)
            nn_d, _ = tree.query(pts, k=2)
            return float(np.median(np.asarray(nn_d, dtype=float)[:, 1]))
    return _VORONOI_MAX_DISTANCE_COVALENT_SCALE * _SURFACE_COVALENT_RADIUS_FALLBACK


def periodic_accessibility_tree(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    max_distance: float,
) -> KDTree:
    """KDTree over periodic images, for candidate-to-framework distance gating.

    A plain ``KDTree(positions)`` reports the *in-cell* nearest-neighbour
    distance, which overestimates the true minimum-image distance for
    candidates near an a/b boundary. The image margin covers ``max_distance``
    so any candidate that would pass the accessibility window is found.
    """
    if not np.any(pbc) or not cell_has_volume(cell):
        return KDTree(positions)
    margin = float(max_distance) + _VORONOI_DEDUP_TOLERANCE
    return KDTree(_build_periodic_images(positions, cell, pbc, margin=margin))


def top_layer_is_planar_from_arrays(
    positions: np.ndarray,
    cell: np.ndarray,
    top_layer_tolerance: float = _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    *,
    top_mask: np.ndarray | None = None,
) -> bool:
    """Check whether the topmost atomic layer of *positions* is approximately flat.

    The fit is done in an orientation-aware slab coordinate system rather than
    assuming the slab normal is Cartesian z. *top_mask* (from
    :func:`top_layer_mask_by_normal`) may be supplied when the caller already
    computed it, avoiding a redundant height scan.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    if top_mask is None:
        top_mask = top_layer_mask_by_normal(positions, cell, float(top_layer_tolerance))
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) == 0:
        return False
    top_pos = positions[top_indices]
    h = _height_along_slab_normal(top_pos, cell)
    if len(top_indices) < 3:
        return float(np.var(h)) < z_variance_threshold
    xy = _project_to_slab_plane(top_pos, cell)
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    coeffs, _residuals, rank, _ = np.linalg.lstsq(A, h, rcond=None)
    if rank < 3:
        return float(np.var(h)) < z_variance_threshold
    h_pred = A @ coeffs
    return float(np.var(h - h_pred)) < z_variance_threshold


def is_top_layer_planar(
    slab: Atoms,
    top_layer_tolerance: float = _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    *,
    top_mask: np.ndarray | None = None,
) -> bool:
    """Check whether the topmost atomic layer is approximately flat."""
    return top_layer_is_planar_from_arrays(
        slab.get_positions(),
        np.asarray(slab.get_cell(), dtype=float),
        top_layer_tolerance,
        z_variance_threshold,
        top_mask=top_mask,
    )


def bounding_box_cell(
    positions: np.ndarray,
    pad: float = _BOUNDING_BOX_CELL_PAD_ANGSTROM,
) -> np.ndarray:
    """Orthorhombic cell spanning atomic positions plus padding."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo + pad, pad)
    return np.diag(span)

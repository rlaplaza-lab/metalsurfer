"""Shared candidate-pipeline helpers used by site generator plugins."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree

from ..._utils import cell_has_volume
from .._constants import (
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _BOUNDING_BOX_CELL_PAD_ANGSTROM,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
)
from ..site_coords import (
    _build_periodic_images,
    _deduplicate_points,
    _height_along_slab_normal,
    _minimum_image_cartesian_delta,
    _pbc_merge_pair_set,
    _periodic_image_offsets,
    _project_to_slab_plane,
    _shift_along_slab_normal,
    _slab_normal,
    _wrap_cartesian,
    top_layer_mask_by_normal,
)
from ..site_np import (
    _convex_hull_surface_mask,
    _surface_atom_normals,
    _try_convex_hull,
)
from .base import slice_candidate_arrays

_EMPTY_ATOM_INDICES: tuple[int, ...] = ()

logger = logging.getLogger(__name__)


@dataclass
class PlanarWidenScratch:
    """Topology Qhull objects retained for planar-slab auto-widen reuse."""

    planar_skip_voronoi: bool = False
    primary_delaunay: Delaunay | None = None
    exp_xy: np.ndarray | None = None
    exp_origin: list[int] | None = None
    exp_tri: Delaunay | None = None


def _slice_optional_enrichment(
    arr: np.ndarray | None,
    mask_or_idx: np.ndarray | slice,
    *,
    n_expected: int,
) -> np.ndarray | None:
    """Slice an optional enrichment array by a boolean mask, index list, or slice."""
    if arr is None:
        return None
    values = np.asarray(arr)
    if len(values) != n_expected:
        raise ValueError(f"enrichment length {len(values)} != expected {n_expected}")
    return values[mask_or_idx]


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
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
    new_normals: np.ndarray | None = None,
    new_clearances: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[tuple[int, ...]],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Append *new_* sites without collapsing already-unique existing sites.

    The existing unique set is frozen: new points are first deduplicated among
    themselves, then dropped when within ``_VORONOI_DEDUP_TOLERANCE`` of any
    existing point (PBC-aware). A new midpoint near two old sites therefore
    cannot merge those old representatives into one.

    Optional *normals* / *clearances* (and matching ``new_*`` arrays) are
    preserved for kept sites. A channel that is missing or length-mismatched
    on either side is dropped (returned as ``None``).
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

    def _combine_channel(
        old: np.ndarray | None,
        new: np.ndarray | None,
        *,
        n_old_expected: int,
        n_new_expected: int,
        new_mask: np.ndarray | None,
        stack: bool,
    ) -> np.ndarray | None:
        """Preserve enrichment when channels align; drop on length/presence mismatch."""
        if old is None and new is None:
            return None
        if n_old_expected == 0:
            if new is None or len(np.asarray(new)) != n_new_expected:
                return None
            return _slice_optional_enrichment(
                new,
                new_mask if new_mask is not None else np.s_[:],
                n_expected=n_new_expected,
            )
        if n_new_expected == 0 or new_mask is None or not np.any(new_mask):
            if old is None or len(np.asarray(old)) != n_old_expected:
                return None
            return np.asarray(old)
        if old is None or new is None:
            return None
        if (
            len(np.asarray(old)) != n_old_expected
            or len(np.asarray(new)) != n_new_expected
        ):
            return None
        new_kept = np.asarray(new)[new_mask]
        if stack:
            return np.vstack([np.asarray(old), new_kept])
        return np.concatenate([np.asarray(old), new_kept])

    def _old_enrichment() -> tuple[np.ndarray | None, np.ndarray | None]:
        return (
            _combine_channel(
                normals,
                None,
                n_old_expected=n_old,
                n_new_expected=0,
                new_mask=None,
                stack=True,
            ),
            _combine_channel(
                clearances,
                None,
                n_old_expected=n_old,
                n_new_expected=0,
                new_mask=None,
                stack=False,
            ),
        )

    if n_new == 0:
        out_n, out_c = _old_enrichment()
        return vertices, nn_dists, source_hints, old_atoms, out_n, out_c

    pbc_arr = np.asarray(pbc, dtype=bool)
    if n_old == 0:
        keep_new = _deduplicate_points(
            new_vertices, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc_arr
        )
        kept = np.nonzero(keep_new)[0]
        out_n = _combine_channel(
            None,
            new_normals,
            n_old_expected=0,
            n_new_expected=n_new,
            new_mask=keep_new,
            stack=True,
        )
        out_c = _combine_channel(
            None,
            new_clearances,
            n_old_expected=0,
            n_new_expected=n_new,
            new_mask=keep_new,
            stack=False,
        )
        return (
            new_vertices[keep_new],
            new_dists[keep_new],
            [new_sources[i] for i in kept],
            [new_atoms[i] for i in kept],
            out_n,
            out_c,
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
        out_n, out_c = _old_enrichment()
        return vertices, nn_dists, source_hints, old_atoms, out_n, out_c

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
        out_n, out_c = _old_enrichment()
        return vertices, nn_dists, source_hints, old_atoms, out_n, out_c

    accepted_idx = np.nonzero(accept)[0]
    keep_and_accept = np.zeros(n_new, dtype=bool)
    keep_and_accept[kept_new_idx[accept]] = True
    out_n = _combine_channel(
        normals,
        new_normals,
        n_old_expected=n_old,
        n_new_expected=n_new,
        new_mask=keep_and_accept,
        stack=True,
    )
    out_c = _combine_channel(
        clearances,
        new_clearances,
        n_old_expected=n_old,
        n_new_expected=n_new,
        new_mask=keep_and_accept,
        stack=False,
    )
    return (
        np.vstack([vertices, cand_verts[accept]]),
        np.concatenate([nn_dists, cand_dists[accept]]),
        source_hints + [cand_sources[i] for i in accepted_idx],
        old_atoms + [cand_atoms[i] for i in accepted_idx],
        out_n,
        out_c,
    )


def candidate_enrichment_frames(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    atom_indices: list[tuple[int, ...]],
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    material_type: str,
    accessibility_tree: KDTree | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normals, clearances)`` aligned with *vertices*.

    Clearances copy accessibility-gated centre-to-centre ``nn_dists``. Normals
    prefer support-centroid lift when ``atom_indices`` are nonempty; otherwise
    slabs use the cell slab normal and other materials use the direction from
    the nearest framework atom.
    """
    verts = np.asarray(vertices, dtype=float).reshape(-1, 3)
    dists = np.asarray(nn_dists, dtype=float).reshape(-1)
    n = len(verts)
    clearances = dists.copy()
    if n == 0:
        return np.empty((0, 3), dtype=float), clearances

    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    pos = np.asarray(positions, dtype=float)
    use_mic = bool(np.any(pbc_arr)) and cell_has_volume(cell_arr)
    slab_n = _slab_normal(cell_arr) if material_type == "slab" else None
    com = None if slab_n is not None else np.mean(pos, axis=0)
    gate = accessibility_tree if accessibility_tree is not None else KDTree(pos)

    normals = np.zeros((n, 3), dtype=float)
    for i, vert in enumerate(verts):
        support = (
            tuple(int(j) for j in atom_indices[i]) if i < len(atom_indices) else ()
        )
        if support:
            pts = []
            for j in support:
                delta = pos[int(j)] - vert
                if use_mic:
                    delta = _minimum_image_cartesian_delta(delta, cell_arr, pbc_arr)
                pts.append(vert + delta)
            centroid = np.mean(np.asarray(pts, dtype=float), axis=0)
            lift = vert - centroid
            nrm = float(np.linalg.norm(lift))
            if nrm >= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                normals[i] = lift / nrm
                continue
            # Anchor already lies on the support plane (no probe lift).
            if com is not None:
                outward = centroid - com
                on = float(np.linalg.norm(outward))
                if on >= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                    normals[i] = outward / on
                    continue
        if slab_n is not None:
            normals[i] = slab_n
            continue
        _, nn_idx = gate.query(vert.reshape(1, 3), k=1)
        nearest = pos[int(np.asarray(nn_idx).ravel()[0]) % len(pos)]
        delta = vert - nearest
        if use_mic:
            delta = _minimum_image_cartesian_delta(delta, cell_arr, pbc_arr)
        nrm = float(np.linalg.norm(delta))
        if nrm >= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            normals[i] = delta / nrm
        elif slab_n is not None:
            normals[i] = slab_n
        else:
            normals[i] = np.array([0.0, 0.0, 1.0], dtype=float)
    return normals, clearances


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


def apply_slab_height_mask(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    atom_indices: list[tuple[int, ...]],
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    top_layer_tolerance: float,
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[tuple[int, ...]],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Drop candidates below the top-layer height band along the slab normal."""
    if len(vertices) == 0:
        return vertices, nn_dists, source_hints, atom_indices, normals, clearances
    heights = _height_along_slab_normal(positions, cell)
    h_surface = float(np.max(heights))
    nn_margin = (
        float(np.median(nn_dists)) if len(nn_dists) > 0 else float(top_layer_tolerance)
    )
    h_min = h_surface - max(float(top_layer_tolerance), nn_margin)
    keep_mask = _height_along_slab_normal(vertices, cell) >= h_min
    return slice_candidate_arrays(
        vertices,
        nn_dists,
        source_hints,
        atom_indices,
        keep_mask,
        normals=normals,
        clearances=clearances,
    )


def inject_atop_sites(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    material_type: str,
    local_tree: KDTree,
    accessibility_tree: KDTree | None,
    median_nn: float | None,
    slab_top_atom_indices: np.ndarray | None,
    has_topology_atop: bool,
    probe_radius: float,
    max_site_distance: float,
    atom_indices: list[tuple[int, ...]] | None = None,
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[tuple[int, ...]],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Inject atop when topology did not already produce any.

    Uses the same height as the topology generator when *median_nn* is supplied.
    *accessibility_tree* (PBC-aware) gates distances under periodic boundaries.
    When *normals* / *clearances* are supplied, frames for injected atops are
    appended so enrichment stays aligned with the merged catalog.
    """
    atoms = (
        list(atom_indices)
        if atom_indices is not None
        else [_EMPTY_ATOM_INDICES for _ in range(len(vertices))]
    )
    if material_type in ("slab", "nanoparticle") and has_topology_atop:
        return vertices, nn_dists, source_hints, atoms, normals, clearances
    if material_type not in ("slab", "nanoparticle"):
        return vertices, nn_dists, source_hints, atoms, normals, clearances

    if median_nn is None:
        ref = (
            positions[slab_top_atom_indices]
            if material_type == "slab" and slab_top_atom_indices is not None
            else positions
        )
        median_nn = median_nn_or_fallback(
            nn_dists if material_type == "slab" else np.empty(0, dtype=float),
            reference_positions=ref,
            cell=cell,
            pbc=pbc,
        )
    atop_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn

    if material_type == "slab":
        if slab_top_atom_indices is None:
            return vertices, nn_dists, source_hints, atoms, normals, clearances
        top_atom_indices = np.asarray(slab_top_atom_indices, dtype=int)
        atom_normals = None
    else:
        hull = _try_convex_hull(positions)
        if hull is None:
            return vertices, nn_dists, source_hints, atoms, normals, clearances
        top_atom_indices = np.nonzero(_convex_hull_surface_mask(positions, hull=hull))[
            0
        ].astype(int)
        if len(top_atom_indices) == 0:
            return vertices, nn_dists, source_hints, atoms, normals, clearances
        atom_normals = _surface_atom_normals(positions, top_atom_indices, hull)

    candidate_verts: list[np.ndarray] = []
    candidate_probes: list[np.ndarray] = []
    candidate_atom_ids: list[int] = []
    candidate_normal_list: list[np.ndarray] = []
    for li, ai in enumerate(top_atom_indices):
        atom_pos = positions[int(ai)]
        if material_type == "slab":
            probe = _shift_along_slab_normal(atom_pos.reshape(1, 3), cell, atop_height)[
                0
            ]
            if np.any(pbc):
                probe = _wrap_cartesian(probe.reshape(1, 3), cell, pbc)[0]
            n_hat = _slab_normal(cell)
            anchor = atom_pos
            if np.any(pbc):
                anchor = _wrap_cartesian(atom_pos.reshape(1, 3), cell, pbc)[0]
        else:
            assert atom_normals is not None
            n_hat = atom_normals[li]
            if float(np.linalg.norm(n_hat)) < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                continue
            probe = atom_pos + atop_height * n_hat
            anchor = atom_pos
        candidate_verts.append(anchor)
        candidate_probes.append(probe)
        candidate_atom_ids.append(int(ai))
        candidate_normal_list.append(np.asarray(n_hat, dtype=float))

    if not candidate_verts:
        return vertices, nn_dists, source_hints, atoms, normals, clearances

    candidate_arr = np.asarray(candidate_verts, dtype=float)
    probe_arr = np.asarray(candidate_probes, dtype=float)
    gate_tree = (
        accessibility_tree
        if accessibility_tree is not None and np.any(pbc)
        else local_tree
    )
    d_nn_all = np.asarray(gate_tree.query(probe_arr, k=1)[0], dtype=float).ravel()
    keep_acc = (d_nn_all >= float(probe_radius)) & (
        d_nn_all <= float(max_site_distance)
    )
    if not np.any(keep_acc):
        return vertices, nn_dists, source_hints, atoms, normals, clearances

    candidate_arr = candidate_arr[keep_acc]
    candidate_dist_arr = d_nn_all[keep_acc]
    kept_atom_ids = [candidate_atom_ids[i] for i in np.nonzero(keep_acc)[0]]
    candidate_sources = ["atop_injected"] * len(candidate_arr)
    candidate_atoms = [(ai,) for ai in kept_atom_ids]
    candidate_normals = np.asarray(
        [candidate_normal_list[i] for i in np.nonzero(keep_acc)[0]], dtype=float
    )
    candidate_clearances = candidate_dist_arr.copy()

    if normals is None or clearances is None or len(normals) != len(vertices):
        normals, clearances = candidate_enrichment_frames(
            vertices,
            nn_dists,
            atoms,
            positions=positions,
            cell=cell,
            pbc=pbc,
            material_type=material_type,
            accessibility_tree=accessibility_tree,
        )

    n_existing = len(vertices)
    vertices, nn_dists, source_hints, atoms, normals, clearances = (
        merge_dedup_site_arrays(
            vertices,
            nn_dists,
            source_hints,
            candidate_arr,
            candidate_dist_arr,
            candidate_sources,
            cell=cell,
            pbc=pbc,
            atom_indices=atoms,
            new_atom_indices=candidate_atoms,
            normals=normals,
            clearances=clearances,
            new_normals=candidate_normals,
            new_clearances=candidate_clearances,
        )
    )
    n_injected = len(vertices) - n_existing
    logger.debug(
        "Injected %d atop candidate sites (%d total sites)",
        max(n_injected, 0),
        len(vertices),
    )

    return vertices, nn_dists, source_hints, atoms, normals, clearances

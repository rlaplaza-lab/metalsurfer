"""Coordinate frame, PBC images, deduplication, and radii helpers for site detection.

The cell-frame primitives (fractional conversion, wrapping, minimum image, slab
frame) live in :mod:`metalsurfer._geom_pbc` so :mod:`metalsurfer.symmetry` can use
them without importing the ``placement`` package. They are re-exported here under
their historical underscore names for existing call sites.
"""

from collections.abc import Sequence

import numpy as np
from scipy.spatial import KDTree

from .._geom_pbc import (
    cart_to_frac as _cart_to_frac,
)
from .._geom_pbc import (
    frac_to_cart as _frac_to_cart,
)
from .._geom_pbc import (
    height_along_slab_normal as _height_along_slab_normal,
)
from .._geom_pbc import (
    minimum_image_cartesian_delta as _minimum_image_cartesian_delta,
)
from .._geom_pbc import (
    minimum_image_fractional_delta as _minimum_image_fractional_delta,
)
from .._geom_pbc import (
    project_to_slab_plane as _project_to_slab_plane,
)
from .._geom_pbc import (
    reciprocal_plane_spacings as _reciprocal_plane_spacings,
)
from .._geom_pbc import (
    shift_along_slab_normal as _shift_along_slab_normal,
)
from .._geom_pbc import (
    slab_normal as _slab_normal,
)
from .._geom_pbc import (
    slab_outward_normal as _slab_outward_normal,
)
from .._geom_pbc import (
    slab_plane_projectors as _slab_plane_projectors,
)
from .._geom_pbc import (
    slab_vacuum_face_sign as _slab_vacuum_face_sign,
)
from .._geom_pbc import (
    wrap_cartesian as _wrap_cartesian,
)
from .._geom_pbc import (
    wrap_fractional as _wrap_fractional,
)
from .._utils import cell_has_volume
from .._utils import union_find_cluster as _union_find_cluster
from ._constants import (
    _ADAPTIVE_GRID_EXPOSURE_N_STEPS,
    _ADAPTIVE_GRID_EXPOSURE_STEP,
    _EXPOSED_SHADOW_NN_SCALE,
    _PORE_THRESHOLD_COVALENT_SCALE,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _STEP_TERRACE_MAX_GAP_ANGSTROM,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _TOP_LAYER_DEPTH_COVALENT_SCALE,
    _TOP_LAYER_DEPTH_MAX_ANGSTROM,
    _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
)
from ._material import MATERIAL_PBC
from .geometry import _get_covalent_radius, framework_radii_from_symbols

__all__ = [
    # Re-exported cell-frame geometry (see metalsurfer._geom_pbc).
    "_cart_to_frac",
    "_frac_to_cart",
    "_height_along_slab_normal",
    "_minimum_image_cartesian_delta",
    "_minimum_image_fractional_delta",
    "_project_to_slab_plane",
    "_reciprocal_plane_spacings",
    "_shift_along_slab_normal",
    "_slab_normal",
    "_slab_outward_normal",
    "_slab_plane_projectors",
    "_slab_vacuum_face_sign",
    "_wrap_cartesian",
    "_wrap_fractional",
    # Defined here.
    "_build_periodic_images",
    "_deduplicate_points",
    "_derive_top_layer_tolerance",
    "_derive_voronoi_distance_window",
    "_keep_mask_from_clusters",
    "_mean_covalent_radius",
    "_pbc_merge_pair_set",
    "_periodic_image_offsets",
    "_union_find_cluster",
    "_wrapped_slab_fractional",
    "derive_pore_threshold",
    "mic_support_positions",
    "top_layer_mask_by_normal",
]


def _primary_height_band_mask(
    heights: np.ndarray,
    tolerance: float,
    *,
    face_sign: float = 1.0,
) -> np.ndarray:
    """Boolean mask for atoms within *tolerance* of the vacuum-exposed extreme.

    ``face_sign`` is :func:`slab_vacuum_face_sign` (``+1`` max-height, ``-1`` min).
    """
    hout = np.asarray(heights, dtype=float) * float(face_sign)
    return hout >= (float(np.max(hout)) - float(tolerance))


def top_layer_mask_by_normal(
    positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float,
    *,
    include_terrace: bool = True,
) -> np.ndarray:
    """Return mask of atoms on the vacuum-richer exposed slab face.

    For a bottom-anchored slab this is the max-height band
    (``h_max - tolerance``). When more vacuum lies toward ``-a×b``, the
    min-height band is used instead. When *include_terrace* is true and a
    discrete terrace sits immediately inward from that primary band
    (stepped/reconstructed surfaces), that terrace is included once if its
    gap is at most ``_STEP_TERRACE_MAX_GAP_ANGSTROM``. The mask never walks
    deeper into the bulk by another full tolerance.

    Parameters
    ----------
    positions
        Atomic Cartesian positions.
    cell
        Unit-cell matrix.
    tolerance
        Height tolerance for the top layer.
    include_terrace
        When false, return only the primary height band (no stepped-surface
        terrace expansion). Used by freeze policy and alloy top-layer selection.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.size == 0:
        return np.zeros(0, dtype=bool)

    heights = _height_along_slab_normal(positions, cell)
    face_sign = _slab_vacuum_face_sign(positions, cell)
    tol = float(tolerance)
    hout = heights * face_sign
    h_ext = float(np.max(hout))
    primary = _primary_height_band_mask(heights, tol, face_sign=face_sign)
    if not include_terrace:
        return primary

    # Steps already inside the primary band (e.g. Δh ≤ tol) stay in primary.
    inward = ~primary
    if not np.any(inward):
        return primary

    terrace_ext = float(np.max(hout[inward]))
    gap = h_ext - terrace_ext
    if gap <= _STEP_TERRACE_MAX_GAP_ANGSTROM:
        terrace = (hout >= terrace_ext - tol) & (hout <= terrace_ext + tol)
        return primary | terrace
    return primary


# ---------------------------------------------------------------------------
# Periodic image generation
# ---------------------------------------------------------------------------


def _periodic_image_offsets(
    cell: np.ndarray,
    pbc: np.ndarray,
    margin: float,
) -> list[np.ndarray]:
    """Return enough image offsets to cover a Cartesian margin around the cell."""
    if not np.any(pbc):
        return [np.zeros(3, dtype=float)]

    spacings = _reciprocal_plane_spacings(cell)
    ranges: list[list[int]] = []
    for dim in range(3):
        if bool(pbc[dim]):
            spacing = float(spacings[dim])
            if not np.isfinite(spacing) or spacing <= 0.0:
                n_img = 1
            else:
                n_img = max(
                    1, int(np.ceil((margin + _VORONOI_DEDUP_TOLERANCE) / spacing))
                )
            ranges.append(list(range(-n_img, n_img + 1)))
        else:
            ranges.append([0])

    offsets: list[np.ndarray] = []
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                offsets.append(i * cell[0] + j * cell[1] + k * cell[2])
    return offsets


def _build_periodic_images(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    margin: float = 0.0,
    *,
    offsets: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Return extended positions including enough periodic images for *margin*."""
    if offsets is None:
        offsets = _periodic_image_offsets(cell, pbc, margin)
    return np.vstack([positions + off for off in offsets])


# ---------------------------------------------------------------------------
# Deduplication helpers
# ---------------------------------------------------------------------------


def _pbc_merge_pair_set(
    coords: np.ndarray,
    radius: float,
    *,
    image_offsets: list[np.ndarray] | None = None,
) -> set[tuple[int, int]]:
    """Origin-index pairs within *radius*, PBC-aware when *image_offsets* given.

    Shared primitive for every periodic union-find dedup pipeline: expands the
    coordinates by *image_offsets* (when provided), queries a KD-tree, maps the
    expanded indices back with ``% n``, and returns each undirected origin pair
    once.
    """
    pts = np.asarray(coords, dtype=float)
    n = len(pts)
    all_pts = np.vstack([pts + off for off in image_offsets]) if image_offsets else pts
    tree = KDTree(all_pts)
    raw_pairs = tree.query_pairs(r=float(radius), output_type="ndarray")
    merge_set: set[tuple[int, int]] = set()
    for a_exp, b_exp in raw_pairs:
        a = int(a_exp) % n
        b = int(b_exp) % n
        if a == b:
            continue
        merge_set.add((min(a, b), max(a, b)))
    return merge_set


def _keep_mask_from_clusters(n: int, pairs: set[tuple[int, int]]) -> np.ndarray:
    """Boolean keep-mask keeping ``min(component)`` of every union-find cluster."""
    components = _union_find_cluster(n, list(pairs))
    keep = np.zeros(n, dtype=bool)
    for comp in components:
        keep[min(comp)] = True
    return keep


def _deduplicate_points(
    points: np.ndarray,
    tolerance: float,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
    image_offsets: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Return a boolean keep-mask that removes near-duplicate points.

    When *cell* and *pbc* are provided, periodic duplicates across the unit-cell
    boundary are also merged. Pass *image_offsets* to reuse a precomputed
    offset list for the same ``(cell, pbc, tolerance)`` margin.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return np.ones(0, dtype=bool)

    offsets = image_offsets
    if offsets is None and cell is not None and pbc is not None and np.any(pbc):
        offsets = _periodic_image_offsets(
            np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool), tolerance
        )
    merge_set = _pbc_merge_pair_set(pts, tolerance, image_offsets=offsets)
    return _keep_mask_from_clusters(n, merge_set)


def _wrapped_slab_fractional(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Fractional coordinates with *a*/*b* in ``[0, 1)`` and *c* unchanged."""
    return _wrap_fractional(_cart_to_frac(points, cell), MATERIAL_PBC["slab"])


def mic_support_positions(
    vertex: np.ndarray,
    positions: np.ndarray,
    support: np.ndarray | list[int] | tuple[int, ...],
    cell: np.ndarray | None = None,
    pbc: Sequence[bool] | np.ndarray | None = None,
) -> np.ndarray:
    """Support atom coordinates as MIC images of *vertex* (primary if no PBC)."""
    vert = np.asarray(vertex, dtype=float).reshape(3)
    pos = np.asarray(positions, dtype=float)
    raw_idx = np.asarray(support, dtype=int).ravel()
    idx = raw_idx[(raw_idx >= 0) & (raw_idx < len(pos))]
    if idx.size == 0:
        return np.empty((0, 3), dtype=float)
    primary = pos[idx]
    if cell is None or pbc is None:
        return primary
    pbc_arr = np.asarray(pbc, dtype=bool)
    cell_arr = np.asarray(cell, dtype=float)
    if not np.any(pbc_arr) or not cell_has_volume(cell_arr):
        return primary
    return vert + _minimum_image_cartesian_delta(primary - vert, cell_arr, pbc_arr)


def project_anchor_to_support_plane(
    xyz: np.ndarray,
    normal: np.ndarray,
    support_positions: np.ndarray,
) -> np.ndarray:
    """Drop *xyz* onto the max-support plane along *normal*.

    Empty supports and a degenerate normal leave *xyz* unchanged.
    """
    vert = np.asarray(xyz, dtype=float).reshape(3).copy()
    supp = np.asarray(support_positions, dtype=float).reshape(-1, 3)
    if len(supp) == 0:
        return vert
    n_hat = np.asarray(normal, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(n_hat))
    if nrm <= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return vert
    n_unit = np.asarray(n_hat / nrm, dtype=float)
    support_h = float(np.max(supp @ n_unit))
    delta = float(np.dot(vert, n_unit)) - support_h
    return np.asarray(vert - delta * n_unit, dtype=float)


def project_supported_anchor(
    xyz: np.ndarray,
    normal: np.ndarray,
    positions: np.ndarray,
    support: np.ndarray | list[int] | tuple[int, ...],
    *,
    cell: np.ndarray | None = None,
    pbc: Sequence[bool] | np.ndarray | None = None,
    max_iters: int = 5,
) -> np.ndarray:
    """Project *xyz* onto its MIC support plane until the MIC image is stable."""
    vert = np.asarray(xyz, dtype=float).reshape(3).copy()
    pos = np.asarray(positions, dtype=float)
    for _ in range(int(max_iters)):
        supp = mic_support_positions(vert, pos, support, cell=cell, pbc=pbc)
        if len(supp) == 0:
            return vert
        updated = project_anchor_to_support_plane(vert, normal, supp)
        if float(np.linalg.norm(updated - vert)) <= 1e-10:
            return updated
        vert = updated
    return vert


def project_vertices_to_support_plane(
    vertices: np.ndarray,
    normals: np.ndarray | None,
    atom_indices: list[tuple[int, ...]] | None,
    positions: np.ndarray,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Project supported vertices onto their support plane (MIC when *cell*/*pbc*)."""
    verts = np.asarray(vertices, dtype=float).copy()
    if normals is None or atom_indices is None or len(verts) == 0:
        return verts
    normals_arr = np.asarray(normals, dtype=float)
    pos = np.asarray(positions, dtype=float)
    for i, support in enumerate(atom_indices):
        if not support:
            continue
        verts[i] = project_supported_anchor(
            verts[i],
            normals_arr[i],
            pos,
            support,
            cell=cell,
            pbc=pbc,
        )
    return verts


# ---------------------------------------------------------------------------
# Parameter derivation helpers
# ---------------------------------------------------------------------------


def _mean_covalent_radius(
    symbols: list[str],
    *,
    fallback: float | None = None,
) -> float:
    """Mean of positive covalent radii for *symbols*.

    When no tabulated positive radii exist, return *fallback* if provided;
    otherwise raise ``ValueError`` (Voronoi window derivation has no safe default).
    """
    radii = [_get_covalent_radius(s) for s in symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        if fallback is not None:
            return float(fallback)
        raise ValueError(
            f"No positive covalent radii for symbols {symbols!r}; "
            "cannot derive Voronoi distance window"
        )
    return float(np.mean(valid))


def _derive_voronoi_distance_window(
    positions: np.ndarray,
    symbols: list[str],
    pbc: np.ndarray,
    cell: np.ndarray | None = None,
) -> tuple[float, float]:
    if len(positions) == 0:
        raise ValueError("Cannot derive Voronoi distance window from empty positions")
    if bool(pbc[0]) and bool(pbc[1]) and not bool(pbc[2]) and cell is not None:
        # Slab: vacuum-richer face along a×b, not Cartesian z or the bulk average.
        heights = _height_along_slab_normal(positions, cell)
        top_depth = _derive_top_layer_tolerance(symbols)
        top_mask = _primary_height_band_mask(
            heights,
            top_depth,
            face_sign=_slab_vacuum_face_sign(positions, cell),
        )
        top_idx = np.nonzero(top_mask)[0]
        top_symbols = [symbols[int(i)] for i in top_idx] if len(top_idx) else symbols
        base_radius = _mean_covalent_radius(top_symbols)
    else:
        # Nanoparticle, porous, and non-periodic: mean over all framework atoms.
        base_radius = _mean_covalent_radius(symbols)

    probe_radius = _VORONOI_PROBE_RADIUS_COVALENT_SCALE * base_radius
    max_distance = _VORONOI_MAX_DISTANCE_COVALENT_SCALE * base_radius
    return float(probe_radius), float(max(max_distance, probe_radius))


def _top_layer_tolerance_from_mean_radius(mean_radius: float) -> float:
    """Covalent-radius-derived top-layer depth from a precomputed mean radius."""
    return float(
        min(
            max(
                _TOP_LAYER_DEPTH_MIN_ANGSTROM,
                _TOP_LAYER_DEPTH_COVALENT_SCALE * mean_radius,
            ),
            _TOP_LAYER_DEPTH_MAX_ANGSTROM,
        )
    )


def _derive_top_layer_tolerance(symbols: list[str]) -> float:
    """Covalent-radius-derived top-layer depth, capped for FCC-like slabs."""
    return _top_layer_tolerance_from_mean_radius(_mean_covalent_radius(symbols))


def _pore_threshold_from_mean_radius(mean_radius: float) -> float:
    """Pore classification threshold from a precomputed mean covalent radius."""
    return max(
        _PORE_THRESHOLD_MIN_ANGSTROM,
        _PORE_THRESHOLD_COVALENT_SCALE * mean_radius,
    )


def derive_pore_threshold(symbols: list[str]) -> float:
    """Return pore classification threshold from mean covalent radius.

    Parameters
    ----------
    symbols
        List of element symbols.
    """
    return _pore_threshold_from_mean_radius(_mean_covalent_radius(symbols))


def _periodic_delta(
    delta: np.ndarray,
    cell: np.ndarray | None,
    pbc: np.ndarray | None,
) -> np.ndarray:
    if (
        cell is None
        or pbc is None
        or not np.any(pbc)
        or not cell_has_volume(np.asarray(cell, dtype=float))
    ):
        return delta
    flat = delta.reshape(-1, 3)
    return _minimum_image_cartesian_delta(
        flat, np.asarray(cell, dtype=float), np.asarray(pbc, dtype=bool)
    ).reshape(delta.shape)


def ray_exposure_mask(
    origins: np.ndarray,
    normals: np.ndarray,
    framework: np.ndarray,
    radii: np.ndarray,
    *,
    exclude_indices: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
    step: float = _ADAPTIVE_GRID_EXPOSURE_STEP,
    n_steps: int = _ADAPTIVE_GRID_EXPOSURE_N_STEPS,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return whether clearance along *normals* is non-decreasing.

    Clearance is the nearest other framework atom's distance minus that atom's
    radius. *exclude_indices[i]* is omitted for origin *i* (the atom itself).
    """
    origins_arr = np.asarray(origins, dtype=float).reshape(-1, 3)
    n = len(origins_arr)
    if n == 0:
        return np.ones(0, dtype=bool)
    normals_arr = np.asarray(normals, dtype=float).reshape(n, 3)
    framework_arr = np.asarray(framework, dtype=float).reshape(-1, 3)
    radii_arr = np.asarray(radii, dtype=float).reshape(-1)
    if len(framework_arr) == 0:
        return np.ones(n, dtype=bool)
    n_steps_i = int(n_steps)
    steps = np.arange(0, n_steps_i + 1, dtype=float)
    pts = (
        origins_arr[:, None, :]
        + steps[None, :, None] * float(step) * normals_arr[:, None, :]
    )
    delta = _periodic_delta(
        pts[:, :, None, :] - framework_arr[None, None, :, :], cell, pbc
    )
    dist = np.linalg.norm(delta, axis=-1) - radii_arr.reshape(1, 1, -1)
    if exclude_indices is not None:
        excluded = np.asarray(exclude_indices, dtype=int).reshape(n)
        for i, ex in enumerate(excluded):
            if 0 <= int(ex) < dist.shape[-1]:
                dist[i, :, int(ex)] = np.inf
    clear = np.min(dist, axis=-1)
    diffs = clear[:, 1:] - clear[:, :-1]
    return np.all(diffs >= -float(tolerance), axis=1)


def shadowed_by_higher_mask(
    candidate_indices: np.ndarray,
    higher_indices: np.ndarray,
    positions: np.ndarray,
    normal: np.ndarray,
    median_nn: float,
    *,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> np.ndarray:
    """Return whether a higher atom lies inside the lateral shadow of *median_nn*.

    The shadow radius is ``_EXPOSED_SHADOW_NN_SCALE × median_nn``. It hides a
    covered subsurface hollow and leaves a monatomic step edge.
    """
    cand = np.asarray(candidate_indices, dtype=int).reshape(-1)
    higher = np.asarray(higher_indices, dtype=int).reshape(-1)
    if len(cand) == 0:
        return np.ones(0, dtype=bool)
    if len(higher) == 0 or not np.isfinite(median_nn) or median_nn <= 0.0:
        return np.zeros(len(cand), dtype=bool)
    direction = np.asarray(normal, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(direction))
    if nrm <= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.zeros(len(cand), dtype=bool)
    n_hat = direction / nrm
    pos = np.asarray(positions, dtype=float)
    radius = _EXPOSED_SHADOW_NN_SCALE * float(median_nn)
    delta = _periodic_delta(pos[cand][:, None, :] - pos[higher][None, :, :], cell, pbc)
    along = np.einsum("ijk,k->ij", delta, n_hat)
    lateral = delta - along[:, :, None] * n_hat.reshape(1, 1, 3)
    lat = np.linalg.norm(lateral, axis=-1)
    # Candidate minus higher: a higher atom has a negative normal component.
    above = along < -_SURFACE_NORMAL_FALLBACK_NORM_EPS
    return np.any((lat <= radius) & above, axis=1)


def exposed_lower_band_indices(
    positions: np.ndarray,
    primary_indices: np.ndarray,
    symbols: list[str],
    cell: np.ndarray,
    pbc: np.ndarray,
    median_nn: float,
) -> np.ndarray:
    """Atoms within one nearest-neighbour below *primary_indices* that are open.

    Covered second-layer atoms (shadowed by the primary band) are excluded.
    A flat multilayer therefore adds nothing. A step or vacancy floor does not.
    """
    pos = np.asarray(positions, dtype=float)
    primary = np.asarray(primary_indices, dtype=int).reshape(-1)
    if (
        len(pos) == 0
        or len(primary) == 0
        or not np.isfinite(median_nn)
        or float(median_nn) <= 0.0
    ):
        return np.zeros(0, dtype=int)
    heights = _height_along_slab_normal(pos, cell)
    face_sign = _slab_vacuum_face_sign(pos, cell)
    hout = heights * face_sign
    h_ext = float(np.max(hout[primary]))
    band = (hout >= h_ext - float(median_nn)) & (hout < h_ext - 1e-6)
    band[primary] = False
    cand = np.nonzero(band)[0]
    if len(cand) == 0:
        return cand
    n_hat = face_sign * _slab_normal(cell)
    radii = framework_radii_from_symbols(symbols)
    # Rim atoms sit above a vacancy floor or a step edge. They must not veto
    # the outward ray; the shadow test is what rejects a covered second layer.
    ray_ok = np.zeros(len(cand), dtype=bool)
    for i, ai in enumerate(cand):
        at_or_inward = hout <= float(hout[int(ai)]) + 1e-6
        fw = np.nonzero(at_or_inward)[0]
        local_ex = int(np.nonzero(fw == int(ai))[0][0])
        ray_ok[i] = bool(
            ray_exposure_mask(
                pos[int(ai)].reshape(1, 3),
                n_hat.reshape(1, 3),
                pos[fw],
                radii[fw],
                exclude_indices=np.array([local_ex], dtype=int),
                cell=cell,
                pbc=pbc,
            )[0]
        )
    shadow = shadowed_by_higher_mask(
        cand,
        primary,
        pos,
        n_hat,
        float(median_nn),
        cell=cell,
        pbc=pbc,
    )
    return cand[ray_ok & ~shadow]

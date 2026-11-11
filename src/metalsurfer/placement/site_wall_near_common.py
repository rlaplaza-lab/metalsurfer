"""Shared wall-near site types, side policy, and NMS for adaptive grid / rolling probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from scipy.spatial import KDTree

from .._geom_pbc import mic_point_distance
from .._utils import cell_has_volume
from ._constants import _ATOP_INJECTION_HEIGHT_FACTOR, _VECTOR_NORM_EPS
from .site_coords import (
    _cart_to_frac,
    _periodic_image_offsets,
    _wrap_fractional,
)

SidePolicy = Literal["all", "positive", "negative", "external"]


@dataclass(frozen=True)
class CandidateSite:
    """Internal wall-near candidate before shared Site construction."""

    position: np.ndarray
    clearance: float
    support_indices: tuple[int, ...]
    support_image_shifts: tuple[tuple[int, int, int], ...]
    support_distances: tuple[float, ...]
    normal: np.ndarray
    score: float


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


def resolve_side_policy_for_pbc(
    pbc: np.ndarray,
    side_policy: SidePolicy,
) -> SidePolicy:
    """Map the shared config default ``positive`` onto PBC geometry.

    Explicit ``all`` / ``negative`` / ``external`` are left unchanged. The
    default ``positive`` means:

    - two periodic axes (slab-like): keep the vacuum-face half-space
    - no periodic axes (cluster): COM-outward ``external``
    - one or three periodic axes: ``all`` (no single vacuum face)
    """
    if side_policy not in ("all", "positive", "negative", "external"):
        raise ValueError(f"Unknown side_policy {side_policy!r}")
    if side_policy != "positive":
        return side_policy
    n_periodic = int(np.count_nonzero(np.asarray(pbc, dtype=bool).reshape(3)))
    if n_periodic == 2:
        return "positive"
    if n_periodic == 0:
        return "external"
    return "all"


def _side_policy_mask(
    normals: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    side_policy: SidePolicy,
    vertices: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> np.ndarray:
    """Apply face / exposure policy from PBC geometry (not material labels).

    - Exactly one non-periodic axis (slab-like): ``a×b``-style face filter for
      ``positive`` / ``negative``.
    - No PBC (finite cluster): COM outward for ``external``.
    - Fully 3D-periodic: no face filter (wall-near shells on all sides).
    """
    if side_policy not in ("all", "positive", "negative", "external"):
        raise ValueError(f"Unknown side_policy {side_policy!r}")
    n = len(normals)
    if n == 0:
        return np.ones(0, dtype=bool)
    keep = np.ones(n, dtype=bool)
    pbc_arr = np.asarray(pbc, dtype=bool).reshape(3)
    n_periodic = int(np.count_nonzero(pbc_arr))

    if n_periodic == 2 and cell_has_volume(cell):
        periodic_axes = [i for i in range(3) if bool(pbc_arr[i])]
        n_hat = np.cross(
            np.asarray(cell[periodic_axes[0]], dtype=float),
            np.asarray(cell[periodic_axes[1]], dtype=float),
        )
        nrm = float(np.linalg.norm(n_hat))
        if nrm <= _VECTOR_NORM_EPS:
            return keep
        n_hat = n_hat / nrm
        dots = np.einsum("ij,j->i", normals, n_hat)
        if side_policy == "positive":
            keep &= dots >= -1e-12
        elif side_policy == "negative":
            keep &= dots <= 1e-12
    elif (
        n_periodic == 0
        and side_policy == "external"
        and vertices is not None
        and positions is not None
    ):
        com = np.mean(np.asarray(positions, dtype=float), axis=0)
        outward = np.asarray(vertices, dtype=float) - com
        norms = np.linalg.norm(outward, axis=1)
        valid = norms > _VECTOR_NORM_EPS
        uhat = np.zeros_like(outward)
        uhat[valid] = outward[valid] / norms[valid, None]
        keep &= np.einsum("ij,ij->i", normals, uhat) >= -1e-12
    return keep


def _pair_within_radius(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> bool:
    """Return True if *a* and *b* are within *radius* under triclinic-safe MIC."""
    return mic_point_distance(a, b, cell, pbc) <= float(radius) + 1e-12


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


def _normals_oppose(n_a: np.ndarray, n_b: np.ndarray) -> bool:
    a = np.asarray(n_a, dtype=float).reshape(3)
    b = np.asarray(n_b, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= _VECTOR_NORM_EPS or nb <= _VECTOR_NORM_EPS:
        return False
    return float(np.dot(a / na, b / nb)) < 0.0


def _nms(
    vertices: np.ndarray,
    nn: np.ndarray,
    scores: np.ndarray,
    merge_r: float,
    cell: np.ndarray,
    pbc: np.ndarray,
    radii: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy NMS; returns ``(kept_vertices, kept_nn, kept_indices)``.

    Non-PBC builds a neighbor graph via ``query_pairs``. Periodic cells use
    image-expanded ball queries. Opposing normals are not suppressed.
    """
    if len(vertices) == 0:
        return vertices, nn, np.empty(0, dtype=int)
    if radii is None:
        radii = np.full(len(vertices), float(merge_r), dtype=float)
    max_r = float(max(float(merge_r), float(np.max(radii))))
    offsets = _image_offsets_for_radius(cell, pbc, max_r)
    suppressed = np.zeros(len(vertices), dtype=bool)
    accepted: list[int] = []
    normals_arr = (
        None if normals is None else np.asarray(normals, dtype=float).reshape(-1, 3)
    )

    def _may_suppress(i: int, j: int) -> bool:
        return normals_arr is None or not _normals_oppose(
            normals_arr[i], normals_arr[j]
        )

    if offsets is None:
        neighbors: list[list[int]] = [[] for _ in range(len(vertices))]
        for i, j in KDTree(vertices).query_pairs(r=max_r):
            ii, jj = int(i), int(j)
            r_ij = float(min(float(radii[ii]), float(radii[jj])))
            if _pair_within_radius(vertices[ii], vertices[jj], r_ij, cell, pbc):
                neighbors[ii].append(jj)
                neighbors[jj].append(ii)
        for i in _nms_rank_order(vertices, scores, cell, pbc):
            ii = int(i)
            if suppressed[ii]:
                continue
            accepted.append(ii)
            for j in neighbors[ii]:
                if _may_suppress(ii, j):
                    suppressed[j] = True
    else:
        n = len(vertices)
        expanded = np.vstack([vertices + off for off in offsets])
        tree = KDTree(expanded)

        def query_ball(pt: np.ndarray, r: float) -> list[int]:
            return list({int(h) % n for h in tree.query_ball_point(pt, r=float(r))})

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
                if _pair_within_radius(
                    vertices[ii], vertices[j], r_ij, cell, pbc
                ) and _may_suppress(ii, j):
                    suppressed[j] = True

    idx = np.asarray(accepted, dtype=int)
    return vertices[idx], nn[idx], idx


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
    """Isotropic NMS at *merge_radius*, returned in score order."""
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0]]

    verts = np.asarray([c.position for c in candidates], dtype=float)
    scores = np.asarray([c.score for c in candidates], dtype=float)
    normals = np.asarray([c.normal for c in candidates], dtype=float)
    _, _, kept_idx = _nms(
        verts,
        np.zeros(len(verts), dtype=float),
        scores,
        merge_r=float(merge_radius),
        cell=cell,
        pbc=pbc,
        normals=normals,
    )
    out = [candidates[int(i)] for i in kept_idx]
    rep_order = _nms_rank_order(
        np.asarray([c.position for c in out], dtype=float),
        np.asarray([c.score for c in out], dtype=float),
        cell,
        pbc,
    )
    return [out[int(i)] for i in rep_order]


def _best_per_support_key(
    candidates: list[CandidateSite],
    *,
    keep_opposing_normals: bool = False,
) -> list[CandidateSite]:
    """Keep the highest-score candidate per support; optionally both faces."""
    groups: dict[frozenset[tuple[int, tuple[int, int, int]]], list[CandidateSite]] = {}
    for c in candidates:
        key = _support_key(c.support_indices, c.support_image_shifts)
        groups.setdefault(key, []).append(c)
    out: list[CandidateSite] = []
    for group in groups.values():
        group_sorted = sorted(group, key=lambda c: float(c.score), reverse=True)
        best = group_sorted[0]
        out.append(best)
        if not keep_opposing_normals:
            continue
        for other in group_sorted[1:]:
            if _normals_oppose(best.normal, other.normal):
                out.append(other)
                break
    return out


def wall_near_clearance_window(
    framework_radii: np.ndarray,
    probe_radius: float,
    max_site_distance: float,
) -> tuple[float, float]:
    """Map probe / max-site distances onto the element-aware clearance window."""
    n_atoms = len(framework_radii)
    mean_r = float(np.mean(framework_radii)) if n_atoms else 0.0
    min_clearance = float(probe_radius) - mean_r
    max_clearance = float(max_site_distance) - mean_r
    if max_clearance < min_clearance:
        max_clearance = min_clearance
    return float(min_clearance), float(max_clearance)


def shell_target_clearance(
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


def candidates_to_arrays(
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
    nn = np.asarray(tree.query(cast(Any, verts), k=1)[0], dtype=float).ravel()
    return verts, nn, clear, atoms, normals


def empty_adaptive_grid_result(
    *,
    spacing: AdaptiveGridSpacing,
    accessibility_tree: KDTree,
) -> AdaptiveGridResult:
    """Empty catalog with the caller's spacing / accessibility tree."""
    return AdaptiveGridResult(
        vertices=np.empty((0, 3), dtype=float),
        nn_dists=np.empty(0, dtype=float),
        clearances=np.empty(0, dtype=float),
        atom_indices=[],
        normals=np.empty((0, 3), dtype=float),
        spacing=spacing,
        accessibility_tree=accessibility_tree,
    )

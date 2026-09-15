"""Nanoparticle surface topology: convex-hull skin + nearest-neighbour graph.

Convex particles (symmetric or not) are handled the same way: atoms on the
hull skin, NN edges for bridges, chordless 3-/4-cycles for hollows, all lifted
along hull-facet normals. Concave dents are outside this model.
"""

from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from scipy.spatial import ConvexHull, KDTree, QhullError

from ._constants import (
    _NP_HULL_OUTSIDE_EPS,
    _NP_HULL_SURFACE_EPS,
    _NP_NN_BOND_MAX_SCALE,
    _NP_NN_BOND_MIN_SCALE,
    _NP_PLANAR_4RING_TOL_SCALE,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _VORONOI_DEDUP_TOLERANCE,
)
from .site_coords import _deduplicate_points


class _NPTopologyResult(NamedTuple):
    """Output of :func:`_generate_nanoparticle_topology_sites`."""

    vertices: np.ndarray
    nn_dists: np.ndarray
    sources: list[str]
    atom_indices: list[tuple[int, ...]]


def _try_convex_hull(positions: np.ndarray) -> ConvexHull | None:
    """Return a ConvexHull, or ``None`` if Qhull cannot build one."""
    pts = np.asarray(positions, dtype=float)
    if len(pts) < 4:
        return None
    try:
        return ConvexHull(pts)
    except (QhullError, ValueError, RuntimeError):
        return None


def _hull_signed_distances(points: np.ndarray, hull: ConvexHull) -> np.ndarray:
    """``(n_points, n_facets)`` signed distances; interior points are ≤ 0."""
    eqs = np.asarray(hull.equations, dtype=float)
    pts = np.asarray(points, dtype=float)
    return pts @ eqs[:, :3].T + eqs[:, 3]


def _convex_hull_surface_mask(
    positions: np.ndarray,
    *,
    eps: float = _NP_HULL_SURFACE_EPS,
    hull: ConvexHull | None = None,
) -> np.ndarray:
    """Atoms on the hull skin (including face-centre / edge atoms)."""
    pts = np.asarray(positions, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    if hull is None:
        hull = _try_convex_hull(pts)
    if hull is None:
        return np.ones(len(pts), dtype=bool)
    return np.max(_hull_signed_distances(pts, hull), axis=1) > -float(eps)


def _outside_convex_hull_mask(
    points: np.ndarray,
    hull: ConvexHull,
    *,
    eps: float = _NP_HULL_OUTSIDE_EPS,
) -> np.ndarray:
    """Strictly outside the hull (used by tests / callers checking lift)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    return np.max(_hull_signed_distances(pts, hull), axis=1) > float(eps)


def _normalize_rows(vecs: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors; zero rows stay zero."""
    out = np.asarray(vecs, dtype=float).copy()
    norms = np.linalg.norm(out, axis=1)
    ok = norms >= _SURFACE_NORMAL_FALLBACK_NORM_EPS
    out[ok] /= norms[ok, None]
    out[~ok] = 0.0
    return out


def _surface_atom_normals(
    positions: np.ndarray,
    surf_idx: np.ndarray,
    hull: ConvexHull,
    *,
    eps: float = _NP_HULL_SURFACE_EPS,
) -> np.ndarray:
    """Outward normals for surface atoms from the hull facets they lie on.

    Face-centre atoms are not Qhull vertices but still lie on facet planes;
    averaging those facet normals works for symmetric and lopsided convex clusters.
    """
    eqs = np.asarray(hull.equations, dtype=float)
    signed = _hull_signed_distances(positions[surf_idx], hull)
    on_facet = np.abs(signed) <= float(eps)
    normals = _normalize_rows(on_facet.astype(float) @ eqs[:, :3])
    missing = np.linalg.norm(normals, axis=1) < _SURFACE_NORMAL_FALLBACK_NORM_EPS
    if np.any(missing):
        com = np.mean(positions, axis=0)
        normals[missing] = _normalize_rows(positions[surf_idx[missing]] - com)
    return normals


def _flip_outward(normal: np.ndarray, from_com: np.ndarray) -> np.ndarray:
    """Flip *normal* so it points away from the cluster interior."""
    n = np.asarray(normal, dtype=float)
    if float(np.dot(n, from_com)) < 0.0:
        return -n
    return n


def _is_planar_quad(pts: np.ndarray, *, tol: float) -> bool:
    p = np.asarray(pts, dtype=float).reshape(4, 3)
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    nrm = float(np.linalg.norm(normal))
    if nrm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return False
    return abs(float(np.dot(p[3] - p[0], normal / nrm))) < float(tol)


def _generate_nanoparticle_topology_sites(
    positions: np.ndarray,
    accessibility_tree: KDTree,
    site_height: float,
    probe_radius: float,
    max_distance: float,
    *,
    metal_nn: float,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> _NPTopologyResult:
    """Atop / bridge / hollow sites on the outer hull skin.

    Returns empty when a hull cannot be built; the caller may then inject atops.
    """
    empty = _NPTopologyResult(
        np.empty((0, 3), dtype=float),
        np.empty(0, dtype=float),
        [],
        [],
    )
    positions = np.asarray(positions, dtype=float)
    if len(positions) == 0:
        return empty

    hull = _try_convex_hull(positions)
    if hull is None:
        return empty

    surf_idx = np.nonzero(_convex_hull_surface_mask(positions, hull=hull))[0].astype(
        int
    )
    if len(surf_idx) == 0:
        return empty

    com = np.mean(positions, axis=0)
    surf_pos = positions[surf_idx]
    atom_out = _surface_atom_normals(positions, surf_idx, hull)
    local_to_global = {int(i): int(g) for i, g in enumerate(surf_idx)}

    min_bond = float(metal_nn) * _NP_NN_BOND_MIN_SCALE
    max_bond = float(metal_nn) * _NP_NN_BOND_MAX_SCALE
    planar_tol = float(metal_nn) * _NP_PLANAR_4RING_TOL_SCALE

    adj: dict[int, set[int]] = defaultdict(set)
    edges: list[tuple[int, int]] = []
    if len(surf_pos) >= 2:
        for i, j in KDTree(surf_pos).query_pairs(r=max_bond):
            d = float(np.linalg.norm(surf_pos[i] - surf_pos[j]))
            if d < min_bond:
                continue
            adj[i].add(j)
            adj[j].add(i)
            edges.append((i, j) if i < j else (j, i))
        edges = sorted(set(edges))

    candidates: list[np.ndarray] = []
    candidate_dists: list[float] = []
    candidate_sources: list[str] = []
    candidate_atoms: list[tuple[int, ...]] = []

    def _add(point: np.ndarray, source: str, atoms: tuple[int, ...]) -> None:
        dist, _ = accessibility_tree.query(
            np.asarray(point, dtype=float).reshape(1, 3), k=1
        )
        d_nn = float(np.asarray(dist, dtype=float).ravel()[0])
        if float(probe_radius) <= d_nn <= float(max_distance):
            candidates.append(np.asarray(point, dtype=float))
            candidate_dists.append(d_nn)
            candidate_sources.append(source)
            candidate_atoms.append(atoms)

    height = float(site_height)
    for i, ai in enumerate(surf_idx):
        n_hat = atom_out[i]
        if float(np.linalg.norm(n_hat)) < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            continue
        _add(positions[int(ai)] + height * n_hat, "topology_atop", (int(ai),))

    for i, j in edges:
        n_hat = _normalize_rows((atom_out[i] + atom_out[j]).reshape(1, 3))[0]
        if float(np.linalg.norm(n_hat)) < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            n_hat = atom_out[i]
        mid = 0.5 * (surf_pos[i] + surf_pos[j])
        _add(
            mid + height * n_hat,
            "topology_bridge",
            (local_to_global[i], local_to_global[j]),
        )

    seen_tris: set[tuple[int, ...]] = set()
    for i, j in edges:
        for k in adj[i] & adj[j]:
            tri = tuple(sorted((i, j, k)))
            if tri in seen_tris:
                continue
            seen_tris.add(tri)
            pts = surf_pos[list(tri)]
            centroid = np.mean(pts, axis=0)
            n_hat = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            nrm = float(np.linalg.norm(n_hat))
            if nrm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                continue
            n_hat = _flip_outward(n_hat / nrm, centroid - com)
            _add(
                centroid + height * n_hat,
                "topology_hollow",
                tuple(local_to_global[m] for m in tri),
            )

    seen_quads: set[tuple[int, ...]] = set()
    for i, j in edges:
        for a in adj[i]:
            if a == j:
                continue
            for b in adj[j]:
                if b in (i, a) or b not in adj[a]:
                    continue
                # Chordless 4-cycle i–a–b–j–i.
                if b in adj[i] or a in adj[j]:
                    continue
                quad = tuple(sorted((i, a, b, j)))
                if quad in seen_quads:
                    continue
                ordered = (i, a, b, j)
                pts = surf_pos[list(ordered)]
                if not _is_planar_quad(pts, tol=planar_tol):
                    continue
                seen_quads.add(quad)
                centroid = np.mean(pts, axis=0)
                n_hat = np.cross(pts[1] - pts[0], pts[3] - pts[0])
                nrm = float(np.linalg.norm(n_hat))
                if nrm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                    continue
                n_hat = _flip_outward(n_hat / nrm, centroid - com)
                _add(
                    centroid + height * n_hat,
                    "topology_hollow",
                    tuple(local_to_global[m] for m in ordered),
                )

    if not candidates:
        return empty

    cand_arr = np.asarray(candidates, dtype=float)
    cand_dist = np.asarray(candidate_dists, dtype=float)
    keep = _deduplicate_points(cand_arr, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    kept = np.nonzero(keep)[0]
    return _NPTopologyResult(
        cand_arr[keep],
        cand_dist[keep],
        [candidate_sources[i] for i in kept],
        [candidate_atoms[i] for i in kept],
    )

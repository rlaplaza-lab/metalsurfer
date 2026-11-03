"""Sparse sites on and beside a previously adsorbed ligand.

Ligand atoms are not fed into the slab top-layer mask (they would hide the
metal). Exposed ligand atoms get one atop site, and a bridge to each substrate
atom inside the neighbour window.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms
from scipy.spatial import KDTree

from ._constants import (
    _NP_NN_BOND_MAX_SCALE,
    _NP_NN_BOND_MIN_SCALE,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _VORONOI_DEDUP_TOLERANCE,
)
from ._material import material_aware_pbc
from .geometry import framework_radii_from_symbols, tangent_basis_from_normal
from .site_classify import site_env_fingerprint
from .site_coords import (
    _minimum_image_cartesian_delta,
    _slab_normal,
    ray_exposure_mask,
)
from .site_plugins.helpers import median_nn_or_fallback
from .site_types import Site

_LIGAND_ATOP_SOURCE = "ligand_atop"
_LIGAND_BRIDGE_SOURCE = "ligand_bridge"


def _unit(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(arr))
    if nrm <= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        fb = np.asarray(fallback, dtype=float).reshape(3)
        fb_nrm = float(np.linalg.norm(fb))
        if fb_nrm <= _SURFACE_NORMAL_FALLBACK_NORM_EPS:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return fb / fb_nrm
    return arr / nrm


def _mic_delta(
    a: np.ndarray,
    b: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    delta = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if np.any(pbc):
        return _minimum_image_cartesian_delta(delta, cell, pbc)
    return delta


def ligand_local_sites(
    substrate: Atoms,
    full_slab: Atoms,
    *,
    material_type: str,
) -> list[Site]:
    """Atop sites on exposed ligand atoms and bridges to nearby substrate atoms."""
    n_sub = len(substrate)
    if len(full_slab) <= n_sub or n_sub == 0:
        return []
    positions = np.asarray(full_slab.get_positions(), dtype=float)
    symbols = list(full_slab.get_chemical_symbols())
    cell = np.asarray(full_slab.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc(material_type), dtype=bool)
    lig_idx = np.arange(n_sub, len(full_slab), dtype=int)
    sub_pos = positions[:n_sub]
    sub_tree = KDTree(sub_pos)
    _, nearest_j = sub_tree.query(positions[lig_idx], k=1)
    nearest_j = np.asarray(nearest_j, dtype=int).ravel()
    fallback = (
        _slab_normal(cell) if material_type == "slab" else np.array([0.0, 0.0, 1.0])
    )
    normals = np.vstack(
        [
            _unit(positions[int(ai)] - sub_pos[int(j)], fallback)
            for ai, j in zip(lig_idx, nearest_j, strict=True)
        ]
    )
    radii = framework_radii_from_symbols(symbols)
    exposed = ray_exposure_mask(
        positions[lig_idx],
        normals,
        positions,
        radii,
        exclude_indices=lig_idx,
        cell=cell,
        pbc=pbc,
    )
    if not np.any(exposed):
        return []
    median_nn = median_nn_or_fallback(
        np.empty(0, dtype=float),
        reference_positions=sub_pos,
        cell=cell,
        pbc=pbc,
    )
    min_bond = _NP_NN_BOND_MIN_SCALE * float(median_nn)
    max_bond = _NP_NN_BOND_MAX_SCALE * float(median_nn)
    sites: list[Site] = []
    for local_i, ai in enumerate(lig_idx):
        if not bool(exposed[local_i]):
            continue
        atom_pos = positions[int(ai)]
        normal = normals[local_i]
        sites.append(
            _make_ligand_site(
                atom_pos,
                normal,
                (int(ai),),
                (0.0,),
                symbols,
                material_type,
                _LIGAND_ATOP_SOURCE,
                "atop",
            )
        )
        neighbor_js = sub_tree.query_ball_point(atom_pos, r=max_bond)
        for j in neighbor_js:
            delta = _mic_delta(atom_pos, sub_pos[int(j)], cell, pbc)
            dist = float(np.linalg.norm(delta))
            if dist < min_bond or dist > max_bond:
                continue
            mid = sub_pos[int(j)] + 0.5 * delta
            sites.append(
                _make_ligand_site(
                    mid,
                    normal,
                    (int(ai), int(j)),
                    (dist, dist),
                    symbols,
                    material_type,
                    _LIGAND_BRIDGE_SOURCE,
                    "bridge",
                )
            )
    return sites


def _make_ligand_site(
    xyz: np.ndarray,
    normal: np.ndarray,
    support: tuple[int, ...],
    distances: tuple[float, ...],
    symbols: list[str],
    material_type: str,
    source: str,
    site_type: str,
) -> Site:
    n_hat = _unit(normal, np.array([0.0, 0.0, 1.0]))
    return Site(
        xyz=np.asarray(xyz, dtype=float),
        normal=n_hat,
        site_type=site_type,
        slab_indices=support,
        material_type=material_type,
        site_source=source,
        env_fingerprint=site_env_fingerprint(support, symbols, distances),
        tangent_basis=tangent_basis_from_normal(n_hat),
    )


def drop_sites_near_catalog(
    extra: list[Site],
    existing: list[Site],
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    tolerance: float = _VORONOI_DEDUP_TOLERANCE,
) -> list[Site]:
    """Drop *extra* sites within *tolerance* of an existing catalog anchor."""
    if not extra or not existing:
        return list(extra)
    kept: list[Site] = []
    anchors = [np.asarray(site.xyz, dtype=float) for site in existing]
    for site in extra:
        xyz = np.asarray(site.xyz, dtype=float)
        too_close = False
        for anchor in anchors:
            delta = _mic_delta(xyz, anchor, cell, pbc)
            if float(np.linalg.norm(delta)) <= float(tolerance):
                too_close = True
                break
        if not too_close:
            kept.append(site)
            anchors.append(xyz)
    return kept

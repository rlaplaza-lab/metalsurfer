"""Occupancy helpers for packing-aware site selection under coverage.

Occupancy compares **site vertices** (Voronoi / topology ``Site.xyz``) to
**existing adsorbate atom positions**, optionally followed by an incoming
in-plane molecular **footprint** disk.  A site is kept when its vertex is at
least ``min_separation`` from every existing adsorbate atom (MIC), and — when
footprint pruning is enabled — when the lateral clearance to each existing
atom (minus that atom's covalent radius) is at least the incoming footprint
radius.  If footprint pruning empties a non-empty vertex mask, the vertex-only
mask is restored (Packmol ``avoid_overlap`` analogue).

``existing_adsorbate_positions(slab_for_sites, full_slab)`` returns the suffix of
``full_slab`` beyond ``len(slab_for_sites)`` — i.e. atoms added after the bare
substrate used for site detection.  ``existing_adsorbate_cloud`` also returns
per-atom covalent radii for footprint pruning.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from ase import Atoms

from .._numeric_defaults import OCCUPANCY_FOOTPRINT_SCALE_DEFAULT
from . import geometry as geom
from .clash import atom_radii_for_symbols
from .site_types import Site

logger = logging.getLogger(__name__)


def existing_adsorbate_positions(
    slab_for_sites: Atoms,
    full_slab: Atoms | None,
) -> np.ndarray | None:
    """Positions of atoms in *full_slab* beyond ``len(slab_for_sites)``, else None.

    Contract: *slab_for_sites* is the bare substrate used for site detection;
    *full_slab* may append previously placed adsorbate atoms.  Returns ``None``
    when there is no adsorbate suffix.

    Parameters
    ----------
    slab_for_sites
        Bare substrate used for site detection.
    full_slab
        Full system that may include pre-adsorbed atoms.
    """
    if full_slab is None:
        return None
    n_sub = len(slab_for_sites)
    if len(full_slab) <= n_sub:
        return None
    return np.asarray(full_slab.get_positions()[n_sub:], dtype=float)


def existing_adsorbate_cloud(
    slab_for_sites: Atoms,
    full_slab: Atoms | None,
    *,
    min_separation: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Existing adsorbate positions and covalent radii, or ``(None, None)``.

    Parameters
    ----------
    slab_for_sites
        Bare substrate used for site detection.
    full_slab
        Full system that may include pre-adsorbed atoms.
    min_separation
        Fallback floor for unknown covalent radii (``dtol/2`` analogue).
    """
    if full_slab is None:
        return None, None
    n_sub = len(slab_for_sites)
    if len(full_slab) <= n_sub:
        return None, None
    pos = np.asarray(full_slab.get_positions()[n_sub:], dtype=float)
    if pos.size == 0:
        return None, None
    symbols = list(full_slab.get_chemical_symbols()[n_sub:])
    radii = atom_radii_for_symbols(symbols, min_separation=float(min_separation))
    return pos, radii


def incoming_inplane_radius(
    conformer: Atoms,
    *,
    footprint_scale: float = OCCUPANCY_FOOTPRINT_SCALE_DEFAULT,
) -> float:
    """COM-centred in-plane envelope after dropping the molecular thickness axis.

    Flat molecules drop the plane normal (largest inertia axis); linear molecules
    drop the bond axis (smallest inertia axis); round molecules drop the shortest
    principal axis. Returns ``footprint_scale * max||p_perp||``.

    Parameters
    ----------
    conformer
        Molecule geometry.
    footprint_scale
        Scale applied to the raw envelope radius.
    """
    pos = np.asarray(conformer.get_positions(), dtype=float)
    if len(pos) == 0:
        return 0.0
    centered = pos - np.mean(pos, axis=0)
    if len(centered) == 1:
        return 0.0
    shape, _, eigenvecs = geom._classify_molecule_shape(centered)
    if shape == "flat":
        # Plane normal = largest-inertia axis.
        axis = eigenvecs[:, 2]
    elif shape == "linear":
        # Bond axis = smallest-inertia axis.
        axis = eigenvecs[:, 0]
    else:
        axis = eigenvecs[:, 0]
    proj = centered - np.outer(centered @ axis, axis)
    norms = np.linalg.norm(proj, axis=1)
    return float(footprint_scale) * float(np.max(norms))


def _normalize_existing_positions(existing: np.ndarray) -> np.ndarray:
    arr = np.asarray(existing, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"existing adsorbate positions must have shape (n, 3), got {arr.shape}"
        )
    return arr


def _sites_clearance_and_vertex_mask(
    sites: Sequence[Site],
    existing: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(vertex_mask, min_3d_dists, mic_vecs)`` for sites×existing."""
    n = len(sites)
    if n == 0:
        return (
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=float),
            np.zeros((0, 0, 3), dtype=float),
        )

    existing_arr = _normalize_existing_positions(existing)
    if existing_arr.size == 0:
        return (
            np.ones(n, dtype=bool),
            np.full(n, np.inf, dtype=float),
            np.zeros((n, 0, 3), dtype=float),
        )

    site_xyz = np.asarray([s.xyz for s in sites], dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    mic_vecs, dists = geom._mol_slab_pairwise_mic(site_xyz, existing_arr, cell_arr, pbc)
    min_dists = np.min(dists, axis=1)
    return min_dists >= float(min_separation), min_dists, mic_vecs


def _sites_footprint_mask(
    sites: Sequence[Site],
    mic_vecs: np.ndarray,
    existing_radii: np.ndarray,
    *,
    incoming_radius: float,
) -> np.ndarray:
    """Lateral disk clearance: ``||d_perp|| - r_atom >= incoming_radius``."""
    n = len(sites)
    if n == 0:
        return np.zeros(0, dtype=bool)
    if mic_vecs.shape[1] == 0:
        return np.ones(n, dtype=bool)

    normals = np.asarray([s.normal for s in sites], dtype=float)
    # Lateral component: remove the projection along each site outward normal.
    dots = np.einsum("sjd,sd->sj", mic_vecs, normals)
    perp = mic_vecs - dots[:, :, None] * normals[:, None, :]
    lateral = np.linalg.norm(perp, axis=2)
    r_exist = np.asarray(existing_radii, dtype=float).reshape(1, -1)
    clearance = lateral - r_exist
    return np.min(clearance, axis=1) >= float(incoming_radius)


def available_site_indices(
    sites: Sequence[Site],
    existing_positions: np.ndarray | None,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
    incoming_footprint_radius: float | None = None,
    existing_radii: np.ndarray | None = None,
    use_footprint: bool = False,
) -> list[int]:
    """Original indices into *sites* that pass occupancy (or all if empty coverage).

    When *use_footprint* is True and *incoming_footprint_radius* / *existing_radii*
    are provided, sites must also clear the lateral footprint disk.  If that
    empties a non-empty vertex mask, the vertex-only indices are returned with
    a warning.

    Parameters
    ----------
    sites
        Sequence of :class:`Site` objects.
    existing_positions
        Existing adsorbate positions or None.
    cell
        Unit cell matrix.
    pbc
        Periodic boundary condition flags.
    min_separation
        Minimum separation distance (Å).
    incoming_footprint_radius
        Optional incoming in-plane disk radius (Å).
    existing_radii
        Optional covalent radii for existing adsorbate atoms.
    use_footprint
        Enable footprint pruning (requires radius inputs).
    """
    if existing_positions is None or np.asarray(existing_positions).size == 0:
        return list(range(len(sites)))

    vertex_mask, _min_dists, mic_vecs = _sites_clearance_and_vertex_mask(
        sites,
        existing_positions,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )
    vertex_indices = [i for i, keep in enumerate(vertex_mask) if keep]
    if not vertex_indices:
        return []

    if (
        use_footprint
        and incoming_footprint_radius is not None
        and existing_radii is not None
        and float(incoming_footprint_radius) > 0.0
    ):
        foot_mask = _sites_footprint_mask(
            sites,
            mic_vecs,
            existing_radii,
            incoming_radius=float(incoming_footprint_radius),
        )
        combined = vertex_mask & foot_mask
        foot_indices = [i for i, keep in enumerate(combined) if keep]
        if not foot_indices:
            logger.warning(
                "Footprint occupancy emptied %d vertex-clear sites; "
                "falling back to vertex-only occupancy",
                len(vertex_indices),
            )
            return vertex_indices
        return foot_indices

    return vertex_indices


def site_clearance_distances(
    sites: Sequence[Site],
    existing_positions: np.ndarray | None,
    *,
    cell: np.ndarray,
    pbc: list[bool],
) -> np.ndarray:
    """Per-site minimum 3D MIC distance to existing adsorbates (inf if none)."""
    n = len(sites)
    if n == 0:
        return np.zeros(0, dtype=float)
    if existing_positions is None or np.asarray(existing_positions).size == 0:
        return np.full(n, np.inf, dtype=float)
    _mask, min_dists, _vecs = _sites_clearance_and_vertex_mask(
        sites,
        existing_positions,
        cell=cell,
        pbc=pbc,
        min_separation=0.0,
    )
    return min_dists


def _positions_mutually_clear(
    a_pos: np.ndarray,
    b_pos: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> bool:
    """Return whether every pair of positions is at least *min_separation* apart (MIC)."""
    a_arr = _normalize_existing_positions(a_pos)
    b_arr = _normalize_existing_positions(b_pos)
    if a_arr.size == 0 or b_arr.size == 0:
        return True
    dists = geom._mol_slab_pairwise_distances(a_arr, b_arr, cell, pbc)
    return bool(np.min(dists) >= float(min_separation))


def results_mutually_clear(
    a_atoms_suffix: Atoms,
    b_atoms_suffix: Atoms,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> bool:
    """Whether two adsorbate fragments can coexist on one slab (MIC).

    Pure n-tuplet primitive: every atom of *a_atoms_suffix* must be at least
    *min_separation* from every atom of *b_atoms_suffix*. Empty fragments are
    trivially clear.

    Parameters
    ----------
    a_atoms_suffix
        Adsorbate-only atoms of the first fragment.
    b_atoms_suffix
        Adsorbate-only atoms of the second fragment.
    cell
        Unit cell matrix of the underlying slab.
    pbc
        Material-aware periodicity flags (see
        :func:`placement._material.material_aware_pbc`).
    min_separation
        Minimum adsorbate-atom to adsorbate-atom distance in Å.
    """
    return _positions_mutually_clear(
        np.asarray(a_atoms_suffix.get_positions(), dtype=float),
        np.asarray(b_atoms_suffix.get_positions(), dtype=float),
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )


def filter_sites_by_occupancy(
    sites: Sequence[Site],
    existing_positions: np.ndarray | None,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
    incoming_footprint_radius: float | None = None,
    existing_radii: np.ndarray | None = None,
    use_footprint: bool = False,
) -> list[Site]:
    """Keep sites that pass occupancy (vertex, optional footprint + fallback).

    *existing_positions* may be ``None`` or empty; then *sites* is returned
    unchanged (order preserved).

    Parameters
    ----------
    sites
        Sequence of :class:`Site` objects.
    existing_positions
        Existing adsorbate positions or None.
    cell
        Unit cell matrix.
    pbc
        Periodic boundary condition flags.
    min_separation
        Minimum separation distance (Å).
    incoming_footprint_radius
        Optional incoming in-plane disk radius (Å).
    existing_radii
        Optional covalent radii for existing adsorbate atoms.
    use_footprint
        Enable footprint pruning.
    """
    keep = available_site_indices(
        sites,
        existing_positions,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
        incoming_footprint_radius=incoming_footprint_radius,
        existing_radii=existing_radii,
        use_footprint=use_footprint,
    )
    return [sites[i] for i in keep]

"""Occupancy helpers for packing-aware site selection under coverage.

Occupancy compares **site vertices** (Voronoi / topology ``Site.xyz``) to
**existing adsorbate atom positions**, not molecule–molecule footprints.  A site
is kept when its vertex is at least ``min_separation`` from every existing
adsorbate atom (MIC when a periodic cell is provided).

``existing_adsorbate_positions(slab_for_sites, full_slab)`` returns the suffix of
``full_slab`` beyond ``len(slab_for_sites)`` — i.e. atoms added after the bare
substrate used for site detection.
"""

from collections.abc import Sequence

import numpy as np
from ase import Atoms

from . import geometry as geom
from .site_types import Site


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


def _sites_clear_occupancy_mask(
    sites: Sequence[Site],
    existing: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> np.ndarray:
    """Boolean mask: True where site xyz is ≥ *min_separation* from all existing atoms (MIC).

    Uses one sites×existing MIC distance matrix (same pattern as
    :func:`geometry._mol_slab_pairwise_distances`).
    """
    n = len(sites)
    if n == 0:
        return np.zeros(0, dtype=bool)

    existing_arr = _normalize_existing_positions(existing)
    if existing_arr.size == 0:
        return np.ones(n, dtype=bool)

    site_xyz = np.asarray([s.xyz for s in sites], dtype=float)
    cell_arr = np.asarray(cell, dtype=float)
    dists = geom._mol_slab_pairwise_distances(site_xyz, existing_arr, cell_arr, pbc)
    min_dists = np.min(dists, axis=1)
    return min_dists >= float(min_separation)


def available_site_indices(
    sites: Sequence[Site],
    existing_positions: np.ndarray | None,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> list[int]:
    """Original indices into *sites* that pass occupancy (or all if *existing_positions* is None/empty).

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
    """
    if existing_positions is None or np.asarray(existing_positions).size == 0:
        return list(range(len(sites)))

    mask = _sites_clear_occupancy_mask(
        sites,
        existing_positions,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )
    return [i for i, keep in enumerate(mask) if keep]


def filter_sites_by_occupancy(
    sites: Sequence[Site],
    existing_positions: np.ndarray | None,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
) -> list[Site]:
    """Keep sites whose xyz is at least *min_separation* from all existing adsorbate atoms (MIC).

    *existing_positions* may be ``None`` or empty; then *sites* is returned
    unchanged (order preserved).  Site vertices are compared to adsorbate atoms,
    not full molecular footprints.

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
    """
    keep = available_site_indices(
        sites,
        existing_positions,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )
    return [sites[i] for i in keep]

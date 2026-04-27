"""Voronoi sites, clustering, and optional spglib-based symmetry reduction.

Simplified and consolidated placement module with improved slab handling.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import Atoms

from ..symmetry import SymmetryAnalyzer
from ._material import material_type_for_placement
from .geometry import (
    _get_covalent_radius,
)
from .sites import (
    _MOL_COVALENT_RADIUS_FALLBACK,
    _NON_SLAB_Z_HI_FROM_NN_SCALE,
    _NON_SLAB_Z_LO_FROM_NN_SCALE,
    _SITE_Z_RADIUS_REFERENCE_ANGSTROM,
    _SITE_Z_RADIUS_SHIFT_SCALE,
    DEFAULT_SYMMETRY_TOLERANCE,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Z-base range computation (used by generators.py)
# ---------------------------------------------------------------------------


def _get_site_surface_radii(
    slab: Atoms,
    site: dict[str, object] | None = None,
) -> float | None:
    """Mean covalent radius of framework atoms nearest to the placement site."""
    from .sites import _derive_top_layer_tolerance, _get_covalent_radius

    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    cell = np.asarray(slab.get_cell(), dtype=float)

    if site is not None and "slab_indices" in site:
        indices = site["slab_indices"]
        if not indices:
            indices = None
    else:
        indices = None

    if indices is None:
        top_depth = _derive_top_layer_tolerance(positions, symbols)
        from .sites import _top_layer_mask_by_normal

        top_mask = _top_layer_mask_by_normal(positions, cell, float(top_depth))
        indices = tuple(int(i) for i in np.nonzero(top_mask)[0])

    radii = [_get_covalent_radius(symbols[int(i)]) for i in indices]
    radii = [r for r in radii if r is not None]
    if not radii:
        return None
    return float(np.mean(radii))


def _compute_site_z_base(
    config,
    slab: Atoms,
    site: dict[str, object] | None,
    mol_symbols: list[str],
) -> tuple[float, float]:
    """Compute z-offset range for placement above *site*."""
    z_lo, z_hi = config.placement_z_range

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    if (
        mat_type != "slab"
        and site is not None
        and site.get("nn_distance") is not None
        and str(site.get("site_type", "")) != "pore"
    ):
        nn = float(site["nn_distance"])
        nn_lo = nn * _NON_SLAB_Z_LO_FROM_NN_SCALE
        nn_hi = nn * _NON_SLAB_Z_HI_FROM_NN_SCALE
        if nn_hi - nn_lo < z_hi - z_lo:
            z_lo, z_hi = nn_lo, nn_hi

    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi
    if site is not None and str(site.get("site_type", "")) == "pore":
        return z_lo, z_hi

    r_surface = _get_site_surface_radii(slab, site)
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else _MOL_COVALENT_RADIUS_FALLBACK

    if r_surface is None:
        return z_lo, z_hi

    delta = _SITE_Z_RADIUS_SHIFT_SCALE * (
        r_mol + r_surface - _SITE_Z_RADIUS_REFERENCE_ANGSTROM
    )
    return z_lo + delta, z_hi + delta


# ---------------------------------------------------------------------------
# Public API (for external imports)
# ---------------------------------------------------------------------------


def get_symmetry_info(
    slab: Atoms,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, object]:
    """Symmetry metadata including spglib space group."""
    symmetry_analyzer = SymmetryAnalyzer(slab, symmetry_tolerance=symmetry_tolerance)
    return symmetry_analyzer.get_symmetry_info()

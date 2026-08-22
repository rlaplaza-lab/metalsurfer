"""Prep-time substrate freeze policy and FixAtoms helpers.

Freeze masks (which substrate atoms stay fixed during placement relaxation)
live here. Site-enumeration top-layer masks are separate — see
:func:`~metalsurfer.placement.site_coords.top_layer_mask_by_normal`.
"""

import logging
from collections import Counter

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from .._numeric_defaults import DEFAULT_TOP_LAYER_TOLERANCE
from ..placement._material import validate_material_type
from ..placement.site_coords import (
    derive_pore_threshold,
    top_layer_mask_by_normal,
)
from ..placement.site_enumeration import get_unified_sites

logger = logging.getLogger(__name__)

_DEFAULT_FROZEN_SUBSTRATE_DISPLACEMENT_TOL_ANG = 0.01

__all__ = [
    "identify_relaxable_surface_indices",
    "top_layer_indices_by_height",
    "compute_frozen_indices",
    "frozen_indices_from_constraints",
    "max_frozen_substrate_displacement",
    "check_frozen_substrate_displacement",
    "format_atom_index_ranges",
    "log_substrate_freeze_policy",
]


def top_layer_indices_by_height(
    positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float,
) -> list[int]:
    """Return atom indices within *tolerance* of the max height along the slab normal.

    Primary height band only — calls
    :func:`~metalsurfer.placement.site_coords.top_layer_mask_by_normal` with
    ``include_terrace=False`` so freeze/alloy paths do not expand stepped terraces.

    Parameters
    ----------
    positions
        Atom positions array.
    cell
        Unit cell matrix.
    tolerance
        Height tolerance in Å.
    """
    mask = top_layer_mask_by_normal(
        positions,
        cell,
        tolerance,
        include_terrace=False,
    )
    return [int(i) for i in np.nonzero(mask)[0]]


def identify_relaxable_surface_indices(
    slab: Atoms,
    *,
    material_type: str = "slab",
    tolerance: float = DEFAULT_TOP_LAYER_TOLERANCE,
    pore_threshold: float | None = None,
) -> list[int]:
    """Return substrate atom indices left free when ``relax_top_layer=True``.

    - **slab:** primary height band along the slab normal (within *tolerance* of
      the maximum height), via
      :func:`top_layer_indices_by_height` with terrace expansion disabled.
    - **nanoparticle:** outermost shell (within *tolerance* of the maximum
      distance from the centre of mass).
    - **porous:** framework atoms on pore walls — closest neighbour of each
      pore-classified Voronoi void site.

    Parameters
    ----------
    slab
        ASE Atoms object.
    material_type
        Type of material: ``"slab"``, ``"nanoparticle"``, or ``"porous"``.
    tolerance
        Distance tolerance in Å.
    pore_threshold
        Pore threshold for porous materials.
    """
    validate_material_type(material_type)

    positions = slab.get_positions()
    n_atoms = len(positions)
    if n_atoms == 0:
        return []

    if material_type == "slab":
        cell = np.asarray(slab.get_cell(), dtype=float)
        return top_layer_indices_by_height(positions, cell, float(tolerance))

    if material_type == "nanoparticle":
        com = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - com, axis=1)
        r_max = float(np.max(dists))
        return [int(i) for i, d in enumerate(dists) if d >= r_max - float(tolerance)]

    symbols = slab.get_chemical_symbols()
    if pore_threshold is None:
        pore_threshold = derive_pore_threshold(symbols)

    sites = get_unified_sites(
        slab,
        material_type="porous",
        top_layer_tolerance=float(tolerance),
        pore_threshold=float(pore_threshold),
        enrich=False,
    )
    boundary: set[int] = set()
    for site in sites:
        if site.site_type != "pore":
            continue
        raw_indices = site.slab_indices
        if not raw_indices:
            continue
        idx = int(raw_indices[0])
        if 0 <= idx < n_atoms:
            boundary.add(idx)

    if not boundary:
        logger.warning(
            "Relax_top_layer=True on porous substrate identified no pore-boundary "
            "atoms; freezing entire substrate during placement relaxation"
        )
    return sorted(boundary)


def compute_frozen_indices(
    slab: Atoms,
    *,
    relax_top_layer: bool = False,
    freeze_symbols: list[str] | None = None,
    top_layer_tolerance: float = DEFAULT_TOP_LAYER_TOLERANCE,
    material_type: str = "slab",
    pore_threshold: float | None = None,
) -> list[int]:
    """Determine which slab atom indices should be frozen during optimisation.

    Prep-time policy helper used by :func:`~metalsurfer.surface_prep.apply_surface_constraints`.
    Default policy: freeze the entire substrate (``relax_top_layer=False``).
    If ``relax_top_layer`` is ``True``, only the interior is frozen; which atoms
    remain free depends on *material_type* (see
    :func:`identify_relaxable_surface_indices`).
    If ``freeze_symbols`` is a non-empty list, only atoms whose symbol is in
    that list are frozen (regardless of layer). ``None`` and ``[]`` both mean
    "use the default layer policy" (freeze everything when
    ``relax_top_layer=False``).

    Parameters
    ----------
    slab
        ASE Atoms object.
    relax_top_layer
        If True, leave the top layer free.
    freeze_symbols
        Chemical symbols to freeze. Empty or ``None`` → layer policy.
    top_layer_tolerance
        Height tolerance for the top layer in Å.
    material_type
        Type of material.
    pore_threshold
        Pore threshold for porous materials.
    """
    n_slab = len(slab)

    if freeze_symbols:
        syms = slab.get_chemical_symbols()
        return [i for i, s in enumerate(syms) if s in freeze_symbols]

    if not relax_top_layer:
        return list(range(n_slab))

    free_indices = set(
        identify_relaxable_surface_indices(
            slab,
            material_type=material_type,
            tolerance=top_layer_tolerance,
            pore_threshold=pore_threshold,
        )
    )
    return [i for i in range(n_slab) if i not in free_indices]


def frozen_indices_from_constraints(atoms: Atoms) -> list[int]:
    """Return frozen atom indices from ASE ``FixAtoms`` constraints on *atoms*.

    Parameters
    ----------
    atoms
        ASE Atoms object.
    """
    indices: list[int] = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices.extend(int(i) for i in np.atleast_1d(constraint.index))
    return sorted(set(indices))


def max_frozen_substrate_displacement(
    optimized: Atoms,
    reference_slab: Atoms,
    *,
    slab_size: int | None = None,
    frozen_indices: list[int] | None = None,
) -> float:
    """Maximum Cartesian displacement (Å) among constrained substrate atoms.

    Parameters
    ----------
    optimized
        Optimized ASE Atoms structure.
    reference_slab
        Reference slab ASE Atoms.
    slab_size
        Number of atoms in the substrate. Defaults to length of *reference_slab*.
    frozen_indices
        Indices of frozen atoms. Defaults to constraints on *reference_slab*.
    """
    if slab_size is None:
        slab_size = len(reference_slab)
    if frozen_indices is None:
        frozen_indices = frozen_indices_from_constraints(reference_slab)
    if not frozen_indices:
        return 0.0
    ref_pos = reference_slab.get_positions()
    opt_pos = optimized.get_positions()
    max_disp = 0.0
    for idx in frozen_indices:
        if idx >= slab_size:
            continue
        max_disp = max(max_disp, float(np.linalg.norm(opt_pos[idx] - ref_pos[idx])))
    return max_disp


def check_frozen_substrate_displacement(
    optimized: Atoms,
    reference_slab: Atoms,
    *,
    slab_size: int | None = None,
    tolerance_ang: float = _DEFAULT_FROZEN_SUBSTRATE_DISPLACEMENT_TOL_ANG,
) -> tuple[bool, str]:
    """Return whether *optimized* kept FixAtoms substrate indices fixed.

    Parameters
    ----------
    optimized
        Optimized ASE Atoms structure.
    reference_slab
        Reference slab ASE Atoms.
    slab_size
        Number of atoms in the substrate. Defaults to length of *reference_slab*.
    tolerance_ang
        Displacement tolerance in Å.
    """
    max_disp = max_frozen_substrate_displacement(
        optimized,
        reference_slab,
        slab_size=slab_size,
    )
    frozen = frozen_indices_from_constraints(reference_slab)
    if not frozen:
        return True, "no FixAtoms constraints on reference slab"
    if max_disp > tolerance_ang:
        return (
            False,
            f"frozen substrate atoms displaced up to {max_disp:.4f} A "
            f"(tolerance {tolerance_ang:.4f} A)",
        )
    return True, f"frozen substrate displacement {max_disp:.6f} A"


def format_atom_index_ranges(indices: list[int]) -> str:
    """Format sorted atom indices as compact ranges (e.g. ``0-31, 40-47``).

    Parameters
    ----------
    indices
        List of atom indices.
    """
    if not indices:
        return "(none)"
    sorted_idx = sorted(set(indices))
    parts: list[str] = []
    start = prev = sorted_idx[0]
    for idx in sorted_idx[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = idx
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(parts)


def _symbol_count_label(indices: list[int], symbols: list[str]) -> str:
    counts = Counter(symbols[i] for i in indices)
    return ", ".join(f"{sym}×{n}" for sym, n in sorted(counts.items()))


def log_substrate_freeze_policy(
    substrate: Atoms,
    *,
    context: str = "Substrate",
) -> None:
    """Log which substrate atoms are frozen vs free during placement relaxation.

    Parameters
    ----------
    substrate
        ASE Atoms object.
    context
        Prefix string for log messages.
    """
    n_substrate = len(substrate)
    symbols = substrate.get_chemical_symbols()
    frozen = frozen_indices_from_constraints(substrate)
    frozen_set = set(frozen)
    moving = [i for i in range(n_substrate) if i not in frozen_set]

    if not frozen:
        logger.info(
            "%s freeze policy: no FixAtoms on %d substrate atoms — all substrate "
            "atoms free to move during placement relaxation",
            context,
            n_substrate,
        )
        return

    if not moving:
        logger.info(
            "%s freeze policy: all %d substrate atoms frozen during placement "
            "relaxation (%s; indices %s)",
            context,
            n_substrate,
            _symbol_count_label(frozen, symbols),
            format_atom_index_ranges(frozen),
        )
        return

    logger.info(
        "%s freeze policy: %d/%d substrate atoms frozen, %d free to move during "
        "placement relaxation",
        context,
        len(frozen),
        n_substrate,
        len(moving),
    )
    logger.info(
        "  frozen (%s): indices %s",
        _symbol_count_label(frozen, symbols),
        format_atom_index_ranges(frozen),
    )
    logger.info(
        "  moving (%s): indices %s",
        _symbol_count_label(moving, symbols),
        format_atom_index_ranges(moving),
    )

"""Material-type detection helpers shared by geometry and sites modules.

Kept in a separate module so that both :mod:`geometry` (which needs
:func:`material_aware_pbc`) and :mod:`sites` (which defines site detection
but imports covalent radii from :mod:`geometry`) can import these utilities
without creating a circular dependency.
"""

import numpy as np
from ase import Atoms

# Slab vacuum detection: atom z-extent / cell-c-length < this → classify as slab
_SLAB_VACUUM_FRACTION = 0.70


def detect_material_type(atoms: Atoms) -> str:
    """Infer material type: ``"slab"``, ``"nanoparticle"``, or ``"porous"``.

    - nanoparticle: no PBC in any direction.
    - slab: periodic in xy (or all three) with a vacuum gap along z.
    - porous: fully 3D-periodic with no significant vacuum gap.
    """
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    if not np.any(pbc):
        return "nanoparticle"

    cell = np.asarray(atoms.get_cell(), dtype=float)
    positions = atoms.get_positions()

    if np.linalg.det(cell) > 0:
        c_length = float(np.linalg.norm(cell[2]))
        if c_length > 0:
            z_span = float(np.max(positions[:, 2]) - np.min(positions[:, 2]))
            if z_span / c_length < _SLAB_VACUUM_FRACTION:
                return "slab"

    if bool(pbc[0]) and bool(pbc[1]) and not bool(pbc[2]):
        return "slab"

    if np.all(pbc):
        return "porous"

    return "slab"


def material_aware_pbc(slab: Atoms) -> list[bool]:
    """Return PBC flags appropriate for distance calculations on *slab*.

    - slab: ``[True, True, False]`` — periodic in xy, free in z.
    - porous: ``[True, True, True]`` — fully 3D periodic.
    - nanoparticle: ``[False, False, False]`` — no PBC.
    """
    mat_type = detect_material_type(slab)
    if mat_type == "porous":
        return [True, True, True]
    if mat_type == "nanoparticle":
        return [False, False, False]
    return [True, True, False]


def _resolve_material_type(
    site: dict[str, object] | None,
    fallback: str = "slab",
) -> str:
    """Resolve material type from a site dictionary with a deterministic fallback."""
    if site is None:
        return fallback
    material_type = site.get("material_type")
    if material_type is None:
        return fallback
    return str(material_type)

"""Material-type helpers shared by geometry and site-enumeration modules.

Kept in a separate module so that both :mod:`geometry` (which needs
:func:`material_aware_pbc`) and site enumeration / Voronoi helpers can import
these utilities without creating a circular dependency.
"""

from typing import Literal

import numpy as np
from ase import Atoms

from ..exceptions import GeometryValidationError
from .site_types import Site

# Shared PBC flags for distance math, ASE layout, and substrate validation.
MATERIAL_PBC: dict[str, tuple[bool, bool, bool]] = {
    "slab": (True, True, False),
    "porous": (True, True, True),
    "nanoparticle": (False, False, False),
}


def validate_material_type(material_type: str) -> str:
    """Validate and return a supported material type string."""
    material_aware_pbc(material_type)
    return material_type


def material_aware_pbc(material_type: str) -> list[bool]:
    """Return PBC flags for distance calculations given explicit *material_type*.

    - slab: ``[True, True, False]`` — periodic in xy, free in z.
    - porous: ``[True, True, True]`` — fully 3D periodic.
    - nanoparticle: ``[False, False, False]`` — no PBC.

    Parameters
    ----------
    material_type
        Material type string (``"slab"``, ``"porous"``, or ``"nanoparticle"``).
    """
    try:
        return list(MATERIAL_PBC[material_type])
    except KeyError as exc:
        raise ValueError(
            f"material_type must be one of {sorted(MATERIAL_PBC)}, "
            f"got {material_type!r}"
        ) from exc


def require_atoms_pbc_matches_material(
    atoms: Atoms,
    material_type: str,
    *,
    context: str = "substrate",
) -> np.ndarray:
    """Return *atoms* PBC after checking it matches *material_type*.

    Geometry follows the ASE mask once the mask is one of the three legal
    patterns. A mismatch or any other combination raises
    :class:`~metalsurfer.exceptions.GeometryValidationError` — PBC is never
    rewritten to fit the type.

    Parameters
    ----------
    atoms
        ASE :class:`~ase.Atoms` substrate.
    material_type
        Declared material type.
    context
        Label for the error message.
    """
    validate_material_type(material_type)
    expected = material_aware_pbc(material_type)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool).reshape(3)
    if not np.array_equal(pbc, np.asarray(expected, dtype=bool)):
        raise GeometryValidationError(
            f"{context} PBC {pbc.tolist()} is inconsistent with "
            f"material_type={material_type!r} (expected {expected}). "
            "Set PBC on the ASE Atoms object during preparation; "
            "metalsurfer does not rewrite mismatched flags."
        )
    return pbc.copy()


def symmetry_mode_for_material(
    material_type: str,
) -> Literal["periodic", "cluster"]:
    """Cluster for nanoparticles; periodic for slabs and porous frameworks."""
    validate_material_type(material_type)
    if material_type == "nanoparticle":
        return "cluster"
    return "periodic"


def calculator_pbc_for_atoms(atoms: Atoms) -> list[bool]:
    """Return PBC flags legal for the UMA/FairChem calculator.

    Slab-style ``[True, True, False]`` and fully periodic substrates map to
    ``[True, True, True]``; non-periodic clusters stay ``[False, False, False]``.
    Call at the calculator boundary on copies so stored substrate PBC is unchanged.

    Parameters
    ----------
    atoms
        ASE :class:`~ase.Atoms` object.
    """
    if any(atoms.get_pbc()):
        return [True, True, True]
    return [False, False, False]


def material_type_for_placement(
    site: Site | None,
    *,
    when_no_site: str,
) -> str:
    """Return ``site.material_type`` or *when_no_site* when *site* is None.

    Parameters
    ----------
    site
        Adsorption :class:`Site` or None.
    when_no_site
        Fallback material type when *site* is None. Must be an explicit
        caller-supplied type (never inferred).
    """
    if site is None:
        return validate_material_type(when_no_site)
    return str(site.material_type)

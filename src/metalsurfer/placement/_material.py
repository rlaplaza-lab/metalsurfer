"""Material-type helpers shared by geometry and site-enumeration modules.

Kept in a separate module so that both :mod:`geometry` (which needs
:func:`material_aware_pbc`) and site enumeration / Voronoi helpers can import
these utilities without creating a circular dependency.
"""

from ase import Atoms

from .site_types import Site

# Shared PBC flags for distance math, ASE layout, and substrate validation.
MATERIAL_PBC: dict[str, tuple[bool, bool, bool]] = {
    "slab": (True, True, False),
    "porous": (True, True, True),
    "nanoparticle": (False, False, False),
}


def material_aware_pbc(material_type: str) -> list[bool]:
    """Return PBC flags for distance calculations given explicit *material_type*.

    - slab: ``[True, True, False]`` — periodic in xy, free in z.
    - porous: ``[True, True, True]`` — fully 3D periodic.
    - nanoparticle: ``[False, False, False]`` — no PBC.
    """
    try:
        return list(MATERIAL_PBC[material_type])
    except KeyError as exc:
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', "
            f"got {material_type!r}"
        ) from exc


def calculator_pbc_for_atoms(atoms: Atoms) -> list[bool]:
    """Return PBC flags legal for the UMA/FairChem calculator.

    Slab-style ``[True, True, False]`` and fully periodic substrates map to
    ``[True, True, True]``; non-periodic clusters stay ``[False, False, False]``.
    Call at the calculator boundary on copies so stored substrate PBC is unchanged.
    """
    if any(atoms.get_pbc()):
        return [True, True, True]
    return [False, False, False]


def material_type_for_placement(
    site: Site | None,
    *,
    when_no_site: str,
) -> str:
    """Return ``site.material_type`` or *when_no_site* when *site* is None."""
    if site is None:
        return when_no_site
    return str(site.material_type)

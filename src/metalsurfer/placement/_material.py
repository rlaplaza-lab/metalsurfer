"""Material-type helpers shared by geometry and sites modules.

Kept in a separate module so that both :mod:`geometry` (which needs
:func:`material_aware_pbc`) and :mod:`sites` (which defines site detection
but imports covalent radii from :mod:`geometry`) can import these utilities
without creating a circular dependency.
"""


def material_aware_pbc(material_type: str) -> list[bool]:
    """Return PBC flags for distance calculations given explicit *material_type*.

    - slab: ``[True, True, False]`` — periodic in xy, free in z.
    - porous: ``[True, True, True]`` — fully 3D periodic.
    - nanoparticle: ``[False, False, False]`` — no PBC.
    """
    if material_type == "porous":
        return [True, True, True]
    if material_type == "nanoparticle":
        return [False, False, False]
    if material_type == "slab":
        return [True, True, False]
    raise ValueError(
        f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
    )


def material_type_for_placement(
    site: dict[str, object] | None,
    *,
    when_no_site: str,
) -> str:
    """Return ``site['material_type']`` or *when_no_site* when *site* is None.

    When *site* is provided, it must be a site dict from
    :func:`sites.get_unified_sites` (or equivalent) including ``material_type``.
    """
    if site is None:
        return when_no_site
    material_type = site.get("material_type")
    if material_type is None:
        raise ValueError(
            "placement site dict must include 'material_type' (use get_unified_sites)"
        )
    return str(material_type)

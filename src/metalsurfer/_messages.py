"""Shared user-facing message literals."""

RDKIT_INSTALL_HINT = "Install it with: pip install rdkit"
MISSING_REFERENCE_ENERGY_FMT = "missing reference energy for %s"
NUM_PLACEMENTS_MUST_BE_SET = "num_placements must be set"
SKIP_DESORPTION_CHECK_WARNING = (
    "skip_desorption_check=True: desorption distance checks disabled; "
    "results may include desorbed structures"
)
UNREADABLE_EXISTING_FILE_WARNING = (
    "Could not read existing %s (%s); it will be replaced"
)


def missing_required_column(key: str) -> str:
    return f"missing required column {key!r}"


def missing_reference_energy(molecule_name: str) -> str:
    return MISSING_REFERENCE_ENERGY_FMT % molecule_name


def num_placements_must_be_set(suffix: str = "") -> str:
    if not suffix:
        return NUM_PLACEMENTS_MUST_BE_SET
    return f"{NUM_PLACEMENTS_MUST_BE_SET} {suffix}"


def could_not_generate_conformers(molecule_name: str) -> str:
    return f"could not generate conformers for {molecule_name}"


def _format_optional_distance(value: float | None) -> str:
    if value is None:
        return "auto"
    return f"{float(value):.2f}"


def no_accessible_sites_message(
    n_atoms: int,
    *,
    probe_radius: float | None,
    max_distance: float | None,
    material_type: str,
    unified: bool = False,
) -> str:
    """Format the empty-site warning used by site enumeration / plugins / context."""
    prefix = (
        "Unified site detection found no sites" if unified else "No accessible sites"
    )
    return (
        f"{prefix} for {n_atoms}-atom structure "
        f"(probe_radius={_format_optional_distance(probe_radius)}, "
        f"max_distance={_format_optional_distance(max_distance)}, "
        f"material_type={material_type!r})"
    )

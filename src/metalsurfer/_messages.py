"""Shared user-facing message literals."""

RDKIT_INSTALL_HINT = "Install it with: pip install rdkit"
MISSING_REFERENCE_ENERGY_FMT = "missing reference energy for %s"
NUM_PLACEMENTS_MUST_BE_SET = "num_placements must be set"


def missing_required_column(key: str) -> str:
    return f"missing required column {key!r}"


def missing_reference_energy(molecule_name: str) -> str:
    return MISSING_REFERENCE_ENERGY_FMT % molecule_name


def num_placements_must_be_set(suffix: str = "") -> str:
    if not suffix:
        return NUM_PLACEMENTS_MUST_BE_SET
    return f"{NUM_PLACEMENTS_MUST_BE_SET} {suffix}"

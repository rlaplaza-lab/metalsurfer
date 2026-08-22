"""Canonical results-directory and artifact-path helpers.

Writers and CSV row builders must agree on the layout under
``results_{surface_type}/``.
"""

from pathlib import Path


def results_dir_for(surface_type: str) -> Path:
    """Return ``results_{surface_type}/`` for a campaign *surface_type* label.

    Parameters
    ----------
    surface_type
        Campaign surface type label.
    """
    return Path(f"results_{surface_type}")


def molecule_all_xyz_dir(results_dir: Path | str, molecule_name: str) -> Path:
    """Directory for full-slab XYZ of all placements of *molecule_name*."""
    return Path(results_dir) / "xyz_structures" / f"{molecule_name}_all"


def molecule_adsorbate_only_dir(results_dir: Path | str, molecule_name: str) -> Path:
    """Directory for adsorbate-only XYZ of all placements of *molecule_name*."""
    return Path(results_dir) / "xyz_structures" / f"{molecule_name}_adsorbate_only"


def molecule_all_vasp_dir(results_dir: Path | str, molecule_name: str) -> Path:
    """Directory for VASP inputs of all placements of *molecule_name*."""
    return Path(results_dir) / "vasp_inputs" / f"{molecule_name}_all"


def saturation_xyz_dir(results_dir: Path | str, label: str) -> Path:
    """Directory for saturation XYZ under *label* (molecule or combo)."""
    return Path(results_dir) / "xyz_structures" / f"{label}_saturation"


def saturation_vasp_dir(results_dir: Path | str, label: str) -> Path:
    """Directory for saturation VASP inputs under *label* (molecule or combo)."""
    return Path(results_dir) / "vasp_inputs" / f"{label}_saturation"

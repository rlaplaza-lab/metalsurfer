"""Typed domain models for adsorption screening results."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from ase import Atoms


@dataclass
class ReferenceEnergies:
    """Clean-slab and isolated-molecule reference energies (eV)."""

    slab_energy: float
    molecule_energies: dict[str, float] = field(default_factory=dict)

    def get_molecule_energy(self, molecule_name: str) -> float | None:
        return self.molecule_energies.get(molecule_name)


@dataclass
class PlacementSpec:
    """Input to placement generator; fully deterministic."""

    conformer_index: int
    orientation_type: Literal["parallel", "EN-down", "vertical", "round"]
    face_flip: bool
    en_atom_index: int | None
    site_index: int  # -1 for random
    site_type: str | None  # "atop", "bridge", "hollow", or None for random
    tilt_deg: float
    azimuth_deg: float
    azimuth_in_plane_deg: float
    z_fraction: float  # 0-1; maps to z_base_lo..z_base_hi
    placement_index: int


@dataclass
class PlacementDescriptor:
    """Output: spec + actual placement values for reproducibility and trend analysis."""

    conformer_index: int
    orientation_type: Literal["parallel", "EN-down", "vertical", "round"]
    face_flip: bool
    en_atom_index: int | None
    site_index: int
    site_type: str | None
    tilt_deg: float
    azimuth_deg: float
    azimuth_in_plane_deg: float
    z_fraction: float
    placement_index: int
    x: float
    y: float
    z: float
    shape: str  # "linear", "flat", "round"
    slab_indices: tuple[int, ...] | None = None  # Surface atom indices for traceability


@dataclass
class ScreeningResult:
    """Single validated placement after optimisation and filtering."""

    molecule: str
    placement_id: int  # Same as placement_descriptor.placement_index
    energy_adslab: float
    energy_slab: float
    energy_adsorbate: float
    energy_adsorption: float
    atoms: Atoms
    distance: float
    placement_descriptor: "PlacementDescriptor"


@dataclass
class TimingInfo:
    """Per-molecule wall-clock timing breakdown (seconds)."""

    molecule: str
    conformer_generation_s: float = 0.0
    placement_generation_s: float = 0.0
    optimization_s: float = 0.0
    validation_s: float = 0.0
    filtering_s: float = 0.0
    total_s: float = 0.0
    n_placements_attempted: int = 0
    n_placements_valid: int = 0
    n_results_after_filter: int = 0


@dataclass
class MoleculeSummary:
    """Aggregate adsorption-energy statistics for one molecule."""

    molecule: str
    n_configurations: int
    e_ads_min: float
    e_ads_max: float
    e_ads_mean: float
    e_ads_std: float
    e_ads_median: float
    best_placement_id: int
    e_ads_best: float


def build_molecule_summary(
    molecule_name: str,
    results: list[ScreeningResult],
) -> MoleculeSummary:
    """Compute aggregate statistics for a set of screening results."""
    if not results:
        raise ValueError("Cannot build molecule summary from empty results")
    energies = [r.energy_adsorption for r in results]
    arr = np.array(energies)
    best_idx = int(np.argmin(arr))
    return MoleculeSummary(
        molecule=molecule_name,
        n_configurations=len(results),
        e_ads_min=float(arr.min()),
        e_ads_max=float(arr.max()),
        e_ads_mean=float(arr.mean()),
        e_ads_std=float(arr.std()),
        e_ads_median=float(np.median(arr)),
        best_placement_id=results[best_idx].placement_id,
        e_ads_best=results[best_idx].energy_adsorption,
    )


@dataclass
class ScreeningRunResult:
    """All results for a single molecule within a screening run."""

    molecule: str
    results: list[ScreeningResult]
    timing: TimingInfo | None = None
    summary: MoleculeSummary | None = None


@dataclass
class SaturationStepResult:
    """Result of one step in a sequential saturation run."""

    step: int
    molecule: str
    n_molecules_on_slab: int
    best_result: ScreeningResult
    all_results: list[ScreeningResult]


@dataclass
class SaturationRunResult:
    """Full saturation run for one molecule: steps until slab is saturated."""

    molecule: str
    steps: list[SaturationStepResult]
    n_molecules_at_saturation: int
    final_slab_atoms: Atoms | None = None

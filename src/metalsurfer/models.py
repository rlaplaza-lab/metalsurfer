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
    orientation_type: Literal[
        "parallel", "EN-down", "vertical", "round", "dissociative"
    ]
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
class PlacementPose:
    """Universal pose parameters for deterministic placement generation."""

    conformer_index: int
    site_index: int
    site_type: str | None
    placement_index: int
    quat_w: float
    quat_x: float
    quat_y: float
    quat_z: float
    x_abs: float
    y_abs: float
    z_fraction: float
    z_abs: float | None = None
    orientation_type: (
        Literal["parallel", "EN-down", "vertical", "round", "dissociative"] | None
    ) = None
    face_flip: bool = False
    en_atom_index: int | None = None
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    azimuth_in_plane_deg: float = 0.0


@dataclass
class PlacementDescriptor:
    """Output: spec + actual placement values for reproducibility and trend analysis."""

    conformer_index: int
    orientation_type: Literal[
        "parallel", "EN-down", "vertical", "round", "dissociative"
    ]
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
    z_offset: float
    x_abs: float | None = None
    y_abs: float | None = None
    surface_ref_z_abs: float | None = None
    z_abs: float | None = None
    shape: str = "round"  # "linear", "flat", "round"
    slab_indices: tuple[int, ...] | None = None  # Surface atom indices for traceability
    placement_mode_resolved: str = "no_sites"
    site_source: str = "no_sites"
    site_reference_frame: str = "global_top_layer"
    site_xy_frac_a: float | None = None
    site_xy_frac_b: float | None = None
    quat_w: float | None = None
    quat_x: float | None = None
    quat_y: float | None = None
    quat_z: float | None = None


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
    slab_size: int
    distance: float
    placement_descriptor: PlacementDescriptor


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
    bo_transfer_enabled: bool = False
    bo_transfer_used: bool = False
    bo_transfer_disabled_reason: str | None = None
    bo_transfer_weight_share: float = 0.0
    bo_transfer_bad_rounds: int = 0
    bo_transfer_last_mae_delta: float | None = None


@dataclass
class SaturationRunResult:
    """Full saturation run for one molecule: steps until slab is saturated."""

    molecule: str
    steps: list[SaturationStepResult]
    n_molecules_at_saturation: int
    final_slab_atoms: Atoms | None = None


@dataclass
class BOStepMemory:
    """Transferable BO memory captured from one saturation step."""

    observed_X_rows: list[dict[str, float]] = field(default_factory=list)
    observed_y: list[float] = field(default_factory=list)
    best_energy: float | None = None


@dataclass
class MultiMolSaturationStepResult:
    """Result of one step in a multi-molecule saturation run.

    All molecules compete at each step; the winner (lowest E_ads) advances
    the slab state. Per-molecule results and placement budgets are stored
    for analysis.
    """

    step: int
    winning_molecule: str
    n_molecules_on_slab: int
    best_result: ScreeningResult
    per_molecule_results: dict[str, list[ScreeningResult]]
    per_molecule_budgets: dict[str, int]
    bo_transfer_enabled: bool = False
    bo_transfer_used: dict[str, bool] = field(default_factory=dict)
    bo_transfer_disabled_reason: dict[str, str | None] = field(default_factory=dict)
    bo_transfer_weight_share: dict[str, float] = field(default_factory=dict)
    bo_transfer_bad_rounds: dict[str, int] = field(default_factory=dict)
    bo_transfer_last_mae_delta: dict[str, float | None] = field(default_factory=dict)


@dataclass
class MultiMolSaturationRunResult:
    """Full multi-molecule saturation run: molecules compete at each step.

    All molecules in *molecules* are evaluated at every saturation step.
    The placement budget is distributed proportionally to molecular complexity
    (number of enumerable placement specs). The step winner (molecule with
    the lowest adsorption energy) advances the slab state.
    """

    molecules: list[str]
    steps: list[MultiMolSaturationStepResult]
    n_molecules_at_saturation: int
    final_slab_atoms: Atoms | None = None
    molecule_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class MoleculeCampaignSummary:
    """Compact campaign-facing summary for a molecule."""

    molecule: str
    n_valid_placements: int
    best_adsorption_energy: float | None
    n_parallel: int = 0
    n_endown: int = 0


@dataclass
class BindingCampaignResult:
    """High-level result contract for script/CLI campaign runs."""

    mode: Literal["non_bo", "bo"]
    surface_type: str
    run_results: list[ScreeningRunResult]
    molecule_summaries: list[MoleculeCampaignSummary]
    total_configurations: int
    n_molecules: int
    t_ref_s: float
    t_total_s: float
    failure_summaries: dict[str, dict[str, object]] = field(default_factory=dict)

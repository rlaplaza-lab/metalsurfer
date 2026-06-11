"""Typed domain models for adsorption screening results."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
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

    def to_row(self) -> dict[str, Any]:
        """Convert descriptor fields to a flat row for tabular exports."""
        x_abs = self.x_abs if self.x_abs is not None else self.x
        y_abs = self.y_abs if self.y_abs is not None else self.y
        surface_ref_z_abs = (
            self.surface_ref_z_abs if self.surface_ref_z_abs is not None else 0.0
        )
        z_abs = (
            self.z_abs if self.z_abs is not None else surface_ref_z_abs + self.z_offset
        )
        row: dict[str, Any] = {
            "conformer_index": self.conformer_index,
            "orientation_type": self.orientation_type,
            "face_flip": self.face_flip,
            "en_atom_index": self.en_atom_index,
            "site_index": self.site_index,
            "site_type": self.site_type,
            "tilt_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "azimuth_in_plane_deg": self.azimuth_in_plane_deg,
            "z_fraction": self.z_fraction,
            "x_abs": x_abs,
            "y_abs": y_abs,
            "z_offset": self.z_offset,
            "surface_ref_z_abs": surface_ref_z_abs,
            "z_abs": z_abs,
            "shape": self.shape,
            "placement_mode_resolved": self.placement_mode_resolved,
            "site_source": self.site_source,
            "site_reference_frame": self.site_reference_frame,
            "site_xy_frac_a": self.site_xy_frac_a,
            "site_xy_frac_b": self.site_xy_frac_b,
            "quat_w": float(self.quat_w) if self.quat_w is not None else 1.0,
            "quat_x": float(self.quat_x) if self.quat_x is not None else 0.0,
            "quat_y": float(self.quat_y) if self.quat_y is not None else 0.0,
            "quat_z": float(self.quat_z) if self.quat_z is not None else 0.0,
        }
        if self.slab_indices is not None:
            row["slab_indices"] = ",".join(str(i) for i in self.slab_indices)
        return row


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

    def to_row(
        self,
        *,
        xyz_path: str | None = None,
        poscar_path: str | None = None,
        context_row: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return a flat row for CSV/dataframe export."""
        row: dict[str, Any] = {
            "molecule": self.molecule,
            "placement_id": self.placement_id,
            "energy_adslab": self.energy_adslab,
            "energy_slab": self.energy_slab,
            "energy_adsorbate": self.energy_adsorbate,
            "energy_adsorption": self.energy_adsorption,
            "distance": self.distance,
        }
        if xyz_path is not None:
            row["xyz_path"] = xyz_path
        if poscar_path is not None:
            row["poscar_path"] = poscar_path
        row.update(self.placement_descriptor.to_row())
        if context_row:
            row.update(dict(context_row))
        return row


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


def _format_failure_summary_text(failure_summary: dict[str, object]) -> str:
    """Produce a human-readable multi-line summary from a failure_summary dict."""
    lines = ["Failure summary:"]
    stage = failure_summary.get("stage", "unknown")
    lines.append(f"  Stage: {stage}")

    if stage in {"reference", "conformers"}:
        reason = failure_summary.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    elif stage == "placement":
        n_attempted = failure_summary.get("n_placements_attempted", "?")
        n_initial = failure_summary.get("n_initial_placements", 0)
        lines.append(f"  Placements attempted: {n_attempted}")
        lines.append(f"  Initial placements: {n_initial}")
        if "n_candidate_specs" in failure_summary:
            lines.append(
                f"  Candidate specs: {failure_summary.get('n_candidate_specs', '?')}"
            )
        if "n_valid_pool" in failure_summary:
            lines.append(f"  Valid pool: {failure_summary.get('n_valid_pool', '?')}")
    elif stage == "validation":
        n_initial = failure_summary.get("n_initial_placements", "?")
        n_opt = failure_summary.get("n_optimized", "?")
        n_opt_fail = failure_summary.get("n_optimization_failed", 0)
        lines.append(f"  Initial placements: {n_initial}")
        lines.append(f"  Optimized: {n_opt} ({n_opt_fail} failed)")
        lines.append("  Passed validation: 0")
        if "n_evaluated" in failure_summary:
            lines.append(f"  BO evaluated: {failure_summary.get('n_evaluated', '?')}")
        if "n_valid_results" in failure_summary:
            lines.append(
                f"  BO valid results: {failure_summary.get('n_valid_results', '?')}"
            )
        validation_failures = failure_summary.get("validation_failures")
        if isinstance(validation_failures, dict):
            items = [
                (str(reason), int(count))
                for reason, count in validation_failures.items()
                if isinstance(count, int)
            ]
            if items:
                lines.append("  Validation failures:")
                for reason, count in sorted(items, key=lambda x: -x[1]):
                    lines.append(f"    {reason}: {count}")
    elif stage == "filter":
        n_before = failure_summary.get("n_before_filter", "?")
        n_after = failure_summary.get("n_after_filter", 0)
        lines.append(f"  Before filter: {n_before}")
        lines.append(f"  After filter: {n_after}")

    return "\n".join(lines)


@dataclass
class ScreeningRunResult:
    """All results for a single molecule within a screening run."""

    molecule: str
    results: list[ScreeningResult]
    timing: TimingInfo | None = None
    summary: MoleculeSummary | None = None

    def to_rows(
        self,
        *,
        results_dir: str | Path | None = None,
        context_row: Mapping[str, Any] | None = None,
        write_vasp_inputs: bool = False,
    ) -> list[dict[str, Any]]:
        """Flatten all placements for this molecule into detailed rows."""
        rows: list[dict[str, Any]] = []
        xyz_dir: Path | None = None
        vasp_dir: Path | None = None
        if results_dir is not None:
            base = Path(results_dir)
            xyz_dir = base / "xyz_structures" / f"{self.molecule}_all"
            if write_vasp_inputs:
                vasp_dir = base / "vasp_inputs" / f"{self.molecule}_all"

        for sr in self.results:
            pid = sr.placement_id
            xyz_path = str(xyz_dir / f"conformer_{pid:03d}.xyz") if xyz_dir else None
            poscar_path = (
                str(vasp_dir / f"conformer_{pid:03d}" / "POSCAR")
                if vasp_dir is not None
                else None
            )
            row = sr.to_row(
                xyz_path=xyz_path,
                poscar_path=poscar_path,
                context_row=context_row,
            )
            row["molecule"] = self.molecule
            rows.append(row)
        return rows

    def to_dataframe(
        self,
        *,
        results_dir: str | Path | None = None,
        context_row: Mapping[str, Any] | None = None,
        write_vasp_inputs: bool = False,
    ) -> pd.DataFrame:
        """Return a detailed pandas DataFrame for this screening run."""
        return pd.DataFrame(
            self.to_rows(
                results_dir=results_dir,
                context_row=context_row,
                write_vasp_inputs=write_vasp_inputs,
            )
        )

    def to_summary_row(self) -> dict[str, Any] | None:
        """Return one summary row for this molecule."""
        if self.summary is None:
            return None
        s = self.summary
        return {
            "molecule": s.molecule,
            "n_configurations": s.n_configurations,
            "E_ads_min": s.e_ads_min,
            "E_ads_max": s.e_ads_max,
            "E_ads_mean": s.e_ads_mean,
            "E_ads_std": s.e_ads_std,
            "E_ads_median": s.e_ads_median,
            "best_placement_id": s.best_placement_id,
            "E_ads_best": s.e_ads_best,
        }


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

    def to_detail_row(
        self,
        *,
        results_dir: str | Path,
        saturation_molecule: str,
        context_row: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one saturation detail row for the winning placement."""
        best = self.best_result
        mol_dir = (
            Path(results_dir) / "xyz_structures" / f"{saturation_molecule}_saturation"
        )
        row: dict[str, Any] = {
            "molecule": saturation_molecule,
            "step": self.step,
            "n_molecules_on_slab": self.n_molecules_on_slab,
            "bo_transfer_enabled": self.bo_transfer_enabled,
            "bo_transfer_used": self.bo_transfer_used,
            "bo_transfer_disabled_reason": self.bo_transfer_disabled_reason,
            "bo_transfer_weight_share": self.bo_transfer_weight_share,
            "bo_transfer_bad_rounds": self.bo_transfer_bad_rounds,
            "bo_transfer_last_mae_delta": self.bo_transfer_last_mae_delta,
            "placement_id": best.placement_id,
            "energy_adslab": best.energy_adslab,
            "energy_slab": best.energy_slab,
            "energy_adsorbate": best.energy_adsorbate,
            "energy_adsorption": best.energy_adsorption,
            "distance": best.distance,
            "step_structure_path": str(mol_dir / f"step_{self.step:03d}_best_slab.xyz"),
            "step_structure_energy_path": str(
                mol_dir / f"step_{self.step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
            ),
            "step_adsorbate_path": str(mol_dir / f"step_{self.step:03d}_adsorbate.xyz"),
        }
        row.update(best.placement_descriptor.to_row())
        if context_row:
            row.update(dict(context_row))
        return row

    def to_rows(
        self,
        *,
        results_dir: str | Path,
        saturation_molecule: str,
        context_row: Mapping[str, Any] | None = None,
        step_prefix: bool = True,
        write_vasp_inputs: bool = False,
    ) -> list[dict[str, Any]]:
        """Return detailed rows for every placement evaluated in this step."""
        step_placements_rel = (
            f"step_{self.step:03d}_placements" if step_prefix else "placements"
        )
        step_xyz = (
            Path(results_dir)
            / "xyz_structures"
            / f"{saturation_molecule}_saturation"
            / step_placements_rel
        )
        step_vasp: Path | None = None
        if write_vasp_inputs:
            step_vasp = (
                Path(results_dir)
                / "vasp_inputs"
                / f"{saturation_molecule}_saturation"
                / step_placements_rel
            )
        rows: list[dict[str, Any]] = []
        for r in self.all_results:
            pid = r.placement_id
            poscar_path = (
                str(step_vasp / f"conformer_{pid:03d}" / "POSCAR")
                if step_vasp is not None
                else None
            )
            rows.append(
                r.to_row(
                    xyz_path=str(step_xyz / f"conformer_{pid:03d}.xyz"),
                    poscar_path=poscar_path,
                    context_row=context_row,
                )
                | {
                    "molecule": saturation_molecule,
                    "step": self.step,
                }
            )
        return rows


@dataclass
class SaturationRunResult:
    """Full saturation run for one molecule: steps until slab is saturated."""

    molecule: str
    steps: list[SaturationStepResult]
    n_molecules_at_saturation: int
    final_slab_atoms: Atoms | None = None

    @staticmethod
    def format_failure_summary(failure_summary: dict[str, object]) -> str:
        """Return a canonical human-readable failure summary."""
        return _format_failure_summary_text(failure_summary)

    def to_flattened_runs(self) -> list[ScreeningRunResult]:
        """Flatten all saturation steps into screening-like run results."""
        flattened_runs: list[ScreeningRunResult] = []
        for step_result in self.steps:
            step_name = f"{self.molecule}_step_{step_result.step:03d}"
            step_results = step_result.all_results
            if not step_results:
                continue
            flattened_runs.append(
                ScreeningRunResult(
                    molecule=step_name,
                    results=step_results,
                    summary=build_molecule_summary(step_name, step_results),
                )
            )
        return flattened_runs

    def format_completion(self, *, label: str, results_dir: str) -> str:
        """Return a canonical multi-line saturation completion summary."""
        return "\n".join(
            [
                f"{label} complete:",
                f"  Molecules at saturation: {self.n_molecules_at_saturation}",
                f"  Total steps: {len(self.steps)}",
                f"  Results saved to {Path(results_dir).as_posix()}/ (XYZ, POSCAR, CSV)",
            ]
        )


@dataclass
class BOStepMemory:
    """Transferable BO memory captured from one saturation step."""

    observed_X_rows: list[dict[str, float]] = field(default_factory=list)
    observed_y: list[float] = field(default_factory=list)
    best_energy: float | None = None
    best_X_row: dict[str, float] | None = None
    step_ages: list[int] | None = None


def windowed_bo_step_memories(
    memories: Sequence[BOStepMemory | None],
    *,
    window: int | None,
) -> BOStepMemory | None:
    """Select the most recent *window* step memories and merge them."""
    if window is None:
        return merge_bo_step_memories(memories)
    if window <= 0:
        return None
    recent = [mem for mem in memories[-window:] if mem is not None]
    return merge_bo_step_memories(recent)


def merge_bo_step_memories(
    memories: Sequence[BOStepMemory | None],
) -> BOStepMemory | None:
    """Concatenate observations from multiple prior saturation steps."""
    rows: list[dict[str, float]] = []
    ys: list[float] = []
    ages: list[int] = []
    best: float | None = None
    eligible = [mem for mem in memories if mem is not None and mem.observed_X_rows]
    n_eligible = len(eligible)
    for index, mem in enumerate(eligible):
        step_age = n_eligible - 1 - index
        rows.extend(mem.observed_X_rows)
        ys.extend(float(v) for v in mem.observed_y)
        ages.extend([step_age] * len(mem.observed_X_rows))
        if mem.best_energy is not None:
            best = (
                float(mem.best_energy)
                if best is None
                else min(best, float(mem.best_energy))
            )
    if not rows:
        return None
    return BOStepMemory(
        observed_X_rows=rows,
        observed_y=ys,
        best_energy=best,
        step_ages=ages,
    )


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
class SaturationCampaignResult:
    """High-level result contract for saturation campaign runs."""

    mode: Literal["non_bo", "bo"]
    surface_type: str
    runs: list[SaturationRunResult | MultiMolSaturationRunResult]
    failure_summary: dict[str, object] = field(default_factory=dict)
    t_ref_s: float = 0.0
    t_total_s: float = 0.0

    def format_failure_summary(self) -> str:
        """Return a canonical human-readable failure summary."""
        return _format_failure_summary_text(self.failure_summary)

    def format_completion(self, *, label: str, results_dir: str) -> str:
        """Return a canonical multi-line saturation completion summary."""
        if not self.runs:
            lines = [f"{label}: no saturation results produced."]
            if self.failure_summary:
                lines.append("")
                lines.append(self.format_failure_summary())
            return "\n".join(lines)

        if len(self.runs) == 1 and isinstance(self.runs[0], SaturationRunResult):
            return self.runs[0].format_completion(label=label, results_dir=results_dir)

        total_steps = sum(len(run.steps) for run in self.runs)
        total_mols = sum(run.n_molecules_at_saturation for run in self.runs)
        return "\n".join(
            [
                f"{label} complete:",
                f"  Molecules at saturation: {total_mols}",
                f"  Total steps: {total_steps}",
                f"  Results saved to {Path(results_dir).as_posix()}/ (XYZ, POSCAR, CSV)",
            ]
        )


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
    """High-level result contract for campaign runs."""

    mode: Literal["non_bo", "bo"]
    surface_type: str
    run_results: list[ScreeningRunResult]
    molecule_summaries: list[MoleculeCampaignSummary]
    total_configurations: int
    n_molecules: int
    t_ref_s: float
    t_total_s: float
    failure_summaries: dict[str, dict[str, object]] = field(default_factory=dict)

    def format_results_saved_line(self, *, results_dir: str) -> str:
        """Return a canonical results output line."""
        return f"Results saved to {Path(results_dir).as_posix()}/ (XYZ, POSCAR, CSV)"

    def format_screening_complete(self) -> str:
        """Return a canonical screening completion line."""
        return f"Screening complete: {self.total_configurations} total configurations"

    def format_summary(self, *, title: str, results_dir: str) -> str:
        """Return a canonical multi-line binding summary block."""
        lines = [
            "=" * 60,
            title,
            "=" * 60,
            "(E_ads = E(slab+molecule) - E(slab) - E(molecule); negative = favorable)",
            "",
        ]
        for item in self.molecule_summaries:
            if item.best_adsorption_energy is None:
                lines.append(f"  {item.molecule:12s}: (no valid placements)")
                continue
            lines.append(
                f"  {item.molecule:12s}: {item.best_adsorption_energy:+.4f} eV  "
                f"({item.n_valid_placements} valid placements)"
            )
        lines.append("")
        lines.append(self.format_results_saved_line(results_dir=results_dir))
        return "\n".join(lines)

    @staticmethod
    def format_failure_summary(failure_summary: dict[str, object]) -> str:
        """Return a canonical human-readable failure summary."""
        return _format_failure_summary_text(failure_summary)

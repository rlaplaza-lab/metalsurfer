"""Typed domain models for adsorption screening results."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from ase import Atoms

from ._csv_coerce import (
    float_or as _row_float_or,
)
from ._csv_coerce import (
    int_or_none as _row_int_or_none,
)
from ._csv_coerce import (
    is_missing as _row_is_missing,
)
from ._csv_coerce import (
    parse_bool as _row_parse_bool,
)
from ._csv_coerce import (
    parse_fragment_positions as _row_parse_fragment_positions,
)
from ._csv_coerce import (
    with_default as _row_with_default,
)
from .reporting import (
    format_failure_summary_text as _format_failure_summary_text,
)
from .reporting import (
    format_results_saved_line as _format_results_saved_line,
)
from .reporting import (
    format_saturation_completion as _format_saturation_completion,
)


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


# In-memory attribute name -> rich CSV column (pre-relax provenance only).
INITIAL_PROVENANCE_COLUMN_MAP: dict[str, str] = {
    "orientation_type": "initial_orientation_type",
    "face_flip": "initial_face_flip",
    "en_atom_index": "initial_en_atom_index",
    "site_index": "initial_site_index",
    "site_type": "initial_site_type",
    "tilt_deg": "initial_tilt_deg",
    "azimuth_deg": "initial_azimuth_deg",
    "azimuth_in_plane_deg": "initial_azimuth_in_plane_deg",
    "z_fraction": "initial_z_fraction",
    "z_offset": "initial_z_offset",
    "surface_ref_z_abs": "initial_surface_ref_z_abs",
    "x": "initial_x",
    "y": "initial_y",
    "shape": "initial_shape",
    "slab_indices": "initial_slab_indices",
    "placement_mode_resolved": "initial_placement_mode_resolved",
    "site_source": "initial_site_source",
    "site_reference_frame": "initial_site_reference_frame",
    "site_xy_frac_a": "initial_site_xy_frac_a",
    "site_xy_frac_b": "initial_site_xy_frac_b",
    "fragment_positions": "initial_fragment_positions",
}


def provenance_export_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    """Map in-memory provenance attrs to ``initial_*`` CSV columns."""
    row: dict[str, Any] = {}
    for attr, export_name in INITIAL_PROVENANCE_COLUMN_MAP.items():
        val = values[attr]
        if attr == "slab_indices":
            row[export_name] = (
                ",".join(str(i) for i in val) if val is not None else None
            )
        elif attr == "fragment_positions":
            row[export_name] = json.dumps(list(val)) if val is not None else None
        else:
            row[export_name] = val
    return row


def _provenance_value_from_row(row: Mapping[str, Any], attr: str, default: Any) -> Any:
    """Resolve a provenance field from its ``initial_*`` CSV column."""
    export_name = INITIAL_PROVENANCE_COLUMN_MAP.get(attr, attr)
    if export_name in row and not _row_is_missing(row.get(export_name)):
        return row.get(export_name)
    return default


@dataclass
class PlacementDescriptor:
    """Initial (pre-relax) placement: spec fields + resolved absolute pose.

    Absolute pose (``x_abs`` / ``y_abs`` / ``z_abs`` + quaternion) is the ML/BO
    feature geometry. Site/orientation fields are enumeration provenance for the
    *initial* placement; adsorbates may move during relaxation. Post-relax
    geometry lives in on-disk structures, not these fields.
    """

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
    # Absolute fragment atom positions for multi-site (dissociative) replay only.
    # Exported only when include_provenance=True (as initial_fragment_positions).
    fragment_positions: tuple[tuple[float, float, float], ...] | None = None

    def to_row(self, *, include_provenance: bool = False) -> dict[str, Any]:
        """Convert descriptor fields to a flat row for tabular exports.

        Lean default: ML feature geometry only. Rich mode adds ``initial_*``
        pre-relax provenance columns (see ``INITIAL_PROVENANCE_COLUMN_MAP``).
        """
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
            "x_abs": x_abs,
            "y_abs": y_abs,
            "z_abs": z_abs,
            "quat_w": _row_float_or(self.quat_w, 1.0),
            "quat_x": _row_float_or(self.quat_x, 0.0),
            "quat_y": _row_float_or(self.quat_y, 0.0),
            "quat_z": _row_float_or(self.quat_z, 0.0),
        }
        if include_provenance:
            prov_vals = {
                attr: getattr(self, attr) for attr in INITIAL_PROVENANCE_COLUMN_MAP
            }
            prov_vals["surface_ref_z_abs"] = surface_ref_z_abs
            row.update(provenance_export_fields(prov_vals))
        return row

    @classmethod
    def from_row(
        cls,
        row: Mapping[str, Any],
        *,
        placement_index: int | None = None,
    ) -> "PlacementDescriptor":
        """Inflate a descriptor from lean or rich (``initial_*``) flat CSV/dict rows."""
        slab_indices_raw = _provenance_value_from_row(row, "slab_indices", None)
        slab_indices = None
        if slab_indices_raw and not _row_is_missing(slab_indices_raw):
            slab_indices = tuple(int(x) for x in str(slab_indices_raw).split(","))

        z_offset = float(
            _provenance_value_from_row(
                row, "z_offset", _row_float_or(row.get("z"), 0.0)
            )
        )
        surface_ref_z_abs = float(
            _provenance_value_from_row(row, "surface_ref_z_abs", 0.0)
        )
        pid = (
            int(placement_index)
            if placement_index is not None
            else int(_row_with_default(row.get("placement_id"), -1))
        )

        def _prov(attr: str, default: Any) -> Any:
            return _provenance_value_from_row(row, attr, default)

        conformer_index_raw = row.get("conformer_index")
        if _row_is_missing(conformer_index_raw) or conformer_index_raw is None:
            raise ValueError(
                "PlacementDescriptor.from_row requires a 'conformer_index' column"
            )
        return cls(
            conformer_index=int(conformer_index_raw),
            orientation_type=cast(
                Literal["parallel", "EN-down", "vertical", "round", "dissociative"],
                _prov("orientation_type", "round"),
            ),
            face_flip=_row_parse_bool(_prov("face_flip", False), default=False),
            en_atom_index=_row_int_or_none(_prov("en_atom_index", None)),
            site_index=int(_prov("site_index", -1)),
            site_type=_prov("site_type", None),
            tilt_deg=float(_prov("tilt_deg", 0.0)),
            azimuth_deg=float(_prov("azimuth_deg", 0.0)),
            azimuth_in_plane_deg=float(_prov("azimuth_in_plane_deg", 0.0)),
            z_fraction=float(_prov("z_fraction", 0.5)),
            placement_index=pid,
            x=float(_prov("x", 0.0)),
            y=float(_prov("y", 0.0)),
            z_offset=z_offset,
            x_abs=_row_float_or(row.get("x_abs"), 0.0),
            y_abs=_row_float_or(row.get("y_abs"), 0.0),
            surface_ref_z_abs=surface_ref_z_abs,
            z_abs=float(
                _row_with_default(row.get("z_abs"), surface_ref_z_abs + z_offset)
            ),
            shape=str(_prov("shape", "round")),
            slab_indices=slab_indices,
            placement_mode_resolved=str(_prov("placement_mode_resolved", "no_sites")),
            site_source=str(_prov("site_source", "no_sites")),
            site_reference_frame=str(_prov("site_reference_frame", "global_top_layer")),
            site_xy_frac_a=float(_prov("site_xy_frac_a", 0.0)),
            site_xy_frac_b=float(_prov("site_xy_frac_b", 0.0)),
            quat_w=_row_float_or(row.get("quat_w"), 1.0),
            quat_x=_row_float_or(row.get("quat_x"), 0.0),
            quat_y=_row_float_or(row.get("quat_y"), 0.0),
            quat_z=_row_float_or(row.get("quat_z"), 0.0),
            fragment_positions=_row_parse_fragment_positions(
                _prov("fragment_positions", None)
            ),
        )


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
    distance: float  # Post-relax min adsorbate–surface distance (Å)
    placement_descriptor: PlacementDescriptor

    def to_row(
        self,
        *,
        xyz_path: str | None = None,
        poscar_path: str | None = None,
        context_row: Mapping[str, Any] | None = None,
        include_provenance: bool = False,
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
        row.update(
            self.placement_descriptor.to_row(include_provenance=include_provenance)
        )
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
        include_provenance: bool = False,
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
                include_provenance=include_provenance,
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
        include_provenance: bool = False,
    ) -> pd.DataFrame:
        """Return a detailed pandas DataFrame for this screening run."""
        return pd.DataFrame(
            self.to_rows(
                results_dir=results_dir,
                context_row=context_row,
                write_vasp_inputs=write_vasp_inputs,
                include_provenance=include_provenance,
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
class BOTransferInfo:
    """Typed BO transfer bookkeeping written by Bayesian saturation steps."""

    transfer_enabled: bool = False
    transfer_used: bool = False
    transfer_disabled_reason: str | None = None
    transfer_bad_rounds: int = 0
    transfer_last_mae_delta: float | None = None
    transfer_weight_share: float = 0.0

    def to_saturation_columns(self) -> dict[str, Any]:
        """Flatten to stable ``bo_transfer_*`` CSV column names."""
        return {
            "bo_transfer_used": self.transfer_used,
            "bo_transfer_disabled_reason": self.transfer_disabled_reason,
            "bo_transfer_weight_share": self.transfer_weight_share,
            "bo_transfer_bad_rounds": self.transfer_bad_rounds,
            "bo_transfer_last_mae_delta": self.transfer_last_mae_delta,
        }


def _saturation_step_structure_paths(
    mol_dir: Path,
    step: int,
    energy_adsorption: float,
) -> dict[str, str]:
    return {
        "step_structure_path": str(mol_dir / f"step_{step:03d}_best_slab.xyz"),
        "step_structure_energy_path": str(
            mol_dir / f"step_{step:03d}_Eads_{energy_adsorption:.4f}.xyz"
        ),
        "step_adsorbate_path": str(mol_dir / f"step_{step:03d}_adsorbate.xyz"),
    }


def _placement_rows_for_results(
    results: Sequence[ScreeningResult],
    *,
    step: int,
    step_xyz: Path,
    step_vasp: Path | None,
    context_row: Mapping[str, Any] | None,
    include_provenance: bool,
    extra: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        r.to_row(
            xyz_path=str(step_xyz / f"conformer_{r.placement_id:03d}.xyz"),
            poscar_path=str(step_vasp / f"conformer_{r.placement_id:03d}" / "POSCAR")
            if step_vasp is not None
            else None,
            context_row=context_row,
            include_provenance=include_provenance,
        )
        | {"step": step, **dict(extra)}
        for r in results
    ]


@dataclass
class SaturationStepResult:
    """Result of one step in a sequential saturation run."""

    step: int
    molecule: str
    n_molecules_on_slab: int
    best_result: ScreeningResult
    all_results: list[ScreeningResult]
    bo_transfer_enabled: bool = False
    transfer: BOTransferInfo | None = None

    def to_detail_row(
        self,
        *,
        results_dir: str | Path,
        saturation_molecule: str,
        context_row: Mapping[str, Any] | None = None,
        include_provenance: bool = False,
    ) -> dict[str, Any]:
        """Return one saturation detail row for the winning placement."""
        best = self.best_result
        mol_dir = (
            Path(results_dir) / "xyz_structures" / f"{saturation_molecule}_saturation"
        )
        info = self.transfer if self.transfer is not None else BOTransferInfo()
        row: dict[str, Any] = {
            "molecule": saturation_molecule,
            "step": self.step,
            "n_molecules_on_slab": self.n_molecules_on_slab,
            "bo_transfer_enabled": self.bo_transfer_enabled,
            **info.to_saturation_columns(),
            "placement_id": best.placement_id,
            "energy_adslab": best.energy_adslab,
            "energy_slab": best.energy_slab,
            "energy_adsorbate": best.energy_adsorbate,
            "energy_adsorption": best.energy_adsorption,
            "distance": best.distance,
            **_saturation_step_structure_paths(
                mol_dir, self.step, best.energy_adsorption
            ),
        }
        row.update(
            best.placement_descriptor.to_row(include_provenance=include_provenance)
        )
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
        include_provenance: bool = False,
    ) -> list[dict[str, Any]]:
        """Return detailed rows for every placement evaluated in this step."""
        step_placements_rel = (
            f"step_{self.step:03d}_placements" if step_prefix else "placements"
        )
        base = Path(results_dir)
        sat = f"{saturation_molecule}_saturation"
        step_xyz = base / "xyz_structures" / sat / step_placements_rel
        step_vasp = (
            base / "vasp_inputs" / sat / step_placements_rel
            if write_vasp_inputs
            else None
        )
        return _placement_rows_for_results(
            self.all_results,
            step=self.step,
            step_xyz=step_xyz,
            step_vasp=step_vasp,
            context_row=context_row,
            include_provenance=include_provenance,
            extra={"molecule": saturation_molecule},
        )


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

    def format_completion(
        self,
        *,
        label: str,
        results_dir: str,
        write_vasp_inputs: bool = False,
    ) -> str:
        """Return a canonical multi-line saturation completion summary."""
        return _format_saturation_completion(
            label=label,
            n_molecules_at_saturation=self.n_molecules_at_saturation,
            n_steps=len(self.steps),
            results_dir=results_dir,
            write_vasp_inputs=write_vasp_inputs,
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
    transfer_by_molecule: dict[str, BOTransferInfo] = field(default_factory=dict)

    def to_detail_row(
        self,
        *,
        results_dir: str | Path,
        molecules_label: str,
        context_row: Mapping[str, Any] | None = None,
        include_provenance: bool = False,
    ) -> dict[str, Any]:
        """Return one saturation detail row for the winning placement."""
        best = self.best_result
        mol_dir = Path(results_dir) / "xyz_structures" / f"{molecules_label}_saturation"
        return best.to_row(
            context_row=context_row,
            include_provenance=include_provenance,
        ) | {
            "molecules": molecules_label,
            "winning_molecule": self.winning_molecule,
            "step": self.step,
            "n_molecules_on_slab": self.n_molecules_on_slab,
            "per_molecule_budgets": str(self.per_molecule_budgets),
            "bo_transfer_enabled": self.bo_transfer_enabled,
            **_saturation_step_structure_paths(
                mol_dir, self.step, best.energy_adsorption
            ),
        }

    def to_rows(
        self,
        *,
        results_dir: str | Path,
        molecules_label: str,
        context_row: Mapping[str, Any] | None = None,
        write_vasp_inputs: bool = False,
        include_provenance: bool = False,
    ) -> list[dict[str, Any]]:
        """Return detailed rows for every placement evaluated in this step."""
        rel = f"step_{self.step:03d}_placements"
        base = Path(results_dir)
        sat = f"{molecules_label}_saturation"
        base_xyz = base / "xyz_structures" / sat / rel
        base_vasp = base / "vasp_inputs" / sat / rel if write_vasp_inputs else None
        rows: list[dict[str, Any]] = []
        for pmol, res_list in self.per_molecule_results.items():
            rows.extend(
                _placement_rows_for_results(
                    res_list,
                    step=self.step,
                    step_xyz=base_xyz / pmol,
                    step_vasp=base_vasp / pmol if base_vasp is not None else None,
                    context_row=context_row,
                    include_provenance=include_provenance,
                    extra={
                        "molecules": molecules_label,
                        "winning_molecule": self.winning_molecule,
                        "molecule": pmol,
                    },
                )
            )
        return rows


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

    def format_completion(
        self,
        *,
        label: str,
        results_dir: str,
        write_vasp_inputs: bool = False,
    ) -> str:
        """Return a canonical multi-line saturation completion summary."""
        if not self.runs:
            lines = [f"{label}: no saturation results produced."]
            if self.failure_summary:
                lines.append("")
                lines.append(self.format_failure_summary())
            return "\n".join(lines)

        if len(self.runs) == 1 and isinstance(self.runs[0], SaturationRunResult):
            return self.runs[0].format_completion(
                label=label,
                results_dir=results_dir,
                write_vasp_inputs=write_vasp_inputs,
            )

        total_steps = sum(len(run.steps) for run in self.runs)
        total_mols = sum(run.n_molecules_at_saturation for run in self.runs)
        return _format_saturation_completion(
            label=label,
            n_molecules_at_saturation=total_mols,
            n_steps=total_steps,
            results_dir=results_dir,
            write_vasp_inputs=write_vasp_inputs,
        )


@dataclass
class MoleculeCampaignSummary:
    """Compact campaign-facing summary for a molecule."""

    molecule: str
    n_valid_placements: int
    best_adsorption_energy: float | None


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

    def format_results_saved_line(
        self,
        *,
        results_dir: str,
        write_vasp_inputs: bool = False,
    ) -> str:
        """Return a canonical results output line."""
        return _format_results_saved_line(
            results_dir=results_dir,
            write_vasp_inputs=write_vasp_inputs,
        )

    def format_screening_complete(self) -> str:
        """Return a canonical screening completion line."""
        return f"Screening complete: {self.total_configurations} total configurations"

    def format_summary(
        self,
        *,
        results_dir: str,
        title: str = "Binding energy summary",
        write_vasp_inputs: bool = False,
    ) -> str:
        """Return a canonical multi-line binding summary block."""
        lines = [
            "=" * 60,
            title,
            "=" * 60,
            "(E_ads = E(slab+molecule) - E(slab) - E(molecule); negative = favorable)",
            "",
        ]
        if self.n_molecules == 0 and self.total_configurations == 0:
            lines.append(
                "No molecules processed (all skipped, empty input, or no valid rows)."
            )
            lines.append("")
        summaries = list(self.molecule_summaries)
        if not summaries and self.run_results:
            for run_result in self.run_results:
                best = (
                    min(r.energy_adsorption for r in run_result.results)
                    if run_result.results
                    else None
                )
                summaries.append(
                    MoleculeCampaignSummary(
                        molecule=run_result.molecule,
                        n_valid_placements=len(run_result.results),
                        best_adsorption_energy=best,
                    )
                )
        for item in summaries:
            if item.best_adsorption_energy is None:
                lines.append(f"  {item.molecule:12s}: (no valid placements)")
                continue
            lines.append(
                f"  {item.molecule:12s}: {item.best_adsorption_energy:+.4f} eV  "
                f"({item.n_valid_placements} valid placements)"
            )
        if self.failure_summaries:
            lines.append("")
            for molecule_name, failure_summary in self.failure_summaries.items():
                lines.append(f"Failures for {molecule_name}:")
                lines.append(self.format_failure_summary(failure_summary))
        lines.append("")
        lines.append(
            self.format_results_saved_line(
                results_dir=results_dir,
                write_vasp_inputs=write_vasp_inputs,
            )
        )
        return "\n".join(lines)

    @staticmethod
    def format_failure_summary(failure_summary: dict[str, object]) -> str:
        """Return a canonical human-readable failure summary."""
        return _format_failure_summary_text(failure_summary)

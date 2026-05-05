"""File I/O: XYZ, VASP inputs, CSV summaries, run metadata."""

import json
import logging
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from ase import Atoms

from .config import AdsorptionConfig
from .ml.schema import SCHEMA_VERSION, config_to_context_row
from .models import (
    MultiMolSaturationRunResult,
    PlacementDescriptor,
    SaturationRunResult,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)

logger = logging.getLogger(__name__)


def _results_dir(surface_type: str) -> Path:
    return Path(f"results_{surface_type}")


def _build_run_metadata(
    *,
    surface_type: str,
    config: AdsorptionConfig,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical run metadata payload."""
    config_dict = {k: v for k, v in asdict(config).items() if not callable(v)}
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "surface_type": surface_type,
        "config": config_dict,
    }
    if extra_fields:
        metadata.update(extra_fields)
    return metadata


def _write_run_metadata_file(results_dir: Path, metadata: dict[str, Any]) -> Path:
    """Write run metadata JSON under *results_dir*."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "run_metadata.json"
    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)
    return path


def _placement_descriptor_to_row(d: PlacementDescriptor) -> dict[str, Any]:
    """Convert PlacementDescriptor fields to a dict for CSV row."""
    x_abs = d.x_abs if d.x_abs is not None else d.x
    y_abs = d.y_abs if d.y_abs is not None else d.y
    z_offset = d.z_offset
    surface_ref_z_abs = d.surface_ref_z_abs if d.surface_ref_z_abs is not None else 0.0
    z_abs = d.z_abs if d.z_abs is not None else surface_ref_z_abs + z_offset
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "conformer_index": d.conformer_index,
        "orientation_type": d.orientation_type,
        "face_flip": d.face_flip,
        "en_atom_index": d.en_atom_index,
        "site_index": d.site_index,
        "site_type": d.site_type,
        "tilt_deg": d.tilt_deg,
        "azimuth_deg": d.azimuth_deg,
        "azimuth_in_plane_deg": d.azimuth_in_plane_deg,
        "x_abs": x_abs,
        "y_abs": y_abs,
        "z_offset": z_offset,
        "surface_ref_z_abs": surface_ref_z_abs,
        "z_abs": z_abs,
        "shape": d.shape,
        "placement_mode_resolved": d.placement_mode_resolved,
        "site_source": d.site_source,
        "site_reference_frame": d.site_reference_frame,
        "site_xy_frac_a": d.site_xy_frac_a,
        "site_xy_frac_b": d.site_xy_frac_b,
    }
    if d.slab_indices is not None:
        row["slab_indices"] = ",".join(str(i) for i in d.slab_indices)
    return row


def write_run_settings(
    surface_type: str,
    config: AdsorptionConfig,
    **run_info: Any,
) -> None:
    """Persist run config and optional run info to run_metadata.json for reproducibility.

    Call this for single-molecule and saturation runs so every results directory
    has the AdsorptionConfig (and optional molecule/count/timing) needed to
    reproduce the computation. Batch runs typically use write_run_metadata instead.
    """
    results_dir = _results_dir(surface_type)
    metadata = _build_run_metadata(
        surface_type=surface_type,
        config=config,
        extra_fields=run_info,
    )
    path = _write_run_metadata_file(results_dir, metadata)
    logger.info("Run settings written to %s", path)


def setup_directories(surface_types: list[str] | None = None) -> None:
    """Create the results directory tree for each surface type."""
    if surface_types is None:
        surface_types = ["manual"]
    for st in surface_types:
        for sub in ("", "/vasp_inputs", "/xyz_structures"):
            os.makedirs(f"results_{st}{sub}", exist_ok=True)


def save_molecule_results(
    molecule_name: str,
    results: list[ScreeningResult],
    surface_type: str = "manual",
    system_name: str | None = None,
    config: AdsorptionConfig | None = None,
) -> None:
    """Write XYZ and VASP input files for each result."""
    if config is None:
        config = AdsorptionConfig()

    results_dir = _results_dir(surface_type)
    vasp_dir = results_dir / "vasp_inputs" / f"{molecule_name}_all"
    xyz_dir = results_dir / "xyz_structures" / f"{molecule_name}_all"
    mol_xyz_dir = results_dir / "xyz_structures" / f"{molecule_name}_adsorbate_only"
    os.makedirs(vasp_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(mol_xyz_dir, exist_ok=True)

    for entry in results:
        pid = entry.placement_id

        xyz_file = xyz_dir / f"conformer_{pid:03d}.xyz"
        _write_clean_xyz(entry.atoms, str(xyz_file))
        adsorbate_xyz_file = mol_xyz_dir / f"conformer_{pid:03d}_adsorbate.xyz"
        adsorbate_atoms = entry.atoms[entry.slab_size :].copy()
        _write_clean_xyz(adsorbate_atoms, str(adsorbate_xyz_file))

        vasp_subdir = vasp_dir / f"conformer_{pid:03d}"
        _write_vasp_inputs(
            entry.atoms,
            str(vasp_subdir),
            molecule_name,
            system_name=system_name,
            config=config,
        )

        logger.info(
            "  Saved placement %d: E_ads = %.4f eV -> %s (adsorbate: %s)",
            pid,
            entry.energy_adsorption,
            xyz_file,
            adsorbate_xyz_file,
        )


def screening_run_result(
    molecule_name: str,
    results: list[ScreeningResult],
) -> ScreeningRunResult:
    """Build a :class:`ScreeningRunResult` for :func:`save_summary_results` after a campaign."""
    summary = build_molecule_summary(molecule_name, results)
    return ScreeningRunResult(
        molecule=molecule_name,
        results=results,
        summary=summary,
    )


def save_single_molecule_results(
    molecule_name: str,
    results: list[ScreeningResult],
    surface_type: str = "manual",
    system_name: str | None = None,
    config: AdsorptionConfig | None = None,
    *,
    write_csv: bool = True,
) -> None:
    """Write XYZ, POSCAR, and CSV for a single molecule's screening results.

    Convenience helper for single-molecule runs (e.g. process_molecule).
    Saves structures and builds a detailed + summary CSV.

    For multi-molecule campaigns, pass ``write_csv=False`` in the loop (structures only),
    accumulate :class:`ScreeningRunResult` via :func:`screening_run_result`, then call
    :func:`save_summary_results` and :func:`write_run_settings` once at the end.
    """
    if not results:
        logger.warning("No results to save for %s", molecule_name)
        return
    save_molecule_results(
        molecule_name,
        results,
        surface_type=surface_type,
        system_name=system_name,
        config=config,
    )
    if not write_csv:
        return
    run_result = screening_run_result(molecule_name, results)
    save_summary_results([run_result], surface_type=surface_type, config=config)
    if config is not None:
        write_run_settings(
            surface_type,
            config,
            molecule=molecule_name,
            n_configurations=len(results),
        )


def save_summary_results(
    run_results: list[ScreeningRunResult],
    surface_type: str = "manual",
    config: AdsorptionConfig | None = None,
) -> None:
    """Write detailed and summary CSV files from typed run results.

    When config is provided, each detailed row is extended with computation
    context (model_name, fmax, stage1_steps, stage2_steps,
    seed, context_hash, etc.) so the run is exactly reproducible.
    """
    results_dir = _results_dir(surface_type)
    context_row = config_to_context_row(config) if config else {}
    all_rows: list[dict[str, Any]] = []
    for rr in run_results:
        xyz_dir = results_dir / "xyz_structures" / f"{rr.molecule}_all"
        vasp_dir = results_dir / "vasp_inputs" / f"{rr.molecule}_all"
        for sr in rr.results:
            pid = sr.placement_id
            row: dict[str, Any] = {
                "molecule": sr.molecule,
                "placement_id": pid,
                "energy_adslab": sr.energy_adslab,
                "energy_slab": sr.energy_slab,
                "energy_adsorbate": sr.energy_adsorbate,
                "energy_adsorption": sr.energy_adsorption,
                "distance": sr.distance,
                "xyz_path": str(xyz_dir / f"conformer_{pid:03d}.xyz"),
                "poscar_path": str(vasp_dir / f"conformer_{pid:03d}" / "POSCAR"),
            }
            row.update(_placement_descriptor_to_row(sr.placement_descriptor))
            if context_row:
                row.update(context_row)
            all_rows.append(row)
    if not all_rows:
        logger.warning("No results to save")
        return

    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "adsorption_energies_detailed.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for rr in run_results:
        if rr.summary is not None:
            s = rr.summary
            summary_rows.append(
                {
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
            )

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(results_dir / "adsorption_energy_summary.csv", index=False)

    logger.info("Saved summary results to %s", results_dir)


def save_saturation_results(
    saturation_results: list[SaturationRunResult],
    surface_type: str = "manual",
    config: AdsorptionConfig | None = None,
) -> None:
    """Write saturation CSV summaries and per-step structures."""
    if config is None:
        config = AdsorptionConfig()

    if not saturation_results:
        logger.warning("No saturation results to save")
        return

    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = f"{results_dir}/xyz_structures"
    os.makedirs(xyz_dir, exist_ok=True)

    # Detailed CSV: one row per step per molecule (with context for reproducibility)
    context_row = config_to_context_row(config)
    detail_rows: list[dict[str, Any]] = []
    for sr in saturation_results:
        mol_dir = f"{xyz_dir}/{sr.molecule}_saturation"
        for step_result in sr.steps:
            best = step_result.best_result
            step = step_result.step
            step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
            step_energy_path = (
                f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
            )
            step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
            detail_row: dict[str, Any] = {
                "molecule": sr.molecule,
                "step": step,
                "n_molecules_on_slab": step_result.n_molecules_on_slab,
                "bo_transfer_enabled": step_result.bo_transfer_enabled,
                "bo_transfer_used": step_result.bo_transfer_used,
                "bo_transfer_disabled_reason": step_result.bo_transfer_disabled_reason,
                "bo_transfer_weight_share": step_result.bo_transfer_weight_share,
                "bo_transfer_bad_rounds": step_result.bo_transfer_bad_rounds,
                "bo_transfer_last_mae_delta": step_result.bo_transfer_last_mae_delta,
                "placement_id": best.placement_id,
                "energy_adslab": best.energy_adslab,
                "energy_slab": best.energy_slab,
                "energy_adsorbate": best.energy_adsorbate,
                "energy_adsorption": best.energy_adsorption,
                "distance": best.distance,
                "step_structure_path": step_structure_path,
                "step_structure_energy_path": step_energy_path,
                "step_adsorbate_path": step_adsorbate_path,
            }
            detail_row.update(_placement_descriptor_to_row(best.placement_descriptor))
            detail_row.update(context_row)
            detail_rows.append(detail_row)

    if detail_rows:
        df = pd.DataFrame(detail_rows)
        df.to_csv(f"{results_dir}/saturation_details.csv", index=False)

    # Summary CSV: one row per molecule
    summary_rows: list[dict[str, Any]] = []
    for sr in saturation_results:
        final_slab_path = (
            f"{xyz_dir}/{sr.molecule}_saturation/final_saturated_slab.xyz"
            if sr.final_slab_atoms is not None
            else ""
        )
        summary_rows.append(
            {
                "molecule": sr.molecule,
                "n_molecules_at_saturation": sr.n_molecules_at_saturation,
                "n_steps": len(sr.steps),
                "final_slab_path": final_slab_path,
            }
        )

    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(f"{results_dir}/saturation_summary.csv", index=False)

    vasp_base = f"{results_dir}/vasp_inputs"
    for sr in saturation_results:
        mol_dir = f"{xyz_dir}/{sr.molecule}_saturation"
        vasp_mol_dir = f"{vasp_base}/{sr.molecule}_saturation"
        os.makedirs(mol_dir, exist_ok=True)
        os.makedirs(vasp_mol_dir, exist_ok=True)
        for step_result in sr.steps:
            step = step_result.step
            best = step_result.best_result
            step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
            step_energy_path = (
                f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
            )
            step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
            # Create copies without calculator to avoid shape mismatches in results arrays
            best_atoms_copy = best.atoms.copy()
            best_atoms_copy.calc = None
            best_atoms_copy.write(step_structure_path, format="extxyz")
            best_atoms_copy.write(step_energy_path, format="extxyz")
            adsorbate = best.atoms[best.slab_size :].copy()
            _write_clean_xyz(adsorbate, step_adsorbate_path)
            vasp_subdir = f"{vasp_mol_dir}/step_{step:03d}"
            _write_vasp_inputs(
                best.atoms,
                vasp_subdir,
                sr.molecule,
                system_name=None,
                config=config,
            )
        if sr.final_slab_atoms is not None:
            final_slab_copy = sr.final_slab_atoms.copy()
            final_slab_copy.calc = None
            final_slab_copy.write(
                f"{mol_dir}/final_saturated_slab.xyz", format="extxyz"
            )

    write_run_settings(
        surface_type,
        config,
        n_molecules=len(saturation_results),
        total_steps=sum(len(sr.steps) for sr in saturation_results),
        n_molecules_at_saturation=sum(
            sr.n_molecules_at_saturation for sr in saturation_results
        ),
    )
    logger.info("Saved saturation results to %s", results_dir)


def save_multi_mol_saturation_results(
    result: MultiMolSaturationRunResult,
    surface_type: str = "manual",
    config: AdsorptionConfig | None = None,
) -> None:
    """Write CSV summaries and per-step structures for a multi-molecule saturation run.

    Output layout mirrors :func:`save_saturation_results` but uses a joined
    molecule name (``mol1_mol2``) for directory names and adds
    ``winning_molecule`` / ``per_molecule_budgets`` columns to the detail CSV.
    """
    if config is None:
        config = AdsorptionConfig()

    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = f"{results_dir}/xyz_structures"
    os.makedirs(xyz_dir, exist_ok=True)

    mol_label = "_".join(result.molecules)
    mol_dir = f"{xyz_dir}/{mol_label}_saturation"
    vasp_mol_dir = f"{results_dir}/vasp_inputs/{mol_label}_saturation"
    os.makedirs(mol_dir, exist_ok=True)
    os.makedirs(vasp_mol_dir, exist_ok=True)

    context_row = config_to_context_row(config)
    detail_rows: list[dict[str, Any]] = []
    for step_result in result.steps:
        best = step_result.best_result
        step = step_result.step
        step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
        step_energy_path = (
            f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
        )
        step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
        detail_row: dict[str, Any] = {
            "molecules": mol_label,
            "winning_molecule": step_result.winning_molecule,
            "step": step,
            "n_molecules_on_slab": step_result.n_molecules_on_slab,
            "per_molecule_budgets": str(step_result.per_molecule_budgets),
            "bo_transfer_enabled": step_result.bo_transfer_enabled,
            "placement_id": best.placement_id,
            "energy_adslab": best.energy_adslab,
            "energy_slab": best.energy_slab,
            "energy_adsorbate": best.energy_adsorbate,
            "energy_adsorption": best.energy_adsorption,
            "distance": best.distance,
            "step_structure_path": step_structure_path,
            "step_structure_energy_path": step_energy_path,
            "step_adsorbate_path": step_adsorbate_path,
        }
        detail_row.update(_placement_descriptor_to_row(best.placement_descriptor))
        detail_row.update(context_row)
        detail_rows.append(detail_row)

    if detail_rows:
        df = pd.DataFrame(detail_rows)
        df.to_csv(f"{results_dir}/saturation_details.csv", index=False)

    # Summary CSV: one row for the whole multi-mol run
    summary_row: dict[str, Any] = {
        "molecules": mol_label,
        "n_molecules_at_saturation": result.n_molecules_at_saturation,
        "n_steps": len(result.steps),
        "molecule_counts": str(result.molecule_counts),
        "final_slab_path": (
            f"{mol_dir}/final_saturated_slab.xyz"
            if result.final_slab_atoms is not None
            else ""
        ),
    }
    pd.DataFrame([summary_row]).to_csv(
        f"{results_dir}/saturation_summary.csv", index=False
    )

    # Write XYZ and VASP inputs for each step
    for step_result in result.steps:
        step = step_result.step
        best = step_result.best_result
        step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
        step_energy_path = (
            f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
        )
        step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
        # Create copies without calculator to avoid shape mismatches in results arrays
        best_atoms_copy = best.atoms.copy()
        best_atoms_copy.calc = None
        best_atoms_copy.write(step_structure_path, format="extxyz")
        best_atoms_copy.write(step_energy_path, format="extxyz")
        adsorbate = best.atoms[best.slab_size :].copy()
        _write_clean_xyz(adsorbate, step_adsorbate_path)
        vasp_subdir = f"{vasp_mol_dir}/step_{step:03d}"
        _write_vasp_inputs(
            best.atoms,
            vasp_subdir,
            step_result.winning_molecule,
            system_name=None,
            config=config,
        )

    if result.final_slab_atoms is not None:
        final_slab_copy = result.final_slab_atoms.copy()
        final_slab_copy.calc = None
        final_slab_copy.write(
            f"{mol_dir}/final_saturated_slab.xyz", format="extxyz"
        )

    write_run_settings(
        surface_type,
        config,
        n_molecules=len(result.molecules),
        total_steps=len(result.steps),
        n_molecules_at_saturation=result.n_molecules_at_saturation,
    )
    logger.info("Saved multi-mol saturation results to %s", results_dir)


def write_run_metadata(
    surface_type: str,
    config: AdsorptionConfig,
    smiles_file: str,
    n_molecules: int,
    total_configs: int,
    t_ref_s: float,
    t_total_s: float,
) -> None:
    """Persist reproducible run metadata as JSON."""
    results_dir = Path(f"results_{surface_type}")

    mol_per_s = n_molecules / t_total_s if t_total_s > 0 else 0.0
    cfg_per_s = total_configs / t_total_s if t_total_s > 0 else 0.0

    metadata = _build_run_metadata(
        surface_type=surface_type,
        config=config,
        extra_fields={
            "input": {
                "smiles_file": smiles_file,
                "n_molecules": n_molecules,
            },
            "timing": {
                "reference_energies_s": round(t_ref_s, 3),
                "total_wall_clock_s": round(t_total_s, 3),
                "molecules_per_second": round(mol_per_s, 4),
                "configs_per_second": round(cfg_per_s, 4),
            },
            "results": {
                "total_molecules": n_molecules,
                "total_configurations": total_configs,
            },
        },
    )
    path = _write_run_metadata_file(results_dir, metadata)
    logger.info("Run metadata written to %s", path)


def _write_vasp_inputs(
    atoms: Atoms,
    vasp_dir: str,
    molecule_name: str,
    system_name: str | None = None,
    config: AdsorptionConfig | None = None,
) -> None:
    if config is None:
        config = AdsorptionConfig()
    os.makedirs(vasp_dir, exist_ok=True)
    atoms.write(f"{vasp_dir}/POSCAR", format="vasp", vasp5=True, direct=True)

    incar = (
        f"SYSTEM = {molecule_name} on {system_name or 'surface'}\n"
        f"ISTART = 0\nICHARG = 2\nENCUT = {config.vasp_encut}\n"
        f"EDIFF = {config.vasp_ediff}\nEDIFFG = {config.vasp_ediffg}\n"
        f"NSW = {config.vasp_nsw}\nIBRION = 2\nISIF = 2\n"
        "LREAL = Auto\nALGO = Normal\nPREC = Normal\n"
        "LWAVE = .FALSE.\nLCHARG = .FALSE.\n"
    )
    with open(f"{vasp_dir}/INCAR", "w") as f:
        f.write(incar)

    kp = config.vasp_kpoints
    kpoints = f"Automatic mesh\n0\nGamma\n{kp[0]} {kp[1]} {kp[2]}\n0 0 0\n"
    with open(f"{vasp_dir}/KPOINTS", "w") as f:
        f.write(kpoints)


def _write_clean_xyz(atoms: Atoms, filename: str) -> None:
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    # Create a copy without calculator to avoid shape mismatches in results arrays
    atoms_copy = atoms.copy()
    atoms_copy.calc = None
    atoms_copy.write(filename, format="extxyz")

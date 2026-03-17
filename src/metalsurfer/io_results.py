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
from .ml.schema import config_to_context_row
from .models import (
    PlacementDescriptor,
    SaturationRunResult,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)

logger = logging.getLogger(__name__)


def _placement_descriptor_to_row(d: PlacementDescriptor) -> dict[str, Any]:
    """Convert PlacementDescriptor fields to a dict for CSV row."""
    row: dict[str, Any] = {
        "conformer_index": d.conformer_index,
        "orientation_type": d.orientation_type,
        "face_flip": d.face_flip,
        "en_atom_index": d.en_atom_index,
        "site_index": d.site_index,
        "site_type": d.site_type,
        "tilt_deg": d.tilt_deg,
        "azimuth_deg": d.azimuth_deg,
        "azimuth_in_plane_deg": d.azimuth_in_plane_deg,
        "x": d.x,
        "y": d.y,
        "z": d.z,
        "shape": d.shape,
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
    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    config_dict = {k: v for k, v in asdict(config).items() if not callable(v)}
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "surface_type": surface_type,
        "config": config_dict,
        **run_info,
    }
    path = f"{results_dir}/run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
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

    results_dir = f"results_{surface_type}"
    vasp_dir = f"{results_dir}/vasp_inputs/{molecule_name}_all"
    xyz_dir = f"{results_dir}/xyz_structures/{molecule_name}_all"
    os.makedirs(vasp_dir, exist_ok=True)
    os.makedirs(xyz_dir, exist_ok=True)

    for entry in results:
        pid = entry.placement_id

        xyz_file = f"{xyz_dir}/conformer_{pid:03d}.xyz"
        _write_clean_xyz(entry.atoms, xyz_file)

        vasp_subdir = f"{vasp_dir}/conformer_{pid:03d}"
        _write_vasp_inputs(
            entry.atoms,
            vasp_subdir,
            molecule_name,
            system_name=system_name,
            config=config,
        )

        logger.info(
            "  Saved placement %d: E_ads = %.4f eV -> %s",
            pid,
            entry.energy_adsorption,
            xyz_file,
        )


def save_single_molecule_results(
    molecule_name: str,
    results: list[ScreeningResult],
    surface_type: str = "manual",
    system_name: str | None = None,
    config: AdsorptionConfig | None = None,
) -> None:
    """Write XYZ, POSCAR, and CSV for a single molecule's screening results.

    Convenience helper for single-molecule runs (e.g. process_molecule).
    Saves structures and builds a detailed + summary CSV.
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
    summary = build_molecule_summary(molecule_name, results)
    run_result = ScreeningRunResult(
        molecule=molecule_name,
        results=results,
        summary=summary,
    )
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
    context (model_name, placement_mode, fmax, stage1_steps, stage2_steps,
    seed, context_hash, etc.) so the run is exactly reproducible.
    """
    results_dir = f"results_{surface_type}"
    context_row = config_to_context_row(config) if config else {}
    all_rows: list[dict[str, Any]] = []
    for rr in run_results:
        xyz_dir = f"{results_dir}/xyz_structures/{rr.molecule}_all"
        vasp_dir = f"{results_dir}/vasp_inputs/{rr.molecule}_all"
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
                "xyz_path": f"{xyz_dir}/conformer_{pid:03d}.xyz",
                "poscar_path": f"{vasp_dir}/conformer_{pid:03d}/POSCAR",
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
    df.to_csv(f"{results_dir}/adsorption_energies_detailed.csv", index=False)

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
        sdf.to_csv(f"{results_dir}/adsorption_energy_summary.csv", index=False)

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
        for step_result in sr.steps:
            best = step_result.best_result
            detail_row: dict[str, Any] = {
                "molecule": sr.molecule,
                "step": step_result.step,
                "n_molecules_on_slab": step_result.n_molecules_on_slab,
                "placement_id": best.placement_id,
                "energy_adslab": best.energy_adslab,
                "energy_slab": best.energy_slab,
                "energy_adsorbate": best.energy_adsorbate,
                "energy_adsorption": best.energy_adsorption,
                "distance": best.distance,
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
        summary_rows.append(
            {
                "molecule": sr.molecule,
                "n_molecules_at_saturation": sr.n_molecules_at_saturation,
                "n_steps": len(sr.steps),
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
            xyz_path = (
                f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
            )
            _write_clean_xyz(best.atoms, xyz_path)
            vasp_subdir = f"{vasp_mol_dir}/step_{step:03d}"
            _write_vasp_inputs(
                best.atoms,
                vasp_subdir,
                sr.molecule,
                system_name=None,
                config=config,
            )
        if sr.final_slab_atoms is not None:
            _write_clean_xyz(
                sr.final_slab_atoms,
                f"{mol_dir}/final_saturated_slab.xyz",
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
    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)

    mol_per_s = n_molecules / t_total_s if t_total_s > 0 else 0.0
    cfg_per_s = total_configs / t_total_s if t_total_s > 0 else 0.0

    config_dict = {k: v for k, v in asdict(config).items() if not callable(v)}
    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "surface_type": surface_type,
        "config": config_dict,
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
    }

    path = f"{results_dir}/run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
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
    with open(filename, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("Generated by metalsurfer\n")
        for sym, pos in zip(
            atoms.get_chemical_symbols(), atoms.get_positions(), strict=False
        ):
            f.write(f"{sym:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}\n")

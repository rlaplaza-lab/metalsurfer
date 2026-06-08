"""File I/O: XYZ, VASP inputs, CSV summaries, run metadata."""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
from ase import Atoms

from .config import AdsorptionConfig
from .ml.schema import SCHEMA_VERSION, config_to_context_row
from .models import (
    MultiMolSaturationRunResult,
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


def setup_directories(
    surface_types: list[str] | None = None,
    *,
    write_vasp_inputs: bool = False,
) -> None:
    """Create the results directory tree for each surface type."""
    if surface_types is None:
        surface_types = ["manual"]
    for st in surface_types:
        os.makedirs(f"results_{st}", exist_ok=True)
        os.makedirs(f"results_{st}/xyz_structures", exist_ok=True)
        if write_vasp_inputs:
            os.makedirs(f"results_{st}/vasp_inputs", exist_ok=True)


def _write_placement_artifacts(
    entry: ScreeningResult,
    *,
    xyz_dir: Path | str,
    adsorbate_xyz_dir: Path | str,
    vasp_parent: Path | str,
    molecule_name: str,
    system_name: str | None,
    config: AdsorptionConfig,
    log_save: bool = True,
) -> None:
    """Write full-slab XYZ, adsorbate-only XYZ, and VASP inputs for one placement."""
    pid = entry.placement_id
    xyz_base = Path(xyz_dir)
    ads_base = Path(adsorbate_xyz_dir)
    vasp_base = Path(vasp_parent)
    xyz_file = xyz_base / f"conformer_{pid:03d}.xyz"
    adsorbate_xyz_file = ads_base / f"conformer_{pid:03d}_adsorbate.xyz"
    vasp_subdir = vasp_base / f"conformer_{pid:03d}"
    _write_clean_xyz(entry.atoms, str(xyz_file))
    adsorbate_atoms = entry.atoms[entry.slab_size :].copy()
    _write_clean_xyz(adsorbate_atoms, str(adsorbate_xyz_file))
    if config.write_vasp_inputs:
        _write_vasp_inputs(
            entry.atoms,
            str(vasp_subdir),
            molecule_name,
            system_name=system_name,
            config=config,
        )
    if log_save:
        logger.info(
            "  Saved placement %d: E_ads = %.4f eV -> %s (adsorbate: %s)",
            pid,
            entry.energy_adsorption,
            xyz_file,
            adsorbate_xyz_file,
        )


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
    xyz_dir = results_dir / "xyz_structures" / f"{molecule_name}_all"
    mol_xyz_dir = results_dir / "xyz_structures" / f"{molecule_name}_adsorbate_only"
    vasp_dir = results_dir / "vasp_inputs" / f"{molecule_name}_all"
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(mol_xyz_dir, exist_ok=True)
    if config.write_vasp_inputs:
        os.makedirs(vasp_dir, exist_ok=True)

    for entry in results:
        _write_placement_artifacts(
            entry,
            xyz_dir=xyz_dir,
            adsorbate_xyz_dir=mol_xyz_dir,
            vasp_parent=vasp_dir,
            molecule_name=molecule_name,
            system_name=system_name,
            config=config,
            log_save=True,
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
    write_vasp = config.write_vasp_inputs if config else False
    all_rows: list[dict[str, Any]] = []
    for rr in run_results:
        for row in rr.to_rows(
            results_dir=results_dir,
            context_row=context_row,
            write_vasp_inputs=write_vasp,
        ):
            row["schema_version"] = SCHEMA_VERSION
            all_rows.append(row)
    if not all_rows:
        logger.warning("No results to save")
        return

    os.makedirs(results_dir, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(results_dir / "adsorption_energies_detailed.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for rr in run_results:
        summary_row = rr.to_summary_row()
        if summary_row is not None:
            summary_rows.append(summary_row)

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        sdf.to_csv(results_dir / "adsorption_energy_summary.csv", index=False)

    logger.info("Saved summary results to %s", results_dir)


def save_saturation_results(
    saturation_results: Sequence[SaturationRunResult | MultiMolSaturationRunResult],
    surface_type: str = "manual",
    config: AdsorptionConfig | None = None,
) -> None:
    """Write saturation CSV summaries and per-step structures.

    When the first entry is a :class:`MultiMolSaturationRunResult`, delegates to
    :func:`save_multi_mol_saturation_results` (only the first element is saved).

    If ``config.saturation_save_all_placements`` is true (default), also writes
    ``saturation_placements_detailed.csv`` and, for each step, every structure in
    ``all_results`` under ``step_{NNN}_placements/`` (mirroring screening
    ``conformer_*`` layout). The per-step best-slab files are always written.
    """
    if config is None:
        config = AdsorptionConfig()

    if not saturation_results:
        logger.warning("No saturation results to save")
        return

    if isinstance(saturation_results[0], MultiMolSaturationRunResult):
        if len(saturation_results) > 1:
            logger.warning(
                "save_saturation_results received %d multi-molecule results; "
                "only the first will be saved",
                len(saturation_results),
            )
        save_multi_mol_saturation_results(
            saturation_results[0],
            surface_type=surface_type,
            config=config,
        )
        return

    single_results = cast(list[SaturationRunResult], list(saturation_results))

    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = f"{results_dir}/xyz_structures"
    os.makedirs(xyz_dir, exist_ok=True)
    write_vasp = config.write_vasp_inputs
    vasp_base = f"{results_dir}/vasp_inputs" if write_vasp else None

    # Detailed CSV: one row per step per molecule (with context for reproducibility)
    context_row = config_to_context_row(config)
    detail_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    save_all = config.saturation_save_all_placements

    for sr in single_results:
        mol_dir = f"{xyz_dir}/{sr.molecule}_saturation"
        vasp_mol_dir = (
            f"{vasp_base}/{sr.molecule}_saturation" if vasp_base is not None else None
        )
        for step_result in sr.steps:
            detail_row = step_result.to_detail_row(
                results_dir=results_dir,
                saturation_molecule=sr.molecule,
                context_row=context_row,
            )
            detail_row["schema_version"] = SCHEMA_VERSION
            detail_rows.append(detail_row)

            if save_all:
                step = step_result.step
                for prow in step_result.to_rows(
                    results_dir=results_dir,
                    saturation_molecule=sr.molecule,
                    context_row=context_row,
                    step_prefix=True,
                    write_vasp_inputs=write_vasp,
                ):
                    prow["schema_version"] = SCHEMA_VERSION
                    placement_rows.append(prow)

    if detail_rows:
        df = pd.DataFrame(detail_rows)
        df.to_csv(f"{results_dir}/saturation_details.csv", index=False)

    if save_all and placement_rows:
        pdf = pd.DataFrame(placement_rows)
        pdf.to_csv(f"{results_dir}/saturation_placements_detailed.csv", index=False)

    # Summary CSV: one row per molecule
    summary_rows: list[dict[str, Any]] = []
    for sr in single_results:
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

    for sr in single_results:
        mol_dir = f"{xyz_dir}/{sr.molecule}_saturation"
        vasp_mol_dir = (
            f"{vasp_base}/{sr.molecule}_saturation" if vasp_base is not None else None
        )
        os.makedirs(mol_dir, exist_ok=True)
        if vasp_mol_dir is not None:
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
            if vasp_mol_dir is not None:
                vasp_subdir = f"{vasp_mol_dir}/step_{step:03d}"
                _write_vasp_inputs(
                    best.atoms,
                    vasp_subdir,
                    sr.molecule,
                    system_name=None,
                    config=config,
                )
            if save_all:
                step_placements_rel = f"step_{step:03d}_placements"
                step_xyz = Path(mol_dir) / step_placements_rel
                os.makedirs(step_xyz, exist_ok=True)
                step_vasp: Path | None = None
                if vasp_mol_dir is not None:
                    step_vasp = Path(vasp_mol_dir) / step_placements_rel
                    os.makedirs(step_vasp, exist_ok=True)
                for r in step_result.all_results:
                    _write_placement_artifacts(
                        r,
                        xyz_dir=step_xyz,
                        adsorbate_xyz_dir=step_xyz,
                        vasp_parent=step_vasp or vasp_mol_dir or "",
                        molecule_name=sr.molecule,
                        system_name=None,
                        config=config,
                        log_save=False,
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
        n_molecules=len(single_results),
        total_steps=sum(len(sr.steps) for sr in single_results),
        n_molecules_at_saturation=sum(
            sr.n_molecules_at_saturation for sr in single_results
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

    When ``config.saturation_save_all_placements`` is true, writes
    ``saturation_placements_detailed.csv`` and per-step placement trees under
    ``step_{NNN}_placements/{molecule}/`` for each molecule's result list.
    """
    if config is None:
        config = AdsorptionConfig()

    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = f"{results_dir}/xyz_structures"
    os.makedirs(xyz_dir, exist_ok=True)

    mol_label = "_".join(result.molecules)
    mol_dir = f"{xyz_dir}/{mol_label}_saturation"
    write_vasp = config.write_vasp_inputs
    vasp_mol_dir = (
        f"{results_dir}/vasp_inputs/{mol_label}_saturation" if write_vasp else None
    )
    os.makedirs(mol_dir, exist_ok=True)
    if vasp_mol_dir is not None:
        os.makedirs(vasp_mol_dir, exist_ok=True)

    context_row = config_to_context_row(config)
    detail_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    save_all = config.saturation_save_all_placements
    for step_result in result.steps:
        best = step_result.best_result
        step = step_result.step
        step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
        step_energy_path = (
            f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
        )
        step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
        detail_row = best.to_row(
            context_row=context_row,
        ) | {
            "molecules": mol_label,
            "winning_molecule": step_result.winning_molecule,
            "step": step,
            "n_molecules_on_slab": step_result.n_molecules_on_slab,
            "per_molecule_budgets": str(step_result.per_molecule_budgets),
            "bo_transfer_enabled": step_result.bo_transfer_enabled,
            "step_structure_path": step_structure_path,
            "step_structure_energy_path": step_energy_path,
            "step_adsorbate_path": step_adsorbate_path,
        }
        detail_row["schema_version"] = SCHEMA_VERSION
        detail_rows.append(detail_row)

        if save_all:
            step_placements_rel = f"step_{step:03d}_placements"
            for pmol, res_list in step_result.per_molecule_results.items():
                step_mol_xyz = Path(mol_dir) / step_placements_rel / pmol
                step_mol_vasp = (
                    Path(vasp_mol_dir) / step_placements_rel / pmol
                    if vasp_mol_dir is not None
                    else None
                )
                for r in res_list:
                    pid = r.placement_id
                    poscar_path = (
                        str(step_mol_vasp / f"conformer_{pid:03d}" / "POSCAR")
                        if step_mol_vasp is not None
                        else None
                    )
                    prow = r.to_row(
                        xyz_path=str(step_mol_xyz / f"conformer_{pid:03d}.xyz"),
                        poscar_path=poscar_path,
                        context_row=context_row,
                    ) | {
                        "molecules": mol_label,
                        "winning_molecule": step_result.winning_molecule,
                        "step": step,
                        "molecule": r.molecule,
                    }
                    prow["schema_version"] = SCHEMA_VERSION
                    placement_rows.append(prow)

    if detail_rows:
        df = pd.DataFrame(detail_rows)
        df.to_csv(f"{results_dir}/saturation_details.csv", index=False)

    if save_all and placement_rows:
        pdf = pd.DataFrame(placement_rows)
        pdf.to_csv(f"{results_dir}/saturation_placements_detailed.csv", index=False)

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
        if vasp_mol_dir is not None:
            vasp_subdir = f"{vasp_mol_dir}/step_{step:03d}"
            _write_vasp_inputs(
                best.atoms,
                vasp_subdir,
                step_result.winning_molecule,
                system_name=None,
                config=config,
            )
        if save_all:
            step_placements_rel = f"step_{step:03d}_placements"
            for pmol, res_list in step_result.per_molecule_results.items():
                if not res_list:
                    continue
                step_mol_xyz = Path(mol_dir) / step_placements_rel / pmol
                os.makedirs(step_mol_xyz, exist_ok=True)
                placement_vasp_dir: Path | None = None
                if vasp_mol_dir is not None:
                    placement_vasp_dir = Path(vasp_mol_dir) / step_placements_rel / pmol
                    os.makedirs(placement_vasp_dir, exist_ok=True)
                for r in res_list:
                    _write_placement_artifacts(
                        r,
                        xyz_dir=step_mol_xyz,
                        adsorbate_xyz_dir=step_mol_xyz,
                        vasp_parent=placement_vasp_dir or vasp_mol_dir or "",
                        molecule_name=pmol,
                        system_name=None,
                        config=config,
                        log_save=False,
                    )

    if result.final_slab_atoms is not None:
        final_slab_copy = result.final_slab_atoms.copy()
        final_slab_copy.calc = None
        final_slab_copy.write(f"{mol_dir}/final_saturated_slab.xyz", format="extxyz")

    write_run_settings(
        surface_type,
        config,
        n_molecules=len(result.molecules),
        total_steps=len(result.steps),
        n_molecules_at_saturation=result.n_molecules_at_saturation,
    )
    logger.info("Saved multi-mol saturation results to %s", results_dir)


def write_run_metadata_from_out(
    run_metadata_out: dict[str, Any],
    *,
    surface_type: str,
    config: AdsorptionConfig,
    molecules: list[tuple[str, str]] | str,
) -> None:
    """Persist run metadata from a populated ``run_metadata_out`` dict."""
    if not run_metadata_out:
        return
    smiles_file = molecules if isinstance(molecules, str) else "<inline-molecules>"
    write_run_metadata(
        surface_type=surface_type,
        config=config,
        smiles_file=smiles_file,
        n_molecules=int(run_metadata_out["n_molecules"]),
        total_configs=int(run_metadata_out["total_configs"]),
        t_ref_s=float(run_metadata_out["t_ref_s"]),
        t_total_s=float(run_metadata_out["t_total_s"]),
    )


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

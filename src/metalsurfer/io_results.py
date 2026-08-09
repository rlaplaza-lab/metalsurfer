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
    MultiMolSaturationStepResult,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)

logger = logging.getLogger(__name__)


def results_dir_for(surface_type: str) -> Path:
    """Return ``results_{surface_type}/`` for a campaign *surface_type* label."""
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


def _merge_run_metadata(
    existing: dict[str, Any] | None,
    updates: dict[str, Any],
) -> dict[str, Any]:
    """Deep-merge *updates* into *existing* run metadata."""
    if existing is None:
        return dict(updates)
    merged = dict(existing)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_run_metadata(
                cast(dict[str, Any], merged[key]),
                cast(dict[str, Any], value),
            )
        else:
            merged[key] = value
    return merged


def _write_run_metadata_file(results_dir: Path, metadata: dict[str, Any]) -> Path:
    """Write run metadata JSON under *results_dir*, merging with any existing file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / "run_metadata.json"
    existing: dict[str, Any] | None = None
    if path.is_file():
        with path.open() as f:
            existing = json.load(f)
    merged = _merge_run_metadata(existing, metadata)
    with path.open("w") as f:
        json.dump(merged, f, indent=2, default=str)
    return path


def write_run_settings(
    surface_type: str,
    config: AdsorptionConfig,
    **run_info: Any,
) -> None:
    """Persist run config and optional run info to ``run_metadata.json``.

    Merges with any existing metadata in the results directory (for example
    timing blocks written by :func:`write_run_metadata`).
    """
    results_dir = results_dir_for(surface_type)
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

    results_dir = results_dir_for(surface_type)
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
    context. Lean default writes ``context_hash`` / ``schema_version`` only;
    set ``export_placement_provenance=True`` for full ``ctx_*`` settings and
    ``initial_*`` placement provenance.
    """
    results_dir = results_dir_for(surface_type)
    include_provenance = bool(config.export_placement_provenance if config else False)
    context_row = (
        config_to_context_row(config, include_provenance=include_provenance)
        if config
        else {}
    )
    write_vasp = config.write_vasp_inputs if config else False
    all_rows: list[dict[str, Any]] = []
    for rr in run_results:
        for row in rr.to_rows(
            results_dir=results_dir,
            context_row=context_row,
            write_vasp_inputs=write_vasp,
            include_provenance=include_provenance,
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


def _saturation_results_dirs(
    surface_type: str, *, write_vasp: bool
) -> tuple[str, str, str | None]:
    """Return ``(results_dir, xyz_dir, vasp_base_or_None)``, creating dirs."""
    results_dir = f"results_{surface_type}"
    os.makedirs(results_dir, exist_ok=True)
    xyz_dir = f"{results_dir}/xyz_structures"
    os.makedirs(xyz_dir, exist_ok=True)
    vasp_base = f"{results_dir}/vasp_inputs" if write_vasp else None
    return results_dir, xyz_dir, vasp_base


def _write_saturation_csv_bundle(
    results_dir: str,
    *,
    detail_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    save_all: bool,
) -> None:
    if detail_rows:
        pd.DataFrame(detail_rows).to_csv(
            f"{results_dir}/saturation_details.csv", index=False
        )
    if save_all and placement_rows:
        pd.DataFrame(placement_rows).to_csv(
            f"{results_dir}/saturation_placements_detailed.csv", index=False
        )
    pd.DataFrame(summary_rows).to_csv(
        f"{results_dir}/saturation_summary.csv", index=False
    )


def _write_saturation_step_bundle(
    best: ScreeningResult,
    *,
    mol_dir: str,
    step: int,
    vasp_mol_dir: str | None,
    molecule_name: str,
    config: AdsorptionConfig,
) -> None:
    _write_saturation_step_xyz(best, mol_dir, step)
    if vasp_mol_dir is not None:
        _write_vasp_inputs(
            best.atoms,
            f"{vasp_mol_dir}/step_{step:03d}",
            molecule_name,
            system_name=None,
            config=config,
        )


def _write_saturation_placement_tree(
    results: Sequence[ScreeningResult],
    *,
    step_xyz: Path,
    step_vasp: Path | None,
    fallback_vasp: str | None,
    molecule_name: str,
    config: AdsorptionConfig,
) -> None:
    if not results:
        return
    os.makedirs(step_xyz, exist_ok=True)
    if step_vasp is not None:
        os.makedirs(step_vasp, exist_ok=True)
    vasp_parent = step_vasp or fallback_vasp or ""
    for entry in results:
        _write_placement_artifacts(
            entry,
            xyz_dir=step_xyz,
            adsorbate_xyz_dir=step_xyz,
            vasp_parent=vasp_parent,
            molecule_name=molecule_name,
            system_name=None,
            config=config,
            log_save=False,
        )


def _write_final_saturated_slab(atoms: Atoms | None, mol_dir: str) -> None:
    if atoms is None:
        return
    final = atoms.copy()
    final.calc = None
    final.write(f"{mol_dir}/final_saturated_slab.xyz", format="extxyz")


def _ensure_saturation_mol_dirs(mol_dir: str, vasp_mol_dir: str | None) -> None:
    os.makedirs(mol_dir, exist_ok=True)
    if vasp_mol_dir is not None:
        os.makedirs(vasp_mol_dir, exist_ok=True)


def _write_saturation_run_structures(
    *,
    mol_dir: str,
    vasp_mol_dir: str | None,
    steps: Sequence[SaturationStepResult | MultiMolSaturationStepResult],
    config: AdsorptionConfig,
    save_all: bool,
    molecule_name_for_step,
) -> None:
    """Write per-step best slabs and optional placement trees for a saturation run."""
    _ensure_saturation_mol_dirs(mol_dir, vasp_mol_dir)
    for step_result in steps:
        step = step_result.step
        _write_saturation_step_bundle(
            step_result.best_result,
            mol_dir=mol_dir,
            step=step,
            vasp_mol_dir=vasp_mol_dir,
            molecule_name=molecule_name_for_step(step_result),
            config=config,
        )
        if not save_all:
            continue
        rel = f"step_{step:03d}_placements"
        if isinstance(step_result, MultiMolSaturationStepResult):
            for pmol, res_list in step_result.per_molecule_results.items():
                _write_saturation_placement_tree(
                    res_list,
                    step_xyz=Path(mol_dir) / rel / pmol,
                    step_vasp=(
                        Path(vasp_mol_dir) / rel / pmol
                        if vasp_mol_dir is not None
                        else None
                    ),
                    fallback_vasp=vasp_mol_dir,
                    molecule_name=pmol,
                    config=config,
                )
        else:
            _write_saturation_placement_tree(
                step_result.all_results,
                step_xyz=Path(mol_dir) / rel,
                step_vasp=(Path(vasp_mol_dir) / rel if vasp_mol_dir else None),
                fallback_vasp=vasp_mol_dir,
                molecule_name=step_result.molecule,
                config=config,
            )


def _persist_saturation_outputs(
    *,
    surface_type: str,
    config: AdsorptionConfig,
    results_dir: str,
    xyz_dir: str,
    vasp_base: str | None,
    detail_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    structure_runs: Sequence[
        tuple[
            str,
            Sequence[SaturationStepResult | MultiMolSaturationStepResult],
            Atoms | None,
            Any,
        ]
    ],
    run_settings_kwargs: dict[str, Any],
    log_message: str,
) -> None:
    """Shared writer for single- and multi-molecule saturation artifacts.

    Writes detail/placement/summary CSVs, per-run step trees + final slabs, and
    run settings. Callers only specialize label columns when building rows.
    """
    save_all = config.saturation_save_all_placements
    _write_saturation_csv_bundle(
        results_dir,
        detail_rows=detail_rows,
        placement_rows=placement_rows,
        summary_rows=summary_rows,
        save_all=save_all,
    )
    for dir_label, steps, final_slab_atoms, molecule_name_for_step in structure_runs:
        mol_dir = f"{xyz_dir}/{dir_label}_saturation"
        vasp_mol_dir = (
            f"{vasp_base}/{dir_label}_saturation" if vasp_base is not None else None
        )
        _write_saturation_run_structures(
            mol_dir=mol_dir,
            vasp_mol_dir=vasp_mol_dir,
            steps=steps,
            config=config,
            save_all=save_all,
            molecule_name_for_step=molecule_name_for_step,
        )
        _write_final_saturated_slab(final_slab_atoms, mol_dir)
    write_run_settings(surface_type, config, **run_settings_kwargs)
    logger.info(log_message, results_dir)


def _collect_saturation_csv_rows(
    steps: Sequence[SaturationStepResult | MultiMolSaturationStepResult],
    *,
    results_dir: str | Path,
    context_row: dict[str, Any],
    include_provenance: bool,
    save_all: bool,
    write_vasp: bool,
    detail_kwargs: dict[str, Any],
    rows_kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    for step_result in steps:
        detail_row = step_result.to_detail_row(
            results_dir=results_dir,
            context_row=context_row,
            include_provenance=include_provenance,
            **detail_kwargs,
        )
        detail_row["schema_version"] = SCHEMA_VERSION
        detail_rows.append(detail_row)
        if not save_all:
            continue
        for prow in step_result.to_rows(
            results_dir=results_dir,
            context_row=context_row,
            write_vasp_inputs=write_vasp,
            include_provenance=include_provenance,
            **rows_kwargs,
        ):
            prow["schema_version"] = SCHEMA_VERSION
            placement_rows.append(prow)
    return detail_rows, placement_rows


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
    if len(single_results) > 1:
        logger.warning(
            "save_saturation_results received %d single-molecule results; "
            "all entries will be saved (no per-molecule truncation)",
            len(single_results),
        )
    write_vasp = config.write_vasp_inputs
    results_dir, xyz_dir, vasp_base = _saturation_results_dirs(
        surface_type, write_vasp=write_vasp
    )
    include_provenance = bool(config.export_placement_provenance)
    context_row = config_to_context_row(config, include_provenance=include_provenance)
    detail_rows: list[dict[str, Any]] = []
    placement_rows: list[dict[str, Any]] = []
    save_all = config.saturation_save_all_placements

    for sr in single_results:
        d_rows, p_rows = _collect_saturation_csv_rows(
            sr.steps,
            results_dir=results_dir,
            context_row=context_row,
            include_provenance=include_provenance,
            save_all=save_all,
            write_vasp=write_vasp,
            detail_kwargs={"saturation_molecule": sr.molecule},
            rows_kwargs={"saturation_molecule": sr.molecule, "step_prefix": True},
        )
        detail_rows.extend(d_rows)
        placement_rows.extend(p_rows)

    summary_rows = [
        {
            "molecule": sr.molecule,
            "n_molecules_at_saturation": sr.n_molecules_at_saturation,
            "n_steps": len(sr.steps),
            "final_slab_path": (
                f"{xyz_dir}/{sr.molecule}_saturation/final_saturated_slab.xyz"
                if sr.final_slab_atoms is not None
                else ""
            ),
        }
        for sr in single_results
    ]
    _persist_saturation_outputs(
        surface_type=surface_type,
        config=config,
        results_dir=results_dir,
        xyz_dir=xyz_dir,
        vasp_base=vasp_base,
        detail_rows=detail_rows,
        placement_rows=placement_rows,
        summary_rows=summary_rows,
        structure_runs=[
            (
                sr.molecule,
                sr.steps,
                sr.final_slab_atoms,
                (lambda _step, mol=sr.molecule: mol),
            )
            for sr in single_results
        ],
        run_settings_kwargs={
            "n_molecules": len(single_results),
            "total_steps": sum(len(sr.steps) for sr in single_results),
            "n_molecules_at_saturation": sum(
                sr.n_molecules_at_saturation for sr in single_results
            ),
        },
        log_message="Saved saturation results to %s",
    )


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

    write_vasp = config.write_vasp_inputs
    results_dir, xyz_dir, vasp_base = _saturation_results_dirs(
        surface_type, write_vasp=write_vasp
    )
    mol_label = "_".join(result.molecules)
    mol_dir = f"{xyz_dir}/{mol_label}_saturation"

    include_provenance = bool(config.export_placement_provenance)
    context_row = config_to_context_row(config, include_provenance=include_provenance)
    detail_rows, placement_rows = _collect_saturation_csv_rows(
        result.steps,
        results_dir=results_dir,
        context_row=context_row,
        include_provenance=include_provenance,
        save_all=config.saturation_save_all_placements,
        write_vasp=write_vasp,
        detail_kwargs={"molecules_label": mol_label},
        rows_kwargs={"molecules_label": mol_label},
    )

    _persist_saturation_outputs(
        surface_type=surface_type,
        config=config,
        results_dir=results_dir,
        xyz_dir=xyz_dir,
        vasp_base=vasp_base,
        detail_rows=detail_rows,
        placement_rows=placement_rows,
        summary_rows=[
            {
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
        ],
        structure_runs=[
            (
                mol_label,
                result.steps,
                result.final_slab_atoms,
                lambda sr: sr.winning_molecule,
            )
        ],
        run_settings_kwargs={
            "n_molecules": len(result.molecules),
            "total_steps": len(result.steps),
            "n_molecules_at_saturation": result.n_molecules_at_saturation,
        },
        log_message="Saved multi-mol saturation results to %s",
    )


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
    """Persist reproducible run metadata as JSON in ``run_metadata.json``.

    Merges with any existing metadata in the results directory (for example
    campaign fields written by :func:`write_run_settings`).
    """
    results_dir = results_dir_for(surface_type)

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


def _write_saturation_step_xyz(best: ScreeningResult, mol_dir: str, step: int) -> None:
    """Write best-slab, energy-tagged, and adsorbate-only XYZ for one saturation step."""
    step_structure_path = f"{mol_dir}/step_{step:03d}_best_slab.xyz"
    step_energy_path = (
        f"{mol_dir}/step_{step:03d}_Eads_{best.energy_adsorption:.4f}.xyz"
    )
    step_adsorbate_path = f"{mol_dir}/step_{step:03d}_adsorbate.xyz"
    best_atoms_copy = best.atoms.copy()
    best_atoms_copy.calc = None
    best_atoms_copy.write(step_structure_path, format="extxyz")
    best_atoms_copy.write(step_energy_path, format="extxyz")
    adsorbate = best.atoms[best.slab_size :].copy()
    _write_clean_xyz(adsorbate, step_adsorbate_path)

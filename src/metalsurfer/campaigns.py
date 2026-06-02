"""High-level campaign APIs for adsorption screening library scripts and workflows."""

from __future__ import annotations

import dataclasses as _dc
import logging
import time
from typing import Any, Literal, cast

from ase import Atoms

from .config import AdsorptionConfig
from .io_results import (
    save_saturation_results,
    save_single_molecule_results,
    save_summary_results,
    screening_run_result,
    setup_directories,
    write_run_metadata,
    write_run_metadata_from_out,
    write_run_settings,
)
from .models import (
    BindingCampaignResult,
    MoleculeCampaignSummary,
    MultiMolSaturationRunResult,
    SaturationCampaignResult,
    SaturationRunResult,
    ScreeningResult,
)
from .optimization import setup_single_model
from .placement import classify_adsorbate_orientation
from .surfaces import SlabContainer, coerce_slab_container
from .workflow import (
    calculate_reference_energies,
    process_molecule,
    process_molecule_bayesian,
    run_saturation_screening,
)

logger = logging.getLogger(__name__)


def _summarize_molecule(
    molecule_name: str,
    results: list[ScreeningResult],
    surface_symbols: set[str],
) -> MoleculeCampaignSummary:
    if not results:
        return MoleculeCampaignSummary(
            molecule=molecule_name,
            n_valid_placements=0,
            best_adsorption_energy=None,
        )
    best = min(results, key=lambda r: r.energy_adsorption)
    slab_size = next(
        (
            i
            for i, sym in enumerate(results[0].atoms.get_chemical_symbols())
            if sym not in surface_symbols
        ),
        None,
    )
    n_parallel = 0
    n_endown = 0
    if slab_size is not None:
        orientations = [
            classify_adsorbate_orientation(r.atoms, slab_size) for r in results
        ]
        n_parallel = sum(1 for ori in orientations if ori == "parallel")
        n_endown = sum(1 for ori in orientations if ori == "EN-down")
    return MoleculeCampaignSummary(
        molecule=molecule_name,
        n_valid_placements=len(results),
        best_adsorption_energy=best.energy_adsorption,
        n_parallel=n_parallel,
        n_endown=n_endown,
    )


def _run_binding_campaign(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    mode: str,
    process_fn,
    system_name: str | None,
    save_results: bool,
    write_settings: bool,
    write_metadata: bool,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    # When molecules is a CSV path, delegate entirely to the workflow
    # screening loop which supports dataset logging and skip-existing semantics.
    if isinstance(molecules, str):
        from .workflow.screening import _run_screening_common as _wf_run

        run_results = _wf_run(
            slab=slab,
            smiles_file=molecules,
            config=config,
            surface_type=surface_type,
            skip_existing=skip_existing,
            run_metadata_out=run_metadata_out,
            process_fn=process_fn,
            completion_label="BO screening complete"
            if mode == "bo"
            else "Screening complete",
        )
        total_configurations = sum(len(rr.results) for rr in run_results)
        if write_metadata and run_metadata_out:
            write_run_metadata_from_out(
                run_metadata_out,
                surface_type=surface_type,
                config=config,
                molecules=molecules,
            )
        return BindingCampaignResult(
            mode="bo" if mode == "bo" else "non_bo",
            surface_type=surface_type,
            run_results=run_results,
            molecule_summaries=[],
            total_configurations=total_configurations,
            n_molecules=len(run_results),
            t_ref_s=float(run_metadata_out.get("t_ref_s", 0.0))
            if run_metadata_out
            else 0.0,
            t_total_s=float(run_metadata_out.get("t_total_s", 0.0))
            if run_metadata_out
            else 0.0,
            failure_summaries={},
        )

    if not molecules:
        raise ValueError("molecules must be a non-empty list")
    process_kwargs = process_kwargs or {}
    slab = coerce_slab_container(slab)
    t_start = time.perf_counter()

    setup_directories([surface_type])
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    smiles_list = [s for s, _ in molecules]
    molecule_names = [n for _, n in molecules]
    t_ref_start = time.perf_counter()
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=molecule_names,
        smiles_list=smiles_list,
        ts_model=ts_model,
        config=config,
    )
    t_ref_s = time.perf_counter() - t_ref_start

    run_results = []
    summaries = []
    failure_summaries: dict[str, dict[str, object]] = {}
    surface_symbols = set(slab.atoms.get_chemical_symbols())

    for smiles, molecule_name in molecules:
        failure_summary: dict[str, object] = {}
        results = process_fn(
            smiles,
            molecule_name,
            slab,
            calculator,
            ref,
            ts_model=ts_model,
            config=config,
            surface_type=surface_type,
            failure_summary_out=failure_summary,
            **process_kwargs,
        )
        if failure_summary:
            failure_summaries[molecule_name] = failure_summary
        summaries.append(_summarize_molecule(molecule_name, results, surface_symbols))
        if not results:
            continue
        if save_results:
            save_single_molecule_results(
                molecule_name,
                results,
                surface_type=surface_type,
                system_name=system_name,
                config=config,
                write_csv=False,
            )
        run_results.append(screening_run_result(molecule_name, results))

    if save_results and run_results:
        save_summary_results(run_results, surface_type=surface_type, config=config)
    total_configurations = sum(len(rr.results) for rr in run_results)
    t_total_s = time.perf_counter() - t_start
    if write_settings:
        write_run_settings(
            surface_type,
            config,
            campaign="multi_molecule_binding",
            n_molecules=len(run_results),
            molecules=[rr.molecule for rr in run_results],
            n_configurations=total_configurations,
            mode=mode,
        )
    if write_metadata:
        write_run_metadata(
            surface_type=surface_type,
            config=config,
            smiles_file="<inline-molecules>",
            n_molecules=len(molecule_names),
            total_configs=total_configurations,
            t_ref_s=t_ref_s,
            t_total_s=t_total_s,
        )
    return BindingCampaignResult(
        mode="bo" if mode == "bo" else "non_bo",
        surface_type=surface_type,
        run_results=run_results,
        molecule_summaries=summaries,
        total_configurations=total_configurations,
        n_molecules=len(molecule_names),
        t_ref_s=t_ref_s,
        t_total_s=t_total_s,
        failure_summaries=failure_summaries,
    )


def run_adsorption(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    system_name: str | None = None,
    save_results: bool = True,
    write_settings: bool = True,
    write_metadata: bool = False,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    """Multi-molecule adsorption screening (non-BO; uses :func:`~metalsurfer.workflow.core.process_molecule`).

    Parameters
    ----------
    slab:
        :class:`~metalsurfer.surfaces.SlabContainer` or plain :class:`ase.Atoms`.
    molecules:
        In-memory ``(smiles, name)`` list or path to a CSV with ``smiles`` and ``name`` columns.
    config:
        Screening configuration.
    surface_type:
        Label used to name the ``results_{surface_type}/`` output directory.
    system_name:
        Optional system identifier written into per-molecule XYZ files.
    save_results:
        Whether to write CSV/XYZ/POSCAR output files.
    write_settings:
        Whether to write a ``run_settings.json`` file.
    write_metadata:
        Whether to write a ``run_metadata.json`` file.
    skip_existing:
        When *molecules* is a CSV path, skip molecules already present in
        the existing summary CSV (has no effect for in-memory lists).
    run_metadata_out:
        Optional dict to populate with timing and count metadata.
    process_kwargs:
        Extra keyword arguments forwarded to
        :func:`~metalsurfer.workflow.core.process_molecule`.
    """
    return _run_binding_campaign(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="non_bo",
        process_fn=process_molecule,
        system_name=system_name,
        save_results=save_results,
        write_settings=write_settings,
        write_metadata=write_metadata,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
        process_kwargs=process_kwargs,
    )


def run_adsorption_bo(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    system_name: str | None = None,
    save_results: bool = True,
    write_settings: bool = True,
    write_metadata: bool = False,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    """Multi-molecule adsorption screening with BO (``bo_enabled`` forced on).

    Parameters
    ----------
    slab:
        :class:`~metalsurfer.surfaces.SlabContainer` or plain :class:`ase.Atoms`.
    molecules:
        In-memory list or CSV path (``smiles``, ``name``).
    config:
        Screening configuration.  ``bo_enabled`` is forced to ``True``.
    surface_type:
        Label used to name the ``results_{surface_type}/`` output directory.
    system_name:
        Optional system identifier for per-molecule XYZ files.
    save_results:
        Whether to write CSV/XYZ/POSCAR output files.
    write_settings:
        Whether to write a ``run_settings.json`` file.
    write_metadata:
        Whether to write a ``run_metadata.json`` file.
    skip_existing:
        Skip molecules already in the summary (CSV input only).
    run_metadata_out:
        Optional dict to populate with timing and count metadata.
    process_kwargs:
        Extra keyword arguments forwarded to
        :func:`~metalsurfer.workflow.bayesian.process_molecule_bayesian`.
    """
    config = _dc.replace(config, bo_enabled=True)
    return _run_binding_campaign(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="bo",
        process_fn=process_molecule_bayesian,
        system_name=system_name,
        save_results=save_results,
        write_settings=write_settings,
        write_metadata=write_metadata,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
        process_kwargs=process_kwargs,
    )


def _save_benchmark_dataset_if_requested(
    results: list[SaturationRunResult | MultiMolSaturationRunResult],
    *,
    surface_type: str,
    config: AdsorptionConfig,
) -> None:
    if not config.save_benchmark_dataset:
        return
    flattened = [
        run
        for sr in results
        if isinstance(sr, SaturationRunResult)
        for run in sr.to_flattened_runs()
    ]
    if flattened:
        save_summary_results(flattened, surface_type=surface_type, config=config)


def _run_saturation_campaign(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    mode: Literal["non_bo", "bo"],
    save_results: bool,
    write_settings: bool,
    write_metadata: bool,
    skip_existing: bool,
    run_metadata_out: dict[str, Any] | None,
) -> SaturationCampaignResult:
    setup_directories([surface_type])
    failure_summary: dict[str, object] = {}
    run_metadata: dict[str, Any] = (
        run_metadata_out if run_metadata_out is not None else {}
    )

    results = run_saturation_screening(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        skip_existing=skip_existing,
        failure_summary_out=failure_summary,
        run_metadata_out=run_metadata,
    )
    runs = cast(
        list[SaturationRunResult | MultiMolSaturationRunResult],
        list(results),
    )

    if save_results:
        save_saturation_results(runs, surface_type=surface_type, config=config)
        _save_benchmark_dataset_if_requested(
            runs, surface_type=surface_type, config=config
        )
    if write_settings:
        write_run_settings(surface_type, config)
    if write_metadata and run_metadata:
        write_run_metadata_from_out(
            run_metadata,
            surface_type=surface_type,
            config=config,
            molecules=molecules,
        )

    return SaturationCampaignResult(
        mode=mode,
        surface_type=surface_type,
        runs=runs,
        failure_summary=failure_summary,
        t_ref_s=float(run_metadata.get("t_ref_s", 0.0)),
        t_total_s=float(run_metadata.get("t_total_s", 0.0)),
    )


def run_saturation(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    save_results: bool = True,
    write_settings: bool = True,
    write_metadata: bool = False,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> SaturationCampaignResult:
    """Sequential saturation (non-BO) until best E_ads ≥ 0 or no valid placements.

    Parameters
    ----------
    slab:
        :class:`~metalsurfer.surfaces.SlabContainer` or plain :class:`ase.Atoms`.
    molecules:
        In-memory list or CSV path (default ``"smiles.csv"``).
    config:
        Screening configuration.
    surface_type:
        Label for the ``results_{surface_type}/`` output directory.
    save_results:
        Whether to write CSV/XYZ/POSCAR output files.
    write_settings:
        Whether to write a ``run_settings.json`` file.
    write_metadata:
        Whether to write a ``run_metadata.json`` file.
    skip_existing:
        Skip molecules already in the saturation summary (CSV input only).
    run_metadata_out:
        Optional dict populated with timing and count metadata.

    With ``save_results`` true, calls :func:`save_saturation_results` with the same ``config``.
    When ``config.save_benchmark_dataset`` is true, also writes
    ``adsorption_energies_detailed.csv`` from flattened step placements.
    """
    if config is None:
        config = AdsorptionConfig()
    return _run_saturation_campaign(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="non_bo",
        save_results=save_results,
        write_settings=write_settings,
        write_metadata=write_metadata,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
    )


def run_saturation_bo(
    *,
    slab: SlabContainer | Atoms,
    molecules: list[tuple[str, str]] | str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    save_results: bool = True,
    write_settings: bool = True,
    write_metadata: bool = False,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> SaturationCampaignResult:
    """Saturation with BO-guided placement selection.

    Parameters
    ----------
    slab:
        :class:`~metalsurfer.surfaces.SlabContainer` or plain :class:`ase.Atoms`.
    molecules:
        In-memory list or CSV path (default ``"smiles.csv"``).
    config:
        Screening configuration; ``bo_enabled`` is set to ``True``.
    surface_type:
        Label for the ``results_{surface_type}/`` output directory.
    save_results:
        Whether to write CSV/XYZ/POSCAR output files.
    write_settings:
        Whether to write a ``run_settings.json`` file.
    write_metadata:
        Whether to write a ``run_metadata.json`` file.
    skip_existing:
        Skip molecules already in the saturation summary (CSV input only).
    run_metadata_out:
        Optional dict populated with timing and count metadata.

    With ``save_results`` true, calls :func:`save_saturation_results` with the same ``config``
    (after ``bo_enabled`` is set true).
    """
    if config is None:
        config = AdsorptionConfig()
    config = _dc.replace(config, bo_enabled=True)
    return _run_saturation_campaign(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="bo",
        save_results=save_results,
        write_settings=write_settings,
        write_metadata=write_metadata,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
    )

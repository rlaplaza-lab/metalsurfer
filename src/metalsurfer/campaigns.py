"""High-level campaign APIs for adsorption screening library scripts and workflows."""

import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from .campaign_schema import CampaignDocument
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
from .reporting import FailureSummary
from .result_paths import ADSORPTION_ENERGIES_DETAILED_CSV, results_dir_for
from .surface_prep import Substrate, prepare_substrate
from .workflow import (
    process_molecule,
    process_molecule_bayesian,
    run_saturation_screening,
)
from .workflow.shared import (
    _bootstrap_screening_run,
    _normalize_molecules_input,
    empty_molecule_input_message,
)

logger = logging.getLogger(__name__)


def _summarize_molecule(
    molecule_name: str,
    results: list[ScreeningResult],
) -> MoleculeCampaignSummary:
    if not results:
        return MoleculeCampaignSummary(
            molecule=molecule_name,
            n_valid_placements=0,
            best_adsorption_energy=None,
        )
    best = min(results, key=lambda r: r.energy_adsorption)
    return MoleculeCampaignSummary(
        molecule=molecule_name,
        n_valid_placements=len(results),
        best_adsorption_energy=best.energy_adsorption,
    )


CampaignState = dict[str, Any]


def _write_campaign_run_settings(
    surface_type: str,
    config: AdsorptionConfig,
    *,
    campaign: str,
    mode: Literal["bo", "non_bo"],
    molecule_names: list[str],
    n_configurations: int | None = None,
) -> None:
    """Write shared campaign fields for run settings."""
    run_info: dict[str, Any] = {
        "campaign": campaign,
        "n_molecules": len(molecule_names),
        "molecules": molecule_names,
    }
    if n_configurations is not None:
        run_info["n_configurations"] = n_configurations
    write_run_settings(surface_type, config, mode=mode, **run_info)


def _run_campaign_lifecycle(
    *,
    surface_type: str,
    config: AdsorptionConfig,
    run: Callable[[], CampaignState],
    build_result: Callable[[CampaignState], Any],
) -> Any:
    """Drive setup, execution, and result construction for a campaign."""
    setup_directories([surface_type], write_vasp_inputs=config.write_vasp_inputs)
    return build_result(run())


def _run_binding_campaign(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    mode: Literal["bo", "non_bo"],
    process_fn,
    system_name: str | None,
    save_results: bool,
    write_settings: bool,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    molecule_pairs, load_status, molecules_source = _normalize_molecules_input(
        molecules,
        skip_existing=skip_existing,
        surface_type=surface_type,
    )
    if not molecule_pairs:
        listed_csv = (
            results_dir_for(surface_type) / ADSORPTION_ENERGIES_DETAILED_CSV
        ).as_posix()
        msg = empty_molecule_input_message(load_status, listed_csv=listed_csv)
        if msg is not None:
            logger.warning(msg)
        return BindingCampaignResult(
            mode=mode,
            surface_type=surface_type,
            run_results=[],
            molecule_summaries=[],
            total_configurations=0,
            n_molecules=0,
            t_ref_s=0.0,
            t_total_s=0.0,
            failure_summaries={},
        )

    if process_kwargs is None:
        process_kwargs = {}
    t_start = time.perf_counter()

    def _run() -> CampaignState:
        bootstrap = _bootstrap_screening_run(substrate, molecule_pairs, config)
        slab = bootstrap.slab
        run_results = []
        summaries = []
        failure_summaries: dict[str, FailureSummary] = {}
        for smiles, molecule_name in molecule_pairs:
            outcome = process_fn(
                smiles,
                molecule_name,
                slab,
                bootstrap.calculator,
                bootstrap.ref,
                ts_model=bootstrap.ts_model,
                config=config,
                surface_type=surface_type,
                reference_smiles=smiles,
                **process_kwargs,
            )
            results = outcome.results
            if outcome.failure_summary:
                failure_summaries[molecule_name] = outcome.failure_summary
            summaries.append(_summarize_molecule(molecule_name, results))
            if not results:
                continue
            if save_results:
                save_single_molecule_results(
                    molecule_name,
                    results,
                    surface_type=surface_type,
                    system_name=system_name,
                    config=config,
                    material_type=slab.material_type,
                    write_csv=False,
                )
            run_results.append(screening_run_result(molecule_name, results))
        return {
            "slab": slab,
            "run_results": run_results,
            "summaries": summaries,
            "failure_summaries": failure_summaries,
            "n_molecules": len(molecule_pairs),
            "t_ref_s": bootstrap.t_ref_s,
        }

    def _build_result(state: CampaignState) -> BindingCampaignResult:
        run_results = state["run_results"]
        if save_results and run_results:
            save_summary_results(
                run_results,
                surface_type=surface_type,
                config=config,
                material_type=state["slab"].material_type,
            )
        total_configurations = sum(len(rr.results) for rr in run_results)
        t_total_s = time.perf_counter() - t_start
        if write_settings:
            _write_campaign_run_settings(
                surface_type,
                config,
                campaign="multi_molecule_binding",
                mode=mode,
                molecule_names=[rr.molecule for rr in run_results],
                n_configurations=total_configurations,
            )
            write_run_metadata(
                surface_type=surface_type,
                config=config,
                smiles_file=molecules_source,
                n_molecules=state["n_molecules"],
                total_configs=total_configurations,
                t_ref_s=state["t_ref_s"],
                t_total_s=t_total_s,
            )
        if run_metadata_out is not None:
            run_metadata_out.update(
                n_molecules=state["n_molecules"],
                total_configs=total_configurations,
                t_ref_s=state["t_ref_s"],
                t_total_s=t_total_s,
            )
        return BindingCampaignResult(
            mode=mode,
            surface_type=surface_type,
            run_results=run_results,
            molecule_summaries=state["summaries"],
            total_configurations=total_configurations,
            n_molecules=state["n_molecules"],
            t_ref_s=state["t_ref_s"],
            t_total_s=t_total_s,
            failure_summaries=state["failure_summaries"],
        )

    return _run_campaign_lifecycle(
        surface_type=surface_type,
        config=config,
        run=_run,
        build_result=_build_result,
    )


def run_adsorption(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    system_name: str | None = None,
    save_results: bool = True,
    write_settings: bool = True,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    """Multi-molecule adsorption screening (non-BO; uses :func:`~metalsurfer.workflow.core.process_molecule`).

    Parameters
    ----------
    substrate:
        Prepared :class:`~metalsurfer.surface_prep.Substrate`. Bare ``Atoms`` are rejected.
    molecules:
        In-memory ``(smiles, name)`` list or path to a CSV with ``smiles`` and ``name`` columns.
    config:
        Screening configuration.
    surface_type:
        Label used to name the ``results_{surface_type}/`` output directory.
    system_name:
        Optional system identifier written into per-molecule XYZ files.
    save_results:
        Whether to write CSV/XYZ output files. VASP bundles require
        ``config.write_vasp_inputs=True``.
    write_settings:
        When True (default), write the full ``run_metadata.json`` (config, campaign
        fields, and timing/counts). Set False to suppress the file.
    skip_existing:
        Skip molecules already listed in ``ADSORPTION_ENERGIES_DETAILED_CSV``
        (in-memory lists and CSV paths).
    run_metadata_out:
        Optional dict to populate with timing and count metadata.
    process_kwargs:
        Extra keyword arguments forwarded to
        :func:`~metalsurfer.workflow.core.process_molecule`.
    """
    return _run_binding_campaign(
        substrate=substrate,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="non_bo",
        process_fn=process_molecule,
        system_name=system_name,
        save_results=save_results,
        write_settings=write_settings,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
        process_kwargs=process_kwargs,
    )


def run_adsorption_bo(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    system_name: str | None = None,
    save_results: bool = True,
    write_settings: bool = True,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
    process_kwargs: dict[str, Any] | None = None,
) -> BindingCampaignResult:
    """Run adsorption with BO-guided placement.

    Same as :func:`run_adsorption`, using
    :func:`~metalsurfer.workflow.bayesian.process_molecule_bayesian`.

    Parameters
    ----------
    substrate
        Prepared :class:`~metalsurfer.surface_prep.Substrate`. Bare ``Atoms`` are rejected.
    molecules
        In-memory ``(smiles, name)`` list or path to a CSV with ``smiles`` and ``name`` columns.
    config
        Screening configuration.
    surface_type
        Label used to name the ``results_{surface_type}/`` output directory.
    system_name
        Optional system identifier written into per-molecule XYZ files.
    save_results
        Whether to write CSV/XYZ output files.
    write_settings
        When True (default), write the full ``run_metadata.json``.
    skip_existing
        Skip molecules already listed in ``ADSORPTION_ENERGIES_DETAILED_CSV``.
    run_metadata_out
        Optional dict to populate with timing and count metadata.
    process_kwargs
        Extra keyword arguments forwarded to
        :func:`~metalsurfer.workflow.bayesian.process_molecule_bayesian`.
    """
    return _run_binding_campaign(
        substrate=substrate,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="bo",
        process_fn=process_molecule_bayesian,
        system_name=system_name,
        save_results=save_results,
        write_settings=write_settings,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
        process_kwargs=process_kwargs,
    )


def _save_benchmark_dataset_if_requested(
    results: list[SaturationRunResult | MultiMolSaturationRunResult],
    *,
    surface_type: str,
    config: AdsorptionConfig,
    material_type: str,
) -> None:
    if not config.save_benchmark_dataset:
        return
    flattened = [run for sr in results for run in sr.to_flattened_runs()]
    if flattened:
        save_summary_results(
            flattened,
            surface_type=surface_type,
            config=config,
            material_type=material_type,
        )
    else:
        logger.warning(
            "save_benchmark_dataset=True but no flattened screening runs to write"
        )


def _run_saturation_campaign(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str,
    mode: Literal["non_bo", "bo"],
    save_results: bool,
    write_settings: bool,
    skip_existing: bool,
    run_metadata_out: dict[str, Any] | None,
) -> SaturationCampaignResult:
    run_metadata: dict[str, Any] = (
        run_metadata_out if run_metadata_out is not None else {}
    )

    def _run() -> CampaignState:
        failure_summary: dict[str, FailureSummary] = {}
        results = run_saturation_screening(
            substrate=substrate,
            molecules=molecules,
            config=config,
            surface_type=surface_type,
            skip_existing=skip_existing,
            failure_summary_out=failure_summary,
            run_metadata_out=run_metadata,
            bo_enabled=(mode == "bo"),
        )
        runs = cast(
            list[SaturationRunResult | MultiMolSaturationRunResult],
            list(results),
        )
        return {"runs": runs, "failure_summary": failure_summary}

    def _build_result(state: CampaignState) -> SaturationCampaignResult:
        runs = state["runs"]
        if save_results:
            save_saturation_results(
                runs,
                surface_type=surface_type,
                config=config,
                material_type=substrate.material_type,
            )
            _save_benchmark_dataset_if_requested(
                runs,
                surface_type=surface_type,
                config=config,
                material_type=substrate.material_type,
            )
        if write_settings:
            molecule_names = [
                name
                for run in runs
                for name in (
                    run.molecules
                    if isinstance(run, MultiMolSaturationRunResult)
                    else [run.molecule]
                )
            ]
            _write_campaign_run_settings(
                surface_type,
                config,
                campaign="saturation",
                mode=mode,
                molecule_names=molecule_names,
            )
            if run_metadata:
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
            failure_summary=state["failure_summary"],
            t_ref_s=float(run_metadata.get("t_ref_s", 0.0)),
            t_total_s=float(run_metadata.get("t_total_s", 0.0)),
        )

    return _run_campaign_lifecycle(
        surface_type=surface_type,
        config=config,
        run=_run,
        build_result=_build_result,
    )


def run_saturation(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str = "manual",
    save_results: bool = True,
    write_settings: bool = True,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> SaturationCampaignResult:
    """Sequential saturation (non-BO) until ranking energy Ω ≥ 0 or no valid placements.

    Optional ``saturation_temperature`` / ``saturation_pressure`` /
    ``saturation_activities`` / ``saturation_omega_shift`` set Ω
    (SATP defaults recover electronic E_ads).

    Parameters
    ----------
    substrate:
        Prepared :class:`~metalsurfer.surface_prep.Substrate`. Bare ``Atoms`` are rejected.
    molecules:
        In-memory list or CSV path (``smiles``, ``name`` columns).
    config:
        Screening configuration.
    surface_type:
        Label for the ``results_{surface_type}/`` output directory.
    save_results:
        Whether to write CSV/XYZ output files. VASP bundles require
        ``config.write_vasp_inputs=True``.
    write_settings:
        When True (default), write the full ``run_metadata.json`` (config, campaign
        fields, and timing/counts when available). Set False to suppress the file.
    skip_existing:
        Skip molecules already listed in ``SATURATION_SUMMARY_CSV``.
    run_metadata_out:
        Optional dict populated with timing and count metadata.

    Notes
    -----
    With ``save_results`` true, calls :func:`save_saturation_results` with the same ``config``.
    When ``config.save_benchmark_dataset`` is true, also writes
    ``ADSORPTION_ENERGIES_DETAILED_CSV`` from flattened step placements.
    """
    return _run_saturation_campaign(
        substrate=substrate,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="non_bo",
        save_results=save_results,
        write_settings=write_settings,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
    )


def run_saturation_bo(
    *,
    substrate: Substrate,
    molecules: list[tuple[str, str]] | str,
    config: AdsorptionConfig,
    surface_type: str = "manual",
    save_results: bool = True,
    write_settings: bool = True,
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> SaturationCampaignResult:
    """Run saturation with BO-guided placement selection.

    Reservoir Ω ranking applies on the shared commit/stop path; surrogate
    labels remain electronic ``E_ads``.

    Parameters
    ----------
    substrate
        Prepared :class:`~metalsurfer.surface_prep.Substrate`. Bare ``Atoms`` are rejected.
    molecules
        In-memory list or CSV path (``smiles``, ``name`` columns).
    config
        Screening configuration.
    surface_type
        Label for the ``results_{surface_type}/`` output directory.
    save_results
        Whether to write CSV/XYZ output files.
    write_settings
        When True (default), write the full ``run_metadata.json``.
    skip_existing
        Skip molecules already listed in ``SATURATION_SUMMARY_CSV``.
    run_metadata_out
        Optional dict populated with timing and count metadata.
    """
    return _run_saturation_campaign(
        substrate=substrate,
        molecules=molecules,
        config=config,
        surface_type=surface_type,
        mode="bo",
        save_results=save_results,
        write_settings=write_settings,
        skip_existing=skip_existing,
        run_metadata_out=run_metadata_out,
    )


_RUNNERS: dict[str, Callable[..., Any]] = {
    "adsorption": run_adsorption,
    "adsorption_bo": run_adsorption_bo,
    "saturation": run_saturation,
    "saturation_bo": run_saturation_bo,
}


def run_campaign(document: CampaignDocument, *, skip_existing: bool = True) -> Any:
    """Prepare substrate from a YAML campaign document and dispatch the runner.

    Parameters
    ----------
    document:
        Parsed campaign from :func:`~metalsurfer.load_campaign_yaml` or
        :func:`~metalsurfer.campaign_schema.parse_campaign_dict`.
    skip_existing:
        Skip molecules already present in prior result CSVs (default: True).
    """
    substrate = prepare_substrate(
        **document.substrate,
        config=document.config,
        results_dir=document.results_dir,
    )
    runner = _RUNNERS[document.campaign]
    return runner(
        substrate=substrate,
        molecules=document.molecules,
        config=document.config,
        surface_type=document.surface_type,
        skip_existing=skip_existing,
    )

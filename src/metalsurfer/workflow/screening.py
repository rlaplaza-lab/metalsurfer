"""Batch screening entry points."""

import logging
import time
from typing import Any

from ase import Atoms
from ase.calculators.calculator import Calculator

from .._logging import log_context
from ..config import AdsorptionConfig
from ..ml.dataset import DatasetLogger
from ..ml.schema import PlacementRecord
from ..models import ReferenceEnergies, ScreeningRunResult, build_molecule_summary
from ..optimization import setup_single_model
from ..surfaces import SlabContainer, coerce_slab_container
from .reference import calculate_reference_energies
from .shared import load_molecules, load_molecules_from_pairs

logger = logging.getLogger(__name__)


def _setup_screening_run(
    slab: SlabContainer | Atoms,
    molecules_input: list[tuple[str, str]] | tuple[str, str] | str,
    config: AdsorptionConfig,
    surface_type: str,
    skip_existing: bool,
    skip_saturation_file: bool = False,
) -> tuple[Calculator, Any, list[str], list[str], ReferenceEnergies, float] | None:
    """Setup model, load molecules, and compute reference energies."""
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    if isinstance(molecules_input, str):
        molecules, smiles_list, load_status = load_molecules(
            molecules_input,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_saturation_file,
        )
    else:
        molecules, smiles_list, load_status = load_molecules_from_pairs(
            molecules_input,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_saturation_file,
        )
    if not molecules:
        if load_status == "all_skipped":
            logger.info("No molecules to process (all already in existing summary)")
        elif load_status == "empty_file":
            logger.info("No molecules to process (file empty or no valid rows)")
        else:
            logger.info("No molecules to process")
        return None
    slab_container = coerce_slab_container(slab)
    t_ref_start = time.perf_counter()
    ref = calculate_reference_energies(
        slab_container,
        calculator,
        molecules,
        smiles_list,
        ts_model,
        config=config,
    )
    t_ref_s = time.perf_counter() - t_ref_start
    return (calculator, ts_model, molecules, smiles_list, ref, t_ref_s)


def _run_screening_common(
    *,
    slab: SlabContainer | Atoms,
    smiles_file: str,
    config: AdsorptionConfig | None,
    surface_type: str,
    skip_existing: bool,
    run_metadata_out: dict[str, Any] | None,
    process_fn,
    completion_label: str,
) -> list[ScreeningRunResult]:
    if config is None:
        config = AdsorptionConfig()

    slab = coerce_slab_container(slab, material_type=config.material_type)

    t_run_start = time.perf_counter()
    with log_context(surface_type=surface_type, seed=config.seed):
        setup = _setup_screening_run(
            slab, smiles_file, config, surface_type, skip_existing
        )
        if setup is None:
            return []

        calculator, ts_model, molecules, smiles_list, ref, t_ref_s = setup
        ds_logger = DatasetLogger(
            f"results_{surface_type}",
            config=config,
            surface_id=surface_type,
        )
        all_run_results: list[ScreeningRunResult] = []
        for smi, mol in zip(smiles_list, molecules, strict=True):
            extra_ml_records: list[PlacementRecord] = []
            mol_results = process_fn(
                smi,
                mol,
                slab,
                calculator,
                ref,
                ts_model=ts_model,
                config=config,
                surface_type=surface_type,
                reference_smiles=smi,
                extra_ml_records_out=extra_ml_records,
            )
            if not mol_results:
                for record in extra_ml_records:
                    ds_logger.add_record(record)
                continue
            all_run_results.append(
                ScreeningRunResult(
                    molecule=mol,
                    results=mol_results,
                    summary=build_molecule_summary(mol, mol_results),
                )
            )
            ds_logger.add_results(mol_results, smiles=smi, surface_id=surface_type)
            for record in extra_ml_records:
                ds_logger.add_record(record)

        ds_logger.flush()

    t_run_total = time.perf_counter() - t_run_start
    total_configs = sum(len(r.results) for r in all_run_results)
    logger.info(
        "%s: %d molecules, %d configs, %.1fs total",
        completion_label,
        len(molecules),
        total_configs,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecules),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_run_results

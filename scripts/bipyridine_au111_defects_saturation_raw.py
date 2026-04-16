#!/usr/bin/env python3
"""Generate high-placement, non-BO saturation data for bipyridine on defected Au(111).

Purpose:
- Create an irregular Au(111) surface by depositing extra Au adatoms (defects).
- Run saturation screening with BO disabled so each step is sampled without BO bias.
- Persist standard saturation outputs and flattened adsorption_energies_detailed.csv for BO benchmarking.
"""

import logging
import os

from metalsurfer import (
    AdsorptionConfig,
    format_failure_summary,
    run_saturation,
    setup_single_model,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_saturation_complete
from metalsurfer.io_results import (
    save_saturation_results,
    save_summary_results,
    setup_directories,
)
from metalsurfer.models import ScreeningRunResult, build_molecule_summary
from metalsurfer.surface_prep import create_slab_from_bulk, deposit_adatoms

SURFACE_TYPE = "bipyridine_au111_defects_saturation_raw"
RESULTS_DIR = f"results_{SURFACE_TYPE}"
BIPYRIDINE_SMILES = "n1ccccc1-c2ccccn2"

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)


def main():
    setup_directories([SURFACE_TYPE])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Create Au(111) 3×3 slab
    slab = create_slab_from_bulk(
        bulk_id="mp-81",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        results_dir=RESULTS_DIR,
    )
    logger.info("Base Au(111) slab atoms: %d", len(slab.atoms))

    # Configuration with benchmark-oriented high sampling
    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=20,
        num_placements=1000,
        bo_enabled=False,
        device="cuda",
        fmax=0.05,
        stage1_steps=80,
        stage2_steps=500,
        auto_resize_slab=True,
        min_pbc_image_separation=10.0,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        debug_write_initial_placements=True,
    )

    # Initialize calculator and deposit Au adatoms
    logger.info("Initializing model...")
    calculator, ts_model = setup_single_model(config.model_name, config.device)

    defect_slab = deposit_adatoms(
        slab,
        adatom_symbol="Au",
        coverage_fraction=0.20,
        calculator=calculator,
        config=config,
        results_dir=RESULTS_DIR,
        seed=42,
    )
    logger.info("Defected Au(111) slab atoms: %d", len(defect_slab.atoms))

    # Run saturation
    failure_summary = {}
    saturation_results = run_saturation(
        slab=defect_slab,
        molecules=[(BIPYRIDINE_SMILES, "bipyridine")],
        config=config,
        surface_type=SURFACE_TYPE,
        skip_existing=False,
        failure_summary_out=failure_summary,
    )

    if not saturation_results:
        logger.error("No saturation results produced.")
        if failure_summary:
            logger.error(format_failure_summary(failure_summary))
        return 1

    save_saturation_results(
        saturation_results, surface_type=SURFACE_TYPE, config=config
    )

    # Flatten per-step results into benchmark CSV format
    flattened_runs = []
    for saturation_run in saturation_results:
        for step_result in saturation_run.steps:
            step_name = f"{saturation_run.molecule}_step_{step_result.step:03d}"
            step_results = step_result.all_results
            if step_results:
                flattened_runs.append(
                    ScreeningRunResult(
                        molecule=step_name,
                        results=step_results,
                        summary=build_molecule_summary(step_name, step_results),
                    )
                )

    if flattened_runs:
        save_summary_results(flattened_runs, surface_type=SURFACE_TYPE, config=config)
        logger.info(
            "Wrote benchmark dataset: %s/adsorption_energies_detailed.csv",
            RESULTS_DIR,
        )

    # Print summary
    first = saturation_results[0]
    print(
        format_saturation_complete(
            label="Bipyridine saturation on defected Au(111)",
            n_molecules_at_saturation=first.n_molecules_at_saturation,
            total_steps=len(first.steps),
            results_dir=RESULTS_DIR,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

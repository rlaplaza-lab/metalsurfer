#!/usr/bin/env python3
"""Generate high-placement, non-BO saturation data for bipyridine on defected Au(111).

Purpose:
- Create an irregular Au(111) surface by depositing extra Au adatoms (defects).
- Run saturation screening with BO disabled so each step is sampled without BO bias.
- Persist standard saturation outputs and also a flattened
  ``adsorption_energies_detailed.csv`` suitable for BO benchmark scripts.

Example:
  python scripts/bipyridine_au111_defects_saturation_raw.py --device cuda
"""

import argparse
import logging
import os
import tempfile

from metalsurfer import (
    AdsorptionConfig,
    format_failure_summary,
    run_saturation,
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bipyridine saturation on defected Au(111) with high, non-BO placement sampling."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        choices=("cuda", "cpu"),
        help="Compute device.",
    )
    parser.add_argument(
        "--num-placements",
        type=int,
        default=1000,
        help="Placements per saturation step (benchmark-oriented high sampling).",
    )
    parser.add_argument(
        "--num-conformers",
        type=int,
        default=20,
        help="Conformers for bipyridine (higher for broader geometry coverage).",
    )
    parser.add_argument(
        "--defect-coverage",
        type=float,
        default=0.20,
        help="Fraction of hollow sites populated by extra Au adatoms.",
    )
    parser.add_argument(
        "--supercell-x",
        type=int,
        default=3,
        help="Au(111) supercell size along x.",
    )
    parser.add_argument(
        "--supercell-y",
        type=int,
        default=3,
        help="Au(111) supercell size along y.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    configure_logging(default_level="DEBUG" if args.debug else "INFO")
    logger = logging.getLogger(__name__)

    setup_directories([SURFACE_TYPE])
    os.makedirs(RESULTS_DIR, exist_ok=True)

    slab = create_slab_from_bulk(
        bulk_id="mp-81",  # fcc Au
        miller_indices=(1, 1, 1),
        supercell=(args.supercell_x, args.supercell_y, 1),
        results_dir=RESULTS_DIR,
    )
    logger.info("Base Au(111) slab atoms: %d", len(slab.atoms))

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=args.seed,
        num_conformers=args.num_conformers,
        num_placements=args.num_placements,
        bo_enabled=False,
        device=args.device,
        fmax=0.05,
        stage1_steps=80,
        stage2_steps=500,
        auto_resize_slab=True,
        min_pbc_image_separation=10.0,
        skip_topology_check=False,
        skip_desorption_check=False,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
    )

    defect_slab = deposit_adatoms(
        slab,
        adatom_symbol="Au",
        coverage_fraction=args.defect_coverage,
        calculator=None,
        config=config,
        results_dir=RESULTS_DIR,
        seed=args.seed,
    )
    logger.info(
        "Defected Au(111) slab atoms after Au adatoms: %d",
        len(defect_slab.atoms),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as handle:
        handle.write(f"{BIPYRIDINE_SMILES},bipyridine\n")
        smiles_path = handle.name

    try:
        failure_summary: dict[str, object] = {}
        saturation_results = run_saturation(
            defect_slab,
            molecules=smiles_path,
            config=config,
            surface_type=SURFACE_TYPE,
            skip_existing=False,
            failure_summary_out=failure_summary,
        )
    finally:
        os.unlink(smiles_path)

    if not saturation_results:
        logger.error("No saturation results produced.")
        if failure_summary:
            logger.error("\n%s", format_failure_summary(failure_summary))
        return 1

    save_saturation_results(
        saturation_results, surface_type=SURFACE_TYPE, config=config
    )

    # Flatten all per-step results into standard benchmark CSV format.
    # This creates adsorption_energies_detailed.csv under results_<surface_type>.
    flattened_runs: list[ScreeningRunResult] = []
    for saturation_run in saturation_results:
        for step_result in saturation_run.steps:
            step_name = f"{saturation_run.molecule}_step_{step_result.step:03d}"
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
    if flattened_runs:
        save_summary_results(flattened_runs, surface_type=SURFACE_TYPE, config=config)
        logger.info(
            "Wrote benchmark-ready dataset: %s/adsorption_energies_detailed.csv",
            RESULTS_DIR,
        )

    first = saturation_results[0]
    print(
        format_saturation_complete(
            label="Bipyridine saturation on defected Au(111)",
            n_molecules_at_saturation=first.n_molecules_at_saturation,
            total_steps=len(first.steps),
            results_dir=RESULTS_DIR,
        )
    )
    print(f"Raw benchmark dataset: {RESULTS_DIR}/adsorption_energies_detailed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

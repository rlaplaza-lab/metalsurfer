#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on Ru(0001) with 10% Sn coverage.

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Uses metalsurfer prepare_substrate to create Sn-covered Ru(0001) surface.
BO pipeline: 300 placements max in passes of 100 (100 initial random + up to 2 BO passes of 100).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import argparse
import logging
import os

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption_bo
from metalsurfer.surface_prep import prepare_substrate

# List of smiles and molecule name pairs
MOLECULES = [
    ("C(=O)C1OC(CO[H])=CC=1", "HMF"),
    ("C(O[H])C1OC(CO[H])=CC=1", "BHMF"),
    ("C(O[H])C1OC(CO[H])CC1", "BHMTHF"),
    ("C(=O)C1OC(C)=CC=1", "5-MF"),
    ("C(O[H])C1OC(C)=CC=1", "MFA"),
    ("C1(C)OC(C)=CC=1", "DMF"),
    ("C(O[H])C1OC(C)CC1", "MTHFA"),
    ("CC1OC(C)CC1", "DMTHF"),
]


def _configure_logging(debug: bool = False) -> None:
    level_name = "DEBUG" if debug else "INFO"
    configure_logging(default_level=level_name)
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(
        description="Furanic molecules on Ru(0001)+10% Sn with BO (up to 300 placements, passes of 100)"
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        help="Device: cuda or cpu",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    args = parser.parse_args()
    debug = args.debug or (
        os.environ.get("METALSURFER_DEBUG", "").lower() in ("1", "true", "yes")
    )
    _configure_logging(debug=debug)
    device = args.device if args.device in ("cuda", "cpu") else "cuda"

    results_subdir = "furanics_ru0001_sn10"
    results_dir = f"results_{results_subdir}"
    os.makedirs(results_dir, exist_ok=True)

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        device=device,
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=False,
        top_layer_tolerance=2.0,  # Include top Ru + Sn in top layer for placement
        bo_enabled=True,
        bo_initial_random=100,
        bo_batch_size=100,
        # bo_total_budget = acquisition batches after initial (not total evals).
        bo_total_budget=2,  # 100 initial + 2×100 ≈ 300 evaluations
    )

    # Create Ru(0001) slab from Materials Project mp-33.
    base_slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    # Deposit Sn atoms at 10% of hollow sites
    slab = prepare_substrate(
        slab=base_slab,
        adatom_symbol="Sn",
        adatom_coverage=0.1,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption_bo(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=results_subdir,
        system_name="Ru_0001_Sn10",
        process_kwargs={"base_slab_for_frozen": base_slab.atoms},
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (Ru(0001) + 10% Sn)",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

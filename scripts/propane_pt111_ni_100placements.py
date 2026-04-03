#!/usr/bin/env python3
"""Run propane adsorption on Pt(111) with Ni adatoms for many placements.

Produces results_propane_pt111_ni/adsorption_energies_detailed.csv for BO benchmarking.
Use conda env metalsurfer: conda run -n metalsurfer python scripts/propane_pt111_ni_100placements.py

Slab: fcc Pt mp-126 (1,1,1) → 2×2×1 supercell → ~10% Ni adatoms at hollow sites
so the surface is heterogeneous and placement matters.

If GPU runs out of memory, set METALSURFER_DEVICE=cpu or pass --device cpu.
"""

import argparse
import logging
import os

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    deposit_adatoms,
    run_adsorption,
)
from metalsurfer._logging import configure_logging


def _configure_logging(debug: bool = False) -> None:
    level_name = "DEBUG" if debug else "INFO"
    configure_logging(default_level=level_name)
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


SURFACE_TYPE = "propane_pt111_ni"
RESULTS_DIR = f"results_{SURFACE_TYPE}"
BULK_ID = "mp-126"  # fcc Pt
MILLER = (1, 1, 1)
# Note: TorchSim's autobatcher has an internal max system \"metric\" (often ~500).
# Larger slabs can exceed it (e.g. metric 510) and force a slow per-system fallback.
SUPERCELL = (2, 2, 1)
NI_ADATOM_COVERAGE = 0.10  # fraction of hollow sites with Ni adatoms
PROPANE_SMILES = "CCC"
NUM_PLACEMENTS = 250


def main():
    parser = argparse.ArgumentParser(
        description="Propane on Pt(111) + Ni adatoms. "
        "You may get fewer than num_placements: placement generation can skip attempts that fail "
        "(e.g. no valid site, overlap, or geometry), so 'Generated N/num_placements' can have N < num_placements."
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("METALSURFER_DEVICE", "cuda"),
        help="Device: cuda or cpu",
    )
    parser.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable DEBUG (default: debug is on)",
    )
    args = parser.parse_args()
    debug = (not args.no_debug) or (
        os.environ.get("METALSURFER_DEBUG", "").lower() in ("1", "true", "yes")
    )
    _configure_logging(debug=debug)
    device = args.device if args.device in ("cuda", "cpu") else "cuda"

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Build Pt(111) slab
    slab = create_slab_from_bulk(
        bulk_id=BULK_ID,
        miller_indices=MILLER,
        supercell=SUPERCELL,
        results_dir=RESULTS_DIR,
    )
    logging.info(
        "Pt(111) slab: %d atoms (from %s %s supercell %s)",
        len(slab.atoms),
        BULK_ID,
        MILLER,
        SUPERCELL,
    )

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=20,  # propane is floppier than CO2
        num_placements=NUM_PLACEMENTS,
        bo_enabled=False,
        device=device,
        fmax=0.05,
        stage1_steps=100,
        stage2_steps=350,
        max_force_convergence=0.08,
        auto_resize_slab=True,
        min_pbc_image_separation=8.0,
        skip_topology_check=True,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
    )

    # 2) Deposit Ni adatoms at hollow sites (~10% coverage)
    slab = deposit_adatoms(
        slab,
        adatom_symbol="Ni",
        coverage_fraction=NI_ADATOM_COVERAGE,
        calculator=None,
        config=config,
        results_dir=RESULTS_DIR,
        seed=config.seed,
    )
    logging.info(
        "Pt(111) + Ni adatoms slab: %d atoms",
        len(slab.atoms),
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[(PROPANE_SMILES, "propane")],
        config=config,
        surface_type=SURFACE_TYPE,
        system_name="pt111_ni_adatoms",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        logging.info(
            "Best E_ads = %.4f eV; %d results -> %s",
            summary.best_adsorption_energy,
            summary.n_valid_placements,
            RESULTS_DIR,
        )
    else:
        logging.warning("No valid placements.")

    return 0 if summary.best_adsorption_energy is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run propane adsorption on Pt(111) with Ni adatoms for many placements.

Produces results_propane_pt111_ni/adsorption_energies_detailed.csv for BO benchmarking.
Use conda env pyadsorbml: conda run -n pyadsorbml python scripts/propane_pt111_ni_100placements.py

Slab: fcc Pt mp-126 (1,1,1) → 2×2×1 supercell → ~10% Ni adatoms at hollow sites
so the surface is heterogeneous and placement matters.

If GPU runs out of memory, set METALSURFER_DEVICE=cpu or pass --device cpu.
"""

import argparse
import logging
import os

from metalsurfer import (
    AdsorptionConfig,
    calculate_reference_energies,
    create_slab_from_bulk,
    deposit_adatoms,
    format_failure_summary,
    process_molecule,
    save_single_molecule_results,
    setup_single_model,
)


def _configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
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
NUM_PLACEMENTS = 120


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

    calculator, ts_model = setup_single_model(config.model_name, config.device)

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

    # 3) Reference energies
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=["propane"],
        smiles_list=[PROPANE_SMILES],
        ts_model=ts_model,
        config=config,
    )
    e_slab = ref.slab_energy
    e_propane = ref.get_molecule_energy("propane")
    if e_propane is None:
        raise RuntimeError("Missing propane reference energy")
    logging.info("E_slab=%.4f eV, E_propane=%.4f eV", e_slab, e_propane)

    # 4) Run placements (non-Bayesian)
    failure_summary: dict[str, object] = {}
    results = process_molecule(
        PROPANE_SMILES,
        "propane",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type=SURFACE_TYPE,
        failure_summary_out=failure_summary,
    )

    if results:
        save_single_molecule_results(
            "propane",
            results,
            surface_type=SURFACE_TYPE,
            system_name="pt111_ni_adatoms",
            config=config,
        )
        best = min(results, key=lambda r: r.energy_adsorption)
        logging.info(
            "Best E_ads = %.4f eV; %d results -> %s",
            best.energy_adsorption,
            len(results),
            RESULTS_DIR,
        )
    else:
        logging.warning("No valid placements.")
        if failure_summary:
            logging.info(format_failure_summary(failure_summary))

    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on Ru(0001) with 10% Sn coverage.

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Uses metalsurfer deposit_adatoms to create Sn-covered Ru(0001) surface.
BO pipeline: 100 placements in batches of 20 (20 initial random + 4 batches of 20).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import argparse
import logging
import os

from metalsurfer import (
    AdsorptionConfig,
    calculate_reference_energies,
    create_slab_from_bulk,
    deposit_adatoms,
    process_molecule_bayesian,
    save_single_molecule_results,
    save_summary_results,
    screening_run_result,
    setup_single_model,
    write_run_settings,
)
from metalsurfer.placement import classify_adsorbate_orientation

# (SMILES, molecule_name)
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
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    if debug:
        logging.getLogger("metalsurfer.filters").setLevel(logging.DEBUG)
        logging.getLogger("metalsurfer.workflow").setLevel(logging.DEBUG)


def main():
    parser = argparse.ArgumentParser(
        description="Furanic molecules on Ru(0001)+10% Sn with BO (100 placements, batches of 20)"
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

    # Create Ru(0001) slab from Materials Project mp-33.
    base_slab = create_slab_from_bulk(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        results_dir=results_dir,
    )

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=100,
        autobatcher_max_memory_padding=0.5,
        device=device,
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=False,
        placement_mode="auto",
        top_layer_tolerance=2.0,  # Include top Ru + Sn in top layer for placement
        bo_enabled=True,
        bo_initial_random=20,
        bo_batch_size=20,
        bo_total_budget=100,
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    # Deposit Sn atoms at 10% of hollow sites
    slab = deposit_adatoms(
        base_slab,
        adatom_symbol="Sn",
        coverage_fraction=0.10,
        calculator=calculator,
        results_dir=results_dir,
        config=config,
    )

    smiles_list = [s for s, _ in MOLECULES]
    molecule_names = [n for _, n in MOLECULES]

    # Reference energies: Sn-covered slab + all isolated molecules
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=molecule_names,
        smiles_list=smiles_list,
        ts_model=ts_model,
        config=config,
    )

    e_slab = ref.slab_energy
    print(f"E_slab (Ru+10% Sn) = {e_slab:.4f} eV")
    for name in molecule_names:
        e_mol = ref.get_molecule_energy(name)
        if e_mol is not None:
            print(f"  E_{name} = {e_mol:.4f} eV")
        else:
            print(f"  E_{name} = (failed)")

    # Process each molecule (BO pipeline)
    summary = []
    all_run_results = []

    for smiles, name in MOLECULES:
        print(f"\n--- Processing {name} ---")
        results = process_molecule_bayesian(
            smiles,
            name,
            slab,
            calculator,
            ref,
            ts_model=ts_model,
            config=config,
            surface_type=results_subdir,
            base_slab_for_frozen=base_slab.atoms,
        )

        if results:
            save_single_molecule_results(
                name,
                results,
                surface_type=results_subdir,
                system_name="Ru_0001_Sn10",
                config=config,
                write_csv=False,
            )
            all_run_results.append(screening_run_result(name, results))
            best = min(results, key=lambda r: r.energy_adsorption)
            surface_symbols = set(slab.atoms.get_chemical_symbols())
            slab_size = next(
                (
                    i
                    for i, s in enumerate(results[0].atoms.get_chemical_symbols())
                    if s not in surface_symbols
                ),
                None,
            )
            orientations = []
            if slab_size is not None:
                orientations = [
                    classify_adsorbate_orientation(r.atoms, slab_size) for r in results
                ]
            n_parallel = sum(1 for o in orientations if o == "parallel")
            print(
                f"Binding energy of {name} on Ru(0001)+10% Sn: "
                f"{best.energy_adsorption:.4f} eV"
            )
            if orientations:
                print(
                    f"  Orientations: {n_parallel}/{len(results)} parallel, "
                    f"{len(results) - n_parallel} EN-down"
                )
            summary.append((name, best.energy_adsorption, len(results)))
        else:
            print(f"No valid placements found for {name}.")
            summary.append((name, None, 0))

    if all_run_results:
        save_summary_results(
            all_run_results,
            surface_type=results_subdir,
            config=config,
        )
        write_run_settings(
            results_subdir,
            config,
            campaign="multi_molecule_binding",
            n_molecules=len(all_run_results),
            molecules=[rr.molecule for rr in all_run_results],
            n_configurations=sum(len(rr.results) for rr in all_run_results),
        )

    # Final summary
    print("\n" + "=" * 60)
    print("Binding energy summary (Ru(0001) + 10% Sn)")
    print("=" * 60)
    print("(E_ads = E(slab+molecule) - E(slab) - E(molecule); negative = favorable)")
    print()
    for name, e_ads, n_results in summary:
        if e_ads is not None:
            print(f"  {name:12s}: {e_ads:+.4f} eV  ({n_results} valid placements)")
        else:
            print(f"  {name:12s}: (no valid placements)")
    print()
    print(f"Results saved to {results_dir}/ (XYZ, POSCAR, CSV)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

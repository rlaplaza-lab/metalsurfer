#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on Ru(0001) with 25% Sn coverage.

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Uses metalsurfer deposit_adatoms to create Sn-covered Ru(0001) surface.
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import logging

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
from metalsurfer.placement import classify_adsorbate_orientation

logging.basicConfig(level=logging.DEBUG)

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


def main():
    results_subdir = "furanics_ru0001_sn25"
    results_dir = f"results_{results_subdir}"

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
        num_placements=25,
        autobatcher_max_memory_padding=0.5,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=True,
        placement_mode="auto",
        top_layer_tolerance=2.0,  # Include top Ru + Sn in top layer for placement
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    # Deposit Sn atoms at 25% of hollow sites
    slab = deposit_adatoms(
        base_slab,
        adatom_symbol="Sn",
        coverage_fraction=0.25,
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
    print(f"E_slab (Ru+25% Sn) = {e_slab:.4f} eV")
    for name in molecule_names:
        e_mol = ref.get_molecule_energy(name)
        if e_mol is not None:
            print(f"  E_{name} = {e_mol:.4f} eV")
        else:
            print(f"  E_{name} = (failed)")

    # Process each molecule
    all_failures = {}
    summary = []

    for smiles, name in MOLECULES:
        print(f"\n--- Processing {name} ---")
        failure_summary = {}
        results = process_molecule(
            smiles,
            name,
            slab,
            calculator,
            ref,
            ts_model=ts_model,
            config=config,
            surface_type=results_subdir,
            base_slab_for_frozen=base_slab.atoms,
            failure_summary_out=failure_summary,
        )

        if failure_summary:
            all_failures[name] = failure_summary

        if results:
            save_single_molecule_results(
                name,
                results,
                surface_type=results_subdir,
                system_name="Ru_0001_Sn25",
                config=config,
            )
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
                f"Binding energy of {name} on Ru(0001)+25% Sn: "
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

    # Final summary
    print("\n" + "=" * 60)
    print("Binding energy summary (Ru(0001) + 25% Sn)")
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

    if all_failures:
        print()
        print(format_failure_summary(all_failures))


if __name__ == "__main__":
    main()

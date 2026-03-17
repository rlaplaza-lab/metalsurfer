#!/usr/bin/env python3
"""Compute binding energies of ethane, ethene, and acetylene on RuCu2 alloy (Ru host, 2/3 Cu) from mp-33.

Molecules: ethane (CC), ethene (C=C), acetylene (C#C).

Uses metalsurfer substitute_alloy to create RuCu2 alloy from Ru(0001) base slab.
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import logging

from metalsurfer import (
    AdsorptionConfig,
    calculate_reference_energies,
    create_slab_from_bulk,
    format_failure_summary,
    process_molecule,
    save_single_molecule_results,
    setup_single_model,
    substitute_alloy,
)
from metalsurfer.placement import classify_adsorbate_orientation

logging.basicConfig(level=logging.DEBUG)

# (SMILES, molecule_name)
MOLECULES = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]


def main():
    results_subdir = "ethane_ethene_acetylene_rucu2"
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
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    # Substitute 2/3 of Ru with Cu to form RuCu2 alloy
    slab = substitute_alloy(
        base_slab,
        host_symbol="Ru",
        guest_symbol="Cu",
        guest_fraction=2.0 / 3.0,
        calculator=calculator,
        config=config,
        results_dir=results_dir,
    )

    smiles_list = [s for s, _ in MOLECULES]
    molecule_names = [n for _, n in MOLECULES]

    # Reference energies: clean slab + all isolated molecules
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=molecule_names,
        smiles_list=smiles_list,
        ts_model=ts_model,
        config=config,
    )

    e_slab = ref.slab_energy
    print(f"E_slab = {e_slab:.4f} eV")
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
            failure_summary_out=failure_summary,
        )

        if failure_summary:
            all_failures[name] = failure_summary

        if results:
            save_single_molecule_results(
                name,
                results,
                surface_type=results_subdir,
                system_name="RuCu2_0001",
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
                f"Binding energy of {name} on RuCu2(0001): {best.energy_adsorption:.4f} eV"
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
    print("Binding energy summary (RuCu2(0001))")
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

#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of vanillin on Ni(111) from mp-23 using metalsurfer.

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
)
from metalsurfer.placement import classify_adsorbate_orientation

logging.basicConfig(level=logging.DEBUG)


def main():
    # Single subdir for slab, placements, and results (avoids path drift)
    results_subdir = "vanillin_ni111"
    results_dir = f"results_{results_subdir}"

    # Create Ni(111) slab from Materials Project mp-23.
    # Use minimal (1,1,1) supercell; auto_resize_slab will expand if needed for PBC separation.
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir=results_dir,
    )

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=25,
        autobatcher_max_memory_padding=0.5,  # 0.9 was too aggressive
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=True,
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    smiles = "c1(C=O)cc(OC)c(O)cc1"

    # Reference energies: clean slab + isolated vanillin
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=["vanillin"],
        smiles_list=[smiles],
        ts_model=ts_model,
        config=config,
    )

    # Diagnostic: verify reference energies
    e_slab = ref.slab_energy
    e_vanillin = ref.get_molecule_energy("vanillin")
    print(f"E_slab={e_slab:.4f} eV, E_vanillin={e_vanillin:.4f} eV")

    # Run placement, optimization, and validation
    failure_summary = {}
    results = process_molecule(
        smiles,
        "vanillin",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type=results_subdir,
        failure_summary_out=failure_summary,
    )

    if results:
        save_single_molecule_results(
            "vanillin",
            results,
            surface_type=results_subdir,
            system_name="Ni_111",
            config=config,
        )
        best = min(results, key=lambda r: r.energy_adsorption)
        # Infer slab_size from result: process_molecule may resize the slab internally,
        # so len(slab.atoms) can be wrong. Use first non-surface atom in result.
        surface_symbols = set(slab.atoms.get_chemical_symbols())
        slab_size = next(
            (
                i
                for i, s in enumerate(results[0].atoms.get_chemical_symbols())
                if s not in surface_symbols
            ),
            None,
        )
        if slab_size is None:
            raise ValueError("Could not find adsorbate atoms in result structure")
        orientations = [
            classify_adsorbate_orientation(r.atoms, slab_size) for r in results
        ]
        n_parallel = sum(1 for o in orientations if o == "parallel")
        print(
            f"\nBinding energy of vanillin on Ni(111): {best.energy_adsorption:.4f} eV"
        )
        print(
            "  (E_ads = E(slab+vanillin) - E(slab) - E(vanillin); negative = favorable)"
        )
        print(
            f"  Orientations: {n_parallel}/{len(results)} parallel, "
            f"{len(results) - n_parallel} EN-down"
        )
        print(f"  Results saved to {results_dir}/ (XYZ, POSCAR, CSV)")
    else:
        print("No valid placements found.")
        if failure_summary:
            print()
            print(format_failure_summary(failure_summary))

    if config.debug_write_initial_placements:
        print(
            f"\nInitial placements (pre-optimization): "
            f"{results_dir}/xyz_structures/vanillin_all/initial_*.xyz"
        )


if __name__ == "__main__":
    main()

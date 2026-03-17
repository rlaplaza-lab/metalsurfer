#!/usr/bin/env python3
"""Compute binding energy of vanillin on H-saturated Ni(111).

Loads the saturated slab from H2 saturation run (e.g. scripts/h2_ni111_saturation.py)
and runs vanillin adsorption using envelope placement for the non-planar H-covered surface.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Prerequisites:
  - Run scripts/h2_ni111_saturation.py first to generate the saturated slab
  - Or provide path to a pre-adsorbed slab XYZ file
"""

import logging
import os

from ase.io import read

from metalsurfer import (
    AdsorptionConfig,
    calculate_reference_energies,
    create_slab_from_atoms,
    create_slab_from_bulk,
    format_failure_summary,
    process_molecule,
    save_single_molecule_results,
    setup_single_model,
)
from metalsurfer.placement import classify_adsorbate_orientation

logging.basicConfig(level=logging.DEBUG)


def main():
    results_subdir = "vanillin_h_saturated_ni111"
    results_dir = f"results_{results_subdir}"
    saturation_dir = "results_h2_ni111_saturation"
    saturated_xyz = (
        f"{saturation_dir}/xyz_structures/H2_saturation/final_saturated_slab.xyz"
    )

    if not os.path.exists(saturated_xyz):
        print(
            f"Saturated slab not found at {saturated_xyz}. "
            "Run scripts/h2_ni111_saturation.py first."
        )
        return 1

    # Load H-saturated slab
    saturated_atoms = read(saturated_xyz)
    saturated_atoms.set_pbc([True, True, True])
    slab = create_slab_from_atoms(saturated_atoms)

    # Clean metal slab for frozen indices (subsurface metal only)
    base_slab = create_slab_from_bulk(
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
        autobatcher_max_memory_padding=0.5,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        placement_mode="auto",  # Uses envelope for non-planar H-covered surface
        top_layer_tolerance=2.0,  # Include top metal + H in top layer for envelope
    )

    calculator, ts_model = setup_single_model(config.model_name, config.device)

    smiles = "c1(C=O)cc(OC)c(O)cc1"

    # Reference energies: saturated slab + isolated vanillin
    ref = calculate_reference_energies(
        slab,
        calculator,
        molecules=["vanillin"],
        smiles_list=[smiles],
        ts_model=ts_model,
        config=config,
    )
    e_saturated = ref.slab_energy
    e_vanillin = ref.get_molecule_energy("vanillin")
    print(f"E(saturated slab) = {e_saturated:.4f} eV, E_vanillin = {e_vanillin:.4f} eV")

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
        base_slab_for_frozen=base_slab.atoms,
        failure_summary_out=failure_summary,
    )

    if results:
        save_single_molecule_results(
            "vanillin",
            results,
            surface_type=results_subdir,
            system_name="Ni_111_H_saturated",
            config=config,
        )
        best = min(results, key=lambda r: r.energy_adsorption)
        surface_symbols = set(base_slab.atoms.get_chemical_symbols())
        slab_size = next(
            (
                i
                for i, s in enumerate(results[0].atoms.get_chemical_symbols())
                if s not in surface_symbols
            ),
            None,
        )
        if slab_size is not None:
            orientations = [
                classify_adsorbate_orientation(r.atoms, slab_size) for r in results
            ]
            n_parallel = sum(1 for o in orientations if o == "parallel")
            print(
                f"\nBinding energy of vanillin on H-saturated Ni(111): "
                f"{best.energy_adsorption:.4f} eV"
            )
            print("  (E_ads = E(slab+vanillin) - E(saturated slab) - E(vanillin))")
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
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

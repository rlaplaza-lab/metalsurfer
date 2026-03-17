#!/usr/bin/env python3
"""Systematically adsorb H2 on Ni(111) until saturation using metalsurfer.

Adds H2 molecules one at a time to the slab; stops when best E_ads >= 0 (slab saturated).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import os
import tempfile

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    format_failure_summary,
    run_saturation_screening,
)
from metalsurfer.io_results import save_saturation_results, setup_directories


def main():
    # Create Ni(111) slab from Materials Project mp-23.
    surface_type = "h2_ni111_saturation"
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir=f"results_{surface_type}",
    )

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=1,  # H2 has only one geometry
        num_placements=50,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,  # Skip memory estimation (slab+H2 ~386 atoms)
        autobatcher_max_atoms_to_try=5000,  # Cap estimation probes if scaler unused
        device="cuda",  # use "cpu" if no GPU
        skip_topology_check=True,  # Allow H2 → 2H (bond breaking)
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    # Create temporary SMILES file with H2 only
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n")
        smiles_path = f.name

    try:
        setup_directories([surface_type])
        failure_summary = {}
        saturation_results = run_saturation_screening(
            slab,
            smiles_file=smiles_path,
            config=config,
            surface_type=surface_type,
            skip_existing=False,
            failure_summary_out=failure_summary,
        )

        if saturation_results:
            save_saturation_results(
                saturation_results,
                surface_type=surface_type,
                config=config,
            )
            sr = saturation_results[0]
            total_steps = len(sr.steps)
            n_at_sat = sr.n_molecules_at_saturation
            print("\nH2 saturation on Ni(111) complete:")
            print(f"  Molecules at saturation: {n_at_sat}")
            print(f"  Total steps: {total_steps}")
            print(f"  Results saved to results_{surface_type}/ (XYZ, POSCAR, CSV)")
        else:
            print("No saturation results (no valid placements found).")
            if failure_summary:
                print()
                print(format_failure_summary(failure_summary))
    finally:
        os.unlink(smiles_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Systematically adsorb vanillin on Ni(111) until saturation using metalsurfer.

Adds vanillin molecules one at a time to the slab; stops when best E_ads >= 0 (slab saturated).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

import os
import tempfile

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    format_failure_summary,
    run_saturation,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_saturation_complete
from metalsurfer.io_results import save_saturation_results, setup_directories


def main():
    configure_logging(default_level="INFO")
    # Create Ni(111) slab from Materials Project mp-23.
    surface_type = "vanillin_ni111_saturation"
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir=f"results_{surface_type}",
    )

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        device="cuda",  # use "cpu" if no GPU
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    # Create temporary SMILES file with vanillin only
    smiles = "c1(C=O)cc(OC)c(O)cc1"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(f"{smiles},vanillin\n")
        smiles_path = f.name

    try:
        setup_directories([surface_type])
        failure_summary = {}
        saturation_results = run_saturation(
            slab,
            molecules=smiles_path,
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
            print("")
            print(
                format_saturation_complete(
                    label="Vanillin saturation on Ni(111)",
                    n_molecules_at_saturation=n_at_sat,
                    total_steps=total_steps,
                    results_dir=f"results_{surface_type}",
                )
            )
        else:
            print("No saturation results (no valid placements found).")
            if failure_summary:
                print()
                print(format_failure_summary(failure_summary))
    finally:
        os.unlink(smiles_path)


if __name__ == "__main__":
    main()

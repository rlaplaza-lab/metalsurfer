#!/usr/bin/env python3
"""Systematically adsorb vanillin on Ni(111) until saturation using metalsurfer.

Adds vanillin molecules one at a time to the slab; stops when best E_ads >= 0 (slab saturated).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
On-disk output follows ``AdsorptionConfig`` (README, saturation section).
"""

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    prepare_slab,
    run_saturation,
)


def main():
    configure_logging(default_level="INFO")
    surface_type = "vanillin_ni111_saturation"
    results_dir = f"results_{surface_type}"

    # Create Ni(111) slab from Materials Project mp-23.
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

    slab = prepare_slab(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    smiles = "c1(C=O)cc(OC)c(O)cc1"
    campaign = run_saturation(
        slab=slab,
        molecules=[(smiles, "vanillin")],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="Vanillin saturation on Ni(111)",
                results_dir=results_dir,
            )
        )
    else:
        print("No saturation results (no valid placements found).")
        if campaign.failure_summary:
            print()
            print(campaign.format_failure_summary())


if __name__ == "__main__":
    main()

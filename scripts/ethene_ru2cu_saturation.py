#!/usr/bin/env python3
"""Ethene saturation on Ru2Cu alloy (Ru host, 1/3 Cu) from mp-33 using metalsurfer.

Adds ethene molecules one at a time until best E_ads >= 0 (slab saturated).
Uses same surface creation pipeline as ethane_ethene_acetylene_ru2cu_binding_energy.py (seed=42).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
On-disk output follows ``AdsorptionConfig`` (README, saturation section).
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "ethene_ru2cu_saturation"
    results_dir = f"results_{surface_type}"

    # Same surface creation as ethane_ethene_acetylene_ru2cu_binding_energy.py (seed=42)
    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        alloy_host="Ru",
        alloy_guest="Cu",
        alloy_fraction=1.0 / 3.0,
        enforce_top_layer_fraction=True,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_saturation(
        slab=slab,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="Ethene saturation on Ru2Cu(0001)",
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

#!/usr/bin/env python3
"""H2 saturation on RuCu alloy (Ru host, 1/2 Cu) from mp-33 using metalsurfer.

Adds H2 molecules one at a time until best E_ads >= 0 (slab saturated).
Uses same surface creation pipeline as ethane_ethene_acetylene_rucu_binding_energy.py (seed=42).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
On-disk output follows ``AdsorptionConfig`` (README, saturation section).
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "h2_rucu_saturation"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=1,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=True,
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
        alloy_fraction=0.5,
        enforce_top_layer_fraction=True,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_saturation(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="H2 saturation on RuCu(0001)",
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

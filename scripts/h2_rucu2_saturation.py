#!/usr/bin/env python3
"""H2 saturation on RuCu2 alloy (Ru host, 2/3 Cu) from mp-33 using metalsurfer.

Adds H2 molecules one at a time until best E_ads >= 0 (slab saturated).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "h2_rucu2_saturation"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        num_conformers=1,
        num_placements=250,
        enable_dissociative_placement=True,
        skip_topology_check=True,
    )

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        alloy_host="Ru",
        alloy_guest="Cu",
        alloy_fraction=2.0 / 3.0,
        enforce_top_layer_fraction=True,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_saturation(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="H2 saturation on RuCu2(0001)",
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

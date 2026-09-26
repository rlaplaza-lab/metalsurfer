#!/usr/bin/env python3
"""Ethene saturation on RuCu alloy (Ru host, 1/2 Cu) from mp-33 using metalsurfer.

Adds ethene molecules one at a time until best E_ads >= 0 (slab saturated).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "ethene_rucu_saturation"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        num_placements=250,
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
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="Ethene saturation on RuCu(0001)",
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

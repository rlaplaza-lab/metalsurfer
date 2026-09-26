#!/usr/bin/env python3
"""H2 saturation on Cu(111) from mp-30.

Adds H2 molecules one at a time until best E_ads >= 0 (slab saturated).
``enable_dissociative_placement`` + ``skip_topology_check`` allow H₂ → 2H.

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "h2_cu_saturation"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=250,
        enable_dissociative_placement=True,
        skip_topology_check=True,
    )

    slab = prepare_substrate(
        bulk_id="mp-30",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
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
                label="H2 saturation on Cu(111)",
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

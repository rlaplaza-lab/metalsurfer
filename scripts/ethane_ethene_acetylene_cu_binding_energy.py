#!/usr/bin/env python3
"""Compute binding energies of ethane, ethene, and acetylene on Cu(111) from mp-30 using metalsurfer.

Molecules: ethane (CC), ethene (C=C), acetylene (C#C).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

MOLECULES = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]


def main() -> int:
    configure_logging(default_level="INFO")
    surface_type = "ethane_ethene_acetylene_cu"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        num_placements=250,
    )

    slab = prepare_substrate(
        bulk_id="mp-30",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=surface_type,
        system_name="Cu_111",
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethane / ethene / acetylene on Cu(111))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

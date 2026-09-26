#!/usr/bin/env python3
"""Compute binding energies of ethane, ethene, and acetylene on Ru(0001) from mp-33 using metalsurfer.

Molecules: ethane (CC), ethene (C=C), acetylene (C#C).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

# List of smiles and molecule name pairs
MOLECULES = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]


def main() -> int:
    configure_logging(default_level="INFO")
    results_subdir = "ethane_ethene_acetylene_ru"
    results_dir = f"results_{results_subdir}"

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        num_placements=250,
    )

    # Create Ru(0001) slab from Materials Project mp-33.
    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=results_subdir,
        system_name="Ru_0001",
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethane / ethene / acetylene on Ru(0001))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

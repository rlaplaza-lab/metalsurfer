#!/usr/bin/env python3
"""Binding energy of H2 on Ni(111) from mp-23.

``enable_dissociative_placement=True`` enables hollow-site pair placements;
``skip_topology_check=True`` keeps fragmented post-relax adsorbates.
Pins ``uma-m-1p1`` / ``oc20`` (not the library default).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate


def main() -> int:
    configure_logging(default_level="INFO")
    surface_type = "h2_ni111"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        model_name="uma-m-1p1",
        task_name="oc20",
        num_conformers=1,
        num_placements=250,
        enable_dissociative_placement=True,
        skip_topology_check=True,
    )

    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Ni_111",
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (H2 / Ni(111))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

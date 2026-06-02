#!/usr/bin/env python3
"""Compute binding energies of ethane/ethene/acetylene on Cu(111)."""

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    prepare_slab,
    run_adsorption,
)

MOLECULES = [("CC", "ethane"), ("C=C", "ethene"), ("C#C", "acetylene")]


def main() -> None:
    configure_logging(default_level="INFO")
    surface_type = "ethane_ethene_acetylene_cu"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        debug_write_initial_placements=True,
    )

    slab = prepare_slab(
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
    print(campaign.format_screening_complete())


if __name__ == "__main__":
    main()

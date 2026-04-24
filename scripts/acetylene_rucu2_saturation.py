#!/usr/bin/env python3
"""Acetylene saturation on RuCu2 alloy (Ru host, 2/3 Cu) from mp-33 using metalsurfer.

Adds acetylene molecules one at a time until best E_ads >= 0 (slab saturated).
Uses same surface creation pipeline as ethane_ethene_acetylene_rucu2_binding_energy.py (seed=42).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    format_failure_summary,
    run_saturation,
    setup_single_model,
    substitute_alloy,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_saturation_complete
from metalsurfer.io_results import save_saturation_results, setup_directories


def main():
    configure_logging(default_level="INFO")
    surface_type = "acetylene_rucu2_saturation"
    results_dir = f"results_{surface_type}"

    # Same surface creation as ethane_ethene_acetylene_rucu2_binding_energy.py (seed=42)
    base_slab = create_slab_from_bulk(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        results_dir=results_dir,
    )

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

    calculator, _ = setup_single_model(config.model_name, config.device)
    slab = substitute_alloy(
        base_slab,
        host_symbol="Ru",
        guest_symbol="Cu",
        guest_fraction=2.0 / 3.0,
        calculator=calculator,
        seed=42,
        enforce_top_layer_fraction=True,
        config=config,
        results_dir=results_dir,
    )

    setup_directories([surface_type])
    failure_summary = {}
    saturation_results = run_saturation(
        slab=slab,
        molecules=[("C#C", "acetylene")],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
        failure_summary_out=failure_summary,
    )

    if saturation_results:
        save_saturation_results(
            saturation_results,
            surface_type=surface_type,
            config=config,
        )
        sr = saturation_results[0]
        total_steps = len(sr.steps)
        n_at_sat = sr.n_molecules_at_saturation
        print("")
        print(
            format_saturation_complete(
                label="Acetylene saturation on RuCu2(0001)",
                n_molecules_at_saturation=n_at_sat,
                total_steps=total_steps,
                results_dir=f"results_{surface_type}",
            )
        )
    else:
        print("No saturation results (no valid placements found).")
        if failure_summary:
            print()
            print(format_failure_summary(failure_summary))


if __name__ == "__main__":
    main()

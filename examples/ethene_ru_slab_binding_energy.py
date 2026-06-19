#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on Ru(0001) slab.

This example creates a Ru(0001) slab and computes ethene adsorption energy using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Uses modest settings for quick demonstration (similar to test suite).
"""

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "ethene_ru_slab"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=3,
        num_placements=5,
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
        supercell=(2, 2, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
        system_name="Ru_0001",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethene / Ru(0001))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

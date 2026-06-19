#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of vanillin on Ni(111) from mp-23 using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate


def main() -> int:
    configure_logging(default_level="INFO")
    # Single subdir for slab, placements, and results (avoids path drift)
    surface_type = "vanillin_ni111"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
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

    # Create Ni(111) slab from Materials Project mp-23 (3×3 in-plane for PBC separation).
    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
    )

    smiles = "c1(C=O)cc(OC)c(O)cc1"
    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, "vanillin")],
        config=config,
        surface_type=surface_type,
        system_name="Ni_111",
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (vanillin / Ni(111))",
            results_dir=results_dir,
        )
    )

    if config.debug_write_initial_placements:
        print(
            f"\nInitial placements (pre-optimization): "
            f"{results_dir}/xyz_structures/vanillin_all/initial_*.xyz"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

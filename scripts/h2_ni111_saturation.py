#!/usr/bin/env python3
"""Systematically adsorb H2 on Ni(111) until saturation using metalsurfer.

Adds H2 molecules one at a time to the slab; stops when best E_ads >= 0 (slab saturated).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
On-disk output follows ``AdsorptionConfig`` (README, saturation section).

If CUDA OOM on 16GB GPUs: ensure no other GPU processes (nvidia-smi), reduce
explicit `num_placements`, or use `autobatcher_max_memory_scaler=400` for ~1 system per batch.
Omit `num_placements` to autotune to GPU parallel capacity instead.
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_saturation
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    surface_type = "h2_ni111_saturation"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=1,  # H2 has only one geometry
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=True,  # Allow H2 → 2H (bond breaking)
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_saturation(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
    )

    print()
    if campaign.runs:
        print(
            campaign.format_completion(
                label="H2 saturation on Ni(111)",
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

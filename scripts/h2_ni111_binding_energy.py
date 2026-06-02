#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of H2 on Ni(111) from mp-23 using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/h2_ni111_binding_energy.py
or reduce num_placements (e.g. 25).
"""

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    prepare_slab,
    run_adsorption,
)


def main():
    configure_logging(default_level="INFO")
    surface_type = "h2_ni111"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=1,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    slab = prepare_slab(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
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
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        total_steps = config.stage1_steps + config.stage2_steps
        print(
            f"\nBinding energy of H2 on Ni(111): {summary.best_adsorption_energy:.4f} eV"
        )
        print("  (E_ads = E(slab+H2) - E(slab) - E(H2); negative = favorable)")
        print(
            f"  Relaxation: {total_steps} steps (stage1: {config.stage1_steps}, stage2: {config.stage2_steps})"
        )
        print(f"  {campaign.format_results_saved_line(results_dir=results_dir)}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()

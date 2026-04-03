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
    create_slab_from_bulk,
    run_adsorption,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_results_saved_line


def main():
    configure_logging(default_level="INFO")
    # Create Ni(111) slab from Materials Project mp-23.
    # Use minimal (1,1,1) supercell; auto_resize_slab will expand if needed for PBC separation.
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir="results_h2_ni111",
    )

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=1,  # H2 has only one geometry
        num_placements=250,
        # Conservative autobatcher settings to avoid CUDA OOM on 15GB GPUs:
        autobatcher_max_memory_padding=0.8,  # 0.9 was too aggressive
        autobatcher_max_memory_scaler=500,  # Skip memory estimation (slab+H2 ~386 atoms)
        autobatcher_max_atoms_to_try=5000,  # Cap estimation probes if scaler unused
        device="cuda",  # use "cpu" if no GPU (torch-sim-atomistic requires GPU for batching)
        skip_topology_check=True,  # Allow H2 → 2H (bond breaking)
        skip_desorption_check=False,  # Keep distance check unless you need to relax it
        stage1_steps=50,
        stage2_steps=500,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type="h2_ni111",
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
        print(f"  {format_results_saved_line('results_h2_ni111')}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()

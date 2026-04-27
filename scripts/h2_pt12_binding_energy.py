#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of H2 on a small Pt nanocluster (12 atoms).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/h2_pt12_binding_energy.py
or reduce num_placements (e.g. 25).
"""

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_atoms,
    run_adsorption,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_results_saved_line


def main():
    configure_logging(default_level="INFO")
    # Create a small Pt nanocluster with 12 atoms using ASE
    from ase import Atoms
    from ase.data import atomic_numbers
    
    # Create a simple but realistic Pt cluster for H2 adsorption
    # Use a small (111) facet-like structure which provides good adsorption sites
    from ase import Atoms
    
    # Create a 3-layer Pt(111)-like nanocluster (12 atoms total)
    # Layer 1 (bottom): 4 atoms
    # Layer 2 (middle): 4 atoms  
    # Layer 3 (top): 4 atoms
    pt_atoms = Atoms(
        symbols=['Pt'] * 12,
        positions=[
            # Bottom layer (z=0)
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.425, 0.0],
            [4.2, 2.425, 0.0],
            # Middle layer (z=2.0)
            [1.4, 0.808, 2.0],
            [4.2, 0.808, 2.0],
            [0.0, 2.425, 2.0],
            [2.8, 2.425, 2.0],
            # Top layer (z=4.0)
            [1.4, 1.617, 4.0],
            [4.2, 1.617, 4.0],
            [0.0, 0.808, 4.0],
            [2.8, 0.808, 4.0],
        ],
        cell=[20, 20, 20],  # Large cell to isolate cluster
        pbc=False  # No periodic boundary conditions for nanocluster
    )
    
    nanocluster = create_slab_from_atoms(pt_atoms)

    config = AdsorptionConfig(
        material_type="nanoparticle",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,  # H2 has only one geometry
        num_placements=5,  # Quick example - limit to 5 placements
        # Conservative autobatcher settings to avoid CUDA OOM on 15GB GPUs:
        autobatcher_max_memory_padding=0.8,  # 0.9 was too aggressive
        autobatcher_max_memory_scaler=500,  # Skip memory estimation (nanocluster+H2 ~14 atoms)
        autobatcher_max_atoms_to_try=5000,  # Cap estimation probes if scaler unused
        device="cuda",  # use "cpu" if no GPU (torch-sim-atomistic requires GPU for batching)
        skip_topology_check=True,  # Allow H2 → 2H (bond breaking for Pt clusters)
        skip_desorption_check=False,  # Keep distance check
        stage1_steps=50,
        stage2_steps=500,
        # For nanoclusters, we need to adjust placement parameters
        placement_z_range=(1.5, 4.0),  # Wider range for cluster adsorption
        min_initial_distance=1.5,  # Minimum distance from cluster atoms
    )

    campaign = run_adsorption(
        slab=nanocluster,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type="h2_pt12",
        system_name="Pt_12",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        total_steps = config.stage1_steps + config.stage2_steps
        print(
            f"\nBinding energy of H2 on Pt12 nanocluster: {summary.best_adsorption_energy:.4f} eV"
        )
        print("  (E_ads = E(nanocluster+H2) - E(nanocluster) - E(H2); negative = favorable)")
        print(
            f"  Relaxation: {total_steps} steps (stage1: {config.stage1_steps}, stage2: {config.stage2_steps})"
        )
        print(f"  {format_results_saved_line('results_h2_pt12')}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of CO2 in a MOF periodic cell.

This example creates a simple MOF-like structure with periodic boundary conditions
and computes CO2 adsorption energy using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/co2_mof_binding_energy.py
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
    # Create a simple MOF-like structure with periodic boundary conditions
    from ase import Atoms
    from ase.data import atomic_numbers
    
    # Create a simple MOF structure: Zn4O(BDC)3-like structure
    # This is a simplified representation with periodic boundary conditions
    mof_atoms = Atoms(
        symbols=['Zn', 'Zn', 'Zn', 'Zn', 'O', 'O', 'O', 'O', 'C', 'C', 'C', 'C', 'O', 'O', 'O', 'O'],
        positions=[
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [4.0, 4.0, 0.0],
            [2.0, 2.0, 2.0],
            [6.0, 2.0, 2.0],
            [2.0, 6.0, 2.0],
            [6.0, 6.0, 2.0],
            [1.0, 1.0, 4.0],
            [5.0, 1.0, 4.0],
            [1.0, 5.0, 4.0],
            [5.0, 5.0, 4.0],
            [1.0, 1.0, 6.0],
            [5.0, 1.0, 6.0],
            [1.0, 5.0, 6.0],
            [5.0, 5.0, 6.0],
        ],
        cell=[8.0, 8.0, 20.0],  # Periodic cell with sufficient vacuum in z-direction
        pbc=True  # Periodic boundary conditions for MOF
    )
    
    mof_slab = create_slab_from_atoms(mof_atoms)

    config = AdsorptionConfig(
        material_type="porous",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,  # CO2 has linear geometry
        num_placements=5,  # Quick example - limit to 5 placements
        # Conservative autobatcher settings to avoid CUDA OOM on 15GB GPUs:
        autobatcher_max_memory_padding=0.8,  # 0.9 was too aggressive
        autobatcher_max_memory_scaler=500,  # Skip memory estimation (MOF+CO2 ~20 atoms)
        autobatcher_max_atoms_to_try=5000,  # Cap estimation probes if scaler unused
        device="cuda",  # use "cpu" if no GPU (torch-sim-atomistic requires GPU for batching)
        skip_topology_check=False,  # Keep topology check for CO2
        skip_desorption_check=False,  # Keep distance check
        stage1_steps=50,
        stage2_steps=500,
        # For MOFs, adjust placement parameters for pore adsorption
        placement_z_range=(2.0, 6.0),  # Wider range for MOF pores
        min_initial_distance=1.8,  # Minimum distance from MOF atoms
    )

    campaign = run_adsorption(
        slab=mof_slab,
        molecules=[("O=C=O", "CO2")],
        config=config,
        surface_type="co2_mof",
        system_name="MOF_cell",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        total_steps = config.stage1_steps + config.stage2_steps
        print(
            f"\nBinding energy of CO2 in MOF: {summary.best_adsorption_energy:.4f} eV"
        )
        print("  (E_ads = E(MOF+CO2) - E(MOF) - E(CO2); negative = favorable)")
        print(
            f"  Relaxation: {total_steps} steps (stage1: {config.stage1_steps}, stage2: {config.stage2_steps})"
        )
        print(f"  {format_results_saved_line('results_co2_mof')}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()
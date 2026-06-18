#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of H2 on a small Pt nanocluster (12 atoms).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

The hand-built Pt₁₂ cluster is MLIP-relaxed during ``prepare_substrate`` (default
``slab_relaxation_mode="ionic_only"``) before adsorption screening.

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/h2_pt12_binding_energy.py
or reduce num_placements (e.g. 25).
"""

from ase import Atoms

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")

    pt_atoms = Atoms(
        symbols=["Pt"] * 12,
        positions=[
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.425, 0.0],
            [4.2, 2.425, 0.0],
            [1.4, 0.808, 2.0],
            [4.2, 0.808, 2.0],
            [0.0, 2.425, 2.0],
            [2.8, 2.425, 2.0],
            [1.4, 1.617, 4.0],
            [4.2, 1.617, 4.0],
            [0.0, 0.808, 4.0],
            [2.8, 0.808, 4.0],
        ],
        cell=[20, 20, 20],
        pbc=False,
    )

    config = AdsorptionConfig(
        material_type="nanoparticle",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,
        num_placements=5,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        placement_z_range=(1.5, 4.0),
        min_initial_distance=1.5,
    )

    nanocluster = prepare_substrate(
        slab=pt_atoms,
        config=config,
        results_dir="results_h2_pt12",
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
        print(
            "  (E_ads = E(nanocluster+H2) - E(nanocluster) - E(H2); negative = favorable)"
        )
        print(
            "  Note: UMA may dissociate H2 on Pt; inspect results_h2_pt12/xyz_structures/H2_adsorbate_only/ for H–H distances."
        )
        print(
            f"  Relaxation: {total_steps} steps (stage1: {config.stage1_steps}, stage2: {config.stage2_steps})"
        )
        print(f"  {campaign.format_results_saved_line(results_dir='results_h2_pt12')}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()

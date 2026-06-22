#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of H2 on a small Pt nanocluster (12 atoms).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

Hand-built Pt₁₂ keeps its input geometry during ``prepare_substrate``
(``slab_relaxation_mode="none"``). All Pt atoms receive ASE ``FixAtoms`` constraints
and stay fixed during placement relaxation. With ``skip_topology_check=True``, H₂ is
placed dissociatively on two cluster sites and may relax to separated H atoms; E_ads
still uses molecular E(H₂) as reference (dissociated minima may show positive E_ads).

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


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "h2_pt12"
    results_dir = f"results_{surface_type}"

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
        slab_relaxation_mode="none",
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
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=nanocluster,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Pt_12",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (H2 / Pt12 nanocluster)",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

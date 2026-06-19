#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of CO2 in a MOF periodic cell.

This example loads a real MOF structure from a CIF file and computes CO2 adsorption
energy using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/co2_mof_binding_energy.py
or reduce num_placements (e.g. 25).

Uses RUBTAK01 MOF structure from:
https://github.com/bafgreat/mofstructure/blob/main/tests/test_data/RUBTAK01.cif
"""

import os

from ase.io import read

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "co2_mof"
    results_dir = f"results_{surface_type}"
    cif_path = os.path.join(os.path.dirname(__file__), "mof_structures", "RUBTAK01.cif")

    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"MOF CIF file not found at {cif_path}. "
            "Please ensure the RUBTAK01.cif file is present in examples/mof_structures/"
        )

    mof_atoms = read(cif_path)

    print(f"Successfully loaded MOF structure from {cif_path}")
    print(f"MOF formula: {mof_atoms.get_chemical_formula()}")
    print(f"MOF has {len(mof_atoms)} atoms")
    print(f"MOF cell: {mof_atoms.cell}")

    config = AdsorptionConfig(
        material_type="porous",
        slab_relaxation_mode="none",  # keep experimental CIF framework geometry
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,
        num_placements=5,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
        placement_z_range=(2.0, 6.0),
        min_initial_distance=1.8,
    )

    mof_slab = prepare_substrate(
        slab=mof_atoms,
        config=config,
        results_dir=results_dir,
        align=False,
    )

    campaign = run_adsorption(
        slab=mof_slab,
        molecules=[("O=C=O", "CO2")],
        config=config,
        surface_type=surface_type,
        system_name="MOF_cell",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (CO2 / MOF)",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

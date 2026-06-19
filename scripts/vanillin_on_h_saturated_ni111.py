#!/usr/bin/env python3
"""Compute binding energy of vanillin on H-saturated Ni(111).

Loads the saturated slab from H2 saturation run (e.g. scripts/h2_ni111_saturation.py)
and runs vanillin adsorption using envelope placement for the non-planar H-covered surface.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Prerequisites:
  - Run scripts/h2_ni111_saturation.py first to generate the saturated slab
"""

import os

import pandas as pd
from ase.io import read

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

SATURATION_DIR = "results_h2_ni111_saturation"


def resolve_saturation_slab_path(saturation_dir: str) -> str:
    summary_path = f"{saturation_dir}/saturation_summary.csv"

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing saturation summary: {summary_path}")
    summary_df = pd.read_csv(summary_path)
    row = summary_df[summary_df["molecule"] == "H2"]
    if row.empty:
        raise ValueError("No H2 row found in saturation_summary.csv")
    final_path = str(row.iloc[0]["final_slab_path"]).strip()
    if not final_path:
        raise ValueError(
            "H2 saturation has no final slab path (likely not saturated yet)"
        )
    if not os.path.exists(final_path):
        raise FileNotFoundError(f"Final slab path does not exist: {final_path}")
    return final_path


def main():
    configure_logging(default_level="INFO")
    surface_type = "vanillin_h_saturated_ni111_final"
    results_dir = f"results_{surface_type}"
    saturation_dir = SATURATION_DIR
    saturated_xyz = resolve_saturation_slab_path(saturation_dir)

    # Load H-saturated slab and prepare for campaign APIs (PBC, constraints).
    saturated_atoms = read(saturated_xyz)

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
        top_layer_tolerance=2.0,  # Include top metal + H in top layer for envelope
    )

    slab = prepare_substrate(
        slab=saturated_atoms,
        config=config,
        results_dir=results_dir,
        align=False,
        slab_relaxation_mode="none",  # Use saturation geometry as the reference
    )

    # Freeze policy for placement relaxation comes from base_slab_for_frozen below.
    # relax_top_layer=True freezes subsurface metal only on the clean reference.
    base_slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
        relax_top_layer=True,
    )

    smiles = "c1(C=O)cc(OC)c(O)cc1"
    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, "vanillin")],
        config=config,
        surface_type=surface_type,
        system_name="Ni_111_H_saturated",
        process_kwargs={"base_slab_for_frozen": base_slab.atoms},
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (vanillin / H-saturated Ni(111))",
            results_dir=results_dir,
        )
    )
    if (
        campaign.molecule_summaries
        and campaign.molecule_summaries[0].best_adsorption_energy is None
    ):
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

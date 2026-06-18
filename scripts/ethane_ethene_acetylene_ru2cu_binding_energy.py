#!/usr/bin/env python3
"""Compute binding energies of ethane, ethene, and acetylene on Ru2Cu alloy (Ru host, 1/3 Cu) from mp-33.

Molecules: ethane (CC), ethene (C=C), acetylene (C#C).

Uses metalsurfer substitute_alloy to create Ru2Cu alloy from Ru(0001) base slab.
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate

# List of smiles and molecule name pairs
MOLECULES = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]


def main():
    configure_logging(default_level="INFO")
    results_subdir = "ethane_ethene_acetylene_ru2cu"
    results_dir = f"results_{results_subdir}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p1",
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

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        alloy_host="Ru",
        alloy_guest="Cu",
        alloy_fraction=1.0 / 3.0,
        enforce_top_layer_fraction=True,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=results_subdir,
        system_name="Ru2Cu_0001",
    )
    print(campaign.format_screening_complete())


if __name__ == "__main__":
    main()

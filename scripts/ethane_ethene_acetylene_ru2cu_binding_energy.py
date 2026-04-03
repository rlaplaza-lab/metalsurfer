#!/usr/bin/env python3
"""Compute binding energies of ethane, ethene, and acetylene on Ru2Cu alloy (Ru host, 1/3 Cu) from mp-33.

Molecules: ethane (CC), ethene (C=C), acetylene (C#C).

Uses metalsurfer substitute_alloy to create Ru2Cu alloy from Ru(0001) base slab.
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import (
    AdsorptionConfig,
    create_slab_from_bulk,
    run_adsorption,
    substitute_alloy,
)
from metalsurfer._logging import configure_logging
from metalsurfer.cli.cli_output import format_screening_complete

# (SMILES, molecule_name)
MOLECULES = [
    ("CC", "ethane"),
    ("C=C", "ethene"),
    ("C#C", "acetylene"),
]


def main():
    configure_logging(default_level="INFO")
    results_subdir = "ethane_ethene_acetylene_ru2cu"
    results_dir = f"results_{results_subdir}"

    # Create Ru(0001) slab from Materials Project mp-33.
    base_slab = create_slab_from_bulk(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        results_dir=results_dir,
    )

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

    # Substitute 1/3 of Ru with Cu to form Ru2Cu alloy
    slab = substitute_alloy(
        base_slab,
        host_symbol="Ru",
        guest_symbol="Cu",
        guest_fraction=1.0 / 3.0,
        calculator=None,
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
    print(format_screening_complete(campaign.total_configurations))


if __name__ == "__main__":
    main()

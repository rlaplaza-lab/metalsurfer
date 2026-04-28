#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on Ru(0001) slab.

This example creates a Ru(0001) slab and computes ethene adsorption energy using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

Uses modest settings for quick demonstration (similar to test suite).
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

    # Create Ru(0001) slab using mp-33 (Ru) bulk structure
    slab = create_slab_from_bulk(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(2, 2, 1),  # Small supercell for quick demo
        results_dir="results_ethene_ru_slab",
    )

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=3,  # Ethene has planar geometry with some flexibility
        num_placements=5,  # Modest number for quick demonstration
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",  # use "cpu" if no GPU
        skip_topology_check=False,  # Keep topology check for ethene
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type="ethene_ru_slab",
        system_name="Ru_0001",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        total_steps = config.stage1_steps + config.stage2_steps
        print(
            f"\nBinding energy of ethene on Ru(0001): {summary.best_adsorption_energy:.4f} eV"
        )
        print("  (E_ads = E(Ru+ethene) - E(Ru) - E(ethene); negative = favorable)")
        print(
            f"  Relaxation: {total_steps} steps (stage1: {config.stage1_steps}, stage2: {config.stage2_steps})"
        )
        print(f"  {format_results_saved_line('results_ethene_ru_slab')}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()

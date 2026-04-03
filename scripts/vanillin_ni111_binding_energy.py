#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of vanillin on Ni(111) from mp-23 using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
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
    # Single subdir for slab, placements, and results (avoids path drift)
    results_subdir = "vanillin_ni111"
    results_dir = f"results_{results_subdir}"

    # Create Ni(111) slab from Materials Project mp-23.
    # Use minimal (1,1,1) supercell; auto_resize_slab will expand if needed for PBC separation.
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir=results_dir,
    )

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
        debug_write_initial_placements=True,
    )

    smiles = "c1(C=O)cc(OC)c(O)cc1"
    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, "vanillin")],
        config=config,
        surface_type=results_subdir,
        system_name="Ni_111",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        print(
            f"\nBinding energy of vanillin on Ni(111): {summary.best_adsorption_energy:.4f} eV"
        )
        print(
            "  (E_ads = E(slab+vanillin) - E(slab) - E(vanillin); negative = favorable)"
        )
        print(
            f"  Orientations: {summary.n_parallel}/{summary.n_valid_placements} parallel, "
            f"{summary.n_endown} EN-down"
        )
        print(f"  {format_results_saved_line(results_dir)}")
    else:
        print("No valid placements found.")

    if config.debug_write_initial_placements:
        print(
            f"\nInitial placements (pre-optimization): "
            f"{results_dir}/xyz_structures/vanillin_all/initial_*.xyz"
        )


if __name__ == "__main__":
    main()

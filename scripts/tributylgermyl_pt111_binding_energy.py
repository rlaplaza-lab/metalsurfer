#!/usr/bin/env python3
"""Binding energy screening of tri-n-butylgermyl anion (R3Ge−, no H on Ge) on Pt(111) from mp-126.

``min_contact_ratio=0.5`` allows close Ge–Pt approaches during relaxation.

The anion is a convenient way to get H-free M–C₃ starting points; use this run to mine
relaxed slab+adsorbate geometries. Do not treat printed E_ads as physical for a neutral process.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate


def main():
    configure_logging(default_level="INFO")
    results_subdir = "tributylgermyl_pt111"
    results_dir = f"results_{results_subdir}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=10,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        min_contact_ratio=0.5,
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    slab = prepare_substrate(
        bulk_id="mp-126",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    smiles = "[Ge-](CCCC)(CCCC)CCCC"
    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, "tributylgermyl")],
        config=config,
        surface_type=results_subdir,
        system_name="Pt_111",
    )
    summary = campaign.molecule_summaries[0]
    if summary.best_adsorption_energy is not None:
        print(
            f"\nBinding energy of tri-n-butylgermyl anion on Pt(111): "
            f"{summary.best_adsorption_energy:.4f} eV"
        )
        print(
            "  (E_ads = E(slab+adsorbate) - E(slab) - E(adsorbate); negative = favorable)"
        )
        print(
            f"  Orientations: {summary.n_parallel}/{summary.n_valid_placements} parallel, "
            f"{summary.n_endown} EN-down"
        )
        print(f"  {campaign.format_results_saved_line(results_dir=results_dir)}")
    else:
        print("No valid placements found.")


if __name__ == "__main__":
    main()

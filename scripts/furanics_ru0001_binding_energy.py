#!/usr/bin/env python3
"""Compute binding energies of furanic molecules on Ru(0001) from mp-33 using metalsurfer.

Molecules: HMF, BHMF, BHMTHF, 5-MF, MFA, DMF, MTHFA, DMTHF.

Uses BO pipeline: 300 placements max in passes of 100 (100 initial random + up to 2 BO passes of 100).
Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption_bo
from metalsurfer.surface_prep import prepare_substrate

# List of smiles and molecule name pairs
MOLECULES = [
    ("C(=O)C1OC(CO[H])=CC=1", "HMF"),
    ("C(O[H])C1OC(CO[H])=CC=1", "BHMF"),
    ("C(O[H])C1OC(CO[H])CC1", "BHMTHF"),
    ("C(=O)C1OC(C)=CC=1", "5-MF"),
    ("C(O[H])C1OC(C)=CC=1", "MFA"),
    ("C1(C)OC(C)=CC=1", "DMF"),
    ("C(O[H])C1OC(C)CC1", "MTHFA"),
    ("CC1OC(C)CC1", "DMTHF"),
]


def main() -> int:
    configure_logging(default_level="INFO")

    results_subdir = "furanics_ru0001"
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
        debug_write_initial_placements=False,
        bo_initial_random=100,
        bo_batch_size=100,
        # bo_total_budget = acquisition batches after initial (not total evals).
        bo_total_budget=2,  # 100 initial + 2×100 ≈ 300 evaluations
    )

    # Create Ru(0001) slab from Materials Project mp-33.
    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(1, 1, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption_bo(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=results_subdir,
        system_name="Ru_0001",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (Ru(0001))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of H2 on Ni(111) from mp-23 using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e . && pip install -e ".[mlip]"

``enable_dissociative_placement=True`` enables dissociative hollow-site pair
placements; ``skip_topology_check=True`` skips post-relax connectivity checks
(E_ads still vs isolated molecular E(H₂)).

Rerun note: ``skip_existing=True`` by default skips molecules already in
``adsorption_energies_detailed.csv``; delete ``results_h2_ni111/`` or pass
``skip_existing=False`` to force a fresh run.

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/h2_ni111_binding_energy.py
or reduce num_placements (e.g. 25).
"""

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.surface_prep import prepare_substrate


def main() -> int:
    configure_logging(default_level="INFO")
    surface_type = "h2_ni111"
    results_dir = f"results_{surface_type}"

    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-m-1p1",
        seed=42,
        num_conformers=1,
        num_placements=250,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        enable_dissociative_placement=True,
        skip_topology_check=True,  # allow fragmented adsorbates after relax
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Ni_111",
    )
    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (H2 / Ni(111))",
            results_dir=results_dir,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

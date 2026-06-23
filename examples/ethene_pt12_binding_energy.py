#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on a small Pt nanocluster (12 atoms).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

The hand-built Pt₁₂ cluster keeps its input geometry during ``prepare_substrate``
(``slab_relaxation_mode="none"``): unrestricted ionic prep relaxation can distort
small hand-built nanoparticles and yield unreliable adsorption energies.

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/ethene_pt12_binding_energy.py
or reduce num_placements (e.g. 25).
"""

from __future__ import annotations

import sys

from ase import Atoms

from metalsurfer import (
    AdsorptionConfig,
    BindingCampaignResult,
    configure_logging,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


def _validate_campaign(campaign: BindingCampaignResult, *, results_dir: str) -> None:
    """Exit non-zero when the demo did not find favorable molecular adsorption."""
    if not campaign.molecule_summaries:
        print("No molecule summaries produced.", file=sys.stderr)
        raise SystemExit(1)

    summary = campaign.molecule_summaries[0]
    if summary.n_valid_placements < 3:
        print(
            f"Expected >= 3 valid placements, got {summary.n_valid_placements}.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)

    best = summary.best_adsorption_energy
    if best is None or best >= 0.0:
        print(
            f"Expected favorable binding (best E_ads < 0 eV), got {best}.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)

    if best < -3.0:
        print(
            f"Best E_ads {best:.4f} eV is unexpectedly strong for ethene on Pt₁₂.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "ethene_pt12"
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
        num_conformers=3,
        num_placements=5,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        slab_relaxation_mode="none",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    nanocluster = prepare_substrate(
        slab=pt_atoms,
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=nanocluster,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
        system_name="Pt_12",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethene / Pt12 nanocluster)",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

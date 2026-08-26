#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on a small Pt nanocluster (12 atoms).

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

The hand-built Pt₁₂ cluster keeps its input geometry during ``prepare_substrate``
(``slab_relaxation_mode="none"``): unrestricted ionic prep relaxation can distort
small hand-built nanoparticles and yield unreliable adsorption energies. The whole
cluster is also frozen during adsorption (default prep ``FixAtoms``); under that
rigid-cluster approximation UMA places the best surviving pose right at
E_ads ≈ 0 eV — a chemisorbed C–Pt contact (~2.1 Å, C=C stretched to ~1.4 Å) whose
missing cluster-relaxation energy offsets the bond. The demo therefore validates
the chemisorption contact rather than a strictly negative E_ads.

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
    results_dir_for,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate

# A relaxed best pose at or below this distance means ethene made a true
# chemisorption contact (physisorption sits around 3+ Å).
CHEMISORPTION_CONTACT_ANG = 2.6
# Generous ceiling rejecting broken runs where every pose ends up strongly
# endothermic.
E_ADS_CEILING_EV = 1.0


def _validate_campaign(campaign: BindingCampaignResult, *, results_dir: str) -> None:
    """Exit non-zero unless a chemisorption-contact pose survived relaxation."""
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

    run_result = campaign.run_results[0]
    best = min(run_result.results, key=lambda r: r.energy_adsorption)
    if best.energy_adsorption >= E_ADS_CEILING_EV:
        print(
            f"Best E_ads {best.energy_adsorption:.4f} eV exceeds the "
            f"{E_ADS_CEILING_EV:.1f} eV ceiling for ethene on Pt₁₂.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)
    if best.distance > CHEMISORPTION_CONTACT_ANG:
        print(
            f"Best pose has no chemisorption contact "
            f"(closest approach {best.distance:.2f} Å > "
            f"{CHEMISORPTION_CONTACT_ANG:.1f} Å).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "ethene_pt12"
    results_dir = str(results_dir_for(surface_type))

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

    # Modest placement count + GPU memory padding for small demo GPUs (~15 GB).
    # Prefer enough samples that at least one chemisorbed ethene pose survives.
    config = AdsorptionConfig(
        material_type="nanoparticle",
        seed=42,
        num_conformers=3,
        num_placements=25,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        slab_relaxation_mode="none",
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
        skip_existing=False,
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

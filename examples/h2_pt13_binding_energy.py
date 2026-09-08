#!/usr/bin/env python3
"""Compute H2 dissociative adsorption on an ASE Pt₁₃ icosahedron.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

Uses ``ase.cluster.Icosahedron("Pt", noshells=2)`` (13 atoms), UMA ionic prep
relaxation, then frozen-cluster dissociative hollow-pair placements. Under
``uma-s-1p2`` / ``oc25`` this yields chemisorbed dissociated H (H–Pt ≈ 1.8 Å,
H–H ≈ 2.2 Å) with favorable E_ads ≈ −1.1 eV — unlike the hand-built Pt₁₂ toy
used by the ethene nanoparticle demo, which is too strained for reliable H₂
thermodynamics after the same protocol.

``enable_dissociative_placement=True`` and ``skip_topology_check=True`` match
the H₂/Ru(0001) demo.

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/h2_pt13_binding_energy.py
"""

from __future__ import annotations

import sys

import numpy as np
from ase.cluster import Icosahedron

from metalsurfer import (
    AdsorptionConfig,
    BindingCampaignResult,
    ScreeningResult,
    configure_logging,
    results_dir_for,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate

# Best-E_ads band (uma-s-1p2 + oc25): observed ≈ −1.128 eV on prep-relaxed Pt₁₃.
E_ADS_CEILING_EV = -0.90
E_ADS_FLOOR_EV = -1.40
CHEMISORPTION_CONTACT_ANG = 2.2


def _pt13_icosahedron():
    atoms = Icosahedron("Pt", noshells=2)
    atoms.set_cell([30.0, 30.0, 30.0])
    atoms.center()
    atoms.pbc = False
    return atoms


def _validate_dissociative_chemisorption(result: ScreeningResult) -> None:
    descriptor = result.placement_descriptor
    if descriptor.orientation_type != "dissociative":
        print(
            f"Expected dissociative placement, got {descriptor.orientation_type}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if descriptor.site_source != "dissociative_hollow_pair":
        print(
            f"Expected dissociative_hollow_pair site source, got {descriptor.site_source}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if result.distance > CHEMISORPTION_CONTACT_ANG:
        print(
            f"Best pose has no chemisorption contact "
            f"(closest approach {result.distance:.2f} Å > "
            f"{CHEMISORPTION_CONTACT_ANG:.1f} Å).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _validate_campaign(campaign: BindingCampaignResult, *, results_dir: str) -> None:
    if not campaign.run_results:
        print("No screening results produced.", file=sys.stderr)
        raise SystemExit(1)

    results = campaign.run_results[0].results
    if len(results) < 3:
        print(
            f"Expected >= 3 valid H2 placements, got {len(results)}.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)

    best = min(results, key=lambda r: r.energy_adsorption)
    _validate_dissociative_chemisorption(best)

    e_ads = best.energy_adsorption
    if not np.isfinite(e_ads) or e_ads >= E_ADS_CEILING_EV:
        print(
            f"Expected favorable dissociative H2 binding on Pt₁₃ "
            f"(best E_ads < {E_ADS_CEILING_EV:.2f} eV), got {e_ads}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if e_ads < E_ADS_FLOOR_EV:
        print(
            f"Best E_ads {e_ads:.4f} eV is below the {E_ADS_FLOOR_EV:.2f} eV "
            "floor for H2 on Pt₁₃ (unexpectedly strong vs QC).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "h2_pt13"
    results_dir = str(results_dir_for(surface_type))

    config = AdsorptionConfig(
        material_type="nanoparticle",
        seed=42,
        num_conformers=1,
        num_placements=10,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        slab_relaxation_mode="ionic_only",
        enable_dissociative_placement=True,
        skip_topology_check=True,
        stage2_steps=500,
    )

    nanocluster = prepare_substrate(
        slab=_pt13_icosahedron(),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=nanocluster,
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Pt_13",
        skip_existing=False,
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (H2 / Pt13 icosahedron, dissociative)",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

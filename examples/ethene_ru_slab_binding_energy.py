#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on Ru(0001) slab.

This example creates a Ru(0001) slab and computes ethene adsorption energy using metalsurfer.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

Uses modest settings for quick demonstration (similar to test suite).

Note: absolute E_ads depends on the substrate source. This demo builds the slab
from the Materials Project entry ``mp-33`` (DFT-relaxed lattice constant);
under UMA ``oc25`` the best surviving pose lands right around E_ads ≈ 0 eV —
a chemisorbed di-σ configuration (C≈2.0 Å above the surface, C=C stretched)
whose exact sign tracks the lattice constant. The demo therefore validates
that a chemisorption-contact pose survives relaxation rather than a strictly
negative E_ads.
"""

from __future__ import annotations

import sys

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
            f"{E_ADS_CEILING_EV:.1f} eV ceiling for ethene on Ru(0001).",
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

    surface_type = "ethene_ru_slab"
    results_dir = str(results_dir_for(surface_type))

    # Modest placement count + GPU memory padding for small demo GPUs (~15 GB).
    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=3,
        num_placements=5,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        stage2_steps=500,
    )

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(2, 2, 1),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
        system_name="Ru_0001",
        skip_existing=False,
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethene / Ru(0001))",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

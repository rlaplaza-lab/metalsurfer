#!/usr/bin/env python3
"""H₂ adsorption on Ru(0001) with dissociative initial placements.

``enable_dissociative_placement=True`` enables hollow-site pair placements.
``skip_topology_check=True`` keeps structures where the H–H bond has broken.
E_ads is always reported vs isolated molecular E(H₂). On Ru(0001) with UMA,
the best minima are typically dissociated H (H–H ≳ 2.5 Å, E_ads ≈ −0.4 eV).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.
"""

from __future__ import annotations

import sys

import numpy as np

from metalsurfer import (
    AdsorptionConfig,
    BindingCampaignResult,
    ScreeningResult,
    configure_logging,
    results_dir_for,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


def _validate_campaign(campaign: BindingCampaignResult, *, results_dir: str) -> None:
    """Exit non-zero when the dissociative slab workflow did not complete."""
    if not campaign.run_results:
        print("No screening results produced.", file=sys.stderr)
        raise SystemExit(1)

    results = campaign.run_results[0].results
    if not results:
        print("No valid H2 placements after filtering.", file=sys.stderr)
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)

    best = min(results, key=lambda r: r.energy_adsorption)
    _validate_dissociative_result(best)

    e_ads = best.energy_adsorption
    # Best-E_ads band (uma-s-1p2 + oc25 QC): observed ≈ −0.40 eV for
    # fully dissociated H (H–H ≈ 2.8 Å, H–Ru ≈ 1.9 Å). Weaker ~−0.11 eV
    # minima remain when H stay closer (~2.0 Å); the lock requires the
    # dissociated basin.
    e_ads_ceiling_ev = -0.30
    e_ads_floor_ev = -0.48
    if not np.isfinite(e_ads) or e_ads >= e_ads_ceiling_ev:
        print(
            f"Expected favorable H2 binding on Ru "
            f"(best E_ads < {e_ads_ceiling_ev:.2f} eV), got {e_ads}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if e_ads < e_ads_floor_ev:
        print(
            f"Best E_ads {e_ads:.4f} eV is below the {e_ads_floor_ev:.2f} eV "
            "floor for H2 on Ru (unexpectedly strong vs QC).",
            file=sys.stderr,
        )
        raise SystemExit(1)


def _validate_dissociative_result(result: ScreeningResult) -> None:
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
    if not (1.5 <= result.distance <= 4.0):
        print(
            f"Adsorbate–surface distance should be 1.5–4 Å, got {result.distance:.2f}.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "h2_ru_slab"
    results_dir = str(results_dir_for(surface_type))

    # Modest placement count: many dissociative trials desorb on this surface.
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=10,
        enable_dissociative_placement=True,
        skip_topology_check=True,
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
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Ru_0001",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (H2 / Ru(0001), dissociative)",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

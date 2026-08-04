#!/usr/bin/env python3
"""Compute H2 adsorption on Ru(0001) with dissociative initial placements.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.
Run from project root: pip install -e ".[mlip]"

``enable_dissociative_placement=True`` enables dissociative hollow-site pair
placements on the periodic slab. ``skip_topology_check=True`` disables
post-relaxation connectivity / decomposition checks so structures where the
H–H bond has broken are retained. E_ads is always reported vs isolated
molecular E(H₂). On Ru(0001) with UMA, many minima relax to molecular H₂
physisorption (~0.75 Å H–H); dissociated minima are also allowed when the
model finds them.

Uses a modest placement count because many dissociative trials desorb on this surface.
Initial z heights use default ``placement_z_range`` scale factors on
``(r_adsorbate + r_surface)`` (see :class:`~metalsurfer.AdsorptionConfig`).

If you hit CUDA OOM on a 15GB GPU, try:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python examples/h2_ru_slab_binding_energy.py
or reduce num_placements (e.g. 25).
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
    if not np.isfinite(e_ads) or e_ads > 5.0:
        print(f"E_ads out of smoke-test range: {e_ads}", file=sys.stderr)
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

    # enable_dissociative_placement: hollow-pair placements.
    # skip_topology_check: allow fragmented adsorbates after relax.
    # Modest placement count + GPU memory padding for small demo GPUs (~15 GB).
    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=1,
        num_placements=10,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        enable_dissociative_placement=True,
        skip_topology_check=True,
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
        molecules=[("[H][H]", "H2")],
        config=config,
        surface_type=surface_type,
        system_name="Ru_0001",
        skip_existing=False,
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

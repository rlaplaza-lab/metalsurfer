#!/usr/bin/env python3
"""Binding energy of CO₂ in a MOF periodic cell (RUBTAK01).

Loads the experimental CIF and keeps the published framework geometry
(``slab_relaxation_mode="none"``).

Requires: ``pip install -e ".[mlip]"``. Run from the project root.

CIF source:
https://github.com/bafgreat/mofstructure/blob/main/tests/test_data/RUBTAK01.cif
"""

from __future__ import annotations

import os
import sys

from ase.io import read

from metalsurfer import (
    AdsorptionConfig,
    BindingCampaignResult,
    configure_logging,
    results_dir_for,
    run_adsorption,
)
from metalsurfer.surface_prep import prepare_substrate


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

    best = summary.best_adsorption_energy
    # Best-E_ads band (uma-s-1p2 + oc25 QC): observed ≈ −0.301 eV.
    e_ads_ceiling_ev = -0.25
    e_ads_floor_ev = -0.34
    if best is None or best >= e_ads_ceiling_ev:
        print(
            f"Expected favorable CO₂ physisorption "
            f"(best E_ads < {e_ads_ceiling_ev:.2f} eV), got {best}.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)

    if best < e_ads_floor_ev:
        print(
            f"Best E_ads {best:.4f} eV is below the {e_ads_floor_ev:.2f} eV "
            "floor for CO₂ in this MOF (unexpectedly strong vs QC).",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)


def main() -> int:
    configure_logging(default_level="INFO")

    surface_type = "co2_mof"
    results_dir = str(results_dir_for(surface_type))
    cif_path = os.path.join(os.path.dirname(__file__), "mof_structures", "RUBTAK01.cif")

    if not os.path.exists(cif_path):
        raise FileNotFoundError(
            f"MOF CIF file not found at {cif_path}. "
            "Please ensure the RUBTAK01.cif file is present in examples/mof_structures/"
        )

    mof_atoms = read(cif_path)

    config = AdsorptionConfig(
        material_type="porous",
        slab_relaxation_mode="none",  # keep experimental CIF framework geometry
        num_conformers=1,
        num_placements=5,
    )

    mof_slab = prepare_substrate(
        slab=mof_atoms,
        config=config,
        results_dir=results_dir,
        align=False,
    )

    campaign = run_adsorption(
        slab=mof_slab,
        molecules=[("O=C=O", "CO2")],
        config=config,
        surface_type=surface_type,
        system_name="MOF_cell",
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (CO2 / MOF)",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

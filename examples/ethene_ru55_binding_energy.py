#!/usr/bin/env python3
"""Compute the binding (adsorption) energy of ethene on an ASE Ru₅₅ icosahedron.

Requires: metalsurfer with MLIP stack (torch-sim-atomistic, fairchem-data-oc, torch) and rdkit.

Uses ``ase.cluster.Icosahedron("Ru", noshells=3, latticeconstant=...)`` (55 atoms).
ASE cannot guess Ru's lattice constant (hcp), so we pass the FCC-equivalent
``a ≈ 3.83 Å`` that matches Ru hcp nearest-neighbour spacing (~2.71 Å).
Keep the input cluster geometry (``slab_relaxation_mode="none"``): ionic prep
can distort the icosahedron and yield unbound ethene under UMA. The whole
cluster is frozen during adsorption (default prep ``FixAtoms``).

Run (conda env metalsurfer)::

  conda activate metalsurfer
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python examples/ethene_ru55_binding_energy.py
"""

from __future__ import annotations

import sys

from ase.cluster import Icosahedron

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
# Provisional best-E_ads band (uma-s-1p2 + oc25) on the ASE icosahedron with
# slab_relaxation_mode="none" and prep-frozen Ru₅₅; e2e A/B saw best ≈ −4.81
# (auto) / −5.00 (adaptive_grid) at 6 placements.
E_ADS_CEILING_EV = -4.20
E_ADS_FLOOR_EV = -6.00

# FCC-equivalent a so NN ≈ Ru hcp a (~2.71 Å): a_fcc = a_hcp * sqrt(2).
_RU_FCC_LATTICE_CONSTANT = 2.71 * (2.0**0.5)


def _ru55_icosahedron():
    atoms = Icosahedron("Ru", noshells=3, latticeconstant=_RU_FCC_LATTICE_CONSTANT)
    atoms.set_cell([40.0, 40.0, 40.0])
    atoms.center()
    atoms.pbc = False
    return atoms


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
            f"{E_ADS_CEILING_EV:.2f} eV ceiling for ethene on Ru₅₅.",
            file=sys.stderr,
        )
        print(campaign.format_summary(results_dir=results_dir), file=sys.stderr)
        raise SystemExit(1)
    if best.energy_adsorption < E_ADS_FLOOR_EV:
        print(
            f"Best E_ads {best.energy_adsorption:.4f} eV is below the "
            f"{E_ADS_FLOOR_EV:.2f} eV floor for ethene on Ru₅₅ "
            "(unexpectedly strong vs QC).",
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

    surface_type = "ethene_ru55"
    results_dir = str(results_dir_for(surface_type))

    config = AdsorptionConfig(
        material_type="nanoparticle",
        seed=42,
        num_conformers=3,
        num_placements=25,
        n_jobs=1,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        slab_relaxation_mode="none",
        stage2_steps=500,
    )

    nanocluster = prepare_substrate(
        slab=_ru55_icosahedron(),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_adsorption(
        slab=nanocluster,
        molecules=[("C=C", "ethene")],
        config=config,
        surface_type=surface_type,
        system_name="Ru_55",
        skip_existing=False,
    )

    print()
    print(
        campaign.format_summary(
            title="Binding energy summary (ethene / Ru55 icosahedron)",
            results_dir=results_dir,
        )
    )
    _validate_campaign(campaign, results_dir=results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

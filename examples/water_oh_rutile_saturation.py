#!/usr/bin/env python3
"""Competitive saturation: water and hydroxide together on rutile TiO2(110).

Two molecules are screened at the same time on one growing slab:

- ``multi_molecule_saturation=True`` — water and OH- compete at every step;
  the placement budget is split by molecular complexity and the best binder
  advances the surface.
- ``saturation_molecules_per_step=2`` (n-tuplet mode) — each step may commit
  up to two mutually clear winners at once, relaxed as ONE composite
  structure. Committed rows carry the shared tuplet E_ads.

The substrate is the classic rutile TiO2(110) surface. Oxides are absent from
the FairChem bulk database used by ``bulk_id=``, so the slab is built with ASE
(``ase.spacegroup`` + ``ase.build.surface``) and passed to
``prepare_substrate(slab=...)``.

Requires: ``pip install -e ".[mlip]"``. Run from the project root::

    python examples/water_oh_rutile_saturation.py
"""

from __future__ import annotations

import sys

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.spacegroup import crystal

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    results_dir_for,
    run_saturation,
)
from metalsurfer.surface_prep import prepare_substrate

SURFACE_TYPE = "water_oh_rutile_saturation"
RUTILE_A = 4.5944  # Å, TiO2 rutile lattice parameter
RUTILE_C = 2.9587  # Å


def build_rutile_tio2_110() -> Atoms:
    """Build a water-sized rutile TiO2(110) slab with ASE.

    The in-plane cell must fit the largest adsorbate plus
    ``min_pbc_image_separation`` (default 8 Å); a (2, 4, 1) repeat of the
    (110) surface cell (~13 x 11.8 Å) satisfies this for water.
    """
    tio2 = crystal(
        ["Ti", "O"],
        basis=[(0, 0, 0), (0.3051, 0.3051, 0)],
        spacegroup=136,
        cellpar=(RUTILE_A, RUTILE_A, RUTILE_C, 90, 90, 90),
    )
    slab = surface(tio2, (1, 1, 0), layers=4, vacuum=10.0)
    return slab.repeat((2, 4, 1))


def main() -> int:
    configure_logging(default_level="INFO")
    results_dir = str(results_dir_for(SURFACE_TYPE))

    config = AdsorptionConfig(
        num_conformers=2,
        num_placements=16,
        multi_molecule_saturation=True,
        saturation_molecules_per_step=2,
        saturation_max_steps=3,
        stage2_steps=300,
    )

    slab = prepare_substrate(
        slab=build_rutile_tio2_110(),
        config=config,
        results_dir=results_dir,
    )

    campaign = run_saturation(
        slab=slab,
        molecules=[
            ("O", "water"),
            ("[OH-]", "hydroxide"),
        ],
        config=config,
        surface_type=SURFACE_TYPE,
    )

    if not campaign.runs:
        print("No saturation runs produced.", file=sys.stderr)
        return 1
    result = campaign.runs[0]

    print()
    print(f"Competitive saturation on {SURFACE_TYPE}:")
    for step_result in result.steps:
        committed = step_result.committed()
        energies = (
            ", ".join(
                f"{unit.molecule}: {unit.energy_adsorption:.3f} eV"
                for unit in committed
            )
            or f"{step_result.best_result.energy_adsorption:.3f} eV (unbound final step)"
        )
        print(
            f"  step {step_result.step:>2d} | on slab: "
            f"{step_result.n_molecules_on_slab:>2d} | committed: "
            f"{step_result.n_added} | {energies}"
        )
    print(f"  coverage at saturation: {result.molecule_counts}")

    # Best single-molecule E_ads band (uma-s-1p2 + oc25 QC): ≈ −3.23 eV.
    # Use per-molecule screening results — committed tuplet E_ads is a shared
    # composite (~sum of units), not a per-adsorbate figure.
    e_ads_ceiling_ev = -3.10
    e_ads_floor_ev = -3.35
    first_bound = next((s for s in result.steps if s.n_added > 0), None)
    if first_bound is None:
        print("No committed saturation step found.", file=sys.stderr)
        return 1
    screened = [
        r.energy_adsorption
        for group in first_bound.per_molecule_results.values()
        for r in group
        if np.isfinite(r.energy_adsorption)
    ]
    if not screened:
        print(
            "No finite per-molecule screening E_ads on first bound step.",
            file=sys.stderr,
        )
        return 1
    best_first = min(screened)
    if best_first >= e_ads_ceiling_ev:
        print(
            f"Expected strong first-step binding "
            f"(best E_ads < {e_ads_ceiling_ev:.2f} eV), "
            f"got {best_first:.4f} eV.",
            file=sys.stderr,
        )
        return 1
    if best_first < e_ads_floor_ev:
        print(
            f"First-step best E_ads {best_first:.4f} eV is below the "
            f"{e_ads_floor_ev:.2f} eV floor (unexpectedly strong vs QC).",
            file=sys.stderr,
        )
        return 1

    print(
        f"\nResults written under {results_dir} "
        f"(first-step best screened E_ads = {best_first:.4f} eV)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

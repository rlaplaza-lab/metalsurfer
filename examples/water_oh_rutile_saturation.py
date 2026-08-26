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
``prepare_substrate(slab=...)``, which equilibrates it, applies PBC and freeze
constraints, and validates it for campaigns.

Requires: ``pip install -e ".[mlip]"`` and a CUDA-capable GPU.

Run from the project root::

    python examples/water_oh_rutile_saturation.py
"""

from __future__ import annotations

import logging
import sys

import numpy as np
from ase import Atoms
from ase.build import surface
from ase.spacegroup import crystal

from metalsurfer import (
    AdsorptionConfig,
    MultiMolSaturationRunResult,
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
    logger = logging.getLogger(__name__)
    results_dir = str(results_dir_for(SURFACE_TYPE))

    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=2,
        num_placements=16,
        multi_molecule_saturation=True,
        saturation_molecules_per_step=2,
        saturation_max_steps=3,
        stage1_steps=50,
        stage2_steps=300,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
    )

    slab_atoms = build_rutile_tio2_110()
    logger.info("Rutile TiO2(110) substrate atoms: %d", len(slab_atoms))
    slab = prepare_substrate(
        slab=slab_atoms,
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
        skip_existing=False,
    )

    if not campaign.runs:
        print("No saturation runs produced.", file=sys.stderr)
        return 1
    result = campaign.runs[0]
    if not isinstance(result, MultiMolSaturationRunResult):
        print(
            "Expected a competitive multi-molecule saturation run.",
            file=sys.stderr,
        )
        return 1
    if not result.steps or result.n_molecules_at_saturation < 1:
        print(
            "No molecule bound within the step limit; nothing to validate.",
            file=sys.stderr,
        )
        return 1

    counts_total = sum(result.molecule_counts.values())
    if counts_total != result.n_molecules_at_saturation:
        print(
            f"molecule_counts {result.molecule_counts} do not sum to "
            f"n_molecules_at_saturation={result.n_molecules_at_saturation}.",
            file=sys.stderr,
        )
        return 1
    # Adsorbate atoms on the final slab must match what the steps committed.
    # All winners of one step share a single composite structure, so count its
    # adsorbate suffix once per bound step.
    n_adsorbate_expected = len(result.final_slab_atoms) - len(slab.atoms)
    n_adsorbate_committed = 0
    for step_result in result.steps:
        units = step_result.committed()
        if units:
            n_adsorbate_committed += len(units[0].atoms) - units[0].slab_size
    if n_adsorbate_expected != n_adsorbate_committed:
        print(
            f"Final slab holds {n_adsorbate_expected} adsorbate atoms, "
            f"but the step record commits {n_adsorbate_committed}.",
            file=sys.stderr,
        )
        return 1

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
    e_ads_rows = [
        unit.energy_adsorption
        for step_result in result.steps
        for unit in step_result.committed()
    ]
    if not np.all(np.isfinite(e_ads_rows)):
        print("Non-finite E_ads rows found.", file=sys.stderr)
        return 1

    print(
        f"\nResults written under {results_dir} "
        "(saturation_details.csv includes a committed_molecule column for "
        "multi-winner steps)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Diagnose placement generation failure reasons on a small slab fixture.

Runs without GPU (``slab_relaxation_mode="none"``, ``device="cpu"``) and prints a
reason histogram so operators can tune distance / strict / VDW knobs before a
full ML campaign.
"""

from __future__ import annotations

from collections import Counter

from ase import Atoms
from ase.build import fcc111

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec_with_reason,
)


def _water_conformer() -> Atoms:
    water = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    water.center()
    return water


def main() -> None:
    slab = fcc111("Pt", size=(2, 2, 3), vacuum=10.0, periodic=True)
    smiles = "O"
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=24,
        slab_relaxation_mode="none",
        device="cpu",
        seed=0,
    )
    conformers = [_water_conformer()]

    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        smiles,
        config.num_placements or 24,
        seed=config.seed,
    )
    reasons: Counter[str] = Counter()
    n_ok = 0
    for spec in specs:
        result, reason = generate_placement_from_spec_with_reason(
            spec, conformers, slab, config, smiles=smiles
        )
        if result is None:
            reasons[reason or "unknown"] += 1
        else:
            n_ok += 1

    print(f"specs={len(specs)} valid={n_ok} failed={sum(reasons.values())}")
    if reasons:
        print("Failure reason histogram:")
        for reason, count in reasons.most_common():
            print(f"  {reason}: {count}")
    else:
        print("No generation failures.")


if __name__ == "__main__":
    main()

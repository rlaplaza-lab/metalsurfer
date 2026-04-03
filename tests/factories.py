"""Shared test data factories (random PlacementRecords for ML / Bayesian tests)."""

from __future__ import annotations

from typing import Literal

import numpy as np

from metalsurfer.ml.schema import ComputationContext, PlacementRecord

Variant = Literal["bayesian", "ml"]


def make_random_placement_records(
    n: int,
    *,
    seed: int = 42,
    variant: Variant = "ml",
    with_binding_energies: bool | None = None,
) -> list[PlacementRecord]:
    """Build *n* randomized records with fixed RNG *seed* for stable tests.

    ``variant="bayesian"`` matches the former ``test_bayesian`` synthetic set
    (single pseudo-molecule, narrower site/azimuth choices, no slab energies).
    ``variant="ml"`` matches the former ``test_ml`` synthetic set (rotating
    molecules, optional binding-energy fields).
    """
    if with_binding_energies is None:
        with_binding_energies = variant == "ml"

    rng = np.random.RandomState(seed)
    records: list[PlacementRecord] = []

    mols = ["ethanol", "methanol", "water", "CO"]
    smiles_map = {
        "ethanol": "CCO",
        "methanol": "CO",
        "water": "O",
        "CO": "[C-]#[O+]",
    }

    for i in range(n):
        if variant == "bayesian":
            molecule, smiles, surface_id = "test", "C", "test"
            site_choices = ["atop", "bridge", "hollow"]
            az_choices = [0, 45, 90, 135, 180]
        else:
            mol = mols[i % len(mols)]
            molecule, smiles, surface_id = mol, smiles_map[mol], "Cu_fcc111"
            site_choices = ["atop", "bridge", "hollow", "envelope"]
            az_choices = [0, 45, 90, 135, 180, 225, 270, 315]

        z_off = float(rng.uniform(2, 3))
        tilt = float(rng.choice([0, 15, 30, 45, 60, 90]))
        e_ads = -0.5 * z_off + 0.01 * tilt + float(rng.normal(0, 0.1))
        x_ = float(rng.uniform(-4, 4))
        y_ = float(rng.uniform(-4, 4))
        surf_z = 7.0

        kwargs: dict = {
            "molecule": molecule,
            "smiles": smiles,
            "surface_id": surface_id,
            "placement_id": i,
            "conformer_index": i % 3,
            "orientation_type": str(
                rng.choice(["parallel", "EN-down", "vertical", "round"])
            ),
            "face_flip": False,
            "en_atom_index": None,
            "site_index": i % 5,
            "site_type": str(rng.choice(site_choices)),
            "tilt_deg": tilt,
            "azimuth_deg": float(rng.choice(az_choices)),
            "azimuth_in_plane_deg": 0.0,
            "z_fraction": 0.5,
            "x": x_,
            "y": y_,
            "x_abs": x_,
            "y_abs": y_,
            "z_offset": z_off,
            "surface_ref_z_abs": surf_z,
            "z_abs": surf_z + z_off,
            "shape": str(rng.choice(["linear", "flat", "round"])),
            "energy_adsorption": e_ads,
            "context": ComputationContext(),
        }
        if with_binding_energies:
            kwargs["energy_adslab"] = float(-150 + e_ads)
            kwargs["energy_slab"] = -145.0
            kwargs["energy_adsorbate"] = -5.0
            kwargs["distance"] = 2.3

        records.append(PlacementRecord(**kwargs))

    return records

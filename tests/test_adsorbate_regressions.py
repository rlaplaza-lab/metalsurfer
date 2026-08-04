"""Focused adsorbate regressions without full MLIP runs."""

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec,
)
from metalsurfer.placement.site_context import clear_site_caches
from metalsurfer.surface_prep import SlabContainer, deposit_adatoms

from .conftest import make_slab


def test_deposit_adatoms_then_generate_placement_from_spec():
    slab = SlabContainer(make_slab(nx=4, ny=4, n_layers=3))
    decorated = deposit_adatoms(
        slab,
        "Sn",
        coverage_fraction=0.15,
        seed=7,
        relaxation_mode="none",
    )
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=8,
        placement_z_range=(2.0, 3.0),
    )
    result = create_conformers_from_smiles(
        "O", config=AdsorptionConfig(num_conformers=1)
    )
    if result is None:
        pytest.skip("RDKit required")
    conformers, _ = result

    clear_site_caches()
    specs = enumerate_placement_specs(
        conformers,
        decorated.atoms,
        config,
        "O",
        n_desired=4,
    )
    assert specs
    placed = generate_placement_from_spec(specs[0], conformers, decorated.atoms, config)
    assert placed is not None
    adsorbate, _descriptor = placed
    assert len(adsorbate) == 3
    assert len(decorated.atoms) > len(slab.atoms)

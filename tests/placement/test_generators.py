"""Placement spec enumeration and materialization."""


import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.models import PlacementSpec
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec,
)
from metalsurfer.placement.site_context import (
    SiteContext,
    clear_site_caches,
)
from metalsurfer.surface_prep import SlabContainer, deposit_adatoms

from ..conftest import (
    make_slab,
    make_water,
)


def test_enumerate_specs_empty_sites_returns_empty():
    """No sites / use_sites=False must not invent site_index=-1 capacity."""
    from metalsurfer.placement.generators import (
        enumerate_placement_specs,
        estimate_placement_spec_capacity,
    )

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0, num_placements=40)
    ctx = SiteContext(
        sites=[],
        use_sites=False,
        source="no_sites",
        raw_unclustered=[],
    )
    specs = enumerate_placement_specs(
        [make_water()],
        slab,
        config,
        "O",
        n_desired=40,
        site_context=ctx,
    )
    assert specs == []
    capacity = estimate_placement_spec_capacity(
        [make_water()], slab, config, "O", site_context=ctx
    )
    assert capacity == 0

def test_enumerate_specs_skips_occupied_site_indices():
    from metalsurfer.placement.generators import enumerate_placement_specs
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0, num_placements=40)
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    assert ctx.use_sites and ctx.sites
    blocked = 0
    site = ctx.sites[blocked]
    water = make_water().copy()
    # Place clearly inside min_initial_distance (equality at the threshold is kept).
    water.set_positions(water.get_positions() + site.xyz + np.array([0.0, 0.0, 0.5]))
    full = slab.copy() + water
    specs = enumerate_placement_specs(
        [make_water()],
        slab,
        config,
        "O",
        n_desired=40,
        site_context=ctx,
        full_slab=full,
    )
    assert specs
    assert all(int(s.site_index) != blocked for s in specs if int(s.site_index) >= 0)

def test_generate_placements_from_specs_preserves_order(monkeypatch):
    """Serial and threaded paths return successes/failures in input order."""

    from metalsurfer.placement import generators as gen_mod
    from metalsurfer.workflow.shared import _materialize_spec_placements

    from ..conftest import make_placement_descriptor

    def _spec(i: int) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=i,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=i,
        )

    def fake_generate(spec, *args, **kwargs):
        if spec.placement_index % 2 == 1:
            return None, "too_close"
        desc = make_placement_descriptor(placement_id=spec.placement_index)
        return (Atoms("H"), desc), None

    monkeypatch.setattr(
        gen_mod, "generate_placement_from_spec_with_reason", fake_generate
    )

    specs = [_spec(i) for i in range(6)]
    slab = make_slab()
    for workers in (1, 4):
        config = AdsorptionConfig(
            material_type="slab",
            num_placements=6,
            placement_materialize_workers=workers,
        )
        combined, ids, _descs, failures = _materialize_spec_placements(
            specs=specs,
            conformers=[make_water()],
            slab_atoms=slab,
            calculator=None,
            config=config,
            smiles="O",
            site_context=None,
        )
        assert ids == [0, 2, 4]
        assert [f.placement_id for f in failures] == [1, 3, 5]
        assert len(combined) == 3


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


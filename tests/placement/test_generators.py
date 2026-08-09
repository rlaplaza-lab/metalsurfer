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


def test_boltzmann_weighting_is_deterministic_and_proportional():
    """conformer_weighting='boltzmann' skews the spec mix toward low-E conformers.

    Uses synthetic conformers (no RDKit needed) so the test is hermetic. The
    Boltzmann prior must (a) be deterministic for fixed seed/energies, and (b)
    produce a conformer_index histogram skewed toward the lowest-energy
    conformer relative to the uniform draw.
    """
    from collections import Counter

    from metalsurfer.placement.generators import enumerate_placement_specs

    clear_site_caches()
    slab = make_slab()
    n_conformers = 4
    conformers = [make_water() for _ in range(n_conformers)]
    energies = [0.0, 0.01, 0.02, 0.03]  # eV, gentle finite spread

    base = dict(
        material_type="slab",
        seed=0,
        num_conformers=n_conformers,
        num_placements=200,
    )

    def hist(**kw):
        cfg = AdsorptionConfig(**{**base, **kw})
        specs = enumerate_placement_specs(
            conformers,
            slab,
            cfg,
            "O",
            n_desired=120,
            conformer_energies=energies,
        )
        return Counter(s.conformer_index for s in specs)

    uni = hist(conformer_weighting="uniform")
    boltz = hist(conformer_weighting="boltzmann", boltzmann_temperature=300.0)
    boltz_again = hist(conformer_weighting="boltzmann", boltzmann_temperature=300.0)

    # Determinism: identical inputs -> identical output.
    assert boltz == boltz_again

    # Boltzmann skews the mix toward the lowest-energy conformer: with a modest
    # spread every conformer still keeps slots, and conformer 0 dominates.
    assert boltz[0] >= uni[0]
    assert boltz[0] > boltz[3], f"Boltzmann should favor low-E conformer: {dict(boltz)}"
    assert all(boltz[c] >= 1 for c in range(n_conformers))


def test_boltzmann_weighting_falls_back_to_uniform_without_energies():
    """Without conformer_energies the per-conformer allocation is the uniform one."""
    from collections import Counter

    from metalsurfer.placement.policy import _weighted_conformer_order

    # No energies => resolve_conformer_weights returns None => the parent
    # stratified draw is used unchanged, i.e. the conformer mix is whatever the
    # seeded prior produces (deterministic, not energy-skewed).
    specs = [
        PlacementSpec(
            conformer_index=ci,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=0,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=0,
        )
        for ci in (list(range(3)) * 40)
    ]
    # Explicit uniform interleave (the weighting applied when weights are equal).
    ordered = _weighted_conformer_order(specs, [1.0, 1.0, 1.0], 120)
    hist = Counter(s.conformer_index for s in ordered)
    assert hist == {0: 40, 1: 40, 2: 40}


def test_boltzmann_weights_helper():
    """Unit checks for the Boltzmann weight resolver and largest-remainder interleave."""
    from metalsurfer.placement.policy import (
        _boltzmann_weights,
        _weighted_conformer_order,
    )

    # Finite, spread energies -> decreasing weights, low E highest.
    w = _boltzmann_weights([0.0, 0.05, 0.1], 300.0)
    assert w is not None
    assert w[0] > w[1] > w[2] > 0.0

    # Degenerate (all equal) -> uniform fallback (None).
    assert _boltzmann_weights([0.0, 0.0, 0.0], 300.0) is None
    # Single finite entry -> None (needs >= 2).
    assert _boltzmann_weights([0.0], 300.0) is None
    # Non-positive temperature -> None.
    assert _boltzmann_weights([0.0, 0.1], 0.0) is None
    # Non-finite energies get weight 0 (not dropped), finite still weighted.
    w2 = _boltzmann_weights([float("nan"), 0.0, 0.2], 300.0)
    assert w2 is not None and w2[0] == 0.0 and w2[1] > w2[2] > 0.0

    # Largest-remainder interleave: prefix histograms stay proportional.
    specs = [
        PlacementSpec(
            conformer_index=ci,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=0,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=0,
        )
        for ci in (list(range(3)) * 10)
    ]
    from collections import Counter

    ordered = _weighted_conformer_order(specs, [1.0, 0.5, 0.0], 13)
    # Conformer 2 has weight 0 -> never selected; counts sum to limit (13).
    assert Counter(s.conformer_index for s in ordered) == {0: 9, 1: 4}

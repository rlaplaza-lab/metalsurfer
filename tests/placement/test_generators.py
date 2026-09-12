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
    generate_placements_from_specs,
)
from metalsurfer.placement.site_context import SiteContext
from metalsurfer.surface_prep import SlabContainer, deposit_adatoms

from ..conftest import (
    make_placement_descriptor,
    make_slab,
    make_water,
)


def test_enumerate_specs_empty_sites_returns_empty():
    """No sites / use_sites=False must not invent site_index=-1 capacity."""
    from metalsurfer.placement.generators import (
        enumerate_placement_specs,
        estimate_placement_spec_capacity,
    )

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
    # No spec may target the blocked site (-1 is the "no site" sentinel, which
    # would also silently bypass the block if emitted here).
    assert all(int(s.site_index) != blocked for s in specs)
    assert all(int(s.site_index) >= 0 for s in specs)


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


def test_generate_placements_from_specs_all_cached_skips_pose_cache(monkeypatch):
    from metalsurfer.placement import generators as gen_mod

    from ..conftest import make_placement_descriptor

    def _spec(index: int) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=index,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=index,
        )

    specs = [_spec(0), _spec(0)]
    cached_adsorbate = Atoms("H")
    descriptor = make_placement_descriptor(placement_id=0)
    cache = {0: (cached_adsorbate, descriptor)}
    build_calls = []
    generate_calls = []

    def fake_build_pose_cache(*args, **kwargs):
        build_calls.append((args, kwargs))
        return object()

    def fake_generate(spec, *args, **kwargs):
        generate_calls.append(spec.placement_index)
        return (
            Atoms("H"),
            make_placement_descriptor(placement_id=spec.placement_index),
        ), None

    monkeypatch.setattr(gen_mod, "build_pose_batch_cache", fake_build_pose_cache)
    monkeypatch.setattr(
        gen_mod, "generate_placement_from_spec_with_reason", fake_generate
    )

    for workers in (1, 4):
        build_calls.clear()
        generate_calls.clear()
        config = AdsorptionConfig(
            material_type="slab", placement_materialize_workers=workers
        )
        results = generate_placements_from_specs(
            specs,
            [make_water()],
            make_slab(),
            config,
            materialization_cache=cache,
        )

        assert build_calls == []
        assert generate_calls == []
        assert len(results) == 2
        for result, reason in results:
            assert reason is None
            assert result is not None
            adsorbate, returned_descriptor = result
            assert adsorbate is not cached_adsorbate
            assert returned_descriptor is descriptor


def test_generate_placements_from_specs_builds_pose_cache_once_for_misses(
    monkeypatch,
):
    from metalsurfer.placement import generators as gen_mod

    from ..conftest import make_placement_descriptor

    def _spec(index: int) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=index,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=index,
        )

    specs = [_spec(index) for index in range(4)]
    slab = make_slab()
    site_slab = make_slab(nx=1, ny=1, n_layers=1)
    conformers = [make_water()]
    config = AdsorptionConfig(material_type="slab", placement_materialize_workers=4)
    hit_adsorbate = Atoms("H")
    hit_descriptor = make_placement_descriptor(placement_id=3)
    cache = {
        0: (hit_adsorbate, make_placement_descriptor(placement_id=0)),
        2: None,
        3: (hit_adsorbate.copy(), hit_descriptor),
    }
    pose_cache = object()
    build_calls = []
    generate_calls = {}

    def fake_build_pose_cache(*args, **kwargs):
        build_calls.append((args, kwargs))
        return pose_cache

    def fake_generate(spec, _conformers, passed_slab, passed_config, **kwargs):
        generate_calls[spec.placement_index] = (
            passed_slab,
            kwargs.get("slab_for_sites"),
            kwargs.get("pose_cache"),
        )
        if spec.placement_index == 1:
            return None, "too_close"
        if spec.placement_index == 2:
            return None, "invalid_site_index"
        return (
            Atoms("H"),
            make_placement_descriptor(placement_id=spec.placement_index),
        ), None

    monkeypatch.setattr(gen_mod, "build_pose_batch_cache", fake_build_pose_cache)
    monkeypatch.setattr(
        gen_mod, "generate_placement_from_spec_with_reason", fake_generate
    )

    results = generate_placements_from_specs(
        specs,
        conformers,
        slab,
        config,
        slab_for_sites=site_slab,
        materialization_cache=cache,
    )

    assert len(build_calls) == 1
    assert build_calls[0][0][0] is site_slab
    assert build_calls[0][0][1] is conformers
    assert build_calls[0][0][2] is config
    assert set(generate_calls) == {1, 2}
    for passed_slab, passed_site_slab, passed_pose_cache in generate_calls.values():
        assert passed_slab is slab
        assert passed_site_slab is site_slab
        assert passed_pose_cache is pose_cache
    assert [reason for _result, reason in results] == [
        None,
        "too_close",
        "invalid_site_index",
        None,
    ]
    assert results[0][0][0] is not hit_adsorbate
    assert results[0][0][1] is cache[0][1]
    assert results[3][0][1] is hit_descriptor


def test_generate_placements_from_specs_without_cache_builds_pose_cache(monkeypatch):
    from metalsurfer.placement import generators as gen_mod

    def _spec(index: int) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=index,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=index,
        )

    specs = [_spec(0)]
    slab = make_slab()
    conformers = [make_water()]
    config = AdsorptionConfig(material_type="slab")

    for materialization_cache in (None, {}):
        pose_cache = object()
        build_calls = []
        generate_calls = []

        def fake_build_pose_cache(
            *args,
            _build_calls=build_calls,
            _pose_cache=pose_cache,
            **kwargs,
        ):
            _build_calls.append((args, kwargs))
            return _pose_cache

        def fake_generate(
            spec,
            *_args,
            _generate_calls=generate_calls,
            **kwargs,
        ):
            _generate_calls.append(kwargs)
            return (
                Atoms("H"),
                make_placement_descriptor(placement_id=spec.placement_index),
            ), None

        monkeypatch.setattr(gen_mod, "build_pose_batch_cache", fake_build_pose_cache)
        monkeypatch.setattr(
            gen_mod, "generate_placement_from_spec_with_reason", fake_generate
        )

        results = generate_placements_from_specs(
            specs,
            conformers,
            slab,
            config,
            materialization_cache=materialization_cache,
        )

        assert len(build_calls) == 1
        assert build_calls[0][0][0] is slab
        assert build_calls[0][0][1] is conformers
        assert build_calls[0][0][2] is config
        assert generate_calls[0]["pose_cache"] is pose_cache
        assert results[0][0] is not None
        assert results[0][1] is None


def test_generate_placements_from_specs_empty_skips_cache_and_pose_cache(
    monkeypatch,
):
    from metalsurfer.placement import generators as gen_mod

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("empty specs must return before cache work")

    monkeypatch.setattr(gen_mod, "build_pose_batch_cache", fail_if_called)
    monkeypatch.setattr(
        gen_mod, "generate_placement_from_spec_with_reason", fail_if_called
    )

    assert (
        generate_placements_from_specs(
            [],
            [make_water()],
            make_slab(),
            AdsorptionConfig(),
            materialization_cache={},
        )
        == []
    )


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
    pytest.importorskip("rdkit", reason="RDKit required for conformer generation")
    result = create_conformers_from_smiles(
        "O", config=AdsorptionConfig(num_conformers=1)
    )
    assert result is not None
    conformers, _ = result

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

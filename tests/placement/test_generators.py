"""Placement spec enumeration and materialization."""

import logging

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.models import PlacementSpec
from metalsurfer.placement.clash import atom_radii_for_symbols
from metalsurfer.placement.generators import (
    _adsorbate_atoms_on_substrate,
    _enumerate_covered_specs,
    _SpecGridInfo,
    enumerate_placement_specs,
    generate_placement_from_spec,
    generate_placements_from_specs,
    log_site_sample,
)
from metalsurfer.placement.geometry import occupied_neighbor_counts
from metalsurfer.placement.site_context import SiteContext
from metalsurfer.placement.site_types import Site, with_symmetry
from metalsurfer.surface_prep import SlabContainer, deposit_adatoms

from ..conftest import (
    make_placement_descriptor,
    make_slab,
    make_water,
)
from ._helpers import _make_site, _round_atop_placement_spec, dissoc_placement_spec


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

    def fake_generate(spec, _conformers, passed_slab, _passed_config, **kwargs):
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
        config=AdsorptionConfig(),
    )
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=8,
        placement_z_range=(2.0, 3.0),
    )
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


def test_occupied_neighbor_counts_reach_the_next_lattice_shell():
    """A line graph counts the neighbor and the next-nearest, not a farther site."""
    positions = {index: np.array([float(index), 0.0, 0.0]) for index in range(5)}
    cell = np.diag([40.0, 40.0, 40.0])
    counts = occupied_neighbor_counts(positions, {0}, cell, [False, False, False])
    assert counts[1] == 1
    assert counts[2] == 1
    assert counts[3] == 0
    assert counts[0] == 0


def test_occupied_neighbor_counts_ignore_bridges_between_hollows():
    """Two hollow steps stay a hollow shell, and 0.0001 Å does not split it.

    Bridges sitting between hollows are a different binding mode. They can
    mark a bridge as an edge, and they do not add to the hollow count.
    """
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([2.0, 0.0, 0.0]),
        3: np.array([3.0, 0.0, 0.0]),
        4: np.array([4.0, 0.0, 0.0]),
    }
    site_types = {0: "hollow", 1: "bridge", 2: "hollow", 3: "bridge", 4: "hollow"}
    cell = np.diag([40.0, 40.0, 40.0])
    pbc = [False, False, False]
    counts = occupied_neighbor_counts(positions, {0}, cell, pbc, site_types=site_types)
    assert counts[0] == 0
    assert counts[2] == 1
    assert counts[4] == 1
    assert counts[1] == 1
    assert counts[3] == 0

    jittered = {index: xyz.copy() for index, xyz in positions.items()}
    jittered[2] = jittered[2] + np.array([1e-4, 0.0, 0.0])
    moved = occupied_neighbor_counts(jittered, {0}, cell, pbc, site_types=site_types)
    assert moved == counts


def test_covered_short_draw_keeps_both_hollows_and_an_edge():
    """Eight slots cover both tied hollows and one positive-count site once."""
    sites = [
        _make_site((0.0, 0.0, 0.0), site_type="atop"),
        _make_site((1.0, 0.0, 0.0), site_type="bridge"),
        _make_site((2.0, 0.0, 0.0), site_type="hollow"),
        _make_site((3.0, 0.0, 0.0), site_type="hollow"),
        _make_site((4.0, 0.0, 0.0), site_type="atop"),
    ]
    info = _sample_grid(
        use_sites=True,
        unique_sites=sites,
        site_indices=[0, 1, 2, 3, 4],
        clearances=np.array([5.0, 5.0, 5.0, 5.0, 1.0]),
        occupied_neighbor_counts={0: 0, 1: 0, 2: 0, 3: 0, 4: 1},
        n_clustered=len(sites),
    )
    specs = _enumerate_covered_specs(
        info,
        n_desired=8,
        parallel_fraction=0.0,
        filter_spec=None,
        seed=7,
        conformer_energies=None,
        conformer_weighting="uniform",
        boltzmann_temperature=300.0,
        n_conformers=1,
    )
    assert len(specs) == 8
    assert {spec.site_index for spec in specs if spec.site_type == "hollow"} == {2, 3}
    assert any(spec.site_index == 4 for spec in specs)
    identities = [
        (
            spec.site_index,
            spec.conformer_index,
            spec.orientation_type,
            spec.tilt_deg,
            spec.azimuth_deg,
            spec.azimuth_in_plane_deg,
            round(float(spec.z_fraction), 5),
        )
        for spec in specs
    ]
    assert len(identities) == len(set(identities))


def test_floating_chain_does_not_occupy_the_surface():
    """Atoms outside the metal-bond cutoff leave every site open."""
    slab = make_slab()
    top = float(np.max(np.asarray(slab.positions)[:, 2]))
    radii = atom_radii_for_symbols(["O", "C", "C"])
    positions = np.array(
        [
            [0.0, 0.0, top + 6.0],
            [1.2, 0.0, top + 8.0],
            [1.2, 0.0, top + 10.0],
        ]
    )
    kept_pos, kept_radii = _adsorbate_atoms_on_substrate(
        positions,
        radii,
        slab,
        cell=np.asarray(slab.cell),
        pbc=list(slab.pbc),
        connectivity_multiplier=1.3,
    )
    assert len(kept_pos) == 0
    assert len(kept_radii) == 0


def test_upright_tail_does_not_thicken_the_footprint():
    """The bound head is kept. A carbon two bonds above it is not."""
    slab = make_slab()
    positions_slab = np.asarray(slab.positions)
    anchor_index = int(np.argmax(positions_slab[:, 2]))
    anchor = positions_slab[anchor_index]
    oxygen_radius = float(atom_radii_for_symbols(["O"])[0])
    carbon_radius = float(atom_radii_for_symbols(["C"])[0])
    metal_radius = float(
        atom_radii_for_symbols([slab.get_chemical_symbols()[anchor_index]])[0]
    )
    oxygen = anchor + np.array([0.0, 0.0, 1.05 * (oxygen_radius + metal_radius)])
    carbon = anchor + np.array([0.4, 0.0, 2.5 * (carbon_radius + metal_radius)])
    kept_pos, _kept_radii = _adsorbate_atoms_on_substrate(
        np.vstack([oxygen, carbon]),
        np.array([oxygen_radius, carbon_radius]),
        slab,
        cell=np.asarray(slab.cell),
        pbc=list(slab.pbc),
        connectivity_multiplier=1.3,
    )
    assert len(kept_pos) == 1
    assert np.allclose(kept_pos[0], oxygen)


def test_enumerate_modes_share_bare_slab_prefix():
    """Normal, sequential, and n-tuple draws match on a fixed site catalog."""
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    slab = make_slab()
    conformers = [make_water()]
    n_desired = 24
    seed = 7
    # Share one clustered catalog so n-tuple symmetry rebuild cannot fork the draw.
    site_context = resolve_site_context_for_sampling(
        slab,
        AdsorptionConfig(
            material_type="slab",
            seed=seed,
            num_placements=n_desired,
            saturation_molecules_per_step=2,
        ),
        symmetry_broken=True,
    )

    def _draw(**overrides: object) -> list[PlacementSpec]:
        config = AdsorptionConfig(
            material_type="slab",
            seed=seed,
            num_placements=n_desired,
            **overrides,
        )
        return enumerate_placement_specs(
            conformers,
            slab,
            config,
            "O",
            n_desired=n_desired,
            site_context=site_context,
        )

    normal = _draw()
    sequential = _draw(saturation_max_steps=2)
    ntuplet = _draw(saturation_molecules_per_step=2, saturation_max_steps=1)
    assert normal

    def _key(specs: list[PlacementSpec]) -> list[tuple]:
        return [
            (
                s.site_index,
                round(float(s.z_fraction), 5),
                float(s.azimuth_deg),
                s.tilt_deg,
            )
            for s in specs
        ]

    assert _key(normal) == _key(sequential) == _key(ntuplet)


def _sample_grid(**overrides: object) -> _SpecGridInfo:
    base: dict[str, object] = {
        "is_dissociative": False,
        "unique_sites": [],
        "use_sites": False,
        "site_indices": [],
        "shape": "round",
        "symbols": ["O"],
        "n_binders": 1,
        "flat_aromatic": False,
        "n_hollow_pairs": 0,
        "site_source": "no_sites",
        "n_clustered": 0,
    }
    base.update(overrides)
    return _SpecGridInfo(**base)


def _orbit(site_type: str, multiplicity: int) -> Site:
    return with_symmetry(
        _make_site((0.0, 0.0, 0.0), site_type=site_type),
        symmetry_multiplicity=multiplicity,
        symmetry_equivalent_sites=(),
    )


def _log_lines(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    return [
        record.message
        for record in caplog.records
        if record.name == "metalsurfer.placement.generators" and record.levelno == level
    ]


def test_log_site_sample_omits_percent_when_no_sites(caplog) -> None:
    with caplog.at_level(logging.DEBUG, logger="metalsurfer.placement.generators"):
        log_site_sample(_sample_grid(), 0)
    assert _log_lines(caplog, logging.INFO) == ["Site sample: 0 placements, 0 sites"]
    assert _log_lines(caplog, logging.DEBUG) == [
        "Site catalog (no_sites): 0 open, 0 catalog"
    ]


def test_log_site_sample_dissociative_pairs(caplog) -> None:
    specs = [
        dissoc_placement_spec(placement_index=i, site_index=i % 12) for i in range(8)
    ]
    info = _sample_grid(
        is_dissociative=True,
        n_hollow_pairs=12,
        site_source="clustered",
        n_clustered=12,
    )
    with caplog.at_level(logging.DEBUG, logger="metalsurfer.placement.generators"):
        log_site_sample(info, 8, specs=specs)
    assert _log_lines(caplog, logging.INFO) == [
        "Site sample: 8 placements, 12 hollow pairs (67%)"
    ]
    assert _log_lines(caplog, logging.DEBUG) == [
        "Site catalog (clustered): 12 hollow pairs, hollow 12 (sampled 8)"
    ]


def test_log_site_sample_symmetry_reduced_breakdown(caplog) -> None:
    sites = [
        *[_orbit("hollow", 4) for _ in range(4)],
        *[_orbit("bridge", 2) for _ in range(5)],
        *[_orbit("atop", 2) for _ in range(3)],
    ]
    specs = [
        *[_round_atop_placement_spec(i, site_type="hollow") for i in range(6)],
        *[_round_atop_placement_spec(i, site_type="bridge") for i in range(6, 12)],
        *[_round_atop_placement_spec(i, site_type="atop") for i in range(12, 16)],
    ]
    info = _sample_grid(
        unique_sites=sites,
        use_sites=True,
        site_indices=list(range(len(sites))),
        site_source="symmetry_aware",
        n_clustered=32,
    )
    with caplog.at_level(logging.DEBUG, logger="metalsurfer.placement.generators"):
        log_site_sample(info, 16, specs=specs)
    assert _log_lines(caplog, logging.INFO) == [
        "Site sample: 16 placements, 12 sites (133%)"
    ]
    assert _log_lines(caplog, logging.DEBUG) == [
        "Site catalog (symmetry_aware): 12 open, 12 catalog, "
        "12 orbits (multiplicity 32, clustered 32), "
        "hollow 4 (sampled 6), bridge 5 (sampled 6), atop 3 (sampled 4)"
    ]


def test_log_site_sample_open_subset_and_catalog_only(caplog) -> None:
    sites = [
        *[_make_site((float(i), 0.0, 0.0), site_type="hollow") for i in range(20)],
        *[_make_site((float(i), 1.0, 0.0), site_type="bridge") for i in range(18)],
        *[_make_site((float(i), 2.0, 0.0), site_type="atop") for i in range(10)],
    ]
    info = _sample_grid(
        unique_sites=sites,
        use_sites=True,
        site_indices=list(range(31)),
        site_source="clustered_under_coverage",
        n_clustered=48,
    )
    with caplog.at_level(logging.DEBUG, logger="metalsurfer.placement.generators"):
        log_site_sample(info, 16)
    assert _log_lines(caplog, logging.INFO) == [
        "Site sample: 16 placements, 31 sites (52%)"
    ]
    assert _log_lines(caplog, logging.DEBUG) == [
        "Site catalog (clustered_under_coverage): 31 open, 48 catalog, "
        "hollow 20, bridge 11"
    ]

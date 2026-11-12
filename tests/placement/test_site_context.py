"""Site-context caching and surface references."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.features import extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.placement.generators import (
    _spec_grid_info,
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
    generate_placement_from_spec,
    generate_placements_from_specs,
)
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _get_unique_sites_for_specs,
    resolve_site_context_for_sampling,
    site_context_for_sampling,
    skip_symmetry_for_sampling,
)
from metalsurfer.placement.site_coords import _wrapped_slab_fractional
from metalsurfer.placement.site_enumeration import get_hollow_sites_for_adatoms

from ..conftest import (
    adsorption_config_factory,
    make_nanoparticle,
    make_placement_descriptor,
    make_slab,
    make_water,
    water_conformers,
)
from ._helpers import (
    _first_successful_placement,
    _tilted_make_slab,
)


def test_site_context_cache_keys_differ_by_symmetry_broken():
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig()

    ctx_broken = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=True, material_type="slab"
    )
    ctx_intact = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    # Unique-sites + sym=True + sym=False.
    assert len(_SITE_CONTEXT_CACHE) == 3
    assert ctx_broken is not ctx_intact


def test_unique_sites_cache_key_includes_atoms_pbc():
    """Every generator hashes atoms PBC; TTF vs TTT are distinct keys."""
    from metalsurfer.placement.site_context import (
        _get_unique_sites_for_specs,
        _unique_sites_cache_key,
    )

    slab_mat = make_slab(nx=2, ny=2)
    slab_mat.set_pbc([True, True, False])
    slab_calc = slab_mat.copy()
    slab_calc.set_pbc([True, True, True])
    config = AdsorptionConfig()

    assert _unique_sites_cache_key(
        slab_mat, config, material_type="nanoparticle"
    ) != _unique_sites_cache_key(
        slab_calc,
        config,
        material_type="nanoparticle",
    )
    _get_unique_sites_for_specs(slab_mat, config, material_type="slab")
    assert len(_SITE_CONTEXT_CACHE) == 1

    # Different material_type must still split the cache.
    np_config = AdsorptionConfig()
    assert _unique_sites_cache_key(
        slab_mat, config, material_type="slab"
    ) != _unique_sites_cache_key(
        slab_mat,
        np_config,
        material_type="nanoparticle",
    )


def test_unique_sites_cache_key_atoms_pbc_for_all_plugins():
    """Topology, adaptive_grid, and rolling_probe all key on atoms PBC."""
    from metalsurfer.placement.site_context import _unique_sites_cache_key

    slab_tt = make_slab(nx=2, ny=2)
    slab_tt.set_pbc([True, True, False])
    slab_ttt = slab_tt.copy()
    slab_ttt.set_pbc([True, True, True])
    for plugin in ("topology", "adaptive_grid", "rolling_probe"):
        cfg = AdsorptionConfig(site_generator=plugin)
        assert _unique_sites_cache_key(
            slab_tt, cfg, material_type="slab"
        ) != _unique_sites_cache_key(
            slab_ttt,
            cfg,
            material_type="slab",
        )


def test_unique_sites_cache_key_includes_site_generator():
    from metalsurfer.placement.site_context import _unique_sites_cache_key

    slab = make_slab(nx=2, ny=2)
    auto = AdsorptionConfig(site_generator="auto")
    topology = AdsorptionConfig(site_generator="topology")
    voronoi = AdsorptionConfig(site_generator="voronoi")
    assert _unique_sites_cache_key(
        slab, auto, material_type="slab"
    ) != _unique_sites_cache_key(
        slab,
        topology,
        material_type="slab",
    )
    assert _unique_sites_cache_key(
        slab, topology, material_type="slab"
    ) != _unique_sites_cache_key(
        slab,
        voronoi,
        material_type="slab",
    )


def test_side_policy_cache_key_for_wall_near_plugins():
    """side_policy splits the cache for adaptive_grid and rolling_probe, not topology."""
    from metalsurfer.placement.site_context import _unique_sites_cache_key

    slab = make_slab(nx=2, ny=2)
    topo_pos = AdsorptionConfig(site_generator="topology", side_policy="positive")
    topo_all = AdsorptionConfig(site_generator="topology", side_policy="all")
    assert _unique_sites_cache_key(
        slab, topo_pos, material_type="slab"
    ) == _unique_sites_cache_key(
        slab,
        topo_all,
        material_type="slab",
    )
    for plugin in ("adaptive_grid", "rolling_probe"):
        pos = AdsorptionConfig(site_generator=plugin, side_policy="positive")
        all_sides = AdsorptionConfig(site_generator=plugin, side_policy="all")
        assert _unique_sites_cache_key(
            slab, pos, material_type="slab"
        ) != _unique_sites_cache_key(
            slab,
            all_sides,
            material_type="slab",
        )


def test_extract_features_depends_only_on_absolute_geometry():
    record = PlacementRecord.from_descriptor(
        make_placement_descriptor(
            placement_id=1,
            x_abs=1.25,
            y_abs=2.5,
            z_abs=7.75,
            quat_w=0.9,
            quat_x=0.1,
            quat_y=0.2,
            quat_z=0.3,
        ),
        molecule="water",
        smiles="O",
        surface_id="test",
        material_type="slab",
    )
    record.descriptor.site_index = 99
    record.descriptor.surface_ref_z_abs = 0.0
    record.descriptor.z_offset = 99.0
    features = extract_features(record)
    assert set(features.keys()) == {
        "x",
        "y",
        "z",
        "conformer_index",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    }
    # COM uses absolute surface-frame coordinates; fractional provenance is ignored.
    assert features["x"] == pytest.approx(1.25)
    assert features["y"] == pytest.approx(2.5)
    assert features["z"] == pytest.approx(7.75)


def test_site_context_cache_key_includes_config_and_symmetry():
    from metalsurfer.placement.site_context import (
        _site_context_cache_key,
        resolve_site_context_for_sampling,
    )

    slab = make_slab()
    c1 = AdsorptionConfig(voronoi_probe_radius=1.0)
    c2 = AdsorptionConfig(voronoi_probe_radius=1.5)
    k1 = _site_context_cache_key(slab, c1, symmetry_broken=False, material_type="slab")
    k2 = _site_context_cache_key(slab, c1, symmetry_broken=True, material_type="slab")
    k3 = _site_context_cache_key(slab, c2, symmetry_broken=False, material_type="slab")
    assert k1 != k2
    assert k1 != k3
    single = AdsorptionConfig(saturation_molecules_per_step=1)
    tuplet = AdsorptionConfig(saturation_molecules_per_step=2)
    assert _site_context_cache_key(
        slab,
        single,
        symmetry_broken=False,
        material_type="slab",
    ) != _site_context_cache_key(
        slab, tuplet, symmetry_broken=False, material_type="slab"
    )
    full = resolve_site_context_for_sampling(
        slab, tuplet, symmetry_broken=False, material_type="slab"
    )
    assert full.clustered_sites is not None
    assert full.source != "symmetry_aware"
    assert len(full.sites) == len(full.clustered_sites)
    a = resolve_site_context_for_sampling(
        slab, c1, symmetry_broken=False, material_type="slab"
    )
    b = resolve_site_context_for_sampling(
        slab, c1, symmetry_broken=False, material_type="slab"
    )
    assert a is b


def test_site_context_cache_key_includes_species_and_symmetry_tol():
    from metalsurfer.placement.site_context import (
        _site_context_cache_key,
        _unique_sites_cache_key,
    )

    cu = make_slab(symbol="Cu")
    ni = make_slab(symbol="Ni")
    # Same lattice geometry, different chemistry.
    ni.set_cell(cu.get_cell(), scale_atoms=False)
    ni.set_positions(cu.get_positions())
    cfg = AdsorptionConfig()
    assert _unique_sites_cache_key(
        cu, cfg, material_type="slab"
    ) != _unique_sites_cache_key(ni, cfg, material_type="slab")

    c_loose = AdsorptionConfig(symmetry_tolerance=0.05)
    c_tight = AdsorptionConfig(symmetry_tolerance=0.01)
    assert _site_context_cache_key(
        cu,
        c_loose,
        symmetry_broken=False,
        material_type="slab",
    ) != _site_context_cache_key(
        cu, c_tight, symmetry_broken=False, material_type="slab"
    )


def test_site_context_cache_key_float_packing_no_collision():
    from metalsurfer.placement.site_context import (
        _pack_optional_float,
        _unique_sites_cache_key,
    )

    # Naive str concat collides for these triples; structured packing must not.
    assert f"{1.5}{20.0}{0.5}" == f"{1.52}{0.0}{0.5}"
    packed_a = (
        _pack_optional_float(1.5)
        + _pack_optional_float(20.0)
        + _pack_optional_float(0.5)
    )
    packed_b = (
        _pack_optional_float(1.52)
        + _pack_optional_float(0.0)
        + _pack_optional_float(0.5)
    )
    assert packed_a != packed_b

    slab = make_slab()
    a = AdsorptionConfig(
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
    )
    b = AdsorptionConfig(
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.6,
    )
    assert _unique_sites_cache_key(
        slab, a, material_type="slab"
    ) != _unique_sites_cache_key(slab, b, material_type="slab")

    c = AdsorptionConfig(
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
        planar_z_variance_threshold=0.01,
    )
    d = AdsorptionConfig(
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
        planar_z_variance_threshold=0.05,
    )
    assert _unique_sites_cache_key(
        slab, c, material_type="slab"
    ) != _unique_sites_cache_key(slab, d, material_type="slab")


def test_surface_reference_uses_prefix_not_symbols():
    from metalsurfer.workflow.shared import _build_surface_reference_slab

    base = make_slab(symbol="Ru")
    # Same-element adatom appended as suffix.
    decorated = base.copy()
    decorated.extend(Atoms("Ru", positions=[[1.0, 1.0, 10.0]]))
    ref = _build_surface_reference_slab(decorated, base, material_type="slab")
    assert len(ref) == len(base)
    assert len(ref) == len(decorated) - 1


def test_tilted_slab_site_xy_frac_uses_full_3d_projection():
    """Descriptor frac a/b are crystal fractions of the full COM, not [x, y, 0]."""
    slab = _tilted_make_slab()
    config = adsorption_config_factory(num_placements=20, placement_z_range=(2.0, 3.0))
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    _adsorbate, descriptor = result
    assert descriptor.z_abs is not None
    cell = np.asarray(slab.get_cell(), dtype=float)
    full = np.array(
        [descriptor.x_abs, descriptor.y_abs, float(descriptor.z_abs)], dtype=float
    )
    expected = _wrapped_slab_fractional(full, cell)[..., :2]
    zeroed = _wrapped_slab_fractional(
        np.array([descriptor.x_abs, descriptor.y_abs, 0.0], dtype=float), cell
    )[..., :2]
    assert not np.allclose(expected, zeroed, atol=1e-6)
    assert descriptor.site_xy_frac_a == pytest.approx(float(expected[0]), abs=1e-9)
    assert descriptor.site_xy_frac_b == pytest.approx(float(expected[1]), abs=1e-9)
    assert descriptor.placement_mode_resolved == "sites"


def _site_xyz_type_key(site) -> tuple:
    xyz = np.asarray(site.xyz, dtype=float)
    return (
        round(float(xyz[0]), 6),
        round(float(xyz[1]), 6),
        round(float(xyz[2]), 6),
        str(site.site_type),
    )


def test_hollow_sites_match_clustered_catalog_slab_and_np():
    """Adatom hollows are the hollow/pore subset of clustered unique sites."""
    for material_type, structure in (
        ("slab", make_slab(nx=3, ny=3)),
        ("nanoparticle", make_nanoparticle()),
    ):
        config = AdsorptionConfig()
        core = _get_unique_sites_for_specs(
            structure, config, material_type=material_type
        )
        assert core.clustered_sites is not None
        expected = {
            _site_xyz_type_key(s)
            for s in core.clustered_sites
            if s.site_type in ("hollow", "pore")
        }
        hollows = get_hollow_sites_for_adatoms(
            structure,
            material_type=material_type,
            site_equivalence_tolerance=config.site_equivalence_tolerance,
        )
        assert {_site_xyz_type_key(s) for s in hollows} == expected


def test_enumerate_without_context_matches_resolved_sampling_catalog():
    """Omitting site_context must still sample the symmetry-aware catalog."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(seed=0)
    resolved = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert resolved.use_sites and resolved.sites
    info = _spec_grid_info(
        [make_water()], slab, config, "O", site_context=None, material_type="slab"
    )
    assert [_site_xyz_type_key(s) for s in info.unique_sites] == [
        _site_xyz_type_key(s) for s in resolved.sites
    ]


def test_shared_site_frame_across_molecules_keeps_site_index():
    """Competitive saturation: one catalog; per-pack footprint views differ."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(seed=0)
    shared = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert shared.use_sites and len(shared.sites) >= 2

    water_pack = [make_water(), make_water()]
    # Second "molecule": same water geometry scaled up in-plane → larger disk.
    big = make_water()
    big_pos = big.get_positions().copy()
    com = big_pos.mean(axis=0)
    big_pos = com + 3.0 * (big_pos - com)
    big.set_positions(big_pos)
    big_pack = [big, big.copy()]

    info_w = _spec_grid_info(
        water_pack, slab, config, "O", site_context=shared, material_type="slab"
    )
    info_b = _spec_grid_info(
        big_pack, slab, config, "O", site_context=shared, material_type="slab"
    )
    # Shared frame: identical site_index catalog identity.
    assert [_site_xyz_type_key(s) for s in info_w.unique_sites] == [
        _site_xyz_type_key(s) for s in shared.sites
    ]
    assert [_site_xyz_type_key(s) for s in info_b.unique_sites] == [
        _site_xyz_type_key(s) for s in shared.sites
    ]
    # Bare slab: no occupancy prune; both packs see all sites.
    assert info_w.site_indices == info_b.site_indices


def test_symmetry_aware_context_preserves_clustered_sites():
    """Sampling sites shrink under symmetry; clustered_sites stay full."""
    slab = make_slab(nx=3, ny=3)
    config = AdsorptionConfig()
    ctx = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert ctx.clustered_sites is not None
    assert ctx.source == "symmetry_aware"
    assert len(ctx.clustered_sites) > len(ctx.sites)


def test_sampling_rebuilds_catalog_when_substrate_symmetry_drops():
    """A pruned context must not supply sites after the substrate moves."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig()
    reduced = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert reduced.source == "symmetry_aware"
    assert reduced.clustered_sites is not None
    old_keys = [_site_xyz_type_key(s) for s in reduced.clustered_sites]

    moved = slab.copy()
    pos = moved.get_positions()
    top = int(np.argmax(pos[:, 2]))
    pos[top, 0] += 0.8
    # The fixture slab freezes every atom; the displacement is geometric.
    moved.set_positions(pos, apply_constraint=False)

    rebuilt = site_context_for_sampling(
        moved,
        config,
        reduced,
        symmetry_broken=True,
        material_type="slab",
    )
    fresh = resolve_site_context_for_sampling(
        moved, config, symmetry_broken=True, material_type="slab"
    )
    assert rebuilt.source != "symmetry_aware"
    assert [_site_xyz_type_key(s) for s in rebuilt.sites] == [
        _site_xyz_type_key(s) for s in fresh.sites
    ]
    assert [_site_xyz_type_key(s) for s in rebuilt.sites] != old_keys


def test_skip_symmetry_for_sampling_when_adsorbate_suffix_present():
    slab = make_slab(nx=2, ny=2)
    covered = slab.copy() + Atoms("H", positions=[[1.0, 1.0, 8.0]])
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=False, slab_for_sites=slab, full_slab=slab
        )
        is False
    )
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=False, slab_for_sites=slab, full_slab=covered
        )
        is True
    )
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=True, slab_for_sites=slab, full_slab=slab
        )
        is True
    )


def test_occupied_surface_samples_full_lattice_and_drops_occupied_sites():
    """After adsorption, sample every clustered site except occupied vertices."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(seed=0)
    reduced = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert reduced.clustered_sites is not None
    assert reduced.source == "symmetry_aware"
    assert len(reduced.clustered_sites) > len(reduced.sites)

    occupied = reduced.clustered_sites[0]
    ads = Atoms("H", positions=[occupied.xyz + np.array([0.0, 0.0, 0.2])])
    full = slab.copy() + ads

    expanded = site_context_for_sampling(
        slab, config, reduced, full_slab=full, material_type="slab"
    )
    assert len(expanded.sites) >= len(reduced.clustered_sites)
    assert [
        _site_xyz_type_key(s) for s in expanded.sites[: len(reduced.clustered_sites)]
    ] == [_site_xyz_type_key(s) for s in reduced.clustered_sites]
    assert (
        site_context_for_sampling(
            slab, config, expanded, full_slab=full, material_type="slab"
        )
        is expanded
    )

    info = _spec_grid_info(
        [make_water()],
        slab,
        config,
        "O",
        site_context=reduced,
        full_slab=full,
        material_type="slab",
    )
    clustered_keys = [_site_xyz_type_key(s) for s in reduced.clustered_sites]
    got_keys = [_site_xyz_type_key(s) for s in info.unique_sites]
    assert got_keys[: len(clustered_keys)] == clustered_keys
    assert any(
        any(int(atom) >= len(slab) for atom in site.slab_indices)
        for site in info.unique_sites[len(clustered_keys) :]
    )
    kept_keys = {_site_xyz_type_key(info.unique_sites[i]) for i in info.site_indices}
    occupied_key = _site_xyz_type_key(occupied)
    assert occupied_key not in kept_keys
    reduced_keys = {_site_xyz_type_key(s) for s in reduced.sites}
    assert kept_keys - reduced_keys, (
        "coverage sampling must include unoccupied orbit copies "
        "that symmetry reduction dropped"
    )

    specs = enumerate_placement_specs(
        [make_water()],
        slab,
        config,
        "O",
        n_desired=32,
        site_context=reduced,
        full_slab=full,
        material_type="slab",
    )
    assert specs
    wide = [
        s
        for s in specs
        if len(reduced.sites) <= s.site_index < len(reduced.clustered_sites)
    ]
    assert wide, "enumerated specs must index into the clustered lattice"
    sampling = site_context_for_sampling(
        slab, config, reduced, full_slab=full, material_type="slab"
    )
    placed = generate_placement_from_spec(
        wide[0],
        [make_water()],
        full,
        config,
        "O",
        site_context=sampling,
        slab_for_sites=slab,
        material_type="slab",
    )
    assert placed is not None


def test_n_tuplet_config_samples_full_lattice_on_clean_slab():
    """Co-adsorption needs distinct orbit copies before anything is adsorbed."""
    slab = make_slab(nx=2, ny=2)
    reduced = resolve_site_context_for_sampling(
        slab, AdsorptionConfig(), symmetry_broken=False, material_type="slab"
    )
    assert reduced.clustered_sites is not None
    assert len(reduced.clustered_sites) > len(reduced.sites)
    tuplet_cfg = AdsorptionConfig(seed=0, saturation_molecules_per_step=2)
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=False,
            slab_for_sites=slab,
            full_slab=slab,
            config=tuplet_cfg,
        )
        is True
    )
    info = _spec_grid_info(
        [make_water()],
        slab,
        tuplet_cfg,
        "O",
        site_context=reduced,
        material_type="slab",
    )
    assert [_site_xyz_type_key(s) for s in info.unique_sites] == [
        _site_xyz_type_key(s) for s in reduced.clustered_sites
    ]
    type_counts: dict[str, int] = {}
    for i in info.site_indices:
        key = str(info.unique_sites[i].site_type)
        type_counts[key] = type_counts.get(key, 0) + 1
    assert max(type_counts.values()) > 1


def test_multi_molecule_coverage_shares_pruned_clustered_catalog():
    """Competing adsorbates see the same expanded lattice minus occupied vertices."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(seed=0)
    reduced = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    assert reduced.clustered_sites is not None
    occupied = reduced.clustered_sites[0]
    full = slab.copy() + Atoms(
        "H", positions=[occupied.xyz + np.array([0.0, 0.0, 0.2])]
    )
    water = make_water()
    oh = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]])
    occupied_key = _site_xyz_type_key(occupied)
    kept = []
    for mol in (water, oh):
        info = _spec_grid_info(
            [mol],
            slab,
            config,
            None,
            site_context=reduced,
            full_slab=full,
            material_type="slab",
        )
        clustered_keys = [_site_xyz_type_key(s) for s in reduced.clustered_sites]
        got_keys = [_site_xyz_type_key(s) for s in info.unique_sites]
        assert got_keys[: len(clustered_keys)] == clustered_keys
        keys = {_site_xyz_type_key(info.unique_sites[i]) for i in info.site_indices}
        assert occupied_key not in keys
        kept.append(keys)
    assert kept[0] == kept[1]


def test_bo_pool_capacity_matches_clustered_catalog_under_coverage():
    """BO enumerate/materialize must use the expanded catalog, not orbit reps."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(seed=0)
    reduced = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False, material_type="slab"
    )
    clustered = resolve_site_context_for_sampling(
        slab, config, symmetry_broken=True, material_type="slab"
    )
    assert reduced.clustered_sites is not None
    occupied = reduced.clustered_sites[0]
    full = slab.copy() + Atoms(
        "H", positions=[occupied.xyz + np.array([0.0, 0.0, 0.2])]
    )
    water = [make_water()]
    cap_from_reduced = estimate_placement_spec_capacity(
        water,
        slab,
        config,
        "O",
        site_context=reduced,
        full_slab=full,
        material_type="slab",
    )
    cap_from_clustered = estimate_placement_spec_capacity(
        water,
        slab,
        config,
        "O",
        site_context=clustered,
        full_slab=full,
        material_type="slab",
    )
    assert cap_from_reduced == cap_from_clustered
    cap_clean = estimate_placement_spec_capacity(
        water, slab, config, "O", site_context=reduced, material_type="slab"
    )
    assert cap_from_reduced != cap_clean

    specs = enumerate_placement_specs(
        water,
        slab,
        config,
        "O",
        n_desired=24,
        site_context=reduced,
        full_slab=full,
        material_type="slab",
    )
    assert specs
    generated = generate_placements_from_specs(
        specs,
        water,
        full,
        config,
        smiles="O",
        site_context=reduced,
        slab_for_sites=slab,
        material_type="slab",
    )
    assert any(result is not None for result, _reason in generated)
    assert any(spec.site_index >= len(reduced.sites) for spec in specs)


def test_unique_sites_context_stores_plugin_vertices():
    _SITE_CONTEXT_CACHE.clear()
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig()
    ctx = _get_unique_sites_for_specs(slab, config, material_type="slab")
    assert ctx.use_sites
    assert ctx.plugin_vertices is not None
    assert ctx.plugin_vertices.ndim == 2
    assert ctx.plugin_vertices.shape[1] == 3
    assert len(ctx.plugin_vertices) > 0
    assert ctx.clustered_sites is not None
    assert len(ctx.clustered_sites) > 0


def test_debug_write_sites_writes_plugin_and_final_overlays(tmp_path, monkeypatch):
    from ase.io import read

    from metalsurfer.surface_prep import Substrate
    from metalsurfer.workflow.shared import _prepare_molecule_screening

    from ..conftest import DummyReferenceEnergies

    _SITE_CONTEXT_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    surface_type = "debug_sites_test"
    slab_atoms = make_slab(nx=4, ny=4)
    water = make_water()
    config = AdsorptionConfig(
        debug_write_sites=True,
        num_placements=2,
        num_conformers=1,
    )
    ctx, failure = _prepare_molecule_screening(
        smiles="O",
        molecule_name="water",
        slab=Substrate(slab_atoms, material_type="slab"),
        calculator=None,
        reference_energies=DummyReferenceEnergies(
            molecule_energies={"water": -10.0},
            conformer_packs={"water": ([water], [0.0])},
        ),
        ts_model=None,
        config=config,
        skip_workload_autotune=True,
        surface_type=surface_type,
    )
    assert failure is None
    assert ctx is not None
    assert ctx.site_context is not None

    xyz_dir = tmp_path / f"results_{surface_type}" / "xyz_structures"
    plugin_path = xyz_dir / "sites_plugin.xyz"
    final_path = xyz_dir / "sites_final.xyz"
    assert plugin_path.exists()
    assert final_path.exists()

    n_sub = len(ctx.slab_for_sites)
    plugin_verts = ctx.site_context.plugin_vertices
    clustered = ctx.site_context.clustered_sites or []
    assert plugin_verts is not None

    plugin_atoms = read(str(plugin_path))
    final_atoms = read(str(final_path))
    assert len(plugin_atoms) == n_sub + len(plugin_verts)
    assert len(final_atoms) == n_sub + len(clustered)
    assert np.all(plugin_atoms.get_atomic_numbers()[n_sub:] == 0)
    assert np.all(final_atoms.get_atomic_numbers()[n_sub:] == 0)


def test_debug_write_sites_step_suffix_from_prepare(tmp_path, monkeypatch):
    from metalsurfer.surface_prep import Substrate
    from metalsurfer.workflow.shared import _prepare_molecule_screening

    from ..conftest import DummyReferenceEnergies

    _SITE_CONTEXT_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    surface_type = "debug_sites_step"
    water = make_water()
    config = AdsorptionConfig(
        debug_write_sites=True,
        num_placements=2,
        num_conformers=1,
    )
    ctx, failure = _prepare_molecule_screening(
        smiles="O",
        molecule_name="water",
        slab=Substrate(make_slab(nx=4, ny=4), material_type="slab"),
        calculator=None,
        reference_energies=DummyReferenceEnergies(
            molecule_energies={"water": -10.0},
            conformer_packs={"water": ([water], [0.0])},
        ),
        ts_model=None,
        config=config,
        skip_workload_autotune=True,
        surface_type=surface_type,
        debug_sites_step=3,
    )
    assert failure is None
    assert ctx is not None
    xyz_dir = tmp_path / f"results_{surface_type}" / "xyz_structures"
    assert (xyz_dir / "sites_plugin_step003.xyz").exists()
    assert (xyz_dir / "sites_final_step003.xyz").exists()
    assert not (xyz_dir / "sites_plugin.xyz").exists()


def test_debug_write_sites_skips_when_site_context_pre_supplied(tmp_path, monkeypatch):
    from metalsurfer.surface_prep import Substrate
    from metalsurfer.workflow.shared import _prepare_molecule_screening

    from ..conftest import DummyReferenceEnergies

    _SITE_CONTEXT_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    surface_type = "debug_sites_presupplied"
    water = make_water()
    slab = make_slab(nx=4, ny=4)
    config = AdsorptionConfig(
        debug_write_sites=True,
        num_placements=2,
        num_conformers=1,
    )
    pre = _get_unique_sites_for_specs(slab, config, material_type="slab")
    ctx, failure = _prepare_molecule_screening(
        smiles="O",
        molecule_name="water",
        slab=Substrate(slab, material_type="slab"),
        calculator=None,
        reference_energies=DummyReferenceEnergies(
            molecule_energies={"water": -10.0},
            conformer_packs={"water": ([water], [0.0])},
        ),
        ts_model=None,
        config=config,
        skip_workload_autotune=True,
        surface_type=surface_type,
        site_context=pre,
        debug_sites_step=1,
    )
    assert failure is None
    assert ctx is not None
    xyz_dir = tmp_path / f"results_{surface_type}" / "xyz_structures"
    assert not (xyz_dir / "sites_plugin_step001.xyz").exists()
    assert not (xyz_dir / "sites_plugin.xyz").exists()


def test_debug_write_sites_off_writes_nothing(tmp_path, monkeypatch):
    from metalsurfer.surface_prep import Substrate
    from metalsurfer.workflow.shared import _prepare_molecule_screening

    from ..conftest import DummyReferenceEnergies

    _SITE_CONTEXT_CACHE.clear()
    monkeypatch.chdir(tmp_path)
    surface_type = "debug_sites_off"
    config = AdsorptionConfig(
        debug_write_sites=False,
        num_placements=2,
        num_conformers=1,
    )
    water = make_water()
    ctx, failure = _prepare_molecule_screening(
        smiles="O",
        molecule_name="water",
        slab=Substrate(make_slab(nx=4, ny=4), material_type="slab"),
        calculator=None,
        reference_energies=DummyReferenceEnergies(
            molecule_energies={"water": -10.0},
            conformer_packs={"water": ([water], [0.0])},
        ),
        ts_model=None,
        config=config,
        skip_workload_autotune=True,
        surface_type=surface_type,
    )
    assert failure is None
    assert ctx is not None
    xyz_dir = tmp_path / f"results_{surface_type}" / "xyz_structures"
    assert not (xyz_dir / "sites_plugin.xyz").exists()
    assert not (xyz_dir / "sites_final.xyz").exists()

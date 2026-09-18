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
    site_context_for_occupied_surface,
    skip_symmetry_for_sampling,
)
from metalsurfer.placement.site_enumeration import get_hollow_sites_for_adatoms
from metalsurfer.workflow import shared as workflow_shared

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
    config = AdsorptionConfig(material_type="slab")

    ctx_broken = workflow_shared.resolve_site_context_for_sampling(
        slab, config, symmetry_broken=True
    )
    ctx_intact = workflow_shared.resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False
    )
    # Unique-sites + sym=True + sym=False.
    assert len(_SITE_CONTEXT_CACHE) == 3
    assert ctx_broken is not ctx_intact


def test_unique_sites_cache_key_uses_material_aware_pbc_not_ase_pbc():
    """ASE calculator PBC vs material PBC must share one unique-sites entry."""
    from metalsurfer.placement.site_context import (
        _get_unique_sites_for_specs,
        _unique_sites_cache_key,
    )

    slab_mat = make_slab(nx=2, ny=2)
    slab_mat.set_pbc([True, True, False])
    slab_calc = slab_mat.copy()
    slab_calc.set_pbc([True, True, True])
    config = AdsorptionConfig(material_type="slab")

    assert _unique_sites_cache_key(slab_mat, config) == _unique_sites_cache_key(
        slab_calc, config
    )
    _get_unique_sites_for_specs(slab_mat, config)
    assert len(_SITE_CONTEXT_CACHE) == 1
    _get_unique_sites_for_specs(slab_calc, config)
    assert len(_SITE_CONTEXT_CACHE) == 1

    # Different material_type must still split the cache.
    np_config = AdsorptionConfig(material_type="nanoparticle")
    assert _unique_sites_cache_key(slab_mat, config) != _unique_sites_cache_key(
        slab_mat, np_config
    )


def test_unique_sites_cache_key_includes_site_generator():
    from metalsurfer.placement.site_context import _unique_sites_cache_key

    slab = make_slab(nx=2, ny=2)
    auto = AdsorptionConfig(material_type="slab", site_generator="auto")
    topology = AdsorptionConfig(material_type="slab", site_generator="topology")
    voronoi = AdsorptionConfig(material_type="slab", site_generator="voronoi")
    assert _unique_sites_cache_key(slab, auto) != _unique_sites_cache_key(
        slab, topology
    )
    assert _unique_sites_cache_key(slab, topology) != _unique_sites_cache_key(
        slab, voronoi
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
    c1 = AdsorptionConfig(material_type="slab", voronoi_probe_radius=1.0)
    c2 = AdsorptionConfig(material_type="slab", voronoi_probe_radius=1.5)
    k1 = _site_context_cache_key(slab, c1, symmetry_broken=False)
    k2 = _site_context_cache_key(slab, c1, symmetry_broken=True)
    k3 = _site_context_cache_key(slab, c2, symmetry_broken=False)
    assert k1 != k2
    assert k1 != k3
    a = resolve_site_context_for_sampling(slab, c1, symmetry_broken=False)
    b = resolve_site_context_for_sampling(slab, c1, symmetry_broken=False)
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
    cfg = AdsorptionConfig(material_type="slab")
    assert _unique_sites_cache_key(cu, cfg) != _unique_sites_cache_key(ni, cfg)

    c_loose = AdsorptionConfig(material_type="slab", symmetry_tolerance=0.05)
    c_tight = AdsorptionConfig(material_type="slab", symmetry_tolerance=0.01)
    assert _site_context_cache_key(
        cu, c_loose, symmetry_broken=False
    ) != _site_context_cache_key(cu, c_tight, symmetry_broken=False)


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
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
    )
    b = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.6,
    )
    assert _unique_sites_cache_key(slab, a) != _unique_sites_cache_key(slab, b)

    c = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
        planar_z_variance_threshold=0.01,
    )
    d = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
        planar_z_variance_threshold=0.05,
    )
    assert _unique_sites_cache_key(slab, c) != _unique_sites_cache_key(slab, d)


def test_surface_reference_uses_prefix_not_symbols():
    from metalsurfer.workflow.shared import _build_surface_reference_slab

    base = make_slab(symbol="Ru")
    # Same-element adatom appended as suffix.
    decorated = base.copy()
    decorated.extend(Atoms("Ru", positions=[[1.0, 1.0, 10.0]]))
    ref = _build_surface_reference_slab(decorated, base)
    assert len(ref) == len(base)
    assert len(ref) == len(decorated) - 1


def test_tilted_slab_site_xy_frac_uses_full_3d_projection():
    """Descriptor frac a/b must project the full COM, not [x, y, 0]."""
    from metalsurfer.placement.site_coords import _slab_plane_projectors

    slab = _tilted_make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    _adsorbate, descriptor = result
    assert descriptor.z_abs is not None
    pinv_ab_T, _ = _slab_plane_projectors(np.asarray(slab.get_cell(), dtype=float))
    full = np.array(
        [descriptor.x_abs, descriptor.y_abs, float(descriptor.z_abs)], dtype=float
    )
    expected = np.mod(full @ pinv_ab_T, 1.0)
    zeroed = np.mod(
        np.array([descriptor.x_abs, descriptor.y_abs, 0.0], dtype=float) @ pinv_ab_T,
        1.0,
    )
    # Bug baseline: zeroing z shifts frac coords on this tilt.
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
        config = AdsorptionConfig(material_type=material_type)
        core = _get_unique_sites_for_specs(structure, config)
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
    config = AdsorptionConfig(material_type="slab", seed=0)
    resolved = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
    assert resolved.use_sites and resolved.sites
    info = _spec_grid_info([make_water()], slab, config, "O", site_context=None)
    assert [_site_xyz_type_key(s) for s in info.unique_sites] == [
        _site_xyz_type_key(s) for s in resolved.sites
    ]


def test_symmetry_aware_context_preserves_clustered_sites():
    """Sampling sites shrink under symmetry; clustered_sites stay full."""
    slab = make_slab(nx=3, ny=3)
    config = AdsorptionConfig(material_type="slab")
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
    assert ctx.clustered_sites is not None
    assert ctx.source == "symmetry_aware"
    assert len(ctx.clustered_sites) > len(ctx.sites)


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
    config = AdsorptionConfig(material_type="slab", seed=0)
    reduced = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
    assert reduced.clustered_sites is not None
    assert reduced.source == "symmetry_aware"
    assert len(reduced.clustered_sites) > len(reduced.sites)

    occupied = reduced.clustered_sites[0]
    ads = Atoms("H", positions=[occupied.xyz + np.array([0.0, 0.0, 0.2])])
    full = slab.copy() + ads

    expanded = site_context_for_occupied_surface(reduced)
    assert expanded.source == "clustered_under_coverage"
    assert [_site_xyz_type_key(s) for s in expanded.sites] == [
        _site_xyz_type_key(s) for s in reduced.clustered_sites
    ]
    assert site_context_for_occupied_surface(expanded) is expanded

    info = _spec_grid_info(
        [make_water()], slab, config, "O", site_context=reduced, full_slab=full
    )
    assert [_site_xyz_type_key(s) for s in info.unique_sites] == [
        _site_xyz_type_key(s) for s in reduced.clustered_sites
    ]
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
    )
    assert specs
    wide = [s for s in specs if s.site_index >= len(reduced.sites)]
    assert wide, "enumerated specs must index into the clustered lattice"
    placed = generate_placement_from_spec(
        wide[0],
        [make_water()],
        full,
        config,
        "O",
        site_context=reduced,
        slab_for_sites=slab,
    )
    assert placed is not None


def test_n_tuplet_config_samples_full_lattice_on_clean_slab():
    """Co-adsorption needs distinct orbit copies before anything is adsorbed."""
    slab = make_slab(nx=2, ny=2)
    reduced = resolve_site_context_for_sampling(
        slab, AdsorptionConfig(material_type="slab"), symmetry_broken=False
    )
    assert reduced.clustered_sites is not None
    assert len(reduced.clustered_sites) > len(reduced.sites)
    tuplet_cfg = AdsorptionConfig(
        material_type="slab", seed=0, saturation_molecules_per_step=2
    )
    assert (
        skip_symmetry_for_sampling(
            symmetry_broken=False,
            slab_for_sites=slab,
            full_slab=slab,
            config=tuplet_cfg,
        )
        is True
    )
    info = _spec_grid_info([make_water()], slab, tuplet_cfg, "O", site_context=reduced)
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
    config = AdsorptionConfig(material_type="slab", seed=0)
    reduced = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
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
            [mol], slab, config, None, site_context=reduced, full_slab=full
        )
        assert [_site_xyz_type_key(s) for s in info.unique_sites] == [
            _site_xyz_type_key(s) for s in reduced.clustered_sites
        ]
        keys = {_site_xyz_type_key(info.unique_sites[i]) for i in info.site_indices}
        assert occupied_key not in keys
        kept.append(keys)
    assert kept[0] == kept[1]


def test_bo_pool_capacity_matches_clustered_catalog_under_coverage():
    """BO enumerate/materialize must use the expanded catalog, not orbit reps."""
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(material_type="slab", seed=0)
    reduced = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
    clustered = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    assert reduced.clustered_sites is not None
    occupied = reduced.clustered_sites[0]
    full = slab.copy() + Atoms(
        "H", positions=[occupied.xyz + np.array([0.0, 0.0, 0.2])]
    )
    water = [make_water()]
    cap_from_reduced = estimate_placement_spec_capacity(
        water, slab, config, "O", site_context=reduced, full_slab=full
    )
    cap_from_clustered = estimate_placement_spec_capacity(
        water, slab, config, "O", site_context=clustered, full_slab=full
    )
    assert cap_from_reduced == cap_from_clustered
    cap_clean = estimate_placement_spec_capacity(
        water, slab, config, "O", site_context=reduced
    )
    assert cap_from_reduced != cap_clean

    specs = enumerate_placement_specs(
        water, slab, config, "O", n_desired=24, site_context=reduced, full_slab=full
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
    )
    assert any(result is not None for result, _reason in generated)
    assert any(spec.site_index >= len(reduced.sites) for spec in specs)

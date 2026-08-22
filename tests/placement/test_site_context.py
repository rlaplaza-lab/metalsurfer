"""Site-context caching and surface references."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.features import extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.placement.site_context import _SITE_CONTEXT_CACHE
from metalsurfer.workflow import shared as workflow_shared

from ..conftest import (
    adsorption_config_factory,
    make_placement_descriptor,
    make_slab,
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

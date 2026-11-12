"""Unit tests for Site adapters and defensive copies."""

import numpy as np
import pytest

from metalsurfer.placement._material import MATERIAL_PBC, material_aware_pbc
from metalsurfer.placement.site_types import Site, with_symmetry


def _site(**overrides) -> Site:
    base = dict(
        xyz=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="atop",
        slab_indices=(0,),
        site_source="voronoi",
        env_fingerprint=(),
    )
    base.update(overrides)
    return Site(**base, material_type="slab")


def test_site_xy_returns_copy():
    site = _site()
    xy = site.xy
    xy[0] = 99.0
    assert site.xyz[0] == pytest.approx(1.0)


def test_site_normal_is_unit_length():
    site = _site(normal=np.array([0.0, 0.0, 2.0]))
    np.testing.assert_allclose(site.normal, [0.0, 0.0, 1.0])
    assert float(np.linalg.norm(site.normal)) == pytest.approx(1.0)


def test_with_symmetry_preserves_geometry_fields():
    site = _site()
    enriched = with_symmetry(
        site, symmetry_multiplicity=3, symmetry_equivalent_sites=((0.0, 0.0, 1.0),)
    )
    np.testing.assert_array_equal(enriched.xyz, site.xyz)
    np.testing.assert_array_equal(enriched.normal, site.normal)
    assert enriched.site_type == site.site_type
    assert enriched.slab_indices == site.slab_indices
    assert enriched.material_type == site.material_type
    assert enriched.symmetry_multiplicity == 3
    assert enriched.symmetry_equivalent_sites == ((0.0, 0.0, 1.0),)


def test_material_pbc_values_are_immutable_tuples():
    assert isinstance(MATERIAL_PBC["slab"], tuple)
    assert material_aware_pbc("slab") == [True, True, False]
    with pytest.raises(ValueError, match="material_type"):
        material_aware_pbc("crystal")

"""Unit tests for Site identity, adapters, and defensive copies."""

import numpy as np
import pytest

from metalsurfer.placement._constants import _SITE_Z_OFFSET_FROM_SURFACE_RADIUS
from metalsurfer.placement._material import MATERIAL_PBC, material_aware_pbc
from metalsurfer.placement.site_types import Site, with_symmetry


def _site(**overrides) -> Site:
    base = dict(
        xyz=np.array([1.0, 2.0, 3.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="atop",
        slab_indices=(0,),
        material_type="slab",
        site_source="voronoi",
        env_fingerprint=(),
    )
    base.update(overrides)
    return Site(**base)


def test_site_eq_hash_uses_rounded_coords_and_types():
    a = _site(xyz=np.array([1.0, 2.0, 3.0]))
    b = _site(xyz=np.array([1.0 + 1e-9, 2.0, 3.0]))
    c = _site(xyz=np.array([1.1, 2.0, 3.0]))
    d = _site(site_type="bridge")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d
    assert len({a, b, c}) == 2


def test_site_xy_returns_copy():
    site = _site()
    xy = site.xy
    xy[0] = 99.0
    assert site.xyz[0] == pytest.approx(1.0)


def test_with_symmetry_preserves_identity_fields():
    site = _site()
    enriched = with_symmetry(
        site, symmetry_multiplicity=3, symmetry_equivalent_sites=((0.0, 0.0),)
    )
    assert enriched == site
    assert enriched.symmetry_multiplicity == 3


def test_material_pbc_values_are_immutable_tuples():
    assert isinstance(MATERIAL_PBC["slab"], tuple)
    assert material_aware_pbc("slab") == [True, True, False]
    with pytest.raises(ValueError, match="material_type"):
        material_aware_pbc("crystal")


def test_pore_z_offset_matches_hollow():
    assert (
        _SITE_Z_OFFSET_FROM_SURFACE_RADIUS["pore"]
        == _SITE_Z_OFFSET_FROM_SURFACE_RADIUS["hollow"]
    )

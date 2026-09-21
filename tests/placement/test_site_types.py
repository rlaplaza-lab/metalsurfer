"""Unit tests for :class:`~metalsurfer.placement.site_types.Site`."""

import numpy as np

from metalsurfer.placement.site_types import Site, site_kind_from_type_and_supports


def test_site_kind_from_type_and_supports():
    assert site_kind_from_type_and_supports("hollow", (0, 1, 2)) == "wall"
    assert site_kind_from_type_and_supports("pore", ()) == "void"
    assert site_kind_from_type_and_supports("atop", ()) == "void"


def test_site_kind_auto_derives_wall_and_void():
    wall = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="hollow",
        slab_indices=(0, 1, 2),
        material_type="slab",
        site_source="test",
        env_fingerprint=(("Ru",), (0,), 1),
    )
    assert wall.kind == "wall"

    pore = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="pore",
        slab_indices=(),
        material_type="porous",
        site_source="test",
        env_fingerprint=((), (), 0),
    )
    assert pore.kind == "void"

    empty_support = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="atop",
        slab_indices=(),
        material_type="slab",
        site_source="test",
        env_fingerprint=((), (), 1),
    )
    assert empty_support.kind == "void"


def test_site_kind_explicit_override():
    site = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="hollow",
        slab_indices=(0, 1, 2),
        material_type="slab",
        site_source="test",
        env_fingerprint=(("Ru",), (0,), 1),
        kind="void",
    )
    assert site.kind == "void"

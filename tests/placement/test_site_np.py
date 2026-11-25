"""Regression tests for the shared NumPy topology helpers."""

import numpy as np
import pytest
from ase.build import fcc111

from metalsurfer.placement import get_unified_sites
from metalsurfer.placement.site_np import try_delaunay


def test_try_delaunay_accepts_three_non_collinear_points():
    tri = try_delaunay(np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]]))
    assert tri is not None
    assert tri.simplices.shape == (1, 3)


@pytest.mark.parametrize(
    "points",
    [
        np.empty((0, 2), dtype=float),
        np.array([[0.0, 0.0], [1.0, 1.0]]),
    ],
)
def test_try_delaunay_returns_none_for_fewer_than_three_points(points):
    assert try_delaunay(points) is None


def test_try_delaunay_returns_none_for_collinear_points():
    assert try_delaunay(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])) is None


def _three_atom_top_layer_slab():
    slab = fcc111("Pt", size=(2, 2, 2), vacuum=10.0, periodic=True)
    slab.set_pbc([True, True, False])
    z = slab.get_positions()[:, 2]
    top = np.nonzero(np.isclose(z, np.max(z)))[0]
    assert len(top) == 4
    del slab[int(top[-1])]
    z = slab.get_positions()[:, 2]
    top = np.nonzero(np.isclose(z, np.max(z)))[0]
    assert len(top) == 3
    return slab


def test_get_unified_sites_three_atom_top_layer_has_bridge_and_hollow():
    sites = get_unified_sites(
        _three_atom_top_layer_slab(), material_type="slab", auto_widen=False
    )
    sources = {s.site_source for s in sites}
    assert "topology_bridge" in sources
    assert "topology_hollow" in sources

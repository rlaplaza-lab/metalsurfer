"""Voronoi vertex generation and enrichment."""

import logging

import numpy as np
from ase.build import fcc111
from scipy.spatial import KDTree

import metalsurfer.placement.site_voronoi as site_voronoi_module
from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement.site_context import (
    _get_unique_sites_for_specs,
    clear_site_caches,
)
from metalsurfer.placement.site_voronoi import (
    _classify_voronoi_site,
    _enrich_along_ridges,
    _voronoi_sites,
)

from ..conftest import (
    make_porous_framework,
    make_slab,
)


def test_classify_voronoi_site_types_for_simple_geometries():
    positions_atop = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    site_type, _ = _classify_voronoi_site(np.array([0.8, 0.0, 0.0]), positions_atop)
    assert site_type == "atop"

    positions_bridge = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
        ]
    )
    site_type, idx = _classify_voronoi_site(
        np.array([1.0, 0.0, 0.0]), positions_bridge, k=4
    )
    assert site_type == "bridge"
    assert len(idx) == 2

    positions_hollow = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, np.sqrt(3.0), 0.0],
            [6.0, 6.0, 0.0],
        ]
    )
    site_type, idx = _classify_voronoi_site(
        np.array([1.0, np.sqrt(3.0) / 3.0, 0.0]), positions_hollow, k=4
    )
    assert site_type == "hollow"
    assert len(idx) == 3


def test_voronoi_nn_distances_match_periodic_image_query_for_porous():
    porous = make_porous_framework()
    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell())
    pbc = np.asarray(porous.get_pbc(), dtype=bool)

    vertices, nn_dists = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
    )
    assert len(vertices) > 0

    shifts = [([-1, 0, 1] if pbc[d] else [0]) for d in range(3)]
    extended = []
    for i in shifts[0]:
        for j in shifts[1]:
            for k in shifts[2]:
                offset = i * cell[0] + j * cell[1] + k * cell[2]
                extended.append(positions + offset)
    extended_positions = np.vstack(extended)

    expected, _ = KDTree(extended_positions).query(vertices, k=1)
    np.testing.assert_allclose(nn_dists, np.ravel(expected), atol=1e-8)


def test_voronoi_enrichment_increases_site_count_on_porous():
    porous = make_porous_framework()
    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell())
    pbc = np.asarray(porous.get_pbc(), dtype=bool)

    vertices_base, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=False,
    )
    vertices_enriched, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=True,
    )

    assert len(vertices_base) > 0
    # Real 3D ridge polygons must yield subdivision candidates on this framework.
    assert len(vertices_enriched) > len(vertices_base)


def test_enrich_along_ridges_walks_polygonal_faces():
    """Closed 3D ridge faces (len > 2) contribute consecutive edges for subdivision.

    A regression to ``if len(ridge) != 2: continue`` skips this face and returns
    the input vertices unchanged.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    nn_dists = np.array([2.0, 2.0, 2.0], dtype=float)
    ridge_vertices = [[0, 1, 2]]
    raw_to_kept = {0: 0, 1: 1, 2: 2}
    extended = vertices.copy()
    tree = KDTree(extended)
    out_verts, _ = _enrich_along_ridges(
        vertices,
        nn_dists,
        ridge_vertices,
        raw_to_kept,
        extended,
        tree,
        probe_radius=0.0,
        max_distance=100.0,
        cell=np.eye(3) * 20.0,
        pbc=np.array([False, False, False], dtype=bool),
    )
    assert len(out_verts) > len(vertices)

    # Control: a ridge that the old len==2 filter would accept still subdivides.
    out_edge, _ = _enrich_along_ridges(
        vertices[:2],
        nn_dists[:2],
        [[0, 1]],
        {0: 0, 1: 1},
        vertices[:2],
        KDTree(vertices[:2]),
        probe_radius=0.0,
        max_distance=100.0,
        cell=np.eye(3) * 20.0,
        pbc=np.array([False, False, False], dtype=bool),
    )
    assert len(out_edge) > 2


def test_voronoi_enrichment_uses_ridge_vertices(monkeypatch):
    class _FakeVoronoi:
        def __init__(self):
            self.vertices = np.array(
                [
                    [2.0, 2.0, 2.0],
                    [6.0, 2.0, 2.0],
                    [4.0, 2.0, 2.0],
                ],
                dtype=float,
            )
            # Input-point connectivity (not valid for Voronoi vertex graph).
            self.ridge_points = np.array([[0, 1]], dtype=int)
            # Voronoi vertex connectivity used for enrichment.
            self.ridge_vertices = [[0, 2], [2, 1]]

    fake_vor = _FakeVoronoi()
    monkeypatch.setattr(site_voronoi_module, "Voronoi", lambda _pts: fake_vor)

    captured = {}

    def _capture_ridges(
        vertices,
        nn_dists,
        ridge_vertices,
        raw_to_kept,
        extended_positions,
        framework_tree,
        probe_radius,
        max_distance,
        *,
        cell,
        pbc,
    ):
        captured["ridge_vertices"] = ridge_vertices
        return vertices, nn_dists

    monkeypatch.setattr(site_voronoi_module, "_enrich_along_ridges", _capture_ridges)

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    pbc = np.array([False, False, False], dtype=bool)

    _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=0.0,
        max_distance=100.0,
        enrich=True,
    )

    assert captured["ridge_vertices"] == fake_vor.ridge_vertices


def test_voronoi_auto_widen_retries_when_first_window_empty(monkeypatch):
    """Empty first Voronoi window triggers one widened retry when enabled."""
    import metalsurfer.placement.site_context as site_context_mod

    clear_site_caches()
    slab = make_slab()
    calls = {"n": 0}
    real = site_context_mod.get_unified_sites

    def fake_get_unified_sites(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return real(*args, **kwargs)

    monkeypatch.setattr(site_context_mod, "get_unified_sites", fake_get_unified_sites)
    ctx = _get_unique_sites_for_specs(slab, AdsorptionConfig(voronoi_auto_widen=True))
    assert calls["n"] == 2
    assert ctx.use_sites
    assert len(ctx.sites) > 0


def test_voronoi_auto_widen_disabled_skips_retry(monkeypatch):
    import metalsurfer.placement.site_context as site_context_mod

    clear_site_caches()
    slab = make_slab()
    calls = {"n": 0}

    def fake_get_unified_sites(*args, **kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(site_context_mod, "get_unified_sites", fake_get_unified_sites)
    ctx = _get_unique_sites_for_specs(slab, AdsorptionConfig(voronoi_auto_widen=False))
    assert calls["n"] == 1
    assert not ctx.use_sites


def test_planar_slab_skips_voronoi_and_says_so(caplog, monkeypatch):
    """Qhull used to raise QH6154 on the coplanar top layer and be swallowed."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    def _fail(*_args, **_kwargs):
        raise AssertionError("_voronoi_sites must not run on a planar slab")

    monkeypatch.setattr(site_enumeration, "_voronoi_sites", _fail)

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.INFO, logger="metalsurfer.placement.site_enumeration"):
        sites = site_enumeration.get_unified_sites(slab, material_type="slab")

    assert sites
    assert any(
        "skipping Voronoi vertex generation" in record.getMessage()
        for record in caplog.records
    )


def test_non_planar_slab_still_runs_voronoi():
    """Only coplanar top layers skip the Voronoi pass."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = slab.get_positions()
    positions[-1, 2] += 1.5  # adatom-like bump breaks coplanarity
    slab.set_positions(positions)

    cell = np.asarray(slab.get_cell(), dtype=float)
    assert not site_enumeration._top_layer_is_planar_from_arrays(
        np.asarray(slab.get_positions(), dtype=float), cell, 0.5
    )


def test_slab_only_enrichment_flag_warns(caplog):
    """voronoi_site_enrichment is porous/nanoparticle-only; slabs get a warning."""
    from metalsurfer.config import AdsorptionConfig
    from metalsurfer.placement.site_context import (
        _get_unique_sites_for_specs,
        clear_site_caches,
    )

    clear_site_caches()
    config = AdsorptionConfig(material_type="slab", voronoi_site_enrichment=False)
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.WARNING, logger="metalsurfer.placement.site_context"):
        _get_unique_sites_for_specs(slab, config)
    clear_site_caches()

    assert any(
        "voronoi_site_enrichment=False has no effect" in record.getMessage()
        for record in caplog.records
    )

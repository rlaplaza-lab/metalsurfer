"""Voronoi vertex generation and enrichment."""

import logging

import numpy as np
import pytest
from ase.build import fcc111
from scipy.spatial import KDTree, Voronoi

import metalsurfer.placement.site_voronoi as site_voronoi_module
from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement._constants import (
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _VORONOI_DEDUP_TOLERANCE,
)
from metalsurfer.placement.site_context import _get_unique_sites_for_specs
from metalsurfer.placement.site_coords import (
    _build_periodic_images,
    _cart_to_frac,
    _deduplicate_points,
    _wrap_cartesian,
    top_layer_mask_by_normal,
)
from metalsurfer.placement.site_voronoi import (
    _classify_voronoi_site_from_neighbors,
    _enrich_along_ridges,
    _voronoi_sites,
)

from ..conftest import (
    make_porous_framework,
    make_slab,
)


def _classify_vertex(vertex, positions, k=_SITE_CLASSIFICATION_NEIGHBOURS):
    k_query = min(k, len(positions))
    dists, idx = KDTree(positions).query(np.asarray(vertex).reshape(1, 3), k=k_query)
    return _classify_voronoi_site_from_neighbors(
        np.asarray(dists, dtype=float).ravel(),
        np.asarray(idx, dtype=int).ravel(),
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
    site_type, _ = _classify_vertex(np.array([0.8, 0.0, 0.0]), positions_atop)
    assert site_type == "atop"

    positions_bridge = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
        ]
    )
    site_type, idx = _classify_vertex(np.array([1.0, 0.0, 0.0]), positions_bridge, k=4)
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
    site_type, idx = _classify_vertex(
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
        symbols=list(porous.get_chemical_symbols()),
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


def test_voronoi_keeps_vertices_that_wrap_into_cell():
    """Unwrapped frac just outside [-margin, 1+margin) must not drop wrapped sites.

    Regression: the old ``inside`` filter used raw fractional coords, so a
    vertex at frac 1.02 was discarded even though wrap maps it to 0.02.
    """
    porous = make_porous_framework()
    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell(), dtype=float)
    pbc = np.asarray(porous.get_pbc(), dtype=bool)

    vertices, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=False,
        symbols=list(porous.get_chemical_symbols()),
    )
    assert len(vertices) > 0

    # Reconstruct what the old unwrapped-frac filter would have kept from the
    # same accessibility window, then show wrap+dedup retains strictly more.
    extension_margin = 4.5 + _VORONOI_DEDUP_TOLERANCE
    extended = _build_periodic_images(positions, cell, pbc, margin=extension_margin)
    vor = Voronoi(extended)
    raw = np.asarray(vor.vertices, dtype=float)
    wrapped = _wrap_cartesian(raw, cell, pbc)
    nn_dists, _ = KDTree(extended).query(wrapped, k=1)
    nn_dists = np.asarray(nn_dists, dtype=float).ravel()
    accessible = (nn_dists >= 1.0) & (nn_dists <= 4.5)
    wrapped_acc = wrapped[accessible]
    raw_acc = raw[accessible]

    frac = _cart_to_frac(raw_acc, cell)
    # Historical unwrapped-frac filter margin (removed from production).
    old_frac_margin = 0.01
    inside = np.ones(len(frac), dtype=bool)
    for dim in range(3):
        if bool(pbc[dim]):
            inside &= (frac[:, dim] >= -old_frac_margin) & (
                frac[:, dim] < 1.0 + old_frac_margin
            )
    old_keep = _deduplicate_points(
        wrapped_acc[inside], _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
    )
    new_keep = _deduplicate_points(
        wrapped_acc, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc
    )
    assert int(np.count_nonzero(new_keep)) > int(np.count_nonzero(old_keep))
    assert len(vertices) == int(np.count_nonzero(new_keep))


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
        symbols=list(porous.get_chemical_symbols()),
    )
    vertices_enriched, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=True,
        symbols=list(porous.get_chemical_symbols()),
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
        symbols=["C"] * len(positions),
    )

    assert captured["ridge_vertices"] == fake_vor.ridge_vertices


def test_voronoi_auto_widen_retries_when_first_window_empty(monkeypatch):
    """Empty first Voronoi window triggers one widened retry when enabled."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    slab = make_slab()
    calls = {"n": 0}
    real = site_enumeration._enumerate_unified_sites

    def fake_enumerate(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return real(*args, **kwargs)

    monkeypatch.setattr(site_enumeration, "_enumerate_unified_sites", fake_enumerate)
    ctx = _get_unique_sites_for_specs(slab, AdsorptionConfig(voronoi_auto_widen=True))
    assert calls["n"] == 2
    assert ctx.use_sites
    assert len(ctx.sites) > 0


def test_voronoi_auto_widen_disabled_skips_retry(monkeypatch):
    import metalsurfer.placement.site_enumeration as site_enumeration

    slab = make_slab()
    calls = {"n": 0}

    def fake_enumerate(*args, **kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(site_enumeration, "_enumerate_unified_sites", fake_enumerate)
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


def test_non_planar_slab_still_runs_voronoi(monkeypatch):
    """Only coplanar top layers skip the Voronoi pass."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    calls = {"n": 0}
    real_voronoi = site_enumeration._voronoi_sites

    def _counting_voronoi(*args, **kwargs):
        calls["n"] += 1
        return real_voronoi(*args, **kwargs)

    monkeypatch.setattr(site_enumeration, "_voronoi_sites", _counting_voronoi)

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = slab.get_positions()
    positions[-1, 2] += 1.5  # adatom-like bump breaks coplanarity
    slab.set_positions(positions)

    cell = np.asarray(slab.get_cell(), dtype=float)
    assert not site_enumeration._top_layer_is_planar_from_arrays(
        np.asarray(slab.get_positions(), dtype=float), cell, 0.5
    )
    sites = site_enumeration.get_unified_sites(slab, material_type="slab")
    assert calls["n"] >= 1
    assert sites


def test_slab_enrichment_flag_does_not_warn_from_site_context(caplog):
    """Planar slabs skip Voronoi in enumeration; site_context must not warn."""
    from metalsurfer.config import AdsorptionConfig
    from metalsurfer.placement.site_context import _get_unique_sites_for_specs

    config = AdsorptionConfig(material_type="slab", voronoi_site_enrichment=False)
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.WARNING, logger="metalsurfer.placement.site_context"):
        _get_unique_sites_for_specs(slab, config)

    assert not any(
        "voronoi_site_enrichment=False has no effect" in record.getMessage()
        for record in caplog.records
    )


def test_porous_classification_uses_periodic_neighbours_near_cell_face():
    """Near-face porous sites must type from MIC k-NN, not in-cell distances."""
    from metalsurfer.placement.site_classify import _build_classification_context
    from metalsurfer.placement.site_coords import derive_pore_threshold
    from metalsurfer.placement.site_enumeration import get_unified_sites

    porous = make_porous_framework()
    sites = get_unified_sites(porous, material_type="porous")
    assert sites, "porous fixture must expose sites"

    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell(), dtype=float)
    pbc = np.asarray(porous.get_pbc(), dtype=bool)
    local_tree = KDTree(positions)
    vertices = np.asarray([s.xyz for s in sites], dtype=float)
    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))

    ctx = _build_classification_context(
        vertices,
        positions,
        local_tree,
        material_type="porous",
        cell=cell,
        pbc=pbc,
        delaunay=None,
    )
    assert ctx.class_dists is not None and ctx.class_idx is not None

    plain_dists, plain_idx = local_tree.query(vertices, k=k_class)
    plain_dists = np.asarray(plain_dists, dtype=float)
    plain_idx = np.asarray(plain_idx, dtype=int)
    if plain_dists.ndim == 1:
        plain_dists = plain_dists.reshape(-1, 1)
        plain_idx = plain_idx.reshape(-1, 1)
    assert np.any(
        ~np.isclose(plain_dists, ctx.class_dists, atol=1e-6)
        | (plain_idx != ctx.class_idx)
    ), "expected at least one near-face site where in-cell k-NN differs from MIC"

    pore_threshold = derive_pore_threshold(list(porous.get_chemical_symbols()))
    for site, d_ref, i_ref in zip(sites, ctx.class_dists, ctx.class_idx, strict=True):
        expected_type, expected_idx = _classify_voronoi_site_from_neighbors(
            d_ref, i_ref, pore_threshold=pore_threshold
        )
        assert site.site_type == expected_type
        assert tuple(site.slab_indices) == expected_idx


def _primitive_in_plane_vectors(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two shortest independent in-plane lattice vectors spanning *pts*."""
    p0 = pts[0]
    diffs = [p - p0 for p in pts[1:]]
    cands: list[np.ndarray] = []
    for d in diffs:
        cands.append(np.asarray(d, dtype=float))
        cands.append(-np.asarray(d, dtype=float))
    cands.sort(key=lambda v: float(np.linalg.norm(v)))
    v1 = cands[0]
    for v in cands[1:]:
        # 2D cross product (avoid NumPy 2.x 2-vector deprecation).
        if abs(v1[0] * v[1] - v1[1] * v[0]) > 1e-6:
            v2 = v
            break
    else:
        raise RuntimeError("could not find two independent in-plane lattice vectors")
    return v1, v2


def _polygon_area(verts: np.ndarray) -> float:
    """Shoelace area of a closed 2D polygon (rows = vertices)."""
    return 0.5 * abs(
        np.dot(verts[:, 0], np.roll(verts[:, 1], -1))
        - np.dot(verts[:, 1], np.roll(verts[:, 0], -1))
    )


def test_voronoi_area_conservation_fcc111():
    """Projected fcc111(2×2) top-layer Voronoi cells tile one surface unit cell.

    The flat top layer is a 2D Bravais lattice. Replicating it and summing the
    Voronoi cell areas of the interior lattice points must reproduce the cell's
    in-plane cross-sectional area: a tessellation partitions space, so the
    central unit cell's worth of Voronoi polygons conserves the surface area.
    """
    slab = fcc111("Pt", (2, 2, 3), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    top_mask = top_layer_mask_by_normal(positions, cell, 0.5)
    top_xy = positions[top_mask][:, :2]
    n_top = len(top_xy)

    # Primitive in-plane lattice vectors derived from the top-layer point set.
    v1, v2 = _primitive_in_plane_vectors(top_xy)

    # Replicate broadly so the central (0, 0)-shift copies are interior points
    # with fully bounded Voronoi cells.
    R = 5
    shifts = np.array(
        [[i, j] for i in range(-R, R + 1) for j in range(-R, R + 1)],
        dtype=float,
    )
    basis = np.stack([v1, v2])
    all_pts = (top_xy[:, None, :] + (shifts @ basis)[None, :, :]).reshape(-1, 2)
    vor = Voronoi(all_pts)

    central = R * (2 * R + 1) + R
    cell_area = abs(float(np.linalg.det(cell[:2, :2])))

    area_sum = 0.0
    bounded = 0
    for io in range(n_top):
        gi = io * len(shifts) + central
        region = vor.regions[vor.point_region[gi]]
        assert region and -1 not in region, "interior Voronoi cell must be bounded"
        area_sum += _polygon_area(vor.vertices[region])
        bounded += 1
    assert bounded == n_top
    assert area_sum == pytest.approx(cell_area, abs=1e-6)


def test_expand_top_layer_ab_images_matches_projection():
    """4.2: the hoisted ortho-basis projection equals the per-offset projection."""
    from ase.build import fcc111

    from metalsurfer._geom_pbc import slab_plane_projectors
    from metalsurfer.placement.site_coords import _project_to_slab_plane

    slab = fcc111("Cu", (3, 3, 3), vacuum=8.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    top_xy = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=float)
    top_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]], dtype=float)

    exp_xy, _, _, _ = site_voronoi_module._expand_top_layer_ab_images(
        top_xy,
        cell=cell,
        pbc=np.array([True, True, False]),
        top_positions_3d=top_3d,
    )

    # Reference: the same expansion computed with the explicit per-offset
    # projection must match the hoisted-basis result.
    _, ortho_basis = slab_plane_projectors(cell)
    ref_xy = []
    for ia in (-1, 0, 1):
        for ib in (-1, 0, 1):
            offset = ia * cell[0] + ib * cell[1]
            off_2d = _project_to_slab_plane(offset.reshape(1, 3), cell)[0]
            for li in range(len(top_xy)):
                ref_xy.append(top_xy[li] + off_2d)
    assert np.allclose(exp_xy, np.asarray(ref_xy), atol=1e-12, rtol=0)

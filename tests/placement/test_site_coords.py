"""Periodic coordinate utilities for site cataloguing."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from metalsurfer.placement.site_classify import _build_site_records
from metalsurfer.placement.site_coords import (
    _deduplicate_points,
    mic_support_positions,
    project_anchor_to_support_plane,
    project_vertices_to_support_plane,
)
from metalsurfer.placement.site_plugins.wall_near import wall_near_batch_to_plugin
from metalsurfer.placement.site_wall_near_common import (
    AdaptiveGridResult,
    AdaptiveGridSpacing,
)


def test_deduplicate_points_returns_expected_keep_mask():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [1.0, 1.0, 1.0],
            [1.01, 1.0, 1.0],
        ]
    )
    keep = _deduplicate_points(points, tolerance=0.03)
    assert keep.dtype == bool
    assert keep.shape == (4,)
    assert int(np.sum(keep)) == 2
    kept_points = points[keep]
    assert len(kept_points) == 2


def test_deduplicate_points_is_order_independent():
    """Union-find dedup should pick the same representative regardless of input order."""
    points_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    points_b = points_a[[0, 2, 1]]
    keep_a = _deduplicate_points(points_a, tolerance=0.05)
    keep_b = _deduplicate_points(points_b, tolerance=0.05)
    assert int(np.sum(keep_a)) == 2
    assert int(np.sum(keep_b)) == 2
    kept_a = np.sort(points_a[keep_a], axis=0)
    kept_b = np.sort(points_b[keep_b], axis=0)
    np.testing.assert_allclose(kept_a, kept_b, atol=1e-10)


def _porous_wall_boundary_fixture():
    cell = np.diag([10.0, 10.0, 10.0])
    pbc = np.array([True, True, True], dtype=bool)
    positions = np.array([[0.5, 5.0, 5.0], [9.5, 5.0, 5.0]], dtype=float)
    normal = np.array([1.0, 0.0, 0.0], dtype=float)
    vert = np.array([0.0, 5.5, 5.0], dtype=float)
    support = (0, 1)
    anchor = project_anchor_to_support_plane(
        vert,
        normal,
        mic_support_positions(vert, positions, support, cell=cell, pbc=pbc),
    )
    return cell, pbc, positions, normal, support, anchor


def test_project_vertices_to_support_plane_uses_mic_on_porous_wall():
    """Primary-cell supports must not yank a face site across the cell."""
    cell, pbc, positions, normal, support, anchor = _porous_wall_boundary_fixture()
    verts = anchor.reshape(1, 3)
    norms = normal.reshape(1, 3)
    primary = project_vertices_to_support_plane(verts, norms, [support], positions)[0]
    mic = project_vertices_to_support_plane(
        verts, norms, [support], positions, cell=cell, pbc=pbc
    )[0]
    assert float(np.linalg.norm(primary - anchor)) == pytest.approx(9.0, abs=1e-9)
    np.testing.assert_allclose(mic, anchor, atol=1e-9)


def test_wall_near_and_classify_keep_mic_anchor_on_porous_wall():
    """Plugin pack + classify stay on a MIC support, not the far primary image."""
    cell, pbc, positions, normal, support, anchor = _porous_wall_boundary_fixture()
    result = AdaptiveGridResult(
        vertices=np.array([anchor], dtype=float),
        nn_dists=np.array([0.5], dtype=float),
        clearances=np.array([0.5], dtype=float),
        atom_indices=[support],
        normals=np.array([normal], dtype=float),
        spacing=AdaptiveGridSpacing(1.0, 1.0, 0.5, 0, 0.5),
        accessibility_tree=KDTree(positions),
    )
    batch = wall_near_batch_to_plugin(
        result, positions=positions, cell=cell, pbc=pbc, source_hint="adaptive_grid"
    )
    np.testing.assert_allclose(batch.vertices[0], anchor, atol=1e-9)
    sites = _build_site_records(
        batch.vertices,
        batch.nn_dists,
        positions,
        ["C", "C"],
        KDTree(positions),
        "porous",
        pore_threshold=2.0,
        cell=cell,
        pbc=pbc,
        source_hints=list(batch.source_hints),
        atom_indices=list(batch.atom_indices),
        normals=batch.normals,
        clearances=batch.clearances,
        probe_radius=0.5,
    )
    assert len(sites) == 1
    xyz = np.asarray(sites[0].xyz, dtype=float)
    assert abs(float(xyz[0])) == pytest.approx(0.5, abs=1e-6)
    assert abs(float(xyz[0]) - 9.5) > 1.0

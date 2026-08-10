"""Periodic coordinate utilities for site cataloguing."""

import numpy as np

from metalsurfer.placement.site_coords import _deduplicate_points


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

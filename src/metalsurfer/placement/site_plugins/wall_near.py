"""Shared plugin packaging for wall-near generators (adaptive grid / rolling probe)."""

from __future__ import annotations

import numpy as np

from ..site_coords import project_vertices_to_support_plane
from ..site_wall_near_common import AdaptiveGridResult
from .base import SiteCandidateBatch, empty_candidate_batch
from .helpers import median_nn_or_fallback


def wall_near_batch_to_plugin(
    result: AdaptiveGridResult,
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    source_hint: str,
) -> SiteCandidateBatch:
    """Project supports and wrap a wall-near engine result as a plugin batch."""
    if len(result.vertices) == 0:
        return empty_candidate_batch()

    vertices = project_vertices_to_support_plane(
        result.vertices,
        result.normals,
        list(result.atom_indices),
        positions,
    )
    return SiteCandidateBatch(
        vertices=vertices,
        nn_dists=result.nn_dists,
        source_hints=[source_hint] * len(result.vertices),
        atom_indices=list(result.atom_indices),
        accessibility_tree=result.accessibility_tree,
        topology_median_nn=float(
            median_nn_or_fallback(
                result.nn_dists, reference_positions=positions, cell=cell, pbc=pbc
            )
        ),
        normals=result.normals,
        clearances=result.clearances,
    )

"""Shared pose placement context and cache types."""

import dataclasses
from dataclasses import dataclass

import numpy as np

from ...models import PlacementPose
from ..site_types import Site


def _require_pose_z_abs(pose: PlacementPose) -> float:
    """Return absolute z; recovery paths must not invent ``0.0`` for missing pose."""
    if pose.z_abs is None:
        raise ValueError("PlacementPose.z_abs is required; no zero fallback")
    return float(pose.z_abs)


@dataclass
class _PlacementContext:
    """Inputs for ``_finalize_placement``: pose, site/material refs, canonical and rotated positions."""

    pose: PlacementPose
    site: Site | None
    mat_type: str
    surface_ref: float
    is_local_ref: bool
    source: str
    canonical_pos: np.ndarray
    use_sites: bool
    rotated_pos: np.ndarray
    normal: np.ndarray
    z_base_lo: float = 0.0
    z_base_hi: float = 0.0
    shape: str = "round"
    com_lo: float | None = None
    com_nominal: float | None = None
    com_hi: float | None = None


@dataclass
class _PoseBatchCache:
    """Per-batch invariants shared across placements on the same substrate."""

    pinv_ab_T: np.ndarray | None = None
    # Mean top-layer covalent radius for z-offset when site has no slab_indices.
    r_surface_top_layer: float | None = None
    cell: np.ndarray | None = None
    n_hat: np.ndarray | None = None
    positions: np.ndarray | None = None
    frames: dict[int, tuple[np.ndarray, str]] = dataclasses.field(default_factory=dict)
    height_intervals: dict[tuple, "_HeightInterval"] = dataclasses.field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _HeightInterval:
    """Feasible COM height along the placement normal for one rigid pose family."""

    com_lo: float
    com_nominal: float
    com_hi: float
    contact_atoms_ok: bool

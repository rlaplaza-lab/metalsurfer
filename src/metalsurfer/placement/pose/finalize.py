"""Validate, describe, and emit placements from poses."""

import logging

import numpy as np
from ase import Atoms

from ..._utils import is_finite_number as _is_finite_number
from ...config import AdsorptionConfig
from ...models import PlacementDescriptor, PlacementPose
from .. import geometry as geom
from .._constants import RECOVERABLE_DISTANCE_REASONS
from ..site_context import SiteContext
from ..site_coords import _slab_plane_projectors
from .build import _context_from_pose
from .checks import (
    _build_slab_distance_scratch,
    _saturation_exclude_count,
    _validate_posed_adsorbate,
)
from .context import _PlacementContext, _PoseBatchCache, _require_pose_z_abs
from .recovery import (
    _recover_distance_failure,
    _recover_z_offset,
    _set_adsorbate_at_center,
)

logger = logging.getLogger(__name__)


def _descriptor_from_placement(
    pose: PlacementPose,
    *,
    z_offset: float,
    surface_ref: float,
    shape: str,
    slab_indices: tuple[int, ...] | None,
    site_source: str,
    site_reference_frame: str,
    site_xy_frac_a: float,
    site_xy_frac_b: float,
    z_abs: float,
    placement_mode_resolved: str = "no_sites",
    fragment_positions: tuple[tuple[float, float, float], ...] | None = None,
) -> PlacementDescriptor:
    """Build a PlacementDescriptor from resolved pose/geometry fields.

    *z_offset* is the COM height above *surface_ref* (pairwise contact-solved
    for wall-near sites; fractional window for free-volume pores).
    """
    return PlacementDescriptor(
        conformer_index=pose.conformer_index,
        orientation_type=pose.orientation_type or "round",
        face_flip=pose.face_flip,
        en_atom_index=pose.en_atom_index,
        site_index=pose.site_index,
        site_type=pose.site_type,
        tilt_deg=pose.tilt_deg,
        azimuth_deg=pose.azimuth_deg,
        azimuth_in_plane_deg=pose.azimuth_in_plane_deg,
        z_fraction=pose.z_fraction,
        placement_index=pose.placement_index,
        x=float(pose.x_abs),
        y=float(pose.y_abs),
        z_offset=z_offset,
        x_abs=float(pose.x_abs),
        y_abs=float(pose.y_abs),
        surface_ref_z_abs=surface_ref,
        z_abs=float(z_abs),
        shape=shape,
        slab_indices=slab_indices,
        placement_mode_resolved=placement_mode_resolved,
        site_source=site_source,
        site_reference_frame=site_reference_frame,
        site_xy_frac_a=site_xy_frac_a,
        site_xy_frac_b=site_xy_frac_b,
        quat_w=float(pose.quat_w),
        quat_x=float(pose.quat_x),
        quat_y=float(pose.quat_y),
        quat_z=float(pose.quat_z),
        fragment_positions=fragment_positions,
    )


def _finalize_placement(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
    allow_distance_recovery: bool = False,
    pose_cache: _PoseBatchCache | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Translate the pre-rotated positions, validate, and build a descriptor."""
    pose = ctx.pose
    if pose.z_abs is None:
        logger.warning("Pose replay requires z_abs for deterministic reconstruction")
        return None, "missing_z_abs"
    z_abs = float(pose.z_abs)

    _set_adsorbate_at_center(
        adsorbate,
        ctx.rotated_pos,
        np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float),
    )

    # Share slab scratch across first validation and any recovery attempts.
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    slab_scratch = _build_slab_distance_scratch(slab, exclude_n, ctx.mat_type)

    fail_reason = _validate_posed_adsorbate(
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        material_type=ctx.mat_type,
        slab_scratch=slab_scratch,
    )
    if fail_reason is not None:
        if (
            allow_distance_recovery
            and config.placement_distance_recovery
            and fail_reason
            and fail_reason in RECOVERABLE_DISTANCE_REASONS
        ):
            ctx, fail_reason = _recover_distance_failure(
                ctx,
                adsorbate,
                slab,
                config,
                fail_reason,
                slab_for_sites=slab_for_sites,
                pose_cache=pose_cache,
                slab_scratch=slab_scratch,
            )
            pose = ctx.pose
            if fail_reason is not None:
                return None, fail_reason
            z_abs = _require_pose_z_abs(pose)
        else:
            return None, fail_reason

    # COM height above surface_ref (contact-solved / fractional window at pose).
    z_offset = _recover_z_offset(ctx, z_abs)
    slab_indices: tuple[int, ...] | None = None
    if ctx.site is not None:
        slab_indices = ctx.site.slab_indices
    cell = (
        pose_cache.cell
        if pose_cache is not None and pose_cache.cell is not None
        else np.asarray(slab.get_cell(), dtype=float)
    )
    if pose_cache is not None and pose_cache.pinv_ab_T is not None:
        pinv_ab_T = pose_cache.pinv_ab_T
    else:
        pinv_ab_T, _ = _slab_plane_projectors(cell)
    # Full 3D point: zeroing z biases frac a/b on tilted (non-orthogonal) cells.
    placement_xyz = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    frac2 = placement_xyz @ pinv_ab_T
    xy_frac = np.mod(frac2, 1.0)

    site_source = ctx.source if ctx.use_sites else "no_sites"
    descriptor = _descriptor_from_placement(
        pose,
        z_offset=z_offset,
        surface_ref=ctx.surface_ref,
        shape=ctx.shape,
        slab_indices=slab_indices,
        site_source=site_source,
        site_reference_frame=(
            "local_site" if ctx.site is not None else "global_top_layer"
        ),
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        z_abs=z_abs,
        placement_mode_resolved="sites" if ctx.use_sites else "no_sites",
    )
    return (adsorbate, descriptor), None


def generate_placement_from_pose(
    pose: PlacementPose,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement using universal pose semantics.

    Parameters
    ----------
    pose
        :class:`~metalsurfer.models.PlacementPose` with placement parameters.
    conformers
        List of adsorbate :class:`~ase.Atoms` conformers.
    slab
        Substrate :class:`~ase.Atoms` (may include pre-adsorbed molecules).
    config
        :class:`~metalsurfer.config.AdsorptionConfig` with placement settings.
    site_context
        Optional cached :class:`SiteContext`.
    slab_for_sites
        Optional bare-substrate reference for surface_ref / site detection
        (same contract as :func:`generate_placement_from_spec`).
    pose_cache
        Optional batch cache (planarity, surface ref) shared across poses on
        the same substrate; see :func:`build_pose_batch_cache`.
    """
    if not conformers:
        return None
    if pose.conformer_index < 0 or pose.conformer_index >= len(conformers):
        logger.warning(
            "Invalid conformer_index=%d for %d conformers",
            pose.conformer_index,
            len(conformers),
        )
        return None
    if not _is_finite_number(pose.x_abs) or not _is_finite_number(pose.y_abs):
        logger.warning("Pose must provide finite x_abs and y_abs")
        return None
    if pose.z_abs is not None and not _is_finite_number(pose.z_abs):
        logger.warning("Pose z_abs must be finite when provided")
        return None

    adsorbate = conformers[pose.conformer_index].copy()
    symbols = adsorbate.get_chemical_symbols()
    cached_frame = (
        pose_cache.frames.get(int(pose.conformer_index))
        if pose_cache is not None
        else None
    )
    if cached_frame is not None:
        canonical_pos, _shape = cached_frame
    else:
        canonical_pos = geom.compute_canonical_molecular_frame(
            adsorbate.get_positions(), symbols=symbols
        )

    ctx = _context_from_pose(
        pose,
        canonical_pos,
        slab,
        config,
        site_context,
        slab_for_sites=slab_for_sites,
        pose_cache=pose_cache,
    )
    if ctx is None:
        return None
    result, fail_reason = _finalize_placement(
        ctx,
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        pose_cache=pose_cache,
    )
    if result is None:
        logger.debug("Pose placement rejected: %s", fail_reason)
        return None
    return result

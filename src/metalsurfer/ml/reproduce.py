"""Deterministic reconstruction of simulation inputs from PlacementRecord.

Given a :class:`PlacementRecord` (or its serialized form), this module
rebuilds placement descriptors and an :class:`~metalsurfer.AdsorptionConfig`
matching the stored computation context for placement replay.
"""

import numpy as np

from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..models import PlacementDescriptor
from ..placement.geometry import normalize_quaternion
from .schema import PlacementRecord


def record_to_placement_descriptor(record: PlacementRecord) -> PlacementDescriptor:
    """Extract a PlacementDescriptor from a PlacementRecord."""
    required_fields = {
        "x_abs": record.x_abs,
        "y_abs": record.y_abs,
        "z_abs": record.z_abs,
        "quat_w": record.quat_w,
        "quat_x": record.quat_x,
        "quat_y": record.quat_y,
        "quat_z": record.quat_z,
    }
    missing = [
        name for name, value in required_fields.items() if not _is_finite_number(value)
    ]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            "PlacementRecord is missing finite deterministic geometry fields: "
            f"{missing_csv}"
        )
    quat = normalize_quaternion(
        np.array(
            [
                float(record.quat_w),
                float(record.quat_x),
                float(record.quat_y),
                float(record.quat_z),
            ],
            dtype=float,
        )
    )
    return PlacementDescriptor(
        conformer_index=record.conformer_index,
        orientation_type=record.orientation_type,
        face_flip=record.face_flip,
        en_atom_index=record.en_atom_index,
        site_index=record.site_index,
        site_type=record.site_type,
        tilt_deg=record.tilt_deg,
        azimuth_deg=record.azimuth_deg,
        azimuth_in_plane_deg=record.azimuth_in_plane_deg,
        z_fraction=record.z_fraction,
        placement_index=record.placement_id,
        x=record.x,
        y=record.y,
        x_abs=record.x_abs,
        y_abs=record.y_abs,
        z_offset=record.z_offset,
        surface_ref_z_abs=record.surface_ref_z_abs,
        z_abs=record.z_abs,
        shape=record.shape,
        slab_indices=record.slab_indices,
        placement_mode_resolved=record.placement_mode_resolved,
        site_source=record.site_source,
        site_reference_frame=record.site_reference_frame,
        site_xy_frac_a=record.site_xy_frac_a,
        site_xy_frac_b=record.site_xy_frac_b,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
    )


def record_to_config(record: PlacementRecord) -> AdsorptionConfig:
    """Build an AdsorptionConfig that matches the computation context."""
    ctx = record.context
    cfg = AdsorptionConfig()
    cfg.model_name = ctx.model_name
    cfg.fmax = ctx.fmax
    cfg.stage1_steps = ctx.stage1_steps
    cfg.stage2_steps = ctx.stage2_steps
    cfg.device = ctx.device
    cfg.seed = ctx.seed
    cfg.placement_z_range = ctx.placement_z_range
    cfg.placement_z_scale_by_covalent_radius = ctx.placement_z_scale_by_covalent_radius
    cfg.min_initial_distance = ctx.min_initial_distance
    cfg.min_contact_ratio = ctx.min_contact_ratio
    cfg.top_layer_tolerance = ctx.top_layer_tolerance
    cfg.symmetry_tolerance = ctx.symmetry_tolerance
    cfg.site_equivalence_tolerance = ctx.site_equivalence_tolerance
    cfg.hollow_site_dedup_tolerance = ctx.hollow_site_dedup_tolerance
    cfg.planar_z_variance_threshold = ctx.planar_z_variance_threshold
    return cfg

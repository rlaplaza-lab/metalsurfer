"""Pose construction, validation, and finalization."""

from .. import geometry as geom
from .build import (
    _context_from_pose as _context_from_pose,
)
from .build import (
    _pose_from_spec as _pose_from_spec,
)
from .build import (
    build_pose_batch_cache as build_pose_batch_cache,
)
from .checks import (
    _build_slab_distance_scratch as _build_slab_distance_scratch,
)
from .checks import (
    _saturation_exclude_count as _saturation_exclude_count,
)
from .checks import (
    _validate_posed_adsorbate as _validate_posed_adsorbate,
)
from .context import (
    _HeightInterval as _HeightInterval,
)
from .context import (
    _PlacementContext as _PlacementContext,
)
from .context import (
    _PoseBatchCache as _PoseBatchCache,
)
from .context import (
    _require_pose_z_abs as _require_pose_z_abs,
)
from .finalize import (
    _descriptor_from_placement as _descriptor_from_placement,
)
from .finalize import (
    _finalize_placement as _finalize_placement,
)
from .finalize import (
    generate_placement_from_pose as generate_placement_from_pose,
)
from .height import (
    _com_floor_on_support_plane as _com_floor_on_support_plane,
)
from .height import (
    _com_height_from_z_fraction as _com_height_from_z_fraction,
)
from .height import (
    _contact_atom_index as _contact_atom_index,
)
from .height import (
    _contact_shell_indices as _contact_shell_indices,
)
from .height import (
    _count_contact_atoms_at_com_height as _count_contact_atoms_at_com_height,
)
from .height import (
    _feasible_height_interval as _feasible_height_interval,
)
from .height import (
    _framework_plane_height as _framework_plane_height,
)
from .height import (
    _height_above_supports as _height_above_supports,
)
from .height import (
    _height_interval_family_key as _height_interval_family_key,
)
from .height import (
    _mol_positions_at_com_height as _mol_positions_at_com_height,
)
from .height import (
    _pair_min_distance_at_com_height as _pair_min_distance_at_com_height,
)
from .height import (
    _pair_worst_penetration_at_com_height as _pair_worst_penetration_at_com_height,
)
from .height import (
    _pairwise_contact_com_height as _pairwise_contact_com_height,
)
from .height import (
    _placement_normal_hat as _placement_normal_hat,
)
from .height import (
    _resolve_surface_ref as _resolve_surface_ref,
)
from .height import (
    _void_height_interval_analytic as _void_height_interval_analytic,
)
from .height import (
    _z_fraction_from_com_height as _z_fraction_from_com_height,
)
from .recovery import (
    _analytic_height_recovery as _analytic_height_recovery,
)
from .recovery import (
    _apply_lateral_offset as _apply_lateral_offset,
)
from .recovery import (
    _clash_recovery_fixed_cloud as _clash_recovery_fixed_cloud,
)
from .recovery import (
    _contact_penetration as _contact_penetration,
)
from .recovery import (
    _contact_penetration_detail as _contact_penetration_detail,
)
from .recovery import (
    _placement_normal as _placement_normal,
)
from .recovery import (
    _recover_distance_failure as _recover_distance_failure,
)
from .recovery import (
    _recover_z_offset as _recover_z_offset,
)
from .recovery import (
    _set_adsorbate_at_center as _set_adsorbate_at_center,
)
from .recovery import (
    _try_clash_descent_recovery as _try_clash_descent_recovery,
)
from .recovery import (
    _xy_recovery_offsets as _xy_recovery_offsets,
)

__all__ = [
    "build_pose_batch_cache",
    "generate_placement_from_pose",
    "geom",
]

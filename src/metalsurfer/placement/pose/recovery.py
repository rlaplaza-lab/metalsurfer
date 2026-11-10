"""Distance and clash recovery for posed adsorbates."""

import dataclasses
import logging
import random

import numpy as np
from ase import Atoms

from ...config import AdsorptionConfig
from .. import geometry as geom
from .._constants import (
    _CONTACT_CLEARANCE_PAD_ANGSTROM,
    _DISTANCE_RECOVERY_XY_ATTEMPTS,
    _DISTANCE_ZERO_EPS,
    _LATERAL_OFFSET_REF_SWITCH_DOT,
    _RECOVERY_INPLANE_PENETRATION_DOT,
    _RECOVERY_NORMAL_PENETRATION_WINDOW_FACTOR,
    _VECTOR_NORM_EPS,
    _XY_RECOVERY_PLACEMENT_MIXER,
    _XY_RECOVERY_SEED_MIXER,
    _XY_RECOVERY_SITE_MIXER,
    RECOVERABLE_DISTANCE_REASONS,
)
from ..clash import (
    atom_radii_for_symbols,
    clash_bounds_for_adsorbate,
    compose_quaternion_with_azimuth,
    resolve_rigid_clash,
)
from ..cutoffs import (
    mixed_covalent_thresholds,
    pair_thresholds,
    radii_or_cached,
    tabulated_radii,
)
from ..occupancy import incoming_inplane_radius
from ..site_coords import _frac_to_cart, _slab_normal, _wrapped_slab_fractional
from .checks import (
    _build_slab_distance_scratch,
    _saturation_exclude_count,
    _validate_posed_adsorbate,
)
from .context import _PlacementContext, _PoseBatchCache, _require_pose_z_abs
from .height import (
    _placement_normal_hat,
    _z_fraction_from_com_height,
)

logger = logging.getLogger(__name__)


def _recover_z_offset(
    ctx: _PlacementContext,
    z_abs: float,
) -> float:
    """Recover COM height above the surface reference from absolute placement.

    Returned ``z_offset`` is the adsorbate COM displacement above
    *surface_ref* along the placement normal (site frame). It includes any
    clearance lift applied at placement time, so ``surface_ref + z_offset``
    reconstructs the COM height along that normal — not the closest-atom gap.
    """
    pose = ctx.pose
    placement = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    n_hat = _placement_normal_hat(ctx.normal)
    return float(np.dot(placement, n_hat) - ctx.surface_ref)


def _placement_normal(
    ctx: _PlacementContext,
    slab: Atoms,
) -> np.ndarray:
    """Return unit normal used for height/lateral recovery (site frame)."""
    n_hat = _placement_normal_hat(ctx.normal)
    if float(np.linalg.norm(np.asarray(ctx.normal, dtype=float))) > _VECTOR_NORM_EPS:
        return n_hat
    # Degenerate stored normal: fall back to slab-cell +z.
    return _placement_normal_hat(_slab_normal(np.asarray(slab.get_cell(), dtype=float)))


def _contact_penetration(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    material_type: str,
    slab_scratch: geom._SlabDistanceScratch | None,
    slab_for_sites: Atoms | None,
) -> tuple[float, float]:
    """Return ``(actual_min_distance, max_pair_penetration)`` vs the substrate gate."""
    actual_min, max_pen, _dot = _contact_penetration_detail(
        adsorbate,
        slab,
        config,
        material_type=material_type,
        slab_scratch=slab_scratch,
        slab_for_sites=slab_for_sites,
        normal=None,
    )
    return actual_min, max_pen


def _contact_penetration_detail(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    material_type: str,
    slab_scratch: geom._SlabDistanceScratch | None,
    slab_for_sites: Atoms | None,
    normal: np.ndarray | None,
) -> tuple[float, float, float]:
    """Return ``(actual_min, max_penetration, |n·pair_dir|)`` for the worst pair.

    ``|n·pair_dir|`` is 1.0 when *normal* is omitted or no penetrating pair exists.
    """
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    mol_pos, slab_pos, mol_syms, slab_syms, _cell, _pbc, dists = (
        geom._mol_slab_contact_arrays(
            adsorbate,
            slab,
            material_type=material_type,
            exclude_slab_atoms=exclude_n,
            slab_scratch=slab_scratch,
        )
    )
    if dists.size == 0:
        return float("inf"), 0.0, 1.0
    mol_r = tabulated_radii(mol_syms, kind="covalent")
    slab_r = radii_or_cached(
        slab_syms,
        None if slab_scratch is None else slab_scratch.slab_cov_r,
        kind="covalent",
    )
    _clears, actual_min, max_pen, abs_dot = geom.covalent_floor_contact_metrics(
        dists,
        mol_pos,
        slab_pos,
        mol_r,
        slab_r,
        min_distance=float(config.min_initial_distance),
        min_contact_ratio=float(config.min_contact_ratio),
        on_floor_pad=_CONTACT_CLEARANCE_PAD_ANGSTROM,
        normal=None if normal is None else _placement_normal_hat(normal),
    )
    return actual_min, max_pen, abs_dot


def _analytic_height_recovery(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    height_mode: str,
    *,
    slab_for_sites: Atoms | None,
    slab_scratch: geom._SlabDistanceScratch | None,
) -> tuple[np.ndarray, float] | None:
    """One signed height nudge along the placement normal; None if nothing to fix."""
    pose = ctx.pose
    z_abs = _require_pose_z_abs(pose)
    origin = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    com_lo = ctx.com_lo
    com_nominal = ctx.com_nominal
    com_hi = ctx.com_hi
    if com_lo is None or com_nominal is None or com_hi is None:
        # Replay / synthetic contexts without a cached interval: build a
        # symmetric window around the posed COM so nudges and z_fraction
        # remain inverses of each other.
        z_span = float(ctx.z_base_hi - ctx.z_base_lo)
        if z_span <= _DISTANCE_ZERO_EPS:
            return None
        com_h0 = float(np.dot(origin, _placement_normal(ctx, slab)))
        half = 0.5 * z_span
        com_lo = com_h0 - half
        com_nominal = com_h0
        com_hi = com_h0 + half
    if float(com_hi) + _DISTANCE_ZERO_EPS < float(com_lo):
        return None

    actual, max_pen = _contact_penetration(
        adsorbate,
        slab,
        config,
        material_type=ctx.mat_type,
        slab_scratch=slab_scratch,
        slab_for_sites=slab_for_sites,
    )
    # Void sites: nudge toward the free-volume centre when too close. Wall:
    # raise away from the support plane. Keyed on site.kind, not material_type.
    is_void = ctx.site is not None and ctx.site.kind == "void"
    if height_mode == "too_close":
        if max_pen <= _DISTANCE_ZERO_EPS:
            return None
        signed = -max_pen if is_void else max_pen
    elif height_mode == "contact_distance_too_large":
        target = float(config.max_closest_approach)
        excess = actual - target
        if excess <= _DISTANCE_ZERO_EPS:
            return None
        signed = excess if is_void else -excess
    else:
        max_d = config.max_initial_distance
        if max_d is None:
            # Fall back to contact window when no hard max_initial_distance.
            if height_mode == "too_far" and config.strict_initial_placement:
                target = float(config.max_closest_approach)
                excess = actual - target
                if excess <= _DISTANCE_ZERO_EPS:
                    return None
                signed = excess if is_void else -excess
            else:
                return None
        else:
            excess = actual - float(max_d)
            if excess <= _DISTANCE_ZERO_EPS:
                return None
            signed = excess if is_void else -excess

    n_hat = _placement_normal(ctx, slab)
    com_h = float(np.dot(origin, n_hat))
    new_com_h = float(min(float(com_hi), max(float(com_lo), com_h + float(signed))))
    clipped_center = origin + (new_com_h - com_h) * n_hat
    new_zf = _z_fraction_from_com_height(
        new_com_h, float(com_lo), float(com_nominal), float(com_hi)
    )
    return clipped_center, new_zf


def _xy_recovery_offsets(
    config: AdsorptionConfig,
    *,
    placement_index: int,
    site_index: int,
) -> list[tuple[float, float]]:
    """Deterministic in-plane recovery offsets within configured XY ranges."""
    x_lo, x_hi = config.placement_x_range
    y_lo, y_hi = config.placement_y_range
    if abs(x_hi - x_lo) < _DISTANCE_ZERO_EPS and abs(y_hi - y_lo) < _DISTANCE_ZERO_EPS:
        return []
    rng = random.Random(
        (int(config.seed) * _XY_RECOVERY_SEED_MIXER)
        ^ (int(placement_index) * _XY_RECOVERY_PLACEMENT_MIXER)
        ^ (int(site_index) * _XY_RECOVERY_SITE_MIXER)
    )
    return [
        (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
        for _ in range(_DISTANCE_RECOVERY_XY_ATTEMPTS)
    ]


def _apply_lateral_offset(
    center: np.ndarray,
    *,
    dx: float,
    dy: float,
    ctx: _PlacementContext,
    slab: Atoms,
    pose_cache: _PoseBatchCache | None = None,
) -> np.ndarray:
    """Apply a lateral recovery offset in the plane perpendicular to the site/slab normal."""
    n_hat = _placement_normal(ctx, slab)
    # Build an orthonormal in-plane basis from Cartesian dx/dy.
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n_hat))) > _LATERAL_OFFSET_REF_SWITCH_DOT:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n_hat, ref)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n_hat, u)
    shifted = np.asarray(center, dtype=float) + float(dx) * u + float(dy) * v
    if ctx.mat_type == "slab":
        cell = (
            pose_cache.cell
            if pose_cache is not None and pose_cache.cell is not None
            else np.asarray(slab.get_cell(), dtype=float)
        )
        return _frac_to_cart(_wrapped_slab_fractional(shifted, cell), cell)
    return shifted


def _set_adsorbate_at_center(
    adsorbate: Atoms,
    rotated_pos: np.ndarray,
    center: np.ndarray,
) -> None:
    """Translate COM-centred *rotated_pos* so the COM sits at *center*."""
    center_arr = np.asarray(center, dtype=float).reshape(3)
    adsorbate.set_positions(np.asarray(rotated_pos, dtype=float) + center_arr)


def _recover_distance_failure(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    fail_reason: str,
    *,
    slab_for_sites: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
    slab_scratch: geom._SlabDistanceScratch | None = None,
) -> tuple[_PlacementContext, str | None]:
    """One analytic height nudge, then clash descent (or XY if clash is off).

    ``too_close`` / ``too_far`` / ``contact_distance_too_large`` / ``vdw_overlap``
    try a single height shift first (skipped when the worst penetration is
    mostly in-plane). Huge normal penetration fails cheaply before Packmol
    clash. Remaining recoverable failures use clash descent when enabled;
    otherwise discrete XY jitter. Pore sites invert the height nudge toward
    the free-volume centre (``site.kind == "void"``), not ``material_type``.
    """
    if fail_reason not in RECOVERABLE_DISTANCE_REASONS:
        return ctx, fail_reason
    height_reasons: tuple[str, ...] = (
        "too_close",
        "too_far",
        "contact_distance_too_large",
        "vdw_overlap",
    )
    height_mode = "too_close" if fail_reason == "vdw_overlap" else fail_reason

    pose = ctx.pose
    origin = np.array([pose.x_abs, pose.y_abs, _require_pose_z_abs(pose)], dtype=float)
    work_zf = float(pose.z_fraction)
    work_center = origin.copy()
    last_reason: str | None = fail_reason
    n_hat = _placement_normal(ctx, slab)
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)

    _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, origin)
    _actual, max_pen, abs_dot = _contact_penetration_detail(
        adsorbate,
        slab,
        config,
        material_type=ctx.mat_type,
        slab_scratch=slab_scratch,
        slab_for_sites=slab_for_sites,
        normal=n_hat,
    )

    # Deep normal penetration: skip clash descent.
    if (
        fail_reason == "too_close"
        and z_span > _DISTANCE_ZERO_EPS
        and max_pen > _RECOVERY_NORMAL_PENETRATION_WINDOW_FACTOR * z_span
        and abs_dot >= _RECOVERY_INPLANE_PENETRATION_DOT
    ):
        return ctx, "too_close"

    try_height = fail_reason in height_reasons
    # Mostly in-plane penetration → clash/XY, not height nudge.
    if (
        try_height
        and fail_reason in ("too_close", "vdw_overlap")
        and max_pen > _DISTANCE_ZERO_EPS
        and abs_dot < _RECOVERY_INPLANE_PENETRATION_DOT
    ):
        try_height = False

    if try_height:
        height_shift = _analytic_height_recovery(
            ctx,
            adsorbate,
            slab,
            config,
            height_mode,
            slab_for_sites=slab_for_sites,
            slab_scratch=slab_scratch,
        )
        if height_shift is not None:
            work_center, work_zf = height_shift
            _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, work_center)
            last_reason = _validate_posed_adsorbate(
                adsorbate,
                slab,
                config,
                slab_for_sites=slab_for_sites,
                material_type=ctx.mat_type,
                slab_scratch=slab_scratch,
            )
            if last_reason is None:
                new_pose = dataclasses.replace(
                    pose,
                    x_abs=float(work_center[0]),
                    y_abs=float(work_center[1]),
                    z_abs=float(work_center[2]),
                    z_fraction=float(work_zf),
                )
                return dataclasses.replace(ctx, pose=new_pose), None
            if last_reason not in RECOVERABLE_DISTANCE_REASONS:
                return ctx, last_reason

    _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, work_center)

    if config.placement_clash_descent and last_reason in (
        "adsorbate_overlap",
        "vdw_overlap",
        "too_close",
    ):
        ctx_out, last_reason = _try_clash_descent_recovery(
            ctx,
            adsorbate,
            slab,
            config,
            fail_reason=last_reason or fail_reason,
            work_center=work_center,
            work_zf=work_zf,
            slab_for_sites=slab_for_sites,
            slab_scratch=slab_scratch,
        )
        if last_reason is None:
            return ctx_out, None
        if last_reason not in RECOVERABLE_DISTANCE_REASONS:
            return ctx, last_reason
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, work_center)

    for dx, dy in _xy_recovery_offsets(
        config,
        placement_index=pose.placement_index,
        site_index=pose.site_index,
    ):
        center = _apply_lateral_offset(
            work_center,
            dx=dx,
            dy=dy,
            ctx=ctx,
            slab=slab,
            pose_cache=pose_cache,
        )
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, center)
        last_reason = _validate_posed_adsorbate(
            adsorbate,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            material_type=ctx.mat_type,
            slab_scratch=slab_scratch,
        )
        if last_reason is None:
            new_pose = dataclasses.replace(
                pose,
                x_abs=float(center[0]),
                y_abs=float(center[1]),
                z_abs=float(center[2]),
                z_fraction=float(work_zf),
            )
            return dataclasses.replace(ctx, pose=new_pose), None
        if last_reason not in RECOVERABLE_DISTANCE_REASONS:
            return ctx, last_reason

    return ctx, last_reason


def _recovery_thresholds(
    adsorbate: Atoms,
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    n_substrate: int,
    use_vdw: bool,
    config: AdsorptionConfig,
) -> np.ndarray:
    """Return acceptance cutoffs for clash recovery.

    A van der Waals rescue scales substrate columns only. Adsorbate columns
    stay on the covalent fusion cutoff.
    """
    if not use_vdw:
        return mixed_covalent_thresholds(
            moving_radii,
            fixed_radii,
            n_substrate=int(n_substrate),
            substrate_scale=float(config.min_contact_ratio),
            adsorbate_scale=float(config.connectivity_multiplier),
            absolute_floor=float(config.min_initial_distance),
        )
    n_sub = int(n_substrate)
    n_fix = int(fixed_radii.shape[0])
    thresholds = np.empty((int(moving_radii.shape[0]), n_fix), dtype=float)
    if n_sub:
        thresholds[:, :n_sub] = pair_thresholds(
            moving_radii,
            fixed_radii[:n_sub],
            float(config.vdw_overlap_scale),
        )
    if n_sub < n_fix:
        moving_cov = atom_radii_for_symbols(list(adsorbate.get_chemical_symbols()))
        thresholds[:, n_sub:] = pair_thresholds(
            moving_cov,
            fixed_radii[n_sub:],
            float(config.connectivity_multiplier),
        )
    return thresholds


def _try_clash_descent_recovery(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    fail_reason: str,
    work_center: np.ndarray,
    work_zf: float,
    slab_for_sites: Atoms | None,
    slab_scratch: geom._SlabDistanceScratch | None,
) -> tuple[_PlacementContext, str | None]:
    """Attempt bounded rigid-body clash descent; return updated ctx or last reason."""
    if slab_scratch is None:
        exclude_n = _saturation_exclude_count(slab, slab_for_sites)
        slab_scratch = _build_slab_distance_scratch(slab, exclude_n, ctx.mat_type)

    use_vdw = fail_reason == "vdw_overlap"
    z_window = max(0.0, float(ctx.z_base_hi - ctx.z_base_lo))
    footprint = incoming_inplane_radius(
        adsorbate,
        footprint_scale=float(config.occupancy_footprint_scale),
    )
    moving_r = atom_radii_for_symbols(
        list(adsorbate.get_chemical_symbols()),
        use_vdw=use_vdw,
    )
    if moving_r.size == 0:
        raise ValueError("clash recovery requires a non-empty adsorbate")
    cutoff = 2.0 * float(np.max(moving_r)) + z_window
    fixed_pos, fixed_radii, n_substrate = _clash_recovery_fixed_cloud(
        slab,
        slab_scratch,
        ads_com=np.asarray(work_center, dtype=float),
        use_vdw=use_vdw,
        neighbor_cutoff=cutoff,
    )
    if fixed_pos is None or fixed_radii is None or n_substrate is None:
        return ctx, fail_reason
    thresholds = _recovery_thresholds(
        adsorbate,
        moving_r,
        fixed_radii,
        n_substrate=n_substrate,
        use_vdw=use_vdw,
        config=config,
    )

    bounds = clash_bounds_for_adsorbate(
        adsorbate,
        config,
        z_window=z_window,
        footprint_radius=footprint,
        moving_radii=moving_r,
    )
    site_frame = geom.compute_surface_site_frame(
        _placement_normal(ctx, slab),
        tangent_basis=ctx.site.tangent_basis if ctx.site is not None else None,
    )
    new_pos, az_delta, ok = resolve_rigid_clash(
        adsorbate,
        fixed_pos,
        fixed_radii,
        origin=np.asarray(work_center, dtype=float),
        site_frame=site_frame,
        cell=slab_scratch.cell,
        pbc=slab_scratch.pbc,
        config=config,
        thresholds=thresholds,
        use_vdw_moving=use_vdw,
        bounds=bounds,
        moving_radii=moving_r,
    )
    if not ok:
        return ctx, fail_reason

    new_center = np.mean(new_pos, axis=0)
    new_rotated = new_pos - new_center
    adsorbate.set_positions(new_pos)

    pose = ctx.pose
    quat_w, quat_x, quat_y, quat_z = compose_quaternion_with_azimuth(
        (pose.quat_w, pose.quat_x, pose.quat_y, pose.quat_z),
        az_delta,
        _placement_normal(ctx, slab),
    )
    new_pose = dataclasses.replace(
        pose,
        x_abs=float(new_center[0]),
        y_abs=float(new_center[1]),
        z_abs=float(new_center[2]),
        z_fraction=float(work_zf),
        quat_w=quat_w,
        quat_x=quat_x,
        quat_y=quat_y,
        quat_z=quat_z,
    )
    new_ctx = dataclasses.replace(ctx, pose=new_pose, rotated_pos=new_rotated)
    last_reason = _validate_posed_adsorbate(
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        material_type=ctx.mat_type,
        slab_scratch=slab_scratch,
    )
    if last_reason is None:
        return new_ctx, None
    # restore adsorbate coords
    _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, work_center)
    return ctx, last_reason


def _clash_recovery_fixed_cloud(
    slab: Atoms,
    slab_scratch: geom._SlabDistanceScratch,
    *,
    ads_com: np.ndarray,
    use_vdw: bool,
    neighbor_cutoff: float,
) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
    """Return nearby ``(positions, radii, n_substrate)`` for clash recovery."""
    fixed_chunks: list[np.ndarray] = []
    radius_chunks: list[np.ndarray] = []
    cutoff = float(neighbor_cutoff)
    n_substrate = 0

    def _nearby_mask(positions: np.ndarray) -> np.ndarray:
        if positions.size == 0:
            return np.zeros(0, dtype=bool)
        dists = geom._mol_slab_pairwise_distances(
            ads_com.reshape(1, 3),
            positions,
            slab_scratch.cell,
            slab_scratch.pbc,
        ).reshape(-1)
        return dists <= cutoff

    near = _nearby_mask(slab_scratch.slab_pos)
    if np.any(near):
        n_substrate = int(np.count_nonzero(near))
        fixed_chunks.append(slab_scratch.slab_pos[near])
        scratch_r = slab_scratch.slab_vdw_r if use_vdw else slab_scratch.slab_cov_r
        syms = [s for s, keep in zip(slab_scratch.slab_syms, near, strict=True) if keep]
        cached = None if scratch_r is None else np.asarray(scratch_r, dtype=float)[near]
        radius_chunks.append(
            radii_or_cached(syms, cached, kind="vdw" if use_vdw else "covalent")
        )

    if slab_scratch.pre_ads_pos is not None and slab_scratch.pre_ads_pos.size:
        near_pre = _nearby_mask(slab_scratch.pre_ads_pos)
        if np.any(near_pre):
            fixed_chunks.append(slab_scratch.pre_ads_pos[near_pre])
            exclude_n = len(slab_scratch.slab_pos)
            pre_syms = list(slab.get_chemical_symbols()[exclude_n:])
            pre_syms_near = [
                s for s, keep in zip(pre_syms, near_pre, strict=True) if keep
            ]
            # Fusion stays on covalent radii. Van der Waals radii are only
            # the substrate steric gate.
            radius_chunks.append(atom_radii_for_symbols(pre_syms_near, use_vdw=False))

    if not fixed_chunks:
        return None, None, None
    return np.vstack(fixed_chunks), np.concatenate(radius_chunks), n_substrate

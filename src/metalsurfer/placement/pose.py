"""Pose construction, validation, and finalization."""

import dataclasses
import logging
import random
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementPose, PlacementSpec
from . import geometry as geom
from ._constants import (
    _DISTANCE_RECOVERY_HEIGHT_STEPS,
    _DISTANCE_RECOVERY_XY_ATTEMPTS,
    _DISTANCE_ZERO_EPS,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _VECTOR_NORM_EPS,
)
from ._material import material_aware_pbc, material_type_for_placement
from .orientation import (
    _is_flat_aromatic,
    _parallel_z_adjustments,
    _site_type_z_offset,
    orient_from_spec,
)
from .site_context import SiteContext, _get_unique_sites_for_specs
from .site_coords import _slab_normal, _slab_plane_projectors
from .site_enumeration import (
    _compute_site_z_base,
    _height_along_slab_normal,
    _is_top_layer_planar,
)
from .site_types import Site

logger = logging.getLogger(__name__)


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
    z_base_lo: float = 0.0
    z_base_hi: float = 0.0
    normal: np.ndarray | None = None


def _clearance_lift_along_normal(
    rotated_pos: np.ndarray,
    normal: np.ndarray,
) -> float:
    """Return how far to lift the COM so the closest atom sits at the intended height.

    ``rotated_pos`` must be COM-centred. Atoms with negative projection on *normal*
    protrude toward the surface; the lift equals ``max(0, -min(r · n))`` so that
    ``z_offset`` / ``z_fraction`` refer to the closest adsorbate atom rather than
    the molecular COM.
    """
    n = np.asarray(normal, dtype=float)
    nrm = float(np.linalg.norm(n))
    if nrm <= _VECTOR_NORM_EPS:
        return 0.0
    n_hat = n / nrm
    heights = np.asarray(rotated_pos, dtype=float) @ n_hat
    if heights.size == 0:
        return 0.0
    return float(max(0.0, -float(np.min(heights))))


def _resolve_surface_ref(
    site: Site | None,
    slab: Atoms,
    mat_type: str,
    *,
    rough_slab_local_z: bool = False,
) -> tuple[float, bool]:
    """Return *(surface_ref, is_local_ref)* for z-offset calculations.

    For slabs the reference is the topmost height along the slab normal so that
    z_offset is the gap above the surface layer.  For nanoparticles and porous
    materials the reference is the Voronoi vertex projected onto the local site
    normal (matching placement along that normal).

    When *rough_slab_local_z* is True and the slab is non-planar, use the
    site's own height along the normal instead of the global maximum.  This
    prevents step-edge sites from getting excessive z offsets.
    """
    if mat_type == "slab":
        cell = np.asarray(slab.get_cell(), dtype=float)
        positions = slab.get_positions()
        if rough_slab_local_z and site is not None and not _is_top_layer_planar(slab):
            return float(_height_along_slab_normal(site.xyz, cell)), True
        return float(np.max(_height_along_slab_normal(positions, cell))), False
    if site is not None:
        site_xyz = np.asarray(site.xyz, dtype=float)
        site_normal = np.asarray(site.normal, dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        if nrm > _VECTOR_NORM_EPS:
            return float(np.dot(site_xyz, site_normal / nrm)), True
        return float(site_xyz[2]), True
    return float(np.max(slab.get_positions()[:, 2])), False


def _pose_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    z_fraction: float | None = None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
) -> tuple[_PlacementContext | None, str | None]:
    """Build a placement context (pose + resolved geometry) from a spec.

    Returns ``(ctx, None)`` on success, or ``(None, reason)`` when placement
    cannot proceed (``"no_sites_found"`` or ``"invalid_site_index"``).
    """
    ads_pos = adsorbate.get_positions().copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
    normal = np.array([0.0, 0.0, 1.0])
    shape, _, _ = geom._classify_molecule_shape(canonical_pos)

    ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
    if not ctx.use_sites or len(ctx.sites) == 0:
        logger.debug(
            "No sites available for spec placement_index=%d",
            spec.placement_index,
        )
        return None, "no_sites_found"
    if not (0 <= spec.site_index < len(ctx.sites)):
        logger.debug(
            "Site_index=%d out of range for %d sites (placement_index=%d)",
            spec.site_index,
            len(ctx.sites),
            spec.placement_index,
        )
        return None, "invalid_site_index"
    site = ctx.sites[spec.site_index]

    site_normal = np.asarray(site.normal, dtype=float)
    if np.linalg.norm(site_normal) > _VECTOR_NORM_EPS:
        normal = site_normal / np.linalg.norm(site_normal)

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    placement_reference_slab = slab_for_sites if slab_for_sites is not None else slab

    z_base_lo, z_base_hi = _compute_site_z_base(
        config, placement_reference_slab, site, symbols
    )
    if spec.site_type:
        offset = _site_type_z_offset(placement_reference_slab, site, spec.site_type)
        z_base_lo += offset
        z_base_hi += offset

    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)
    if flat_aromatic and spec.orientation_type == "parallel" and mat_type != "porous":
        z_floor, z_lo_shrink, z_hi_shrink = _parallel_z_adjustments(
            placement_reference_slab, site, symbols
        )
        z_base_lo = max(z_floor, z_base_lo - z_lo_shrink)
        z_base_hi = max(
            z_base_lo + _PARALLEL_Z_MIN_HI_MARGIN,
            z_base_hi - z_hi_shrink,
        )

    zf = spec.z_fraction if z_fraction is None else z_fraction
    z_offset = z_base_lo + zf * (z_base_hi - z_base_lo)

    # Slab: top-layer z (Voronoi vertex z can sit between layers). NP/pore: local vertex.
    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        placement_reference_slab,
        mat_type,
        rough_slab_local_z=config.rough_slab_local_z,
    )

    oriented = orient_from_spec(
        canonical_pos, normal=normal, symbols=symbols, spec=spec
    )
    rotated_pos = oriented.rotated_pos
    quat = oriented.quat

    # Clearance-aware lift: place so the closest atom (not the COM) sits at
    # z_offset. Skipped for porous frameworks — local normals are not a unique
    # "away from wall" direction inside confined pores.
    # Descriptor z_offset recovered later is still COM height above surface_ref
    # (includes this lift).
    apply_lift = mat_type != "porous"

    if mat_type == "slab":
        cell = np.asarray(placement_reference_slab.get_cell(), dtype=float)
        n_hat = _slab_normal(cell)
        lift = _clearance_lift_along_normal(rotated_pos, n_hat) if apply_lift else 0.0
        base = np.asarray(site.xyz, dtype=float)
        base_h = float(np.dot(base, n_hat))
        # Intended height of the closest atom; COM sits higher by *lift*.
        target_h = float(surface_ref + z_offset + lift)
        placement_center = base + (target_h - base_h) * n_hat
    else:
        lift = _clearance_lift_along_normal(rotated_pos, normal) if apply_lift else 0.0
        placement_center = (
            np.asarray(site.xyz, dtype=float) + float(z_offset + lift) * normal
        )

    pose = PlacementPose(
        conformer_index=spec.conformer_index,
        site_index=spec.site_index,
        site_type=spec.site_type,
        placement_index=spec.placement_index,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
        x_abs=float(placement_center[0]),
        y_abs=float(placement_center[1]),
        z_fraction=float(zf),
        z_abs=float(placement_center[2]),
        orientation_type=spec.orientation_type,
        face_flip=spec.face_flip,
        en_atom_index=spec.en_atom_index,
        tilt_deg=spec.tilt_deg,
        azimuth_deg=spec.azimuth_deg,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
    )
    return (
        _PlacementContext(
            pose=pose,
            site=site,
            mat_type=mat_type,
            surface_ref=float(surface_ref),
            is_local_ref=is_local_ref,
            source=ctx.source,
            canonical_pos=canonical_pos,
            use_sites=True,
            rotated_pos=rotated_pos,
            z_base_lo=float(z_base_lo),
            z_base_hi=float(z_base_hi),
            normal=np.asarray(normal, dtype=float),
        ),
        None,
    )


def _context_from_pose(
    pose: PlacementPose,
    canonical_pos: np.ndarray,
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None,
) -> _PlacementContext | None:
    """Replay path: normalize quaternion, rotate *canonical_pos*, resolve site for ``_finalize_placement``."""
    raw_q = np.array([pose.quat_w, pose.quat_x, pose.quat_y, pose.quat_z], dtype=float)
    if float(np.linalg.norm(raw_q)) < _VECTOR_NORM_EPS:
        logger.warning(
            "Degenerate quaternion (norm < %.1e) for placement_index=%d; skipping",
            _VECTOR_NORM_EPS,
            pose.placement_index,
        )
        return None
    quat = geom.normalize_quaternion(raw_q)
    rotated_pos = (geom.quaternion_to_rotation_matrix(quat) @ canonical_pos.T).T

    ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
    site = None
    if ctx.use_sites and 0 <= pose.site_index < len(ctx.sites):
        site = ctx.sites[pose.site_index]
    mat_type = material_type_for_placement(site, when_no_site=config.material_type)
    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        slab,
        mat_type,
        rough_slab_local_z=config.rough_slab_local_z,
    )

    pose_normalized = dataclasses.replace(
        pose,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
    )

    return _PlacementContext(
        pose=pose_normalized,
        site=site,
        mat_type=mat_type,
        surface_ref=float(surface_ref),
        is_local_ref=is_local_ref,
        source=ctx.source if ctx.use_sites else "no_sites",
        canonical_pos=canonical_pos,
        use_sites=ctx.use_sites,
        rotated_pos=rotated_pos,
    )


def _recover_z_offset(
    ctx: _PlacementContext, z_abs: float, slab: Atoms | None = None
) -> float:
    """Recover COM height above the surface reference from absolute placement.

    Returned ``z_offset`` is the adsorbate COM displacement above
    *surface_ref* (along the slab/site normal).  It includes any clearance
    lift applied at placement time, so ``surface_ref + z_offset`` reconstructs
    the COM height along that normal — not the closest-atom gap.

    For slabs this is ``dot(placement, n_slab) - surface_ref``.  For
    nanoparticle / pore sites it is ``dot(placement, n_site) - surface_ref``
    when the site normal is usable (``surface_ref`` is the site projected onto
    the same normal).
    """
    pose = ctx.pose
    site = ctx.site
    placement = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    if ctx.mat_type == "slab" and slab is not None:
        cell = np.asarray(slab.get_cell(), dtype=float)
        n_hat = _slab_normal(cell)
        return float(np.dot(placement, n_hat) - ctx.surface_ref)
    if site is not None:
        site_normal = np.asarray(site.normal, dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        if nrm > _VECTOR_NORM_EPS:
            return float(np.dot(placement, site_normal / nrm) - ctx.surface_ref)
    return float(z_abs - ctx.surface_ref)


def _saturation_exclude_count(
    slab: Atoms,
    slab_for_sites: Atoms | None,
) -> int | None:
    """Return substrate atom count for saturation exclude, else None."""
    if slab_for_sites is None:
        return None
    n_sub = len(slab_for_sites)
    if n_sub < len(slab):
        return n_sub
    return None


def _placement_normal(ctx: _PlacementContext, slab: Atoms) -> np.ndarray:
    """Return unit normal used for height/lateral recovery."""
    if ctx.mat_type == "slab":
        return _slab_normal(np.asarray(slab.get_cell(), dtype=float))
    normal = ctx.normal
    if normal is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    normal = np.asarray(normal, dtype=float)
    nrm = float(np.linalg.norm(normal))
    if nrm > _VECTOR_NORM_EPS:
        return normal / nrm
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _center_with_height_delta(
    center: np.ndarray,
    *,
    old_z_fraction: float,
    new_z_fraction: float,
    ctx: _PlacementContext,
    slab: Atoms,
) -> np.ndarray:
    """Shift *center* along the placement normal by the height-window delta."""
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)
    delta_h = (new_z_fraction - old_z_fraction) * z_span
    if abs(delta_h) < _DISTANCE_ZERO_EPS:
        return np.asarray(center, dtype=float).copy()
    n_hat = _placement_normal(ctx, slab)
    return np.asarray(center, dtype=float) + float(delta_h) * n_hat


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
        (int(config.seed) * 1_000_003)
        ^ (int(placement_index) * 97)
        ^ (int(site_index) * 1_009)
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
) -> np.ndarray:
    """Apply a lateral recovery offset in the plane perpendicular to the site/slab normal."""
    n_hat = _placement_normal(ctx, slab)
    # Build an orthonormal in-plane basis from Cartesian dx/dy.
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n_hat, ref)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n_hat, u)
    shifted = np.asarray(center, dtype=float) + float(dx) * u + float(dy) * v
    if ctx.mat_type == "slab":
        cell = np.asarray(slab.get_cell(), dtype=float)
        # MIC-wrap the in-plane a–b components; keep height along normal.
        pinv_ab_T, _ = _slab_plane_projectors(cell)
        frac2 = shifted @ pinv_ab_T
        frac2 = np.mod(frac2, 1.0)
        # Reconstruct Cartesian from fractional a,b plus original height along normal.
        planar = frac2[0] * cell[0] + frac2[1] * cell[1]
        h = float(np.dot(shifted, n_hat))
        base_h = float(np.dot(planar, n_hat))
        return planar + (h - base_h) * n_hat
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
) -> tuple[_PlacementContext, str | None]:
    """Nudge height then XY after distance/overlap failures; return updated ctx or last reason.

    ``too_close`` / ``too_far`` try height first, then lateral XY.
    ``adsorbate_overlap`` skips height (rarely helps) and tries lateral XY only.
    For porous frameworks, ``vdw_overlap`` is treated like ``too_close`` (shrink
    toward the free-volume site center, then XY).
    """
    recoverable = ("too_close", "too_far", "adsorbate_overlap", "vdw_overlap")
    if fail_reason not in recoverable:
        return ctx, fail_reason
    # Height recovery only for contact-distance failures (and porous VDW).
    height_reasons: tuple[str, ...] = ("too_close", "too_far")
    if ctx.mat_type == "porous":
        height_reasons = ("too_close", "too_far", "vdw_overlap")
    # Map porous VDW to the same shrink-toward-center strategy as too_close.
    height_mode = fail_reason
    if fail_reason == "vdw_overlap" and ctx.mat_type == "porous":
        height_mode = "too_close"

    pose = ctx.pose
    if pose.z_abs is None:
        return ctx, fail_reason

    zf = float(pose.z_fraction)
    origin = np.array([pose.x_abs, pose.y_abs, float(pose.z_abs)], dtype=float)
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)

    height_candidates: list[float] = []
    if fail_reason in height_reasons and z_span > 1e-9:
        for step in range(1, _DISTANCE_RECOVERY_HEIGHT_STEPS + 1):
            frac = step / float(_DISTANCE_RECOVERY_HEIGHT_STEPS + 1)
            if height_mode == "too_close":
                # Slabs/NPs: raise away from the surface. Porous: Voronoi sites
                # already sit in free volume — shrink toward the site center.
                if ctx.mat_type == "porous":
                    cand = zf * (1.0 - frac)
                else:
                    cand = zf + (1.0 - zf) * frac
            else:
                if ctx.mat_type == "porous":
                    cand = zf + (1.0 - zf) * frac
                else:
                    cand = zf * (1.0 - frac)
            cand = float(min(1.0, max(0.0, cand)))
            if abs(cand - zf) > 1e-9:
                height_candidates.append(cand)

    last_reason: str | None = fail_reason
    for cand_zf in height_candidates:
        center = _center_with_height_delta(
            origin,
            old_z_fraction=zf,
            new_z_fraction=cand_zf,
            ctx=ctx,
            slab=slab,
        )
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, center)
        last_reason = _validate_posed_adsorbate(
            adsorbate,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            material_type=ctx.mat_type,
        )
        if last_reason is None:
            new_pose = dataclasses.replace(
                pose,
                x_abs=float(center[0]),
                y_abs=float(center[1]),
                z_abs=float(center[2]),
                z_fraction=float(cand_zf),
            )
            return dataclasses.replace(ctx, pose=new_pose), None
        if last_reason not in recoverable:
            return ctx, last_reason

    work_zf = zf
    if height_mode == "too_close" and height_candidates:
        work_zf = (
            min(height_candidates)
            if ctx.mat_type == "porous"
            else max(height_candidates)
        )
    elif height_mode == "too_far" and height_candidates:
        work_zf = (
            max(height_candidates)
            if ctx.mat_type == "porous"
            else min(height_candidates)
        )

    work_center = _center_with_height_delta(
        origin,
        old_z_fraction=zf,
        new_z_fraction=work_zf,
        ctx=ctx,
        slab=slab,
    )

    for dx, dy in _xy_recovery_offsets(
        config,
        placement_index=pose.placement_index,
        site_index=pose.site_index,
    ):
        center = _apply_lateral_offset(work_center, dx=dx, dy=dy, ctx=ctx, slab=slab)
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, center)
        last_reason = _validate_posed_adsorbate(
            adsorbate,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            material_type=ctx.mat_type,
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
        if last_reason not in recoverable:
            return ctx, last_reason

    return ctx, last_reason or fail_reason


def _validate_posed_adsorbate(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
    material_type: str | None = None,
) -> str | None:
    """Run distance, adsorbate-separation, and optional contact-quality checks.

    Returns a failure reason token, or ``None`` when the placement is accepted.
    *material_type* defaults to ``config.material_type``; callers with a resolved
    placement context should pass ``ctx.mat_type``.
    """
    mat_type = material_type if material_type is not None else config.material_type
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    ok, _, dist_reason = geom.check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
        reject_vdw_overlaps=config.reject_vdw_overlaps,
        vdw_overlap_scale=config.vdw_overlap_scale,
        exclude_slab_atoms=exclude_n,
        material_type=mat_type,
    )
    if not ok:
        return dist_reason or "distance_check_failed"

    if exclude_n is not None:
        pre_ads = np.asarray(slab.get_positions()[exclude_n:], dtype=float)
        sep_ok, _ = geom.check_adsorbate_separation(
            adsorbate,
            pre_ads,
            min_separation=config.min_adsorbate_separation,
            cell=np.asarray(slab.get_cell(), dtype=float),
            pbc=material_aware_pbc(mat_type),
        )
        if not sep_ok:
            return "adsorbate_overlap"

    if config.strict_initial_placement or config.require_multiple_contact:
        contact_ok, contact_reason = geom.check_initial_contact_quality(
            adsorbate,
            slab,
            strict_initial_placement=config.strict_initial_placement,
            require_multiple_contact=config.require_multiple_contact,
            max_closest_approach=float(config.max_closest_approach),
            min_contact_atoms=int(config.min_contact_atoms),
            contact_distance_threshold=config.contact_distance_threshold,
            exclude_slab_atoms=exclude_n,
            material_type=mat_type,
        )
        if not contact_ok:
            return contact_reason

    return None


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
    placement_mode_resolved: str = "no_sites",
    fragment_positions: tuple[tuple[float, float, float], ...] | None = None,
) -> PlacementDescriptor:
    """Build a PlacementDescriptor from resolved pose/geometry fields.

    *z_offset* is the COM height above *surface_ref* (includes clearance lift).
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
        z_abs=float(pose.z_abs) if pose.z_abs is not None else None,
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

    fail_reason = _validate_posed_adsorbate(
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        material_type=ctx.mat_type,
    )
    if fail_reason is not None:
        if (
            allow_distance_recovery
            and config.placement_distance_recovery
            and fail_reason
            in ("too_close", "too_far", "adsorbate_overlap", "vdw_overlap")
        ):
            ctx, fail_reason = _recover_distance_failure(
                ctx,
                adsorbate,
                slab,
                config,
                fail_reason,
                slab_for_sites=slab_for_sites,
            )
            pose = ctx.pose
            if fail_reason is not None:
                return None, fail_reason
            if pose.z_abs is None:
                return None, "missing_z_abs"
            z_abs = float(pose.z_abs)
        else:
            return None, fail_reason

    # COM height above surface_ref (includes clearance lift applied at pose time).
    z_offset = _recover_z_offset(ctx, z_abs, slab)
    slab_indices: tuple[int, ...] | None = None
    if ctx.site is not None:
        slab_indices = ctx.site.slab_indices
    cell = np.asarray(slab.get_cell(), dtype=float)
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
        shape=geom._classify_molecule_shape(ctx.canonical_pos)[0],
        slab_indices=slab_indices,
        site_source=site_source,
        site_reference_frame=("local_site" if ctx.is_local_ref else "global_top_layer"),
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        placement_mode_resolved="sites" if ctx.use_sites else "no_sites",
    )
    return (adsorbate, descriptor), None


def generate_placement_from_pose(
    pose: PlacementPose,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement using universal pose semantics.

    Parameters
    ----------
    pose
        :class:`~metalsurfer.models.PlacementPose` with placement parameters.
    conformers
        List of adsorbate :class:`~ase.Atoms` conformers.
    slab
        Substrate :class:`~ase.Atoms`.
    config
        :class:`~metalsurfer.config.AdsorptionConfig` with placement settings.
    site_context
        Optional cached :class:`SiteContext`.
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
    canonical_pos = geom.compute_canonical_molecular_frame(
        adsorbate.get_positions(), symbols=symbols
    )

    ctx = _context_from_pose(pose, canonical_pos, slab, config, site_context)
    if ctx is None:
        return None
    result, fail_reason = _finalize_placement(ctx, adsorbate, slab, config)
    if result is None:
        logger.debug("Pose placement rejected: %s", fail_reason)
        return None
    return result

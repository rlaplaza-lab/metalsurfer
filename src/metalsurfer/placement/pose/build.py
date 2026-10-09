"""Pose construction from placement specs and absolute pose replay."""

import dataclasses
import logging

import numpy as np
from ase import Atoms

from ...config import AdsorptionConfig
from ...models import PlacementPose, PlacementSpec
from .. import geometry as geom
from .._constants import _PARALLEL_Z_MIN_HI_MARGIN, _VECTOR_NORM_EPS
from .._material import material_aware_pbc, material_type_for_placement
from ..orientation import (
    _exclusive_marked_binders,
    _is_flat_aromatic,
    _marked_binder_indices,
    _parallel_z_adjustments,
    orient_from_spec,
)
from ..site_context import SiteContext, site_context_for_sampling
from ..site_coords import (
    _derive_top_layer_tolerance,
    _slab_normal,
    _slab_plane_projectors,
    top_layer_mask_by_normal,
)
from ..site_enumeration import _compute_site_z_base, _get_site_surface_radii
from .context import _HeightInterval, _PlacementContext, _PoseBatchCache
from .height import (
    _com_floor_on_support_plane,
    _com_height_from_z_fraction,
    _feasible_height_interval,
    _height_above_supports,
    _height_interval_family_key,
    _placement_normal_hat,
    _resolve_surface_ref,
)

logger = logging.getLogger(__name__)


def build_pose_batch_cache(
    slab: Atoms,
    conformers: list[Atoms],
    config: AdsorptionConfig,
) -> _PoseBatchCache:
    """Precompute plane projectors, top-layer radius, and per-conformer frames."""
    _ = config
    cache = _PoseBatchCache()
    cell = np.asarray(slab.get_cell(), dtype=float)
    positions = np.asarray(slab.get_positions(), dtype=float)
    symbols = list(slab.get_chemical_symbols())
    cache.cell = cell
    cache.n_hat = _slab_normal(cell)
    cache.positions = positions
    cache.pinv_ab_T, _ = _slab_plane_projectors(cell)
    # Element-derived top depth (same as _get_site_surface_radii without
    # top_indices), not config.top_layer_tolerance.
    radii_indices = np.nonzero(
        top_layer_mask_by_normal(
            positions, cell, float(_derive_top_layer_tolerance(symbols))
        )
    )[0]
    cache.r_surface_top_layer = _get_site_surface_radii(
        slab, None, top_indices=radii_indices
    )
    for i, conf in enumerate(conformers):
        ads_pos = conf.get_positions()
        symbols = conf.get_chemical_symbols()
        canonical = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
        shape, _, _ = geom._classify_molecule_shape(canonical)
        cache.frames[i] = (canonical, shape)
    return cache


def _pose_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
) -> tuple[_PlacementContext | None, str | None]:
    """Build a placement context (pose + resolved geometry) from a spec.

    *slab* may already contain previously placed adsorbates (saturation); when
    it does, *slab_for_sites* must be the bare substrate used for site
    enumeration and ``surface_ref``. Occupancy and clash checks still use the
    full *slab*.

    Returns ``(ctx, None)`` on success, or ``(None, reason)`` when placement
    cannot proceed (``"no_sites_found"`` or ``"invalid_site_index"``).
    """
    ads_pos = adsorbate.get_positions().copy()
    symbols = adsorbate.get_chemical_symbols()
    cached_frame = (
        pose_cache.frames.get(int(spec.conformer_index))
        if pose_cache is not None
        else None
    )
    if cached_frame is not None:
        canonical_pos, shape = cached_frame
    else:
        canonical_pos = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
        shape, _, _ = geom._classify_molecule_shape(canonical_pos)
    normal = np.array([0.0, 0.0, 1.0])

    reference = slab_for_sites if slab_for_sites is not None else slab
    if site_context is not None:
        ctx = site_context
    else:
        ctx = site_context_for_sampling(reference, config, None, full_slab=slab)
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
        normal = geom._safe_normalize(site_normal)

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    ref_slab = reference

    if site.slab_indices:
        r_surface = _get_site_surface_radii(ref_slab, site)
    elif pose_cache is not None and pose_cache.r_surface_top_layer is not None:
        r_surface = pose_cache.r_surface_top_layer
    else:
        r_surface = _get_site_surface_radii(ref_slab, None)

    z_base_lo, z_base_hi = _compute_site_z_base(
        config, ref_slab, site, symbols, r_surface=r_surface
    )

    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)
    if flat_aromatic and spec.orientation_type == "parallel" and site.kind != "void":
        z_floor, z_lo_shrink, z_hi_shrink = _parallel_z_adjustments(
            ref_slab, site, symbols, r_surface=r_surface
        )
        z_base_lo = max(z_floor, z_base_lo - z_lo_shrink)
        z_base_hi = max(
            z_base_lo + _PARALLEL_Z_MIN_HI_MARGIN,
            z_base_hi - z_hi_shrink,
        )

    zf = float(spec.z_fraction)

    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        ref_slab,
        mat_type,
        cell=pose_cache.cell if pose_cache is not None else None,
        positions=pose_cache.positions if pose_cache is not None else None,
    )

    marked = _marked_binder_indices(smiles)
    exclusive = _exclusive_marked_binders(smiles)
    oriented = orient_from_spec(
        canonical_pos,
        normal=normal,
        symbols=symbols,
        spec=spec,
        marked_indices=marked,
        exclusive_marked=exclusive,
        tangent_basis=site.tangent_basis if site is not None else None,
    )
    rotated_pos = oriented.rotated_pos
    quat = oriented.quat

    place_normal = _placement_normal_hat(normal)
    if float(np.linalg.norm(np.asarray(site.normal, dtype=float))) <= _VECTOR_NORM_EPS:
        if pose_cache is not None and pose_cache.n_hat is not None:
            place_normal = _placement_normal_hat(pose_cache.n_hat)
        else:
            cell = (
                pose_cache.cell
                if pose_cache is not None and pose_cache.cell is not None
                else np.asarray(ref_slab.get_cell(), dtype=float)
            )
            place_normal = _placement_normal_hat(_slab_normal(cell))

    base = np.asarray(site.xyz, dtype=float)
    base_h = float(np.dot(base, place_normal))
    pos_for_support = (
        pose_cache.positions
        if pose_cache is not None and pose_cache.positions is not None
        else np.asarray(ref_slab.get_positions(), dtype=float)
    )
    cell_arr = (
        pose_cache.cell
        if pose_cache is not None and pose_cache.cell is not None
        else np.asarray(ref_slab.get_cell(), dtype=float)
    )
    family_key = _height_interval_family_key(spec)
    interval: _HeightInterval | None = None
    if pose_cache is not None:
        interval = pose_cache.height_intervals.get(family_key)
    if interval is None:
        interval = _feasible_height_interval(
            rotated_pos,
            symbols,
            site=site,
            place_normal=place_normal,
            slab_positions=pos_for_support,
            slab_symbols=list(ref_slab.get_chemical_symbols()),
            cell=cell_arr,
            pbc=material_aware_pbc(mat_type),
            config=config,
            r_surface=float(r_surface),
            z_base_lo=float(z_base_lo),
            z_base_hi=float(z_base_hi),
        )
        if pose_cache is not None and interval is not None:
            pose_cache.height_intervals[family_key] = interval
    if interval is None:
        return None, "infeasible_pose"
    if not interval.contact_atoms_ok:
        return None, "insufficient_contact_atoms"

    com_h = _com_height_from_z_fraction(
        zf, interval.com_lo, interval.com_nominal, interval.com_hi
    )
    if site.kind != "void":
        com_h = max(
            com_h,
            _com_floor_on_support_plane(rotated_pos, place_normal, base_h),
        )
    placement_center = base + (com_h - base_h) * place_normal
    contact_ref = _height_above_supports(
        site,
        pos_for_support,
        place_normal,
        reduce="max",
        fallback=float(surface_ref),
    )
    ctx_z_lo = float(interval.com_lo) - float(contact_ref)
    ctx_z_hi = float(interval.com_hi) - float(contact_ref)
    if ctx_z_hi < ctx_z_lo:
        ctx_z_hi = ctx_z_lo

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
            z_base_lo=float(ctx_z_lo),
            z_base_hi=float(ctx_z_hi),
            normal=np.asarray(place_normal, dtype=float),
            shape=shape,
            com_lo=float(interval.com_lo),
            com_nominal=float(interval.com_nominal),
            com_hi=float(interval.com_hi),
        ),
        None,
    )


def _context_from_pose(
    pose: PlacementPose,
    canonical_pos: np.ndarray,
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None,
    slab_for_sites: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
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

    reference = slab_for_sites if slab_for_sites is not None else slab
    if site_context is not None:
        ctx = site_context
    else:
        ctx = site_context_for_sampling(reference, config, None, full_slab=slab)
    site = None
    if ctx.use_sites and 0 <= pose.site_index < len(ctx.sites):
        site = ctx.sites[pose.site_index]
    mat_type = material_type_for_placement(site, when_no_site=config.material_type)
    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        reference,
        mat_type,
        cell=pose_cache.cell if pose_cache is not None else None,
        positions=pose_cache.positions if pose_cache is not None else None,
    )

    pose_normalized = dataclasses.replace(
        pose,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
    )

    if site is not None:
        site_normal = np.asarray(site.normal, dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        normal = (
            site_normal / nrm
            if nrm > _VECTOR_NORM_EPS
            else np.array([0.0, 0.0, 1.0], dtype=float)
        )
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

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
        normal=normal,
        shape=geom._classify_molecule_shape(canonical_pos)[0],
    )

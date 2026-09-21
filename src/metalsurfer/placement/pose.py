"""Pose construction, validation, and finalization."""

import dataclasses
import logging
import random
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms

from .._utils import cell_has_volume
from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementPose, PlacementSpec
from . import geometry as geom
from ._constants import (
    _DISTANCE_RECOVERY_XY_ATTEMPTS,
    _DISTANCE_ZERO_EPS,
    _LATERAL_OFFSET_REF_SWITCH_DOT,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _RECOVERY_INPLANE_PENETRATION_DOT,
    _RECOVERY_NORMAL_PENETRATION_WINDOW_FACTOR,
    _VECTOR_NORM_EPS,
    _XY_RECOVERY_PLACEMENT_MIXER,
    _XY_RECOVERY_SEED_MIXER,
    _XY_RECOVERY_SITE_MIXER,
    RECOVERABLE_DISTANCE_REASONS,
)
from ._material import material_aware_pbc, material_type_for_placement
from .clash import (
    atom_radii_for_symbols,
    clash_bounds_for_adsorbate,
    compose_quaternion_with_azimuth,
    resolve_rigid_clash,
)
from .occupancy import incoming_inplane_radius
from .orientation import (
    _is_flat_aromatic,
    _parallel_z_adjustments,
    orient_from_spec,
)
from .site_context import SiteContext, site_context_for_sampling
from .site_coords import (
    _derive_top_layer_tolerance,
    _slab_normal,
    _slab_plane_projectors,
    top_layer_mask_by_normal,
)
from .site_enumeration import (
    _compute_site_z_base,
    _get_site_surface_radii,
    _height_along_slab_normal,
)
from .site_plugins.helpers import (
    top_layer_is_planar_from_arrays as _top_layer_is_planar_from_arrays,
)
from .site_types import Site

logger = logging.getLogger(__name__)


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


@dataclass
class _PoseBatchCache:
    """Per-batch invariants shared across placements on the same substrate."""

    top_layer_planar: bool | None = None
    pinv_ab_T: np.ndarray | None = None
    # Mean covalent radius of the bare substrate top layer; computed once per
    # batch and reused for z-offset scaling when a spec's site has no explicit
    # slab_indices. Read-only across worker threads.
    r_surface_top_layer: float | None = None
    # Legacy global strip (unused for height; kept for callers that still fill it).
    global_surface_ref: float | None = None
    cell: np.ndarray | None = None
    n_hat: np.ndarray | None = None
    positions: np.ndarray | None = None
    # conformer_index -> (canonical_pos, shape)
    frames: dict[int, tuple[np.ndarray, str]] = dataclasses.field(default_factory=dict)
    # Per-screen height intervals keyed by pose family (not shared across molecules).
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


def _height_interval_family_key(spec: PlacementSpec) -> tuple:
    """Cache key shared by sibling ``z_fraction`` values of one rigid pose."""
    return (
        int(spec.conformer_index),
        str(spec.orientation_type),
        int(spec.site_index),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        bool(spec.face_flip),
        spec.en_atom_index,
    )


def _com_height_from_z_fraction(
    zf: float, lo: float, nominal: float, hi: float
) -> float:
    """Map unit ``z_fraction`` onto *[lo, hi]* with ``0.5`` at *nominal*."""
    z = float(zf)
    if z <= 0.5:
        return float(lo + (z / 0.5) * (nominal - lo))
    return float(nominal + ((z - 0.5) / 0.5) * (hi - nominal))


def build_pose_batch_cache(
    slab: Atoms,
    conformers: list[Atoms],
    config: AdsorptionConfig,
) -> _PoseBatchCache:
    """Precompute slab planarity, plane projectors, and per-conformer frames."""
    cache = _PoseBatchCache()
    cell = np.asarray(slab.get_cell(), dtype=float)
    positions = np.asarray(slab.get_positions(), dtype=float)
    symbols = list(slab.get_chemical_symbols())
    cache.cell = cell
    cache.n_hat = _slab_normal(cell)
    cache.positions = positions
    cache.pinv_ab_T, _ = _slab_plane_projectors(cell)
    top_tol = float(config.top_layer_tolerance)
    top_mask = top_layer_mask_by_normal(positions, cell, top_tol)
    cache.top_layer_planar = bool(
        _top_layer_is_planar_from_arrays(
            positions,
            cell,
            top_tol,
            float(config.planar_z_variance_threshold),
            top_mask=top_mask,
        )
    )
    # Radii use the element-derived top depth (same as _get_site_surface_radii
    # without top_indices), not config.top_layer_tolerance used for planarity.
    radii_indices = np.nonzero(
        top_layer_mask_by_normal(
            positions, cell, float(_derive_top_layer_tolerance(symbols))
        )
    )[0]
    cache.r_surface_top_layer = _get_site_surface_radii(
        slab, None, top_indices=radii_indices
    )
    if config.material_type == "slab":
        cache.global_surface_ref = float(
            np.max(_height_along_slab_normal(positions, cell))
        )
    for i, conf in enumerate(conformers):
        ads_pos = conf.get_positions()
        symbols = conf.get_chemical_symbols()
        canonical = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
        shape, _, _ = geom._classify_molecule_shape(canonical)
        cache.frames[i] = (canonical, shape)
    return cache


def _placement_normal_hat(normal: np.ndarray) -> np.ndarray:
    """Return the unit vector along *normal*, or +z when degenerate."""
    n = np.asarray(normal, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(n))
    if nrm <= _VECTOR_NORM_EPS:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return n / nrm


def _contact_atom_index(
    rotated_pos: np.ndarray,
    normal: np.ndarray,
    symbols: list[str],
    *,
    orientation_type: str | None,
    en_atom_index: int | None,
) -> int:
    """Atom whose height defines covalent contact for the orientation family."""
    n_hat = _placement_normal_hat(normal)
    heights = np.asarray(rotated_pos, dtype=float) @ n_hat
    # Parallel / face-down: closest atom (often H on a ring).
    if orientation_type == "parallel":
        return int(np.argmin(heights))
    # Binder-aligned (EN-down / round / vertical): prefer the binder.
    if en_atom_index is not None and 0 <= int(en_atom_index) < len(symbols):
        return int(en_atom_index)
    binders = geom._binding_atom_candidates(symbols)
    if binders:
        return int(min(binders, key=lambda i: float(heights[i])))
    return int(np.argmin(heights))


def _framework_plane_height(
    site: Site,
    positions: np.ndarray,
    n_hat: np.ndarray,
    *,
    reduce: Literal["max", "mean"] = "max",
) -> float | None:
    """Height of coordinating framework atoms along *n_hat*, or ``None``.

    Site vertices from any plugin may already sit above their supports
    (topology lift, clearance snap, free-volume). Clearance and height
    offsets must use this plane so lift is applied once.

    ``"max"`` clears every support (contact); ``"mean"`` is the NP shell ref.
    """
    if not site.slab_indices:
        return None
    idx = np.asarray(site.slab_indices, dtype=int)
    n_pos = len(positions)
    idx = idx[(idx >= 0) & (idx < n_pos)]
    if idx.size == 0:
        return None
    heights = np.asarray(positions, dtype=float)[idx] @ _placement_normal_hat(n_hat)
    if reduce == "mean":
        return float(np.mean(heights))
    return float(np.max(heights))


def _height_above_supports(
    site: Site,
    positions: np.ndarray,
    n_hat: np.ndarray,
    *,
    reduce: Literal["max", "mean"] = "max",
    fallback: float,
) -> float:
    """Support-plane height along *n_hat*, or *fallback* when supports are empty."""
    fw = _framework_plane_height(site, positions, n_hat, reduce=reduce)
    return float(fallback) if fw is None else fw


def _pairwise_contact_com_height(
    rotated_pos: np.ndarray,
    symbols: list[str],
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    slab_symbols: list[str],
    cell: np.ndarray,
    pbc: list[bool],
    config: AdsorptionConfig,
    r_surface: float,
) -> float:
    """Smallest COM height along *place_normal* clearing every mol–slab pair.

    Lateral seed is *site_xyz*; orientation is fixed in *rotated_pos* (COM-
    centred). Uses the same pair gate as validation (covalent / optional VDW).
    Height is solved from MIC vectors at the support-plane seed so PBC images
    and non-Cartesian normals stay consistent.
    """
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    rotated = np.asarray(rotated_pos, dtype=float)
    slab_pos = np.asarray(slab_positions, dtype=float)
    if len(slab_pos) == 0 or len(rotated) == 0:
        return base_h

    def _allowed(sym_m: str, sym_s: str | None) -> float:
        return geom.min_pair_clearance_angstrom(
            sym_m,
            sym_s,
            min_distance=float(config.min_initial_distance),
            min_contact_ratio=float(config.min_contact_ratio),
            reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
            vdw_overlap_scale=float(config.vdw_overlap_scale),
            r_surface_fallback=float(r_surface),
        )

    mol_seed = rotated + base
    mic_vecs, _ = geom._mol_slab_pairwise_mic(mol_seed, slab_pos, cell, pbc)
    mic_n = np.einsum("ijd,d->ij", mic_vecs, n_hat)
    perp = mic_vecs - mic_n[:, :, None] * n_hat.reshape(1, 1, 3)
    lateral = np.linalg.norm(perp, axis=2)

    h_needed = float("-inf")
    for i, sym_m in enumerate(symbols):
        for j, sym_s in enumerate(slab_symbols):
            allowed = _allowed(sym_m, sym_s)
            lat = float(lateral[i, j])
            if lat >= allowed:
                continue
            # Raising COM by Δh adds Δh to every mic·n (same lattice image).
            vertical = float(np.sqrt(max(0.0, allowed * allowed - lat * lat)))
            h_pair = base_h + vertical - float(mic_n[i, j])
            if h_pair > h_needed:
                h_needed = h_pair

    if not np.isfinite(h_needed):
        # Every pair already clears laterally; sit the lowest atom at its
        # nearest-support gate so the pose is still a contact, not a hover.
        rel_h = rotated @ n_hat
        i_low = int(np.argmin(rel_h))
        j_near = int(np.argmin(lateral[i_low]))
        gate = _allowed(symbols[i_low], slab_symbols[j_near])
        h_needed = base_h + gate - float(rel_h[i_low])
    # 3D pair gates in a hollow can sit atoms below the nuclear plane; keep
    # every atom on the vacuum side of the support-plane anchor.
    h_needed = max(
        float(h_needed),
        _com_floor_on_support_plane(rotated, n_hat, base_h),
    )
    # Pad so reconstructed MIC distances clear ``dists < allowed`` under
    # quaternion / wrap float noise (not a chemistry slack).
    return float(h_needed) + 1e-6


def _pair_min_distance_at_com_height(
    rotated_pos: np.ndarray,
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    com_h: float,
) -> float:
    """Minimum mol–slab MIC distance with COM at *com_h* along *place_normal*."""
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    center = base + (float(com_h) - base_h) * n_hat
    mol = np.asarray(rotated_pos, dtype=float) + center
    dists = geom._mol_slab_pairwise_distances(mol, slab_positions, cell, pbc)
    if dists.size == 0:
        return float("inf")
    return float(np.min(dists))


def _count_contact_atoms_at_com_height(
    rotated_pos: np.ndarray,
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    com_h: float,
    contact_threshold: float,
) -> int:
    """Count adsorbate atoms within *contact_threshold* of any slab atom."""
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    center = base + (float(com_h) - base_h) * n_hat
    mol = np.asarray(rotated_pos, dtype=float) + center
    dists = geom._mol_slab_pairwise_distances(mol, slab_positions, cell, pbc)
    if dists.size == 0:
        return 0
    return int(np.sum(np.any(dists <= float(contact_threshold), axis=1)))


def _feasible_height_interval(
    rotated_pos: np.ndarray,
    symbols: list[str],
    *,
    site: Site,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    slab_symbols: list[str],
    cell: np.ndarray,
    pbc: list[bool],
    config: AdsorptionConfig,
    r_surface: float,
    z_base_lo: float,
    z_base_hi: float,
) -> _HeightInterval | None:
    """Feasible COM height interval along *place_normal* for one rigid pose.

    Wall-near (``site_type != "pore"``): nominal is the pairwise contact solve;
    lower bound is that height; upper bound is limited by
    ``max_initial_distance`` / ``max_closest_approach`` when configured.

    Pore: nominal is the void-centre height; the interval is the largest span
    of the probe window ``[centre ± ½·diversity]`` that still clears the pair
    gate. If nothing clears, the interval collapses to the least-penetrating
    sample on that same probe grid (never a legacy offset outside the window).
    """
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site.xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    diversity = max(float(z_base_hi - z_base_lo), _PARALLEL_Z_MIN_HI_MARGIN)
    is_pore = site.site_type == "pore"

    if is_pore:
        com_nominal = base_h
        half = 0.5 * diversity
        probe_lo = com_nominal - half
        probe_hi = com_nominal + half
        n_probe = 21
        cleared_mask = np.zeros(n_probe, dtype=bool)
        deficits = np.full(n_probe, np.inf)
        heights = np.linspace(probe_lo, probe_hi, n_probe)
        for k, h in enumerate(heights):
            center = base + (float(h) - base_h) * n_hat
            mol = np.asarray(rotated_pos, dtype=float) + center
            mic_vecs, _ = geom._mol_slab_pairwise_mic(mol, slab_positions, cell, pbc)
            if mic_vecs.size == 0:
                cleared_mask[k] = True
                deficits[k] = 0.0
                continue
            dists = np.linalg.norm(mic_vecs, axis=2)
            worst = 0.0
            for i, sym_m in enumerate(symbols):
                for j, sym_s in enumerate(slab_symbols):
                    allowed = geom.min_pair_clearance_angstrom(
                        sym_m,
                        sym_s,
                        min_distance=float(config.min_initial_distance),
                        min_contact_ratio=float(config.min_contact_ratio),
                        reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
                        vdw_overlap_scale=float(config.vdw_overlap_scale),
                        r_surface_fallback=float(r_surface),
                    )
                    short = float(allowed) - float(dists[i, j])
                    if short > worst:
                        worst = short
            cleared_mask[k] = worst <= 0.0
            deficits[k] = worst
        if not np.any(cleared_mask):
            best_def = float(np.min(deficits))
            cands = np.nonzero(np.abs(deficits - best_def) <= 1e-9)[0]
            centre_i = int(np.argmin(np.abs(heights - base_h)))
            pick = int(cands[np.argmin(np.abs(cands - centre_i))])
            h_pick = float(heights[pick])
            com_lo = h_pick
            com_nominal = h_pick
            com_hi = h_pick
        else:
            mid_i = int(np.argmin(np.abs(heights - com_nominal)))
            if not cleared_mask[mid_i]:
                cleared_idx = np.nonzero(cleared_mask)[0]
                mid_i = int(cleared_idx[np.argmin(np.abs(cleared_idx - mid_i))])
                com_nominal = float(heights[mid_i])
            lo_i = mid_i
            while lo_i > 0 and cleared_mask[lo_i - 1]:
                lo_i -= 1
            hi_i = mid_i
            while hi_i + 1 < n_probe and cleared_mask[hi_i + 1]:
                hi_i += 1
            com_lo = float(heights[lo_i])
            com_hi = float(heights[hi_i])
    else:
        com_nominal = _pairwise_contact_com_height(
            rotated_pos,
            symbols,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=slab_positions,
            slab_symbols=slab_symbols,
            cell=cell,
            pbc=pbc,
            config=config,
            r_surface=r_surface,
        )
        com_lo = float(com_nominal)
        actual = _pair_min_distance_at_com_height(
            rotated_pos,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=slab_positions,
            cell=cell,
            pbc=pbc,
            com_h=com_nominal,
        )
        upper_targets: list[float] = []
        if config.max_initial_distance is not None:
            upper_targets.append(float(config.max_initial_distance))
        if config.strict_initial_placement:
            upper_targets.append(float(config.max_closest_approach))
        if upper_targets:
            target = min(upper_targets)
            # Raising along the normal ≈ increases min distance 1:1 when the
            # closest pair is normal-aligned; clamp the diversity window.
            slack = max(0.0, float(target) - float(actual))
            com_hi = com_nominal + min(slack, 0.5 * diversity)
        else:
            com_hi = com_nominal + 0.5 * diversity

    if com_hi + _DISTANCE_ZERO_EPS < com_lo:
        return None

    min_contacts = int(config.min_contact_atoms)
    if config.require_multiple_contact:
        min_contacts = max(2, min_contacts)
    need_contact_check = bool(
        config.strict_initial_placement or config.require_multiple_contact
    )
    if need_contact_check:
        n_contact = _count_contact_atoms_at_com_height(
            rotated_pos,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=slab_positions,
            cell=cell,
            pbc=pbc,
            com_h=com_nominal,
            contact_threshold=float(config.contact_distance_threshold),
        )
        contact_ok = n_contact >= min_contacts
        if config.strict_initial_placement:
            actual_nom = _pair_min_distance_at_com_height(
                rotated_pos,
                site_xyz=base,
                place_normal=place_normal,
                slab_positions=slab_positions,
                cell=cell,
                pbc=pbc,
                com_h=com_nominal,
            )
            if actual_nom > float(config.max_closest_approach):
                contact_ok = False
    else:
        contact_ok = True

    return _HeightInterval(
        com_lo=float(com_lo),
        com_nominal=float(com_nominal),
        com_hi=float(max(com_hi, com_lo)),
        contact_atoms_ok=contact_ok,
    )


def _com_floor_on_support_plane(
    rotated_pos: np.ndarray,
    n_hat: np.ndarray,
    base_h: float,
) -> float:
    """COM height that places the lowest adsorbate atom on the support plane."""
    rel_h = np.asarray(rotated_pos, dtype=float) @ np.asarray(n_hat, dtype=float)
    return float(base_h - np.min(rel_h))


def _resolve_surface_ref(
    site: Site | None,
    slab: Atoms,
    mat_type: str,
    *,
    rough_slab_local_z: bool = False,
    top_layer_tolerance: float | None = None,
    planar_z_variance_threshold: float | None = None,
    top_layer_planar: bool | None = None,
    global_surface_ref: float | None = None,
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Return *(surface_ref, is_local_ref)* for height / z-offset calculations.

    Placement always uses the **site frame**: ``surface_ref`` is the
    coordinating-atom plane along ``site.normal`` (max of supports), or the
    site vertex for pores / empty supports. Never a plugin-lifted ``site.xyz``.
    ``is_local_ref`` is always ``True`` when a site is present.

    *rough_slab_local_z* / planarity / *global_surface_ref* are accepted for
    call-site compatibility but ignored — height is local for every material.
    Slab-cell ``+z`` is only the fallback when *site* is ``None`` or the site
    normal is degenerate.
    """
    del (
        rough_slab_local_z,
        top_layer_tolerance,
        planar_z_variance_threshold,
        top_layer_planar,
        global_surface_ref,
    )
    cell_arr = (
        np.asarray(cell, dtype=float)
        if cell is not None
        else np.asarray(slab.get_cell(), dtype=float)
    )
    pos = (
        np.asarray(positions, dtype=float)
        if positions is not None
        else np.asarray(slab.get_positions(), dtype=float)
    )
    if site is not None:
        site_xyz = np.asarray(site.xyz, dtype=float)
        site_normal = np.asarray(site.normal, dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        if nrm <= _VECTOR_NORM_EPS:
            # Degenerate site normal: fall back to slab-cell height of the vertex.
            return float(_height_along_slab_normal(site_xyz, cell_arr)), True
        n_hat = site_normal / nrm
        vertex_h = float(np.dot(site_xyz, n_hat))
        if site.site_type == "pore" or not site.slab_indices:
            return vertex_h, True
        return (
            _height_above_supports(site, pos, n_hat, reduce="max", fallback=vertex_h),
            True,
        )
    # No site: Cartesian / radial fallback (stale site_index is the usual cause).
    if cell_has_volume(cell_arr):
        return (
            float(np.max(_height_along_slab_normal(pos, cell_arr))),
            False,
        )
    com = np.mean(pos, axis=0)
    logger.debug(
        "Resolving surface reference for %s without a site; using radial "
        "distance from COM (stale or out-of-range site_index is the likely cause)",
        mat_type,
    )
    return (
        float(np.max(np.linalg.norm(pos - com, axis=1))),
        False,
    )


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
    # Use the caller-provided catalog as-is so site_index stays stable.
    # Resolve (and possibly expand under coverage) only when omitted (public API).
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

    # Fetch the surface radius once per pose. Per-site radii (non-empty
    # slab_indices) still need a per-spec fetch; the top-layer radius is taken
    # from the batch cache when available, else computed once here.
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
    if (
        flat_aromatic
        and spec.orientation_type == "parallel"
        and site.site_type != "pore"
    ):
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

    oriented = orient_from_spec(
        canonical_pos,
        normal=normal,
        symbols=symbols,
        spec=spec,
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
    if site.site_type != "pore":
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
    # Replay must use the same catalog the original placement indexed into.
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


def _recover_z_offset(
    ctx: _PlacementContext,
    z_abs: float,
    slab: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
) -> float:
    """Recover COM height above the surface reference from absolute placement.

    Returned ``z_offset`` is the adsorbate COM displacement above
    *surface_ref* along the placement normal (site frame). It includes any
    clearance lift applied at placement time, so ``surface_ref + z_offset``
    reconstructs the COM height along that normal — not the closest-atom gap.
    """
    del slab, pose_cache  # height is always along ctx.normal / site frame
    pose = ctx.pose
    placement = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    n_hat = _placement_normal_hat(ctx.normal)
    return float(np.dot(placement, n_hat) - ctx.surface_ref)


def _saturation_exclude_count(
    slab: Atoms,
    slab_for_sites: Atoms | None,
) -> int | None:
    """Prefix length of substrate atoms when *slab* has pre-adsorbed suffix."""
    if slab_for_sites is None:
        return None
    n = len(slab_for_sites)
    if n >= len(slab):
        return None
    return n


def _placement_normal(
    ctx: _PlacementContext,
    slab: Atoms,
    pose_cache: _PoseBatchCache | None = None,
) -> np.ndarray:
    """Return unit normal used for height/lateral recovery (site frame)."""
    del pose_cache
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
    actual_min = float(np.min(dists))
    mol_r = np.array(
        [
            r if (r := geom._get_covalent_radius(s)) is not None else np.nan
            for s in mol_syms
        ],
        dtype=float,
    )
    if slab_scratch is not None and slab_scratch.slab_cov_r is not None:
        slab_r = np.asarray(slab_scratch.slab_cov_r, dtype=float)
    else:
        slab_r = np.array(
            [
                r if (r := geom._get_covalent_radius(s)) is not None else np.nan
                for s in slab_syms
            ],
            dtype=float,
        )
    allowed = (mol_r[:, None] + slab_r[None, :]) * float(config.min_contact_ratio)
    np.maximum(allowed, float(config.min_initial_distance), out=allowed)
    np.nan_to_num(allowed, nan=float(config.min_initial_distance), copy=False)
    penetration = np.maximum(0.0, allowed - dists)
    max_pen = float(np.max(penetration))
    if max_pen <= _DISTANCE_ZERO_EPS or normal is None:
        return actual_min, max_pen, 1.0
    mi, si = np.unravel_index(int(np.argmax(penetration)), penetration.shape)
    pair_vec = np.asarray(slab_pos[si], dtype=float) - np.asarray(
        mol_pos[mi], dtype=float
    )
    pair_nrm = float(np.linalg.norm(pair_vec))
    if pair_nrm <= _VECTOR_NORM_EPS:
        return actual_min, max_pen, 1.0
    n_hat = _placement_normal_hat(normal)
    abs_dot = abs(float(np.dot(pair_vec / pair_nrm, n_hat)))
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
    pose_cache: _PoseBatchCache | None = None,
) -> tuple[np.ndarray, float] | None:
    """One signed height nudge along the placement normal; None if nothing to fix."""
    pose = ctx.pose
    zf = float(pose.z_fraction)
    z_abs = _require_pose_z_abs(pose)
    origin = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)
    if z_span <= _DISTANCE_ZERO_EPS:
        return None

    actual, max_pen = _contact_penetration(
        adsorbate,
        slab,
        config,
        material_type=ctx.mat_type,
        slab_scratch=slab_scratch,
        slab_for_sites=slab_for_sites,
    )
    # Pore sites: nudge toward the void centre when too close. Wall-near: raise
    # away from the support plane. Keyed on site_type, not material_type.
    is_pore = ctx.site is not None and ctx.site.site_type == "pore"
    if height_mode == "too_close":
        if max_pen <= _DISTANCE_ZERO_EPS:
            return None
        signed = -max_pen if is_pore else max_pen
    elif height_mode == "contact_distance_too_large":
        # Same physics as too_far: molecule is outside the contact window.
        target = float(config.max_closest_approach)
        excess = actual - target
        if excess <= _DISTANCE_ZERO_EPS:
            return None
        signed = excess if is_pore else -excess
    else:
        max_d = config.max_initial_distance
        if max_d is None:
            # Fall back to contact window when no hard max_initial_distance.
            if height_mode == "too_far" and config.strict_initial_placement:
                target = float(config.max_closest_approach)
                excess = actual - target
                if excess <= _DISTANCE_ZERO_EPS:
                    return None
                signed = excess if is_pore else -excess
            else:
                return None
        else:
            excess = actual - float(max_d)
            if excess <= _DISTANCE_ZERO_EPS:
                return None
            signed = excess if is_pore else -excess

    n_hat = _placement_normal(ctx, slab, pose_cache=pose_cache)
    # Clamp the nudge into the feasible recovery window via z_fraction.
    new_zf = float(min(1.0, max(0.0, zf + signed / z_span)))
    clipped_center = origin + float((new_zf - zf) * z_span) * n_hat
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
    n_hat = _placement_normal(ctx, slab, pose_cache=pose_cache)
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
        # MIC-wrap the in-plane a–b components; keep height along normal.
        if pose_cache is not None and pose_cache.pinv_ab_T is not None:
            pinv_ab_T = pose_cache.pinv_ab_T
        else:
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
    pose_cache: _PoseBatchCache | None = None,
    slab_scratch: geom._SlabDistanceScratch | None = None,
) -> tuple[_PlacementContext, str | None]:
    """One analytic height nudge, then clash descent (or XY if clash is off).

    ``too_close`` / ``too_far`` / ``contact_distance_too_large`` / ``vdw_overlap``
    try a single height shift first (skipped when the worst penetration is
    mostly in-plane). Huge normal penetration fails cheaply before Packmol
    clash. Remaining recoverable failures use clash descent when enabled;
    otherwise discrete XY jitter. Pore sites invert the height nudge toward
    the free-volume centre (``site_type == "pore"``), not ``material_type``.
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
    n_hat = _placement_normal(ctx, slab, pose_cache=pose_cache)
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
            pose_cache=pose_cache,
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
            pose_cache=pose_cache,
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
    pose_cache: _PoseBatchCache | None = None,
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
        min_separation=float(config.min_adsorbate_separation),
        use_vdw=use_vdw,
    )
    cutoff = 2.0 * float(np.max(moving_r) if moving_r.size else footprint) + z_window
    fixed_pos, fixed_radii = _clash_recovery_fixed_cloud(
        slab,
        slab_scratch,
        ads_com=np.asarray(work_center, dtype=float),
        use_vdw=use_vdw,
        min_initial_distance=float(config.min_initial_distance),
        min_adsorbate_separation=float(config.min_adsorbate_separation),
        neighbor_cutoff=cutoff,
    )
    if fixed_pos is None or fixed_radii is None:
        return ctx, fail_reason

    bounds = clash_bounds_for_adsorbate(
        adsorbate,
        config,
        z_window=z_window,
        footprint_radius=footprint,
        moving_radii=moving_r,
    )
    site_frame = geom.compute_surface_site_frame(
        _placement_normal(ctx, slab, pose_cache=pose_cache),
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
        include_substrate_min_sep=(fail_reason == "adsorbate_overlap"),
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
        _placement_normal(ctx, slab, pose_cache=pose_cache),
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
    # Restore adsorbate coordinates to the pre-descent context; otherwise the
    # Atoms object retains new_pos while we return the original ctx.
    _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, work_center)
    return ctx, last_reason


def _clash_recovery_fixed_cloud(
    slab: Atoms,
    slab_scratch: geom._SlabDistanceScratch,
    *,
    ads_com: np.ndarray,
    use_vdw: bool,
    min_initial_distance: float,
    min_adsorbate_separation: float,
    neighbor_cutoff: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Nearby substrate / pre-adsorbate atoms for clash recovery."""
    fixed_chunks: list[np.ndarray] = []
    radius_chunks: list[np.ndarray] = []
    cutoff = float(neighbor_cutoff)

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
        fixed_chunks.append(slab_scratch.slab_pos[near])
        if use_vdw and slab_scratch.slab_vdw_r is not None:
            radius_chunks.append(np.asarray(slab_scratch.slab_vdw_r, dtype=float)[near])
        elif slab_scratch.slab_cov_r is not None:
            radius_chunks.append(np.asarray(slab_scratch.slab_cov_r, dtype=float)[near])
        else:
            syms = [
                s for s, keep in zip(slab_scratch.slab_syms, near, strict=True) if keep
            ]
            radius_chunks.append(
                atom_radii_for_symbols(
                    syms, min_separation=min_initial_distance, use_vdw=use_vdw
                )
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
            radius_chunks.append(
                atom_radii_for_symbols(
                    pre_syms_near,
                    min_separation=min_adsorbate_separation,
                    use_vdw=use_vdw,
                )
            )

    if not fixed_chunks:
        return None, None
    fixed_radii = np.concatenate(radius_chunks)
    floor = min_adsorbate_separation / 2.0
    fixed_radii = np.where(np.isfinite(fixed_radii), fixed_radii, floor)
    return np.vstack(fixed_chunks), fixed_radii


def _build_slab_distance_scratch(
    slab: Atoms,
    exclude_n: int | None,
    mat_type: str,
) -> geom._SlabDistanceScratch:
    """Build the invariant slab-side slice used across candidate validations."""
    slab_syms = list(slab.get_chemical_symbols())
    if exclude_n is not None:
        slab_pos = np.asarray(slab.get_positions()[:exclude_n], dtype=float)
        slab_syms = slab_syms[:exclude_n]
        pre_ads_pos = np.asarray(slab.get_positions()[exclude_n:], dtype=float)
    else:
        slab_pos = np.asarray(slab.get_positions(), dtype=float)
        pre_ads_pos = None
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(mat_type)
    slab_cov_r = np.array(
        [
            r if (r := geom._get_covalent_radius(s)) is not None else np.nan
            for s in slab_syms
        ],
        dtype=float,
    )
    slab_vdw_r = np.array(
        [
            r if (r := geom._get_vdw_radius(s)) is not None else np.nan
            for s in slab_syms
        ],
        dtype=float,
    )
    return geom._SlabDistanceScratch(
        slab_pos=slab_pos,
        cell=cell,
        pbc=pbc,
        slab_syms=slab_syms,
        slab_cov_r=slab_cov_r,
        slab_vdw_r=slab_vdw_r,
        pre_ads_pos=pre_ads_pos,
    )


def _validate_posed_adsorbate(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
    material_type: str | None = None,
    slab_scratch: geom._SlabDistanceScratch | None = None,
) -> str | None:
    """Run distance, adsorbate-separation, and optional contact-quality checks.

    Returns a failure reason token, or ``None`` when the placement is accepted.
    *material_type* defaults to ``config.material_type``; callers with a resolved
    placement context should pass ``ctx.mat_type``.

    The mol↔slab MIC distance matrix is computed **once** and reused for the
    distance gate and the contact-quality gate.  When *slab_scratch* is provided,
    the slab side is reused from it instead of re-slicing the ASE ``Atoms``.
    """
    mat_type = material_type if material_type is not None else config.material_type
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    if slab_scratch is None:
        slab_scratch = _build_slab_distance_scratch(slab, exclude_n, mat_type)

    _mol_pos, _slab_pos, _mol_syms, _slab_syms, _cell, _pbc, dists = (
        geom._mol_slab_contact_arrays(
            adsorbate,
            slab,
            material_type=mat_type,
            exclude_slab_atoms=exclude_n,
            slab_scratch=slab_scratch,
        )
    )
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
        pairwise_distances=dists,
        slab_scratch=slab_scratch,
    )
    if not ok:
        return dist_reason

    if exclude_n is not None:
        pre_ads = (
            slab_scratch.pre_ads_pos
            if slab_scratch.pre_ads_pos is not None
            else np.asarray(slab.get_positions()[exclude_n:], dtype=float)
        )
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
            pairwise_distances=dists,
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
    z_offset = _recover_z_offset(ctx, z_abs, slab, pose_cache=pose_cache)
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
            "local_site"
            if ctx.is_local_ref or ctx.site is not None
            else "global_top_layer"
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

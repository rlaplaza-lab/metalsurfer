"""Height intervals, contact shells, and surface-reference resolution."""

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from ..._utils import cell_has_volume
from ...config import AdsorptionConfig
from ...models import PlacementSpec
from .. import geometry as geom
from .._constants import (
    _CONTACT_CLEARANCE_PAD_ANGSTROM,
    _CONTACT_SHELL_MARGIN_ANGSTROM,
    _DISTANCE_ZERO_EPS,
    _HEIGHT_INTERVAL_BISECT_STEPS,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _VECTOR_NORM_EPS,
)
from ..site_enumeration import _height_along_slab_normal
from ..site_types import Site
from .context import _HeightInterval

logger = logging.getLogger(__name__)


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


def _z_fraction_from_com_height(
    com_h: float, lo: float, nominal: float, hi: float
) -> float:
    """Inverse of :func:`_com_height_from_z_fraction` (``0.5`` at *nominal*)."""
    h = float(com_h)
    lo_f, nom_f, hi_f = float(lo), float(nominal), float(hi)
    if abs(hi_f - lo_f) <= _DISTANCE_ZERO_EPS:
        return 0.5
    if h <= nom_f:
        span = nom_f - lo_f
        if span <= _DISTANCE_ZERO_EPS:
            return 0.5
        return float(max(0.0, min(0.5, 0.5 * (h - lo_f) / span)))
    span = hi_f - nom_f
    if span <= _DISTANCE_ZERO_EPS:
        return 0.5
    return float(max(0.5, min(1.0, 0.5 + 0.5 * (h - nom_f) / span)))


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
    marked_indices: Sequence[int] = (),
    exclusive_marked: bool = False,
) -> int:
    """Atom whose height defines covalent contact for the orientation family."""
    n_hat = _placement_normal_hat(normal)
    heights = np.asarray(rotated_pos, dtype=float) @ n_hat
    # Parallel / face-down: closest atom (often H on a ring).
    if orientation_type == "parallel":
        return int(np.argmin(heights))
    # Binder-aligned (EN-down / round / vertical): prefer the binder.
    # *en_atom_index* is an index into the merged binder list (policy
    # semantics), not a raw atom index; resolve it the same way
    # :func:`geom._surface_aligned_rotation` does.
    binders = geom._binding_atom_candidates(
        symbols, marked_indices, exclusive=exclusive_marked
    )
    if en_atom_index is not None and binders and 0 <= int(en_atom_index) < len(binders):
        return int(binders[int(en_atom_index)])
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
    return float(h_needed) + _CONTACT_CLEARANCE_PAD_ANGSTROM


def _mol_positions_at_com_height(
    rotated_pos: np.ndarray,
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    com_h: float,
) -> np.ndarray:
    """Translate *rotated_pos* so the COM sits at *com_h* along *place_normal*."""
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    center = base + (float(com_h) - float(np.dot(base, n_hat))) * n_hat
    return np.asarray(rotated_pos, dtype=float) + center


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
    mol = _mol_positions_at_com_height(
        rotated_pos, site_xyz=site_xyz, place_normal=place_normal, com_h=com_h
    )
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
    mol = _mol_positions_at_com_height(
        rotated_pos, site_xyz=site_xyz, place_normal=place_normal, com_h=com_h
    )
    dists = geom._mol_slab_pairwise_distances(mol, slab_positions, cell, pbc)
    if dists.size == 0:
        return 0
    return int(np.sum(np.any(dists <= float(contact_threshold), axis=1)))


def _pair_worst_penetration_at_com_height(
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
    com_h: float,
) -> float:
    """Worst pair penetration (Å) at *com_h*; ≤0 means fully cleared."""
    mol = _mol_positions_at_com_height(
        rotated_pos, site_xyz=site_xyz, place_normal=place_normal, com_h=com_h
    )
    mic_vecs, _ = geom._mol_slab_pairwise_mic(mol, slab_positions, cell, pbc)
    if mic_vecs.size == 0:
        return 0.0
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
    return float(worst)


def _void_height_interval_analytic(
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
    diversity: float,
) -> tuple[float, float, float]:
    """Return ``(com_lo, com_nominal, com_hi)`` for a void site.

    Nominal is the void centre when it clears; otherwise the nearest cleared
    height inside ``[centre ± ½·diversity]``. Bounds expand from the nominal
    by bisection. If nothing clears, collapse to the least-penetrating height.
    """
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    half = 0.5 * float(diversity)
    probe_lo = base_h - half
    probe_hi = base_h + half

    def _pen(h: float) -> float:
        return _pair_worst_penetration_at_com_height(
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
            com_h=float(h),
        )

    centre_pen = _pen(base_h)
    if centre_pen <= 0.0:
        com_nominal = base_h
    else:
        # Nearest cleared height to the centre via bisection on each side.
        # Non-monotonic windows fall back to the least-penetrating of the
        # probed edge / mid samples (no fixed 21-point grid).
        def _nearest_cleared(edge: float) -> float | None:
            if _pen(edge) > 0.0:
                return None
            lo_b, hi_b = (edge, base_h) if edge < base_h else (base_h, edge)
            # lo_b cleared (or is edge), hi_b may penetrate; find cleared bound nearest centre.
            for _ in range(_HEIGHT_INTERVAL_BISECT_STEPS):
                mid = 0.5 * (lo_b + hi_b)
                if edge < base_h:
                    # search in [edge, centre]: want highest cleared
                    if _pen(mid) <= 0.0:
                        lo_b = mid
                    else:
                        hi_b = mid
                else:
                    # search in [centre, edge]: want lowest cleared
                    if _pen(mid) <= 0.0:
                        hi_b = mid
                    else:
                        lo_b = mid
            return float(lo_b if edge < base_h else hi_b)

        candidates = [
            h
            for h in (_nearest_cleared(probe_lo), _nearest_cleared(probe_hi))
            if h is not None
        ]
        if candidates:
            com_nominal = float(min(candidates, key=lambda h: abs(h - base_h)))
        else:
            mid = 0.5 * (probe_lo + probe_hi)
            samples = (probe_lo, mid, probe_hi, base_h)
            pens = [_pen(h) for h in samples]
            best = min(pens)
            picks = [
                h
                for h, p in zip(samples, pens, strict=True)
                if abs(p - best) <= _DISTANCE_ZERO_EPS
            ]
            h_pick = float(min(picks, key=lambda h: abs(h - base_h)))
            return h_pick, h_pick, h_pick

    def _bisect_edge(toward_lo: bool) -> float:
        lo_b, hi_b = (probe_lo, com_nominal) if toward_lo else (com_nominal, probe_hi)
        if _pen(lo_b if toward_lo else hi_b) <= 0.0:
            return float(lo_b if toward_lo else hi_b)
        # Find the farthest cleared edge from nominal by binary search.
        for _ in range(_HEIGHT_INTERVAL_BISECT_STEPS):
            mid = 0.5 * (lo_b + hi_b)
            if _pen(mid) <= 0.0:
                if toward_lo:
                    hi_b = mid
                else:
                    lo_b = mid
            else:
                if toward_lo:
                    lo_b = mid
                else:
                    hi_b = mid
        return float(hi_b if toward_lo else lo_b)

    com_lo = _bisect_edge(True)
    com_hi = _bisect_edge(False)
    return float(com_lo), float(com_nominal), float(com_hi)


def _contact_shell_indices(
    site_xyz: np.ndarray,
    *,
    rotated_pos: np.ndarray,
    symbols: list[str],
    slab_positions: np.ndarray,
    slab_symbols: list[str],
    config: AdsorptionConfig,
    r_surface: float,
    cell: np.ndarray,
    pbc: list[bool],
    place_normal: np.ndarray,
) -> np.ndarray:
    """Return substrate indices that can affect contact along *place_normal*.

    Membership is a cylinder around the site: lateral (MIC) distance ≤
    molecule extent + max pair gate. That keeps atoms along the approach
    column (needed on porous walls) while dropping far in-plane spectators.
    """
    n_slab = len(slab_positions)
    if n_slab == 0:
        return np.empty(0, dtype=int)
    mol_extent = float(
        np.max(np.linalg.norm(np.asarray(rotated_pos, dtype=float), axis=1))
    )
    max_gate = 0.0
    for sym_m in symbols:
        for sym_s in slab_symbols:
            gate = geom.min_pair_clearance_angstrom(
                sym_m,
                sym_s,
                min_distance=float(config.min_initial_distance),
                min_contact_ratio=float(config.min_contact_ratio),
                reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
                vdw_overlap_scale=float(config.vdw_overlap_scale),
                r_surface_fallback=float(r_surface),
            )
            if gate > max_gate:
                max_gate = float(gate)
    radius = mol_extent + max_gate + _CONTACT_SHELL_MARGIN_ANGSTROM
    base = np.asarray(site_xyz, dtype=float).reshape(3)
    slab_pos = np.asarray(slab_positions, dtype=float)
    pbc_flags = [bool(x) for x in pbc]
    n_hat = _placement_normal_hat(place_normal)

    if any(pbc_flags):
        deltas = slab_pos - base
        mic_vecs, _dists = find_mic(
            deltas, np.asarray(cell, dtype=float), pbc=pbc_flags
        )
        mic = np.asarray(mic_vecs, dtype=float)
        lat = mic - np.outer(mic @ n_hat, n_hat)
    else:
        deltas = slab_pos - base
        lat = deltas - np.outer(deltas @ n_hat, n_hat)
    lat_d = np.linalg.norm(lat, axis=1)

    return np.asarray(np.nonzero(lat_d <= radius)[0], dtype=int)


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

    Wall (``site.kind == "wall"``): nominal is the pairwise contact solve;
    lower bound is that height; upper bound is limited by
    ``max_initial_distance`` / ``max_closest_approach`` when configured.

    Void: nominal is the void-centre height when it clears; otherwise the
    nearest cleared height inside ``[centre ± ½·diversity]``. Bounds expand
    from the nominal by bisection on pair penetration.
    """
    base = np.asarray(site.xyz, dtype=float)
    diversity = max(float(z_base_hi - z_base_lo), _PARALLEL_Z_MIN_HI_MARGIN)
    is_void = site.kind == "void"

    shell_idx = _contact_shell_indices(
        base,
        rotated_pos=rotated_pos,
        symbols=symbols,
        slab_positions=slab_positions,
        slab_symbols=slab_symbols,
        config=config,
        r_surface=r_surface,
        cell=cell,
        pbc=pbc,
        place_normal=place_normal,
    )
    if shell_idx.size > 0:
        shell_pos = np.asarray(slab_positions, dtype=float)[shell_idx]
        shell_syms = [slab_symbols[int(i)] for i in shell_idx]
    else:
        shell_pos = slab_positions
        shell_syms = slab_symbols

    if is_void:
        com_lo, com_nominal, com_hi = _void_height_interval_analytic(
            rotated_pos,
            symbols,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=shell_pos,
            slab_symbols=shell_syms,
            cell=cell,
            pbc=pbc,
            config=config,
            r_surface=r_surface,
            diversity=diversity,
        )
    else:
        com_nominal = _pairwise_contact_com_height(
            rotated_pos,
            symbols,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=shell_pos,
            slab_symbols=shell_syms,
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
            slab_positions=shell_pos,
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
    cell: np.ndarray | None = None,
    positions: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Return *(surface_ref, is_local_ref)* for height / z-offset calculations.

    Placement always uses the **site frame**: ``surface_ref`` is the
    coordinating-atom plane along ``site.normal`` (max of supports), or the
    site vertex for pores / empty supports. Never a plugin-lifted ``site.xyz``.
    ``is_local_ref`` is always ``True`` when a site is present.
    Slab-cell ``+z`` is only the fallback when *site* is ``None`` or the site
    normal is degenerate.
    """
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
        if site.kind == "void":
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

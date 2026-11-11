"""Height intervals, contact shells, and surface-reference resolution."""

import logging
from collections.abc import Callable, Sequence
from typing import Literal, NamedTuple

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
    _VECTOR_NORM_EPS,
    _WALL_SHELL_MIN_GAP_ANGSTROM,
    _WALL_VDW_Z_FRACTION,
)
from .._material import material_aware_pbc
from ..cutoffs import pair_thresholds, tabulated_radii, within_cutoff
from ..site_coords import mic_support_positions
from ..site_enumeration import _height_along_slab_normal
from ..site_types import Site
from .context import _HeightInterval

logger = logging.getLogger(__name__)


def _pair_clearance(
    config: AdsorptionConfig, mol_symbol: str, slab_symbol: str
) -> float:
    return geom.min_pair_clearance_angstrom(
        mol_symbol,
        slab_symbol,
        min_distance=float(config.min_initial_distance),
        min_contact_ratio=float(config.min_contact_ratio),
        reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
        vdw_overlap_scale=float(config.vdw_overlap_scale),
    )


def _vdw_pair_clearance(
    config: AdsorptionConfig, mol_symbol: str, slab_symbol: str
) -> float:
    """Van der Waals pair gate (Å): ``vdw_overlap_scale × (R_i + R_j)``."""
    vdw = tabulated_radii((mol_symbol, slab_symbol), kind="vdw")
    return float(
        pair_thresholds(vdw[:1], vdw[1:], float(config.vdw_overlap_scale))[0, 0]
    )


def _com_hi_for_vdw_landing(nominal: float, landing: float) -> float:
    """Choose ``com_hi`` so ``z_fraction`` 0.9 lands on *landing*.

    With the lower bound equal to *nominal*, 0.7 then sits halfway between
    the contact and that landing.
    """
    frac = (_WALL_VDW_Z_FRACTION - 0.5) / 0.5
    gap = max(0.0, float(landing) - float(nominal))
    if frac <= _DISTANCE_ZERO_EPS:
        return float(landing)
    return float(nominal) + gap / frac


def _void_half_width_for_wall_step(gap: float) -> float:
    """Half-width whose 0.2 ``z_fraction`` step matches the wall 0.5→0.7 step.

    The wall step is half of *gap*. On a symmetric void window a 0.2 step is
    ``0.4 × half_width``.
    """
    wall_step = 0.5 * max(float(gap), 0.0)
    return wall_step / 0.4


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


def _framework_plane_height(
    site: Site,
    positions: np.ndarray,
    n_hat: np.ndarray,
    *,
    reduce: Literal["max", "mean"] = "max",
    cell: np.ndarray | None = None,
    pbc: list[bool] | np.ndarray | None = None,
) -> float | None:
    """Height of coordinating framework atoms along *n_hat*, or ``None``.

    Site vertices from any plugin may already sit above their supports
    (topology lift, clearance snap, free-volume). Clearance and height
    offsets must use this plane so lift is applied once. Supports use
    Cartesian MIC of ``site.xyz`` when *cell* / *pbc* are set.

    ``"max"`` clears every support (contact); ``"mean"`` is the NP shell ref.
    """
    if not site.slab_indices:
        return None
    supp = mic_support_positions(
        site.xyz, positions, site.slab_indices, cell=cell, pbc=pbc
    )
    if len(supp) == 0:
        return None
    heights = supp @ _placement_normal_hat(n_hat)
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
    cell: np.ndarray | None = None,
    pbc: list[bool] | np.ndarray | None = None,
) -> float:
    """Support-plane height along *n_hat*, or *fallback* when supports are empty."""
    fw = _framework_plane_height(
        site, positions, n_hat, reduce=reduce, cell=cell, pbc=pbc
    )
    return float(fallback) if fw is None else fw


class _ContactFrame(NamedTuple):
    """MIC geometry shared by every clearance gate of one rigid pose."""

    mic_n: np.ndarray
    lateral: np.ndarray
    base_h: float
    rotated: np.ndarray
    n_hat: np.ndarray
    empty: bool


def _contact_frame(
    rotated_pos: np.ndarray,
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
) -> _ContactFrame:
    """Minimum-image mol–shell geometry at the support-plane seed."""
    n_hat = _placement_normal_hat(place_normal)
    base = np.asarray(site_xyz, dtype=float)
    base_h = float(np.dot(base, n_hat))
    rotated = np.asarray(rotated_pos, dtype=float)
    slab_pos = np.asarray(slab_positions, dtype=float)
    if len(slab_pos) == 0 or len(rotated) == 0:
        return _ContactFrame(
            mic_n=np.empty((0, 0), dtype=float),
            lateral=np.empty((0, 0), dtype=float),
            base_h=base_h,
            rotated=rotated,
            n_hat=n_hat,
            empty=True,
        )
    mol_seed = rotated + base
    mic_vecs, _ = geom._mol_slab_pairwise_mic(mol_seed, slab_pos, cell, pbc)
    mic_n = np.einsum("ijd,d->ij", mic_vecs, n_hat)
    perp = mic_vecs - mic_n[:, :, None] * n_hat.reshape(1, 1, 3)
    lateral = np.linalg.norm(perp, axis=2)
    return _ContactFrame(
        mic_n=mic_n,
        lateral=lateral,
        base_h=base_h,
        rotated=rotated,
        n_hat=n_hat,
        empty=False,
    )


def _allowed_clearances(
    symbols: Sequence[str],
    slab_symbols: Sequence[str],
    config: AdsorptionConfig,
    clearance: Callable[[AdsorptionConfig, str, str], float] | None,
) -> np.ndarray:
    """Pair gates shaped ``(n_mol, n_slab)``. Unique element pairs are tabulated once."""
    gate = _pair_clearance if clearance is None else clearance
    mol_index: dict[str, int] = {}
    slab_index: dict[str, int] = {}
    for sym in symbols:
        if sym not in mol_index:
            mol_index[sym] = len(mol_index)
    for sym in slab_symbols:
        if sym not in slab_index:
            slab_index[sym] = len(slab_index)
    table = np.empty((len(mol_index), len(slab_index)), dtype=float)
    for sym_m, i in mol_index.items():
        for sym_s, j in slab_index.items():
            table[i, j] = float(gate(config, sym_m, sym_s))
    mol_ids = np.fromiter(
        (mol_index[s] for s in symbols), dtype=int, count=len(symbols)
    )
    slab_ids = np.fromiter(
        (slab_index[s] for s in slab_symbols), dtype=int, count=len(slab_symbols)
    )
    return table[np.ix_(mol_ids, slab_ids)]


def _com_height_from_frame(
    frame: _ContactFrame,
    allowed: np.ndarray,
    *,
    apply_anchor_floor: bool,
) -> float:
    """COM height that clears *allowed*, from a frame already solved at the seed."""
    if frame.empty:
        return frame.base_h
    lateral = frame.lateral
    active = lateral < allowed
    vertical = np.sqrt(np.maximum(0.0, allowed * allowed - lateral * lateral))
    h_pair = frame.base_h + vertical - frame.mic_n
    h_needed = float(np.max(h_pair[active])) if np.any(active) else float("-inf")
    if not np.isfinite(h_needed):
        # Every pair already clears laterally; sit the lowest atom at its
        # nearest-support gate so the pose is still a contact, not a hover.
        rel_h = frame.rotated @ frame.n_hat
        i_low = int(np.argmin(rel_h))
        j_near = int(np.argmin(lateral[i_low]))
        h_needed = frame.base_h + float(allowed[i_low, j_near]) - float(rel_h[i_low])
    if apply_anchor_floor:
        h_needed = max(
            float(h_needed),
            _com_floor_on_support_plane(frame.rotated, frame.n_hat, frame.base_h),
        )
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
    symbols: list[str],
    *,
    site_xyz: np.ndarray,
    place_normal: np.ndarray,
    slab_positions: np.ndarray,
    slab_symbols: list[str],
    cell: np.ndarray,
    pbc: list[bool],
    com_h: float,
    contact_covalent_scale: float,
) -> int:
    """Count adsorbate atoms within a covalent-sum contact of any slab atom."""
    mol = _mol_positions_at_com_height(
        rotated_pos, site_xyz=site_xyz, place_normal=place_normal, com_h=com_h
    )
    dists = geom._mol_slab_pairwise_distances(mol, slab_positions, cell, pbc)
    if dists.size == 0:
        return 0
    thresh = pair_thresholds(
        tabulated_radii(symbols, kind="covalent"),
        tabulated_radii(slab_symbols, kind="covalent"),
        contact_covalent_scale,
    )
    return int(np.sum(np.any(within_cutoff(dists, thresh), axis=1)))


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
    allowed = _allowed_clearances(symbols, slab_symbols, config, None)
    mol0 = _mol_positions_at_com_height(
        rotated_pos, site_xyz=base, place_normal=place_normal, com_h=base_h
    )

    def _pen(h: float) -> float:
        mol = mol0 + (float(h) - base_h) * n_hat
        dists = geom._mol_slab_pairwise_distances(mol, slab_positions, cell, pbc)
        if dists.size == 0:
            return 0.0
        return float(np.max(allowed - dists))

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
    for sym_m in set(symbols):
        for sym_s in set(slab_symbols):
            gate = max(
                _pair_clearance(config, sym_m, sym_s),
                _vdw_pair_clearance(config, sym_m, sym_s),
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
) -> _HeightInterval | None:
    """Feasible COM height interval along *place_normal* for one rigid pose.

    Wall (``site.kind == "wall"``): nominal and lower bound are the covalent
    contact. ``z_fraction`` 0.5, 0.7, and 0.9 land on that contact, the
    halfway point, and the van der Waals standoff. An explicit
    ``max_initial_distance`` or strict ``max_closest_approach`` caps the
    standoff.

    Void: nominal is the void-centre height when it clears. The half-width
    makes a 0.2 ``z_fraction`` step the same distance as the wall step from
    0.5 to 0.7. Bounds then shrink to the cleared span.
    """
    base = np.asarray(site.xyz, dtype=float)
    is_void = site.kind == "void"

    shell_idx = _contact_shell_indices(
        base,
        rotated_pos=rotated_pos,
        symbols=symbols,
        slab_positions=slab_positions,
        slab_symbols=slab_symbols,
        config=config,
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

    frame = _contact_frame(
        rotated_pos,
        site_xyz=base,
        place_normal=place_normal,
        slab_positions=shell_pos,
        cell=cell,
        pbc=pbc,
    )
    if frame.empty:
        contact_h = frame.base_h
        vdw_h = contact_h
        raw_contact = contact_h
        raw_vdw = contact_h
    else:
        cov_allowed = _allowed_clearances(symbols, shell_syms, config, None)
        vdw_allowed = _allowed_clearances(
            symbols, shell_syms, config, _vdw_pair_clearance
        )
        contact_h = _com_height_from_frame(frame, cov_allowed, apply_anchor_floor=True)
        vdw_h = _com_height_from_frame(frame, vdw_allowed, apply_anchor_floor=True)
        raw_contact = _com_height_from_frame(
            frame, cov_allowed, apply_anchor_floor=False
        )
        raw_vdw = _com_height_from_frame(frame, vdw_allowed, apply_anchor_floor=False)
    landing = max(float(vdw_h), float(contact_h) + _WALL_SHELL_MIN_GAP_ANGSTROM)

    if is_void:
        raw_landing = max(
            float(raw_vdw), float(raw_contact) + _WALL_SHELL_MIN_GAP_ANGSTROM
        )
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
            diversity=2.0
            * _void_half_width_for_wall_step(raw_landing - float(raw_contact)),
        )
    else:
        com_nominal = float(contact_h)
        com_lo = float(com_nominal)
        upper_targets: list[float] = []
        if config.max_initial_distance is not None:
            upper_targets.append(float(config.max_initial_distance))
        if config.strict_initial_placement:
            upper_targets.append(float(config.max_closest_approach))
        if upper_targets:
            actual = _pair_min_distance_at_com_height(
                rotated_pos,
                site_xyz=base,
                place_normal=place_normal,
                slab_positions=shell_pos,
                cell=cell,
                pbc=pbc,
                com_h=com_nominal,
            )
            slack = max(0.0, float(min(upper_targets)) - float(actual))
            landing = min(landing, com_nominal + slack)
            landing = max(landing, com_nominal)
        com_hi = _com_hi_for_vdw_landing(com_nominal, landing)

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
            symbols,
            site_xyz=base,
            place_normal=place_normal,
            slab_positions=shell_pos,
            slab_symbols=shell_syms,
            cell=cell,
            pbc=pbc,
            com_h=com_nominal,
            contact_covalent_scale=float(config.contact_covalent_scale),
        )
        contact_ok = n_contact >= min_contacts
        if config.strict_initial_placement:
            actual_nom = _pair_min_distance_at_com_height(
                rotated_pos,
                site_xyz=base,
                place_normal=place_normal,
                slab_positions=shell_pos,
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
            _height_above_supports(
                site,
                pos,
                n_hat,
                reduce="max",
                fallback=vertex_h,
                cell=cell_arr,
                pbc=material_aware_pbc(mat_type),
            ),
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

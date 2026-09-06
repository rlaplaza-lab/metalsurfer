"""Packmol-style overlap penalty and bounded rigid-body clash descent.

Used by placement distance recovery and n-tuplet near-miss / pre-relax packing.
Does not call Packmol; the merit function matches the distance term of Martinez
et al., J. Comput. Chem. 30, 2157 (2009).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from ase import Atoms
from scipy.optimize import minimize

from ..config import AdsorptionConfig
from . import geometry as geom
from ._constants import (
    _ADSORBATE_COVALENT_RADIUS_FALLBACK,
    _CLASH_AZIMUTH_DELTA_EPS_DEG,
    _CLASH_DESCENT_AZIMUTH_BOUND_DEG,
    _CLASH_DESCENT_MAXITER,
    _CLASH_DESCENT_SUCCESS_F,
    _CLASH_DESCENT_SUCCESS_VIOLATION_ANGSTROM,
    _CLASH_LATERAL_FOOTPRINT_SCALE,
    _DISTANCE_ZERO_EPS,
    _TUPLET_CLASH_RESCUE_COVALENT_SCALE,
    _VECTOR_NORM_EPS,
)

logger = logging.getLogger(__name__)

__all__ = [
    "atom_radii_for_symbols",
    "clash_bounds_for_adsorbate",
    "compose_quaternion_with_azimuth",
    "overlap_penalty",
    "resolve_rigid_clash",
    "tuplet_clash_rescue_floor",
]


def clash_bounds_for_adsorbate(
    adsorbate: Atoms,
    config: AdsorptionConfig,
    *,
    z_window: float | None = None,
    footprint_radius: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return ``(x_range, y_range, dz_bound)`` scaled by footprint / height window.

    Zero-width XY ranges stay disabled (height-only recovery).
    """
    min_sep = float(config.min_adsorbate_separation)
    r_char = float(footprint_radius) if footprint_radius is not None else 0.0
    if r_char <= _DISTANCE_ZERO_EPS:
        radii = atom_radii_for_symbols(
            list(adsorbate.get_chemical_symbols()),
            min_separation=min_sep,
            use_vdw=False,
        )
        r_char = (
            float(np.mean(radii))
            if radii.size
            else max(min_sep / 2.0, _ADSORBATE_COVALENT_RADIUS_FALLBACK)
        )

    lat = float(_CLASH_LATERAL_FOOTPRINT_SCALE) * r_char
    x_lo, x_hi = (float(v) for v in config.placement_x_range)
    y_lo, y_hi = (float(v) for v in config.placement_y_range)
    if (
        abs(x_hi - x_lo) <= _DISTANCE_ZERO_EPS
        and abs(y_hi - y_lo) <= _DISTANCE_ZERO_EPS
    ):
        x_range, y_range = (x_lo, x_hi), (y_lo, y_hi)
    else:
        x_range = (min(x_lo, -lat), max(x_hi, lat))
        y_range = (min(y_lo, -lat), max(y_hi, lat))
    dz_bound = (
        float(z_window)
        if z_window is not None and float(z_window) > _DISTANCE_ZERO_EPS
        else r_char
    )
    return x_range, y_range, dz_bound


def tuplet_clash_rescue_floor(
    moving_symbols: Sequence[str],
    fixed_symbols: Sequence[str],
    *,
    min_separation: float,
) -> float:
    """Skip n-tuplet rescue when atoms are closer than this (stacked nuclei).

    Floor is ``scale * min(covalent_i + covalent_j)`` over symbol pairs, never
    above ``min_separation`` so a configured packing gap still governs.
    """
    mov_r = atom_radii_for_symbols(moving_symbols, min_separation=min_separation)
    fix_r = atom_radii_for_symbols(fixed_symbols, min_separation=min_separation)
    if mov_r.size == 0 or fix_r.size == 0:
        return float(_TUPLET_CLASH_RESCUE_COVALENT_SCALE) * float(min_separation)
    pair_min = float(np.min(mov_r[:, None] + fix_r[None, :]))
    return min(
        float(min_separation),
        float(_TUPLET_CLASH_RESCUE_COVALENT_SCALE) * pair_min,
    )


def atom_radii_for_symbols(
    symbols: Sequence[str],
    *,
    min_separation: float,
    use_vdw: bool = False,
) -> np.ndarray:
    """Per-atom radii (Å); unknown symbols fall back to ``min_separation / 2``.

    Parameters
    ----------
    symbols
        Chemical symbols.
    min_separation
        Floor used when a tabulated radius is missing (Packmol ``dtol/2``).
    use_vdw
        If True, prefer van der Waals radii; otherwise covalent.
    """
    floor = float(min_separation) / 2.0
    out = np.empty(len(symbols), dtype=float)
    for i, sym in enumerate(symbols):
        r = geom._get_vdw_radius(sym) if use_vdw else geom._get_covalent_radius(sym)
        out[i] = float(r) if r is not None else floor
    return out


def compose_quaternion_with_azimuth(
    quat_wxyz: Sequence[float] | np.ndarray,
    az_delta_deg: float | None,
    normal: np.ndarray,
) -> tuple[float, float, float, float]:
    """Left-compose a surface-normal azimuth into a ``(w, x, y, z)`` quaternion."""
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    if az_delta_deg is None or abs(float(az_delta_deg)) <= float(
        _CLASH_AZIMUTH_DELTA_EPS_DEG
    ):
        return float(q[0]), float(q[1]), float(q[2]), float(q[3])
    R_old = geom.quaternion_to_rotation_matrix(q)
    R_az = geom._rotation_around_axis(normal, float(az_delta_deg))
    q_new = geom.rotation_matrix_to_quaternion(R_az @ R_old)
    return float(q_new[0]), float(q_new[1]), float(q_new[2]), float(q_new[3])


def _pair_thresholds(
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    min_separation: float | None,
) -> np.ndarray:
    """Pairwise separation thresholds ``(n_moving, n_fixed)``."""
    thresh = moving_radii[:, None] + fixed_radii[None, :]
    if min_separation is not None:
        thresh = np.maximum(thresh, float(min_separation))
    return thresh


def overlap_penalty(
    moving_pos: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float | None = None,
) -> float:
    """Packmol distance-term merit: sum of squared positive overlaps.

    ``f = sum_ij [max(0, (r_i + r_j)^2 - d_ij^2)]^2``. When *min_separation*
    is set, each pair threshold is ``max(r_i + r_j, min_separation)``.

    Parameters
    ----------
    moving_pos
        Moving atom positions ``(n, 3)``.
    moving_radii
        Radii for moving atoms ``(n,)``.
    fixed_pos
        Fixed atom positions ``(m, 3)``.
    fixed_radii
        Radii for fixed atoms ``(m,)``.
    cell
        Unit cell matrix.
    pbc
        Periodic boundary flags.
    min_separation
        Optional hard floor on pairwise separation (Å).
    """
    f, _grad = _overlap_penalty_and_pos_grad(
        moving_pos,
        moving_radii,
        fixed_pos,
        fixed_radii,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )
    return f


def _overlap_penalty_and_pos_grad(
    moving_pos: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float | None,
) -> tuple[float, np.ndarray]:
    """Return ``(f, df/dp)`` with ``df/dp`` shape ``(n_moving, 3)``."""
    mov = np.asarray(moving_pos, dtype=float)
    fix = np.asarray(fixed_pos, dtype=float)
    if mov.size == 0 or fix.size == 0:
        return 0.0, np.zeros_like(mov)
    r_m = np.asarray(moving_radii, dtype=float).reshape(-1)
    r_f = np.asarray(fixed_radii, dtype=float).reshape(-1)
    if r_m.shape[0] != mov.shape[0] or r_f.shape[0] != fix.shape[0]:
        raise ValueError(
            "radii length must match positions: "
            f"moving {r_m.shape[0]} vs {mov.shape[0]}, "
            f"fixed {r_f.shape[0]} vs {fix.shape[0]}"
        )
    mic_vecs, dists = geom._mol_slab_pairwise_mic(mov, fix, cell, pbc)
    thresh = _pair_thresholds(r_m, r_f, min_separation)
    overlap = np.maximum(0.0, thresh * thresh - dists * dists)
    f = float(np.sum(overlap * overlap))
    # df/dp_i = sum_j -4 * o_ij * mic_vec_ij  for overlapping pairs.
    weights = -4.0 * overlap  # (n, m)
    grad = np.einsum("ij,ijk->ik", weights, mic_vecs)
    return f, grad


def _max_pair_violation(
    moving_pos: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float | None,
) -> float:
    """Largest positive (threshold - distance) over pairs; 0 if clear."""
    mov = np.asarray(moving_pos, dtype=float)
    fix = np.asarray(fixed_pos, dtype=float)
    if mov.size == 0 or fix.size == 0:
        return 0.0
    r_m = np.asarray(moving_radii, dtype=float).reshape(-1)
    r_f = np.asarray(fixed_radii, dtype=float).reshape(-1)
    _, dists = geom._mol_slab_pairwise_mic(mov, fix, cell, pbc)
    thresh = _pair_thresholds(r_m, r_f, min_separation)
    return float(np.max(np.maximum(0.0, thresh - dists)))


def _apply_rigid_state(
    base_pos: np.ndarray,
    origin: np.ndarray,
    site_frame: np.ndarray,
    normal: np.ndarray,
    state: np.ndarray,
) -> np.ndarray:
    """Map ``[dx, dy, dz, d_az_deg]`` to world-frame positions."""
    dx, dy, dz, d_az = (
        float(state[0]),
        float(state[1]),
        float(state[2]),
        float(state[3]),
    )
    centered = np.asarray(base_pos, dtype=float) - origin
    if abs(d_az) > float(_CLASH_AZIMUTH_DELTA_EPS_DEG):
        R = geom._rotation_around_axis(normal, d_az)
        centered = (R @ centered.T).T
    shift = site_frame @ np.array([dx, dy, dz], dtype=float)
    return centered + origin + shift


def _rigid_state_objective_and_jac(
    state: np.ndarray,
    *,
    base_pos: np.ndarray,
    origin: np.ndarray,
    site_frame: np.ndarray,
    normal: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float | None,
) -> tuple[float, np.ndarray]:
    """Value and analytic Jacobian of the Packmol merit wrt rigid state."""
    pos = _apply_rigid_state(base_pos, origin, site_frame, normal, state)
    f, pos_grad = _overlap_penalty_and_pos_grad(
        pos,
        moving_radii,
        fixed_pos,
        fixed_radii,
        cell=cell,
        pbc=pbc,
        min_separation=min_separation,
    )
    # d(pos)/d(local translation k) equals the k-th site-frame basis vector.
    jac = np.zeros(4, dtype=float)
    for k in range(3):
        jac[k] = float(np.sum(pos_grad * site_frame[:, k]))
    # Azimuth in degrees: rotate the COM-centred base positions about *normal*.
    d_az = float(state[3])
    centered0 = np.asarray(base_pos, dtype=float) - origin
    theta = np.radians(d_az)
    # Rodrigues derivative: dR/dθ = cosθ K + sinθ K² for skew-symmetric K(n).
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    K = np.array(
        [[0.0, -nz, ny], [nz, 0.0, -nx], [-ny, nx, 0.0]],
        dtype=float,
    )
    dR_dtheta = np.cos(theta) * K + np.sin(theta) * (K @ K)
    dpos_daz = (dR_dtheta @ centered0.T).T * (np.pi / 180.0)
    jac[3] = float(np.sum(pos_grad * dpos_daz))
    return f, jac


def resolve_rigid_clash(
    adsorbate: Atoms,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    origin: np.ndarray,
    site_frame: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
    config: AdsorptionConfig,
    rotate_azimuth: bool = True,
    include_substrate_min_sep: bool = False,
    use_vdw_moving: bool = False,
    bounds: tuple[tuple[float, float], tuple[float, float], float] | None = None,
) -> tuple[np.ndarray, float | None, bool]:
    """Bounded L-BFGS-B rigid-body descent to clear overlaps with *fixed* atoms.

    State is ``[dx, dy, dz, d_az_deg]`` in the local site frame. Bounds default to
    :func:`clash_bounds_for_adsorbate`, or pass ``(x_range, y_range, dz_bound)``.
    """
    base_pos = np.asarray(adsorbate.get_positions(), dtype=float)
    origin_arr = np.asarray(origin, dtype=float).reshape(3)
    frame = np.asarray(site_frame, dtype=float).reshape(3, 3)
    normal = frame[:, 2].copy()
    nrm = float(np.linalg.norm(normal))
    if nrm > _VECTOR_NORM_EPS:
        normal = normal / nrm

    moving_radii = atom_radii_for_symbols(
        list(adsorbate.get_chemical_symbols()),
        min_separation=float(config.min_adsorbate_separation),
        use_vdw=use_vdw_moving,
    )
    min_sep = (
        float(config.min_adsorbate_separation) if include_substrate_min_sep else None
    )
    fix = np.asarray(fixed_pos, dtype=float)
    fix_r = np.asarray(fixed_radii, dtype=float).reshape(-1)

    f0 = overlap_penalty(
        base_pos,
        moving_radii,
        fix,
        fix_r,
        cell=cell,
        pbc=pbc,
        min_separation=min_sep,
    )
    if f0 <= _CLASH_DESCENT_SUCCESS_F:
        return base_pos.copy(), 0.0 if rotate_azimuth else None, True

    x_range, y_range, dz_b = bounds or clash_bounds_for_adsorbate(adsorbate, config)
    az_b = float(_CLASH_DESCENT_AZIMUTH_BOUND_DEG) if rotate_azimuth else 0.0
    bounds_list = [
        (float(x_range[0]), float(x_range[1])),
        (float(y_range[0]), float(y_range[1])),
        (-float(dz_b), float(dz_b)),
        (-az_b, az_b),
    ]

    def objective(state: np.ndarray) -> tuple[float, np.ndarray]:
        return _rigid_state_objective_and_jac(
            state,
            base_pos=base_pos,
            origin=origin_arr,
            site_frame=frame,
            normal=normal,
            moving_radii=moving_radii,
            fixed_pos=fix,
            fixed_radii=fix_r,
            cell=cell,
            pbc=pbc,
            min_separation=min_sep,
        )

    result = minimize(
        objective,
        x0=np.zeros(4, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds_list,
        options={"maxiter": int(_CLASH_DESCENT_MAXITER)},
    )
    best_state = np.asarray(result.x, dtype=float)
    best_pos = _apply_rigid_state(base_pos, origin_arr, frame, normal, best_state)
    f_best = float(result.fun)
    viol = _max_pair_violation(
        best_pos,
        moving_radii,
        fix,
        fix_r,
        cell=cell,
        pbc=pbc,
        min_separation=min_sep,
    )
    ok = f_best <= _CLASH_DESCENT_SUCCESS_F or viol <= float(
        _CLASH_DESCENT_SUCCESS_VIOLATION_ANGSTROM
    )
    az_delta = float(best_state[3]) if rotate_azimuth else None
    if not ok:
        logger.debug(
            "clash descent failed: f=%.3e violation=%.4f A",
            f_best,
            viol,
        )
        return best_pos, az_delta, False
    return best_pos, az_delta, True

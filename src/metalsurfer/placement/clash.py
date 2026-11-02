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
    _CLASH_AZIMUTH_DELTA_EPS_DEG,
    _CLASH_DESCENT_AZIMUTH_BOUND_DEG,
    _CLASH_DESCENT_MAXITER,
    _CLASH_LATERAL_FOOTPRINT_SCALE,
    _DISTANCE_ZERO_EPS,
    _TUPLET_CLASH_RESCUE_COVALENT_SCALE,
    _VECTOR_NORM_EPS,
)
from .cutoffs import clear_of_cutoff, tabulated_radii

logger = logging.getLogger(__name__)

# Merit target sits this far outside the acceptance cutoff so a zero merit
# is strictly clear. The accept/reject predicate uses the unpadded threshold.
_MERIT_CLEARANCE_PAD_ANGSTROM: float = 1e-4

__all__ = [
    "atom_radii_for_symbols",
    "clash_bounds_for_adsorbate",
    "compose_quaternion_with_azimuth",
    "overlap_penalty",
    "resolve_rigid_clash",
    "tuplet_clash_rescue_floor",
]


def atom_radii_for_symbols(
    symbols: Sequence[str],
    *,
    use_vdw: bool = False,
) -> np.ndarray:
    """Return per-atom tabulated radii (Å). Unknown symbols raise.

    Parameters
    ----------
    symbols
        Chemical symbols.
    use_vdw
        If True, use van der Waals radii; otherwise covalent.
    """
    return tabulated_radii(symbols, kind="vdw" if use_vdw else "covalent")


def clash_bounds_for_adsorbate(
    adsorbate: Atoms,
    config: AdsorptionConfig,
    *,
    z_window: float | None = None,
    footprint_radius: float | None = None,
    moving_radii: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Return ``(x_range, y_range, dz_bound)`` scaled by footprint / height window.

    Zero-width XY ranges stay disabled (height-only recovery).
    When *footprint_radius* is missing or near-zero, prefer *moving_radii* (mean)
    over recomputing atom radii.
    """
    r_char = float(footprint_radius) if footprint_radius is not None else 0.0
    if r_char <= _DISTANCE_ZERO_EPS:
        moving = (
            np.asarray(moving_radii, dtype=float) if moving_radii is not None else None
        )
        if moving is not None and moving.size:
            r_char = float(np.mean(moving))
        else:
            radii = atom_radii_for_symbols(
                list(adsorbate.get_chemical_symbols()),
                use_vdw=False,
            )
            if radii.size == 0:
                raise ValueError(
                    "clash_bounds_for_adsorbate requires a non-empty adsorbate"
                )
            r_char = float(np.mean(radii))

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
) -> float:
    """Skip n-tuplet rescue when atoms are closer than this (stacked nuclei).

    Floor is ``scale * min(covalent_i + covalent_j)`` over symbol pairs.
    """
    if not moving_symbols or not fixed_symbols:
        raise ValueError(
            "tuplet_clash_rescue_floor requires non-empty moving and fixed symbols"
        )
    mov_r = atom_radii_for_symbols(moving_symbols)
    fix_r = atom_radii_for_symbols(fixed_symbols)
    pair_min = float(np.min(mov_r[:, None] + fix_r[None, :]))
    return float(_TUPLET_CLASH_RESCUE_COVALENT_SCALE) * pair_min


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


def _normalize_pair_scales(
    pair_scales: np.ndarray | float | None,
    n_fixed: int,
) -> np.ndarray | None:
    """Broadcast a scalar / per-fixed scale array, or return ``None``."""
    if pair_scales is None:
        return None
    if isinstance(pair_scales, np.ndarray):
        scales = np.asarray(pair_scales, dtype=float).reshape(-1)
        if scales.shape[0] != n_fixed:
            raise ValueError(
                f"pair_scales length ({scales.shape[0]}) must match n_fixed ({n_fixed})"
            )
        return scales
    scale = float(pair_scales)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"pair_scales must be positive and finite, got {scale!r}")
    return np.full(n_fixed, scale, dtype=float)


def _pair_thresholds(
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    pair_scales: np.ndarray | None,
) -> np.ndarray:
    """Pairwise separation thresholds ``(n_moving, n_fixed)``.

    Base gap is ``r_i + r_j``. When *pair_scales* is set, column *j* uses
    ``pair_scales[j] * (r_i + r_j)``.
    """
    thresh = moving_radii[:, None] + fixed_radii[None, :]
    if pair_scales is not None:
        thresh = thresh * pair_scales[None, :]
    return thresh


def _acceptance_thresholds(
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    pair_scales: np.ndarray | float | None,
    thresholds: np.ndarray | None,
) -> np.ndarray:
    """Return acceptance thresholds. Explicit *thresholds* win over *pair_scales*."""
    if thresholds is not None:
        acc = np.asarray(thresholds, dtype=float)
        expected = (int(moving_radii.shape[0]), int(fixed_radii.shape[0]))
        if acc.shape != expected:
            raise ValueError(
                f"thresholds shape {acc.shape} must be {expected} (n_moving, n_fixed)"
            )
        return acc
    scales = _normalize_pair_scales(pair_scales, int(fixed_radii.shape[0]))
    return _pair_thresholds(moving_radii, fixed_radii, scales)


def overlap_penalty(
    moving_pos: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    pair_scales: np.ndarray | float | None = None,
    thresholds: np.ndarray | None = None,
) -> float:
    """Packmol distance-term merit: sum of squared positive overlaps.

    ``f = sum_ij [max(0, merit_ij^2 - d_ij^2)]^2``. Acceptance thresholds are
    ``scale_j * (r_i + r_j)`` or an explicit matrix. The merit target is those
    thresholds plus :data:`_MERIT_CLEARANCE_PAD_ANGSTROM`, so a zero merit is
    strictly outside the closed cutoff.

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
    pair_scales
        Optional per-fixed-atom (or scalar) multiplier on ``r_i + r_j``.
    """
    f, _grad = _overlap_penalty_and_pos_grad(
        moving_pos,
        moving_radii,
        fixed_pos,
        fixed_radii,
        cell=cell,
        pbc=pbc,
        pair_scales=pair_scales,
        thresholds=thresholds,
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
    pair_scales: np.ndarray | float | None,
    thresholds: np.ndarray | None = None,
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
    acceptance = _acceptance_thresholds(r_m, r_f, pair_scales, thresholds)
    thresh = acceptance + _MERIT_CLEARANCE_PAD_ANGSTROM
    overlap = np.maximum(0.0, thresh * thresh - dists * dists)
    f = float(np.sum(overlap * overlap))
    # df/dp_i = sum_j -4 * o_ij * mic_vec_ij  for overlapping pairs.
    weights = -4.0 * overlap  # (n, m)
    grad = np.einsum("ij,ijk->ik", weights, mic_vecs)
    return f, grad


def _overlap_f_and_max_violation(
    moving_pos: np.ndarray,
    moving_radii: np.ndarray,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    cell: np.ndarray,
    pbc: list[bool],
    pair_scales: np.ndarray | float | None = None,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return ``(overlap_penalty_f, max_pair_violation)`` from one MIC.

    Violation is measured against the unpadded acceptance threshold.
    """
    mov = np.asarray(moving_pos, dtype=float)
    fix = np.asarray(fixed_pos, dtype=float)
    if mov.size == 0 or fix.size == 0:
        return 0.0, 0.0
    r_m = np.asarray(moving_radii, dtype=float).reshape(-1)
    r_f = np.asarray(fixed_radii, dtype=float).reshape(-1)
    _, dists = geom._mol_slab_pairwise_mic(mov, fix, cell, pbc)
    acceptance = _acceptance_thresholds(r_m, r_f, pair_scales, thresholds)
    merit = acceptance + _MERIT_CLEARANCE_PAD_ANGSTROM
    overlap = np.maximum(0.0, merit * merit - dists * dists)
    f = float(np.sum(overlap * overlap))
    viol = float(np.max(np.maximum(0.0, acceptance - dists)))
    return f, viol


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
    pair_scales: np.ndarray | float | None,
    thresholds: np.ndarray | None,
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
        pair_scales=pair_scales,
        thresholds=thresholds,
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
    fixed_pair_scales: np.ndarray | float | None = None,
    thresholds: np.ndarray | None = None,
    use_vdw_moving: bool = False,
    bounds: tuple[tuple[float, float], tuple[float, float], float] | None = None,
    moving_radii: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None, bool]:
    """Bounded L-BFGS-B rigid-body descent to clear overlaps with *fixed* atoms.

    State is ``[dx, dy, dz, d_az_deg]`` in the local site frame. Bounds default to
    :func:`clash_bounds_for_adsorbate`, or pass ``(x_range, y_range, dz_bound)``.

    *thresholds* are the acceptance cutoffs ``(n_moving, n_fixed)``. When
    omitted, *fixed_pair_scales* multiplies ``r_i + r_j`` per fixed atom.
    Success requires every pair strictly outside those unpadded thresholds.
    The merit itself targets the thresholds plus a 1e-4 Å pad.
    """
    base_pos = np.asarray(adsorbate.get_positions(), dtype=float)
    origin_arr = np.asarray(origin, dtype=float).reshape(3)
    frame = np.asarray(site_frame, dtype=float).reshape(3, 3)
    normal = frame[:, 2].copy()
    nrm = float(np.linalg.norm(normal))
    if nrm > _VECTOR_NORM_EPS:
        normal = normal / nrm

    if moving_radii is None:
        moving_radii = atom_radii_for_symbols(
            list(adsorbate.get_chemical_symbols()),
            use_vdw=use_vdw_moving,
        )
    else:
        moving_radii = np.asarray(moving_radii, dtype=float).reshape(-1)
    fix = np.asarray(fixed_pos, dtype=float)
    fix_r = np.asarray(fixed_radii, dtype=float).reshape(-1)
    acceptance = _acceptance_thresholds(
        moving_radii, fix_r, fixed_pair_scales, thresholds
    )

    def _pairs_clear(pos: np.ndarray) -> bool:
        if pos.size == 0 or fix.size == 0:
            return True
        _, dists = geom._mol_slab_pairwise_mic(pos, fix, cell, pbc)
        return bool(np.all(clear_of_cutoff(dists, acceptance)))

    if _pairs_clear(base_pos):
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
            pair_scales=None,
            thresholds=acceptance,
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
    f_best, viol = _overlap_f_and_max_violation(
        best_pos,
        moving_radii,
        fix,
        fix_r,
        cell=cell,
        pbc=pbc,
        thresholds=acceptance,
    )
    ok = _pairs_clear(best_pos)
    az_delta = float(best_state[3]) if rotate_azimuth else None
    if not ok:
        logger.debug(
            "clash descent failed: f=%.3e violation=%.4f A",
            f_best,
            viol,
        )
        return best_pos, az_delta, False
    return best_pos, az_delta, True

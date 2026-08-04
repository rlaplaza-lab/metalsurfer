"""Rotation, inertia, and distance helpers for adsorbate placement."""

import functools
import logging
import random
from typing import Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii
from ase.geometry import find_mic

from ._constants import (
    _ADSORBATE_SEPARATION_COVALENT_SUM_SCALE,
    _BINDER_ALIGNMENT_TARGET_DOT,
    _BINDER_VECTOR_MIN_NORM,
    _CONTACT_ATOM_VARIANCE_MAX,
    _CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM,
    _CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM,
    _CONTACT_QUALITY_COVALENT_SUM_SCALE,
    _FLAT_SHAPE_I1_I3_MAX,
    _FLAT_SHAPE_I2_I3_MIN,
    _FRAME_PROJECTION_TIE_EPS,
    _FRAME_REF_ALIGNMENT_DOT_THRESHOLD,
    _INERTIA_EPS,
    _LINEAR_SHAPE_RATIO_MAX,
    _MIN_CONTACT_RATIO_DEFAULT,
    _MIN_DISTANCE_COVALENT_FALLBACK_SCALE,
    _MIN_DISTANCE_HARD_FALLBACK_ANGSTROM,
    _MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
    _PRINCIPAL_AXIS_LONG_ALIGN_MIN_DOT,
    _PRINCIPAL_AXIS_ROT_AXIS_MIN_NORM,
    _PRINCIPAL_AXIS_ROTATION_STEP_DEG,
    _PRINCIPAL_AXIS_ROTATION_STEPS,
    _PRINCIPAL_AXIS_SHORT_ALIGN_MAX_DOT,
    _QUATERNION_NORM_EPS,
    _ROTATION_ALIGN_AXIS_SWITCH_DOT,
    _ROTATION_ALIGN_DOT_ANTIPARALLEL,
    _ROTATION_ALIGN_DOT_PARALLEL,
    _VDW_RADIUS_FROM_COVALENT_SCALE,
    _VECTOR_NORM_EPS,
)
from ._material import material_aware_pbc

logger = logging.getLogger(__name__)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Return normalized quaternion [w, x, y, z] with canonical sign."""
    q: np.ndarray = np.asarray(quat, dtype=float).reshape(4)
    nrm = float(np.linalg.norm(q))
    if nrm < _QUATERNION_NORM_EPS:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.asarray(q / nrm, dtype=float)
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    w, x, y, z = normalize_quaternion(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    r = np.asarray(R, dtype=float)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    else:
        if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
    return normalize_quaternion(np.array([w, x, y, z], dtype=float))


def best_fit_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the proper rotation matrix that best maps source to target."""
    src = np.asarray(source, dtype=float)
    dst = np.asarray(target, dtype=float)
    H = src.T @ dst
    U, _S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    return R


def compute_canonical_molecular_frame(
    ads_pos: np.ndarray, symbols: list[str] | None = None
) -> np.ndarray:
    """Return centred positions in a deterministic principal-axis frame."""
    pos = np.asarray(ads_pos, dtype=float).copy()
    pos -= np.mean(pos, axis=0)
    _, eigenvecs = _compute_inertia_tensor(pos)
    frame = eigenvecs.copy()
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0

    syms = (
        symbols if symbols is not None and len(symbols) == len(pos) else [""] * len(pos)
    )
    for axis in range(3):
        projections = pos @ frame[:, axis]
        idx = int(np.argmax(np.abs(projections)))
        if projections[idx] < 0 or (
            abs(projections[idx]) < _FRAME_PROJECTION_TIE_EPS
            and syms[idx]
            and syms[idx] < syms[int(np.argmin(np.abs(projections)))]
        ):
            frame[:, axis] *= -1.0
    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0
    return (frame.T @ pos.T).T


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector, returning the zero vector if norm is near-zero."""
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > _VECTOR_NORM_EPS else np.zeros_like(v)


def compute_surface_site_frame(normal: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal frame whose z-axis is surface normal."""
    z_axis = _safe_normalize(np.asarray(normal, dtype=float))
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, z_axis)) > _FRAME_REF_ALIGNMENT_DOT_THRESHOLD:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _safe_normalize(ref - np.dot(ref, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


@functools.cache
def _get_covalent_radius(symbol: str) -> float | None:
    z = atomic_numbers.get(symbol)
    if z is None or z >= len(ase_covalent_radii):
        return None
    r = float(ase_covalent_radii[z])
    return r if r > 0.0 else None


@functools.cache
def _get_vdw_radius(symbol: str) -> float | None:
    """VdW radius (Å) for overlap checks; NaN tabulated values fall back to ~1.2× covalent."""
    z = atomic_numbers.get(symbol)
    if z is None or z >= len(ase_vdw_radii):
        return None
    r = float(ase_vdw_radii[z])
    if r > 0.0 and not np.isnan(r):
        return r
    cov = _get_covalent_radius(symbol)
    if cov is None:
        return None
    # Tabulated VdW may be NaN (e.g. some transition metals in ASE). Use a mild
    # covalent scale so "touching" physisorptive distances (~3+ Å) are not
    # misclassified as overlaps while still flagging sub-Å spurious approaches.
    return float(cov * _VDW_RADIUS_FROM_COVALENT_SCALE)


def _cell_has_volume(cell: np.ndarray) -> bool:
    """True when *cell* has non-zero volume (supports left-handed cells)."""
    return abs(float(np.linalg.det(np.asarray(cell, dtype=float)))) > 0.0


def _mol_slab_pairwise_distances(
    mol_pos: np.ndarray,
    slab_pos: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool],
) -> np.ndarray:
    """Minimum-image distances between each mol atom and each slab atom, shape (n_m, n_s)."""
    m, s = len(mol_pos), len(slab_pos)
    if m == 0 or s == 0:
        return np.zeros((m, s))
    diffs = mol_pos[:, None, :] - slab_pos[None, :, :]
    if _cell_has_volume(cell) and np.any(pbc):
        diffs_flat = diffs.reshape(-1, 3)
        _, mic_dists = find_mic(diffs_flat, cell, pbc=pbc)
        return mic_dists.reshape(m, s)
    return np.linalg.norm(diffs, axis=2)


def _random_rotation_matrix(rng: random.Random) -> np.ndarray:
    """Return a uniformly random 3x3 rotation matrix (Arvo's method)."""
    u1 = rng.random()
    u2 = rng.random()
    u3 = rng.random()

    theta = 2.0 * np.pi * u1
    phi = 2.0 * np.pi * u2
    z_val = u3

    v = np.array(
        [
            np.cos(phi) * np.sqrt(z_val),
            np.sin(phi) * np.sqrt(z_val),
            np.sqrt(1.0 - z_val),
        ]
    )

    st, ct = np.sin(theta), np.cos(theta)
    Rz = np.array([[ct, st, 0], [-st, ct, 0], [0, 0, 1]])

    # Householder reflection
    H = np.eye(3) - 2.0 * np.outer(v, v)
    return -H @ Rz


def _rodrigues(axis: np.ndarray, c: float, s: float) -> np.ndarray:
    """Rodrigues rotation: I + s·K + (1-c)·K² for a unit *axis*."""
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=float,
    )
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def _rotation_around_axis(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotation matrix for rotation by angle_deg around axis (unit vector)."""
    a = np.radians(angle_deg)
    ax = np.asarray(axis, dtype=float) / (np.linalg.norm(axis) + _VECTOR_NORM_EPS)
    return _rodrigues(ax, np.cos(a), np.sin(a))


def _rotation_to_align_vector_to_target(
    vec: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Rotation matrix that aligns vec (unit) to target (unit)."""
    vec = np.asarray(vec, dtype=float) / (np.linalg.norm(vec) + _VECTOR_NORM_EPS)
    target = np.asarray(target, dtype=float) / (
        np.linalg.norm(target) + _VECTOR_NORM_EPS
    )
    dot = np.clip(np.dot(vec, target), -1.0, 1.0)
    if dot > _ROTATION_ALIGN_DOT_PARALLEL:
        return np.eye(3)
    if dot < _ROTATION_ALIGN_DOT_ANTIPARALLEL:
        # Anti-parallel: 180° rotation around any perpendicular axis.
        axis = (
            np.array([1, 0, 0])
            if abs(vec[0]) < _ROTATION_ALIGN_AXIS_SWITCH_DOT
            else np.array([0, 1, 0])
        )
        axis = np.cross(vec, axis)
        axis /= np.linalg.norm(axis)
        return _rodrigues(axis, -1.0, 0.0)
    axis = np.cross(vec, target)
    axis /= np.linalg.norm(axis)
    return _rodrigues(axis, dot, np.sqrt(max(0.0, 1.0 - dot * dot)))


def _binding_atom_candidates(symbols: list[str]) -> list[int]:
    """Indices of atoms likely to bind (O, N, S, halogens)."""
    binders = {"O", "N", "S", "F", "Cl", "Br", "I"}
    return [i for i, s in enumerate(symbols) if s in binders]


def _compute_inertia_tensor(
    positions: np.ndarray, masses: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Compute inertia tensor and return (eigenvalues, eigenvectors).

    I_ij = sum_k m_k * (|r_k|^2 δ_ij - r_{k,i} r_{k,j}) with COM at origin.
    Eigenvalues are ordered I1 <= I2 <= I3. Eigenvectors are columns of the
    returned matrix, ordered by increasing eigenvalue.

    For flat (planar) molecules: I3 = I1 + I2 (perpendicular axis theorem), so
    eigenvecs[:, 2] is the plane normal.
    """
    pos = np.asarray(positions, dtype=float)
    n = len(pos)
    if masses is None:
        masses = np.ones(n)
    else:
        masses = np.asarray(masses, dtype=float)
        if len(masses) != n:
            raise ValueError(
                f"masses length {len(masses)} does not match positions length {n}"
            )
    com = np.average(pos, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for i in range(n):
        r = pos[i] - com
        m = masses[i]
        inertia[0, 0] += m * (r[1] ** 2 + r[2] ** 2)
        inertia[1, 1] += m * (r[0] ** 2 + r[2] ** 2)
        inertia[2, 2] += m * (r[0] ** 2 + r[1] ** 2)
        inertia[0, 1] -= m * r[0] * r[1]
        inertia[0, 2] -= m * r[0] * r[2]
        inertia[1, 2] -= m * r[1] * r[2]
    inertia[1, 0] = inertia[0, 1]
    inertia[2, 0] = inertia[0, 2]
    inertia[2, 1] = inertia[1, 2]
    eigenvals, eigenvecs = np.linalg.eigh(inertia)
    order = np.argsort(eigenvals)
    return eigenvals[order], eigenvecs[:, order]


def _classify_molecule_shape(
    positions: np.ndarray,
) -> tuple[Literal["linear", "flat", "round"], np.ndarray, np.ndarray]:
    """Classify molecular shape from inertia tensor eigenvalues.

    Returns (shape, eigenvals, eigenvecs) where eigenvals are I1 <= I2 <= I3
    and eigenvecs columns are the corresponding principal axes.
    """
    eigenvals, eigenvecs = _compute_inertia_tensor(positions)
    I1, I2, I3 = eigenvals[0], eigenvals[1], eigenvals[2]
    eps = _INERTIA_EPS
    I3_safe = max(I3, eps)
    I2_safe = max(I2, eps)

    if I1 / I3_safe < _LINEAR_SHAPE_RATIO_MAX:
        return "linear", eigenvals, eigenvecs
    # Flat (oblate): I1 ≈ I2 << I3 (perpendicular axis theorem)
    if (
        I1 / I3_safe < _FLAT_SHAPE_I1_I3_MAX
        and I2_safe / I3_safe > _FLAT_SHAPE_I2_I3_MIN
    ):
        return "flat", eigenvals, eigenvecs
    return "round", eigenvals, eigenvecs


def _flat_orientation_from_principal_axis(
    ads_pos: np.ndarray,
    normal: np.ndarray,
    azimuth_in_plane_deg: float = 0.0,
    face_flip: bool = False,
) -> np.ndarray:
    """Rotate adsorbate so its molecular plane is parallel to the surface.

    Aligns the plane normal (axis of largest inertia for flat molecules;
    perpendicular axis theorem: I3 = I1 + I2 for planar bodies) with the
    surface normal. Returns centred positions.

    When face_flip is True, flips the plane normal so the other face of
    the molecule faces the surface.
    """
    pos = np.asarray(ads_pos, dtype=float).copy()
    com = np.mean(pos, axis=0)
    pos -= com
    normal = np.asarray(normal, dtype=float) / (
        np.linalg.norm(normal) + _VECTOR_NORM_EPS
    )

    _, eigenvecs = _compute_inertia_tensor(pos)
    plane_normal = eigenvecs[:, 2]
    if np.dot(plane_normal, normal) < 0:
        plane_normal = -plane_normal
    if face_flip:
        plane_normal = -plane_normal

    R = _rotation_to_align_vector_to_target(plane_normal, normal)
    pos = (R @ pos.T).T

    R_az = _rotation_around_axis(normal, azimuth_in_plane_deg)
    return (R_az @ pos.T).T


def _surface_aligned_rotation(
    ads_pos: np.ndarray,
    normal: np.ndarray,
    symbols: list[str] | None = None,
    en_binder_index: int | None = None,
    *,
    en_atom_index: int | None = None,
) -> np.ndarray:
    """Rotate adsorbate so a binding vector points toward surface. Returns centred positions.

    When *en_binder_index* (or legacy alias *en_atom_index*) is provided and valid,
    use that index into the filtered electronegative-atom list from
    :func:`_binding_atom_candidates`, not a raw atom index.  Otherwise select
    the binder with highest dot product toward the surface normal.
    """
    binder_idx = en_binder_index if en_binder_index is not None else en_atom_index
    pos = np.asarray(ads_pos, dtype=float).copy()
    com = np.mean(pos, axis=0)
    pos -= com
    normal = np.asarray(normal, dtype=float) / (
        np.linalg.norm(normal) + _VECTOR_NORM_EPS
    )

    binders = _binding_atom_candidates(symbols) if symbols else []
    if binders:
        if binder_idx is not None and binder_idx in range(len(binders)):
            i = binders[binder_idx]
        else:
            best_dot = -1.0
            i = binders[0]
            for idx in binders:
                # pos is already centered above; avoid subtracting COM again.
                v = pos[idx]
                nv = np.linalg.norm(v)
                if nv > _BINDER_VECTOR_MIN_NORM:
                    dot = np.dot(v / nv, normal)
                    if dot > best_dot:
                        best_dot = dot
                        i = idx
        v = pos[i]
        nv = np.linalg.norm(v)
        if nv > _BINDER_VECTOR_MIN_NORM:
            best_vec = v / nv
            dot = np.dot(best_vec, normal)
            if dot < _BINDER_ALIGNMENT_TARGET_DOT:
                R = _rotation_to_align_vector_to_target(-best_vec, normal)
                pos = (R @ pos.T).T
    else:
        rotated, _ = _principal_axis_rotation(pos, normal)
        if rotated is not None:
            pos = rotated
        else:
            pos = np.asarray(ads_pos, dtype=float).copy() - com
    return pos


def _rotation_with_tilt(
    pos: np.ndarray, normal: np.ndarray, tilt_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Apply tilt and azimuth to positions (centred at origin)."""
    frame = compute_surface_site_frame(normal)
    pos_local = (frame.T @ np.asarray(pos, dtype=float).T).T
    R_tilt = _rotation_around_axis(np.array([1.0, 0.0, 0.0]), tilt_deg)
    R_az = _rotation_around_axis(np.array([0.0, 0.0, 1.0]), azimuth_deg)
    pos_local = (R_az @ R_tilt @ pos_local.T).T
    return (frame @ pos_local.T).T


def _principal_axis_rotation(
    adsorbate_positions: np.ndarray,
    normal_vector: np.ndarray,
) -> tuple[np.ndarray | None, float]:
    """Rotate the adsorbate around its principal axes to maximise clearance.

    Returns centred-at-origin positions (no surface_z offset) and the best
    min-z score.
    """
    pos = np.asarray(adsorbate_positions, dtype=float).copy()
    com = np.mean(pos, axis=0)
    normal = np.asarray(normal_vector, dtype=float)
    nrm = float(np.linalg.norm(normal))
    if nrm > _VECTOR_NORM_EPS:
        normal = normal / nrm

    _, eigenvecs = _compute_inertia_tensor(pos)
    principal_axes = eigenvecs

    # Gentle pre-alignment toward surface normal. Skip when already flat-ish:
    # shortest axis already aligned with n, or longest not aligned with n.
    shortest_axis = principal_axes[:, 0]
    longest_axis = principal_axes[:, -1]
    needs_prealign = (
        abs(float(np.dot(shortest_axis, normal))) < _PRINCIPAL_AXIS_SHORT_ALIGN_MAX_DOT
        and abs(float(np.dot(longest_axis, normal)))
        > _PRINCIPAL_AXIS_LONG_ALIGN_MIN_DOT
    )
    if needs_prealign:
        if float(np.dot(shortest_axis, normal)) < 0:
            shortest_axis = -shortest_axis
        rot_ax = np.cross(shortest_axis, normal)
        norm = float(np.linalg.norm(rot_ax))
        if norm > _PRINCIPAL_AXIS_ROT_AXIS_MIN_NORM:
            rot_ax /= norm
            angle_deg = 0.5 * np.degrees(
                np.arccos(np.clip(float(np.dot(shortest_axis, normal)), -1, 1))
            )
            R_pre = _rotation_around_axis(rot_ax, angle_deg)
            pos = (R_pre @ (pos - com).T).T + com
            # Axes change after the pre-align rotation.
            _, principal_axes = _compute_inertia_tensor(pos)
            com = np.mean(pos, axis=0)

    best_score = float("-inf")
    best_positions: np.ndarray | None = None

    for ax_idx in range(3):
        axis = principal_axes[:, ax_idx].copy()
        if float(np.dot(axis, normal)) < 0:
            axis = -axis
        for step in range(_PRINCIPAL_AXIS_ROTATION_STEPS):
            R = _rotation_around_axis(axis, step * _PRINCIPAL_AXIS_ROTATION_STEP_DEG)
            test = (R @ (pos - com).T).T + com
            # score by clearance along the surface normal (higher = more clearance)
            clearance = float(np.min(test @ normal))
            if clearance > best_score:
                best_score = clearance
                best_positions = test.copy()

    # re-centre at origin so caller controls the final offset
    if best_positions is not None:
        best_positions -= np.mean(best_positions, axis=0)

    return best_positions, best_score


def _mol_slab_contact_arrays(
    molecule_atoms: Atoms,
    slab: Atoms,
    *,
    material_type: str = "slab",
    exclude_slab_atoms: int | None = None,
    pairwise_distances: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray, list[bool], np.ndarray]:
    """Slice mol/slab positions and symbols; return MIC pairwise distances."""
    mol_syms = list(molecule_atoms.get_chemical_symbols())
    mol_pos = molecule_atoms.get_positions()
    slab_syms = list(slab.get_chemical_symbols())
    if exclude_slab_atoms is not None:
        slab_pos = slab.get_positions()[:exclude_slab_atoms]
        slab_syms = slab_syms[:exclude_slab_atoms]
    else:
        slab_pos = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(material_type)
    dists = (
        pairwise_distances
        if pairwise_distances is not None
        else _mol_slab_pairwise_distances(mol_pos, slab_pos, cell, pbc)
    )
    return mol_pos, slab_pos, mol_syms, slab_syms, cell, pbc, dists


def detect_vdw_overlaps(
    molecule_atoms: Atoms,
    slab: Atoms,
    *,
    material_type: str = "slab",
    vdw_scale: float = 1.0,
    exclude_slab_atoms: int | None = None,
    pairwise_distances: np.ndarray | None = None,
) -> tuple[list[tuple[int, int, float, float]], float]:
    """Detect VDW overlaps between molecule and slab atoms.

    Returns (overlaps, min_distance) where overlaps entries are
    (mol_idx, slab_idx, distance, overlap_amount).

    When *pairwise_distances* is provided (shape ``(n_mol, n_slab)``), skip
    recomputing MIC distances via :func:`_mol_slab_pairwise_distances`.
    """
    _mol_pos, _slab_pos, mol_syms, slab_syms, _cell, _pbc, dists = (
        _mol_slab_contact_arrays(
            molecule_atoms,
            slab,
            material_type=material_type,
            exclude_slab_atoms=exclude_slab_atoms,
            pairwise_distances=pairwise_distances,
        )
    )
    min_distance = float(np.min(dists)) if dists.size else float("inf")

    if dists.size == 0:
        return [], min_distance

    # Unknown radii become NaN so they propagate to a NaN vdw_sum; the
    # subsequent ``> 0`` comparison is False for NaN, excluding those pairs.
    mol_radii = np.array(
        [r if (r := _get_vdw_radius(s)) is not None else np.nan for s in mol_syms],
        dtype=float,
    )
    slab_radii = np.array(
        [r if (r := _get_vdw_radius(s)) is not None else np.nan for s in slab_syms],
        dtype=float,
    )
    vdw_sum = vdw_scale * (mol_radii[:, None] + slab_radii[None, :])
    overlap_amount = vdw_sum - dists
    with np.errstate(invalid="ignore"):
        mask = overlap_amount > 0.0
    rows, cols = np.nonzero(mask)
    overlaps: list[tuple[int, int, float, float]] = [
        (int(i), int(j), float(dists[i, j]), float(overlap_amount[i, j]))
        for i, j in zip(rows, cols, strict=False)
    ]
    return overlaps, min_distance


def calculate_contact_quality(
    molecule_atoms: Atoms,
    slab: Atoms,
    contact_distance_threshold: float | None = None,
    exclude_slab_atoms: int | None = None,
    *,
    material_type: str = "slab",
) -> dict[str, float | int]:
    """Contact metrics: min distance, covalent ratio at closest pair, and pair counts."""
    _mol_pos, _slab_pos, mol_syms, slab_syms, _cell, _pbc, dists = (
        _mol_slab_contact_arrays(
            molecule_atoms,
            slab,
            material_type=material_type,
            exclude_slab_atoms=exclude_slab_atoms,
        )
    )
    mol_size, slab_size = dists.shape
    if mol_size == 0 or slab_size == 0:
        return {
            "contact_distance": float("inf"),
            "contact_ratio": 1.0,
            "num_contacting_atoms": 0,
            "num_contact_pairs": 0,
            "contact_atom_variance": 0.0,
        }

    flat_idx = int(np.argmin(dists.ravel()))
    i_closest, j_closest = divmod(flat_idx, slab_size)
    contact_distance = float(dists[i_closest, j_closest])
    r1 = _get_covalent_radius(mol_syms[i_closest])
    r2 = _get_covalent_radius(slab_syms[j_closest])
    contact_ratio = (
        contact_distance / (r1 + r2) if (r1 is not None and r2 is not None) else 1.0
    )

    if contact_distance_threshold is None:
        if r1 is not None and r2 is not None:
            contact_distance_threshold = _CONTACT_QUALITY_COVALENT_SUM_SCALE * (r1 + r2)
        else:
            contact_distance_threshold = _MIN_DISTANCE_HARD_FALLBACK_ANGSTROM
    mask = dists <= contact_distance_threshold
    contact_pairs = int(np.count_nonzero(mask))
    contacting_atoms = {int(i) for i in np.where(np.any(mask, axis=1))[0]}
    contact_distances = dists[mask].ravel().tolist() if contact_pairs else []
    contact_atom_variance = (
        float(np.var(contact_distances)) if contact_distances else 0.0
    )

    return {
        "contact_distance": contact_distance,
        "contact_ratio": contact_ratio,
        "num_contacting_atoms": len(contacting_atoms),
        "num_contact_pairs": contact_pairs,
        "contact_atom_variance": contact_atom_variance,
    }


def calculate_min_distance(
    positions1: np.ndarray,
    positions2: np.ndarray,
    cell: np.ndarray | None = None,
    use_pbc: bool = True,
    pbc: list[bool] | None = None,
) -> float:
    """Minimum interatomic distance between two position arrays.

    Uses ASE's :func:`~ase.geometry.find_mic` for minimum-image distances
    with non-orthogonal cells via :func:`_mol_slab_pairwise_distances`.

    When *cell* is periodic (``abs(det) > 0``), *pbc* must be provided explicitly so
    slab ([True, True, False]), nanoparticle ([False, False, False]), and
    porous ([True, True, True]) calculations cannot accidentally fall back to
    incorrect full-3D periodicity.
    """
    p1 = np.asarray(positions1)
    p2 = np.asarray(positions2)
    if use_pbc and cell is not None and _cell_has_volume(cell):
        if pbc is None:
            raise ValueError(
                "pbc must be provided when cell is periodic; "
                "pass slab/cluster/porous flags explicitly"
            )
        cell_arr = np.asarray(cell, dtype=float)
        pbc_list = list(pbc)
    else:
        cell_arr = np.eye(3)
        pbc_list = [False, False, False]
    dists = _mol_slab_pairwise_distances(p1, p2, cell_arr, pbc_list)
    return float(np.min(dists))


def check_initial_placement_distance(
    molecule_atoms: Atoms,
    slab: Atoms,
    min_distance: float = _MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
    min_contact_ratio: float = _MIN_CONTACT_RATIO_DEFAULT,
    max_initial_distance: float | None = None,
    reject_vdw_overlaps: bool = False,
    vdw_overlap_scale: float = 1.0,
    exclude_slab_atoms: int | None = None,
    *,
    material_type: str = "slab",
) -> tuple[bool, float, str | None]:
    """Check if the initial placement satisfies distance constraints.

    Lower bound uses ``max(min_distance, covalent_sum * min_contact_ratio)``
    for the closest pair when radii are known.
    Optional upper bound and vdw-overlap checks when configured.
    ``exclude_slab_atoms`` limits the slab side (saturation with pre-adsorbed atoms).
    PBC comes from *material_type* via :func:`material_aware_pbc`.

    Returns ``(ok, actual_min_distance, reason)`` with ``reason`` one of
    ``None``, ``\"too_close\"``, ``\"too_far\"``, ``\"vdw_overlap\"``,
    ``\"empty_geometry\"``.
    """
    _mol_pos, _slab_pos, mol_syms, slab_syms, _cell, _pbc, dists = (
        _mol_slab_contact_arrays(
            molecule_atoms,
            slab,
            material_type=material_type,
            exclude_slab_atoms=exclude_slab_atoms,
        )
    )
    if dists.size == 0 or dists.shape[0] == 0 or dists.shape[1] == 0:
        return False, float("inf"), "empty_geometry"

    flat_idx = int(np.argmin(dists.ravel()))
    mol_idx, slab_idx = divmod(flat_idx, dists.shape[1])
    actual_min = float(dists[mol_idx, slab_idx])

    r1 = _get_covalent_radius(mol_syms[mol_idx])
    r2 = _get_covalent_radius(slab_syms[slab_idx])
    if r1 is not None and r2 is not None:
        min_allowed = max(float(min_distance), (r1 + r2) * float(min_contact_ratio))
    else:
        min_allowed = max(
            float(min_distance),
            _MIN_DISTANCE_COVALENT_FALLBACK_SCALE
            * _MIN_DISTANCE_HARD_FALLBACK_ANGSTROM,
        )
        logger.debug(
            "Unknown covalent radius for %s or %s; using conservative min distance %.2f A",
            mol_syms[mol_idx],
            slab_syms[slab_idx],
            min_allowed,
        )

    if actual_min < min_allowed:
        return False, actual_min, "too_close"
    if max_initial_distance is not None and actual_min > max_initial_distance:
        return False, actual_min, "too_far"

    if reject_vdw_overlaps:
        overlaps, _ = detect_vdw_overlaps(
            molecule_atoms,
            slab,
            material_type=material_type,
            vdw_scale=vdw_overlap_scale,
            exclude_slab_atoms=exclude_slab_atoms,
            pairwise_distances=dists,
        )
        if overlaps:
            logger.debug(
                "VDW overlap detected: %d overlapping atom pairs, max overlap %.3f A",
                len(overlaps),
                max(o[3] for o in overlaps),
            )
            return False, actual_min, "vdw_overlap"

    return True, actual_min, None


def check_adsorbate_separation(
    new_adsorbate: Atoms,
    pre_adsorbed_positions: np.ndarray,
    min_separation: float | None = None,
    cell: np.ndarray | None = None,
    pbc: list[bool] | None = None,
) -> tuple[bool, float]:
    """Check separation between new adsorbate and pre-adsorbed atoms.

    Used in saturation mode where slab already contains previously placed
    adsorbates. Ensures new placements don't collide with existing ones.

    Args:
        new_adsorbate: Atoms object representing new molecule to place
        pre_adsorbed_positions: (N, 3) array of pre-adsorbed atom positions
        min_separation: minimum allowed distance (Å) between atoms
        cell: unit cell (required if pbc is used)
        pbc: periodic boundary conditions [x, y, z]

    Returns:
        (ok, min_distance) where ok=True if separation is adequate
    """
    if len(pre_adsorbed_positions) == 0:
        return True, float("inf")

    new_pos = new_adsorbate.get_positions()
    pbc_requested = pbc is not None and any(pbc)
    if pbc_requested:
        if cell is None or not _cell_has_volume(cell):
            raise ValueError(
                "cell with non-zero volume must be provided when pbc is requested; "
                "pass slab/cluster/porous cell explicitly"
            )
        cell_arr = np.asarray(cell, dtype=float)
        dmat = _mol_slab_pairwise_distances(
            new_pos, pre_adsorbed_positions, cell_arr, list(pbc)
        )
        min_dist = float(np.min(dmat)) if dmat.size else float("inf")
    elif cell is not None and _cell_has_volume(cell) and pbc is not None:
        # Explicit all-False pbc with a cell: still use the pairwise helper.
        cell_arr = np.asarray(cell, dtype=float)
        dmat = _mol_slab_pairwise_distances(
            new_pos, pre_adsorbed_positions, cell_arr, list(pbc)
        )
        min_dist = float(np.min(dmat)) if dmat.size else float("inf")
    else:
        diffs = new_pos[:, None, :] - pre_adsorbed_positions[None, :, :]
        min_dist = float(np.min(np.linalg.norm(diffs, axis=2)))

    if min_separation is None:
        new_syms = new_adsorbate.get_chemical_symbols()
        new_r = [_get_covalent_radius(s) for s in new_syms]
        valid_new = [r for r in new_r if r is not None]
        ref_radius = (
            float(np.mean(valid_new))
            if valid_new
            else _MIN_DISTANCE_HARD_FALLBACK_ANGSTROM / 2.0
        )
        min_separation = _ADSORBATE_SEPARATION_COVALENT_SUM_SCALE * (2.0 * ref_radius)
    ok = min_dist >= min_separation
    return ok, min_dist


def check_initial_contact_quality(
    molecule_atoms: Atoms,
    slab: Atoms,
    *,
    strict_initial_placement: bool = False,
    require_multiple_contact: bool = False,
    max_closest_approach: float = _CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM,
    min_contact_atoms: int = 1,
    contact_distance_threshold: float = _CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM,
    exclude_slab_atoms: int | None = None,
    material_type: str = "slab",
) -> tuple[bool, str]:
    """Contact-quality gate for initial placements; returns (ok, reason_token)."""
    if not strict_initial_placement and not require_multiple_contact:
        return True, "strict_placement_checks_disabled"

    if len(molecule_atoms) < 1:
        return False, "empty_adsorbate"

    metrics = calculate_contact_quality(
        molecule_atoms,
        slab,
        contact_distance_threshold=contact_distance_threshold,
        exclude_slab_atoms=exclude_slab_atoms,
        material_type=material_type,
    )

    contact_dist = float(metrics["contact_distance"])
    num_contacting = int(metrics["num_contacting_atoms"])

    if contact_dist > max_closest_approach:
        return False, "contact_distance_too_large"

    min_contacts = int(min_contact_atoms)
    if require_multiple_contact:
        min_contacts = max(2, min_contacts)

    if num_contacting < min_contacts:
        return False, "insufficient_contact_atoms"

    if require_multiple_contact and num_contacting > 1:
        contact_atom_var = float(metrics["contact_atom_variance"])
        if contact_atom_var > _CONTACT_ATOM_VARIANCE_MAX:
            return False, "contact_distance_variance_too_high"

    return True, "placement_geometry_valid"

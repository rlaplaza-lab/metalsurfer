"""Rotation, inertia, and distance helpers for adsorbate placement."""

import functools
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii
from ase.geometry import find_mic

from .._utils import cell_has_volume
from ._constants import (
    _ADSORBATE_SEPARATION_COVALENT_SUM_SCALE,
    _ADSORBATE_SEPARATION_MIN_HARD_FLOOR_ANGSTROM,
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


@dataclass(frozen=True)
class _SlabDistanceScratch:
    """Invariant slab-side slice reused across candidate validations.

    Holds the pre-sliced slab positions, symbols, cell, and PBC so that
    repeated distance checks during height/XY recovery do not re-slice the ASE
    ``Atoms`` object or recompute the slab-side MIC geometry. *pre_ads_pos* is
    the slab slice *excluded* from the mol↔slab contact check (used by the
    separate adsorbate-separation check during saturation).
    """

    slab_pos: np.ndarray
    cell: np.ndarray
    pbc: list[bool]
    slab_syms: list[str]
    slab_cov_r: np.ndarray | None = None
    pre_ads_pos: np.ndarray | None = None


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Return normalized quaternion [w, x, y, z] with canonical sign.

    Antipodal quaternions ``q`` and ``-q`` represent the same rotation. After
    unit-norm, the sign is chosen so ``w > 0``, or when ``w == 0`` so the first
    non-zero of ``(x, y, z)`` is positive (lexicographic). Signed zeros are
    collapsed so the canonical form is hash-stable.

    Parameters
    ----------
    quat
        Input quaternion array (4-element).
    """
    q: np.ndarray = np.asarray(quat, dtype=float).reshape(4)
    nrm = float(np.linalg.norm(q))
    if nrm < _QUATERNION_NORM_EPS:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = np.asarray(q / nrm, dtype=float)
    if q[0] < 0.0 or (q[0] == 0.0 and (q[1], q[2], q[3]) < (0.0, 0.0, 0.0)):
        q = -q
    return np.where(q == 0.0, 0.0, q)


def normalize_quaternions(quats: np.ndarray) -> np.ndarray:
    """Normalize a batch of quaternions to unit length with canonical sign.

    Parameters
    ----------
    quats
        Array of shape ``(n, 4)`` with rows ``[w, x, y, z]``.
    """
    arr = np.asarray(quats, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(f"quats must have shape (n, 4), got {arr.shape}")
    if arr.shape[0] == 0:
        return arr.copy()
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = normalize_quaternion(arr[i])
    return out


def quaternion_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to a 3x3 rotation matrix.

    Parameters
    ----------
    quat
        Input quaternion array (4-element).
    """
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
    """Convert rotation matrix to quaternion [w, x, y, z].

    Parameters
    ----------
    R
        3×3 rotation matrix.
    """
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


def compute_canonical_molecular_frame(
    ads_pos: np.ndarray, symbols: list[str] | None = None
) -> np.ndarray:
    """Return centred positions in a deterministic principal-axis frame.

    Parameters
    ----------
    ads_pos
        Adsorbate positions (n, 3).
    symbols
        Optional chemical symbols for tie-breaking.
    """
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
    """Return deterministic orthonormal frame whose z-axis is surface normal.

    Parameters
    ----------
    normal
        Surface normal vector (3-element).
    """
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
    if cell_has_volume(cell) and np.any(pbc):
        diffs_flat = diffs.reshape(-1, 3)
        _, mic_dists = find_mic(diffs_flat, cell, pbc=pbc)
        return mic_dists.reshape(m, s)
    return np.linalg.norm(diffs, axis=2)


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
    """Return indices of atoms likely to bind (O, N, S, halogens)."""
    binders = {"O", "N", "S", "F", "Cl", "Br", "I"}
    return [i for i, s in enumerate(symbols) if s in binders]


def _compute_inertia_tensor(
    positions: np.ndarray,
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
    masses = np.ones(n)
    com = np.average(pos, axis=0, weights=masses)
    r = pos - com  # (n, 3)
    m = masses[:, None]  # (n, 1)
    # I = sum_k m_k * (|r_k|^2 * I_3 - r_k r_k^T)
    r2 = np.einsum("ij,ij->i", r, r)  # |r_k|^2
    outer = r[:, :, None] * r[:, None, :]  # (n, 3, 3)
    inertia = np.einsum("i->", m[:, 0] * r2) * np.eye(3) - np.einsum(
        "i,ijk->jk", m[:, 0], outer
    )
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
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate adsorbate so its molecular plane is parallel to the surface.

    Aligns the plane normal (axis of largest inertia for flat molecules;
    perpendicular axis theorem: I3 = I1 + I2 for planar bodies) with the
    surface normal. Returns ``(centred_positions, R)`` where *R* maps the
    centred input onto *centred_positions* (used to compose a full rotation
    instead of fitting it with Kabsch).

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

    R_align = _rotation_to_align_vector_to_target(plane_normal, normal)
    pos = (R_align @ pos.T).T

    R_az = _rotation_around_axis(normal, azimuth_in_plane_deg)
    pos = (R_az @ pos.T).T
    R_total = R_az @ R_align
    return pos, R_total


def _surface_aligned_rotation(
    ads_pos: np.ndarray,
    normal: np.ndarray,
    symbols: list[str] | None = None,
    en_binder_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate adsorbate so a binding vector points toward surface. Returns ``(centred_positions, R)``.

    When *en_binder_index* is provided and valid, use that index into the filtered
    electronegative-atom list from :func:`_binding_atom_candidates`, not a raw
    atom index.  Otherwise select the binder with highest dot product toward the
    surface normal.

    *R* maps the centred input onto *centred_positions*.
    """
    binder_idx = en_binder_index
    pos = np.asarray(ads_pos, dtype=float).copy()
    com = np.mean(pos, axis=0)
    pos -= com
    normal = np.asarray(normal, dtype=float) / (
        np.linalg.norm(normal) + _VECTOR_NORM_EPS
    )

    R_total = np.eye(3)
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
            if dot > -_BINDER_ALIGNMENT_TARGET_DOT:
                R = _rotation_to_align_vector_to_target(-best_vec, normal)
                pos = (R @ pos.T).T
                R_total = R
    else:
        rotated, _, best_R = _principal_axis_rotation(pos, normal)
        if rotated is not None:
            pos = rotated
            R_total = best_R
        else:
            pos = np.asarray(ads_pos, dtype=float).copy() - com
            R_total = np.eye(3)
    return pos, R_total


def _rotation_with_tilt(
    pos: np.ndarray, normal: np.ndarray, tilt_deg: float, azimuth_deg: float
) -> tuple[np.ndarray, np.ndarray]:
    """Apply tilt and azimuth to positions (centred at origin). Returns ``(rotated_positions, R)``.

    *R* is the full rotation matrix ``frame @ R_az @ R_tilt @ frame.T`` mapping
    the centred input onto *rotated_positions*.
    """
    frame = compute_surface_site_frame(normal)
    pos_local = (frame.T @ np.asarray(pos, dtype=float).T).T
    R_tilt = _rotation_around_axis(np.array([1.0, 0.0, 0.0]), tilt_deg)
    R_az = _rotation_around_axis(np.array([0.0, 0.0, 1.0]), azimuth_deg)
    pos_local = (R_az @ R_tilt @ pos_local.T).T
    rotated = (frame @ pos_local.T).T
    R_total = frame @ R_az @ R_tilt @ frame.T
    return rotated, R_total


def _principal_axis_rotation(
    adsorbate_positions: np.ndarray,
    normal_vector: np.ndarray,
) -> tuple[np.ndarray | None, float, np.ndarray]:
    """Rotate the adsorbate around its principal axes to maximise clearance.

    Returns centred-at-origin positions (no surface_z offset), the best
    min-z score, and *best_R*, the rotation matrix mapping the centred input
    onto the returned positions (used to compose a full rotation).
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
    R_total = np.eye(3)
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
            R_total = R_pre
            # Axes change after the pre-align rotation.
            _, principal_axes = _compute_inertia_tensor(pos)
            com = np.mean(pos, axis=0)

    best_score = float("-inf")
    best_positions: np.ndarray | None = None
    best_R = np.eye(3)

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
                best_R = R @ R_total

    # re-centre at origin so caller controls the final offset
    if best_positions is not None:
        best_positions -= np.mean(best_positions, axis=0)

    return best_positions, best_score, best_R


def _mol_slab_contact_arrays(
    molecule_atoms: Atoms,
    slab: Atoms,
    *,
    material_type: str = "slab",
    exclude_slab_atoms: int | None = None,
    pairwise_distances: np.ndarray | None = None,
    slab_scratch: _SlabDistanceScratch | None = None,
) -> tuple[
    np.ndarray, np.ndarray, list[str], list[str], np.ndarray, list[bool], np.ndarray
]:
    """Slice mol/slab positions and symbols; return MIC pairwise distances.

    When *slab_scratch* is provided, the slab side (positions, symbols, cell,
    pbc) is reused from the scratch instead of re-slicing the ASE ``Atoms``
    object.  This keeps the invariant slab slice fixed across distance-recovery
    candidate validations.
    """
    mol_syms = list(molecule_atoms.get_chemical_symbols())
    mol_pos = molecule_atoms.get_positions()
    if slab_scratch is not None:
        slab_pos = slab_scratch.slab_pos
        slab_syms = slab_scratch.slab_syms
        cell = slab_scratch.cell
        pbc = slab_scratch.pbc
    else:
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

    Parameters
    ----------
    molecule_atoms
        Adsorbate :class:`~ase.Atoms` object.
    slab
        Substrate :class:`~ase.Atoms` object.
    material_type
        Material type for PBC flags.
    vdw_scale
        Scale factor for VDW radii.
    exclude_slab_atoms
        Number of substrate atoms to exclude from checks.
    pairwise_distances
        Optional precomputed pairwise distance matrix.
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
    pairwise_distances: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Contact metrics: min distance, covalent ratio at closest pair, and pair counts.

    Parameters
    ----------
    molecule_atoms
        Adsorbate :class:`~ase.Atoms` object.
    slab
        Substrate :class:`~ase.Atoms` object.
    contact_distance_threshold
        Optional distance threshold for contact counting.
    exclude_slab_atoms
        Number of substrate atoms to exclude from checks.
    material_type
        Material type for PBC flags.
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

    Parameters
    ----------
    positions1
        First set of positions (n, 3).
    positions2
        Second set of positions (m, 3).
    cell
        Optional unit cell matrix.
    use_pbc
        Whether to use periodic boundary conditions.
    pbc
        Optional periodic boundary condition flags [x, y, z].
    """
    p1 = np.asarray(positions1)
    p2 = np.asarray(positions2)
    if use_pbc and cell is not None and cell_has_volume(cell):
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
    pairwise_distances: np.ndarray | None = None,
    slab_scratch: _SlabDistanceScratch | None = None,
) -> tuple[bool, float, str | None]:
    r"""Check if the initial placement satisfies distance constraints.

    Lower bound uses ``max(min_distance, covalent_sum * min_contact_ratio)``
    for the closest pair when radii are known.
    Optional upper bound and vdw-overlap checks when configured.
    ``exclude_slab_atoms`` limits the slab side (saturation with pre-adsorbed atoms).
    PBC comes from *material_type* via :func:`material_aware_pbc`.

    Returns ``(ok, actual_min_distance, reason)`` with ``reason`` one of
    ``None``, ``\"too_close\"``, ``\"too_far\"``, ``\"vdw_overlap\"``,
    ``\"empty_geometry\"``.

    Parameters
    ----------
    molecule_atoms
        Adsorbate :class:`~ase.Atoms` object.
    slab
        Substrate :class:`~ase.Atoms` object.
    min_distance
        Minimum allowed distance (Å).
    min_contact_ratio
        Minimum contact ratio relative to covalent radii.
    max_initial_distance
        Optional maximum allowed distance (Å).
    reject_vdw_overlaps
        Whether to reject VDW overlaps.
    vdw_overlap_scale
        Scale factor for VDW overlap detection.
    exclude_slab_atoms
        Number of substrate atoms to exclude from checks.
    material_type
        Material type for PBC flags.
    """
    _mol_pos, _slab_pos, mol_syms, slab_syms, _cell, _pbc, dists = (
        _mol_slab_contact_arrays(
            molecule_atoms,
            slab,
            material_type=material_type,
            exclude_slab_atoms=exclude_slab_atoms,
            pairwise_distances=pairwise_distances,
            slab_scratch=slab_scratch,
        )
    )
    if dists.size == 0 or dists.shape[0] == 0 or dists.shape[1] == 0:
        return False, float("inf"), "empty_geometry"

    actual_min = float(np.min(dists))

    mol_r = np.array(
        [r if (r := _get_covalent_radius(s)) is not None else np.nan for s in mol_syms],
        dtype=float,
    )
    # Reuse precomputed slab covalent radii from the scratch when available
    # (otherwise recompute, as before).
    if slab_scratch is not None and slab_scratch.slab_cov_r is not None:
        slab_r = np.asarray(slab_scratch.slab_cov_r, dtype=float)
    else:
        slab_r = np.array(
            [
                r if (r := _get_covalent_radius(s)) is not None else np.nan
                for s in slab_syms
            ],
            dtype=float,
        )
    allowed = (mol_r[:, None] + slab_r[None, :]) * float(min_contact_ratio)
    np.maximum(allowed, float(min_distance), out=allowed)
    # Unknown radii: fall back to flat min_distance for that pair.
    np.nan_to_num(allowed, nan=float(min_distance), copy=False)
    if np.any(dists < allowed):
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

    Parameters
    ----------
    new_adsorbate
        Atoms object representing new molecule to place.
    pre_adsorbed_positions
        (N, 3) array of pre-adsorbed atom positions.
    min_separation
        Minimum allowed distance (\u00c5) between atoms.
    cell
        Unit cell (required if pbc is used).
    pbc
        Periodic boundary conditions [x, y, z]. When *cell* has non-zero volume
        (a periodic substrate), *pbc* must be provided explicitly; passing
        ``pbc=None`` with a volumed cell raises :class:`ValueError`.

    Returns
    -------
    tuple[bool, float]
        (ok, min_distance) where ok is True if separation is adequate.
    """
    if len(pre_adsorbed_positions) == 0:
        return True, float("inf")

    if pbc is None and cell is not None and cell_has_volume(cell):
        raise ValueError(
            "pbc must be provided when cell is periodic; "
            "pass slab/cluster/porous flags explicitly"
        )

    new_pos = new_adsorbate.get_positions()
    if (
        pbc is not None
        and any(pbc)
        and not (cell is not None and cell_has_volume(cell))
    ):
        raise ValueError(
            "cell with non-zero volume must be provided when pbc is requested; "
            "pass slab/cluster/porous cell explicitly"
        )
    if pbc is not None and cell is not None and cell_has_volume(cell):
        cell_arr, pbc_list = np.asarray(cell, dtype=float), list(pbc)
    else:
        cell_arr, pbc_list = np.eye(3), [False, False, False]
    dmat = _mol_slab_pairwise_distances(
        new_pos, pre_adsorbed_positions, cell_arr, pbc_list
    )
    min_dist = float(np.min(dmat)) if dmat.size else float("inf")

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
    # When pre-adsorbed molecules are present (saturation), enforce a physically
    # meaningful floor so two molecules 1.5 A H..H apart (effectively bonded) are
    # rejected. With none present the caller is checking adsorbate vs substrate,
    # which is governed by check_initial_placement_distance instead.
    if len(pre_adsorbed_positions) > 0:
        min_separation = max(
            float(min_separation), _ADSORBATE_SEPARATION_MIN_HARD_FLOOR_ANGSTROM
        )
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
    pairwise_distances: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Contact-quality gate for initial placements; returns (ok, reason_token).

    Parameters
    ----------
    molecule_atoms
        Adsorbate :class:`~ase.Atoms` object.
    slab
        Substrate :class:`~ase.Atoms` object.
    strict_initial_placement
        Whether to enforce strict placement checks.
    require_multiple_contact
        Whether to require multiple contacting atoms.
    max_closest_approach
        Maximum allowed closest approach distance (Å).
    min_contact_atoms
        Minimum number of atoms that must make contact.
    contact_distance_threshold
        Distance threshold for contact counting (Å).
    exclude_slab_atoms
        Number of substrate atoms to exclude from checks.
    material_type
        Material type for PBC flags.
    """
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
        pairwise_distances=pairwise_distances,
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

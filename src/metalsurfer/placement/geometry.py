"""Rotation, inertia, and distance helpers for adsorbate placement."""

import logging
import random
from typing import Literal

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.data import covalent_radii as ase_covalent_radii
from ase.geometry import find_mic

from ._material import material_aware_pbc  # noqa: F401 – re-exported

logger = logging.getLogger(__name__)


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    """Return normalized quaternion [w, x, y, z] with canonical sign."""
    q = np.asarray(quat, dtype=float).reshape(4)
    nrm = float(np.linalg.norm(q))
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if nrm < 1e-12 else q / nrm
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
            abs(projections[idx]) < 1e-10
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
    return v / nrm if nrm > 1e-12 else np.zeros_like(v)


def compute_surface_site_frame(normal: np.ndarray) -> np.ndarray:
    """Return deterministic orthonormal frame whose z-axis is surface normal."""
    z_axis = _safe_normalize(np.asarray(normal, dtype=float))
    if z_axis[2] < 0:
        z_axis = -z_axis
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(np.dot(ref, z_axis)) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _safe_normalize(ref - np.dot(ref, z_axis) * z_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


def _get_covalent_radius(symbol: str) -> float | None:
    try:
        z = atomic_numbers[symbol]
        if z < len(ase_covalent_radii):
            r = float(ase_covalent_radii[z])
            if r > 0.0:
                return r
    except (KeyError, IndexError):
        pass
    logger.debug("No covalent radius for symbol %s", symbol)
    return None


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
    ax = np.asarray(axis, dtype=float) / (np.linalg.norm(axis) + 1e-12)
    return _rodrigues(ax, np.cos(a), np.sin(a))


def _rotation_to_align_vector_to_target(
    vec: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Rotation matrix that aligns vec (unit) to target (unit)."""
    vec = np.asarray(vec, dtype=float) / (np.linalg.norm(vec) + 1e-12)
    target = np.asarray(target, dtype=float) / (np.linalg.norm(target) + 1e-12)
    dot = np.clip(np.dot(vec, target), -1.0, 1.0)
    if dot > 0.9999:
        return np.eye(3)
    if dot < -0.9999:
        # Anti-parallel: 180° rotation around any perpendicular axis.
        axis = np.array([1, 0, 0]) if abs(vec[0]) < 0.9 else np.array([0, 1, 0])
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
            masses = np.ones(n)
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
    eps = 1e-8
    I3_safe = max(I3, eps)
    I2_safe = max(I2, eps)

    if I1 / I3_safe < 0.02:
        return "linear", eigenvals, eigenvecs
    # Flat (oblate): I1 ≈ I2 << I3 (perpendicular axis theorem)
    if I1 / I3_safe < 0.55 and I2_safe / I3_safe > 0.45:
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
    normal = np.asarray(normal, dtype=float) / (np.linalg.norm(normal) + 1e-12)
    if normal[2] < 0:
        normal = -normal

    _, eigenvecs = _compute_inertia_tensor(pos)
    plane_normal = eigenvecs[:, 2]
    if np.dot(plane_normal, normal) < 0:
        plane_normal = -plane_normal
    if face_flip:
        plane_normal = -plane_normal

    R = _rotation_to_align_vector_to_target(plane_normal, normal)
    pos = (R @ pos.T).T

    R_az = _rotation_around_axis(normal, azimuth_in_plane_deg)
    pos = (R_az @ pos.T).T

    return pos


def _surface_aligned_rotation(
    ads_pos: np.ndarray,
    normal: np.ndarray,
    placement_id: int,
    symbols: list[str] | None = None,
    en_atom_index: int | None = None,
) -> np.ndarray:
    """Rotate adsorbate so a binding vector points toward surface. Returns centred positions.

    When en_atom_index is provided and valid, use that specific electronegative atom.
    Otherwise select the one with highest dot product toward the surface normal.
    """
    pos = np.asarray(ads_pos, dtype=float).copy()
    com = np.mean(pos, axis=0)
    pos -= com
    normal = np.asarray(normal, dtype=float) / (np.linalg.norm(normal) + 1e-12)
    if normal[2] < 0:
        normal = -normal

    binders = _binding_atom_candidates(symbols) if symbols else []
    if binders:
        if en_atom_index is not None and en_atom_index in range(len(binders)):
            i = binders[en_atom_index]
        else:
            best_dot = -1.0
            i = binders[0]
            for idx in binders:
                # pos is already centered above; avoid subtracting COM again.
                v = pos[idx]
                nv = np.linalg.norm(v)
                if nv > 0.1:
                    dot = np.dot(v / nv, normal)
                    if dot > best_dot:
                        best_dot = dot
                        i = idx
        v = pos[i]
        nv = np.linalg.norm(v)
        if nv > 0.1:
            best_vec = v / nv
            dot = np.dot(best_vec, normal)
            if dot < 0.95:
                R = _rotation_to_align_vector_to_target(-best_vec, normal)
                pos = (R @ pos.T).T
    else:
        pos, _ = _principal_axis_rotation(pos, normal, placement_id)
    return pos


def _rotation_with_tilt(
    pos: np.ndarray, normal: np.ndarray, tilt_deg: float, azimuth_deg: float
) -> np.ndarray:
    """Apply tilt and azimuth to positions (centred at origin)."""
    normal = np.asarray(normal, dtype=float) / (np.linalg.norm(normal) + 1e-12)
    up = np.array([0, 0, 1.0])
    R_align = _rotation_to_align_vector_to_target(up, normal)
    pos = (R_align @ pos.T).T
    R_tilt = _rotation_around_axis(np.array([1, 0, 0]), tilt_deg)
    R_az = _rotation_around_axis(up, azimuth_deg)
    return (R_align.T @ R_az @ R_tilt @ R_align @ pos.T).T


def _principal_axis_rotation(
    adsorbate_positions: np.ndarray,
    normal_vector: np.ndarray,
    placement_id: int,
) -> tuple[np.ndarray, float]:
    """Rotate the adsorbate around its principal axes to maximise clearance.

    Returns centred-at-origin positions (no surface_z offset) and the best
    min-z score. Always returns a valid position array.
    """
    pos = np.asarray(adsorbate_positions, dtype=float).copy()
    com = np.mean(pos, axis=0)

    _, eigenvecs = _compute_inertia_tensor(pos)
    principal_axes = eigenvecs

    # gentle pre-alignment toward surface normal
    shortest_axis = principal_axes[:, 0]
    longest_axis = principal_axes[:, -1]
    if (
        abs(np.dot(shortest_axis, normal_vector)) < 0.7
        and abs(np.dot(longest_axis, normal_vector)) > 0.3
    ):
        if np.dot(shortest_axis, normal_vector) < 0:
            shortest_axis = -shortest_axis
        rot_ax = np.cross(shortest_axis, normal_vector)
        norm = np.linalg.norm(rot_ax)
        if norm > 1e-6:
            rot_ax /= norm
            angle_deg = 0.5 * np.degrees(
                np.arccos(np.clip(np.dot(shortest_axis, normal_vector), -1, 1))
            )
            R_pre = _rotation_around_axis(rot_ax, angle_deg)
            pos = (R_pre @ (pos - com).T).T + com

    best_score = float("-inf")
    best_positions: np.ndarray | None = None

    for ax_idx in range(3):
        axis = principal_axes[:, ax_idx].copy()
        if np.dot(axis, normal_vector) < 0:
            axis = -axis
        for step in range(36):
            R = _rotation_around_axis(axis, step * 10.0)
            test = (R @ (pos - com).T).T + com
            # score by min-z (higher = more clearance)
            min_z = float(np.min(test[:, 2]))
            if min_z > best_score:
                best_score = min_z
                best_positions = test.copy()

    # re-centre at origin so caller controls the final offset
    if best_positions is None:
        best_positions = np.asarray(adsorbate_positions, dtype=float).copy()
    best_positions -= np.mean(best_positions, axis=0)

    return best_positions, best_score


def calculate_min_distance(
    positions1: np.ndarray,
    positions2: np.ndarray,
    cell: np.ndarray | None = None,
    use_pbc: bool = True,
    pbc: list[bool] | None = None,
) -> float:
    """Minimum interatomic distance between two position arrays.

    Uses ASE's :func:`~ase.geometry.find_mic` for robust minimum-image
    convention handling, supporting non-orthogonal cells.  When *pbc* is
    not supplied, it is inferred from the cell determinant (full 3D PBC
    when det > 0, no PBC otherwise).  Callers should pass the actual
    structure PBC for correctness across slabs, nanoparticles, and porous
    materials.
    """
    if use_pbc and cell is not None and np.linalg.det(cell) > 0:
        if pbc is None:
            pbc = [True, True, True]
        p1 = np.asarray(positions1)
        p2 = np.asarray(positions2)
        diffs = p1[:, None, :] - p2[None, :, :]
        diffs_flat = diffs.reshape(-1, 3)
        mic_diffs, mic_dists = find_mic(diffs_flat, cell, pbc=pbc)
        return float(np.min(mic_dists))
    else:
        p1 = positions1.reshape(-1, 1, 3)
        p2 = positions2.reshape(1, -1, 3)
        return float(np.min(np.linalg.norm(p1 - p2, axis=2)))


def calculate_min_distance_pair(
    positions1: np.ndarray,
    positions2: np.ndarray,
    cell: np.ndarray | None = None,
    pbc: list[bool] | None = None,
) -> tuple[float, int, int]:
    """Like :func:`calculate_min_distance` but also returns the index pair.

    Returns ``(min_distance, idx1, idx2)`` where *idx1* indexes into
    *positions1* and *idx2* indexes into *positions2*.
    """
    if pbc is None:
        pbc = [True, True, True]
    p1 = np.asarray(positions1)
    p2 = np.asarray(positions2)
    if cell is not None and np.linalg.det(cell) > 0:
        diffs = p1[:, None, :] - p2[None, :, :]
        diffs_flat = diffs.reshape(-1, 3)
        _, mic_dists = find_mic(diffs_flat, cell, pbc=pbc)
        flat_idx = int(np.argmin(mic_dists))
    else:
        diffs = p1[:, None, :] - p2[None, :, :]
        dists = np.linalg.norm(diffs.reshape(-1, 3), axis=1)
        mic_dists = dists
        flat_idx = int(np.argmin(mic_dists))
    idx1, idx2 = divmod(flat_idx, len(p2))
    return float(mic_dists[flat_idx]), idx1, idx2


def check_initial_placement_distance(
    molecule_atoms: Atoms,
    slab: Atoms,
    min_distance: float = 1.5,
    min_contact_ratio: float = 0.8,
    max_initial_distance: float | None = None,
) -> tuple[bool, float]:
    """Check if the initial placement satisfies distance constraints.

    Lower bound: no atom may be within covalent binding distance. Uses
    (r_mol + r_surf) * min_contact_ratio for the closest atom pair, ensuring
    we avoid bond breaking or formation in standard cases.
    Upper bound: when max_initial_distance is set, reject placements too far
    (desorption-prone starts). Post-optimization, check_desorption uses
    binding_distance_threshold to reject structures that drifted too far.

    PBC flags are derived from the slab material type via
    :func:`material_aware_pbc` so that slabs, nanoparticles, and porous
    materials all receive correct periodic-image handling — consistent with
    the post-optimisation desorption check in :mod:`~metalsurfer.filters`.
    """
    mol_syms = molecule_atoms.get_chemical_symbols()
    slab_syms = slab.get_chemical_symbols()
    mol_pos = molecule_atoms.get_positions()
    slab_pos = slab.get_positions()
    cell = molecule_atoms.get_cell()
    pbc = material_aware_pbc(slab)

    actual_min, mol_idx, slab_idx = calculate_min_distance_pair(
        mol_pos, slab_pos, cell=cell, pbc=pbc
    )

    r1 = _get_covalent_radius(mol_syms[mol_idx])
    r2 = _get_covalent_radius(slab_syms[slab_idx])
    if r1 is not None and r2 is not None:
        min_allowed = (r1 + r2) * min_contact_ratio
    else:
        min_allowed = max(min_distance, 2.0)
        logger.debug(
            "Unknown covalent radius for %s or %s; using conservative min distance %.2f A",
            mol_syms[mol_idx],
            slab_syms[slab_idx],
            min_allowed,
        )

    if actual_min < min_allowed:
        return False, actual_min
    if max_initial_distance is not None and actual_min > max_initial_distance:
        return False, actual_min
    return True, actual_min

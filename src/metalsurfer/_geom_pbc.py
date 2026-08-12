"""Cell-frame geometry: fractional/Cartesian conversion, wrapping, MIC, slab frame.

Private, dependency-light module (numpy only, plus ``_numeric_defaults``): it is a
top-level sibling of :mod:`metalsurfer.symmetry` on purpose. Importing any
``metalsurfer.placement`` submodule executes ``placement/__init__.py``, which loads
``placement.site_enumeration`` and therefore ``metalsurfer.symmetry``; hosting these
helpers under ``placement`` would make a module-scope import from ``symmetry`` a
circular import. ``metalsurfer.placement.site_coords`` re-exports every name here
under its historical underscore alias.

All routines use the ASE row-vector cell convention: ``r_cart = r_frac @ cell``,
with ``cell[0]``/``cell[1]``/``cell[2]`` the lattice vectors a/b/c as rows.
"""

import numpy as np

from ._numeric_defaults import SURFACE_NORMAL_FALLBACK_NORM_EPS


def cart_to_frac(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert Cartesian row-vectors to fractional coordinates for ASE cells.

    Parameters
    ----------
    points
        Cartesian coordinates, shape (..., 3).
    cell
        3x3 cell matrix with lattice vectors as rows.
    """
    arr = np.asarray(points, dtype=float)
    inv_cell = np.linalg.inv(cell)
    return arr @ inv_cell


def frac_to_cart(points_frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert fractional row-vectors to Cartesian coordinates.

    Parameters
    ----------
    points_frac
        Fractional coordinates, shape (..., 3).
    cell
        3x3 cell matrix with lattice vectors as rows.
    """
    return np.asarray(points_frac, dtype=float) @ cell


def wrap_fractional(frac: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Wrap fractional coordinates to [0, 1) on periodic axes only.

    Parameters
    ----------
    frac
        Fractional coordinates.
    pbc
        Boolean periodic-boundary flags for each axis.
    """
    wrapped = np.asarray(frac, dtype=float).copy()
    for dim in range(3):
        if bool(pbc[dim]):
            wrapped[..., dim] -= np.floor(wrapped[..., dim])
    return wrapped


def wrap_cartesian(points: np.ndarray, cell: np.ndarray, pbc: np.ndarray) -> np.ndarray:
    """Wrap Cartesian points into the reference cell along periodic axes.

    Parameters
    ----------
    points
        Cartesian coordinates.
    cell
        3x3 cell matrix.
    pbc
        Boolean periodic-boundary flags for each axis.
    """
    if not np.any(pbc):
        return np.asarray(points, dtype=float).copy()
    frac = cart_to_frac(points, cell)
    return frac_to_cart(wrap_fractional(frac, pbc), cell)


def minimum_image_fractional_delta(
    delta_frac: np.ndarray, pbc: np.ndarray, *, copy: bool = True
) -> np.ndarray:
    """Apply the minimum-image convention to fractional coordinate differences.

    With ``copy=False`` the folding is done in place and the input buffer is
    returned. Only pass ``copy=False`` for a freshly allocated, caller-owned
    array; it exists so large ``n x n x 3`` intermediates are not duplicated.

    Parameters
    ----------
    delta_frac
        Fractional coordinate differences.
    pbc
        Boolean periodic-boundary flags for each axis.
    copy
        Whether to copy the input before modifying.
    """
    delta = np.asarray(delta_frac, dtype=float)
    if copy:
        delta = delta.copy()
    for dim in range(3):
        if bool(pbc[dim]):
            delta[..., dim] -= np.round(delta[..., dim])
    return delta


def reciprocal_plane_spacings(cell: np.ndarray) -> np.ndarray:
    """Distance between adjacent lattice planes normal to each cell vector.

    Parameters
    ----------
    cell
        3x3 cell matrix.
    """
    inv_cell = np.linalg.inv(cell)
    spacings = np.empty(3, dtype=float)
    for dim in range(3):
        g = inv_cell[:, dim]
        norm_g = float(np.linalg.norm(g))
        spacings[dim] = 1.0 / norm_g if norm_g > 0.0 else np.inf
    return spacings


def slab_plane_projectors(cell: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return projectors for slab-plane coordinates.

    Parameters
    ----------
    cell
        3x3 cell matrix.

    Returns
    -------
    (pinv_ab_T, ortho_basis)
        pinv_ab_T : (3, 2) array
            Right-multiplier for least-squares coordinates in the span of a/b.
            For Cartesian row vectors r, in-plane coordinates are r @ pinv_ab_T.
        ortho_basis : (2, 3) array
            Two orthonormal basis vectors spanning the same plane.
    """
    a = np.asarray(cell[0], dtype=float)
    b = np.asarray(cell[1], dtype=float)

    ab = np.column_stack([a, b])
    pinv_ab = np.linalg.pinv(ab)
    pinv_ab_T = pinv_ab.T

    norm_a = float(np.linalg.norm(a))
    if norm_a < SURFACE_NORMAL_FALLBACK_NORM_EPS:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1 = a / norm_a

    b_perp = b - np.dot(b, e1) * e1
    norm_b_perp = float(np.linalg.norm(b_perp))
    if norm_b_perp < SURFACE_NORMAL_FALLBACK_NORM_EPS:
        trial = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(trial, e1)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        b_perp = trial - np.dot(trial, e1) * e1
        norm_b_perp = float(np.linalg.norm(b_perp))
    e2 = b_perp / max(norm_b_perp, SURFACE_NORMAL_FALLBACK_NORM_EPS)
    ortho_basis = np.vstack([e1, e2])
    return pinv_ab_T, ortho_basis


def slab_normal(cell: np.ndarray) -> np.ndarray:
    """Return unit normal to the slab plane spanned by cell a and b.

    Parameters
    ----------
    cell
        3x3 cell matrix.
    """
    a = np.asarray(cell[0], dtype=float)
    b = np.asarray(cell[1], dtype=float)
    n = np.cross(a, b)
    norm_n = float(np.linalg.norm(n))
    if norm_n < SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.array([0.0, 0.0, 1.0])
    return n / norm_n


def height_along_slab_normal(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Signed coordinate of points along the slab normal.

    Parameters
    ----------
    points
        Cartesian coordinates.
    cell
        3x3 cell matrix.
    """
    n = slab_normal(cell)
    arr = np.asarray(points, dtype=float)
    return arr @ n


def shift_along_slab_normal(
    points: np.ndarray, cell: np.ndarray, distance: float
) -> np.ndarray:
    """Translate points by *distance* along the slab normal.

    Parameters
    ----------
    points
        Cartesian coordinates.
    cell
        3x3 cell matrix.
    distance
        Shift distance in Å.
    """
    n = slab_normal(cell)
    arr = np.asarray(points, dtype=float)
    return arr + float(distance) * n


def project_to_slab_plane(points: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Project Cartesian points to a 2D orthonormal basis spanning a/b.

    Parameters
    ----------
    points
        Cartesian coordinates.
    cell
        3x3 cell matrix.
    """
    _, ortho_basis = slab_plane_projectors(cell)
    arr = np.asarray(points, dtype=float)
    return arr @ ortho_basis.T

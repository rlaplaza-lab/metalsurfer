"""Aromatic heuristics, parallel fraction, and adsorbate orientation classification."""


from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..exceptions import DependencyMissingError
from ..models import PlacementSpec
from . import geometry as geom
from ._constants import (
    _MOL_COVALENT_RADIUS_FALLBACK,
    _ORIENTATION_CLASSIFICATION_PARALLEL_DOT_THRESHOLD,
    _PARALLEL_FRACTION_HIGH_BINDER_RATIO,
    _PARALLEL_FRACTION_HIGH_RATIO_CUTOFF,
    _PARALLEL_FRACTION_LOW_BINDER_RATIO,
    _PARALLEL_FRACTION_MEDIUM_BINDER_RATIO,
    _PARALLEL_FRACTION_MEDIUM_RATIO_CUTOFF,
    _PARALLEL_FRACTION_NO_BINDERS,
    _PARALLEL_FRACTION_NO_RING,
    _PARALLEL_FRACTION_SINGLE_BINDER,
    _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
    _PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE,
    _PARALLEL_Z_HI_SHRINK_FALLBACK_ANGSTROM,
    _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE,
    _PARALLEL_Z_LO_SHRINK_FALLBACK_ANGSTROM,
    _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE,
    _SITE_Z_OFFSET_FROM_SURFACE_RADIUS,
    _VECTOR_NORM_EPS,
)
from .site_coords import _slab_normal
from .site_enumeration import _get_site_surface_radii
from .site_types import Site


def _rdkit_chem():
    """Return ``rdkit.Chem`` (lazy import for dependency_behavior tests)."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyMissingError(
            "rdkit",
            "placement aromatic and ring heuristics",
            "pip install rdkit",
        ) from exc
    return Chem


def _is_flat_aromatic(
    shape: str,
    smiles: str | None,
    symbols: list[str],
) -> bool:
    """True if the adsorbate is flat with aromatic EN atoms (parallel-placement candidate).

    With SMILES, requires RDKit aromatic rings plus electronegative binders.
    Without SMILES, any flat molecule that has binder candidates qualifies
    (no aromatic-ring check is possible from symbols alone).
    """
    if shape != "flat":
        return False
    binders = geom._binding_atom_candidates(symbols)
    if smiles is not None:
        return _is_flat_aromatic_with_en(smiles)
    return bool(binders)


def _is_flat_aromatic_with_en(smiles: str) -> bool:
    """True if molecule has aromatic rings and electronegative (binding) atoms.

    Uses the heavy-atom SMILES graph only (no explicit H addition); aromaticity
    and binder identity are defined on heavy atoms.
    """
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    has_en = bool(geom._binding_atom_candidates(symbols))
    return bool(aromatic and has_en)


def _mean_molecule_covalent_radius(symbols: list[str]) -> float:
    radii = [geom._get_covalent_radius(s) for s in symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        return _MOL_COVALENT_RADIUS_FALLBACK
    return float(np.mean(valid))


def _radius_sum_for_site(
    slab: Atoms,
    site: Site | None,
    mol_symbols: list[str],
) -> float | None:
    r_surface = _get_site_surface_radii(slab, site)
    if r_surface is None:
        return None
    return r_surface + _mean_molecule_covalent_radius(mol_symbols)


def _site_type_z_offset(
    slab: Atoms,
    site: Site | None,
    site_type: str | None,
) -> float:
    if not site_type or site_type not in _SITE_Z_OFFSET_FROM_SURFACE_RADIUS:
        return 0.0
    r_surface = _get_site_surface_radii(slab, site)
    if r_surface is None:
        return 0.0
    return _SITE_Z_OFFSET_FROM_SURFACE_RADIUS[site_type] * r_surface


def _parallel_z_adjustments(
    slab: Atoms,
    site: Site | None,
    mol_symbols: list[str],
) -> tuple[float, float, float]:
    radius_sum = _radius_sum_for_site(slab, site, mol_symbols)
    if radius_sum is None:
        return (
            _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
            _PARALLEL_Z_LO_SHRINK_FALLBACK_ANGSTROM,
            _PARALLEL_Z_HI_SHRINK_FALLBACK_ANGSTROM,
        )
    return (
        max(
            _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
            _PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE * radius_sum,
        ),
        _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE * radius_sum,
        _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE * radius_sum,
    )


def classify_adsorbate_orientation(
    atoms: Atoms,
    slab_size: int,
    threshold: float = _ORIENTATION_CLASSIFICATION_PARALLEL_DOT_THRESHOLD,
    *,
    normal: np.ndarray | None = None,
) -> str:
    """Classify adsorbate plane as ``"parallel"``, ``"tilted"``, or ``"unknown"``.

    Uses the inertia-tensor plane normal vs the surface normal. For flat
    molecules, the plane normal is the axis of largest inertia (eigenvecs[:, 2]),
    per the perpendicular axis theorem. Parallel = ring approximately horizontal;
    tilted = ring not parallel to the surface.

    Returns ``"unknown"`` when the adsorbate has fewer than 3 atoms (no plane).

    *normal* defaults to the slab normal from the cell (``a×b``); pass an explicit
    unit vector for non-standard frames.
    """
    pos = atoms.get_positions()[slab_size:]
    if len(pos) < 3:
        return "unknown"
    masses = atoms.get_masses()[slab_size:]
    _, eigenvecs = geom._compute_inertia_tensor(pos, masses)
    plane_normal = eigenvecs[:, 2]
    if normal is None:
        cell = np.asarray(atoms.get_cell(), dtype=float)
        surface_normal = _slab_normal(cell)
    else:
        surface_normal = np.asarray(normal, dtype=float)
        nrm = float(np.linalg.norm(surface_normal))
        surface_normal = (
            surface_normal / nrm
            if nrm > _VECTOR_NORM_EPS
            else np.array([0.0, 0.0, 1.0])
        )
    if float(np.dot(plane_normal, surface_normal)) < 0:
        plane_normal = -plane_normal
    dot = abs(float(np.dot(plane_normal, surface_normal)))
    return "parallel" if dot > threshold else "tilted"


def _estimate_parallel_fraction(
    symbols: list[str],
    smiles: str | None,
) -> float:
    """Estimate the fraction of placements that should be parallel (π-stacking).

    Returns a value in [0.3, 0.8] based on binder count and (when SMILES is
    available) aromatic ring size:

    - no binders (e.g. benzene) → 0.8
    - single binder (e.g. pyridine, phenol) → 0.3
    - multiple binders: ratio of binders to ring atoms selects 0.8 / 0.5 / 0.3

    Without SMILES, ring size falls back to the carbon-atom count, so the same
    molecule can score differently than with SMILES aromatic atoms.
    """
    binders = geom._binding_atom_candidates(symbols)
    n_binders = len(binders)
    if n_binders == 0:
        return _PARALLEL_FRACTION_NO_BINDERS
    if n_binders == 1:
        return _PARALLEL_FRACTION_SINGLE_BINDER

    n_ring = 0
    if smiles is not None:
        Chem = _rdkit_chem()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            n_ring = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    if n_ring == 0:
        n_ring = sum(1 for s in symbols if s == "C")

    if n_ring == 0:
        return _PARALLEL_FRACTION_NO_RING
    ratio = n_binders / n_ring
    if ratio >= _PARALLEL_FRACTION_HIGH_RATIO_CUTOFF:
        return _PARALLEL_FRACTION_HIGH_BINDER_RATIO
    if ratio >= _PARALLEL_FRACTION_MEDIUM_RATIO_CUTOFF:
        return _PARALLEL_FRACTION_MEDIUM_BINDER_RATIO
    return _PARALLEL_FRACTION_LOW_BINDER_RATIO


@dataclass
class OrientedAdsorbate:
    """Canonical-frame adsorbate after orientation, before site translation."""

    rotated_pos: np.ndarray
    canonical_pos: np.ndarray
    quat: np.ndarray  # (w, x, y, z)
    normal: np.ndarray


def _finish_orientation(
    canonical_pos: np.ndarray,
    base_pos: np.ndarray,
    normal: np.ndarray,
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    rotated_pos = geom._rotation_with_tilt(
        base_pos, normal, spec.tilt_deg, spec.azimuth_deg
    )
    rot_mat = geom.best_fit_rotation(canonical_pos, rotated_pos)
    quat = geom.rotation_matrix_to_quaternion(rot_mat)
    return OrientedAdsorbate(
        rotated_pos=rotated_pos,
        canonical_pos=canonical_pos,
        quat=np.asarray(quat, dtype=float),
        normal=np.asarray(normal, dtype=float),
    )


def _orient_parallel(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    base_pos = geom._flat_orientation_from_principal_axis(
        canonical_pos,
        normal,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
        face_flip=spec.face_flip,
    )
    return _finish_orientation(canonical_pos, base_pos, normal, spec)


def _orient_binder_aligned(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    symbols: list[str],
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    base_pos = geom._surface_aligned_rotation(
        canonical_pos,
        normal,
        symbols,
        en_binder_index=spec.en_atom_index,
    )
    return _finish_orientation(canonical_pos, base_pos, normal, spec)


def orient_from_spec(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    symbols: list[str],
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    """Select parallel vs binder-aligned orientation from *spec.orientation_type*."""
    if spec.orientation_type == "parallel":
        return _orient_parallel(canonical_pos, normal=normal, spec=spec)
    return _orient_binder_aligned(
        canonical_pos, normal=normal, symbols=symbols, spec=spec
    )

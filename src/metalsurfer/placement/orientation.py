"""Aromatic heuristics, parallel fraction, and adsorbate orientation."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from ase import Atoms

from ..exceptions import DependencyMissingError
from ..models import PlacementSpec
from . import geometry as geom
from ._constants import (
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
    _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE,
    _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE,
    _SITE_Z_OFFSET_FROM_SURFACE_RADIUS,
)
from .site_coords import _mean_covalent_radius
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
    """Check whether the adsorbate is flat with aromatic EN atoms (parallel-placement candidate).

    With SMILES, requires RDKit aromatic rings plus electronegative binders.
    Without SMILES, any flat molecule that has binder candidates qualifies
    (no aromatic-ring check is possible from symbols alone).
    """
    if shape != "flat":
        return False
    if smiles is not None:
        return _is_flat_aromatic_with_en(smiles)
    binders = geom._binding_atom_candidates(symbols)
    return bool(binders)


@lru_cache(maxsize=256)
def _smiles_aromatic_binder_info(smiles: str) -> tuple[bool, int] | None:
    """Parse *smiles* once: ``(flat_aromatic_with_en, n_aromatic_atoms)`` or None."""
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    n_aromatic = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    has_en = bool(geom._binding_atom_candidates(symbols))
    return bool(n_aromatic > 0 and has_en), n_aromatic


def _is_flat_aromatic_with_en(smiles: str) -> bool:
    """Check whether the molecule has aromatic rings and electronegative (binding) atoms.

    Uses the heavy-atom SMILES graph only (no explicit H addition); aromaticity
    and binder identity are defined on heavy atoms.
    """
    info = _smiles_aromatic_binder_info(smiles)
    return bool(info is not None and info[0])


def _aromatic_ring_atom_count(smiles: str) -> int | None:
    """Return aromatic heavy-atom count for *smiles*, or None if parse fails."""
    info = _smiles_aromatic_binder_info(smiles)
    return None if info is None else info[1]


def _radius_sum_for_site(
    slab: Atoms,
    site: Site | None,
    mol_symbols: list[str],
    r_surface: float | None = None,
) -> float:
    if r_surface is None:
        r_surface = _get_site_surface_radii(slab, site)
    return r_surface + _mean_covalent_radius(mol_symbols)


def _site_type_z_offset(
    slab: Atoms,
    site: Site | None,
    site_type: str | None,
    r_surface: float | None = None,
) -> float:
    if not site_type or site_type not in _SITE_Z_OFFSET_FROM_SURFACE_RADIUS:
        return 0.0
    multiplier = _SITE_Z_OFFSET_FROM_SURFACE_RADIUS[site_type]
    # A zero multiplier never depends on the surface radius; short-circuit so a
    # caller may omit a precomputed *r_surface* without changing the result.
    if multiplier == 0.0:
        return 0.0
    if r_surface is None:
        r_surface = _get_site_surface_radii(slab, site)
    return multiplier * r_surface


def _parallel_z_adjustments(
    slab: Atoms,
    site: Site | None,
    mol_symbols: list[str],
    r_surface: float | None = None,
) -> tuple[float, float, float]:
    radius_sum = _radius_sum_for_site(slab, site, mol_symbols, r_surface=r_surface)
    return (
        max(
            _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
            _PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE * radius_sum,
        ),
        _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE * radius_sum,
        _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE * radius_sum,
    )


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
        counted = _aromatic_ring_atom_count(smiles)
        if counted is not None:
            n_ring = counted
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
    quat: np.ndarray  # (w, x, y, z)


def _finish_orientation(
    canonical_pos: np.ndarray,
    base_pos: np.ndarray,
    normal: np.ndarray,
    spec: PlacementSpec,
    *,
    R_base: np.ndarray,
) -> OrientedAdsorbate:
    rotated_pos, R_tilt = geom._rotation_with_tilt(
        base_pos, normal, spec.tilt_deg, spec.azimuth_deg
    )
    R_total = R_tilt @ R_base
    quat = geom.rotation_matrix_to_quaternion(R_total)
    return OrientedAdsorbate(
        rotated_pos=rotated_pos,
        quat=np.asarray(quat, dtype=float),
    )


def _orient_parallel(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    base_pos, R_base = geom._flat_orientation_from_principal_axis(
        canonical_pos,
        normal,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
        face_flip=spec.face_flip,
    )
    return _finish_orientation(canonical_pos, base_pos, normal, spec, R_base=R_base)


def _orient_binder_aligned(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    symbols: list[str],
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    base_pos, R_base = geom._surface_aligned_rotation(
        canonical_pos,
        normal,
        symbols,
        en_binder_index=spec.en_atom_index,
    )
    return _finish_orientation(canonical_pos, base_pos, normal, spec, R_base=R_base)


def orient_from_spec(
    canonical_pos: np.ndarray,
    *,
    normal: np.ndarray,
    symbols: list[str],
    spec: PlacementSpec,
) -> OrientedAdsorbate:
    """Select parallel vs binder-aligned orientation from *spec.orientation_type*.

    Parameters
    ----------
    canonical_pos
        Canonical adsorbate positions.
    normal
        Surface normal vector.
    symbols
        Chemical symbols of the adsorbate atoms.
    spec
        :class:`~metalsurfer.models.PlacementSpec` defining the orientation.
    """
    if spec.orientation_type == "parallel":
        return _orient_parallel(canonical_pos, normal=normal, spec=spec)
    return _orient_binder_aligned(
        canonical_pos, normal=normal, symbols=symbols, spec=spec
    )

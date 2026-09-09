"""Post-relaxation filtering: decomposition, desorption, energy cap, deduplication.

:func:`filter_results` runs one pipeline. Decomposition can check connectivity
(at multiple radii), formula, bond-pair counts, and coordination vs reference SMILES.

:func:`adsorbate_connected_components` splits the adsorbate region into bonded
fragments; saturation uses it for topology checks before best-slab selection
when ``saturation_discard_topology_rearrangements`` is enabled.
:func:`check_decomposition` is used by :func:`filter_results`.
"""

import logging
import time
from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from scipy.sparse.csgraph import connected_components

from ._geom_pbc import cart_to_frac, frac_to_cart, wrap_fractional
from ._logging import warn_once
from ._utils import cell_has_volume
from .config import AdsorptionConfig
from .exceptions import DependencyMissingError
from .models import ScreeningResult
from .placement._material import material_aware_pbc
from .placement.geometry import _mol_slab_pairwise_distances, calculate_min_distance

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - simulated in dependency_behavior tests
    Chem = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# connectivity helpers
# ---------------------------------------------------------------------------


def _pbc_for_cell(material_type: str, cell: np.ndarray) -> list[bool]:
    """Material-aware PBC, disabled when *cell* has no usable volume."""
    pbc = material_aware_pbc(material_type)
    if not cell_has_volume(cell):
        return [False, False, False]
    return pbc


def _mic_pairwise_distances(
    coords: np.ndarray,
    atoms: Atoms,
    *,
    material_type: str = "slab",
) -> np.ndarray:
    """Return an (n, n) MIC-aware distance matrix for *coords*.

    Uses :func:`material_aware_pbc` (not ``atoms.get_pbc()``) so calculator
    TTT promotion cannot invent vacuum-wrap bonds on slabs.
    """
    coords = np.asarray(coords, dtype=float)
    n = len(coords)
    if n <= 1:
        return np.zeros((n, n))
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = _pbc_for_cell(material_type, cell)
    # Shared placement primitive (same MIC math as every other distance gate).
    return _mol_slab_pairwise_distances(coords, coords, cell, pbc)


def _covalent_threshold_matrix(syms: np.ndarray, multiplier: float) -> np.ndarray:
    """Build (n, n) matrix of ``multiplier * (r_cov_i + r_cov_j)``."""
    z = np.array([atomic_numbers[s] for s in syms])
    r_cov = covalent_radii[z]
    return multiplier * (r_cov[:, None] + r_cov[None, :])


def _nonsurface_distance_and_threshold(
    atoms: Atoms,
    surface_symbols: list[str] | None,
    multiplier: float,
    *,
    material_type: str = "slab",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mask out surface atoms and return (syms, coords, dist_matrix, threshold).

    Shared preamble for the bond-count and coordination-fingerprint filters: the
    non-surface symbol mask, the MIC-aware pairwise distances, and the covalent
    threshold matrix are computed identically in both.
    """
    syms = np.array(atoms.get_chemical_symbols())
    mask = (
        ~np.isin(syms, surface_symbols)
        if surface_symbols is not None
        else np.ones(len(syms), dtype=bool)
    )
    coords = atoms.get_positions()[mask]
    syms = syms[mask]
    dist_matrix = _mic_pairwise_distances(coords, atoms, material_type=material_type)
    threshold = _covalent_threshold_matrix(syms, multiplier)
    return syms, coords, dist_matrix, threshold


def _adjacency_mask(
    dist_matrix: np.ndarray, threshold_matrix: np.ndarray
) -> np.ndarray:
    """Upper-triangle boolean mask where distance <= threshold."""
    bonded = dist_matrix <= threshold_matrix
    np.fill_diagonal(bonded, False)
    return np.triu(bonded, k=1)


def _bond_counts_from_dist(
    syms: np.ndarray,
    dist_matrix: np.ndarray,
    threshold: np.ndarray,
) -> Counter:
    """Count bonds (by element-pair) from a precomputed distance matrix."""
    bonded = _adjacency_mask(dist_matrix, threshold)
    pairs_i, pairs_j = np.nonzero(bonded)

    bonds: Counter = Counter()
    for i, j in zip(pairs_i, pairs_j, strict=True):
        bonds[frozenset({syms[i], syms[j]})] += 1
    return bonds


def _mol_from_smiles(smiles: str):
    """Parse SMILES to an RDKit mol with Hs, or None if rdkit missing or parse fails."""
    if Chem is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.AddHs(mol)
    except (ValueError, TypeError, RuntimeError) as e:
        logger.debug("Failed to parse SMILES: %s", e)
        return None


def _bond_counts_from_smiles(smiles: str) -> Counter | None:
    """Derive bond counts from an RDKit molecule (reference connectivity)."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    bonds: Counter = Counter()
    for bond in mol.GetBonds():
        s1 = bond.GetBeginAtom().GetSymbol()
        s2 = bond.GetEndAtom().GetSymbol()
        bonds[frozenset({s1, s2})] += 1
    return bonds


def _formula_from_smiles(smiles: str) -> Counter | None:
    """Return the molecular formula as a Counter of element symbols."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    formula: Counter = Counter()
    for atom in mol.GetAtoms():
        formula[atom.GetSymbol()] += 1
    return formula


def _formula_from_atoms(
    atoms: Atoms,
    surface_symbols: list[str] | None = None,
) -> Counter:
    """Return elemental composition of non-surface atoms."""
    syms = atoms.get_chemical_symbols()
    formula: Counter = Counter()
    for s in syms:
        if surface_symbols is not None and s in surface_symbols:
            continue
        formula[s] += 1
    return formula


def _coordination_fingerprint_from_dist(
    syms: np.ndarray,
    dist_matrix: np.ndarray,
    threshold: np.ndarray,
) -> dict[str, list[int]]:
    """Per-element sorted coordination numbers from a distance matrix."""
    bonded = dist_matrix <= threshold
    np.fill_diagonal(bonded, False)
    coord_counts = bonded.sum(axis=1).astype(int)

    fingerprint: dict[str, list[int]] = {}
    for i, s in enumerate(syms):
        fingerprint.setdefault(s, []).append(int(coord_counts[i]))
    for key in fingerprint:
        fingerprint[key].sort()
    return fingerprint


def _coordination_fingerprint_from_smiles(
    smiles: str,
) -> dict[str, list[int]] | None:
    """Per-element sorted coordination numbers from the SMILES graph."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    fingerprint: dict[str, list[int]] = {}
    for atom in mol.GetAtoms():
        s = atom.GetSymbol()
        degree = atom.GetDegree()
        fingerprint.setdefault(s, []).append(degree)
    for key in fingerprint:
        fingerprint[key].sort()
    return fingerprint


def _connected_components_from_coords(
    coords: np.ndarray,
    syms: np.ndarray,
    atoms: Atoms,
    multiplier: float,
    *,
    material_type: str = "slab",
) -> list[np.ndarray]:
    """Return boolean masks (one per fragment) for bonded clusters in *coords*."""
    if len(coords) <= 1:
        return [np.ones(len(coords), dtype=bool)] if len(coords) == 1 else []

    dist_matrix = _mic_pairwise_distances(coords, atoms, material_type=material_type)
    threshold = _covalent_threshold_matrix(syms, multiplier)
    bonded = dist_matrix <= threshold
    np.fill_diagonal(bonded, False)

    n_components, labels = connected_components(bonded, directed=False)
    masks: list[np.ndarray] = []
    for label in range(n_components):
        mask = labels == label
        if np.any(mask):
            masks.append(mask)
    return masks


def adsorbate_connected_components(
    atoms: Atoms,
    base_slab_len: int,
    connectivity_multiplier: float,
    *,
    material_type: str = "slab",
) -> list[Atoms]:
    """Split ``atoms[base_slab_len:]`` into connected adsorbate fragments.

    Parameters
    ----------
    atoms
        Combined slab+adsorbate ASE Atoms.
    base_slab_len
        Number of atoms belonging to the base slab.
    connectivity_multiplier
        Covalent-radius multiplier for bond detection.
    material_type
        Geometric PBC via :func:`material_aware_pbc`.
    """
    ads = atoms[base_slab_len:]
    if len(ads) == 0:
        return []

    syms = np.array(ads.get_chemical_symbols())
    coords = ads.get_positions()
    masks = _connected_components_from_coords(
        coords,
        syms,
        ads,
        connectivity_multiplier,
        material_type=material_type,
    )
    return [ads[mask] for mask in masks]


def _is_molecule_connected_from_dist(
    syms: np.ndarray,
    dist_matrix: np.ndarray,
    threshold: np.ndarray,
) -> bool:
    """Return ``True`` if atoms form a single connected fragment."""
    if len(syms) <= 1:
        return True
    bonded = dist_matrix <= threshold
    np.fill_diagonal(bonded, False)
    n_components, _ = connected_components(bonded, directed=False)
    return n_components == 1


# ---------------------------------------------------------------------------
# individual filter predicates
# ---------------------------------------------------------------------------


def check_decomposition(
    atoms: Atoms,
    reference_smiles: str | None,
    surface_symbols: list[str] | None,
    connectivity_multiplier: float,
    adsorbate_prefix_atoms: int | None = None,
    *,
    material_type: str = "slab",
) -> tuple[bool, str]:
    """Return ``(ok, reason)`` indicating whether the adsorbate decomposed.

    ``ok=False`` means the adsorbate decomposed or rearranged.
    When *adsorbate_prefix_atoms* is set, only ``atoms[adsorbate_prefix_atoms:]``
    is checked (sequential saturation: slab already contains prior adsorbates).
    That slice must match ASE ordering ``slab_passed_to_filter + new_adsorbate``,
    consistent with :func:`check_desorption`. When unset, all non-surface atoms
    in *atoms* are checked (legacy behaviour).

    Parameters
    ----------
    atoms
        Structure to check.
    reference_smiles
        SMILES of the original molecule for reference connectivity.
    surface_symbols
        Element symbols of the surface atoms.
    connectivity_multiplier
        Covalent-radius multiplier for bond detection.
    adsorbate_prefix_atoms
        When set, only ``atoms[adsorbate_prefix_atoms:]`` is checked.
    material_type
        Geometric PBC via :func:`material_aware_pbc` (see :func:`check_desorption`).
    """
    if adsorbate_prefix_atoms is not None:
        if adsorbate_prefix_atoms < 0 or adsorbate_prefix_atoms > len(atoms):
            return (
                False,
                f"invalid adsorbate_prefix_atoms ({adsorbate_prefix_atoms} "
                f"for {len(atoms)} atoms)",
            )
        atoms = atoms[adsorbate_prefix_atoms:]
        if len(atoms) == 0:
            return False, "no adsorbate atoms after prefix"
        surface_symbols = None

    syms, _coords, dist_matrix, threshold = _nonsurface_distance_and_threshold(
        atoms,
        surface_symbols,
        connectivity_multiplier,
        material_type=material_type,
    )
    if not _is_molecule_connected_from_dist(syms, dist_matrix, threshold):
        return False, f"adsorbate not connected (multiplier={connectivity_multiplier})"

    if reference_smiles is None:
        return True, "connectivity intact (no SMILES reference for deeper checks)"

    if Chem is None:
        raise DependencyMissingError(
            "rdkit",
            "check_decomposition",
            "pip install rdkit",
        )

    ref_formula = _formula_from_smiles(reference_smiles)
    if ref_formula is not None:
        actual_formula = _formula_from_atoms(atoms, surface_symbols=surface_symbols)
        if actual_formula != ref_formula:
            return (
                False,
                f"formula mismatch: expected {dict(ref_formula)}, "
                f"got {dict(actual_formula)}",
            )

    ref_bonds = _bond_counts_from_smiles(reference_smiles)
    if ref_bonds is not None:
        actual_bonds = _bond_counts_from_dist(syms, dist_matrix, threshold)
        if actual_bonds != ref_bonds:
            return (
                False,
                f"bond pattern mismatch: expected {dict(ref_bonds)}, "
                f"got {dict(actual_bonds)}",
            )

    ref_coord = _coordination_fingerprint_from_smiles(reference_smiles)
    if ref_coord is not None:
        actual_coord = _coordination_fingerprint_from_dist(syms, dist_matrix, threshold)
        if actual_coord != ref_coord:
            return (
                False,
                f"coordination fingerprint mismatch: expected {ref_coord}, "
                f"got {actual_coord}",
            )

    if ref_formula is None and ref_bonds is None and ref_coord is None:
        logger.warning(
            "Could not parse reference SMILES %r for decomposition check; "
            "falling back to connectivity-only screening",
            reference_smiles,
        )
        return True, (
            "connectivity intact (SMILES unparseable; skipped formula/bond/coord checks)"
        )

    return True, "connectivity intact"


def _adsorbate_surface_min_distance(
    atoms: Atoms,
    slab: Atoms,
    surface_symbols: list[str] | None = None,
    *,
    material_type: str = "slab",
    surface_prefix_atoms: int | None = None,
) -> float | None:
    """Minimum adsorbate-to-surface distance using material-aware PBC.

    *slab* supplies ``slab_size`` (new adsorbate starts at ``len(slab)``).
    Substrate positions come from the optimized ``atoms[:slab_size]`` slice so
    relaxed substrate motion is reflected in the distance check.

    When *surface_prefix_atoms* is set, only ``atoms[:surface_prefix_atoms]``
    counts as the binding surface (bare-substrate prefix under saturation).
    Prefer that over *surface_symbols* on organic/porous frameworks where prior
    adsorbates share element symbols with the substrate. *surface_symbols*
    remains a fallback for callers that identify metal atoms by species.

    Returns ``None`` when the combined structure has no adsorbate atoms.
    """
    slab_size = len(slab)
    adsorbate = atoms[slab_size:]
    if len(adsorbate) == 0:
        return None

    cell = atoms.get_cell()
    optimized_slab = atoms[:slab_size]
    slab_positions = optimized_slab.get_positions()
    if surface_prefix_atoms is not None:
        if surface_prefix_atoms < 0 or surface_prefix_atoms > slab_size:
            raise ValueError(
                f"surface_prefix_atoms ({surface_prefix_atoms}) must be in "
                f"[0, {slab_size}] (len(slab))"
            )
        slab_positions = slab_positions[:surface_prefix_atoms]
    elif surface_symbols:
        slab_syms = np.array(optimized_slab.get_chemical_symbols())
        mask = np.isin(slab_syms, surface_symbols)
        if np.any(mask):
            slab_positions = slab_positions[mask]

    pbc_for_dist = material_aware_pbc(material_type)
    return calculate_min_distance(
        adsorbate.get_positions(),
        slab_positions,
        cell,
        use_pbc=True,
        pbc=pbc_for_dist,
    )


def check_desorption(
    atoms: Atoms,
    slab: Atoms,
    binding_threshold: float = 4.0,
    surface_symbols: list[str] | None = None,
    *,
    material_type: str = "slab",
    surface_prefix_atoms: int | None = None,
) -> tuple[bool, str]:
    """Return ``(ok, reason)``; ``ok=False`` means the adsorbate desorbed.

    Complementary to placement: initial placement uses min_contact_ratio to avoid
    covalent binding; this post-optimization filter rejects structures that
    drifted too far from the surface (binding_threshold, default 4.0 Å).

    Periodicity note: this is a *geometric* criterion, so it uses the substrate's
    real periodicity (:func:`material_aware_pbc`) rather than the calculator's
    promoted 3D PBC. For a slab the difference is the ``c`` axis, which is the
    vacuum direction: under 3D PBC an adsorbate that has flown off the surface
    minimum-image-wraps to the bottom of the periodic image and is scored as
    still bound. With ``MIN_CALCULATOR_CELL_C_ANG = 18`` a molecule 12 Å above a
    slab whose top sits at z=3 gives an image distance of 3 Å, i.e. below the
    default 4 Å threshold, so desorption would go undetected. Porous frameworks
    keep ``[True, True, True]`` because there the wrap is physical. The same
    material-aware convention applies to :func:`check_decomposition` and
    duplicate RMSD.

    Parameters
    ----------
    atoms
        Optimized structure to check.
    slab
        Reference slab Atoms.
    binding_threshold
        Maximum allowed adsorbate-surface distance in Å.
    surface_symbols
        Element symbols of the surface atoms (fallback when no bare prefix).
    material_type
        Material type string (e.g. "slab", "porous", "nanoparticle").
    surface_prefix_atoms
        Bare-substrate atom count; when set, prior adsorbates in the coverage
        prefix are ignored by index (preferred for organic/porous substrates).
    """
    min_d = _adsorbate_surface_min_distance(
        atoms,
        slab,
        surface_symbols=surface_symbols,
        material_type=material_type,
        surface_prefix_atoms=surface_prefix_atoms,
    )
    if min_d is None:
        return False, "no adsorbate atoms found"
    if min_d > binding_threshold:
        return False, f"adsorbate too far from surface ({min_d:.2f} A)"

    return True, f"adsorbed (min distance {min_d:.2f} A)"


# ---------------------------------------------------------------------------
# duplicate detection
# ---------------------------------------------------------------------------


def _adsorbate_rmsd(
    a1: Atoms,
    a2: Atoms,
    surface_symbols: list[str] | None = None,
    *,
    material_type: str = "slab",
    prefix_atoms: int | None = None,
) -> float:
    """MIC-aware RMSD between adsorbate atoms (material-aware PBC).

    When *prefix_atoms* is set, only the trailing suffix
    ``positions[prefix_atoms:]`` is compared (saturation uniqueness on the
    newly added adsorbate). Otherwise, when *surface_symbols* is set, all
    non-surface atoms are compared.
    """
    pos1 = a1.get_positions()
    pos2 = a2.get_positions()
    if prefix_atoms is not None:
        if prefix_atoms < 0 or len(pos1) < prefix_atoms or len(pos2) < prefix_atoms:
            return float("inf")
        pos1, pos2 = pos1[prefix_atoms:], pos2[prefix_atoms:]
        if len(pos1) != len(pos2) or len(pos1) == 0:
            return float("inf")
    elif surface_symbols:
        s1 = np.array(a1.get_chemical_symbols())
        s2 = np.array(a2.get_chemical_symbols())
        m1 = ~np.isin(s1, surface_symbols)
        m2 = ~np.isin(s2, surface_symbols)
        if m1.sum() != m2.sum():
            return float("inf")
        pos1, pos2 = pos1[m1], pos2[m2]
        if len(pos1) == 0:
            return float("inf")
    cell = np.asarray(a1.get_cell(), dtype=float)
    pbc = _pbc_for_cell(material_type, cell)
    diffs, _ = find_mic(pos1 - pos2, cell, pbc)
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


def _trailing_adsorbate_com(
    atoms: Atoms,
    prefix_atoms: int,
    *,
    material_type: str,
) -> np.ndarray:
    """Equal-weight COM of ``positions[prefix_atoms:]``, wrapped on periodic axes.

    Atoms are MIC-unwrapped relative to the first adsorbate atom before averaging
    so a molecule that straddles a periodic face does not pull the COM into the
    cell interior.
    """
    pos = np.asarray(atoms.get_positions()[prefix_atoms:], dtype=float)
    if len(pos) == 0:
        return np.zeros(3, dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = _pbc_for_cell(material_type, cell)
    if any(pbc) and len(pos) > 1:
        ref = pos[0]
        mic_deltas, _ = find_mic(pos - ref, cell, pbc)
        com = (ref + mic_deltas).mean(axis=0)
    else:
        com = pos.mean(axis=0)
    if not any(pbc):
        return com
    frac = wrap_fractional(
        cart_to_frac(com.reshape(1, 3), cell), np.asarray(pbc, dtype=bool)
    )
    return frac_to_cart(frac, cell).reshape(3)


def _bin_counts_for_cell(cell: np.ndarray, bin_size: float) -> tuple[int, int, int]:
    lengths = np.linalg.norm(np.asarray(cell, dtype=float), axis=1)
    counts: list[int] = []
    for length in lengths:
        if length <= 0.0 or bin_size <= 0.0:
            counts.append(1)
        else:
            counts.append(max(1, int(np.ceil(float(length) / bin_size))))
    return (counts[0], counts[1], counts[2])


def _com_bin_key(
    com: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
    bin_size: float,
) -> tuple[int, int, int]:
    """Fractional-space grid key; periodic axes wrap into ``[0, n_bins)``."""
    cell_arr = np.asarray(cell, dtype=float)
    pbc_arr = np.asarray(pbc, dtype=bool)
    n_bins = _bin_counts_for_cell(cell_arr, bin_size)
    if not cell_has_volume(cell_arr):
        return (
            int(np.floor(float(com[0]) / bin_size)),
            int(np.floor(float(com[1]) / bin_size)),
            int(np.floor(float(com[2]) / bin_size)),
        )
    frac = wrap_fractional(cart_to_frac(com.reshape(1, 3), cell_arr), pbc_arr)[0]
    keys: list[int] = []
    for i in range(3):
        bi = int(np.floor(float(frac[i]) * n_bins[i]))
        if pbc_arr[i]:
            bi %= n_bins[i]
        keys.append(bi)
    return (keys[0], keys[1], keys[2])


def _neighbor_bin_keys(
    key: tuple[int, int, int],
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
    bin_size: float,
) -> list[tuple[int, int, int]]:
    """Return *key* and its 3×3×3 neighborhood (modular wrap on periodic axes)."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    n_bins = _bin_counts_for_cell(np.asarray(cell, dtype=float), bin_size)
    neighbors: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                coords = [key[0] + dx, key[1] + dy, key[2] + dz]
                for i in range(3):
                    if pbc_arr[i]:
                        coords[i] %= n_bins[i]
                neighbors.append((coords[0], coords[1], coords[2]))
    return neighbors


def _mic_com_distance(
    com1: np.ndarray,
    com2: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
) -> float:
    cell_arr = np.asarray(cell, dtype=float)
    pbc_list = list(pbc) if cell_has_volume(cell_arr) else [False, False, False]
    diffs, _ = find_mic((com1 - com2).reshape(1, 3), cell_arr, pbc_list)
    return float(np.linalg.norm(diffs[0]))


def _deduplicate_by_energy_and_rmsd(
    results: list[ScreeningResult],
    *,
    prefix_atoms: int,
    surface_symbols: list[str] | None,
    material_type: str,
    energy_dedup_threshold: float,
    rmsd_dedup_threshold: float,
) -> tuple[list[ScreeningResult], list[ScreeningResult]]:
    """Keep lowest-energy unique poses; prune with energy + COM bins before RMSD.

    Uniqueness is on the trailing adsorbate (``prefix_atoms``), not prior
    coverage. Full pairwise RMSD runs only for uniques that share the energy
    window, a neighboring COM grid cell, and MIC COM distance within
    *rmsd_dedup_threshold* (index-aligned RMSD cannot be smaller than COM MIC).
    """
    sorted_results = sorted(results, key=lambda r: r.energy_adsorption)
    unique: list[ScreeningResult] = []
    unique_energy: list[float] = []
    unique_n_ads: list[int] = []
    unique_com: list[np.ndarray] = []
    bins: dict[tuple[int, int, int], list[int]] = {}
    deduplicated: list[ScreeningResult] = []

    ref_cell = np.asarray(sorted_results[0].atoms.get_cell(), dtype=float)
    pbc = _pbc_for_cell(material_type, ref_cell)
    bin_size = float(rmsd_dedup_threshold) if rmsd_dedup_threshold > 0.0 else 0.1

    for entry in sorted_results:
        e = entry.energy_adsorption
        n_ads = len(entry.atoms) - prefix_atoms
        if n_ads < 0:
            unique.append(entry)
            unique_energy.append(e)
            unique_n_ads.append(n_ads)
            unique_com.append(np.zeros(3, dtype=float))
            continue
        com = _trailing_adsorbate_com(
            entry.atoms, prefix_atoms, material_type=material_type
        )
        key = _com_bin_key(com, ref_cell, pbc, bin_size)
        is_dup = False

        candidate_idxs: list[int] = []
        seen_idx: set[int] = set()
        for nkey in _neighbor_bin_keys(key, ref_cell, pbc, bin_size):
            for u_idx in bins.get(nkey, ()):
                if u_idx in seen_idx:
                    continue
                seen_idx.add(u_idx)
                energy_gap = abs(e - unique_energy[u_idx])
                if energy_gap >= energy_dedup_threshold:
                    continue
                if n_ads != unique_n_ads[u_idx]:
                    continue
                if (
                    _mic_com_distance(com, unique_com[u_idx], ref_cell, pbc)
                    > rmsd_dedup_threshold
                ):
                    continue
                candidate_idxs.append(u_idx)

        # Prefer lower-energy uniques first (stable with energy sort of `unique`).
        for u_idx in sorted(candidate_idxs):
            rmsd = _adsorbate_rmsd(
                entry.atoms,
                unique[u_idx].atoms,
                surface_symbols=surface_symbols,
                material_type=material_type,
                prefix_atoms=prefix_atoms,
            )
            if rmsd < rmsd_dedup_threshold:
                is_dup = True
                break

        if not is_dup:
            u_idx = len(unique)
            unique.append(entry)
            unique_energy.append(e)
            unique_n_ads.append(n_ads)
            unique_com.append(com)
            bins.setdefault(key, []).append(u_idx)
        else:
            deduplicated.append(entry)

    return unique, deduplicated


# ---------------------------------------------------------------------------
# unified filter entry point
# ---------------------------------------------------------------------------


def filter_results(
    results: list[ScreeningResult],
    slab: Atoms,
    surface_symbols: list[str] | None = None,
    reference_smiles: str | None = None,
    config: AdsorptionConfig | None = None,
    duplicate_results_out: list[ScreeningResult] | None = None,
    *,
    surface_prefix_atoms: int | None = None,
) -> list[ScreeningResult]:
    """Apply decomposition, desorption and duplicate filters in sequence.

    Parameters
    ----------
    results:
        Typed :class:`ScreeningResult` objects from the compute pipeline.
    slab:
        Reference slab Atoms. Contract: *slab* must be the cumulative substrate
        prefix of every ``entry.atoms`` (bare slab + all previously accepted
        adsorbates). It is used for the desorption distance check and, when
        decomposition is enabled, as the atom-count prefix ``len(slab)`` for
        validating only the newly added adsorbate in each ``entry.atoms``.
        Duplicate RMSD uniqueness is also on that trailing adsorbate suffix,
        not prior coverage. A suffix/formula mismatch now raises
        :class:`ValueError` instead of silently dropping the entry as
        decomposed.
    surface_symbols:
        Element symbols of the surface (e.g. ``["Ru"]`` or ``["Ru", "Cu"]``).
        Fallback for desorption when *surface_prefix_atoms* is unset.
    reference_smiles:
        SMILES of the original molecule; enables formula, bond-count, and
        coordination-fingerprint decomposition checks.
    config:
        Screening configuration (provides thresholds). When
        ``config.skip_topology_check`` is True, decomposition checks are skipped
        (e.g. for dissociative adsorption). When ``config.skip_desorption_check``
        is True, desorption distance checks are skipped.
    duplicate_results_out:
        Optional sink for entries removed by duplicate filtering.
    surface_prefix_atoms:
        Bare-substrate atom count for desorption (preferred over symbol masking
        when prior adsorbates may share elements with the framework).
    """
    if config is None:
        config = AdsorptionConfig()

    if not results:
        return results

    # -- decomposition --------------------------------------------------------
    if config.skip_topology_check:
        warn_once(
            logger,
            "skip_topology",
            "skip_topology_check=True: decomposition checks disabled; "
            "results may include decomposed/rearranged structures",
        )
    if config.skip_desorption_check:
        warn_once(
            logger,
            "skip_desorption",
            "skip_desorption_check=True: desorption distance checks disabled; "
            "results may include desorbed structures",
        )

    if not config.skip_topology_check:
        t0 = time.perf_counter()
        kept: list[ScreeningResult] = []
        decomp_count = 0
        decomp_reasons: dict[str, list[int]] = {}  # reason -> [placement_ids]
        prefix = len(slab)
        ref_formula = (
            _formula_from_smiles(reference_smiles) if reference_smiles else None
        )
        if ref_formula is not None:
            expected_suffix = sum(ref_formula.values())
            for entry in results:
                got = len(entry.atoms) - prefix
                if got != expected_suffix:
                    raise ValueError(
                        "filter_results slab prefix contract violated for "
                        f"placement_id={entry.placement_id}: len(entry.atoms)="
                        f"{len(entry.atoms)} minus len(slab)={prefix} leaves {got} "
                        f"adsorbate atoms, but reference_smiles {reference_smiles!r} has "
                        f"{expected_suffix}. Pass the cumulative substrate prefix (bare "
                        "slab + all previously accepted adsorbates) as `slab`."
                    )
        for entry in results:
            ok, reason = check_decomposition(
                entry.atoms,
                reference_smiles=reference_smiles,
                surface_symbols=surface_symbols,
                connectivity_multiplier=config.connectivity_multiplier,
                adsorbate_prefix_atoms=prefix,
                material_type=config.material_type,
            )
            if ok:
                kept.append(entry)
            else:
                decomp_count += 1
                if reason not in decomp_reasons:
                    decomp_reasons[reason] = []
                decomp_reasons[reason].append(entry.placement_id)
                logger.debug(
                    "Decomposition filter (pid=%s): %s",
                    entry.placement_id,
                    reason,
                )
        t_decomp = time.perf_counter() - t0
        if decomp_count:
            logger.info(
                "Filtered %d decomposed structures (%.3fs)",
                decomp_count,
                t_decomp,
            )
            for reason, pids in sorted(
                decomp_reasons.items(), key=lambda x: -len(x[1])
            ):
                example = pids[0] if pids else None
                logger.info(
                    "  decomposition reason (%d): %s  [example placement_id=%s]",
                    len(pids),
                    reason,
                    example,
                )
        results = kept

    # -- desorption -----------------------------------------------------------
    if not config.skip_desorption_check:
        t0 = time.perf_counter()
        kept = []
        desorb_count = 0
        for entry in results:
            ok, reason = check_desorption(
                entry.atoms,
                slab,
                binding_threshold=config.binding_distance_threshold,
                surface_symbols=surface_symbols,
                material_type=config.material_type,
                surface_prefix_atoms=surface_prefix_atoms,
            )
            if ok:
                kept.append(entry)
            else:
                desorb_count += 1
                logger.debug(
                    "Desorption filter (pid=%s): %s",
                    entry.placement_id,
                    reason,
                )
        t_desorb = time.perf_counter() - t0
        if desorb_count:
            logger.info(
                "Filtered %d desorbed structures (%.3fs)",
                desorb_count,
                t_desorb,
            )
        results = kept

    # -- duplicates -----------------------------------------------------------
    if len(results) <= 1:
        return results

    t0 = time.perf_counter()
    unique, deduplicated = _deduplicate_by_energy_and_rmsd(
        results,
        prefix_atoms=len(slab),
        surface_symbols=surface_symbols,
        material_type=config.material_type,
        energy_dedup_threshold=config.energy_dedup_threshold,
        rmsd_dedup_threshold=config.rmsd_dedup_threshold,
    )
    t_dedup = time.perf_counter() - t0

    dup_count = len(results) - len(unique)
    if dup_count:
        logger.info(
            "Filtered %d duplicate results (%d/%d remain, %.3fs)",
            dup_count,
            len(unique),
            len(results),
            t_dedup,
        )
    if duplicate_results_out is not None and deduplicated:
        duplicate_results_out.extend(deduplicated)
    return unique

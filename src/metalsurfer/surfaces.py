"""Generic surface creation and modification helpers.

This module provides three capabilities:

1. **Base slab creation** from a Materials-Project bulk ID + Miller indices.
2. **Alloy substitution** – randomly replace a fraction of host atoms with a
   different element, evaluate several random variants and keep the
   lowest-energy one.
3. **Adatom deposition** – place adatoms at hollow sites above the top layer
   (identified via Delaunay triangulation), again selecting the best variant.

All functions work with plain ASE ``Atoms`` objects, wrapped inside a
minimal ``SlabContainer`` so callers can reference ``slab.atoms`` the same
way existing workflows do.
"""

import logging
import math
import os

import numpy as np
from ase import Atoms

from .config import AdsorptionConfig
from .exceptions import (
    DependencyMissingError,
    GeometryValidationError,
    OptimizationError,
)
from .io_results import _write_clean_xyz

logger = logging.getLogger(__name__)


class SlabContainer:
    """Minimal wrapper that mimics the FAIRChem ``Slab`` interface.

    Existing workflows access the ASE Atoms via ``slab.atoms``.  This
    container lets generic code produce an object with the same API
    without requiring the FAIRChem data package.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms


# ---------------------------------------------------------------------------
# Base slab creation
# ---------------------------------------------------------------------------


def create_slab_from_bulk(
    bulk_id: str,
    miller_indices: tuple = (0, 0, 1),
    supercell: tuple = (2, 2, 1),
    results_dir: str = "results",
) -> SlabContainer:
    """Create a surface slab from a Materials Project bulk entry.

    Parameters
    ----------
    bulk_id:
        Materials Project ID (e.g. ``"mp-33"`` for Ru).
    miller_indices:
        3-index Miller notation for the desired surface.
    supercell:
        Repeat factors applied after slab generation.
    results_dir:
        Directory where reference POSCAR / XYZ files are saved.
    """
    try:
        from fairchem.data.oc.core import Bulk, Slab
    except ImportError as exc:
        raise DependencyMissingError(
            "fairchem-data-oc",
            "create_slab_from_bulk",
            "Install with: pip install fairchem-data-oc",
        ) from exc

    logger.info(
        "Creating slab from %s, Miller %s, supercell %s...",
        bulk_id,
        miller_indices,
        supercell,
    )

    bulk = Bulk(bulk_src_id_from_db=bulk_id)
    slabs = Slab.from_bulk_get_specific_millers(
        bulk=bulk, specific_millers=miller_indices
    )
    slab = slabs[0]

    logger.info(
        "Original slab: %d atoms, cell: %s",
        len(slab.atoms),
        slab.atoms.cell.lengths(),
    )
    slab.atoms = slab.atoms.repeat(supercell)
    logger.info(
        "Expanded slab: %d atoms, cell: %s",
        len(slab.atoms),
        slab.atoms.cell.lengths(),
    )

    os.makedirs(results_dir, exist_ok=True)
    _write_clean_xyz(slab.atoms, f"{results_dir}/clean_slab.xyz")
    slab.atoms.write(
        f"{results_dir}/clean_slab_POSCAR",
        format="vasp",
        vasp5=True,
        direct=True,
    )
    logger.info("Saved clean slab reference files to %s", results_dir)

    return SlabContainer(slab.atoms)


def create_slab_from_atoms(atoms: Atoms) -> SlabContainer:
    """Wrap an existing ASE ``Atoms`` object into a :class:`SlabContainer`."""
    return SlabContainer(atoms.copy())


# ---------------------------------------------------------------------------
# Alloy substitution
# ---------------------------------------------------------------------------


def _evaluate_variant_energy(variant: Atoms, calculator, context: str = "") -> float:
    """Run single-point energy on variant; return inf on failure."""
    try:
        variant.calc = calculator
        return float(variant.get_potential_energy())
    except (RuntimeError, ValueError) as exc:
        if context:
            logger.warning("%s failed: %s", context, exc)
        return float("inf")


def substitute_alloy(
    slab: SlabContainer,
    host_symbol: str,
    guest_symbol: str,
    guest_fraction: float,
    calculator=None,
    n_variants: int = 5,
    seed: int | None = None,
    relax: bool = True,
    config: AdsorptionConfig | None = None,
    results_dir: str = "results",
) -> SlabContainer:
    """Randomly replace *guest_fraction* of *host_symbol* atoms with *guest_symbol*.

    Multiple random variants are generated and scored; the lowest-energy
    variant is kept.  If *relax* is ``True`` the chosen slab is relaxed
    with ``calculator`` before returning.
    """
    if config is None:
        config = AdsorptionConfig()

    if not 0.0 <= guest_fraction <= 1.0:
        raise ValueError(
            f"guest_fraction must be between 0 and 1, got {guest_fraction}"
        )

    if seed is None:
        seed = config.seed

    base = slab.atoms.copy()
    host_indices = [
        i for i, s in enumerate(base.get_chemical_symbols()) if s == host_symbol
    ]
    n_replace = int(round(len(host_indices) * guest_fraction))

    if n_replace == 0:
        logger.info(
            "guest_fraction=%.2f yields 0 replacements; returning base slab",
            guest_fraction,
        )
        return SlabContainer(base)

    if guest_fraction >= 1.0:
        atoms = base.copy()
        syms = [
            guest_symbol if s == host_symbol else s
            for s in atoms.get_chemical_symbols()
        ]
        atoms.set_chemical_symbols(syms)
        return SlabContainer(atoms)

    rng = np.random.RandomState(seed)
    best_energy = float("inf")
    best_atoms = None

    for v in range(n_variants):
        replace_idx = rng.choice(host_indices, size=n_replace, replace=False)
        variant = base.copy()
        syms = variant.get_chemical_symbols()
        for i in replace_idx:
            syms[i] = guest_symbol
        variant.set_chemical_symbols(syms)

        if calculator is not None:
            energy = _evaluate_variant_energy(
                variant, calculator, context=f"Variant {v}"
            )
            if energy < best_energy:
                best_energy = energy
                best_atoms = variant.copy()
        elif best_atoms is None:
            best_atoms = variant.copy()

    if best_atoms is None:
        raise GeometryValidationError(
            "Failed to generate any valid alloy slab variants"
        )

    if relax and calculator is not None:
        try:
            from ase.optimize import LBFGS

            logger.info("Relaxing alloy slab geometry...")
            best_atoms.calc = calculator
            dyn = LBFGS(best_atoms, logfile=None)
            dyn.run(fmax=config.fmax)
            logger.info(
                "Post-relax slab energy: %.4f eV",
                best_atoms.get_potential_energy(),
            )
        except (RuntimeError, ValueError) as exc:
            raise OptimizationError(f"Alloy slab relaxation failed: {exc}") from exc

    os.makedirs(results_dir, exist_ok=True)
    label = f"{host_symbol}_{guest_symbol}_{int(guest_fraction * 100)}"
    _write_clean_xyz(best_atoms, f"{results_dir}/clean_{label}_slab.xyz")
    best_atoms.write(
        f"{results_dir}/clean_{label}_slab_POSCAR",
        format="vasp",
        vasp5=True,
        direct=True,
    )
    logger.info("Saved alloy slab (%s) to %s", label, results_dir)

    return SlabContainer(best_atoms)


# ---------------------------------------------------------------------------
# Adatom deposition
# ---------------------------------------------------------------------------


def deposit_adatoms(
    slab: SlabContainer,
    adatom_symbol: str,
    coverage_fraction: float,
    calculator=None,
    n_variants: int = 5,
    adsorption_height: float = 1.8,
    seed: int | None = None,
    results_dir: str = "results",
    config: AdsorptionConfig | None = None,
) -> SlabContainer:
    """Place *adatom_symbol* atoms at hollow sites above the top layer.

    Uses Delaunay triangulation of the top-layer xy coordinates to
    identify candidate hollow sites.  *coverage_fraction* of the
    available sites are filled.  The lowest-energy variant is kept.
    """
    if config is None:
        config = AdsorptionConfig()

    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError(
            f"coverage_fraction must be between 0 and 1, got {coverage_fraction}"
        )

    if coverage_fraction == 0.0:
        logger.info("coverage_fraction=0; returning unmodified slab")
        return SlabContainer(slab.atoms.copy())

    if seed is None:
        seed = config.seed

    from .placement import get_hollow_sites_for_adatoms

    base = slab.atoms.copy()
    positions = base.get_positions()
    z_max = float(np.max(positions[:, 2]))

    top_mask = positions[:, 2] >= (z_max - 0.5)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        raise GeometryValidationError(
            "Cannot identify top surface layer for adatom placement "
            f"(found {len(top_indices)} atoms within 0.5 A of z_max)"
        )

    candidate_sites = get_hollow_sites_for_adatoms(
        base, top_layer_tolerance=0.5, dedup_tolerance=0.2
    )

    if not candidate_sites:
        raise GeometryValidationError(
            "No candidate hollow sites found for adatom placement"
        )

    n_place = max(
        1, int(round(coverage_fraction * len(candidate_sites)))
    )  # >=1 guaranteed since coverage_fraction > 0

    rng = np.random.RandomState(seed)
    best_energy = float("inf")
    best_atoms = None

    for v in range(n_variants):
        chosen = rng.choice(len(candidate_sites), size=n_place, replace=False)
        variant = base.copy()
        ad_positions = [
            [candidate_sites[i][0], candidate_sites[i][1], z_max + adsorption_height]
            for i in chosen
        ]
        adatoms = Atoms(
            symbols=[adatom_symbol] * len(ad_positions),
            positions=ad_positions,
        )
        adatoms.set_cell(variant.get_cell())
        adatoms.set_pbc(variant.get_pbc())
        variant += adatoms

        if calculator is not None:
            energy = _evaluate_variant_energy(
                variant, calculator, context=f"Adatom variant {v}"
            )
            if energy < best_energy:
                best_energy = energy
                best_atoms = variant.copy()
        elif best_atoms is None:
            best_atoms = variant.copy()

    if best_atoms is None:
        raise GeometryValidationError(
            "Failed to generate any valid adatom-deposited slab"
        )

    os.makedirs(results_dir, exist_ok=True)
    pct = int(round(coverage_fraction * 100))
    label = f"{adatom_symbol}{pct}"
    _write_clean_xyz(best_atoms, f"{results_dir}/clean_slab_{label}.xyz")
    best_atoms.write(
        f"{results_dir}/clean_slab_{label}_POSCAR",
        format="vasp",
        vasp5=True,
        direct=True,
    )
    logger.info(
        "Created adatom-deposited slab (%s, %.0f%%): E=%.4f eV",
        adatom_symbol,
        coverage_fraction * 100,
        best_energy,
    )

    return SlabContainer(best_atoms)


# ---------------------------------------------------------------------------
# Molecule-aware slab auto-sizing
# ---------------------------------------------------------------------------


def _molecule_diameter(conformers: list[Atoms]) -> float:
    """Maximum pairwise interatomic distance across all conformer geometries."""
    max_dist = 0.0
    for conf in conformers:
        pos = conf.get_positions()
        if len(pos) < 2:
            continue
        diffs = pos[:, None, :] - pos[None, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        max_dist = max(max_dist, float(np.sqrt(np.max(dists_sq))))
    return max_dist


def _perpendicular_heights_2d(cell: np.ndarray) -> tuple[float, float]:
    """Perpendicular heights of the in-plane parallelogram.

    For cell vectors *a* = ``cell[0, :2]`` and *b* = ``cell[1, :2]``:

    * ``h_a`` = perpendicular distance between the sides parallel to *a*
      (i.e. the height in the *b* direction).
    * ``h_b`` = perpendicular distance between the sides parallel to *b*
      (i.e. the height in the *a* direction).

    When the cell is repeated ``(nx, ny, 1)``, the new heights are
    ``ny * h_a`` and ``nx * h_b``.
    """
    a = cell[0, :2].astype(float)
    b = cell[1, :2].astype(float)
    area = abs(float(a[0] * b[1] - a[1] * b[0]))
    len_a = float(np.linalg.norm(a))
    len_b = float(np.linalg.norm(b))
    if len_a < 1e-10 or len_b < 1e-10:
        raise GeometryValidationError("2D cell vectors are degenerate or zero-length")
    return area / len_a, area / len_b


def compute_minimum_supercell(
    cell: np.ndarray,
    molecule_diameter: float,
    min_separation: float = 8.0,
) -> tuple[int, int]:
    """Minimal ``(nx, ny)`` repeat factors ensuring PBC image separation.

    The cell is repeated so that both perpendicular heights of the
    in-plane parallelogram exceed ``molecule_diameter + min_separation``.
    """
    required = molecule_diameter + min_separation
    h_a, h_b = _perpendicular_heights_2d(cell)
    if h_a < 1e-10 or h_b < 1e-10:
        raise GeometryValidationError(
            "Cell perpendicular heights are degenerate; cannot compute minimum supercell"
        )
    # ny repeats scale h_a (perp to a); nx repeats scale h_b (perp to b)
    ny = max(1, math.ceil(required / h_a))
    nx = max(1, math.ceil(required / h_b))
    return (nx, ny)


def auto_resize_slab_for_molecule(
    slab: SlabContainer,
    conformers: list[Atoms],
    min_separation: float = 8.0,
) -> tuple[SlabContainer, bool]:
    """Resize *slab* in-plane so periodic images are well separated.

    Returns ``(slab, was_resized)`` where *was_resized* is ``True``
    when the cell was expanded.
    """
    diameter = _molecule_diameter(conformers)
    cell = np.array(slab.atoms.get_cell())
    nx, ny = compute_minimum_supercell(cell, diameter, min_separation)

    if nx <= 1 and ny <= 1:
        return slab, False

    new_atoms = slab.atoms.repeat((nx, ny, 1))
    logger.info(
        "Auto-resized slab by (%d, %d, 1) for molecule diameter %.2f A "
        "(min separation %.2f A): %d -> %d atoms",
        nx,
        ny,
        diameter,
        min_separation,
        len(slab.atoms),
        len(new_atoms),
    )
    return SlabContainer(new_atoms), True

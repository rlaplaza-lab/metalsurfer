"""Slab construction (bulk + Miller), alloy substitution, adatoms, and ``SlabContainer``."""

import logging
import math
import os
from typing import Literal

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from .config import AdsorptionConfig
from .exceptions import (
    DependencyMissingError,
    GeometryValidationError,
    OptimizationError,
)
from .io_results import _write_clean_xyz

logger = logging.getLogger(__name__)
SLAB_RELAXATION_MODE = Literal["none", "ionic_only", "cell_only", "full"]
SLAB_RELAXATION_OPTIMIZER = Literal["lbfgs", "bfgs", "fire"]


def _import_chain(exc: BaseException | None) -> list[BaseException]:
    """Flatten exception.__cause__ into a list (root first)."""
    chain: list[BaseException] = []
    cur: BaseException | None = exc
    while cur is not None:
        chain.append(cur)
        cur = cur.__cause__
    return chain


def _is_pkg_resources_missing(exc: ImportError) -> bool:
    """True if *exc* or its :attr:`__cause__` chain indicates missing ``pkg_resources``."""
    for err in _import_chain(exc):
        if isinstance(err, ModuleNotFoundError) and err.name == "pkg_resources":
            return True
        if "pkg_resources" in str(err):
            return True
    return False


def _dependency_error_for_slab_import(exc: ImportError) -> DependencyMissingError:
    """Map a failed ``fairchem.data.oc`` import to an actionable :class:`DependencyMissingError`."""
    if _is_pkg_resources_missing(exc):
        return DependencyMissingError(
            "setuptools",
            "create_slab_from_bulk",
            "Install with: pip install 'setuptools<82' (setuptools 82+ removed pkg_resources)",
        )
    msg_l = str(exc).lower()
    if "fairchem" in msg_l:
        return DependencyMissingError(
            "fairchem-data-oc",
            "create_slab_from_bulk",
            "Install with: pip install fairchem-data-oc",
        )
    return DependencyMissingError(
        "fairchem-data-oc",
        "create_slab_from_bulk",
        f"Install with: pip install fairchem-data-oc. Underlying error: {exc!s}",
    )


class SlabContainer:
    """Minimal wrapper that mimics the FAIRChem ``Slab`` interface.

    Existing workflows access the ASE Atoms via ``slab.atoms``.  This
    container lets generic code produce an object with the same API
    without requiring the FAIRChem data package.
    """

    def __init__(self, atoms: Atoms):
        self.atoms = atoms


def coerce_slab_container(slab: SlabContainer | Atoms) -> SlabContainer:
    """Normalize slab-like input to :class:`SlabContainer`.

    Accepts either a pre-wrapped ``SlabContainer`` or a plain ASE ``Atoms``
    object. ``Atoms`` inputs are defensively copied to avoid mutating caller
    state across workflow steps.
    """
    if isinstance(slab, SlabContainer):
        return slab
    if isinstance(slab, Atoms):
        return SlabContainer(slab.copy())
    raise TypeError(
        f"slab must be a SlabContainer or ase.Atoms, got {type(slab).__name__}"
    )


# ---------------------------------------------------------------------------
# Base slab creation
# ---------------------------------------------------------------------------


def create_slab_from_bulk(
    bulk_id: str,
    miller_indices: tuple = (0, 0, 1),
    supercell: tuple = (2, 2, 1),
    results_dir: str = "results",
    calculator=None,
    config: AdsorptionConfig | None = None,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
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
    calculator:
        Optional calculator used when slab relaxation is requested.
    config:
        Optional adsorption config. Used for relaxation defaults.
    relaxation_mode:
        One of ``"none"``, ``"ionic_only"``, ``"cell_only"``, or ``"full"``.
    relaxation_optimizer:
        One of ``"lbfgs"``, ``"bfgs"``, or ``"fire"``.
    relaxation_fmax:
        Force convergence threshold for slab relaxation.
    relaxation_steps:
        Maximum optimisation steps for slab relaxation.
    """
    try:
        from fairchem.data.oc.core import Bulk, Slab
    except ImportError as exc:
        raise _dependency_error_for_slab_import(exc) from exc

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
    if not slabs:
        raise GeometryValidationError(
            "No slabs were generated for bulk_id="
            f"{bulk_id!r}, miller_indices={miller_indices!r}. "
            "Try a different Miller index, bulk source, or supercell."
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
    mode, opt_name, fmax, steps = _resolve_slab_relaxation_settings(
        config,
        relaxation_mode=relaxation_mode,
        relaxation_optimizer=relaxation_optimizer,
        relaxation_fmax=relaxation_fmax,
        relaxation_steps=relaxation_steps,
    )
    if mode != "none":
        logger.info(
            "Relaxing clean slab with mode=%s, optimizer=%s, fmax=%.4f, steps=%d",
            mode,
            opt_name,
            fmax,
            steps,
        )
        slab.atoms = _relax_slab_structure(
            slab.atoms,
            calculator,
            mode=mode,
            optimizer_name=opt_name,
            fmax=fmax,
            steps=steps,
            context="create_slab_from_bulk",
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


def _resolve_slab_relaxation_settings(
    config: AdsorptionConfig | None,
    *,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
) -> tuple[SLAB_RELAXATION_MODE, SLAB_RELAXATION_OPTIMIZER, float, int]:
    """Resolve per-call slab relaxation settings with config fallbacks."""
    resolved_config = config if config is not None else AdsorptionConfig()
    mode = (
        relaxation_mode
        if relaxation_mode is not None
        else resolved_config.slab_relaxation_mode
    )
    optimizer = (
        relaxation_optimizer
        if relaxation_optimizer is not None
        else resolved_config.slab_relaxation_optimizer
    )
    fmax = (
        relaxation_fmax
        if relaxation_fmax is not None
        else resolved_config.slab_relaxation_fmax
    )
    if fmax is None:
        fmax = resolved_config.fmax
    steps = (
        relaxation_steps
        if relaxation_steps is not None
        else resolved_config.slab_relaxation_steps
    )
    return mode, optimizer, fmax, steps


def _relax_slab_structure(
    atoms: Atoms,
    calculator,
    *,
    mode: SLAB_RELAXATION_MODE,
    optimizer_name: SLAB_RELAXATION_OPTIMIZER,
    fmax: float,
    steps: int,
    context: str = "slab",
) -> Atoms:
    """Relax a slab using the selected mode and optimizer preset."""
    if mode == "none":
        return atoms
    if calculator is None:
        raise ValueError(
            f"{context} relaxation mode={mode!r} requires a calculator, got None"
        )

    optimizer_map = {"lbfgs": LBFGS, "bfgs": BFGS, "fire": FIRE}
    opt_cls = optimizer_map[optimizer_name]
    relaxed = atoms.copy()
    relaxed.calc = calculator

    if mode == "ionic_only":
        dyn = opt_cls(relaxed, logfile=None)
    elif mode == "cell_only":
        relaxed.set_constraint(FixAtoms(indices=list(range(len(relaxed)))))
        dyn = opt_cls(UnitCellFilter(relaxed), logfile=None)
    else:  # mode == "full"
        dyn = opt_cls(UnitCellFilter(relaxed), logfile=None)

    try:
        dyn.run(fmax=fmax, steps=steps)
    except (RuntimeError, ValueError) as exc:
        raise OptimizationError(
            f"{context} relaxation failed in mode={mode!r}: {exc}"
        ) from exc
    finally:
        # Keep downstream workflow behaviour unchanged by stripping prep constraints.
        relaxed.set_constraint()

    return relaxed


def substitute_alloy(
    slab: SlabContainer | Atoms,
    host_symbol: str,
    guest_symbol: str,
    guest_fraction: float,
    calculator=None,
    n_variants: int = 5,
    seed: int | None = None,
    relax: bool = True,
    enforce_top_layer_fraction: bool = False,
    top_layer_tolerance: float | None = None,
    config: AdsorptionConfig | None = None,
    results_dir: str = "results",
) -> SlabContainer:
    """Randomly replace *guest_fraction* of *host_symbol* atoms with *guest_symbol*.

    Multiple random variants are generated and scored; the lowest-energy
    variant is kept.  If *relax* is ``True`` the chosen slab is relaxed
    with ``calculator`` before returning.

    When *enforce_top_layer_fraction* is ``True``, each random variant uses a
    constrained substitution pattern so the top surface layer composition
    follows *guest_fraction* as closely as possible (subject to integer site
    counts).
    """
    if config is None:
        config = AdsorptionConfig()

    slab = coerce_slab_container(slab)

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

    top_host_indices: list[int] = []
    subsurface_host_indices: list[int] = []
    n_top_replace = 0
    n_sub_replace = 0
    if enforce_top_layer_fraction:
        tol = (
            top_layer_tolerance
            if top_layer_tolerance is not None
            else config.top_layer_tolerance
        )
        if tol <= 0:
            raise ValueError(
                "top_layer_tolerance must be positive when enforcing top-layer "
                f"composition, got {tol}"
            )

        positions = base.get_positions()
        z_max = float(np.max(positions[:, 2]))
        top_set = set(np.nonzero(positions[:, 2] >= (z_max - tol))[0].tolist())
        top_host_indices = [idx for idx in host_indices if idx in top_set]
        subsurface_host_indices = [idx for idx in host_indices if idx not in top_set]

        n_top_replace = int(round(len(top_host_indices) * guest_fraction))
        n_top_replace = max(0, min(n_top_replace, len(top_host_indices)))
        n_sub_replace = n_replace - n_top_replace

        if n_sub_replace > len(subsurface_host_indices):
            deficit = n_sub_replace - len(subsurface_host_indices)
            n_sub_replace = len(subsurface_host_indices)
            n_top_replace = min(len(top_host_indices), n_top_replace + deficit)
        elif n_sub_replace < 0:
            n_top_replace = max(0, n_top_replace + n_sub_replace)
            n_sub_replace = 0

        assigned = n_top_replace + n_sub_replace
        if assigned != n_replace:
            remaining = n_replace - assigned
            n_top_extra = min(remaining, len(top_host_indices) - n_top_replace)
            n_top_replace += n_top_extra
            remaining -= n_top_extra
            n_sub_extra = min(remaining, len(subsurface_host_indices) - n_sub_replace)
            n_sub_replace += n_sub_extra
            remaining -= n_sub_extra
            if remaining != 0:
                raise GeometryValidationError(
                    "Could not allocate alloy substitutions while enforcing "
                    "top-layer composition constraints"
                )

    for v in range(n_variants):
        if enforce_top_layer_fraction:
            top_choice = (
                rng.choice(top_host_indices, size=n_top_replace, replace=False)
                if n_top_replace > 0
                else np.array([], dtype=int)
            )
            sub_choice = (
                rng.choice(
                    subsurface_host_indices,
                    size=n_sub_replace,
                    replace=False,
                )
                if n_sub_replace > 0
                else np.array([], dtype=int)
            )
            replace_idx = np.concatenate([top_choice, sub_choice]).astype(int)
        else:
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
            logger.info("Relaxing alloy slab geometry...")
            best_atoms.calc = calculator
            dyn = LBFGS(best_atoms, logfile="-")
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
    slab: SlabContainer | Atoms,
    adatom_symbol: str,
    coverage_fraction: float,
    calculator=None,
    n_variants: int = 5,
    adsorption_height: float = 1.8,
    seed: int | None = None,
    results_dir: str = "results",
    config: AdsorptionConfig | None = None,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
) -> SlabContainer:
    """Place *adatom_symbol* atoms at hollow sites above the top layer.

    Uses Delaunay triangulation of the top-layer xy coordinates to
    identify candidate hollow sites.  *coverage_fraction* of the
    available sites are filled.  The lowest-energy variant is kept.
    Optional relaxation presets can be applied to each generated
    adatom variant before energy ranking.
    """
    if config is None:
        config = AdsorptionConfig()
    mode, opt_name, fmax, steps = _resolve_slab_relaxation_settings(
        config,
        relaxation_mode=relaxation_mode,
        relaxation_optimizer=relaxation_optimizer,
        relaxation_fmax=relaxation_fmax,
        relaxation_steps=relaxation_steps,
    )

    slab = coerce_slab_container(slab)

    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError(
            f"coverage_fraction must be between 0 and 1, got {coverage_fraction}"
        )

    if coverage_fraction == 0.0:
        logger.info("coverage_fraction=0; returning unmodified slab")
        return SlabContainer(slab.atoms.copy())
    if mode != "none" and calculator is None:
        raise ValueError(
            f"deposit_adatoms relaxation mode={mode!r} requires a calculator"
        )

    if seed is None:
        seed = config.seed

    from .placement import get_hollow_sites_for_adatoms

    base = slab.atoms.copy()
    positions = base.get_positions()
    z_max = float(np.max(positions[:, 2]))
    top_tol = config.top_layer_tolerance

    top_mask = positions[:, 2] >= (z_max - top_tol)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        raise GeometryValidationError(
            "Cannot identify top surface layer for adatom placement "
            f"(found {len(top_indices)} atoms within {top_tol} A of z_max)"
        )

    candidate_sites = get_hollow_sites_for_adatoms(
        base,
        top_layer_tolerance=top_tol,
        dedup_tolerance=config.hollow_site_dedup_tolerance,
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
            candidate = variant
            if mode != "none":
                candidate = _relax_slab_structure(
                    variant,
                    calculator,
                    mode=mode,
                    optimizer_name=opt_name,
                    fmax=fmax,
                    steps=steps,
                    context=f"deposit_adatoms variant {v}",
                )
            energy = _evaluate_variant_energy(
                candidate, calculator, context=f"Adatom variant {v}"
            )
            if energy < best_energy:
                best_energy = energy
                best_atoms = candidate.copy()
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
    slab: SlabContainer | Atoms,
    conformers: list[Atoms],
    min_separation: float = 8.0,
) -> tuple[SlabContainer, bool]:
    """Resize *slab* in-plane so periodic images are well separated.

    Returns ``(slab, was_resized)`` where *was_resized* is ``True``
    when the cell was expanded.
    """
    slab = coerce_slab_container(slab)

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

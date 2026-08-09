"""Slab construction (bulk + Miller), alloy substitution, adatoms, and ``SlabContainer``."""

import copy
import logging
import math
import os

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import UnitCellFilter
from ase.optimize import BFGS, FIRE, LBFGS

from .._numeric_defaults import MIN_CALCULATOR_CELL_C_ANG
from ..config import (
    SLAB_RELAXATION_MODE,
    SLAB_RELAXATION_OPTIMIZER,
    AdsorptionConfig,
    resolve_adsorption_config,
)
from ..exceptions import (
    DependencyMissingError,
    GeometryValidationError,
    OptimizationError,
)
from ..io_results import _write_clean_xyz
from ..placement._constants import _MEAN_COVALENT_RADIUS_FALLBACK
from ..placement._material import MATERIAL_PBC, material_aware_pbc
from ..placement.geometry import _get_covalent_radius
from ..placement.site_coords import _periodic_image_offsets
from ..placement.site_enumeration import get_hollow_sites_for_adatoms
from .freeze import (
    compute_frozen_indices,
    frozen_indices_from_constraints,
    top_layer_indices_by_height,
)

logger = logging.getLogger(__name__)

DEFAULT_SLAB_TOP_VACUUM_ANG = 15.0
_INVERTED_SLAB_VACUUM_MARGIN_ANG = 1.0
_MIN_CALCULATOR_CELL_C_ANG = MIN_CALCULATOR_CELL_C_ANG
_CELL_DET_EPS = 1e-8


def _cell_determinant(cell: np.ndarray) -> float:
    return float(np.linalg.det(cell))


def _fractional_spans(atoms: Atoms) -> np.ndarray:
    """Return fractional coordinate span along each lattice vector."""
    frac = atoms.get_scaled_positions(wrap=False)
    return np.max(frac, axis=0) - np.min(frac, axis=0)


def _vacuum_margins_ang(atoms: Atoms) -> np.ndarray:
    """Vacuum margin (Å) along each lattice direction from fractional spans."""
    cell = np.array(atoms.get_cell(), dtype=float)
    frac_span = _fractional_spans(atoms)
    lengths = np.array([float(np.linalg.norm(cell[i])) for i in range(3)])
    return lengths * (1.0 - frac_span)


def _dependency_error_for_slab_import(exc: ImportError) -> DependencyMissingError:
    """Map a failed ``fairchem.data.oc`` import to an actionable :class:`DependencyMissingError`."""
    cur: BaseException | None = exc
    while cur is not None:
        if (
            isinstance(cur, ModuleNotFoundError) and cur.name == "pkg_resources"
        ) or "pkg_resources" in str(cur):
            return DependencyMissingError(
                "setuptools",
                "create_slab_from_bulk",
                "Install with: pip install 'setuptools<82' (setuptools 82+ removed pkg_resources)",
            )
        cur = cur.__cause__
    extra = f". Underlying error: {exc!s}" if "fairchem" not in str(exc).lower() else ""
    return DependencyMissingError(
        "fairchem-data-oc",
        "create_slab_from_bulk",
        f"Install with: pip install fairchem-data-oc{extra}",
    )


class SlabContainer:
    """Minimal wrapper that mimics the FAIRChem ``Slab`` interface.

    Existing workflows access the ASE Atoms via ``slab.atoms``.  This
    container lets generic code produce an object with the same API
    without requiring the FAIRChem data package.

    ``finalized`` is set by :func:`~metalsurfer.surface_prep.finalize_substrate`
    after full vacuum/cell/PBC validation so campaign entry points can skip
    repeating that work.
    """

    def __init__(self, atoms: Atoms, *, finalized: bool = False):
        self.atoms = atoms
        self.finalized = finalized


def ensure_slab_z_alignment(
    atoms: Atoms,
    *,
    min_top_vacuum: float = DEFAULT_SLAB_TOP_VACUUM_ANG,
    min_cell_c: float = _MIN_CALCULATOR_CELL_C_ANG,
) -> Atoms:
    """Bottom-anchor a slab and ensure vacuum above the top surface layer.

    Returns a copy of *atoms* shifted so ``min(z) == 0`` and with the c-vector
    extended to at least ``max(z_max + min_top_vacuum, min_cell_c)``.
    """
    aligned = atoms.copy()
    pos = aligned.get_positions()
    z_min = float(np.min(pos[:, 2]))
    z_max = float(np.max(pos[:, 2]))

    cell = np.array(aligned.get_cell(), dtype=float)
    c_len = float(np.linalg.norm(cell[2]))
    if c_len > 0.0:
        vacuum_below = z_min
        vacuum_above = c_len - z_max
        if vacuum_below > vacuum_above + _INVERTED_SLAB_VACUUM_MARGIN_ANG:
            logger.warning(
                "Slab has more vacuum below the substrate (%.1f A) than above "
                "(%.1f A); re-aligning to bottom-anchored layout with vacuum "
                "above max(z)",
                vacuum_below,
                vacuum_above,
            )

    if z_min != 0.0:
        pos = pos.copy()
        pos[:, 2] -= z_min
        # FairChem slabs may carry FixAtoms; bypass so the whole slab shifts.
        aligned.set_positions(pos, apply_constraint=False)
        z_max = float(np.max(aligned.get_positions()[:, 2]))

    target_c = max(min_cell_c, z_max + min_top_vacuum)
    if c_len < target_c:
        if abs(cell[2, 0]) < 1e-6 and abs(cell[2, 1]) < 1e-6:
            cell[2, 2] = target_c
        elif c_len > 0.0:
            cell[2] = cell[2] * (target_c / c_len)
        else:
            cell[2, 2] = target_c
        aligned.set_cell(cell)

    return aligned


def apply_surface_constraints(
    atoms: Atoms,
    *,
    relax_top_layer: bool = False,
    freeze_symbols: list[str] | None = None,
    top_layer_tolerance: float = 0.5,
    material_type: str = "slab",
) -> Atoms:
    """Attach ASE ``FixAtoms`` to *atoms* according to the surface freeze policy.

    Call during substrate preparation **before** passing the structure to
    campaign API entry points. Placement relaxation reads these constraints via
    ``frozen_indices_from_constraints`` only.

    Default ``relax_top_layer=False`` freezes every substrate atom. Set
    ``relax_top_layer=True`` for a material-aware shortcut that leaves the
    exposed surface free (see :func:`~metalsurfer.identify_relaxable_surface_indices`;
    requires the correct *material_type*, default ``"slab"``). For full control,
    attach custom ASE constraints to *atoms* yourself and skip this helper.
    """
    result = atoms.copy()
    frozen = compute_frozen_indices(
        result,
        relax_top_layer=relax_top_layer,
        freeze_symbols=freeze_symbols,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
    )
    if relax_top_layer and not frozen and len(result) > 0:
        logger.warning(
            "Relax_top_layer=True left no atoms frozen (tolerance=%.3f A, "
            "material_type=%s); freezing entire substrate instead",
            top_layer_tolerance,
            material_type,
        )
        frozen = list(range(len(result)))
    result.set_constraint(FixAtoms(indices=frozen))
    return result


def coerce_slab_container(
    slab: SlabContainer | Atoms,
    *,
    copy: bool = False,
) -> SlabContainer:
    """Normalize slab-like input to :class:`SlabContainer` without modification.

    Accepts either a pre-wrapped ``SlabContainer`` or a plain ASE ``Atoms``
    object. Policy:

    - Already a ``SlabContainer``: returned as-is unless ``copy=True``, in which
      case a new container wrapping ``atoms.copy()`` is returned (``finalized``
      flag is preserved).
    - Plain ``Atoms``: always wrapped in a new container around ``atoms.copy()``
      so caller state is not mutated across workflow steps.

    Geometry alignment, sizing, and constraints must be applied via the prep
    helpers before calling campaign APIs.
    """
    if isinstance(slab, SlabContainer):
        if not copy:
            return slab
        return SlabContainer(slab.atoms.copy(), finalized=slab.finalized)
    if isinstance(slab, Atoms):
        return SlabContainer(slab.copy())
    raise TypeError(
        f"slab must be a SlabContainer or ase.Atoms, got {type(slab).__name__}"
    )


def _warn_missing_fixatoms(slab: Atoms) -> None:
    if frozen_indices_from_constraints(slab):
        return
    logger.warning(
        "Substrate has no FixAtoms constraints; all %d substrate atoms will "
        "relax during adsorption. Call apply_surface_constraints during "
        "substrate preparation to freeze atoms",
        len(slab),
    )


def validate_substrate(
    slab: Atoms,
    *,
    material_type: str,
    config: AdsorptionConfig | None = None,
    conformers: list[Atoms] | None = None,
    require_bottom_anchor: bool = True,
) -> None:
    """Validate that *slab* is ready for campaign API entry points.

    Raises :class:`~metalsurfer.exceptions.GeometryValidationError` when the
    substrate is misaligned, undersized, or incompatible with *material_type*.

    When *conformers* is provided, also checks in-plane image separation
    (prefer calling :func:`validate_substrate_conformer_sizing` from screening
    prep after resize instead of bundling it here).
    """
    cfg = resolve_adsorption_config(config)
    pos = slab.get_positions()
    if len(pos) == 0:
        raise GeometryValidationError("Substrate has no atoms")

    z_min = float(np.min(pos[:, 2]))
    z_max = float(np.max(pos[:, 2]))
    if material_type == "slab" and require_bottom_anchor and abs(z_min) > 0.05:
        raise GeometryValidationError(
            f"Substrate is not bottom-anchored (min(z)={z_min:.3f} A; expected ~0). "
            "Call ensure_slab_z_alignment during substrate preparation."
        )

    cell = np.array(slab.get_cell(), dtype=float)
    c_len = float(np.linalg.norm(cell[2]))
    pbc = np.array(slab.get_pbc(), dtype=bool)
    if material_type not in MATERIAL_PBC:
        raise GeometryValidationError(
            f"Unknown material_type={material_type!r}; "
            "expected 'slab', 'porous', or 'nanoparticle'"
        )
    expected_pbc = material_aware_pbc(material_type)
    if not np.array_equal(pbc, expected_pbc):
        raise GeometryValidationError(
            f"Substrate PBC {pbc.tolist()} is inconsistent with "
            f"material_type={material_type!r} (expected {expected_pbc}). "
            "Set PBC on the ASE Atoms object during preparation."
        )

    cell_det = _cell_determinant(cell)
    if material_type in ("slab", "porous") and abs(cell_det) <= _CELL_DET_EPS:
        raise GeometryValidationError(
            f"Substrate cell is degenerate (det={cell_det:.2e}). "
            "Provide a non-zero periodic cell during preparation."
        )

    if material_type == "slab":
        a_len = float(np.linalg.norm(cell[0]))
        b_len = float(np.linalg.norm(cell[1]))
        if a_len <= _CELL_DET_EPS or b_len <= _CELL_DET_EPS:
            raise GeometryValidationError(
                f"Slab in-plane lattice vectors must be positive (a={a_len:.3f} A, "
                f"b={b_len:.3f} A)."
            )
        if c_len <= 0.0:
            raise GeometryValidationError("Slab cell c-vector length must be positive")
        vacuum_above = c_len - z_max
        if vacuum_above < DEFAULT_SLAB_TOP_VACUUM_ANG * 0.5:
            raise GeometryValidationError(
                f"Insufficient vacuum above the top surface layer "
                f"({vacuum_above:.1f} A above z_max). "
                f"Call ensure_slab_z_alignment with at least "
                f"{DEFAULT_SLAB_TOP_VACUUM_ANG:.0f} A top vacuum."
            )
        if c_len < _MIN_CALCULATOR_CELL_C_ANG:
            raise GeometryValidationError(
                f"Slab c-vector ({c_len:.1f} A) is below the minimum "
                f"{_MIN_CALCULATOR_CELL_C_ANG:.0f} A required by the calculator. "
                "Extend vacuum via ensure_slab_z_alignment during preparation."
            )

    if material_type == "porous" and c_len < _MIN_CALCULATOR_CELL_C_ANG:
        raise GeometryValidationError(
            f"Porous cell c-vector ({c_len:.1f} A) is below the minimum "
            f"{_MIN_CALCULATOR_CELL_C_ANG:.0f} A required by the calculator."
        )

    if material_type == "nanoparticle" and abs(cell_det) > _CELL_DET_EPS:
        margins = _vacuum_margins_ang(slab)
        min_margin = float(np.min(margins))
        if min_margin < cfg.min_pbc_image_separation:
            raise GeometryValidationError(
                f"Nanoparticle simulation cell is too tight "
                f"(minimum vacuum margin {min_margin:.1f} A along a lattice "
                f"direction; need at least {cfg.min_pbc_image_separation:.1f} A). "
                "Build a larger orthorhombic cell around the cluster before "
                "calling campaign APIs."
            )

    _warn_missing_fixatoms(slab)

    if conformers:
        validate_substrate_conformer_sizing(
            slab,
            conformers=conformers,
            config=cfg,
        )


def validate_substrate_conformer_sizing(
    slab: Atoms,
    *,
    conformers: list[Atoms],
    config: AdsorptionConfig | None = None,
) -> None:
    """Ensure in-plane image separation is adequate for *conformers*."""
    cfg = resolve_adsorption_config(config)
    cell = np.array(slab.get_cell(), dtype=float)
    diameter = _molecule_diameter(conformers)
    nx, ny = compute_minimum_supercell(
        cell,
        diameter,
        cfg.min_pbc_image_separation,
    )
    if nx > 1 or ny > 1:
        raise GeometryValidationError(
            f"In-plane periodic image separation is too small for adsorbate "
            f"diameter {diameter:.1f} A (needs repeat at least ({nx}, {ny}, 1)). "
            "Call auto_resize_substrate_for_molecule during substrate preparation."
        )


def _check_api_substrate_invariants(slab: Atoms) -> None:
    """Lightweight campaign-entry checks (constraints present / FixAtoms type)."""
    if len(slab) == 0:
        raise GeometryValidationError("Substrate has no atoms")
    _warn_missing_fixatoms(slab)


def accept_substrate_for_api(
    slab: SlabContainer | Atoms,
    *,
    config: AdsorptionConfig,
) -> SlabContainer:
    """Wrap and return a substrate ready for campaign APIs.

    When *slab* is already a finalized :class:`SlabContainer` (from
    :func:`~metalsurfer.surface_prep.finalize_substrate`), only API invariants
    are checked — full vacuum/cell validation is not repeated. Plain ``Atoms``
    or non-finalized containers still run :func:`validate_substrate`.
    """
    container = coerce_slab_container(slab)
    if container.finalized:
        _check_api_substrate_invariants(container.atoms)
    else:
        validate_substrate(
            container.atoms,
            material_type=config.material_type,
            config=config,
            conformers=None,
            require_bottom_anchor=False,
        )
    return container


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

    Applies :func:`ensure_slab_z_alignment`.

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
        "Creating slab from %s, Miller %s, supercell %s",
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

    slab.atoms = ensure_slab_z_alignment(slab.atoms)

    cfg = resolve_adsorption_config(config)
    _save_reference_slab_artifacts(
        slab.atoms,
        results_dir=results_dir,
        stem="clean_slab",
        write_vasp=cfg.write_vasp_inputs,
    )
    logger.info("Saved clean slab reference files to %s", results_dir)

    return SlabContainer(slab.atoms)


def create_slab_from_atoms(
    atoms: Atoms,
    *,
    material_type: str = "slab",
    align: bool = True,
) -> SlabContainer:
    """Wrap an existing ASE ``Atoms`` object into a :class:`SlabContainer`.

    When *material_type* is ``"slab"`` and *align* is true, applies
    :func:`ensure_slab_z_alignment`. Pass ``align=False`` for pre-aligned
    DFT or experimental structures.
    """
    container = coerce_slab_container(atoms)
    if material_type == "slab" and align:
        container.atoms = ensure_slab_z_alignment(container.atoms)
    return container


# ---------------------------------------------------------------------------
# Alloy substitution
# ---------------------------------------------------------------------------


def _consider_variant(
    candidate: Atoms,
    *,
    calculator,
    best_energy: float,
    best_atoms: Atoms | None,
    context: str,
    keep_last_without_calculator: bool = False,
) -> tuple[float, Atoms | None]:
    """Update best variant when *candidate* is better (or first without calculator)."""
    if calculator is None:
        # No energy ranking available. By default keep the first variant so the
        # result is deterministic regardless of seed; ``deposit_adatoms`` opts into
        # last-wins so its (seed-dependent) placement differs across runs.
        if best_atoms is None or keep_last_without_calculator:
            return best_energy, candidate.copy()
        return best_energy, best_atoms
    try:
        candidate.calc = calculator
        energy = float(candidate.get_potential_energy())
        if energy < best_energy:
            return energy, candidate.copy()
    except (RuntimeError, ValueError) as exc:
        if context:
            logger.warning("%s failed: %s", context, exc)
    return best_energy, best_atoms


def _save_reference_slab_artifacts(
    atoms: Atoms,
    *,
    results_dir: str,
    stem: str,
    write_vasp: bool,
) -> None:
    """Write ``{stem}.xyz`` and optional VASP POSCAR under *results_dir*."""
    os.makedirs(results_dir, exist_ok=True)
    _write_clean_xyz(atoms, f"{results_dir}/{stem}.xyz")
    if write_vasp:
        atoms.write(
            f"{results_dir}/{stem}_POSCAR",
            format="vasp",
            vasp5=True,
            direct=True,
        )


def _resolve_slab_relaxation_settings(
    config: AdsorptionConfig | None,
    *,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
) -> tuple[SLAB_RELAXATION_MODE, SLAB_RELAXATION_OPTIMIZER, float, int]:
    """Resolve per-call slab relaxation settings with config fallbacks."""
    cfg = config or AdsorptionConfig()
    mode = relaxation_mode if relaxation_mode is not None else cfg.slab_relaxation_mode
    opt = (
        relaxation_optimizer
        if relaxation_optimizer is not None
        else cfg.slab_relaxation_optimizer
    )
    fmax = (
        relaxation_fmax
        if relaxation_fmax is not None
        else (cfg.slab_relaxation_fmax or cfg.fmax)
    )
    steps = (
        relaxation_steps if relaxation_steps is not None else cfg.slab_relaxation_steps
    )
    return mode, opt, fmax, steps


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
    # Snapshot the caller's constraints so we can restore them on the relaxed
    # copy afterwards. ASE's relaxations (esp. ``cell_only``) install their own
    # constraints and the finally block would otherwise leave the structure
    # unconstrained, which breaks downstream frozen-substrate drift checks.
    caller_constraints = copy.deepcopy(atoms.constraints)
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
        # Restore the caller's constraints (replacing any installed by the
        # relaxation mode) rather than leaving the structure unconstrained.
        relaxed.set_constraint(*copy.deepcopy(caller_constraints))

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
            "Guest_fraction=%.2f yields 0 replacements; returning base slab",
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
        cell = np.asarray(base.get_cell(), dtype=float)
        top_set = set(top_layer_indices_by_height(positions, cell, float(tol)))
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

        best_energy, best_atoms = _consider_variant(
            variant,
            calculator=calculator,
            best_energy=best_energy,
            best_atoms=best_atoms,
            context=f"Variant {v}",
        )

    if best_atoms is None:
        raise GeometryValidationError(
            "Failed to generate any valid alloy slab variants"
        )

    if relax and calculator is not None:
        try:
            logger.info("Relaxing alloy slab geometry")
            best_atoms.calc = calculator
            dyn = LBFGS(best_atoms, logfile="-")
            dyn.run(fmax=config.fmax)
            logger.info(
                "Post-relax slab energy: %.4f eV",
                best_atoms.get_potential_energy(),
            )
        except (RuntimeError, ValueError) as exc:
            raise OptimizationError(f"Alloy slab relaxation failed: {exc}") from exc

    cfg = resolve_adsorption_config(config)
    label = f"{host_symbol}_{guest_symbol}_{int(guest_fraction * 100)}"
    _save_reference_slab_artifacts(
        best_atoms,
        results_dir=results_dir,
        stem=f"clean_{label}_slab",
        write_vasp=cfg.write_vasp_inputs,
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
    min_adatom_separation: float | None = None,
    seed: int | None = None,
    results_dir: str = "results",
    config: AdsorptionConfig | None = None,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
) -> SlabContainer:
    """Place *adatom_symbol* atoms at hollow sites above the top layer.

    Candidate hollow/pore sites come from unified site detection. Placement
    height is ``site.xyz + site.normal * adsorption_height`` (normal-aware).
    *coverage_fraction* of the available sites are filled. The lowest-energy
    variant is kept. Optional relaxation presets can be applied to each
    generated adatom variant before energy ranking.
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
        logger.info("Coverage_fraction=0; returning unmodified slab")
        return SlabContainer(slab.atoms.copy())
    if mode != "none" and calculator is None:
        raise ValueError(
            f"deposit_adatoms relaxation mode={mode!r} requires a calculator"
        )

    if seed is None:
        seed = config.seed

    base = slab.atoms.copy()
    positions = base.get_positions()
    cell = np.asarray(base.get_cell(), dtype=float)
    top_tol = config.top_layer_tolerance

    top_indices = np.asarray(
        top_layer_indices_by_height(positions, cell, float(top_tol)), dtype=int
    )
    if len(top_indices) < 3:
        raise GeometryValidationError(
            "Cannot identify top surface layer for adatom placement "
            f"(found {len(top_indices)} atoms within {top_tol} A of max height "
            "along the slab normal)"
        )

    candidate_sites = get_hollow_sites_for_adatoms(
        base,
        top_layer_tolerance=top_tol,
        dedup_tolerance=config.hollow_site_dedup_tolerance,
        material_type=config.material_type,
    )

    if not candidate_sites:
        raise GeometryValidationError(
            "No candidate hollow sites found for adatom placement"
        )

    n_place = max(
        1, int(round(coverage_fraction * len(candidate_sites)))
    )  # >=1 guaranteed since coverage_fraction > 0

    if min_adatom_separation is None:
        sym_r = _get_covalent_radius(adatom_symbol)
        sym_r = sym_r if sym_r is not None else _MEAN_COVALENT_RADIUS_FALLBACK
        min_adatom_separation = 2.0 * float(sym_r)
    min_adatom_separation = float(min_adatom_separation)

    rng = np.random.RandomState(seed)
    best_energy = float("inf")
    best_atoms = None

    pbc = list(material_aware_pbc(config.material_type))
    cell = np.asarray(base.get_cell(), dtype=float)
    image_offsets = _periodic_image_offsets(
        cell, np.asarray(pbc, dtype=bool), min_adatom_separation
    )

    for v in range(n_variants):
        # Greedily accept candidate hollow sites whose adatom position keeps at
        # least ``min_adatom_separation`` from every already-accepted adatom
        # (under MIC). Random ``rng.choice`` can pack adatoms 0.78 A apart.
        accepted: list[np.ndarray] = []
        order = rng.permutation(len(candidate_sites))
        for i in order:
            site = candidate_sites[int(i)]
            normal = np.asarray(site.normal, dtype=float)
            nrm = float(np.linalg.norm(normal))
            normal = normal / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
            pos = site.xyz + float(adsorption_height) * normal
            too_close = False
            for acc in accepted:
                if (
                    np.min([np.linalg.norm(pos - (acc + off)) for off in image_offsets])
                    < min_adatom_separation
                ):
                    too_close = True
                    break
            if too_close:
                continue
            accepted.append(pos)
            if len(accepted) >= n_place:
                break
        if len(accepted) < n_place:
            logger.warning(
                "deposit_adatoms: achieved coverage %d/%d sites at min separation "
                "%.2f A (requested %d); continuing with the achievable count",
                len(accepted),
                n_place,
                min_adatom_separation,
                n_place,
            )
        variant = base.copy()
        adatoms = Atoms(
            symbols=[adatom_symbol] * len(accepted),
            positions=np.asarray(accepted, dtype=float),
        )
        adatoms.set_cell(variant.get_cell())
        adatoms.set_pbc(variant.get_pbc())
        variant += adatoms

        candidate = variant
        if calculator is not None and mode != "none":
            candidate = _relax_slab_structure(
                variant,
                calculator,
                mode=mode,
                optimizer_name=opt_name,
                fmax=fmax,
                steps=steps,
                context=f"deposit_adatoms variant {v}",
            )
        best_energy, best_atoms = _consider_variant(
            candidate,
            calculator=calculator,
            best_energy=best_energy,
            best_atoms=best_atoms,
            context=f"Adatom variant {v}",
            keep_last_without_calculator=True,
        )

    if best_atoms is None:
        raise GeometryValidationError(
            "Failed to generate any valid adatom-deposited slab"
        )

    cfg = resolve_adsorption_config(config)
    pct = int(round(coverage_fraction * 100))
    label = f"{adatom_symbol}{pct}"
    _save_reference_slab_artifacts(
        best_atoms,
        results_dir=results_dir,
        stem=f"clean_slab_{label}",
        write_vasp=cfg.write_vasp_inputs,
    )
    logger.info(
        "Created adatom-deposited slab (%s, %.0f%%): E=%.4f eV",
        adatom_symbol,
        coverage_fraction * 100,
        best_energy,
    )

    # Base-slab FixAtoms indices become stale after atoms are appended; refresh
    # so adatoms are frozen with the rest of the substrate (default freeze-all).
    best_atoms = apply_surface_constraints(best_atoms)
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


def auto_resize_substrate_for_molecule(
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

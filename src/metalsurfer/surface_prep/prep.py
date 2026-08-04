"""Orchestration helpers for campaign-ready substrate preparation."""

from __future__ import annotations

import ase.io
import numpy as np
from ase import Atoms

from .. import optimization
from ..config import (
    SLAB_RELAXATION_MODE,
    SLAB_RELAXATION_OPTIMIZER,
    AdsorptionConfig,
)
from ..placement._material import material_aware_pbc
from ._surfaces import (
    SlabContainer,
    _relax_slab_structure,
    _resolve_slab_relaxation_settings,
    apply_surface_constraints,
    auto_resize_substrate_for_molecule,
    coerce_slab_container,
    create_slab_from_atoms,
    create_slab_from_bulk,
    deposit_adatoms,
    ensure_slab_z_alignment,
    substitute_alloy,
    validate_substrate,
)

__all__ = [
    "apply_material_pbc",
    "finalize_substrate",
    "prepare_substrate",
    "relax_substrate",
    "resize_substrate_for_molecule",
]


def _anchor_atoms_bottom(atoms: Atoms) -> Atoms:
    """Translate *atoms* so the lowest atom sits at z = 0."""
    result = atoms.copy()
    z_min = float(np.min(result.get_positions()[:, 2]))
    if abs(z_min) > 1e-6:
        result.translate((0.0, 0.0, -z_min))
    return result


def apply_material_pbc(atoms: Atoms, material_type: str) -> None:
    """Set ASE boundary conditions expected for *material_type*.

    Slabs use ``[True, True, False]``, porous frameworks ``[True, True, True]``,
    and nanoparticles ``[False, False, False]``. Uses the shared
    :func:`~metalsurfer.placement._material.material_aware_pbc` map.
    """
    atoms.set_pbc(material_aware_pbc(material_type))


def relax_substrate(
    slab: SlabContainer | Atoms,
    calculator,
    config: AdsorptionConfig | None = None,
    *,
    relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    relaxation_fmax: float | None = None,
    relaxation_steps: int | None = None,
    context: str = "relax_substrate",
) -> SlabContainer:
    """Equilibrate a substrate with ASE using prep-time relaxation presets.

    Knobs mirror :class:`~metalsurfer.AdsorptionConfig` ``slab_relaxation_*``
    fields. Explicit arguments override *config* when provided.
    """
    cfg = config if config is not None else AdsorptionConfig()
    container = coerce_slab_container(slab, copy=True)
    mode, opt_name, fmax, steps = _resolve_slab_relaxation_settings(
        cfg,
        relaxation_mode=relaxation_mode,
        relaxation_optimizer=relaxation_optimizer,
        relaxation_fmax=relaxation_fmax,
        relaxation_steps=relaxation_steps,
    )
    if mode == "none":
        return container
    container.atoms = _relax_slab_structure(
        container.atoms,
        calculator,
        mode=mode,
        optimizer_name=opt_name,
        fmax=fmax,
        steps=steps,
        context=context,
    )
    return container


def finalize_substrate(
    slab: SlabContainer | Atoms,
    config: AdsorptionConfig | None = None,
    *,
    conformers: list[Atoms] | None = None,
    align: bool | None = None,
    require_bottom_anchor: bool | None = None,
    relax_top_layer: bool = False,
    freeze_symbols: list[str] | None = None,
    top_layer_tolerance: float = 0.5,
) -> SlabContainer:
    """Apply PBC, freeze constraints, and validate a substrate for campaign APIs.

    Freeze policy is written to ASE ``FixAtoms`` via
    :func:`~metalsurfer.surface_prep.apply_surface_constraints`. Campaign APIs
    read those constraints only.

    When ``relax_top_layer=True``, pass *config* with the correct
    ``material_type`` — freeze geometry is chosen from
    :attr:`~metalsurfer.AdsorptionConfig.material_type` (defaults to ``"slab"``
    when *config* is omitted). Use the same *config* in campaign APIs.

    This is the last step of substrate preparation. It does **not** build slabs
    from bulk, resize in-plane cells, or deposit adatoms — use
    :func:`prepare_substrate` or the lower-level helpers first, then call this
    after custom modification steps when needed.
    """
    cfg = config if config is not None else AdsorptionConfig()
    container = coerce_slab_container(slab, copy=True)
    material_type = cfg.material_type
    should_align = align if align is not None else material_type == "slab"
    check_bottom_anchor = (
        should_align if require_bottom_anchor is None else require_bottom_anchor
    )
    if material_type == "slab" and should_align:
        container.atoms = ensure_slab_z_alignment(container.atoms)
    apply_material_pbc(container.atoms, material_type)
    container.atoms = apply_surface_constraints(
        container.atoms,
        relax_top_layer=relax_top_layer,
        freeze_symbols=freeze_symbols,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
    )
    validate_substrate(
        container.atoms,
        material_type=material_type,
        config=cfg,
        conformers=conformers,
        require_bottom_anchor=check_bottom_anchor,
    )
    container.finalized = True
    return container


def _resolved_relaxation_mode(
    config: AdsorptionConfig,
    override: SLAB_RELAXATION_MODE | None,
) -> SLAB_RELAXATION_MODE:
    if override is not None:
        return override
    return config.slab_relaxation_mode


def prepare_substrate(
    *,
    bulk_id: str | None = None,
    miller_indices: tuple[int, int, int] = (0, 0, 1),
    supercell: tuple[int, int, int] = (2, 2, 1),
    slab_file: str | None = None,
    slab: SlabContainer | Atoms | None = None,
    results_dir: str = "results_manual",
    alloy_host: str | None = None,
    alloy_guest: str | None = None,
    alloy_fraction: float = 0.0,
    enforce_top_layer_fraction: bool = False,
    adatom_symbol: str | None = None,
    adatom_coverage: float = 0.0,
    config: AdsorptionConfig | None = None,
    align: bool | None = None,
    slab_relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    slab_relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    slab_relaxation_fmax: float | None = None,
    slab_relaxation_steps: int | None = None,
    adatom_relaxation_mode: SLAB_RELAXATION_MODE | None = None,
    adatom_relaxation_optimizer: SLAB_RELAXATION_OPTIMIZER | None = None,
    adatom_relaxation_fmax: float | None = None,
    adatom_relaxation_steps: int | None = None,
    relax_top_layer: bool = False,
    freeze_symbols: list[str] | None = None,
    top_layer_tolerance: float = 0.5,
) -> SlabContainer:
    """Build or load a substrate, optionally modify it, and finalize for campaigns.

    By default, equilibrates substrate **ionic positions** with ASE/MLIP
    (``slab_relaxation_mode="ionic_only"`` on *config*) and attaches ASE
    ``FixAtoms`` for adsorption (default ``relax_top_layer=False``: freeze every
    substrate atom during placement relaxation). Campaign APIs read those
    constraints from the returned substrate only.

    Returns a campaign-ready substrate with material-appropriate PBC, validation
    passed, and an optimized reference geometry for ``E(slab)``.

    Provide exactly one of *bulk_id*, *slab_file*, or *slab* to supply the
    starting structure. When *slab* is given, bulk/file loading is skipped and
    only alloy/adatom modification steps are applied before finalization.

    Set ``slab_relaxation_mode="none"`` to skip prep equilibration (experimental
    geometries only). Set ``relax_top_layer=True`` to leave the exposed surface
    free during placement optimization (behaviour depends on
    ``AdsorptionConfig.material_type``; see
    :func:`~metalsurfer.identify_relaxable_surface_indices`). Attach custom ASE
    constraints yourself when you need finer control than this shortcut.

    Prep-time relaxation knobs mirror :class:`~metalsurfer.AdsorptionConfig`
    ``slab_relaxation_*`` fields. Explicit ``slab_relaxation_*`` and
    ``adatom_relaxation_*`` arguments override *config* for each stage.

    MLIP model and device come from *config* (``model_name``, ``device``).
    """
    sources = [bulk_id is not None, slab_file is not None, slab is not None]
    if sum(sources) != 1:
        raise ValueError(
            "Exactly one of 'bulk_id', 'slab_file', or 'slab' must be provided"
        )

    cfg = config if config is not None else AdsorptionConfig()
    material_type = cfg.material_type
    should_align = align if align is not None else material_type == "slab"
    from_loaded = slab is not None or slab_file is not None

    slab_relax_mode = _resolved_relaxation_mode(cfg, slab_relaxation_mode)
    needs_calculator = (
        (alloy_guest and alloy_fraction > 0)
        or (adatom_symbol and adatom_coverage > 0)
        or slab_relax_mode != "none"
    )

    calculator = None
    if needs_calculator:
        calculator, _ = optimization.setup_single_model(cfg.model_name, cfg.device)

    if slab is not None:
        slab_container = coerce_slab_container(slab, copy=True)
    elif slab_file is not None:
        loaded = ase.io.read(slab_file)
        if isinstance(loaded, list):
            if len(loaded) != 1:
                raise ValueError(
                    f"slab_file must contain a single structure, got {len(loaded)}"
                )
            atoms = loaded[0]
        else:
            atoms = loaded
        if not isinstance(atoms, Atoms):
            raise TypeError(f"slab_file did not yield ASE Atoms, got {type(atoms)!r}")
        slab_container = create_slab_from_atoms(
            atoms,
            material_type=material_type,
            align=should_align,
        )
    else:
        if bulk_id is None:
            raise ValueError(
                "bulk_id is required when neither slab_file nor slab is provided"
            )
        slab_container = create_slab_from_bulk(
            bulk_id=bulk_id,
            miller_indices=miller_indices,
            supercell=supercell,
            results_dir=results_dir,
            calculator=calculator,
            config=cfg,
            relaxation_mode=slab_relaxation_mode,
            relaxation_optimizer=slab_relaxation_optimizer,
            relaxation_fmax=slab_relaxation_fmax,
            relaxation_steps=slab_relaxation_steps,
        )

    if alloy_guest and alloy_fraction > 0:
        host = alloy_host
        if host is None:
            host = sorted(set(slab_container.atoms.get_chemical_symbols()))[0]
        slab_container = substitute_alloy(
            slab_container,
            host_symbol=host,
            guest_symbol=alloy_guest,
            guest_fraction=alloy_fraction,
            calculator=calculator,
            enforce_top_layer_fraction=enforce_top_layer_fraction,
            config=cfg,
            results_dir=results_dir,
        )

    if adatom_symbol and adatom_coverage > 0:
        slab_container = deposit_adatoms(
            slab_container,
            adatom_symbol=adatom_symbol,
            coverage_fraction=adatom_coverage,
            calculator=calculator,
            config=cfg,
            results_dir=results_dir,
            relaxation_mode=adatom_relaxation_mode,
            relaxation_optimizer=adatom_relaxation_optimizer,
            relaxation_fmax=adatom_relaxation_fmax,
            relaxation_steps=adatom_relaxation_steps,
        )

    if from_loaded and slab_relax_mode != "none":
        slab_container = relax_substrate(
            slab_container,
            calculator,
            cfg,
            relaxation_mode=slab_relaxation_mode,
            relaxation_optimizer=slab_relaxation_optimizer,
            relaxation_fmax=slab_relaxation_fmax,
            relaxation_steps=slab_relaxation_steps,
            context="prepare_substrate",
        )

    if material_type == "slab" and should_align:
        slab_container.atoms = ensure_slab_z_alignment(slab_container.atoms)
    elif material_type == "nanoparticle":
        slab_container.atoms = _anchor_atoms_bottom(slab_container.atoms)

    return finalize_substrate(
        slab_container,
        cfg,
        align=False,
        require_bottom_anchor=should_align,
        relax_top_layer=relax_top_layer,
        freeze_symbols=freeze_symbols,
        top_layer_tolerance=top_layer_tolerance,
    )


def resize_substrate_for_molecule(
    slab: SlabContainer | Atoms,
    conformers: list[Atoms],
    config: AdsorptionConfig | None = None,
    *,
    relax_top_layer: bool = False,
    freeze_symbols: list[str] | None = None,
    top_layer_tolerance: float = 0.5,
) -> SlabContainer:
    """Expand *slab* in-plane when conformers require larger image separation.

    Re-applies material PBC and freeze constraints after resizing. Call after
    :func:`prepare_substrate` and conformer generation, before campaign APIs.
    """
    cfg = config if config is not None else AdsorptionConfig()
    resized, was_resized = auto_resize_substrate_for_molecule(
        slab,
        conformers,
        cfg.min_pbc_image_separation,
    )
    if not was_resized:
        return coerce_slab_container(resized)
    return finalize_substrate(
        resized,
        cfg,
        conformers=conformers,
        align=False,
        relax_top_layer=relax_top_layer,
        freeze_symbols=freeze_symbols,
        top_layer_tolerance=top_layer_tolerance,
    )

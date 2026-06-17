"""Slab construction and modification; re-exports helpers and ``prepare_slab``."""

from __future__ import annotations

from typing import Literal

import ase.io
from ase import Atoms

from .config import AdsorptionConfig
from .surfaces import (
    SlabContainer,
    auto_resize_slab_for_molecule,
    coerce_slab_container,
    compute_minimum_supercell,
    create_slab_from_atoms,
    create_slab_from_bulk,
    deposit_adatoms,
    substitute_alloy,
)

__all__ = [
    "SlabContainer",
    "create_slab_from_bulk",
    "create_slab_from_atoms",
    "prepare_slab",
    "substitute_alloy",
    "deposit_adatoms",
    "auto_resize_slab_for_molecule",
    "compute_minimum_supercell",
]


def prepare_slab(
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
    model_name: str = "uma-s-1p1",
    device: str = "cuda",
    config: AdsorptionConfig | None = None,
    create_relaxation_mode: Literal["none", "ionic_only", "cell_only", "full"]
    | None = None,
    create_relaxation_optimizer: Literal["lbfgs", "bfgs", "fire"] | None = None,
    create_relaxation_fmax: float | None = None,
    create_relaxation_steps: int | None = None,
    adatom_relaxation_mode: Literal["none", "ionic_only", "cell_only", "full"]
    | None = None,
    adatom_relaxation_optimizer: Literal["lbfgs", "bfgs", "fire"] | None = None,
    adatom_relaxation_fmax: float | None = None,
    adatom_relaxation_steps: int | None = None,
) -> SlabContainer:
    """Build or load a slab, then optionally alloy-substitute and/or deposit adatoms.

    Applies :func:`~metalsurfer.surfaces.ensure_slab_z_alignment`.

    Provide exactly one of *bulk_id*, *slab_file*, or *slab* to supply the
    starting structure.  When *slab* is given, bulk/file loading is skipped and
    only alloy/adatom modification steps are applied.

    Parameters
    ----------
    bulk_id:
        Materials Project structure ID (e.g. ``"mp-33"``).  Mutually
        exclusive with *slab_file* and *slab*.
    miller_indices:
        Miller indices ``(h, k, l)`` for the surface orientation.
    supercell:
        Supercell repetition ``(nx, ny, nz)``.
    slab_file:
        Path to a pre-built POSCAR or XYZ file to load instead of querying
        the Materials Project.  Mutually exclusive with *bulk_id* and *slab*.
    slab:
        Existing :class:`~metalsurfer.surfaces.SlabContainer` or ASE
        :class:`ase.Atoms` to use as the starting structure.  Mutually
        exclusive with *bulk_id* and *slab_file*.
    results_dir:
        Directory used to save slab structure files during construction.
    alloy_host:
        Element symbol of the host atom type to partially replace.  When
        ``None`` the majority element is used automatically.
    alloy_guest:
        Element symbol of the substituted (guest) atom.
    alloy_fraction:
        Fraction of *alloy_host* sites to replace with *alloy_guest*
        (0–1).  Alloy substitution is skipped when this is 0.
    enforce_top_layer_fraction:
        When true, constrain alloy substitution so the top surface layer
        composition follows *alloy_fraction* as closely as possible.
    adatom_symbol:
        Element symbol of adatom to deposit onto the surface.
    adatom_coverage:
        Surface coverage fraction for adatom deposition (0–1).  Deposition
        is skipped when this is 0.
    model_name:
        MLIP model name used for energy-ranking alloy/adatom configurations.
        Only loaded when alloy substitution or adatom deposition is requested.
    device:
        Compute device (``"cuda"`` or ``"cpu"``) for the MLIP calculator.
    config:
        Optional :class:`~metalsurfer.config.AdsorptionConfig` forwarded to
        :func:`substitute_alloy` and :func:`deposit_adatoms`.
    create_relaxation_mode, create_relaxation_optimizer, create_relaxation_fmax, create_relaxation_steps:
        Optional relaxation controls forwarded to :func:`create_slab_from_bulk`.
    adatom_relaxation_mode, adatom_relaxation_optimizer, adatom_relaxation_fmax, adatom_relaxation_steps:
        Optional relaxation controls forwarded to :func:`deposit_adatoms`.

    Returns
    -------
    SlabContainer
        Prepared slab ready to be passed to :func:`~metalsurfer.campaigns.run_adsorption`.
    """
    sources = [bulk_id is not None, slab_file is not None, slab is not None]
    if sum(sources) != 1:
        raise ValueError(
            "Exactly one of 'bulk_id', 'slab_file', or 'slab' must be provided"
        )

    material_type = (
        config.material_type if config is not None else AdsorptionConfig().material_type
    )
    preserve_slab_frame = (
        config.preserve_slab_frame if config is not None else False
    )

    requested_create_relax_mode = create_relaxation_mode
    if requested_create_relax_mode is None and config is not None:
        requested_create_relax_mode = config.slab_relaxation_mode
    needs_calculator = (alloy_guest and alloy_fraction > 0) or (
        adatom_symbol and adatom_coverage > 0
    )
    if (
        bulk_id is not None
        and slab_file is None
        and requested_create_relax_mode is not None
        and requested_create_relax_mode != "none"
    ):
        needs_calculator = True
    calculator = None
    if needs_calculator:
        from .optimization import setup_single_model

        calculator, _ = setup_single_model(model_name, device)

    if slab is not None:
        slab_container = coerce_slab_container(
            slab,
            material_type=material_type,
            preserve_slab_frame=preserve_slab_frame,
        )
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
            preserve_slab_frame=preserve_slab_frame,
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
            config=config,
            relaxation_mode=create_relaxation_mode,
            relaxation_optimizer=create_relaxation_optimizer,
            relaxation_fmax=create_relaxation_fmax,
            relaxation_steps=create_relaxation_steps,
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
            config=config,
            results_dir=results_dir,
        )

    if adatom_symbol and adatom_coverage > 0:
        slab_container = deposit_adatoms(
            slab_container,
            adatom_symbol=adatom_symbol,
            coverage_fraction=adatom_coverage,
            calculator=calculator,
            config=config,
            results_dir=results_dir,
            relaxation_mode=adatom_relaxation_mode,
            relaxation_optimizer=adatom_relaxation_optimizer,
            relaxation_fmax=adatom_relaxation_fmax,
            relaxation_steps=adatom_relaxation_steps,
        )

    return slab_container

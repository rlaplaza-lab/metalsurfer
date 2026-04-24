"""Slab construction and modification; re-exports helpers and ``prepare_slab``."""

from __future__ import annotations

from .surfaces import (
    SlabContainer,
    auto_resize_slab_for_molecule,
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
    results_dir: str = "results_manual",
    alloy_host: str | None = None,
    alloy_guest: str | None = None,
    alloy_fraction: float = 0.0,
    adatom_symbol: str | None = None,
    adatom_coverage: float = 0.0,
    model_name: str = "uma-s-1p1",
    device: str = "cuda",
    config=None,
) -> SlabContainer:
    """Build or load a slab, then optionally alloy-substitute and/or deposit adatoms.

    Exactly one of *bulk_id* or *slab_file* is required.

    Parameters
    ----------
    bulk_id:
        Materials Project structure ID (e.g. ``"mp-33"``).  Mutually
        exclusive with *slab_file*.
    miller_indices:
        Miller indices ``(h, k, l)`` for the surface orientation.
    supercell:
        Supercell repetition ``(nx, ny, nz)``.
    slab_file:
        Path to a pre-built POSCAR or XYZ file to load instead of querying
        the Materials Project.  Mutually exclusive with *bulk_id*.
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

    Returns
    -------
    SlabContainer
        Prepared slab ready to be passed to :func:`~metalsurfer.campaigns.run_adsorption`.
    """
    if bulk_id is None and slab_file is None:
        raise ValueError("Exactly one of 'bulk_id' or 'slab_file' must be provided")
    if bulk_id is not None and slab_file is not None:
        raise ValueError("'bulk_id' and 'slab_file' are mutually exclusive")

    if slab_file is not None:
        import ase.io

        atoms = ase.io.read(slab_file)
        slab = create_slab_from_atoms(atoms)
    else:
        slab = create_slab_from_bulk(
            bulk_id=bulk_id,
            miller_indices=miller_indices,
            supercell=supercell,
            results_dir=results_dir,
        )

    needs_calculator = (alloy_guest and alloy_fraction > 0) or (
        adatom_symbol and adatom_coverage > 0
    )
    calculator = None
    if needs_calculator:
        from .optimization import setup_single_model

        calculator, _ = setup_single_model(model_name, device)

    if alloy_guest and alloy_fraction > 0:
        host = alloy_host
        if host is None:
            host = sorted(set(slab.atoms.get_chemical_symbols()))[0]
        slab = substitute_alloy(
            slab,
            host_symbol=host,
            guest_symbol=alloy_guest,
            guest_fraction=alloy_fraction,
            calculator=calculator,
            config=config,
            results_dir=results_dir,
        )

    if adatom_symbol and adatom_coverage > 0:
        slab = deposit_adatoms(
            slab,
            adatom_symbol=adatom_symbol,
            coverage_fraction=adatom_coverage,
            calculator=calculator,
            config=config,
            results_dir=results_dir,
        )

    return slab

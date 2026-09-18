"""Reference-energy setup helpers for workflow orchestration."""

import logging
from collections.abc import Mapping

import numpy as np
from ase import Atoms

from .._logging import warn_once
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..exceptions import OptimizationError
from ..models import ReferenceEnergies
from ..optimization import (
    clear_autobatcher_cache,
    optimize_isolated_molecules_batched,
)
from ..surface_prep import SlabContainer
from .shared import _prepare_atoms_for_calculator

logger = logging.getLogger(__name__)


def _lookup_atom_ref(
    atom_refs: object,
    task_name: str,
    atomic_number: int,
    charge: int = 0,
) -> float | None:
    """Return a UMA isolated-atom energy (eV), or ``None`` if untabulated.

    Mirrors FairChem ``FAIRChemCalculator._get_single_atom_energies``:
    nested ``{Z: {charge: E}}`` (omol) or a list/sequence indexed by Z.
    ``oc25`` shares the OC20 catalysis table (no ``oc25`` key in the YAML).
    """
    if not isinstance(atom_refs, Mapping):
        return None
    stem = str(task_name).removesuffix("_elem_refs")
    aliased_oc25 = stem == "oc25"
    if aliased_oc25:
        stem = "oc20"
    table = atom_refs.get(stem)
    if table is None:
        table = atom_refs.get(f"{stem}_elem_refs")
    if table is None:
        return None
    if aliased_oc25:
        warn_once(
            logger,
            "uma_atom_ref_oc25",
            "UMA oc25 isolated-atom refs are not tabulated; using oc20 atom_refs",
        )
    z = int(atomic_number)
    try:
        energy = table.get(z, {}).get(int(charge))
    except (AttributeError, TypeError):
        try:
            energy = table[z]
        except (IndexError, KeyError, TypeError):
            return None
    if energy is None:
        return None
    try:
        value = float(energy)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _predictor_atom_refs(ts_model: object, calculator: object) -> object:
    for obj in (ts_model, getattr(calculator, "_model", None)):
        if obj is None:
            continue
        refs = getattr(getattr(obj, "predictor", None), "atom_refs", None)
        if refs is not None:
            return refs
    return None


def calculate_reference_energies(
    slab: SlabContainer,
    calculator,
    molecules: list[str],
    smiles_list: list[str],
    ts_model=None,
    config: AdsorptionConfig | None = None,
) -> ReferenceEnergies:
    """Compute clean-slab and isolated-molecule energies.

    Parameters
    ----------
    slab
        Substrate container.
    calculator
        ASE calculator instance.
    molecules
        List of molecule names.
    smiles_list
        SMILES strings aligned with molecules.
    ts_model
        Transition-state model (optional).
    config
        Adsorption configuration.
    """
    if config is None:
        config = AdsorptionConfig()

    slab_copy = slab.atoms.copy()
    _prepare_atoms_for_calculator(slab_copy, label="reference slab")
    slab_copy.calc = calculator
    slab_energy = slab_copy.get_potential_energy()
    if not np.isfinite(slab_energy):
        raise OptimizationError(
            f"Clean slab energy is not finite: {slab_energy}. "
            "The calculator may have failed; check GPU stability and model output."
        )
    if abs(slab_energy) < 1e-6:
        raise OptimizationError(
            f"Clean slab energy is effectively zero ({slab_energy:.6e} eV). "
            "A real slab cannot have zero energy; the calculator likely returned "
            "a default. Check that the ML model produced valid output."
        )
    logger.info("Clean slab energy: %.4f eV", slab_energy)

    molecule_energies: dict[str, float] = {}
    conformer_packs: dict[str, tuple[list[Atoms], list[float]]] = {}
    atom_refs = _predictor_atom_refs(ts_model, calculator)
    for mol_name, smiles in zip(molecules, smiles_list, strict=True):
        logger.info("Calculating isolated %s energy", mol_name)
        result = create_conformers_from_smiles(
            smiles, calculator=calculator, config=config, ts_model=ts_model
        )
        if result is None:
            if config.fail_on_conformer_failure:
                raise RuntimeError(
                    f"Could not create conformers for {mol_name} from SMILES: {smiles}"
                )
            logger.warning("Could not create %s from SMILES: %s", mol_name, smiles)
            continue
        conformers, conformer_energies = result
        if conformers and len(conformers[0]) == 1:
            atoms0 = conformers[0]
            z = int(atoms0.get_atomic_numbers()[0])
            energy = _lookup_atom_ref(
                atom_refs,
                config.task_name,
                z,
                int(atoms0.info.get("charge", 0)),
            )
            if energy is None:
                reason = f"No UMA isolated-atom energy for {mol_name} (Z={z})"
                if config.fail_on_conformer_failure or config.fail_on_missing_reference:
                    raise OptimizationError(reason)
                logger.warning("%s; omitting molecule", reason)
                continue
            pack_energies = [energy] * len(conformers)
            conformer_packs[mol_name] = (list(conformers), pack_energies)
            molecule_energies[mol_name] = energy
            logger.info(
                "%s isolated energy: %.4f eV (UMA atom_refs)",
                mol_name,
                energy,
            )
            continue
        # Keep the pre–isolated-opt pack for placement (must not use post-opt geoms).
        conformer_packs[mol_name] = (list(conformers), list(conformer_energies))
        opt_results = optimize_isolated_molecules_batched(
            conformers,
            ts_model,
            fmax=config.fmax,
            steps=config.reference_optimization_steps,
            config=config,
        )
        if not opt_results:
            if config.fail_on_conformer_failure:
                raise OptimizationError(
                    f"Failed to optimise any conformers for {mol_name}"
                )
            logger.error("Failed to optimise any conformers for %s", mol_name)
            continue
        best_e, best_i = min(
            ((e, i) for i, (_, e) in enumerate(opt_results)),
            key=lambda item: item[0],
        )
        molecule_energies[mol_name] = best_e
        logger.info(
            "%s isolated energy: %.4f eV (best conformer: %d)",
            mol_name,
            best_e,
            best_i,
        )

    # Model/substrate boundary: the reference stage is done, so drop the probed
    # capacity estimates too rather than carrying them into later stages.
    clear_autobatcher_cache(clear_capacity=True)
    return ReferenceEnergies(
        slab_energy=slab_energy,
        molecule_energies=molecule_energies,
        conformer_packs=conformer_packs,
    )

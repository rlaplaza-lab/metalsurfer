"""Reference-energy setup helpers for workflow orchestration."""

import logging

import numpy as np

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


def calculate_reference_energies(
    slab: SlabContainer,
    calculator,
    molecules: list[str],
    smiles_list: list[str],
    ts_model=None,
    config: AdsorptionConfig | None = None,
) -> ReferenceEnergies:
    """Compute clean-slab and isolated-molecule energies."""
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
        conformers, _ = result
        opt_results = optimize_isolated_molecules_batched(
            conformers,
            ts_model,
            fmax=config.fmax,
            steps=config.reference_optimization_steps,
            config=config,
        )
        energies = [(e, i) for i, (_, e) in enumerate(opt_results)]
        if energies:
            energies.sort()
            molecule_energies[mol_name] = energies[0][0]
            logger.info(
                "%s isolated energy: %.4f eV (best conformer: %d)",
                mol_name,
                energies[0][0],
                energies[0][1],
            )
        else:
            if config.fail_on_conformer_failure:
                raise OptimizationError(
                    f"Failed to optimise any conformers for {mol_name}"
                )
            logger.error("Failed to optimise any conformers for %s", mol_name)

    # Model/substrate boundary: the reference stage is done, so drop the probed
    # capacity estimates too rather than carrying them into later stages.
    clear_autobatcher_cache(clear_capacity=True)
    return ReferenceEnergies(
        slab_energy=slab_energy,
        molecule_energies=molecule_energies,
    )

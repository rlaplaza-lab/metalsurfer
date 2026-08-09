"""Conformer generation from SMILES via RDKit and deduplication."""

import logging
import random
from typing import Any

import numpy as np
from ase import Atoms

from .config import AdsorptionConfig

logger = logging.getLogger(__name__)

Chem: Any = None
AllChem: Any = None
try:
    from rdkit import Chem as _rdkit_chem
    from rdkit.Chem import AllChem as _rdkit_allchem

    Chem = _rdkit_chem
    AllChem = _rdkit_allchem
except (ImportError, AttributeError):
    pass


def create_conformers_from_smiles(
    smiles: str,
    calculator=None,
    config: AdsorptionConfig | None = None,
    ts_model=None,
) -> tuple[list[Atoms], list[float]] | None:
    """Generate 3-D conformers for *smiles* via RDKit, optionally score them.

    When *ts_model* is provided, all conformers are scored in a single batched
    ``ts.static()`` call instead of one-by-one through the calculator. This is
    significantly faster on GPU.

    Returns ``(conformers, energies)`` or ``None`` on failure.
    """
    if Chem is None or AllChem is None:
        raise RuntimeError(
            "RDKit is required for conformer generation. "
            "Install it with: pip install rdkit"
        )
    if config is None:
        config = AdsorptionConfig()

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error("Could not parse SMILES: %s", smiles)
        return None

    mol = Chem.AddHs(mol)

    conf_ids = AllChem.EmbedMultipleConfs(
        mol, numConfs=config.num_conformers, randomSeed=config.seed
    )
    if len(conf_ids) == 0:
        logger.warning(
            "Could not generate conformers for %s, trying single conformer",
            smiles,
        )
        AllChem.EmbedMolecule(mol, randomSeed=config.seed)
        if mol.GetNumConformers() == 0:
            logger.error("Failed to generate any conformer for %s", smiles)
            return None
        conf_ids = [0]

    for conf_id in conf_ids:
        mmff_result = AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)
        if mmff_result == -1:
            logger.warning(
                "MMFF optimization failed for conformer %d of %s (unsupported atom types)",
                conf_id,
                smiles,
            )
        elif mmff_result == 1:
            logger.debug(
                "MMFF did not converge for conformer %d of %s", conf_id, smiles
            )

    conformers: list[Atoms] = []

    for conf_id in conf_ids:
        conf = mol.GetConformer(conf_id)
        positions = []
        symbols = []
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            positions.append([pos.x, pos.y, pos.z])
            symbols.append(atom.GetSymbol())

        atoms = Atoms(symbols=symbols, positions=np.array(positions))
        if calculator is not None or ts_model is not None:
            atoms.cell = [config.vacuum_box_size] * 3
            atoms.set_pbc([False, False, False])
        conformers.append(atoms)

    if ts_model is not None and len(conformers) > 0:
        from .optimization import batch_static

        results = batch_static(conformers, ts_model)
        energies = [e for e, _f in results]
    elif calculator is not None:
        energies = []
        for atoms in conformers:
            atoms.calc = calculator
            energies.append(atoms.get_potential_energy())
    else:
        energies = [0.0] * len(conformers)

    logger.info("Created %d conformers for %s", len(conformers), smiles)
    if (calculator is not None or ts_model is not None) and energies:
        logger.info("Energies range: %.4f to %.4f eV", min(energies), max(energies))

    conformers, energies = remove_duplicate_conformers(
        conformers,
        energies,
        distance_threshold=config.rmsd_dedup_threshold,
        energy_threshold=config.energy_dedup_threshold,
    )
    return conformers, energies


def remove_duplicate_conformers(
    conformers: list[Atoms],
    energies: list[float],
    distance_threshold: float = 0.1,
    energy_threshold: float = 0.05,
) -> tuple[list[Atoms], list[float]]:
    """Remove near-duplicate conformers by centred RMSD + energy."""
    if len(conformers) <= 1:
        return conformers, energies

    sorted_indices = np.argsort(energies)
    sorted_conformers = [conformers[i] for i in sorted_indices]
    sorted_energies = [energies[i] for i in sorted_indices]

    unique_conformers: list[Atoms] = []
    unique_energies: list[float] = []

    for conformer, energy in zip(sorted_conformers, sorted_energies, strict=True):
        is_duplicate = False
        for uc, ue in zip(unique_conformers, unique_energies, strict=True):
            if abs(energy - ue) < energy_threshold:
                rmsd = _kabsch_rmsd(conformer.get_positions(), uc.get_positions())
                if rmsd < distance_threshold:
                    is_duplicate = True
                    break
        if not is_duplicate:
            unique_conformers.append(conformer)
            unique_energies.append(energy)

    logger.info(
        "Removed %d duplicate conformers (%d/%d remaining)",
        len(conformers) - len(unique_conformers),
        len(unique_conformers),
        len(conformers),
    )
    return unique_conformers, unique_energies


def _kabsch_rmsd(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """RMSD after optimal Kabsch rotation, so orientation-equivalent conformers match.

    Both point sets are centred on their centroids, then ``pos_a`` is rotated by the
    optimal rotation (SVD of the cross-covariance) before measuring RMSD against
    ``pos_b``. Returns ``inf`` when the atom counts disagree.
    """
    if np.asarray(pos_a).shape != np.asarray(pos_b).shape:
        return float("inf")
    a = np.asarray(pos_a, dtype=float) - np.mean(pos_a, axis=0)
    b = np.asarray(pos_b, dtype=float) - np.mean(pos_b, axis=0)
    H = a.T @ b
    U, _S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    rot = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    a_rot = a @ rot.T
    return float(np.sqrt(np.mean(np.sum((a_rot - b) ** 2, axis=1))))


def select_conformer_boltzmann(
    conformers: list[Atoms],
    energies: list[float],
    temperature: float = 300.0,
    rng: random.Random | None = None,
) -> Atoms:
    """Pick a conformer by Boltzmann weighting and return a *copy*.

    Parameters
    ----------
    rng:
        Seeded ``random.Random`` instance for reproducibility.  When
        ``None`` a module-level default is used (non-deterministic).
    """
    _rng = rng if rng is not None else random

    if len(conformers) == 1:
        return conformers[0].copy()

    if len(energies) != len(conformers):
        logger.warning(
            "Mismatch between conformers and energies, using random selection"
        )
        return _rng.choice(conformers).copy()

    valid_indices = [i for i, e in enumerate(energies) if np.isfinite(e)]
    if not valid_indices:
        logger.warning("No valid energies found, using random selection")
        return _rng.choice(conformers).copy()

    valid_energies = [energies[i] for i in valid_indices]
    min_energy = min(valid_energies)

    kB_T = 8.617e-5 * temperature
    weights = [np.exp(-(e - min_energy) / kB_T) for e in valid_energies]
    total = sum(weights)
    weights = [w / total for w in weights]

    rand_val = _rng.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if rand_val <= cumulative:
            return conformers[valid_indices[i]].copy()

    # Fallback for floating-point rounding: use last valid conformer
    return conformers[valid_indices[-1]].copy()


def select_conformer_for_placement(
    conformers: list[Atoms],
    energies: list[float],
    placement_id: int,
    sampling: str = "cycle",
    temperature: float = 300.0,
    rng: random.Random | None = None,
) -> Atoms:
    """Select a conformer for placement with diversity-aware sampling.

    .. deprecated::
        Superseded by spec-based placement. Conformer choice is carried
        explicitly by ``PlacementSpec.conformer_index`` and enumerated by the
        placement policy, so nothing in the pipeline calls this. It is kept only
        for backwards compatibility; ``AdsorptionConfig.conformer_sampling`` and
        ``boltzmann_temperature`` are no-ops for the same reason.

    When *sampling* is "cycle", cycles through conformers by placement_id to
    ensure all conformers are used. When "boltzmann", uses Boltzmann weighting.
    When "mixed", alternates: even placement_ids use cycle, odd use Boltzmann.
    """
    if not conformers:
        raise ValueError("conformers must not be empty")
    if len(conformers) == 1:
        return conformers[0].copy()

    if sampling == "cycle":
        idx = placement_id % len(conformers)
        return conformers[idx].copy()

    if sampling == "mixed":
        if placement_id % 2 == 0:
            idx = placement_id % len(conformers)
            return conformers[idx].copy()
        return select_conformer_boltzmann(
            conformers, energies, temperature=temperature, rng=rng
        )

    return select_conformer_boltzmann(
        conformers, energies, temperature=temperature, rng=rng
    )

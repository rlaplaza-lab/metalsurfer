"""Shared substrate and motif helpers for edge-placement snapshot tests."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
from ase import Atoms
from ase.build import fcc111

from metalsurfer._geom_pbc import minimum_image_cartesian_delta
from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.filters import adsorbate_connected_components
from metalsurfer.models import ScreeningResult
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.surface_prep import (
    apply_surface_constraints,
    prepare_substrate,
    resize_substrate_for_molecule,
)
from metalsurfer.workflow.placement_fill import fill_materialized_placements

_AU_AU_ANG = 2.88
_HBOND_HO_LO_ANG = 1.40
_HBOND_HO_HI_ANG = 2.50
_SLAB_PBC = material_aware_pbc("slab")
_COVERAGE_PLACEMENTS = 100

_PME3_SMILES = "C[P:1](C)C"
_PYRIDONE_SMILES = "O=c1cccc[nH]1"
_OLEIC_SMILES = r"CCCCCCCC/C=C\CCCCCCCC(=O)O"

_PME3_FORMULA = {"C": 3, "H": 9, "P": 1}
_PYRIDONE_FORMULA = {"C": 5, "H": 5, "N": 1, "O": 1}
_OLEIC_FORMULA = {"C": 18, "H": 34, "O": 2}


def au4_tetrahedron() -> Atoms:
    scale = _AU_AU_ANG / math.sqrt(8.0)
    cluster = Atoms(
        "Au4",
        positions=[
            [scale, scale, scale],
            [scale, -scale, -scale],
            [-scale, scale, -scale],
            [-scale, -scale, scale],
        ],
        cell=[25.0, 25.0, 25.0],
        pbc=False,
    )
    cluster.center()
    return cluster


def assert_intact(
    ads: Atoms, formula: dict[str, int], config: AdsorptionConfig
) -> None:
    symbols = ads.get_chemical_symbols()
    assert {sym: symbols.count(sym) for sym in sorted(set(symbols))} == formula
    parts = adsorbate_connected_components(
        ads,
        0,
        config.connectivity_multiplier,
        material_type=config.material_type,
    )
    assert len(parts) == 1


def cu111_for(
    smiles: str, config: AdsorptionConfig, results_dir: str
) -> tuple[Atoms, list[Atoms]]:
    pack = create_conformers_from_smiles(smiles, config=config)
    assert pack is not None
    conformers, _energies = pack
    assert conformers
    prepared = prepare_substrate(
        slab=apply_surface_constraints(
            fcc111("Cu", size=(1, 1, 3), a=3.6, vacuum=10.0)
        ),
        config=config,
        results_dir=results_dir,
    )
    return resize_substrate_for_molecule(prepared, conformers, config).atoms, conformers


def _symbol_positions(atoms: Atoms, symbol: str) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    return pos[
        [i for i, sym in enumerate(atoms.get_chemical_symbols()) if sym == symbol]
    ]


def _min_distance(
    pos_a: np.ndarray, pos_b: np.ndarray, cell: np.ndarray, pbc: list[bool]
) -> float:
    best = float("inf")
    pos_b = np.asarray(pos_b, dtype=float)
    flags = np.asarray(pbc, dtype=bool)
    for point in np.asarray(pos_a, dtype=float):
        mic = minimum_image_cartesian_delta(pos_b - point, cell, flags)
        best = min(best, float(np.min(np.linalg.norm(mic, axis=-1))))
    return best


def _nh_hydrogen(mol: Atoms) -> np.ndarray:
    nitrogen = _symbol_positions(mol, "N")
    hydrogens = _symbol_positions(mol, "H")
    assert len(nitrogen) == 1 and len(hydrogens)
    return hydrogens[int(np.argmin(np.linalg.norm(hydrogens - nitrogen[0], axis=1)))]


def has_hbond_contact(left: Atoms, right: Atoms, *, cell: np.ndarray) -> bool:
    for donor, acceptor in ((left, right), (right, left)):
        gap = _min_distance(
            _nh_hydrogen(donor).reshape(1, 3),
            _symbol_positions(acceptor, "O"),
            cell,
            _SLAB_PBC,
        )
        if _HBOND_HO_LO_ANG <= gap <= _HBOND_HO_HI_ANG:
            return True
    return False


def materialize_pose_pool(
    *,
    smiles: str,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    site_context,
    n_desired: int = _COVERAGE_PLACEMENTS,
    full_slab: Atoms | None = None,
) -> list[ScreeningResult]:
    """Rigid first-*n_desired* placements as unrelaxed screening stubs."""
    coverage = full_slab if full_slab is not None else slab
    fill = fill_materialized_placements(
        conformers=conformers,
        slab_for_sites=slab,
        config=replace(config, num_placements=n_desired),
        smiles=smiles,
        site_context=site_context,
        slab_atoms=coverage,
        calculator=None,
    )
    slab_size = len(coverage)
    return [
        ScreeningResult(
            molecule="adsorbate",
            placement_id=pid,
            energy_adslab=0.0,
            energy_slab=0.0,
            energy_adsorbate=0.0,
            energy_adsorption=0.0,
            atoms=combined,
            slab_size=slab_size,
            distance=0.0,
            placement_descriptor=desc,
        )
        for combined, pid, desc in zip(
            fill.combined, fill.placement_ids, fill.descriptors, strict=True
        )
    ]


def p_nearer_au_than_carbons(ads: Atoms, au_pos: np.ndarray) -> bool:
    pos = np.asarray(ads.get_positions(), dtype=float)
    symbols = ads.get_chemical_symbols()
    if "P" not in symbols:
        return False
    p_pos = pos[symbols.index("P")]
    p_au = float(np.linalg.norm(au_pos - p_pos, axis=1).min())
    heavy = pos[[i for i, sym in enumerate(symbols) if sym not in {"H", "P"}]]
    return bool(len(heavy)) and p_au < float(
        np.linalg.norm(heavy[:, None, :] - au_pos, axis=-1).min()
    )


def carboxyl_down(ads: Atoms, slab_pos: np.ndarray, cell: np.ndarray) -> bool:
    o_cu = _min_distance(_symbol_positions(ads, "O"), slab_pos, cell, _SLAB_PBC)
    c_cu = _min_distance(_symbol_positions(ads, "C"), slab_pos, cell, _SLAB_PBC)
    return o_cu < c_cu

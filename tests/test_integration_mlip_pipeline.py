"""Integration tests: real MLIP pipeline via process_molecule."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from ase import Atoms
from ase.build import hcp0001
from ase.io import read

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import ScreeningResult
from metalsurfer.optimization import setup_single_model
from metalsurfer.placement._constants import (
    _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
)
from metalsurfer.surface_prep import apply_surface_constraints, prepare_substrate
from metalsurfer.workflow import calculate_reference_energies, process_molecule
from tests.conftest import (
    E_ADS_MLIP_TOL,
    GPU_AUTOBATCH,
    GPU_MLIP_MARKS,
    adsorbate_symbol_pair_distance,
    pair_distance,
)

pytestmark = GPU_MLIP_MARKS

_MLIP_CASE_IDS = ("ethene_ru", "h2_ru", "h2_pt12", "co2_mof")


def _pt12_cluster() -> Atoms:
    return Atoms(
        symbols=["Pt"] * 12,
        positions=[
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.425, 0.0],
            [4.2, 2.425, 0.0],
            [1.4, 0.808, 2.0],
            [4.2, 0.808, 2.0],
            [0.0, 2.425, 2.0],
            [2.8, 2.425, 2.0],
            [1.4, 1.617, 4.0],
            [4.2, 1.617, 4.0],
            [0.0, 0.808, 4.0],
            [2.8, 0.808, 4.0],
        ],
        cell=[20, 20, 20],
        pbc=False,
    )


def _local_ru_001_slab() -> Atoms:
    """Local Ru(0001) slab via ASE hcp0001 (fcc111 needs an explicit a for Ru)."""
    return apply_surface_constraints(hcp0001("Ru", size=(4, 4, 3), vacuum=10.0))


def _run_mlip_pipeline(case_id: str) -> tuple[list[ScreeningResult], int]:
    if case_id == "ethene_ru":
        num_placements = 12
        config = AdsorptionConfig(
            material_type="slab",
            seed=42,
            num_conformers=3,
            num_placements=num_placements,
            device="cuda",
            min_pbc_image_separation=4.5,
            slab_relaxation_mode="none",
            **GPU_AUTOBATCH,
        )
        assert config.skip_topology_check is False
        assert config.skip_desorption_check is False
        slab = prepare_substrate(
            slab=_local_ru_001_slab(),
            config=config,
            results_dir="results_test_ethene",
        )
        smiles, name, surface_type = "C=C", "ethene", "Ru_001"
    elif case_id == "h2_ru":
        num_placements = 10
        config = AdsorptionConfig(
            material_type="slab",
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            device="cuda",
            enable_dissociative_placement=True,
            skip_topology_check=True,
            min_pbc_image_separation=4.5,
            slab_relaxation_mode="none",
            **GPU_AUTOBATCH,
        )
        assert config.skip_desorption_check is False
        slab = prepare_substrate(
            slab=_local_ru_001_slab(),
            config=config,
            results_dir="results_test_h2_ru_slab",
        )
        smiles, name, surface_type = "[H][H]", "H2", "h2_ru_slab"
    elif case_id == "h2_pt12":
        num_placements = 5
        config = AdsorptionConfig(
            material_type="nanoparticle",
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            device="cuda",
            slab_relaxation_mode="none",
            enable_dissociative_placement=True,
            skip_topology_check=True,
            **GPU_AUTOBATCH,
        )
        slab = prepare_substrate(
            slab=_pt12_cluster(),
            config=config,
            results_dir="results_test_h2_pt12",
        )
        smiles, name, surface_type = "[H][H]", "H2", "h2_pt12"
    elif case_id == "co2_mof":
        num_placements = 5
        config = AdsorptionConfig(
            material_type="porous",
            slab_relaxation_mode="none",
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            device="cuda",
            # MOF pores: UMA often leaves residual forces ~0.06–0.1 eV/Å after
            # the default stage budget; allow that window for this e2e gate.
            max_force_convergence=0.15,
            stage1_steps=75,
            stage2_steps=250,
            **GPU_AUTOBATCH,
        )
        assert config.skip_topology_check is False
        assert config.skip_desorption_check is False
        cif_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "examples",
            "mof_structures",
            "RUBTAK01.cif",
        )
        slab = prepare_substrate(
            slab=read(cif_path),
            config=config,
            results_dir="results_test_co2_mof",
            align=False,
        )
        smiles, name, surface_type = "O=C=O", "CO2", "co2_mof"
    else:
        raise ValueError(f"unknown case_id: {case_id}")

    if case_id != "co2_mof":
        assert config.stage1_steps == 50
        assert config.stage2_steps == 150
    else:
        assert config.stage1_steps >= 50
        assert config.stage2_steps >= 150

    calculator, ts_model = setup_single_model(
        config.model_name, config.device, task_name=config.task_name
    )
    ref = calculate_reference_energies(
        slab, calculator, [name], [smiles], ts_model=ts_model, config=config
    )
    results = process_molecule(
        smiles,
        name,
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type=surface_type,
    ).results
    return results, num_placements


def _assert_ethene_ru(results: list[ScreeningResult], num_placements: int) -> None:
    min_ok = max(6, int(math.ceil(0.5 * num_placements)))
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )

    e_ads = np.array([r.energy_adsorption for r in results])
    # Bounds tightened against the uma-s-1p2 + oc25 reference run
    # (observed: E_ads in [-0.14, 0.01], median -0.01, spread 0.15).
    assert e_ads.min() < 0, (
        f"Best E_ads should be negative (favorable binding), got min {e_ads.min():.3f}"
    )
    assert np.median(e_ads) < 0, (
        f"Median E_ads should be negative, got {np.median(e_ads):.3f}; all: {e_ads}"
    )
    assert np.all(e_ads < 0.15), (
        f"E_ads should stay below 0.15 eV for ethene on Ru, got {e_ads}"
    )
    assert np.all(e_ads >= -0.6), (
        f"E_ads should be >= -0.6 eV for ethene on Ru, got min {e_ads.min():.3f}"
    )

    spread = float(e_ads.max() - e_ads.min())
    assert spread >= 0.03, (
        f"Expected distribution of E_ads (spread >= 0.03 eV), got spread {spread:.4f}"
    )

    slab_size = len(results[0].atoms) - 6
    for r in results:
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=E_ADS_MLIP_TOL,
        )
        assert 1.5 <= r.distance <= 4.0, (
            f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
        )
        ads = r.atoms[r.slab_size :]
        assert len(ads) == 6
        assert sorted(ads.get_chemical_symbols()) == ["C", "C", "H", "H", "H", "H"]
        assert r.placement_descriptor is not None
        assert r.placement_descriptor.surface_ref_z_abs is not None
        cc = adsorbate_symbol_pair_distance(r.atoms, slab_size, "C")
        assert 1.30 <= cc <= 1.48, (  # UMA on Ru(0001): ~1.455 Å
            f"C=C bond length should be ~1.34–1.46 Å (1.30–1.48), got {cc:.3f}"
        )


def _assert_h2_ru(results: list[ScreeningResult], num_placements: int) -> None:
    min_ok = max(5, int(math.ceil(0.5 * num_placements)))
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )

    e_ads = np.array([r.energy_adsorption for r in results])
    # Bounds tightened against the uma-s-1p2 + oc25 reference run
    # (observed: every placement at E_ads = -0.1185 eV, distance 1.75 A).
    assert np.all(np.isfinite(e_ads))
    assert float(e_ads.min()) < 0.0, (
        f"Best E_ads should be negative for H2 on Ru, got {e_ads}"
    )
    assert np.all(e_ads < 0.1), (
        f"E_ads should stay below 0.1 eV for H2 on Ru, got {e_ads}"
    )
    assert np.all(e_ads >= -0.4), (
        f"E_ads should be >= -0.4 eV for H2 on Ru, got min {e_ads.min():.3f}"
    )

    site_ids = set()
    for r in results:
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=E_ADS_MLIP_TOL,
        )
        assert r.placement_descriptor.orientation_type == "dissociative"
        assert r.placement_descriptor.site_source == "dissociative_hollow_pair"
        assert 1.5 <= r.distance <= 4.0, (
            f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
        )
        if r.placement_descriptor.site_index is not None:
            site_ids.add(int(r.placement_descriptor.site_index))

    assert len(results) >= 2, (
        f"Expected multi-site coverage, got {len(results)} results"
    )
    assert len(site_ids) >= 2, f"Expected multi-site coverage, got {site_ids}"

    slab_size = len(results[0].atoms) - 2
    hh_lengths = [
        adsorbate_symbol_pair_distance(r.atoms, slab_size, "H") for r in results
    ]
    for hh in hh_lengths:
        # Molecular H2 or dissociated up to production adjacent-sep cap.
        assert (0.7 <= hh <= 0.9) or (
            1.5 <= hh <= _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM
        ), f"H–H should be molecular or dissociated, got {hh:.3f} (all={hh_lengths})"


def _assert_h2_pt12(results: list[ScreeningResult], num_placements: int) -> None:
    min_ok = max(2, int(math.ceil(0.4 * num_placements)))
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )

    e_ads = np.array([r.energy_adsorption for r in results])
    # Bounds tightened against the uma-s-1p2 + oc25 reference run (observed:
    # E_ads in [-0.10, 1.24]; two physisorbed/desorbed outliers near +1.2 eV).
    assert np.all(np.isfinite(e_ads))
    assert float(e_ads.min()) < 0.1, (
        f"Best E_ads should be near-binding (<0.1 eV) for H2 on Pt12, got {e_ads}"
    )
    assert np.all(e_ads < 1.5), (
        f"E_ads should stay below a weak-binding ceiling (< 1.5 eV), got {e_ads}"
    )
    assert np.all(e_ads >= -0.8), (
        f"E_ads should be >= -0.8 eV for H2 on Pt12, got min {e_ads.min():.3f}"
    )

    assert len(results) >= 2, f"Expected multiple configs, got {len(results)} results"
    spread = float(e_ads.max() - e_ads.min())
    assert spread >= 0.01, (
        f"Expected distinct E_ads when multiple unique configs remain, "
        f"got spread {spread:.4f}"
    )

    slab_size = len(results[0].atoms) - 2
    for r in results:
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=E_ADS_MLIP_TOL,
        )
        assert r.placement_descriptor is not None
        assert r.placement_descriptor.orientation_type == "dissociative"
        assert 1.5 <= r.distance <= 4.0, (
            f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
        )
        hh = adsorbate_symbol_pair_distance(r.atoms, slab_size, "H")
        assert (0.7 <= hh <= 0.9) or (
            1.5 <= hh <= _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM
        ), f"H–H should be molecular or dissociated on cluster, got {hh:.3f}"


def _co_bond_lengths(atoms, slab_size: int) -> tuple[float, float]:
    ads = atoms[slab_size:]
    syms = ads.get_chemical_symbols()
    c_indices = [i for i, s in enumerate(syms) if s == "C"]
    o_indices = [i for i, s in enumerate(syms) if s == "O"]
    if len(c_indices) != 1 or len(o_indices) != 2:
        return float("nan"), float("nan")
    pos = ads.get_positions()
    cell = atoms.get_cell()
    return (
        pair_distance(pos[c_indices[0]], pos[o_indices[0]], cell=cell),
        pair_distance(pos[c_indices[0]], pos[o_indices[1]], cell=cell),
    )


def _assert_co2_mof(results: list[ScreeningResult], num_placements: int) -> None:
    min_ok = max(4, int(math.ceil(0.8 * num_placements)))
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )

    e_ads = np.array([r.energy_adsorption for r in results])
    # Bounds tightened against the uma-s-1p2 + oc25 reference run
    # (observed: E_ads in [-0.22, -0.13] — all placements bind physisorptively).
    assert np.all(e_ads < 0.05), (
        f"E_ads should stay in a physisorption window (< 0.05 eV), got {e_ads}"
    )
    assert np.all(e_ads >= -0.8), (
        f"E_ads should be >= -0.8 eV for CO2 in MOF, got min {e_ads.min():.3f}"
    )
    assert float(e_ads.min()) < 0.0, (
        f"Best E_ads should be favorable physisorption (< 0 eV) for CO2 in MOF, "
        f"got {e_ads}"
    )

    spread = float(e_ads.max() - e_ads.min())
    assert spread >= 0.02, (
        f"Expected distribution of E_ads (spread >= 0.02 eV), got spread {spread:.4f}"
    )

    site_ids = set()
    slab_size = len(results[0].atoms) - 3
    for r in results:
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=E_ADS_MLIP_TOL,
        )
        assert 1.5 <= r.distance <= 4.0, (
            f"Adsorbate–surface distance should be 1.5–4.0 Å, got {r.distance:.2f}"
        )
        assert r.placement_descriptor is not None
        assert r.placement_descriptor.surface_ref_z_abs is not None
        if r.placement_descriptor.site_index is not None:
            site_ids.add(int(r.placement_descriptor.site_index))
        co1, co2 = _co_bond_lengths(r.atoms, slab_size)
        assert 1.08 <= co1 <= 1.30, (
            f"C–O bond length should be ~1.16 Å (1.08–1.30), got {co1:.3f}"
        )
        assert 1.08 <= co2 <= 1.30, (
            f"C–O bond length should be ~1.16 Å (1.08–1.30), got {co2:.3f}"
        )
        ads = r.atoms[slab_size:]
        syms = ads.get_chemical_symbols()
        assert sorted(syms) == ["C", "O", "O"]
        c_idx = syms.index("C")
        o_idxs = [i for i, s in enumerate(syms) if s == "O"]
        pos = ads.get_positions()
        cell = np.asarray(r.atoms.get_cell(), dtype=float)
        v1 = pos[o_idxs[0]] - pos[c_idx]
        v2 = pos[o_idxs[1]] - pos[c_idx]
        v1 = v1 - np.round(v1 @ np.linalg.inv(cell)) @ cell
        v2 = v2 - np.round(v2 @ np.linalg.inv(cell)) @ cell
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        angle = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
        assert 172.0 <= angle <= 180.0, (
            f"O–C–O angle should be ~180° (172–180), got {angle:.1f}"
        )
    assert len(results) >= 2, (
        f"Expected multi-site coverage, got {len(results)} results"
    )
    assert len(site_ids) >= 2, f"Expected multi-site coverage, got {site_ids}"


@pytest.mark.parametrize("case_id", _MLIP_CASE_IDS)
def test_mlip_pipeline(case_id: str, workdir) -> None:
    # workdir chdirs into a temp dir so results_test_* do not land in the repo root.
    results, num_placements = _run_mlip_pipeline(case_id)
    match case_id:
        case "ethene_ru":
            _assert_ethene_ru(results, num_placements)
        case "h2_ru":
            _assert_h2_ru(results, num_placements)
        case "h2_pt12":
            _assert_h2_pt12(results, num_placements)
        case "co2_mof":
            _assert_co2_mof(results, num_placements)
        case _:
            raise AssertionError(f"unknown case_id: {case_id}")

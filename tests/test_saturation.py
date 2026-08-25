"""Tests for sequential saturation feature."""

import logging
import os
import re
import tempfile
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.io_results import (
    _saturation_molecule_label,
    save_multi_mol_saturation_results,
    save_saturation_results,
    setup_directories,
)
from metalsurfer.models import (
    BOStepMemory,
    BOTransferInfo,
    MultiMolSaturationRunResult,
    MultiMolSaturationStepResult,
    SaturationRunResult,
    SaturationStepResult,
    _saturation_step_eads_xyz_name,
)
from metalsurfer.placement import (
    distribute_placement_budget,
    get_hollow_sites_for_adatoms,
)
from metalsurfer.placement.generators import estimate_molecule_complexity
from metalsurfer.surface_prep import (
    SlabContainer,
    auto_resize_substrate_for_molecule,
    prepare_substrate,
)
from metalsurfer.symmetry import SymmetryAnalysisError
from metalsurfer.workflow import load_molecules, run_saturation_screening
from metalsurfer.workflow.saturation import (
    _filter_saturation_topology_results,
    _reference_smiles_units_multi_molecule,
    _saturation_adsorbate_topology_ok,
    _saturation_symmetry_broken_vs_reference,
    _slab_after_saturation_step,
)
from metalsurfer.workflow.shared import (
    MoleculeScreenOutcome,
    ScreeningRunBootstrap,
    _build_surface_reference_slab,
)

from .conftest import (
    DummyReferenceEnergies,
    NoopDatasetLogger,
    assert_paths_exist,
    gpu_mlip_test,
    make_h2,
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)
from .factories import (
    REF_A_B,
    REF_CONSTANT,
    REF_WATER_CO2,
    make_saturation_run,
)


def _mock_saturation_config(**kwargs) -> AdsorptionConfig:
    """AdsorptionConfig for mocked saturation loops without real adsorbate geometries."""
    return AdsorptionConfig(
        num_placements=100,
        saturation_discard_topology_rearrangements=False,
        **kwargs,
    )


def _uniform_placement_budget(
    complexities: dict[str, float], total: int
) -> dict[str, int]:
    n = len(complexities)
    return {mol: total // n for mol in complexities}


def _result_for_step(
    molecule: str, current_slab: SlabContainer, e_ads: float, placement_id: int = 0
):
    return MoleculeScreenOutcome(
        results=[
            make_screening_result(
                molecule=molecule,
                placement_id=placement_id,
                energy_adsorption=e_ads,
                atoms=current_slab.atoms.copy(),
                slab_size=len(current_slab.atoms),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(
                    placement_id=placement_id
                ),
            )
        ]
    )


def _make_schedule_process(
    schedules: dict[str, list[float]],
) -> Callable[..., list]:
    counts = {m: 0 for m in schedules}

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        counts[mol] += 1
        vals = schedules[mol]
        idx = min(counts[mol] - 1, len(vals) - 1)
        return _result_for_step(mol, current_slab, vals[idx])

    return _fake_process_molecule


def _make_bootstrap_mock(
    *,
    molecule: str,
    smiles: str,
    ref: DummyReferenceEnergies,
    slab: SlabContainer | Atoms | None = None,
) -> ScreeningRunBootstrap:
    if slab is None:
        slab = SlabContainer(Atoms("Pt", positions=[[0, 0, 0]], cell=[10, 10, 10]))
    elif not isinstance(slab, SlabContainer):
        slab = SlabContainer(slab)
    return ScreeningRunBootstrap(
        calculator=object(),
        ts_model=None,
        ref=ref,
        t_ref_s=0.0,
        slab=slab,
    )


@pytest.fixture(autouse=True)
def _noop_dataset_logger(monkeypatch):
    """Avoid DatasetLogger side effects across saturation unit tests."""
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", NoopDatasetLogger
    )


def _patch_single_mol_saturation_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    molecule: str,
    smiles: str,
    ref: DummyReferenceEnergies,
    process_molecule: Callable[..., list],
    slab_energy: float = -10.0,
) -> None:
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._normalize_molecules_input",
        lambda *_a, **_kw: ([(smiles, molecule)], "ok", "<inline-molecules>"),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._bootstrap_screening_run",
        lambda slab, *_a, **_kw: _make_bootstrap_mock(
            molecule=molecule, smiles=smiles, ref=ref, slab=slab
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: slab_energy,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", NoopDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", process_molecule
    )


def _patch_multi_mol_saturation_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    molecules: list[str],
    smiles_list: list[str],
    ref: DummyReferenceEnergies,
    process_molecule: Callable[..., list] | None = None,
    process_molecule_bayesian: Callable[..., list] | None = None,
    slab_energy: float = -100.0,
    budget_fn: Callable[[dict[str, float], int], dict[str, int]] | None = None,
) -> None:
    """Shared monkeypatches for competitive multi-molecule saturation tests."""
    if (process_molecule is None) == (process_molecule_bayesian is None):
        raise ValueError(
            "pass exactly one of process_molecule or process_molecule_bayesian"
        )

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._normalize_molecules_input",
        lambda *_a, **_kw: (
            list(zip(smiles_list, molecules, strict=True)),
            "ok",
            "<inline-molecules>",
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._bootstrap_screening_run",
        lambda slab, *_a, **_kw: ScreeningRunBootstrap(
            calculator=object(),
            ts_model=None,
            ref=ref,
            t_ref_s=0.0,
            slab=slab if isinstance(slab, SlabContainer) else SlabContainer(slab),
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: slab_energy,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger",
        NoopDatasetLogger,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        budget_fn or _uniform_placement_budget,
    )
    if process_molecule is not None:
        monkeypatch.setattr(
            "metalsurfer.workflow.saturation.process_molecule",
            process_molecule,
        )
    else:
        monkeypatch.setattr(
            "metalsurfer.workflow.saturation.process_molecule_bayesian",
            process_molecule_bayesian,
        )


# ---------------------------------------------------------------------------
# load_molecules skip_saturation_file
# ---------------------------------------------------------------------------


def test_load_molecules_skip_saturation_file(workdir):
    """When skip_saturation_file=True, skip molecules in saturation_summary.csv."""
    csv_path = workdir / "smiles.csv"
    csv_path.write_text("O,water\nCCO,ethanol\n")
    results_dir = workdir / "results_manual"
    results_dir.mkdir(exist_ok=True)
    summary = pd.DataFrame({"molecule": ["water"], "n_molecules_at_saturation": [3]})
    summary.to_csv(results_dir / "saturation_summary.csv", index=False)
    molecules, smiles, _ = load_molecules(
        str(csv_path),
        skip_existing=False,
        skip_saturation_file=True,
        surface_type="manual",
    )
    assert "water" not in molecules
    assert "ethanol" in molecules


# ---------------------------------------------------------------------------
# save_saturation_results
# ---------------------------------------------------------------------------


def test_save_saturation_results_empty_list_returns_early(workdir):
    """save_saturation_results with empty list returns without writing."""
    setup_directories(["empty_test"])
    save_saturation_results([], surface_type="empty_test")
    assert not (workdir / "results_empty_test" / "saturation_summary.csv").exists()


def test_save_saturation_results_warns_on_multiple_single_results(workdir, caplog):
    """Multiple single-molecule results in the list trigger a truncation warning."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    sr = make_saturation_run(
        atoms=combined,
        slab_size=len(slab),
        final_slab_atoms=combined.copy(),
    )
    setup_directories(["multi_single_test"])
    with caplog.at_level(logging.WARNING, logger="metalsurfer.io_results"):
        save_saturation_results([sr, sr], surface_type="multi_single_test")
    assert any(
        re.search(r"received 2 single-molecule", r.message) for r in caplog.records
    )


def test_save_saturation_results_writes_csv_and_xyz(workdir):
    """save_saturation_results writes saturation_summary, details, and XYZ."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    best = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-1.0,
        atoms=combined,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    other = make_screening_result(
        molecule="water",
        placement_id=1,
        energy_adsorption=-0.5,
        atoms=combined.copy(),
        slab_size=len(slab),
        distance=2.6,
        placement_descriptor=make_placement_descriptor(placement_id=1),
    )
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=best,
        all_results=[best, other],
        bo_transfer_enabled=True,
        transfer=BOTransferInfo(
            transfer_used=True,
            transfer_disabled_reason=None,
            transfer_weight_share=0.2,
            transfer_bad_rounds=0,
            transfer_last_mae_delta=-0.01,
        ),
    )
    sr = SaturationRunResult(
        molecule="water",
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=combined.copy(),
    )
    setup_directories(["saturation_test"])
    save_saturation_results([sr], surface_type="saturation_test")
    output_dir = workdir / "results_saturation_test"
    summary_path = output_dir / "saturation_summary.csv"
    details_path = output_dir / "saturation_details.csv"
    stable_xyz_path = (
        output_dir / "xyz_structures/water_saturation/step_001_best_slab.xyz"
    )
    eads_xyz = _saturation_step_eads_xyz_name(1, best.energy_adsorption)
    assert_paths_exist(
        output_dir,
        [
            "saturation_summary.csv",
            "saturation_details.csv",
            "saturation_placements_detailed.csv",
            f"xyz_structures/water_saturation/{eads_xyz}",
            "xyz_structures/water_saturation/step_001_best_slab.xyz",
            "xyz_structures/water_saturation/step_001_placements/conformer_000.xyz",
            "xyz_structures/water_saturation/step_001_placements/conformer_001.xyz",
            "xyz_structures/water_saturation/step_001_placements/conformer_000_adsorbate.xyz",
            "xyz_structures/water_saturation/step_001_placements/conformer_001_adsorbate.xyz",
            "xyz_structures/water_saturation/final_saturated_slab.xyz",
        ],
    )
    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    assert summary_df.iloc[0]["molecule"] == "water"
    assert summary_df.iloc[0]["n_molecules_at_saturation"] == 1
    assert str(summary_df.iloc[0]["final_slab_path"]).endswith(
        "water_saturation/final_saturated_slab.xyz"
    )

    details_df = pd.read_csv(details_path)
    assert len(details_df) == 1
    assert details_df.iloc[0]["step"] == 1
    assert bool(details_df.iloc[0]["bo_transfer_enabled"]) is True
    assert bool(details_df.iloc[0]["bo_transfer_used"]) is True
    assert float(details_df.iloc[0]["bo_transfer_weight_share"]) > 0.0
    assert str(details_df.iloc[0]["step_structure_path"]).endswith(
        "water_saturation/step_001_best_slab.xyz"
    )
    step_energy_path = str(details_df.iloc[0]["step_structure_energy_path"])
    assert "water_saturation/step_001_Eads_" in step_energy_path
    assert step_energy_path.endswith(".xyz")

    # extxyz writes preserve full lattice/PBC metadata for downstream reload.
    loaded_step = read(stable_xyz_path)
    cell = loaded_step.get_cell()
    assert cell.lengths()[0] > 0.0
    assert cell.lengths()[1] > 0.0
    assert bool(loaded_step.get_pbc()[0])
    assert bool(loaded_step.get_pbc()[1])

    placements_df = pd.read_csv(output_dir / "saturation_placements_detailed.csv")
    assert len(placements_df) == 2
    assert set(placements_df["placement_id"]) == {0, 1}
    assert "poscar_path" not in placements_df.columns


def test_save_saturation_results_writes_vasp_when_enabled(workdir):
    """VASP bundles are written only when write_vasp_inputs=True."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    best = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-1.0,
        atoms=combined,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=best,
        all_results=[best],
    )
    sr = SaturationRunResult(
        molecule="water",
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=combined.copy(),
    )
    cfg = AdsorptionConfig(write_vasp_inputs=True)
    save_saturation_results([sr], surface_type="saturation_vasp_test", config=cfg)
    output_dir = workdir / "results_saturation_vasp_test"
    assert_paths_exist(
        output_dir,
        [
            "vasp_inputs/water_saturation/step_001/POSCAR",
            "vasp_inputs/water_saturation/step_001_placements/conformer_000/POSCAR",
        ],
    )


def test_save_saturation_results_skips_all_placements_when_disabled(workdir):
    """When saturation_save_all_placements is False, omit step_*_placements tree."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    best = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-1.0,
        atoms=combined,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=best,
        all_results=[best],
        bo_transfer_enabled=False,
        transfer=BOTransferInfo(),
    )
    sr = SaturationRunResult(
        molecule="water",
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=combined.copy(),
    )
    setup_directories(["saturation_no_placements"])
    cfg = AdsorptionConfig(saturation_save_all_placements=False)
    save_saturation_results([sr], surface_type="saturation_no_placements", config=cfg)
    output_dir = workdir / "results_saturation_no_placements"
    assert not (output_dir / "saturation_placements_detailed.csv").exists()
    assert not (
        output_dir / "xyz_structures/water_saturation/step_001_placements"
    ).exists()


# ---------------------------------------------------------------------------
# slab_for_sites uses resized slab (placement coverage)
# ---------------------------------------------------------------------------


def test_saturation_slab_for_sites_uses_resized_slab():
    """Placement site detection uses metal atoms of current slab, not original base.

    When auto_resize expands the slab, hollow sites must be computed from the
    resized slab so placements span the full cell. This test verifies that
    resizing increases hollow site count (sites scale with unit cells).

    Uses a deliberately small slab (2x2x2, cell ~5.4 A) and min_separation=8.0
    so that resize is required (5.4 < 0.74 + 8 = 8.74).
    """
    slab = SlabContainer(make_slab(nx=2, ny=2, n_layers=2))
    cell_diag = min(
        slab.atoms.get_cell()[0, 0],
        slab.atoms.get_cell()[1, 1],
    )
    h2 = make_h2()
    min_sep = 8.0
    required = 0.74 + min_sep
    assert cell_diag < required, (
        f"Test setup: slab cell {cell_diag:.1f} A must be < {required:.1f} A "
        "to trigger resize"
    )
    resized, was_resized = auto_resize_substrate_for_molecule(
        slab, [h2], min_separation=min_sep
    )
    assert was_resized, (
        "Slab must be resized for this test; check auto_resize_substrate_for_molecule logic"
    )
    sites_original = get_hollow_sites_for_adatoms(slab.atoms)
    sites_resized = get_hollow_sites_for_adatoms(resized.atoms)
    assert len(sites_resized) > len(sites_original), (
        "Resized slab should have more hollow sites than original"
    )


def test_saturation_validate_posed_adsorbate_overlap_reason():
    """Under coverage, clash with prior adsorbate yields adsorbate_overlap."""
    from metalsurfer.placement.pose import _validate_posed_adsorbate

    slab = make_slab(nx=2, ny=2, n_layers=3)
    water = Atoms(
        "OH2",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    pos = water.get_positions()
    pos -= np.mean(pos, axis=0)
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.2
    pos[:, 0] += 2.0
    pos[:, 1] += 2.0
    water.set_positions(pos)
    covered = slab + water
    reason = _validate_posed_adsorbate(
        water.copy(), covered, AdsorptionConfig(), slab_for_sites=slab
    )
    assert reason == "adsorbate_overlap"


# ---------------------------------------------------------------------------
# run_saturation_screening (real GPU integration test)
# ---------------------------------------------------------------------------


def test_slab_after_saturation_step_restores_material_pbc():
    """Calculator-time [T,T,T] PBC must not leak into the next saturation step."""
    slab = make_slab()
    combined = slab.copy()
    combined.set_pbc([True, True, True])
    config = AdsorptionConfig(material_type="slab")

    restored = _slab_after_saturation_step(combined, config)

    assert list(restored.atoms.get_pbc()) == [True, True, False]


@gpu_mlip_test
def test_run_saturation_screening_h2_ni111_real_gpu(workdir):
    """Saturation screening with real MLIP on GPU: H2 on Ni(111).

    Runs actual run_saturation_screening (no mocks). Verifies:
    - Saturation loop terminates (capped by saturation_max_steps for bounded runtime)
    - Steps and n_molecules_at_saturation are consistent
    - All E_ads in steps are physically reasonable
    """
    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        seed=42,
        num_conformers=1,
        num_placements=6,
        device="cuda",
        material_type="slab",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=16,
        stage2_steps=80,
        saturation_max_steps=8,
    )
    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir="results_test_saturation",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n")
        smiles_path = f.name

    try:
        results = run_saturation_screening(
            slab,
            molecules=smiles_path,
            config=config,
            surface_type="test_saturation",
            skip_existing=False,
        )
    finally:
        os.unlink(smiles_path)

    if not results:
        pytest.fail(
            "Saturation screening returned no results; expected at least one "
            "SaturationRunResult for H2 on Ni(111)"
        )

    assert len(results) == 1
    sr = results[0]
    assert sr.molecule == "H2"
    if not sr.steps:
        pytest.fail(
            "Saturation produced zero steps; expected at least one recorded step"
        )

    # Saturation logic: last step either has E_ads >= 0 (stopped) or E_ads < 0 (added)
    last = sr.steps[-1]
    if last.best_result.energy_adsorption >= 0:
        n_at_sat = last.n_molecules_on_slab
    else:
        n_at_sat = last.n_molecules_on_slab + 1
    assert sr.n_molecules_at_saturation == n_at_sat

    # All E_ads should be physically reasonable (not wildly unphysical)
    for step in sr.steps:
        e = step.best_result.energy_adsorption
        assert -5.0 <= e <= 5.0, f"E_ads {e:.3f} eV out of reasonable range"


def test_run_saturation_screening_symmetry_none_falls_back_to_c1(monkeypatch, caplog):
    """Saturation continues with full site sampling when symmetry fails."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config()

    monkeypatch.setattr(
        "metalsurfer.symmetry.SymmetryAnalyzer.detect_symmetry_breaking",
        lambda *_a, **_kw: (_ for _ in ()).throw(
            SymmetryAnalysisError("spglib.get_symmetry_dataset returned None")
        ),
    )

    symmetry_flags: list[bool] = []

    def _fake_process_molecule(
        _smi,
        mol,
        current_slab,
        *_args,
        symmetry_broken=False,
        **_kwargs,
    ):
        symmetry_flags.append(bool(symmetry_broken))
        step_idx = len(symmetry_flags)
        e_ads = -0.3 if step_idx == 1 else 0.2
        return _result_for_step(mol, current_slab, e_ads)

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=REF_CONSTANT),
        process_molecule=_fake_process_molecule,
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            molecules="unused.csv",
            config=config,
            surface_type="symmetry_fallback",
            skip_existing=False,
        )

    assert len(out) == 1
    assert len(out[0].steps) == 2
    assert symmetry_flags == [False, True]
    assert any("assuming C1" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# Multi-molecule saturation: budget distribution
# ---------------------------------------------------------------------------


def test_distribute_placement_budget_proportional():
    """Budget is split proportionally to complexity scores."""
    budgets = distribute_placement_budget({"A": 100.0, "B": 400.0}, 250)
    assert budgets["A"] + budgets["B"] == 250
    # B has 4× complexity → should receive roughly 4× the placements
    assert budgets["B"] > budgets["A"]


def test_distribute_placement_budget_sums_to_total():
    """Allocations always sum to exactly total_budget regardless of rounding."""
    for total in (10, 11, 13, 100, 250):
        budgets = distribute_placement_budget({"X": 1.0, "Y": 3.0, "Z": 2.0}, total)
        assert sum(budgets.values()) == total, (
            f"total={total}: sum={sum(budgets.values())}"
        )


def test_distribute_placement_budget_min_one():
    """Every molecule gets at least 1 placement even with tiny complexity."""
    budgets = distribute_placement_budget(
        {"tiny": 1.0, "huge": 10000.0}, total_budget=5
    )
    assert budgets["tiny"] >= 1
    assert budgets["huge"] >= 1
    assert sum(budgets.values()) == 5


def test_distribute_placement_budget_below_n_keeps_top_complexity():
    """When budget < n molecules, keep top-budget by complexity (1 each)."""
    budgets = distribute_placement_budget(
        {"low": 1.0, "mid": 10.0, "high": 100.0},
        total_budget=2,
    )
    assert budgets == {"high": 1, "mid": 1}
    assert "low" not in budgets
    budgets_one = distribute_placement_budget(
        {"a": 5.0, "b": 50.0, "c": 1.0},
        total_budget=1,
    )
    assert budgets_one == {"b": 1}


def test_saturation_symmetry_uses_substrate_prefix_not_adsorbates():
    """Covered slab must not auto-break symmetry vs bare base (L1)."""
    base = make_slab()
    covered = base.copy()
    covered.extend(
        Atoms(
            "O",
            positions=[[1.0, 1.0, float(np.max(base.positions[:, 2])) + 2.0]],
        )
    )
    substrate = _build_surface_reference_slab(covered, base)
    assert not _saturation_symmetry_broken_vs_reference(
        substrate,
        base,
        symmetry_tolerance=0.1,
    )
    # Full covered vs bare still breaks (historical adsorbate contamination).
    assert _saturation_symmetry_broken_vs_reference(
        covered,
        base,
        symmetry_tolerance=0.1,
    )


def test_distribute_placement_budget_extreme_skew_sums_to_total():
    """Min-1 floor must not overshoot total_budget under extreme score skew."""
    complexities = {"A": 1000.0, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}
    budgets = distribute_placement_budget(complexities, total_budget=5)
    assert sum(budgets.values()) == 5
    assert all(v >= 1 for v in budgets.values())
    budgets4 = distribute_placement_budget(
        {f"m{i}": (1000.0 if i == 0 else 1.0) for i in range(4)},
        total_budget=4,
    )
    assert sum(budgets4.values()) == 4
    assert all(v >= 1 for v in budgets4.values())


@pytest.mark.parametrize(
    "complexities,total,expected",
    [
        ({"H2": 50.0}, 100, lambda b: b["H2"] == 100),
        (
            {"A": 100.0, "B": 100.0},
            10,
            lambda b: sum(b.values()) == 10 and abs(b["A"] - b["B"]) <= 1,
        ),
    ],
)
def test_distribute_placement_budget_edge_cases(complexities, total, expected):
    budgets = distribute_placement_budget(complexities, total_budget=total)
    assert expected(budgets)


# ---------------------------------------------------------------------------
# Multi-molecule saturation: estimate_molecule_complexity
# ---------------------------------------------------------------------------


def test_estimate_molecule_complexity_positive():
    """Complexity score must be >= 1.0 for any valid molecule."""
    slab = make_slab()
    # minimal linear molecule (CO-like)
    linear = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    config = AdsorptionConfig(num_conformers=1, num_placements=50)
    score = estimate_molecule_complexity([linear], slab, config, smiles="[C-]#[O+]")
    assert score >= 1.0


def test_estimate_molecule_complexity_more_conformers_higher_score():
    """More conformers always give a higher complexity score (n_conformers is a direct multiplier)."""
    slab = make_slab()
    config = AdsorptionConfig(num_conformers=3, num_placements=50)

    mol = make_h2()

    score_one = estimate_molecule_complexity([mol], slab, config, smiles="[H][H]")
    score_three = estimate_molecule_complexity(
        [mol, mol, mol], slab, config, smiles="[H][H]"
    )

    assert score_three > score_one, (
        f"Three-conformer score ({score_three}) should exceed single-conformer ({score_one})"
    )


# ---------------------------------------------------------------------------
# Multi-molecule saturation: loop logic (mocked)
# ---------------------------------------------------------------------------


def test_multi_mol_saturation_picks_best_across_molecules(monkeypatch):
    """The step winner is the molecule with the overall lowest E_ads."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies(REF_WATER_CO2)

    fake_process = _make_schedule_process({"water": [-0.5, 0.1], "CO2": [-1.2, 0.1]})

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["water", "CO2"],
        smiles_list=["O", "O=C=O"],
        ref=ref,
        process_molecule=fake_process,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="multi_mol_test",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert isinstance(result, MultiMolSaturationRunResult)
    assert len(result.steps) == 2
    assert result.steps[0].winning_molecule == "CO2"


def test_multi_mol_saturation_terminates_on_positive_eads(monkeypatch):
    """Multi-mol saturation stops when best E_ads >= 0."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies(REF_A_B)

    fake_process = _make_schedule_process({"A": [0.5], "B": [0.5]})

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["A", "B"],
        smiles_list=["smiles_a", "smiles_b"],
        ref=ref,
        process_molecule=fake_process,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="multi_mol_termination",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    # Exactly 1 step before saturation (positive E_ads on step 1)
    assert len(result.steps) == 1
    assert result.steps[0].best_result.energy_adsorption >= 0
    assert result.n_molecules_at_saturation == 0


def test_multi_mol_saturation_step_result_structure(monkeypatch):
    """MultiMolSaturationStepResult contains expected per-molecule fields."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies({"mol1": -5.0, "mol2": -5.0})

    call_count = [0]

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_count[0] += 1
        e_ads = -0.3 if call_count[0] <= 2 else 0.2
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=e_ads,
                    atoms=current_slab.atoms.copy(),
                    slab_size=len(current_slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                )
            ]
        )

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["mol1", "mol2"],
        smiles_list=["s1", "s2"],
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="step_result_structure",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert isinstance(result, MultiMolSaturationRunResult)
    step = result.steps[0]
    assert isinstance(step, MultiMolSaturationStepResult)
    assert step.step == 1
    assert step.winning_molecule in {"mol1", "mol2"}
    assert set(step.per_molecule_results.keys()) == {"mol1", "mol2"}
    assert set(step.per_molecule_budgets.keys()) == {"mol1", "mol2"}
    assert isinstance(result.molecule_counts, dict)


def test_multi_mol_saturation_single_molecule_fallback(monkeypatch, caplog):
    """When multi_molecule_saturation=True but only 1 molecule, falls back gracefully."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies({"water": -5.0})

    call_count = [0]

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_count[0] += 1
        e_ads = -0.5 if call_count[0] == 1 else 0.1
        return _result_for_step(mol, current_slab, e_ads)

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            molecules="unused.csv",
            config=config,
            surface_type="single_mol_fallback",
            skip_existing=False,
        )

    assert len(out) == 1
    assert isinstance(out[0], SaturationRunResult)
    assert any("falling back" in rec.getMessage().lower() for rec in caplog.records)


def test_multi_mol_saturation_molecule_counts_tracked(monkeypatch):
    """molecule_counts tracks how many steps each molecule won."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(
        multi_molecule_saturation=True,
        saturation_max_steps=2,
    )

    ref = DummyReferenceEnergies(REF_A_B)

    call_index = [0]
    energy_table = {
        ("A", 1): -0.8,
        ("B", 2): -0.5,
        ("A", 3): -0.3,
        ("B", 4): -0.7,
    }

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_index[0] += 1
        key = (mol, call_index[0])
        assert key in energy_table, f"unexpected saturation call {key}"
        e_ads = energy_table[key]
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=e_ads,
                    atoms=current_slab.atoms.copy(),
                    slab_size=len(current_slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                )
            ]
        )

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["A", "B"],
        smiles_list=["sa", "sb"],
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="molecule_counts_test",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert len(result.steps) == 2
    assert result.molecule_counts == {"A": 1, "B": 1}


def test_multi_mol_saturation_molecule_counts_omit_unbound_final_step(monkeypatch):
    """Unbound terminal winners must not inflate molecule_counts vs molecules on slab."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(
        multi_molecule_saturation=True,
        saturation_max_steps=3,
    )

    ref = DummyReferenceEnergies(REF_A_B)

    call_index = [0]
    # Step 1: A wins bound. Step 2: B wins unbound → stop. Counts must be A=1 only.
    energy_table = {
        ("A", 1): -0.8,
        ("B", 2): -0.5,
        ("A", 3): 0.2,
        ("B", 4): 0.1,
    }

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_index[0] += 1
        key = (mol, call_index[0])
        assert key in energy_table, f"unexpected saturation call {key}"
        e_ads = energy_table[key]
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=e_ads,
                    atoms=current_slab.atoms.copy(),
                    slab_size=len(current_slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                )
            ]
        )

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["A", "B"],
        smiles_list=["sa", "sb"],
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="molecule_counts_unbound",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert len(result.steps) == 2
    assert result.steps[-1].best_result.energy_adsorption >= 0
    assert result.molecule_counts == {"A": 1, "B": 0}
    assert sum(result.molecule_counts.values()) == result.n_molecules_at_saturation == 1


def test_multi_mol_saturation_bo_uses_independent_memory_per_adsorbate(monkeypatch):
    """Competing BO saturation carries BO state forward independently per molecule."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies(REF_WATER_CO2)

    call_counts: dict[str, int] = {"water": 0, "CO2": 0}
    prior_memory_seen: dict[str, list[BOStepMemory | None]] = {"water": [], "CO2": []}

    def _fake_process_molecule_bayesian(
        _smi,
        mol,
        current_slab,
        *_args,
        bo_step_memory_in=None,
        **_kwargs,
    ):
        call_counts[mol] += 1
        prior_memory_seen[mol].append(bo_step_memory_in)

        step_idx = call_counts[mol]
        if step_idx == 1:
            assert bo_step_memory_in is None
        else:
            assert bo_step_memory_in is not None
            assert bo_step_memory_in.observed_X_rows == [
                {"step": float(step_idx - 1), "mol_len": float(len(mol))}
            ]
            assert bo_step_memory_in.observed_y == [float(step_idx - 1)]
            assert bo_step_memory_in.best_energy == pytest.approx(-0.1 * (step_idx - 1))

        memory = BOStepMemory(
            observed_X_rows=[{"step": float(step_idx), "mol_len": float(len(mol))}],
            observed_y=[float(step_idx)],
            best_energy=-0.1 * step_idx,
        )

        energies = {
            "water": [-0.5, 0.4],
            "CO2": [-1.2, 0.3],
        }
        e_ads = energies[mol][step_idx - 1]
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=step_idx,
                    energy_adsorption=e_ads,
                    atoms=current_slab.atoms.copy(),
                    slab_size=len(current_slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(
                        placement_id=step_idx
                    ),
                )
            ],
            bo_memory=memory,
            transfer_info=BOTransferInfo(),
        )

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["water", "CO2"],
        smiles_list=["O", "O=C=O"],
        ref=ref,
        process_molecule_bayesian=_fake_process_molecule_bayesian,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="multi_mol_bo_memory",
        skip_existing=False,
        bo_enabled=True,
    )

    assert len(out) == 1
    result = out[0]
    assert len(result.steps) == 2
    assert result.steps[0].winning_molecule == "CO2"
    assert result.steps[1].best_result.energy_adsorption >= 0
    assert prior_memory_seen["water"][0] is None
    assert prior_memory_seen["CO2"][0] is None
    assert prior_memory_seen["water"][1] is not prior_memory_seen["CO2"][1]


def test_multi_mol_saturation_bo_rejects_shared_memory_objects(monkeypatch):
    """Competing BO saturation rejects shared BOStepMemory objects across adsorbates."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies(REF_A_B)

    shared_memory = BOStepMemory(
        observed_X_rows=[{"step": 1.0}],
        observed_y=[1.0],
        best_energy=-0.5,
    )

    def _fake_process_molecule_bayesian(
        _smi,
        mol,
        current_slab,
        *_args,
        **_kwargs,
    ):
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=-0.2,
                    atoms=current_slab.atoms.copy(),
                    slab_size=len(current_slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                )
            ],
            bo_memory=shared_memory,
            transfer_info=BOTransferInfo(),
        )

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["A", "B"],
        smiles_list=["sa", "sb"],
        ref=ref,
        process_molecule_bayesian=_fake_process_molecule_bayesian,
    )

    with pytest.raises(RuntimeError, match="independent per adsorbate"):
        run_saturation_screening(
            slab,
            molecules="unused.csv",
            config=config,
            surface_type="multi_mol_bo_shared_memory",
            skip_existing=False,
            bo_enabled=True,
        )


@gpu_mlip_test
def test_run_saturation_screening_multi_mol_bo_real_gpu(workdir):
    """Smoke-level GPU integration test for BO-enabled competing saturation."""
    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        task_name="oc20",
        seed=42,
        num_conformers=1,
        num_placements=2,
        device="cuda",
        material_type="slab",
        multi_molecule_saturation=True,
        bo=BOConfig(initial_random=1, batch_size=1, total_budget=2),
        saturation_max_steps=1,
        skip_topology_check=True,
        skip_desorption_check=False,
        # Short stages leave residual forces ~0.07–0.6 eV/Å on this smoke;
        # open the force gate so BO can still record a competitive step.
        max_force_convergence=0.7,
        stage1_steps=20,
        stage2_steps=60,
    )
    slab = prepare_substrate(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(3, 3, 1),
        config=config,
        results_dir="results_test_multi_mol_bo_gpu",
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n[C-]#[O+],CO\n")
        smiles_path = f.name

    try:
        results = run_saturation_screening(
            slab,
            molecules=smiles_path,
            config=config,
            surface_type="test_multi_mol_bo_gpu",
            skip_existing=False,
            bo_enabled=True,
        )
    finally:
        os.unlink(smiles_path)

    assert len(results) == 1
    result = results[0]

    assert isinstance(result, MultiMolSaturationRunResult)
    if not result.steps:
        pytest.fail(
            "Multi-molecule BO saturation produced zero steps; expected at least "
            "one competitive placement attempt"
        )
    step0 = result.steps[0]
    # Keys match molecules that got conformers and entered the competitive loop.
    assert set(step0.per_molecule_results) == set(step0.transfer_by_molecule)
    assert any(len(v) >= 1 for v in step0.per_molecule_results.values()), (
        "Expected at least one competitive adsorbate with valid placements; "
        f"got counts: { {k: len(v) for k, v in step0.per_molecule_results.items()} }"
    )


# ---------------------------------------------------------------------------
# Multi-molecule saturation: I/O
# ---------------------------------------------------------------------------


def test_save_multi_mol_saturation_results_writes_csv(workdir):
    """save_multi_mol_saturation_results writes summary, details, and XYZ."""
    slab = make_slab()
    mol_a_atoms = place_molecule_on_slab(slab, make_water())
    best_a = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-0.8,
        atoms=mol_a_atoms,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    second_water = make_screening_result(
        molecule="water",
        placement_id=1,
        energy_adsorption=-0.4,
        atoms=mol_a_atoms.copy(),
        slab_size=len(slab),
        distance=2.6,
        placement_descriptor=make_placement_descriptor(placement_id=1),
    )
    mol_b_atoms = place_molecule_on_slab(slab, make_water())
    best_co2 = make_screening_result(
        molecule="CO2",
        placement_id=0,
        energy_adsorption=-0.3,
        atoms=mol_b_atoms,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    step = MultiMolSaturationStepResult(
        step=1,
        winning_molecule="water",
        n_molecules_on_slab=0,
        best_result=best_a,
        per_molecule_results={"water": [best_a, second_water], "CO2": [best_co2]},
        per_molecule_budgets={"water": 75, "CO2": 25},
        bo_transfer_enabled=True,
        transfer_by_molecule={
            "water": BOTransferInfo(
                transfer_used=True,
                transfer_weight_share=0.2,
                transfer_bad_rounds=0,
                transfer_last_mae_delta=-0.01,
            ),
            "CO2": BOTransferInfo(
                transfer_used=False,
                transfer_disabled_reason="insufficient_overlap",
            ),
        },
    )
    result = MultiMolSaturationRunResult(
        molecules=["water", "CO2"],
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=mol_a_atoms.copy(),
        molecule_counts={"water": 1, "CO2": 0},
    )

    setup_directories(["multi_mol_io_test"])
    save_multi_mol_saturation_results(result, surface_type="multi_mol_io_test")

    output_dir = workdir / "results_multi_mol_io_test"
    summary_path = output_dir / "saturation_summary.csv"
    details_path = output_dir / "saturation_details.csv"
    mol_label = _saturation_molecule_label(["water", "CO2"])
    assert_paths_exist(
        output_dir,
        [
            "saturation_summary.csv",
            "saturation_details.csv",
            "saturation_placements_detailed.csv",
            f"xyz_structures/{mol_label}_saturation",
            f"xyz_structures/{mol_label}_saturation/step_001_placements/water/conformer_000.xyz",
            f"xyz_structures/{mol_label}_saturation/step_001_placements/water/conformer_001.xyz",
            f"xyz_structures/{mol_label}_saturation/step_001_placements/CO2/conformer_000.xyz",
        ],
    )

    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    assert mol_label in str(summary_df.iloc[0]["molecules"])
    assert int(summary_df.iloc[0]["n_molecules_at_saturation"]) == 1

    details_df = pd.read_csv(details_path)
    assert len(details_df) == 1
    assert details_df.iloc[0]["winning_molecule"] == "water"
    assert "water" in str(details_df.iloc[0]["per_molecule_budgets"])
    assert bool(details_df.iloc[0]["bo_transfer_enabled"]) is True
    assert bool(details_df.iloc[0]["bo_transfer_used"]) is True
    assert float(details_df.iloc[0]["bo_transfer_weight_share"]) == pytest.approx(0.2)

    placements_df = pd.read_csv(output_dir / "saturation_placements_detailed.csv")
    assert len(placements_df) == 3
    assert set(placements_df["molecule"]) == {"water", "CO2"}
    water_rows = placements_df[placements_df["molecule"] == "water"]
    co2_rows = placements_df[placements_df["molecule"] == "CO2"]
    assert bool(water_rows.iloc[0]["bo_transfer_used"]) is True
    assert float(water_rows.iloc[0]["bo_transfer_weight_share"]) == pytest.approx(0.2)
    assert bool(co2_rows.iloc[0]["bo_transfer_used"]) is False
    assert (
        str(co2_rows.iloc[0]["bo_transfer_disabled_reason"]) == "insufficient_overlap"
    )

    # Second combo into the same surface_type must preserve the first summary row.
    other_atoms = place_molecule_on_slab(slab, make_water())
    other_best = make_screening_result(
        molecule="ethanol",
        placement_id=0,
        energy_adsorption=-0.5,
        atoms=other_atoms,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    other_step = MultiMolSaturationStepResult(
        step=1,
        winning_molecule="ethanol",
        n_molecules_on_slab=0,
        best_result=other_best,
        per_molecule_results={"ethanol": [other_best]},
        per_molecule_budgets={"ethanol": 50, "methanol": 50},
        bo_transfer_enabled=False,
    )
    other = MultiMolSaturationRunResult(
        molecules=["ethanol", "methanol"],
        steps=[other_step],
        n_molecules_at_saturation=1,
        final_slab_atoms=other_atoms.copy(),
        molecule_counts={"ethanol": 1, "methanol": 0},
    )
    save_multi_mol_saturation_results(other, surface_type="multi_mol_io_test")
    summary_after = pd.read_csv(summary_path)
    labels = set(summary_after["molecules"].astype(str))
    assert mol_label in labels
    assert _saturation_molecule_label(["ethanol", "methanol"]) in labels
    assert len(summary_after) == 2


def test_save_saturation_results_dispatches_multi_mol(workdir):
    """save_saturation_results([MultiMolSaturationRunResult]) delegates to multi-mol I/O."""
    slab = make_slab()
    mol_a_atoms = place_molecule_on_slab(slab, make_water())
    best_a = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-0.8,
        atoms=mol_a_atoms,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    step = MultiMolSaturationStepResult(
        step=1,
        winning_molecule="water",
        n_molecules_on_slab=0,
        best_result=best_a,
        per_molecule_results={"water": [best_a]},
        per_molecule_budgets={"water": 100},
        bo_transfer_enabled=False,
    )
    result = MultiMolSaturationRunResult(
        molecules=["water"],
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=mol_a_atoms.copy(),
        molecule_counts={"water": 1},
    )
    setup_directories(["saturation_dispatch_mm"])
    save_saturation_results([result], surface_type="saturation_dispatch_mm")
    out = workdir / "results_saturation_dispatch_mm"
    assert (out / "saturation_summary.csv").exists()
    assert (
        out
        / "xyz_structures/water_saturation/step_001_placements/water/conformer_000.xyz"
    ).exists()


# ---------------------------------------------------------------------------
# Saturation topology rearrangement guard
# ---------------------------------------------------------------------------


def _slab_with_two_waters(*, separated: bool) -> tuple[Atoms, Atoms, Atoms]:
    """Return (bare_slab, slab_plus_one_water, two_water_structure)."""
    slab = make_slab()
    first = place_molecule_on_slab(
        slab, make_water(), z_offset=3.0, x_shift=2.0, y_shift=2.0
    )
    w2 = make_water().copy()
    slab_z = float(np.max(slab.get_positions()[:, 2]))
    pos2 = w2.get_positions().copy()
    pos2 -= np.mean(pos2, axis=0)
    ads_pos = first.get_positions()[len(slab) :]
    anchor = np.mean(ads_pos, axis=0)
    if separated:
        pos2[:, 0] += 7.0
        pos2[:, 1] += 2.0
    else:
        pos2[:, 0] = anchor[0] + 1.2
        pos2[:, 1] = anchor[1]
    pos2[:, 2] += slab_z + 3.0
    w2.set_positions(pos2)
    combined = first + w2
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    return slab, first, combined


def test_saturation_topology_guard_validates_separated_waters():
    slab, _first, combined = _slab_with_two_waters(separated=True)
    config = AdsorptionConfig(connectivity_multiplier=1.3)
    ok, reason = _saturation_adsorbate_topology_ok(
        combined,
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert ok, reason


def test_saturation_topology_guard_is_connectivity_only():
    slab, _first, separated = _slab_with_two_waters(separated=True)
    # Remove one H from the second adsorbate unit: still two fragments, but
    # per-fragment SMILES checks would reject this against ["O", "O"].
    mismatched_units = separated[:-1]
    config = AdsorptionConfig(connectivity_multiplier=1.3)
    ok, reason = _saturation_adsorbate_topology_ok(
        mismatched_units,
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert ok, reason
    assert reason == ""


def test_saturation_topology_guard_rejects_coupled_waters():
    slab, _first, combined = _slab_with_two_waters(separated=False)
    config = AdsorptionConfig(connectivity_multiplier=1.3)
    ok, reason = _saturation_adsorbate_topology_ok(
        combined,
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert not ok
    assert "expected 2 adsorbate units" in reason


def test_filter_saturation_topology_results_keeps_lower_energy_intact():
    slab, slab_plus_one, separated = _slab_with_two_waters(separated=True)
    _, _, coupled = _slab_with_two_waters(separated=False)
    config = AdsorptionConfig(connectivity_multiplier=1.3)
    bad = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-2.0,
        atoms=coupled,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    good = make_screening_result(
        molecule="water",
        placement_id=1,
        energy_adsorption=-1.0,
        atoms=separated,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=1),
    )
    filtered = _filter_saturation_topology_results(
        [bad, good],
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 1


def test_saturation_topology_guard_disabled_preserves_ranking():
    slab, _, coupled = _slab_with_two_waters(separated=False)
    _, _, separated = _slab_with_two_waters(separated=True)
    config = AdsorptionConfig(
        connectivity_multiplier=1.3,
        saturation_discard_topology_rearrangements=False,
    )
    bad = make_screening_result(
        molecule="water",
        placement_id=0,
        energy_adsorption=-2.0,
        atoms=coupled,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    good = make_screening_result(
        molecule="water",
        placement_id=1,
        energy_adsorption=-1.0,
        atoms=separated,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=1),
    )
    filtered = _filter_saturation_topology_results(
        [bad, good],
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert len(filtered) == 2
    assert min(filtered, key=lambda r: r.energy_adsorption).placement_id == 0


def test_saturation_step2_selects_intact_not_rearranged(monkeypatch):
    """Step-2 best-slab selection skips rearranged low-energy candidates."""
    slab = SlabContainer(make_slab())
    bare, slab_plus_one, _ = _slab_with_two_waters(separated=True)
    _, _, coupled = _slab_with_two_waters(separated=False)
    _, _, separated = _slab_with_two_waters(separated=True)
    config = AdsorptionConfig(connectivity_multiplier=1.3)

    call_count = [0]

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MoleculeScreenOutcome(
                results=[
                    make_screening_result(
                        molecule=mol,
                        placement_id=0,
                        energy_adsorption=-0.5,
                        atoms=slab_plus_one.copy(),
                        slab_size=len(bare),
                        distance=2.5,
                        placement_descriptor=make_placement_descriptor(placement_id=0),
                    )
                ]
            )
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=-2.0,
                    atoms=coupled.copy(),
                    slab_size=len(slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                ),
                make_screening_result(
                    molecule=mol,
                    placement_id=1,
                    energy_adsorption=-1.0,
                    atoms=separated.copy(),
                    slab_size=len(slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=1),
                ),
            ]
        )

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=REF_CONSTANT),
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="topology_guard_single",
        skip_existing=False,
    )

    assert len(out) == 1
    assert len(out[0].steps) == 2
    assert out[0].steps[1].best_result.placement_id == 1
    assert np.allclose(
        out[0].steps[1].best_result.atoms.get_positions(),
        separated.get_positions(),
    )


def test_multi_mol_saturation_topology_guard_step2(monkeypatch):
    """Multi-molecule step-2 ranking uses topology-valid candidates only."""
    slab = SlabContainer(make_slab())
    _, _, coupled = _slab_with_two_waters(separated=False)
    _, _, separated = _slab_with_two_waters(separated=True)
    config = AdsorptionConfig(
        multi_molecule_saturation=True,
        connectivity_multiplier=1.3,
        num_placements=100,
    )

    step_idx = [0]

    bare, slab_plus_one, _ = _slab_with_two_waters(separated=True)

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        step_idx[0] += 1
        if step_idx[0] <= 2:
            if step_idx[0] == 1 and mol == "water":
                return MoleculeScreenOutcome(
                    results=[
                        make_screening_result(
                            molecule=mol,
                            placement_id=0,
                            energy_adsorption=-0.4,
                            atoms=slab_plus_one.copy(),
                            slab_size=len(bare),
                            distance=2.5,
                            placement_descriptor=make_placement_descriptor(
                                placement_id=0
                            ),
                        )
                    ]
                )
            return _result_for_step(mol, current_slab, -0.4)
        if mol == "water":
            return MoleculeScreenOutcome(
                results=[
                    make_screening_result(
                        molecule=mol,
                        placement_id=0,
                        energy_adsorption=-2.0,
                        atoms=coupled.copy(),
                        slab_size=len(slab.atoms),
                        distance=2.5,
                        placement_descriptor=make_placement_descriptor(placement_id=0),
                    ),
                    make_screening_result(
                        molecule=mol,
                        placement_id=1,
                        energy_adsorption=-1.0,
                        atoms=separated.copy(),
                        slab_size=len(slab.atoms),
                        distance=2.5,
                        placement_descriptor=make_placement_descriptor(placement_id=1),
                    ),
                ]
            )
        return _result_for_step(mol, current_slab, -0.2)

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["water", "CO2"],
        smiles_list=["O", "O=C=O"],
        ref=DummyReferenceEnergies(REF_WATER_CO2),
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=config,
        surface_type="topology_guard_multi",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert len(result.steps) >= 2
    step2 = result.steps[1]
    assert step2.winning_molecule == "water"
    assert step2.best_result.placement_id == 1


def test_reference_smiles_units_multi_molecule_counts_placing_unit():
    """The screened candidate includes the unit being placed, so +1 it."""
    active_molecules = ["water", "CO2"]
    active_smiles = {"water": "O", "CO2": "O=C=O"}
    units = _reference_smiles_units_multi_molecule(
        active_molecules,
        active_smiles,
        molecule_counts={"water": 2, "CO2": 1},
        placing_molecule="CO2",
    )
    assert units == ["O", "O", "O=C=O", "O=C=O"]


def test_reference_smiles_units_multi_molecule_fresh_slab():
    """Even with nothing yet on the slab, placing a unit yields one entry."""
    units = _reference_smiles_units_multi_molecule(
        ["water", "CO2"],
        {"water": "O", "CO2": "O=C=O"},
        molecule_counts={"water": 0, "CO2": 0},
        placing_molecule="water",
    )
    assert units == ["O"]


def test_reference_smiles_units_multi_molecule_parity_with_single():
    """For one active molecule the length must equal ``step`` (one per unit placed)."""
    step = 3
    for placing in ("water", "CO2"):
        molecule_counts = (
            {"water": step - 1, "CO2": 0}
            if placing == "water"
            else {
                "water": 0,
                "CO2": step - 1,
            }
        )
        units = _reference_smiles_units_multi_molecule(
            ["water", "CO2"],
            {"water": "O", "CO2": "O=C=O"},
            molecule_counts=molecule_counts,
            placing_molecule=placing,
        )
        assert len(units) == step


def test_saturation_topology_guard_all_filtered_stops(monkeypatch, caplog):
    """If every candidate rearranges, saturation stops without advancing."""
    slab = SlabContainer(make_slab())
    bare, slab_plus_one, _ = _slab_with_two_waters(separated=True)
    _, _, coupled = _slab_with_two_waters(separated=False)
    config = AdsorptionConfig(connectivity_multiplier=1.3)

    call_count = [0]

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return MoleculeScreenOutcome(
                results=[
                    make_screening_result(
                        molecule=mol,
                        placement_id=0,
                        energy_adsorption=-0.5,
                        atoms=slab_plus_one.copy(),
                        slab_size=len(bare),
                        distance=2.5,
                        placement_descriptor=make_placement_descriptor(placement_id=0),
                    )
                ]
            )
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=-2.0,
                    atoms=coupled.copy(),
                    slab_size=len(slab.atoms),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                ),
            ]
        )

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=REF_CONSTANT),
        process_molecule=_fake_process_molecule,
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            molecules="unused.csv",
            config=config,
            surface_type="topology_guard_stop",
            skip_existing=False,
        )

    assert len(out) == 1
    assert len(out[0].steps) == 1
    assert any(
        "topology rearrangement guard" in rec.getMessage() for rec in caplog.records
    )


def test_single_mol_saturation_resolves_workload_config_once(monkeypatch):
    """Unset placement sizes must autotune once, then reuse the written-back config."""
    from dataclasses import replace

    slab = SlabContainer(make_slab())
    bare = slab.atoms.copy()
    resolve_calls: list[int] = []

    def _fake_resolve(config, **_kwargs):
        resolve_calls.append(1)
        return replace(config, num_placements=4)

    def _fake_process(_smi, mol, current_slab, *_args, **kwargs):
        assert kwargs.get("skip_workload_autotune") is True
        assert kwargs["config"].num_placements == 4
        return MoleculeScreenOutcome(
            results=[
                make_screening_result(
                    molecule=mol,
                    placement_id=0,
                    energy_adsorption=-0.5,
                    atoms=place_molecule_on_slab(current_slab.atoms, make_water()),
                    slab_size=len(bare),
                    distance=2.5,
                    placement_descriptor=make_placement_descriptor(placement_id=0),
                )
            ]
        )

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=REF_CONSTANT),
        process_molecule=_fake_process,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.resolve_saturation_step_workload_config",
        _fake_resolve,
    )

    out = run_saturation_screening(
        slab,
        molecules="unused.csv",
        config=AdsorptionConfig(
            num_placements=None,
            saturation_max_steps=2,
            saturation_discard_topology_rearrangements=False,
        ),
        surface_type="autotune_once",
        skip_existing=False,
    )

    assert len(resolve_calls) == 1
    assert len(out) == 1
    assert len(out[0].steps) == 2

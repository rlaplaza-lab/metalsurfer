"""Tests for sequential saturation feature."""

import logging
import os
import tempfile
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from ase.io import read

from metalsurfer.config import AdsorptionConfig
from metalsurfer.io_results import (
    save_multi_mol_saturation_results,
    save_saturation_results,
    setup_directories,
)
from metalsurfer.models import (
    BOStepMemory,
    MultiMolSaturationRunResult,
    MultiMolSaturationStepResult,
    SaturationRunResult,
    SaturationStepResult,
)
from metalsurfer.placement import (
    distribute_placement_budget,
    get_hollow_sites_for_adatoms,
)
from metalsurfer.placement.generators import estimate_molecule_complexity
from metalsurfer.surface_prep import (
    SlabContainer,
    auto_resize_slab_for_molecule,
    create_slab_from_bulk,
)
from metalsurfer.symmetry import SymmetryAnalysisError
from metalsurfer.workflow import load_molecules, run_saturation_screening
from metalsurfer.workflow.saturation import (
    _filter_saturation_topology_results,
    _saturation_adsorbate_topology_ok,
)

from .conftest import (
    DummyReferenceEnergies,
    NoopDatasetLogger,
    assert_paths_exist,
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)
from .optional_deps import cuda_available, has_mlip_stack


def _mock_saturation_config(**kwargs) -> AdsorptionConfig:
    """AdsorptionConfig for mocked saturation loops without real adsorbate geometries."""
    return AdsorptionConfig(
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
    return [
        make_screening_result(
            molecule=molecule,
            placement_id=placement_id,
            energy_adsorption=e_ads,
            atoms=current_slab.atoms.copy(),
            slab_size=len(current_slab.atoms),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=placement_id),
        )
    ]


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
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, [molecule], [smiles], ref, 0.0),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: slab_energy,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", NoopDatasetLogger
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
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, molecules, smiles_list, ref, 0.0),
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
        bo_transfer_used=True,
        bo_transfer_disabled_reason=None,
        bo_transfer_weight_share=0.2,
        bo_transfer_bad_rounds=0,
        bo_transfer_last_mae_delta=-0.01,
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
    assert_paths_exist(
        output_dir,
        [
            "saturation_summary.csv",
            "saturation_details.csv",
            "saturation_placements_detailed.csv",
            "xyz_structures/water_saturation/step_001_Eads_-1.0000.xyz",
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

    # extxyz writes preserve full lattice/PBC so slabs can be auto-resized later.
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
        bo_transfer_used=False,
        bo_transfer_disabled_reason=None,
        bo_transfer_weight_share=0.0,
        bo_transfer_bad_rounds=0,
        bo_transfer_last_mae_delta=None,
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
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    min_sep = 8.0
    required = 0.74 + min_sep
    assert cell_diag < required, (
        f"Test setup: slab cell {cell_diag:.1f} A must be < {required:.1f} A "
        "to trigger resize"
    )
    resized, was_resized = auto_resize_slab_for_molecule(
        slab, [h2], min_separation=min_sep
    )
    assert was_resized, (
        "Slab must be resized for this test; check auto_resize_slab_for_molecule logic"
    )
    sites_original = get_hollow_sites_for_adatoms(slab.atoms)
    sites_resized = get_hollow_sites_for_adatoms(resized.atoms)
    assert len(sites_resized) > len(sites_original), (
        "Resized slab should have more hollow sites than original"
    )


# ---------------------------------------------------------------------------
# run_saturation_screening (real GPU integration test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mlip
@pytest.mark.gpu
@pytest.mark.no_fork  # CUDA incompatible with pytest-forked
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
@pytest.mark.skipif(
    not cuda_available,
    reason="CUDA GPU required; skipped in CI (no GPU)",
)
def test_run_saturation_screening_h2_ni111_real_gpu():
    """Saturation screening with real MLIP on GPU: H2 on Ni(111).

    Runs actual run_saturation_screening (no mocks). Verifies:
    - Saturation loop terminates (E_ads >= 0 or no valid placements)
    - Steps and n_molecules_at_saturation are consistent
    - All E_ads in steps are physically reasonable
    """
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir="results_test_saturation",
    )

    # Keep this as a smoke-level integration test for runtime and CI stability.
    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=1,
        num_placements=6,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=16,
        stage2_steps=80,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n")
        smiles_path = f.name

    try:
        results = run_saturation_screening(
            slab,
            smiles_file=smiles_path,
            config=config,
            surface_type="test_saturation",
            skip_existing=False,
        )
    finally:
        os.unlink(smiles_path)

    assert len(results) == 1
    sr = results[0]
    assert sr.molecule == "H2"
    assert len(sr.steps) >= 1

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
        ref=DummyReferenceEnergies(constant_energy=-1.0),
        process_molecule=_fake_process_molecule,
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            smiles_file="unused.csv",
            config=config,
            surface_type="symmetry_fallback",
            skip_existing=False,
        )

    assert len(out) == 1
    assert len(out[0].steps) == 2
    assert symmetry_flags == [False, True]
    assert any("assuming C1" in rec.getMessage() for rec in caplog.records)


def test_run_saturation_screening_auto_resize_only_step1(monkeypatch):
    """Saturation passes allow_auto_resize=True only on first step."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config()

    allow_auto_resize_flags: list[bool] = []

    def _fake_process_molecule(
        _smi,
        mol,
        current_slab,
        *_args,
        **kwargs,
    ):
        allow_auto_resize_flags.append(bool(kwargs.get("allow_auto_resize", True)))
        step_idx = len(allow_auto_resize_flags)
        e_ads = -0.3 if step_idx == 1 else 0.2
        return _result_for_step(mol, current_slab, e_ads)

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=-1.0),
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="resize_step1_only",
        skip_existing=False,
    )

    assert len(out) == 1
    assert len(out[0].steps) == 2
    assert allow_auto_resize_flags == [True, False]


def test_apply_substrate_resize_from_step_metadata():
    from metalsurfer.workflow.saturation import (
        _apply_substrate_resize_from_step_metadata,
    )

    base = make_slab()
    resized = base.repeat((2, 2, 1))
    updated = _apply_substrate_resize_from_step_metadata(
        base,
        {
            "slab_was_resized": True,
            "substrate_atoms_after_resize": resized,
        },
    )
    assert len(updated) == len(resized)
    assert _apply_substrate_resize_from_step_metadata(base, {}) is base


def test_multi_mol_step1_presize_expands_substrate_for_all_molecules(monkeypatch):
    """Every competitor on step 1 sees the same pre-resized substrate footprint."""
    slab = SlabContainer(make_slab())
    original_n = len(slab.atoms)
    config = _mock_saturation_config(
        multi_molecule_saturation=True,
        auto_resize_slab=True,
    )
    ref = DummyReferenceEnergies({"A": -5.0, "B": -5.0})

    def mock_resize(slab_in, _conformers, _min_sep, **_kwargs):
        return SlabContainer(slab_in.atoms.repeat((2, 2, 1))), True

    slab_sizes_seen: list[tuple[str, int]] = []

    def fake_process(_smi, mol, current_slab, *_args, **_kwargs):
        slab_sizes_seen.append((mol, len(current_slab.atoms)))
        return _result_for_step(mol, current_slab, 0.5)

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.auto_resize_slab_for_molecule",
        mock_resize,
    )
    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["B", "A"],
        smiles_list=["smiles_b", "smiles_a"],
        ref=ref,
        process_molecule=fake_process,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="multi_mol_presize",
        skip_existing=False,
    )

    assert len(out) == 1
    expected_n = original_n * 4
    assert len(slab_sizes_seen) == 2
    assert all(size == expected_n for _, size in slab_sizes_seen)


def test_single_mol_saturation_bo_updates_base_slab_after_resize(monkeypatch):
    """Single-molecule BO saturation expands base_slab after step-1 auto-resize."""
    slab = SlabContainer(make_slab())
    original_n = len(slab.atoms)
    config = _mock_saturation_config(bo_enabled=True, auto_resize_slab=True)
    ref = DummyReferenceEnergies({"water": -5.0})

    base_slab_lens: list[int] = []
    call_count = 0

    def fake_bayesian(_smi, _mol, current_slab, *_args, **kwargs):
        nonlocal call_count
        call_count += 1
        base = kwargs.get("base_slab_for_frozen")
        if base is not None:
            base_slab_lens.append(len(base))
        step_metadata = kwargs.get("step_metadata_out")
        if step_metadata is not None:
            resized = current_slab.atoms.repeat((2, 2, 1))
            step_metadata["slab_was_resized"] = True
            step_metadata["substrate_atoms_after_resize"] = resized
        e_ads = -0.4 if call_count == 1 else 0.5
        return _result_for_step("water", current_slab, e_ads)

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, ["water"], ["O"], ref, 0.0),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -10.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", NoopDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule_bayesian",
        fake_bayesian,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="bo_resize_base_slab",
        skip_existing=False,
    )

    assert len(out) == 1
    assert len(out[0].steps) == 2
    assert base_slab_lens[0] == original_n
    assert base_slab_lens[1] == original_n * 4


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

    mol = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

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

    ref = DummyReferenceEnergies({"water": -5.0, "CO2": -10.0})

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
        smiles_file="unused.csv",
        config=config,
        surface_type="multi_mol_test",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert isinstance(result, MultiMolSaturationRunResult)
    assert len(result.steps) >= 1
    assert result.steps[0].winning_molecule == "CO2"


def test_multi_mol_saturation_terminates_on_positive_eads(monkeypatch):
    """Multi-mol saturation stops when best E_ads >= 0."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies({"A": -5.0, "B": -5.0})

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
        smiles_file="unused.csv",
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
        return [
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

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["mol1", "mol2"],
        smiles_list=["s1", "s2"],
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
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
            smiles_file="unused.csv",
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
    config = _mock_saturation_config(multi_molecule_saturation=True)

    ref = DummyReferenceEnergies({"A": -5.0, "B": -5.0})

    step_count = [0]

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        step_count[0] += 1
        # Step 1: A=-0.8, B=-0.5 → A wins; Step 2: A=-0.3, B=-0.7 → B wins;
        # Step 3: both positive → saturated
        energies = {
            1: {"A": -0.8, "B": -0.5},
            2: {"A": -0.3, "B": -0.7},
            3: {"A": 0.1, "B": 0.2},
        }
        # Determine current saturation step from slab size (rough proxy)
        # Use step_count: each step calls both mols, so step = ceil(count/2)
        sat_step = (step_count[0] + 1) // 2
        sat_step = min(sat_step, 3)
        e_ads = energies[sat_step][mol]
        return [
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

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["A", "B"],
        smiles_list=["sa", "sb"],
        ref=ref,
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="molecule_counts_test",
        skip_existing=False,
    )

    assert len(out) == 1
    result = out[0]
    assert result.molecule_counts["A"] + result.molecule_counts["B"] == len(
        result.steps
    )


def test_multi_mol_saturation_bo_uses_independent_memory_per_adsorbate(monkeypatch):
    """Competing BO saturation carries BO state forward independently per molecule."""
    slab = SlabContainer(make_slab())
    config = _mock_saturation_config(multi_molecule_saturation=True, bo_enabled=True)

    ref = DummyReferenceEnergies({"water": -5.0, "CO2": -10.0})

    call_counts: dict[str, int] = {"water": 0, "CO2": 0}
    prior_memory_seen: dict[str, list[BOStepMemory | None]] = {"water": [], "CO2": []}

    def _fake_process_molecule_bayesian(
        _smi,
        mol,
        current_slab,
        *_args,
        bo_step_memory_in=None,
        bo_step_memory_out=None,
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

        if bo_step_memory_out is not None:
            bo_step_memory_out["memory"] = BOStepMemory(
                observed_X_rows=[{"step": float(step_idx), "mol_len": float(len(mol))}],
                observed_y=[float(step_idx)],
                best_energy=-0.1 * step_idx,
            )

        energies = {
            "water": [-0.5, 0.4],
            "CO2": [-1.2, 0.3],
        }
        e_ads = energies[mol][step_idx - 1]
        return [
            make_screening_result(
                molecule=mol,
                placement_id=step_idx,
                energy_adsorption=e_ads,
                atoms=current_slab.atoms.copy(),
                slab_size=len(current_slab.atoms),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=step_idx),
            )
        ]

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["water", "CO2"],
        smiles_list=["O", "O=C=O"],
        ref=ref,
        process_molecule_bayesian=_fake_process_molecule_bayesian,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="multi_mol_bo_memory",
        skip_existing=False,
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
    config = _mock_saturation_config(multi_molecule_saturation=True, bo_enabled=True)

    ref = DummyReferenceEnergies({"A": -5.0, "B": -5.0})

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
        bo_step_memory_out=None,
        **_kwargs,
    ):
        if bo_step_memory_out is not None:
            bo_step_memory_out["memory"] = shared_memory
        return [
            make_screening_result(
                molecule=mol,
                placement_id=0,
                energy_adsorption=-0.2,
                atoms=current_slab.atoms.copy(),
                slab_size=len(current_slab.atoms),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            )
        ]

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
            smiles_file="unused.csv",
            config=config,
            surface_type="multi_mol_bo_shared_memory",
            skip_existing=False,
        )


@pytest.mark.slow
@pytest.mark.mlip
@pytest.mark.gpu
@pytest.mark.no_fork
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
@pytest.mark.skipif(
    not cuda_available,
    reason="CUDA GPU required; skipped in CI (no GPU)",
)
def test_run_saturation_screening_multi_mol_bo_real_gpu():
    """Smoke-level GPU integration test for BO-enabled competing saturation."""
    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        results_dir="results_test_multi_mol_bo_gpu",
    )

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=1,
        num_placements=2,
        device="cuda",
        material_type="slab",
        multi_molecule_saturation=True,
        bo_enabled=True,
        bo_initial_random=1,
        bo_batch_size=1,
        bo_total_budget=2,
        saturation_max_steps=1,
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=8,
        stage2_steps=32,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n[C-]#[O+],CO\n")
        smiles_path = f.name

    try:
        results = run_saturation_screening(
            slab,
            smiles_file=smiles_path,
            config=config,
            surface_type="test_multi_mol_bo_gpu",
            skip_existing=False,
        )
    finally:
        os.unlink(smiles_path)

    assert len(results) == 1
    result = results[0]

    assert isinstance(result, MultiMolSaturationRunResult)
    assert len(result.steps) >= 1
    step0 = result.steps[0]
    # Keys match molecules that got conformers and entered the competitive loop.
    assert set(step0.per_molecule_results) == set(step0.bo_transfer_used)
    assert len(step0.per_molecule_results["H2"]) >= 1


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
        bo_transfer_enabled=False,
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
    assert_paths_exist(
        output_dir,
        [
            "saturation_summary.csv",
            "saturation_details.csv",
            "saturation_placements_detailed.csv",
            "xyz_structures/water_CO2_saturation",
            "xyz_structures/water_CO2_saturation/step_001_placements/water/conformer_000.xyz",
            "xyz_structures/water_CO2_saturation/step_001_placements/water/conformer_001.xyz",
            "xyz_structures/water_CO2_saturation/step_001_placements/CO2/conformer_000.xyz",
        ],
    )

    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    assert "water_CO2" in str(summary_df.iloc[0]["molecules"])
    assert int(summary_df.iloc[0]["n_molecules_at_saturation"]) == 1

    details_df = pd.read_csv(details_path)
    assert len(details_df) == 1
    assert details_df.iloc[0]["winning_molecule"] == "water"
    assert "water" in str(details_df.iloc[0]["per_molecule_budgets"])

    placements_df = pd.read_csv(output_dir / "saturation_placements_detailed.csv")
    assert len(placements_df) == 3
    assert set(placements_df["molecule"]) == {"water", "CO2"}


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
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
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
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    ok, reason = _saturation_adsorbate_topology_ok(
        mismatched_units,
        base_slab_len=len(slab),
        reference_unit_smiles=["O", "O"],
        config=config,
    )
    assert ok, reason
    assert reason == "adsorbate connectivity intact"


def test_saturation_topology_guard_rejects_coupled_waters():
    slab, _first, combined = _slab_with_two_waters(separated=False)
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
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
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
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
        connectivity_multipliers=[1.3],
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
    config = AdsorptionConfig(connectivity_multipliers=[1.3])

    call_count = [0]

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [
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
        return [
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

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=-1.0),
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
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
        connectivity_multipliers=[1.3],
    )

    step_idx = [0]

    bare, slab_plus_one, _ = _slab_with_two_waters(separated=True)

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        step_idx[0] += 1
        if step_idx[0] <= 2:
            if step_idx[0] == 1 and mol == "water":
                return [
                    make_screening_result(
                        molecule=mol,
                        placement_id=0,
                        energy_adsorption=-0.4,
                        atoms=slab_plus_one.copy(),
                        slab_size=len(bare),
                        distance=2.5,
                        placement_descriptor=make_placement_descriptor(placement_id=0),
                    )
                ]
            return _result_for_step(mol, current_slab, -0.4)
        if mol == "water":
            return [
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
        return _result_for_step(mol, current_slab, -0.2)

    _patch_multi_mol_saturation_mocks(
        monkeypatch,
        molecules=["water", "CO2"],
        smiles_list=["O", "O=C=O"],
        ref=DummyReferenceEnergies({"water": -5.0, "CO2": -10.0}),
        process_molecule=_fake_process_molecule,
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
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


def test_saturation_topology_guard_all_filtered_stops(monkeypatch, caplog):
    """If every candidate rearranges, saturation stops without advancing."""
    slab = SlabContainer(make_slab())
    bare, slab_plus_one, _ = _slab_with_two_waters(separated=True)
    _, _, coupled = _slab_with_two_waters(separated=False)
    config = AdsorptionConfig(connectivity_multipliers=[1.3])

    call_count = [0]

    def _fake_process_molecule(_smi, mol, current_slab, *_args, **_kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return [
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
        return [
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

    _patch_single_mol_saturation_mocks(
        monkeypatch,
        molecule="water",
        smiles="O",
        ref=DummyReferenceEnergies(constant_energy=-1.0),
        process_molecule=_fake_process_molecule,
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            smiles_file="unused.csv",
            config=config,
            surface_type="topology_guard_stop",
            skip_existing=False,
        )

    assert len(out) == 1
    assert len(out[0].steps) == 1
    assert any(
        "topology rearrangement guard" in rec.getMessage() for rec in caplog.records
    )

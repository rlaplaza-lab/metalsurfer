"""Tests for sequential saturation feature."""

import logging
import os
import tempfile

import pandas as pd
import pytest
from ase.io import read

from metalsurfer.config import AdsorptionConfig
from metalsurfer.io_results import save_saturation_results, setup_directories
from metalsurfer.models import (
    BOStepMemory,
    SaturationRunResult,
    SaturationStepResult,
)
from metalsurfer.placement import get_hollow_sites_for_adatoms
from metalsurfer.surface_prep import (
    SlabContainer,
    auto_resize_slab_for_molecule,
    create_slab_from_bulk,
)
from metalsurfer.symmetry import SymmetryAnalysisError
from metalsurfer.workflow import load_molecules, run_saturation_screening

from .conftest import (
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)
from .optional_deps import cuda_available, has_mlip_stack

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
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=best,
        all_results=[best],
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
    summary_path = workdir / "results_saturation_test" / "saturation_summary.csv"
    details_path = workdir / "results_saturation_test" / "saturation_details.csv"
    xyz_path = (
        workdir
        / "results_saturation_test/xyz_structures/water_saturation/step_001_Eads_-1.0000.xyz"
    )
    stable_xyz_path = (
        workdir
        / "results_saturation_test/xyz_structures/water_saturation/step_001_best_slab.xyz"
    )
    final_xyz_path = (
        workdir
        / "results_saturation_test/xyz_structures/water_saturation/final_saturated_slab.xyz"
    )
    poscar_path = (
        workdir / "results_saturation_test/vasp_inputs/water_saturation/step_001/POSCAR"
    )
    assert summary_path.exists()
    assert details_path.exists()
    assert xyz_path.exists()
    assert stable_xyz_path.exists()
    assert final_xyz_path.exists()
    assert poscar_path.exists()
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
    from ase import Atoms

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
    config = AdsorptionConfig()

    class _DummyRef:
        molecule_energies: dict[str, float] = {}

        def get_molecule_energy(self, _mol: str) -> float:
            return -1.0

    class _DummyDatasetLogger:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_results(self, *_args, **_kwargs):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, ["water"], ["O"], _DummyRef(), 0.0),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy", lambda *_a, **_kw: -10.0
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
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
    config = AdsorptionConfig()

    class _DummyRef:
        molecule_energies: dict[str, float] = {}

        def get_molecule_energy(self, _mol: str) -> float:
            return -1.0

    class _DummyDatasetLogger:
        def __init__(self, *_args, **_kwargs):
            pass

        def add_results(self, *_args, **_kwargs):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, ["water"], ["O"], _DummyRef(), 0.0),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy", lambda *_a, **_kw: -10.0
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )

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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
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


# ---------------------------------------------------------------------------
# Multi-molecule saturation: budget distribution
# ---------------------------------------------------------------------------


def test_distribute_placement_budget_proportional():
    """Budget is split proportionally to complexity scores."""
    from metalsurfer.placement import distribute_placement_budget

    budgets = distribute_placement_budget({"A": 100.0, "B": 400.0}, 250)
    assert budgets["A"] + budgets["B"] == 250
    # B has 4× complexity → should receive roughly 4× the placements
    assert budgets["B"] > budgets["A"]


def test_distribute_placement_budget_sums_to_total():
    """Allocations always sum to exactly total_budget regardless of rounding."""
    from metalsurfer.placement import distribute_placement_budget

    for total in (10, 11, 13, 100, 250):
        budgets = distribute_placement_budget({"X": 1.0, "Y": 3.0, "Z": 2.0}, total)
        assert sum(budgets.values()) == total, (
            f"total={total}: sum={sum(budgets.values())}"
        )


def test_distribute_placement_budget_min_one():
    """Every molecule gets at least 1 placement even with tiny complexity."""
    from metalsurfer.placement import distribute_placement_budget

    budgets = distribute_placement_budget(
        {"tiny": 1.0, "huge": 10000.0}, total_budget=5
    )
    assert budgets["tiny"] >= 1
    assert budgets["huge"] >= 1
    assert sum(budgets.values()) == 5


def test_distribute_placement_budget_single_molecule():
    """Single molecule gets the entire budget."""
    from metalsurfer.placement import distribute_placement_budget

    budgets = distribute_placement_budget({"H2": 50.0}, total_budget=100)
    assert budgets["H2"] == 100


def test_distribute_placement_budget_equal_complexity():
    """Equal complexities → roughly equal split (may differ by 1 due to rounding)."""
    from metalsurfer.placement import distribute_placement_budget

    budgets = distribute_placement_budget({"A": 100.0, "B": 100.0}, total_budget=10)
    assert sum(budgets.values()) == 10
    assert abs(budgets["A"] - budgets["B"]) <= 1


# ---------------------------------------------------------------------------
# Multi-molecule saturation: estimate_molecule_complexity
# ---------------------------------------------------------------------------


def test_estimate_molecule_complexity_positive():
    """Complexity score must be >= 1.0 for any valid molecule."""
    from ase import Atoms

    from metalsurfer.placement.generators import estimate_molecule_complexity

    slab = make_slab()
    # minimal linear molecule (CO-like)
    linear = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    config = AdsorptionConfig(num_conformers=1, num_placements=50)
    score = estimate_molecule_complexity([linear], slab, config, smiles="[C-]#[O+]")
    assert score >= 1.0


def test_estimate_molecule_complexity_more_conformers_higher_score():
    """More conformers always give a higher complexity score (n_conformers is a direct multiplier)."""
    from ase import Atoms

    from metalsurfer.placement.generators import estimate_molecule_complexity

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
    config = AdsorptionConfig(multi_molecule_saturation=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"water": -5.0, "CO2": -10.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

    # Step 1: water E_ads=-0.5, CO2 E_ads=-1.2 → CO2 wins
    # Step 2: water E_ads=+0.1 → saturated
    call_counts: dict[str, int] = {"water": 0, "CO2": 0}

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_counts[mol] += 1
        call_idx = call_counts[mol]
        energies = {"water": [-0.5, 0.1], "CO2": [-1.2, 0.1]}
        e_ads = energies[mol][min(call_idx - 1, len(energies[mol]) - 1)]
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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["water", "CO2"],
            ["O", "O=C=O"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda smi, *_a, **_kw: (
            [make_water()],
            [0.0],
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
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
    from metalsurfer.models import MultiMolSaturationRunResult

    assert isinstance(result, MultiMolSaturationRunResult)
    assert len(result.steps) >= 1
    assert result.steps[0].winning_molecule == "CO2"


def test_multi_mol_saturation_terminates_on_positive_eads(monkeypatch):
    """Multi-mol saturation stops when best E_ads >= 0."""
    slab = SlabContainer(make_slab())
    config = AdsorptionConfig(multi_molecule_saturation=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"A": -5.0, "B": -5.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

    call_count = [0]

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_count[0] += 1
        # Both molecules return positive E_ads immediately → saturated after step 1
        e_ads = 0.5
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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["A", "B"],
            ["smiles_a", "smiles_b"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda smi, *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
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
    config = AdsorptionConfig(multi_molecule_saturation=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"mol1": -5.0, "mol2": -5.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["mol1", "mol2"],
            ["s1", "s2"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda smi, *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
    )

    out = run_saturation_screening(
        slab,
        smiles_file="unused.csv",
        config=config,
        surface_type="step_result_structure",
        skip_existing=False,
    )

    from metalsurfer.models import (
        MultiMolSaturationRunResult,
        MultiMolSaturationStepResult,
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
    config = AdsorptionConfig(multi_molecule_saturation=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"water": -5.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

    call_count = [0]

    def _fake_process_molecule(smi, mol, current_slab, *_args, **kwargs):
        call_count[0] += 1
        e_ads = -0.5 if call_count[0] == 1 else 0.1
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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (object(), None, ["water"], ["O"], _DummyRef(), 0.0),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy", lambda *_a, **_kw: -10.0
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
    )

    with caplog.at_level(logging.WARNING):
        out = run_saturation_screening(
            slab,
            smiles_file="unused.csv",
            config=config,
            surface_type="single_mol_fallback",
            skip_existing=False,
        )

    # Should fall back to standard single-molecule SaturationRunResult
    assert len(out) == 1
    from metalsurfer.models import SaturationRunResult

    assert isinstance(out[0], SaturationRunResult)
    assert any("falling back" in rec.getMessage().lower() for rec in caplog.records)


def test_multi_mol_saturation_molecule_counts_tracked(monkeypatch):
    """molecule_counts tracks how many steps each molecule won."""
    slab = SlabContainer(make_slab())
    config = AdsorptionConfig(multi_molecule_saturation=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"A": -5.0, "B": -5.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["A", "B"],
            ["sa", "sb"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda smi, *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule", _fake_process_molecule
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
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
    config = AdsorptionConfig(multi_molecule_saturation=True, bo_enabled=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"water": -5.0, "CO2": -10.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["water", "CO2"],
            ["O", "O=C=O"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda _smi, *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule_bayesian",
        _fake_process_molecule_bayesian,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
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
    config = AdsorptionConfig(multi_molecule_saturation=True, bo_enabled=True)

    class _DummyRef:
        molecule_energies: dict[str, float] = {"A": -5.0, "B": -5.0}

        def get_molecule_energy(self, mol: str) -> float:
            return self.molecule_energies[mol]

    class _DummyDatasetLogger:
        def __init__(self, *_a, **_kw):
            pass

        def add_results(self, *_a, **_kw):
            pass

        def flush(self):
            pass

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

    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._setup_screening_run",
        lambda *_a, **_kw: (
            object(),
            None,
            ["A", "B"],
            ["sa", "sb"],
            _DummyRef(),
            0.0,
        ),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation._compute_slab_energy",
        lambda *_a, **_kw: -100.0,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.DatasetLogger", _DummyDatasetLogger
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.create_conformers_from_smiles",
        lambda _smi, *_a, **_kw: ([make_water()], [0.0]),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.process_molecule_bayesian",
        _fake_process_molecule_bayesian,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.saturation.distribute_placement_budget",
        lambda complexities, total: {
            mol: total // len(complexities) for mol in complexities
        },
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
    """Heavy GPU integration test for BO-enabled competing saturation."""
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
        num_placements=4,
        device="cuda",
        multi_molecule_saturation=True,
        bo_enabled=True,
        bo_initial_random=2,
        bo_batch_size=1,
        bo_total_budget=4,
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=16,
        stage2_steps=80,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\nC#O,CO\n")
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

    from metalsurfer.models import MultiMolSaturationRunResult

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
    from metalsurfer.io_results import save_multi_mol_saturation_results
    from metalsurfer.models import (
        MultiMolSaturationRunResult,
        MultiMolSaturationStepResult,
    )

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

    summary_path = workdir / "results_multi_mol_io_test" / "saturation_summary.csv"
    details_path = workdir / "results_multi_mol_io_test" / "saturation_details.csv"
    xyz_dir = (
        workdir
        / "results_multi_mol_io_test"
        / "xyz_structures"
        / "water_CO2_saturation"
    )

    assert summary_path.exists()
    assert details_path.exists()
    assert xyz_dir.exists()

    summary_df = pd.read_csv(summary_path)
    assert len(summary_df) == 1
    assert "water_CO2" in str(summary_df.iloc[0]["molecules"])
    assert int(summary_df.iloc[0]["n_molecules_at_saturation"]) == 1

    details_df = pd.read_csv(details_path)
    assert len(details_df) == 1
    assert details_df.iloc[0]["winning_molecule"] == "water"
    assert "water" in str(details_df.iloc[0]["per_molecule_budgets"])

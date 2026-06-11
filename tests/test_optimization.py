"""Tests for optimization module: TorchSimCalculator, setup_single_model."""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

import metalsurfer.optimization as optimization_mod
from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import (
    TorchSimCalculator,
    optimize_isolated_molecules_batched,
    setup_single_model,
)
from tests.optional_deps import has_mlip_stack

from .conftest import make_slab


def _make_atoms_with_cell() -> Atoms:
    """Atoms with PBC for TorchSim (needs cell)."""
    slab = make_slab(nx=2, ny=2, n_layers=2)
    slab.set_pbc([True, True, True])
    return slab


@pytest.fixture
def stub_autobatcher(monkeypatch: pytest.MonkeyPatch):
    optimization_mod._AUTOBATCHER_CACHE.clear()
    monkeypatch.setattr(optimization_mod, "ts", object())
    monkeypatch.setattr(
        optimization_mod,
        "InFlightAutoBatcher",
        lambda *args, **kwargs: object(),
    )
    yield
    optimization_mod._AUTOBATCHER_CACHE.clear()


@pytest.mark.mlip
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
class TestTorchSimCalculator:
    """Unit tests with mocked model."""

    def test_calculator_interface_with_mock_model(self):
        """TorchSimCalculator returns energy/forces via ts.static() with mock."""
        import torch

        mock_model = type("MockModel", (), {})()
        mock_model.device = torch.device("cpu")
        mock_model.dtype = torch.float32

        energy_val = -42.5
        n_atoms = 6

        calc = TorchSimCalculator(mock_model)
        atoms = _make_atoms_with_cell()
        atoms = atoms[:n_atoms]
        atoms.set_cell([10.0, 10.0, 15.0])
        atoms.set_pbc([True, True, True])

        fake_result = [
            {
                "potential_energy": torch.tensor([[energy_val]], dtype=torch.float32),
                "forces": torch.randn(n_atoms, 3, dtype=torch.float32) * 0.1,
            }
        ]
        with patch("torch_sim.static", return_value=fake_result):
            calc.calculate(atoms, ["energy", "forces"])
            assert "energy" in calc.results
            assert abs(calc.results["energy"] - energy_val) < 1e-6
            assert "forces" in calc.results
            assert calc.results["forces"].shape == (n_atoms, 3)

            e = calc.get_potential_energy(atoms)
            assert abs(e - energy_val) < 1e-6
            f = calc.get_forces(atoms)
            assert f.shape == (n_atoms, 3)


@pytest.mark.mlip
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
class TestSetupSingleModel:
    """Integration tests with real FairChemModel."""

    def test_setup_single_model_returns_calculator_and_model(self):
        """setup_single_model returns (calculator, ts_model) tuple."""
        calculator, ts_model = setup_single_model("uma-s-1p1", "cpu")
        assert calculator is not None
        assert ts_model is not None
        assert isinstance(calculator, TorchSimCalculator)

    def test_torchsim_calculator_single_point_with_real_model(self):
        """TorchSimCalculator from setup_single_model gives finite energy/forces."""
        calculator, _ = setup_single_model("uma-s-1p1", "cpu")
        atoms = _make_atoms_with_cell()
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert np.all(np.isfinite(forces))

    def test_optimize_isolated_sequentially_skips_autobatcher(self):
        """With optimize_isolated_sequentially=True, _get_inflight_autobatcher is not called."""
        calculator, ts_model = setup_single_model("uma-s-1p1", "cpu")
        config = AdsorptionConfig(
            optimize_isolated_sequentially=True,
            device="cpu",
        )
        conformers = [_make_atoms_with_cell()[:6]]  # small system
        for c in conformers:
            c.set_cell([10.0, 10.0, 15.0])
            c.set_pbc([True, True, True])

        with patch("metalsurfer.optimization._get_inflight_autobatcher") as mock_get_ab:
            results = optimize_isolated_molecules_batched(
                conformers,
                ts_model,
                fmax=0.1,
                steps=5,
                config=config,
            )
            mock_get_ab.assert_not_called()
        assert len(results) == 1
        atoms, energy = results[0]
        assert np.isfinite(energy)


def test_get_inflight_autobatcher_saturation_reuses_small_growth(stub_autobatcher):
    """Saturation mode reuses prior key for small max_n_atoms increase."""
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(
        device="cpu",
        saturation_autobatcher_reuse_growth_atoms=8,
        saturation_autobatcher_reuse_growth_fraction=0.0,
    )
    ab1, key1, reused1 = optimization_mod._get_inflight_autobatcher(
        model,
        100,
        config=config,
        saturation_reuse=True,
    )
    ab2, key2, reused2 = optimization_mod._get_inflight_autobatcher(
        model,
        105,
        config=config,
        saturation_reuse=True,
    )
    assert ab1 is not None
    assert ab2 is ab1
    assert key2 == key1
    assert reused1 is False
    assert reused2 is True


def test_get_inflight_autobatcher_non_saturation_uses_exact_size_key(stub_autobatcher):
    """Non-saturation mode should not reuse different max_n_atoms keys."""
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(device="cpu")
    ab1, key1, reused1 = optimization_mod._get_inflight_autobatcher(
        model,
        100,
        config=config,
        saturation_reuse=False,
    )
    ab2, key2, reused2 = optimization_mod._get_inflight_autobatcher(
        model,
        105,
        config=config,
        saturation_reuse=False,
    )
    assert ab1 is not None
    assert ab2 is not None
    assert ab2 is not ab1
    assert key2 != key1
    assert reused1 is False
    assert reused2 is False


def test_get_inflight_autobatcher_uses_explicit_probe_cap(stub_autobatcher):
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(device="cpu", autobatcher_max_atoms_to_try=123_456)
    _, key, _ = optimization_mod._get_inflight_autobatcher(
        model,
        100,
        config=config,
        max_atoms_to_try=7_000,
    )
    assert key is not None
    assert int(key[6]) == 7_000


def test_resolve_autobatcher_max_atoms_to_try_uses_config_override():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=12_345)
    cap, source = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=40,
        config=config,
    )
    assert cap == 12_345
    assert source == "config_override"


def test_resolve_autobatcher_max_atoms_to_try_uses_dynamic_policy():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    cap, source = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=10,
        config=config,
    )
    assert cap == 10_000
    assert source == "dynamic"


def test_resolve_autobatcher_max_atoms_to_try_applies_floor_and_ceiling():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    floor_cap, floor_source = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=20,
        n_systems=2,
        config=config,
    )
    ceiling_cap, ceiling_source = (
        optimization_mod._resolve_autobatcher_max_atoms_to_try(
            max_n_atoms=2_000,
            n_systems=100,
            config=config,
        )
    )
    assert floor_cap == 5_000
    assert ceiling_cap == 200_000
    assert floor_source == "dynamic"
    assert ceiling_source == "dynamic"


def test_resolve_autobatcher_max_atoms_to_try_buckets_nearby_workloads():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    cap_a, source_a = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=10,
        config=config,
    )
    cap_b, source_b = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=11,
        config=config,
    )
    assert cap_a == 10_000
    assert cap_b == 10_000
    assert source_a == "dynamic"
    assert source_b == "dynamic"


def test_estimate_parallel_relaxation_capacity_fallback_without_torchsim(
    monkeypatch: pytest.MonkeyPatch,
):
    optimization_mod._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(optimization_mod, "ts", None)
    monkeypatch.setattr(optimization_mod, "determine_max_batch_size", None)
    config = AdsorptionConfig()
    atoms = _make_atoms_with_cell()
    capacity = optimization_mod.estimate_parallel_relaxation_capacity(
        ts_model=object(),
        representative_atoms=atoms,
        config=config,
        frozen_indices=[],
    )
    assert capacity == 1


def test_estimate_parallel_relaxation_capacity_uses_memory_scaler(
    monkeypatch: pytest.MonkeyPatch,
):
    optimization_mod._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(optimization_mod, "ts", object())
    monkeypatch.setattr(optimization_mod, "ts_constraints", object())
    monkeypatch.setattr(
        optimization_mod,
        "_make_state_with_frozen_constraint",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        optimization_mod,
        "calculate_memory_scalers",
        lambda state, memory_scales_with: [100.0],
    )
    config = AdsorptionConfig(
        autobatcher_max_memory_scaler=1200.0,
        autobatcher_max_memory_padding=0.5,
    )
    atoms = _make_atoms_with_cell()
    capacity = optimization_mod.estimate_parallel_relaxation_capacity(
        ts_model=object(),
        representative_atoms=atoms,
        config=config,
        frozen_indices=[],
    )
    assert capacity == 6


def test_resolve_autobatcher_max_atoms_to_try_is_conservative_vs_estimate():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    max_n_atoms = 333
    n_systems = 9
    cap, source = optimization_mod._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=max_n_atoms,
        n_systems=n_systems,
        config=config,
    )
    estimated = (
        optimization_mod._DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER * max_n_atoms * n_systems
    )
    assert cap >= estimated
    assert source == "dynamic"

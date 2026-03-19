"""Tests for optimization module: TorchSimCalculator, setup_single_model."""

from unittest.mock import patch

import numpy as np
import pytest
from ase import Atoms

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
        assert results[0] is not None
        atoms, energy = results[0]
        assert np.isfinite(energy)


@pytest.mark.mlip
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
def test_optimize_isolated_sequentially_skips_autobatcher_unit():
    """Unit test: with optimize_isolated_sequentially=True, _get_inflight_autobatcher is not called.

    Uses mocks so it runs without GPU; requires torch/torch_sim for the optimizer.
    """
    from unittest.mock import MagicMock

    import torch

    # Minimal mock ts_model
    mock_ts_model = MagicMock()
    mock_ts_model.device = torch.device("cpu")
    mock_ts_model.dtype = torch.float32

    conformers = [_make_atoms_with_cell()[:6]]
    for c in conformers:
        c.set_cell([10.0, 10.0, 15.0])
        c.set_pbc([True, True, True])

    config = AdsorptionConfig(optimize_isolated_sequentially=True, device="cpu")

    mock_state = MagicMock()
    mock_state.to_atoms.return_value = [c.copy() for c in conformers]
    mock_state.energy = [torch.tensor(-1.0)]

    with (
        patch("metalsurfer.optimization._get_inflight_autobatcher") as mock_get_ab,
        patch("torch_sim.optimize", return_value=mock_state),
        patch(
            "torch_sim.generate_force_convergence_fn",
            return_value=MagicMock(),
        ),
    ):
        results = optimize_isolated_molecules_batched(
            conformers,
            mock_ts_model,
            fmax=0.1,
            steps=5,
            config=config,
        )

        mock_get_ab.assert_not_called()
    assert len(results) == 1
    assert results[0] is not None
    atoms, energy = results[0]
    assert energy == -1.0

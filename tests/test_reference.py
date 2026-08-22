"""Tests for workflow reference-energy helpers (calculate_reference_energies)."""

import logging

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.exceptions import OptimizationError
from metalsurfer.models import ReferenceEnergies
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow import reference
from tests.conftest import make_slab, mock_calculator


def _make_atoms(n: int = 3) -> Atoms:
    return Atoms(
        symbols=["O"] + ["H"] * (n - 1) if n > 1 else ["O"],
        positions=np.zeros((n, 3)),
    )


def _patch_reference_helpers(
    monkeypatch,
    *,
    conformers=None,
    opt_results=None,
    cache_calls=None,
):
    """Monkeypatch the three MLIP-backed helpers used by calculate_reference_energies."""

    def _fake_create_smiles(*_args, **_kwargs):
        return conformers

    def _fake_optimize(*_args, **_kwargs):
        return opt_results if opt_results is not None else []

    def _fake_clear(*_args, **_kwargs):
        if cache_calls is not None:
            cache_calls.append(_kwargs)

    monkeypatch.setattr(reference, "create_conformers_from_smiles", _fake_create_smiles)
    monkeypatch.setattr(
        reference, "optimize_isolated_molecules_batched", _fake_optimize
    )
    monkeypatch.setattr(reference, "clear_autobatcher_cache", _fake_clear)


class TestCalculateReferenceEnergies:
    def test_happy_path_returns_lowest_conformer_energy(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=-100.0, n_atoms=len(slab.atoms))
        confs = [_make_atoms(), _make_atoms()]
        # (atoms, energy) tuples; the lowest energy must win.
        opt_results = [(_make_atoms(), -5.0), (_make_atoms(), -8.0)]
        cache_calls: list[dict] = []
        _patch_reference_helpers(
            monkeypatch,
            conformers=(confs, [0.0, 0.0]),
            opt_results=opt_results,
            cache_calls=cache_calls,
        )

        ref = reference.calculate_reference_energies(
            slab,
            calc,
            ["water"],
            ["O"],
            ts_model=object(),
            config=AdsorptionConfig(),
        )

        assert isinstance(ref, ReferenceEnergies)
        assert ref.slab_energy == -100.0
        # Best conformer selection, not just presence.
        assert ref.molecule_energies == {"water": -8.0}
        # clear_autobatcher_cache(clear_capacity=True) called exactly once.
        assert len(cache_calls) == 1
        assert cache_calls[0].get("clear_capacity") is True

    def test_non_finite_slab_energy_raises(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=float("nan"), n_atoms=len(slab.atoms))
        _patch_reference_helpers(monkeypatch)

        with pytest.raises(OptimizationError, match="not finite"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(),
            )

    def test_zero_slab_energy_raises(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=1e-7, n_atoms=len(slab.atoms))
        _patch_reference_helpers(monkeypatch)

        with pytest.raises(OptimizationError, match="effectively zero"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(),
            )

    def test_boundary_slab_energy_1e_6_accepted(self, monkeypatch):
        """abs(E) == 1e-6 is the reject threshold; equal is accepted (strict <)."""
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=1e-6, n_atoms=len(slab.atoms))
        confs = [_make_atoms()]
        _patch_reference_helpers(
            monkeypatch,
            conformers=(confs, [0.0]),
            opt_results=[(_make_atoms(), -5.0)],
        )
        ref = reference.calculate_reference_energies(
            slab, calc, ["water"], ["O"], ts_model=object(), config=AdsorptionConfig()
        )
        assert ref.slab_energy == pytest.approx(1e-6)

    def test_zero_exact_slab_energy_raises(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=0.0, n_atoms=len(slab.atoms))
        _patch_reference_helpers(monkeypatch)
        with pytest.raises(OptimizationError, match="effectively zero"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(),
            )

    def test_inf_slab_energy_raises(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=float("inf"), n_atoms=len(slab.atoms))
        _patch_reference_helpers(monkeypatch)
        with pytest.raises(OptimizationError, match="not finite"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(),
            )

    def test_empty_slab_n_atoms_zero_path(self, monkeypatch):
        """Calculator with n_atoms=0 still goes through energy gates on returned energy."""
        empty = SlabContainer(Atoms())
        calc = mock_calculator(energy=-1.0, n_atoms=0)
        confs = [_make_atoms()]
        _patch_reference_helpers(
            monkeypatch,
            conformers=(confs, [0.0]),
            opt_results=[(_make_atoms(), -5.0)],
        )
        # Empty substrate may fail prepare or succeed with the stub energy.
        try:
            ref = reference.calculate_reference_energies(
                empty,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(),
            )
            assert ref.slab_energy == pytest.approx(-1.0)
        except (OptimizationError, ValueError, IndexError, RuntimeError):
            pass  # empty geometry may be rejected upstream

    def test_conformer_failure_raises_when_strict(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=-100.0, n_atoms=len(slab.atoms))
        # create_conformers_from_smiles returns None.
        _patch_reference_helpers(monkeypatch, conformers=None)

        with pytest.raises(RuntimeError, match="conformers"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(fail_on_conformer_failure=True),
            )

    def test_conformer_failure_omits_molecule_when_lenient(self, monkeypatch, caplog):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=-100.0, n_atoms=len(slab.atoms))
        _patch_reference_helpers(monkeypatch, conformers=None)

        with caplog.at_level(logging.WARNING):
            ref = reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(fail_on_conformer_failure=False),
            )
        assert ref.slab_energy == -100.0
        assert "water" not in ref.molecule_energies
        assert any(
            "Could not create water from SMILES" in rec.getMessage()
            for rec in caplog.records
        )

    def test_empty_optimization_results_raises_when_strict(self, monkeypatch):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=-100.0, n_atoms=len(slab.atoms))
        confs = [_make_atoms()]
        # create_conformers succeeds, but optimize returns no results.
        _patch_reference_helpers(monkeypatch, conformers=(confs, [0.0]), opt_results=[])

        with pytest.raises(OptimizationError, match="any conformers"):
            reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(fail_on_conformer_failure=True),
            )

    def test_empty_optimization_results_omits_molecule_when_lenient(
        self, monkeypatch, caplog
    ):
        slab = SlabContainer(make_slab())
        calc = mock_calculator(energy=-100.0, n_atoms=len(slab.atoms))
        confs = [_make_atoms()]
        _patch_reference_helpers(monkeypatch, conformers=(confs, [0.0]), opt_results=[])

        with caplog.at_level(logging.ERROR):
            ref = reference.calculate_reference_energies(
                slab,
                calc,
                ["water"],
                ["O"],
                ts_model=object(),
                config=AdsorptionConfig(fail_on_conformer_failure=False),
            )
        assert ref.slab_energy == -100.0
        assert "water" not in ref.molecule_energies
        assert any(
            "Failed to optimise any conformers for water" in rec.getMessage()
            for rec in caplog.records
        )

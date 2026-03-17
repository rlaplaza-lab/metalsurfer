"""Tests for graceful behavior when optional dependencies are missing.

Uses ``unittest.mock.patch`` to simulate missing packages without
actually uninstalling them.  The restore ``importlib.reload()`` is
placed **after** the ``patch.dict`` context manager exits so that
``sys.modules`` is fully restored before the module is re-imported;
this prevents module-level sentinels (e.g. ``Chem = None`` in
``conformers.py``) from staying broken for subsequent tests.

Tests that reload the optimization module are skipped when torch is
installed, since optimization does a top-level import of torch and
patching is unreliable once torch is loaded.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.exceptions import DependencyMissingError
from tests.optional_deps import has_torch

pytestmark = pytest.mark.dependency_behavior


# ---------------------------------------------------------------------------
# RDKit missing — conformer generation
# ---------------------------------------------------------------------------


class TestMissingRDKit:
    def test_conformer_generation_raises_clear_error(self):
        """create_conformers_from_smiles must raise RuntimeError with install hint."""
        import importlib

        import metalsurfer.conformers as mod

        with patch.dict(
            sys.modules, {"rdkit": None, "rdkit.Chem": None, "rdkit.Chem.AllChem": None}
        ):
            importlib.reload(mod)
            with pytest.raises(RuntimeError, match="RDKit is required"):
                mod.create_conformers_from_smiles(
                    "O", config=AdsorptionConfig(num_conformers=1)
                )
        importlib.reload(mod)

    def test_filter_helpers_degrade_gracefully(self):
        """RDKit-dependent filter helpers should return None, not crash."""
        import importlib

        import metalsurfer.filters as fmod

        with patch.dict(sys.modules, {"rdkit": None, "rdkit.Chem": None}):
            importlib.reload(fmod)
            assert fmod._formula_from_smiles("O") is None
            assert fmod._bond_counts_from_smiles("O") is None
            assert fmod._coordination_fingerprint_from_smiles("O") is None
        importlib.reload(fmod)

    def test_check_decomposition_raises_when_rdkit_missing(self):
        """check_decomposition must raise DependencyMissingError when RDKit missing and reference_smiles provided."""
        import importlib

        import metalsurfer.filters as fmod

        with patch.dict(sys.modules, {"rdkit": None, "rdkit.Chem": None}):
            importlib.reload(fmod)
            from ase import Atoms

            atoms = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            with pytest.raises(DependencyMissingError, match="rdkit"):
                fmod.check_decomposition(
                    atoms,
                    reference_smiles="O",
                    surface_symbols=None,
                    connectivity_multipliers=[1.3],
                )
        importlib.reload(fmod)

    def test_is_flat_aromatic_with_en_raises_when_rdkit_missing(self):
        """_is_flat_aromatic_with_en must raise DependencyMissingError when RDKit missing."""
        import importlib

        import metalsurfer.placement.generators as gmod

        with patch.dict(sys.modules, {"rdkit": None, "rdkit.Chem": None}):
            importlib.reload(gmod)
            with pytest.raises(DependencyMissingError, match="rdkit"):
                gmod._is_flat_aromatic_with_en("c1ccccc1O")
        importlib.reload(gmod)


# ---------------------------------------------------------------------------
# torch-sim missing — batched optimisation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    has_torch,
    reason="torch installed; optimization does top-level import, patching unreliable",
)
class TestMissingTorchSim:
    def test_optimize_isolated_raises_clear_error(self):
        """optimize_isolated_molecules_batched must raise DependencyMissingError."""
        import importlib

        import metalsurfer.optimization as omod

        with patch.dict(
            sys.modules,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            importlib.reload(omod)
            from ase import Atoms

            water = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            with pytest.raises(DependencyMissingError, match="torch-sim"):
                omod.optimize_isolated_molecules_batched([water], ts_model=MagicMock())
        importlib.reload(omod)

    def test_optimize_slab_raises_clear_error(self):
        """optimize_adsorbate_slab_batched must raise DependencyMissingError."""
        import importlib

        import metalsurfer.optimization as omod

        with patch.dict(
            sys.modules,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            importlib.reload(omod)
            from ase import Atoms

            slab = Atoms("Ru4", positions=[[i, 0, 0] for i in range(4)])
            slab.set_cell([10, 10, 10])
            slab.set_pbc(True)
            combined = slab.copy()
            with pytest.raises(DependencyMissingError, match="torch-sim"):
                omod.optimize_adsorbate_slab_batched(
                    [combined], slab, ts_model=MagicMock()
                )
        importlib.reload(omod)

    def test_autobatcher_returns_none_when_unavailable(self):
        """_get_inflight_autobatcher should return None gracefully."""
        import importlib

        import metalsurfer.optimization as omod

        with patch.dict(
            sys.modules,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            importlib.reload(omod)
            result = omod._get_inflight_autobatcher(ts_model=None, max_n_atoms=0)
            assert result is None
        importlib.reload(omod)


# ---------------------------------------------------------------------------
# FAIRChem missing — calculator setup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    has_torch,
    reason="torch installed; optimization does top-level import, patching unreliable",
)
class TestMissingFAIRChem:
    def test_setup_calculator_raises(self):
        """setup_calculator must raise DependencyMissingError when fairchem is missing."""
        import importlib

        import metalsurfer.optimization as omod

        with patch.dict(
            sys.modules,
            {
                "fairchem": None,
                "fairchem.core": None,
            },
        ):
            importlib.reload(omod)
            with pytest.raises(DependencyMissingError, match="fairchem"):
                omod.setup_calculator()
        importlib.reload(omod)

    def test_setup_torchsim_model_raises(self):
        """setup_torchsim_model must raise DependencyMissingError when torch_sim.models is missing."""
        import importlib

        import metalsurfer.optimization as omod

        with patch.dict(
            sys.modules,
            {
                "torch_sim.models": None,
                "torch_sim.models.fairchem": None,
            },
        ):
            importlib.reload(omod)
            with pytest.raises(DependencyMissingError, match="torch-sim"):
                omod.setup_torchsim_model()
        importlib.reload(omod)

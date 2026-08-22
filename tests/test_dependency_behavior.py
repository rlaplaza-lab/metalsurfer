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

import builtins
import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.exceptions import DependencyMissingError
from metalsurfer.surface_prep import create_slab_from_bulk
from tests.optional_deps import has_torch

pytestmark = pytest.mark.dependency_behavior

_real_import = builtins.__import__


@contextmanager
def _reload_with_missing(module, missing: dict[str, object]):
    """Temporarily hide optional deps, reload module, then restore cleanly."""
    with patch.dict(sys.modules, missing):
        importlib.reload(module)
        yield module
    importlib.reload(module)


# ---------------------------------------------------------------------------
# RDKit missing — conformer generation
# ---------------------------------------------------------------------------


class TestMissingRDKit:
    def test_conformer_generation_raises_clear_error(self):
        """create_conformers_from_smiles must raise DependencyMissingError with install hint."""
        import metalsurfer.conformers as mod

        with (
            _reload_with_missing(
                mod, {"rdkit": None, "rdkit.Chem": None, "rdkit.Chem.AllChem": None}
            ),
            pytest.raises(DependencyMissingError, match="rdkit"),
        ):
            mod.create_conformers_from_smiles(
                "O", config=AdsorptionConfig(num_conformers=1)
            )

    def test_filter_helpers_degrade_gracefully(self):
        """RDKit-dependent filter helpers should return None, not crash."""
        import metalsurfer.filters as fmod

        with _reload_with_missing(fmod, {"rdkit": None, "rdkit.Chem": None}):
            assert fmod._formula_from_smiles("O") is None
            assert fmod._bond_counts_from_smiles("O") is None
            assert fmod._coordination_fingerprint_from_smiles("O") is None

    def test_check_decomposition_raises_when_rdkit_missing(self):
        """check_decomposition must raise DependencyMissingError when RDKit missing and reference_smiles provided."""
        import metalsurfer.filters as fmod

        with _reload_with_missing(fmod, {"rdkit": None, "rdkit.Chem": None}):
            from ase import Atoms

            atoms = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            with pytest.raises(DependencyMissingError, match="rdkit"):
                fmod.check_decomposition(
                    atoms,
                    reference_smiles="O",
                    surface_symbols=None,
                    connectivity_multiplier=1.3,
                )

    def test_is_flat_aromatic_with_en_raises_when_rdkit_missing(self):
        """_is_flat_aromatic_with_en must raise DependencyMissingError when RDKit missing."""
        import metalsurfer.placement.orientation as omod

        with (
            _reload_with_missing(omod, {"rdkit": None, "rdkit.Chem": None}),
            pytest.raises(DependencyMissingError, match="rdkit"),
        ):
            omod._is_flat_aromatic_with_en("c1ccccc1O")


# ---------------------------------------------------------------------------
# torch-sim missing — batched optimisation
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    has_torch,
    reason="torch installed (covered in CI job without torch); patching unreliable when present",
)
class TestMissingTorchSim:
    def test_optimize_isolated_raises_clear_error(self):
        """optimize_isolated_molecules_batched must raise DependencyMissingError."""
        import metalsurfer.optimization as omod

        with _reload_with_missing(
            omod,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            from ase import Atoms

            water = Atoms("OH2", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
            with pytest.raises(DependencyMissingError, match="torch-sim"):
                omod.optimize_isolated_molecules_batched(
                    [water],
                    ts_model=MagicMock(),
                    config=AdsorptionConfig(),
                )

    def test_optimize_slab_raises_clear_error(self):
        """optimize_adsorbate_slab_batched must raise DependencyMissingError."""
        import metalsurfer.optimization as omod

        with _reload_with_missing(
            omod,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            from ase import Atoms

            slab = Atoms("Ru4", positions=[[i, 0, 0] for i in range(4)])
            slab.set_cell([10, 10, 10])
            slab.set_pbc(True)
            combined = slab.copy()
            with pytest.raises(DependencyMissingError, match="torch-sim"):
                omod.optimize_adsorbate_slab_batched(
                    [combined], slab, ts_model=MagicMock()
                )

    def test_autobatcher_returns_none_when_unavailable(self):
        """_get_inflight_autobatcher should degrade gracefully to a null triple.

        The accessor always returns ``(autobatcher, cache_key,
        reused_prior_estimate)`` so the unpacking call sites in
        ``_optimize`` keep working when the optional MLIP stack is missing.
        """
        import metalsurfer.optimization as omod

        with _reload_with_missing(
            omod,
            {
                "torch_sim": None,
                "torch_sim.autobatching": None,
                "torch_sim.constraints": None,
            },
        ):
            autobatcher, cache_key, reused = omod._cache._get_inflight_autobatcher(
                ts_model=None, max_n_atoms=0
            )
            assert autobatcher is None
            assert cache_key is None
            assert reused is False


# ---------------------------------------------------------------------------
# create_slab_from_bulk — import errors
# ---------------------------------------------------------------------------


class TestCreateSlabFromBulkImportErrors:
    """``create_slab_from_bulk`` must report setuptools vs fairchem-data-oc accurately."""

    def test_raises_setuptools_hint_when_pkg_resources_missing(self, tmp_path):
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fairchem.data.oc.core":
                raise ModuleNotFoundError("pkg_resources")
            return _real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(DependencyMissingError, match="setuptools"),
        ):
            create_slab_from_bulk(
                "mp-23", results_dir=str(tmp_path / "results_test_dep")
            )

    def test_raises_fairchem_data_oc_hint_when_fairchem_missing(self, tmp_path):
        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fairchem.data.oc.core":
                raise ModuleNotFoundError("No module named 'fairchem.data.oc'")
            return _real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(DependencyMissingError, match="fairchem-data-oc"),
        ):
            create_slab_from_bulk(
                "mp-23", results_dir=str(tmp_path / "results_test_dep")
            )


# ---------------------------------------------------------------------------
# FAIRChem missing — calculator setup
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    has_torch,
    reason="torch installed (covered in CI job without torch); patching unreliable when present",
)
class TestMissingFAIRChem:
    def test_setup_single_model_raises(self):
        """setup_single_model must raise DependencyMissingError when torch_sim is missing."""
        import metalsurfer.optimization as omod

        with (
            _reload_with_missing(
                omod,
                {"torch_sim.models": None, "torch_sim.models.fairchem": None},
            ),
            pytest.raises(DependencyMissingError, match="torch-sim"),
        ):
            omod.setup_single_model()

    def test_setup_torchsim_model_raises(self):
        """setup_torchsim_model must raise DependencyMissingError when torch_sim.models is missing."""
        import metalsurfer.optimization as omod

        with (
            _reload_with_missing(
                omod,
                {"torch_sim.models": None, "torch_sim.models.fairchem": None},
            ),
            pytest.raises(DependencyMissingError, match="torch-sim"),
        ):
            omod.setup_torchsim_model()

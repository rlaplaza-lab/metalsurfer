"""Tests that core imports work without optional heavy dependencies."""

import sys
from contextlib import contextmanager

import pytest


@contextmanager
def _isolated_metalsurfer_modules():
    saved = {
        mod_name: module
        for mod_name, module in sys.modules.items()
        if mod_name.startswith("metalsurfer")
    }
    for mod_name in list(saved):
        del sys.modules[mod_name]
    try:
        yield
    finally:
        for mod_name in list(sys.modules):
            if mod_name.startswith("metalsurfer"):
                del sys.modules[mod_name]
        sys.modules.update(saved)


def test_core_import_without_heavy_deps(monkeypatch):
    """Importing metalsurfer must succeed even if torch/fairchem/rdkit
    are unavailable.  The lazy __getattr__ should defer those imports.
    """
    blocked = {
        "torch",
        "torch_sim",
        "torch_sim.autobatching",
        "torch_sim.constraints",
        "torch_sim.models",
        "torch_sim.models.fairchem",
        "fairchem",
        "fairchem.core",
        "fairchem.data",
        "fairchem.data.oc",
        "fairchem.data.oc.core",
        "rdkit",
        "rdkit.Chem",
        "sklearn",
        "sklearn.neighbors",
    }

    original_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def _blocking_import(name, *args, **kwargs):
        top = name.split(".")[0]
        if name in blocked or top in blocked:
            raise ImportError(f"Blocked for test: {name}")
        return original_import(name, *args, **kwargs)

    with _isolated_metalsurfer_modules(), monkeypatch.context() as m:
        m.setattr("builtins.__import__", _blocking_import)
        import metalsurfer

        assert hasattr(metalsurfer, "AdsorptionConfig")
        assert hasattr(metalsurfer, "ReferenceEnergies")
        assert hasattr(metalsurfer, "ScreeningResult")
        assert hasattr(metalsurfer, "DependencyMissingError")

        cfg = metalsurfer.AdsorptionConfig(num_conformers=3)
        assert cfg.num_conformers == 3


def test_typed_models_importable():
    """Typed models can be imported directly from the package."""
    import metalsurfer
    from metalsurfer import (
        BindingCampaignResult,
        MoleculeCampaignSummary,
        MoleculeSummary,
        ReferenceEnergies,
        ScreeningResult,
        ScreeningRunResult,
        TimingInfo,
    )

    exported_models = {
        "ReferenceEnergies": ReferenceEnergies,
        "BindingCampaignResult": BindingCampaignResult,
        "MoleculeCampaignSummary": MoleculeCampaignSummary,
        "ScreeningResult": ScreeningResult,
        "MoleculeSummary": MoleculeSummary,
        "ScreeningRunResult": ScreeningRunResult,
        "TimingInfo": TimingInfo,
    }
    assert set(exported_models).issubset(set(metalsurfer.__all__))
    assert all(isinstance(symbol, type) for symbol in exported_models.values())


def test_exceptions_importable():
    """Exception classes can be imported directly from the package."""
    from metalsurfer import (
        DependencyMissingError,
        GeometryValidationError,
        OptimizationError,
    )

    assert issubclass(DependencyMissingError, RuntimeError)
    assert issubclass(GeometryValidationError, ValueError)
    assert issubclass(OptimizationError, RuntimeError)


def test_lazy_getattr_loads_module():
    """Lazy __getattr__ loads submodules on first access of deferred symbols."""
    import metalsurfer

    # SlabContainer is in _LAZY_MODULES["surfaces"]
    slab_container = metalsurfer.SlabContainer
    assert slab_container.__name__ == "SlabContainer"


def test_surface_prep_import_path():
    """Surface-prep helpers are available under a dedicated import path."""
    from metalsurfer.surface_prep import (
        SlabContainer,
        create_slab_from_atoms,
        create_slab_from_bulk,
        deposit_adatoms,
        substitute_alloy,
    )

    assert SlabContainer.__name__ == "SlabContainer"
    assert callable(create_slab_from_bulk)
    assert callable(create_slab_from_atoms)
    assert callable(substitute_alloy)
    assert callable(deposit_adatoms)


def test_lazy_getattr_raises_for_unknown():
    """Lazy __getattr__ raises AttributeError for unknown attributes."""
    import metalsurfer

    with pytest.raises(AttributeError, match="has no attribute '_nonexistent'"):
        _ = metalsurfer._nonexistent


def test_device_resolution_fallback_for_ci():
    """Optimization resolves cuda->cpu when CUDA unavailable (e.g. GitHub Actions).

    Ensures tests and CI run without false failures when config defaults to
    device='cuda' but no GPU is present.
    """
    from unittest.mock import patch

    from metalsurfer.optimization import _resolve_device

    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda:0") in ("cuda:0", "cpu")  # cpu when no GPU
    # When cuda requested but unavailable, should return cpu
    with patch("metalsurfer.optimization.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        assert _resolve_device("cuda") == "cpu"
    with patch("metalsurfer.optimization.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        assert _resolve_device("cuda") == "cuda"

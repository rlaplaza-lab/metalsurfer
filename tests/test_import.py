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


def test_campaign_entry_points_in_all():
    """Primary run_* APIs are listed in __all__ for discoverability."""
    import metalsurfer

    for name in (
        "run_adsorption",
        "run_adsorption_bo",
        "run_saturation",
        "run_saturation_bo",
    ):
        assert name in metalsurfer.__all__


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

    # SlabContainer is lazy-exported from surface_prep
    slab_container = metalsurfer.SlabContainer
    assert slab_container.__name__ == "SlabContainer"


def test_surface_prep_import_path():
    """Surface-prep helpers are available under the unified surface_prep module."""
    from metalsurfer.surface_prep import (
        SlabContainer,
        accept_substrate_for_api,
        apply_material_pbc,
        auto_resize_substrate_for_molecule,
        create_slab_from_atoms,
        create_slab_from_bulk,
        deposit_adatoms,
        finalize_substrate,
        prepare_substrate,
        relax_substrate,
        resize_substrate_for_molecule,
        substitute_alloy,
        validate_substrate,
    )

    assert SlabContainer.__name__ == "SlabContainer"
    assert callable(create_slab_from_bulk)
    assert callable(create_slab_from_atoms)
    assert callable(substitute_alloy)
    assert callable(deposit_adatoms)
    assert callable(prepare_substrate)
    assert callable(finalize_substrate)
    assert callable(relax_substrate)
    assert callable(apply_material_pbc)
    assert callable(validate_substrate)
    assert callable(accept_substrate_for_api)
    assert callable(auto_resize_substrate_for_molecule)
    assert callable(resize_substrate_for_molecule)


def test_removed_surface_prep_aliases_are_not_exported():
    import metalsurfer
    import metalsurfer.surface_prep as surface_prep

    removed = (
        "prepare_slab",
        "resize_slab_for_molecule",
        "auto_resize_slab_for_molecule",
    )
    for name in removed:
        assert name not in surface_prep.__all__
        with pytest.raises(AttributeError):
            _ = getattr(surface_prep, name)
        with pytest.raises(AttributeError):
            _ = getattr(metalsurfer, name)


def test_removed_lazy_exports_are_not_available():
    """Removed public symbols must not reappear in lazy exports."""
    import metalsurfer

    removed = ("precompute_results",)
    for name in removed:
        with pytest.raises(AttributeError, match=f"has no attribute {name!r}"):
            _ = getattr(metalsurfer, name)


def test_removed_internal_modules_are_gone():
    """Deleted helper modules must not remain importable."""
    import importlib.util

    assert importlib.util.find_spec("metalsurfer._timing") is None


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

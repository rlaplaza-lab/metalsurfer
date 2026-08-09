"""Machine learning pipeline for binding energy prediction from placement descriptors.

The sklearn-backed modules (``regression``, ``predict``) are imported lazily via
:func:`__getattr__` so ``import metalsurfer`` does not pay for sklearn up front.
The lightweight, sklearn-free helpers (``dataset``, ``features``, ``schema``) are
imported eagerly because other modules depend on them at import time.
"""

from .dataset import DatasetLogger, load_dataset
from .features import extract_features, extract_features_from_dataset
from .schema import ComputationContext, PlacementRecord

__all__ = [
    "BindingEnergyPredictor",
    "ComputationContext",
    "DatasetLogger",
    "PlacementRecord",
    "evaluate_model",
    "extract_features",
    "extract_features_from_dataset",
    "grouped_cross_validate",
    "load_dataset",
    "predict",
    "regression",
    "train_model",
]

_LAZY_MODULES = {"regression", "predict"}
_LAZY_FUNCTIONS = {
    "evaluate_model": "regression",
    "grouped_cross_validate": "regression",
    "train_model": "regression",
    "BindingEnergyPredictor": "predict",
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    mod_name = _LAZY_FUNCTIONS.get(name)
    if mod_name is not None:
        import importlib

        return getattr(importlib.import_module(f".{mod_name}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

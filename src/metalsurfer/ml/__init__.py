"""Placement-record dataset, features, and BO surrogate estimators.

The sklearn-backed ``regression`` module is imported lazily via
:func:`__getattr__` so ``import metalsurfer`` does not pay for sklearn up front.
The lightweight, sklearn-free helpers (``dataset``, ``features``, ``schema``) are
imported eagerly because other modules depend on them at import time.
"""

from .dataset import DatasetLogger, load_dataset
from .features import (
    FEATURE_ABS_COLUMNS,
    FEATURE_NAMES,
    extract_features,
    extract_features_from_dataset,
    placement_pose_from_features,
)
from .schema import ComputationContext, PlacementRecord

__all__ = [
    "ComputationContext",
    "DatasetLogger",
    "FEATURE_ABS_COLUMNS",
    "FEATURE_NAMES",
    "PlacementRecord",
    "extract_features",
    "extract_features_from_dataset",
    "load_dataset",
    "placement_pose_from_features",
    "regression",
]

_LAZY_MODULES = {"regression"}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        import importlib

        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

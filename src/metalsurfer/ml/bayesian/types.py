"""Shared type aliases for Bayesian optimisation helpers."""

from typing import Literal

from ..regression import TreeSurrogateKind

AcquisitionType = Literal["lcb", "ei", "pi"]
InitialSamplingType = Literal["random", "spread", "spread_xyz", "stratified"]
SurrogateType = Literal[
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "gaussian_process",
    "ensemble",
]
TransferCapableSurrogateType = Literal[
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "ensemble",
]
DEFAULT_ENSEMBLE_MEMBERS: tuple[
    TreeSurrogateKind | Literal["ridge", "gaussian_process"], ...
] = (
    "random_forest",
    "extra_trees",
    "ridge",
    "gaussian_process",
)

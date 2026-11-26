"""Shared type aliases for Bayesian optimisation helpers."""

from typing import Literal

from ...config import (
    BOAcquisition,
    BOInitialSampling,
    BOSurrogate,
    BOTransferCapableSurrogate,
)
from ..regression import TreeSurrogateKind

AcquisitionType = BOAcquisition
InitialSamplingType = BOInitialSampling
SurrogateType = BOSurrogate
TransferCapableSurrogateType = BOTransferCapableSurrogate

DEFAULT_ENSEMBLE_MEMBERS: tuple[
    TreeSurrogateKind | Literal["ridge", "gaussian_process"], ...
] = (
    "random_forest",
    "extra_trees",
    "ridge",
    "gaussian_process",
)

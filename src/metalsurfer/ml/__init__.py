"""Machine learning pipeline for binding energy prediction from placement descriptors."""

from .bayesian import (
    AcquisitionType,
    build_candidate_features,
    build_spec_features_geometry_aware,
    ei_scores,
    lcb_scores,
    pi_scores,
    predict_with_uncertainty,
    score_and_select,
    train_surrogate,
)
from .dataset import DatasetLogger, load_dataset
from .features import extract_features, extract_features_from_dataset
from .predict import BindingEnergyPredictor
from .regression import (
    evaluate_model,
    grouped_cross_validate,
    train_model,
)
from .schema import ComputationContext, PlacementRecord

__all__ = [
    "AcquisitionType",
    "BindingEnergyPredictor",
    "ComputationContext",
    "DatasetLogger",
    "PlacementRecord",
    "build_candidate_features",
    "build_spec_features_geometry_aware",
    "ei_scores",
    "evaluate_model",
    "extract_features",
    "extract_features_from_dataset",
    "grouped_cross_validate",
    "load_dataset",
    "pi_scores",
    "predict_with_uncertainty",
    "score_and_select",
    "train_model",
    "train_surrogate",
    "lcb_scores",
]

"""Surrogate training, uncertainty-aware prediction, and acquisition scoring for BO."""

from .acquisition import (
    build_spec_features_geometry_aware,
    ei_scores,
    lcb_scores,
    pi_scores,
    score_and_select,
    select_candidates,
    select_candidates_batch_diverse,
    select_initial_bo_indices,
    splice_exploration_picks,
)
from .surrogate import (
    EnsembleRegressor,
    matern_length_scale_for_n_features,
    predict_with_uncertainty,
    train_surrogate,
)
from .transfer import (
    TransferSurrogateResult,
    build_cumulative_refit_surrogate,
    build_transfer_surrogate,
    cumulative_refit_training_set,
    prior_proximity_weights,
    prior_recency_weights,
    prior_similarity_to_current,
)
from .transfer import _capped_prior_weights as _capped_prior_weights
from .types import (
    DEFAULT_ENSEMBLE_MEMBERS,
    AcquisitionType,
    InitialSamplingType,
    SurrogateType,
    TransferCapableSurrogateType,
)

__all__ = [
    "AcquisitionType",
    "DEFAULT_ENSEMBLE_MEMBERS",
    "EnsembleRegressor",
    "InitialSamplingType",
    "SurrogateType",
    "TransferCapableSurrogateType",
    "TransferSurrogateResult",
    "build_cumulative_refit_surrogate",
    "build_spec_features_geometry_aware",
    "build_transfer_surrogate",
    "cumulative_refit_training_set",
    "ei_scores",
    "lcb_scores",
    "matern_length_scale_for_n_features",
    "pi_scores",
    "predict_with_uncertainty",
    "prior_proximity_weights",
    "prior_recency_weights",
    "prior_similarity_to_current",
    "score_and_select",
    "select_candidates",
    "select_candidates_batch_diverse",
    "select_initial_bo_indices",
    "splice_exploration_picks",
    "train_surrogate",
]

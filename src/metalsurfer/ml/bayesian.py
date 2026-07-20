"""Surrogate training, uncertainty-aware prediction, and acquisition scoring for BO."""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from ase import Atoms
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import BO_INITIAL_SAMPLING_OPTIONS, AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement import generators as placement_generators
from .features import extract_features
from .regression import (
    TreeSurrogateKind,
    _build_estimator,
    train_model,
    tree_regressor_for_bayesian_surrogate,
)
from .schema import PlacementRecord

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
TreeSurrogateType = Literal["random_forest", "extra_trees"]
DEFAULT_ENSEMBLE_MEMBERS: tuple[
    TreeSurrogateType | Literal["ridge", "gaussian_process"], ...
] = (
    "random_forest",
    "extra_trees",
    "ridge",
    "gaussian_process",
)

logger = logging.getLogger(__name__)


def matern_length_scale_for_n_features(n_features: int) -> float:
    """Characteristic length scale for BO GP: sqrt(number of features)."""
    if n_features < 1:
        raise ValueError(f"n_features must be >= 1, got {n_features}")
    return float(np.sqrt(n_features))


def _gaussian_process_regressor(
    n_features: int,
    random_state: int,
) -> GaussianProcessRegressor:
    """Matern GP with fixed length scale sqrt(n_features)."""
    length_scale = matern_length_scale_for_n_features(n_features)
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(
        length_scale=length_scale,
        length_scale_bounds="fixed",
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=0,
    )


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Average several BO surrogates; combine mean and disagreement as uncertainty."""

    def __init__(
        self,
        member_surrogates: tuple[str, ...] = DEFAULT_ENSEMBLE_MEMBERS,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.member_surrogates = member_surrogates
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.members_: list[Pipeline] = []

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "EnsembleRegressor":
        self.members_ = []
        for spec in self.member_surrogates:
            if spec == "ensemble":
                raise ValueError("EnsembleRegressor cannot nest another ensemble")
            weight = (
                sample_weight
                if spec in ("random_forest", "extra_trees", "ridge")
                and sample_weight is not None
                else None
            )
            self.members_.append(
                train_surrogate(
                    X,
                    y,
                    surrogate=spec,  # type: ignore[arg-type]
                    n_estimators=self.n_estimators,
                    random_state=self.random_state,
                    sample_weight=weight,
                )
            )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        mu, _ = self.predict_with_uncertainty(X)
        return mu

    def predict_with_uncertainty(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.members_:
            raise RuntimeError("EnsembleRegressor is not fitted")
        mus: list[np.ndarray] = []
        sigmas: list[np.ndarray] = []
        for member in self.members_:
            mu_i, sigma_i = predict_with_uncertainty(member, X)
            mus.append(np.asarray(mu_i, dtype=float).ravel())
            sigmas.append(np.asarray(sigma_i, dtype=float).ravel())
        mus_arr = np.vstack(mus)
        mu_ens = mus_arr.mean(axis=0)
        sigmas_arr = np.vstack(sigmas)
        aleatoric = np.mean(np.square(sigmas_arr), axis=0)
        epistemic = np.var(mus_arr, axis=0)
        sigma_ens = np.sqrt(np.maximum(aleatoric + epistemic, 0.0))
        return mu_ens, sigma_ens


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


def _tree_pipeline_fit_kwargs(
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    if sample_weight is None:
        return {}
    return {"regressor__sample_weight": np.asarray(sample_weight, dtype=float)}


_RESIDUAL_STD_FLOOR = 1e-3


def _attach_residual_uncertainty(
    pipeline: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> None:
    """Store residual RMSE and scaled training features for distance-aware σ.

    Deterministic surrogates (ridge / HGB) have no epistemic σ from the
    estimator itself. Using residual RMSE plus nearest-neighbour distance in
    scaled feature space restores usable EI/PI/LCB without changing the mean
    predictor.
    """
    regressor = pipeline.named_steps["regressor"]
    scaler = pipeline.named_steps.get("scaler")
    X_scaled = scaler.transform(X) if scaler is not None else np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    resid = y_arr - np.asarray(pipeline.predict(X), dtype=float).ravel()
    n = int(resid.size)
    p = int(np.asarray(X_scaled).shape[1]) if np.ndim(X_scaled) == 2 else 1
    dof = max(n - p - 1, 1)
    residual_std = float(np.sqrt(np.sum(np.square(resid)) / dof))
    regressor.bo_residual_std_ = max(residual_std, _RESIDUAL_STD_FLOOR)
    regressor.bo_X_train_scaled_ = np.asarray(X_scaled, dtype=float)


def _sigma_from_residual(
    regressor: Any,
    X_eval: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Build per-candidate σ from attached residual stats, else zeros.

    Uses residual RMSE with mild nearest-neighbour inflation, capped at
    ``2 * residual_std`` so EI/PI do not chase arbitrarily far pool points.
    """
    residual_std = getattr(regressor, "bo_residual_std_", None)
    if residual_std is None or not np.isfinite(residual_std) or residual_std <= 0:
        return np.zeros_like(mu)
    base = float(residual_std)
    X_train = getattr(regressor, "bo_X_train_scaled_", None)
    if X_train is None or len(X_train) == 0:
        return np.full_like(mu, base)
    X_e = np.asarray(X_eval, dtype=float)
    d = cdist(X_e, np.asarray(X_train, dtype=float)).min(axis=1)
    if len(X_train) >= 2:
        d_train = cdist(
            np.asarray(X_train, dtype=float), np.asarray(X_train, dtype=float)
        )
        np.fill_diagonal(d_train, np.inf)
        lengthscale = float(np.median(d_train.min(axis=1)))
        lengthscale = max(lengthscale, _RESIDUAL_STD_FLOOR)
    else:
        lengthscale = 1.0
    # Mild distance tempering; cap prevents EI from ignoring the mean.
    sigma = base * (1.0 + 0.25 * (d / lengthscale))
    return np.minimum(sigma, 2.0 * base)


def train_surrogate(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    surrogate: SurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    **kwargs: Any,
) -> Pipeline:
    """Fit a surrogate on observed placement data.

    Tree ensembles (``random_forest``, ``extra_trees``) return a single-step
    ``Pipeline`` with a regressor only. ``gradient_boost`` and ``ridge`` return
    a ``scaler`` + ``regressor`` pipeline from :func:`regression.train_model`.
    Per-sample ``sample_weight`` is supported for tree ensembles and ``ridge``.
    """
    if surrogate in ("random_forest", "extra_trees"):
        tree_kind: TreeSurrogateKind = (
            "random_forest" if surrogate == "random_forest" else "extra_trees"
        )
        reg = tree_regressor_for_bayesian_surrogate(
            tree_kind,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs,
        )
        pipeline = Pipeline([("regressor", reg)])
        pipeline.fit(X, y, **_tree_pipeline_fit_kwargs(sample_weight))
        logger.info(
            "Trained %s surrogate on %d samples (%d trees)",
            surrogate,
            len(np.asarray(y)),
            n_estimators,
        )
        return pipeline
    if surrogate == "ridge":
        pipeline = _build_estimator("ridge", random_state=random_state, **kwargs)
        pipeline.fit(X, y, **_tree_pipeline_fit_kwargs(sample_weight))
        _attach_residual_uncertainty(pipeline, X, y)
        logger.info(
            "Trained ridge surrogate on %d samples (residual_std=%.4f)",
            len(np.asarray(y)),
            float(pipeline.named_steps["regressor"].bo_residual_std_),
        )
        return pipeline
    if surrogate == "gradient_boost":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is only supported for tree surrogates and ridge, "
                f"not {surrogate!r}"
            )
        pipeline = train_model(
            X,
            y,
            model_type="gradient_boost",
            random_state=random_state,
            **kwargs,
        )
        _attach_residual_uncertainty(pipeline, X, y)
        return pipeline
    if surrogate == "gaussian_process":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is only supported for tree surrogates and ridge, "
                f"not {surrogate!r}"
            )
        n_features = int(X.shape[1]) if hasattr(X, "shape") else len(X[0])
        reg = _gaussian_process_regressor(n_features, random_state)
        pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
        pipeline.fit(X, y)
        logger.info(
            "Trained gaussian_process surrogate on %d samples "
            "(Matern length_scale=%.4f = sqrt(%d))",
            len(np.asarray(y)),
            matern_length_scale_for_n_features(n_features),
            n_features,
        )
        return pipeline
    if surrogate == "ensemble":
        reg = EnsembleRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
        )
        pipeline = Pipeline([("regressor", reg)])
        pipeline.fit(X, y, **_tree_pipeline_fit_kwargs(sample_weight))
        logger.info(
            "Trained ensemble surrogate on %d samples (%d members: %s)",
            len(np.asarray(y)),
            len(reg.member_surrogates),
            ", ".join(reg.member_surrogates),
        )
        return pipeline
    raise ValueError(f"Unknown surrogate: {surrogate!r}")


@dataclass
class TransferSurrogateResult:
    """Outcome of one BO round's transfer-augmented surrogate training."""

    surrogate: Pipeline
    transfer_used_this_round: bool
    transfer_weight_share: float
    transfer_mae_delta: float | None
    transfer_bad_rounds: int
    transfer_disabled: bool
    transfer_disabled_reason: str | None


def prior_similarity_to_current(
    X_prior: pd.DataFrame,
    X_current: pd.DataFrame,
    *,
    lengthscale: float,
) -> np.ndarray:
    """Similarity of each prior row to the nearest current-step placement (feature space)."""
    if len(X_prior) == 0 or len(X_current) == 0:
        return np.array([], dtype=float)
    columns = list(X_current.columns)
    prior_vals = X_prior.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
    current_vals = X_current.reindex(columns=columns, fill_value=0.0).to_numpy(
        dtype=float
    )
    dists = np.linalg.norm(
        prior_vals[:, None, :] - current_vals[None, :, :],
        axis=2,
    )
    min_dist = np.min(dists, axis=1)
    return np.exp(-min_dist / float(lengthscale))


def prior_recency_weights(
    step_ages: np.ndarray | list[int],
    *,
    lengthscale: float,
) -> np.ndarray:
    """Exponential decay for older saturation-step observations (age 0 = most recent)."""
    ages = np.asarray(step_ages, dtype=float)
    if ages.size == 0:
        return np.array([], dtype=float)
    return np.exp(-ages / float(lengthscale))


def prior_placement_downweight(
    X_prior: pd.DataFrame,
    placement_X: pd.DataFrame,
    *,
    lengthscale: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Reduce transfer weight for prior rows near an executed placement site."""
    if len(X_prior) == 0:
        return np.array([], dtype=float)
    if len(placement_X) == 0:
        return np.ones(len(X_prior), dtype=float)
    columns = list(X_prior.columns)
    prior_vals = X_prior.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
    place_vals = placement_X.reindex(columns=columns, fill_value=0.0).to_numpy(
        dtype=float
    )
    anchor = place_vals[0]
    dists = np.linalg.norm(prior_vals - anchor[None, :], axis=1)
    near = np.exp(-dists / float(lengthscale))
    return np.maximum(floor, 1.0 - near)


def prior_proximity_weights(
    X_prior: pd.DataFrame,
    X_anchor: pd.DataFrame,
    *,
    lengthscale: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Downweight prior observations near executed placement sites in feature space."""
    if len(X_prior) == 0 or len(X_anchor) == 0:
        return np.array([], dtype=float)
    columns = list(X_anchor.columns)
    prior_vals = X_prior.reindex(columns=columns, fill_value=0.0).to_numpy(dtype=float)
    anchor_vals = X_anchor.reindex(columns=columns, fill_value=0.0).to_numpy(
        dtype=float
    )
    dists = np.linalg.norm(
        prior_vals[:, None, :] - anchor_vals[None, :, :],
        axis=2,
    )
    min_dist = np.empty(len(prior_vals), dtype=float)
    for j in range(len(prior_vals)):
        row_dists = dists[j]
        other_dists = row_dists[row_dists > 1e-12]
        min_dist[j] = float(np.min(other_dists)) if other_dists.size else np.inf
    proximity = np.exp(-min_dist / float(lengthscale))
    # No other anchors: treat as fully transferable.
    proximity = np.where(np.isfinite(min_dist), proximity, 1.0)
    return np.maximum(floor, proximity)


def cumulative_refit_sample_weights(
    n_current: int,
    X_prior: pd.DataFrame,
    X_anchor: pd.DataFrame,
    *,
    weight_cap: float,
    proximity_lengthscale: float,
    proximity_floor: float = 0.0,
) -> np.ndarray:
    """Sample weights for cumulative refit: current rows at 1.0, prior rows proximity-decayed."""
    n_prior = len(X_prior)
    if n_prior == 0:
        return np.ones(n_current, dtype=float)
    prox = prior_proximity_weights(
        X_prior,
        X_anchor,
        lengthscale=proximity_lengthscale,
        floor=proximity_floor,
    )
    if float(np.sum(prox)) <= 0.0:
        return np.concatenate(
            [np.ones(n_current, dtype=float), np.zeros(n_prior, dtype=float)]
        )
    max_transfer_weight = n_current * weight_cap / max(1.0 - weight_cap, 1e-8)
    prior_weights = prox / max(float(np.sum(prox)), 1e-8) * max_transfer_weight
    return np.concatenate(
        [np.ones(n_current, dtype=float), prior_weights.astype(float)]
    )


def build_transfer_surrogate(
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    observed_X_prev: pd.DataFrame | list[dict[str, float]],
    observed_y_prev: np.ndarray | list[float],
    *,
    surrogate: TreeSurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    weight_cap: float = 0.35,
    similarity_lengthscale: float = 1.0,
    min_similarity: float = 0.05,
    mae_tolerance: float = 0.0,
    transfer_bad_rounds: int = 0,
    trust_patience: int = 2,
    proximity_lengthscale: float | None = None,
    proximity_floor: float = 0.0,
    prior_step_ages: list[int] | None = None,
    recency_lengthscale: float | None = None,
    prior_placement_X: pd.DataFrame
    | list[dict[str, float]]
    | dict[str, float]
    | None = None,
    occupancy_lengthscale: float | None = None,
    occupancy_floor: float = 0.0,
) -> TransferSurrogateResult:
    """Train baseline or transfer-weighted surrogate with production trust gating.

    Mirrors the transfer block in :func:`workflow.bayesian.process_molecule_bayesian`.
    """
    y_current = np.asarray(y_current, dtype=float)
    baseline = train_surrogate(
        X_current,
        y_current,
        surrogate=surrogate,
        n_estimators=n_estimators,
        random_state=random_state,
    )
    if len(observed_X_prev) == 0 or len(observed_y_prev) == 0:
        return TransferSurrogateResult(
            surrogate=baseline,
            transfer_used_this_round=False,
            transfer_weight_share=0.0,
            transfer_mae_delta=None,
            transfer_bad_rounds=transfer_bad_rounds,
            transfer_disabled=False,
            transfer_disabled_reason=None,
        )

    X_prev = (
        pd.DataFrame(observed_X_prev)
        if not isinstance(observed_X_prev, pd.DataFrame)
        else observed_X_prev.copy()
    )
    y_prev = np.asarray(observed_y_prev, dtype=float)
    X_prev = X_prev.reindex(columns=X_current.columns, fill_value=0.0)
    prox_lengthscale = (
        float(similarity_lengthscale)
        if proximity_lengthscale is None
        else float(proximity_lengthscale)
    )
    occ_lengthscale = (
        prox_lengthscale
        if occupancy_lengthscale is None
        else float(occupancy_lengthscale)
    )
    recency_ls = (
        float(similarity_lengthscale)
        if recency_lengthscale is None
        else float(recency_lengthscale)
    )
    step_ages_arr: np.ndarray | None = (
        None if prior_step_ages is None else np.asarray(prior_step_ages, dtype=int)
    )
    similarity = prior_similarity_to_current(
        X_prev,
        X_current,
        lengthscale=float(similarity_lengthscale),
    )
    mask = similarity >= min_similarity
    if step_ages_arr is not None and len(step_ages_arr) == len(mask):
        step_ages_arr = step_ages_arr[mask]
    X_prev = X_prev.loc[mask]
    y_prev = y_prev[mask]
    similarity = similarity[mask]

    if len(X_prev) == 0:
        return TransferSurrogateResult(
            surrogate=baseline,
            transfer_used_this_round=False,
            transfer_weight_share=0.0,
            transfer_mae_delta=None,
            transfer_bad_rounds=transfer_bad_rounds,
            transfer_disabled=False,
            transfer_disabled_reason=None,
        )

    recency = (
        prior_recency_weights(step_ages_arr, lengthscale=recency_ls)
        if step_ages_arr is not None and len(step_ages_arr) == len(X_prev)
        else np.ones(len(X_prev), dtype=float)
    )
    if prior_placement_X is not None:
        if isinstance(prior_placement_X, pd.DataFrame):
            placement_df = prior_placement_X.copy()
        elif isinstance(prior_placement_X, dict):
            placement_df = pd.DataFrame([prior_placement_X])
        else:
            placement_df = pd.DataFrame(prior_placement_X)
        placement_df = placement_df.reindex(columns=X_current.columns, fill_value=0.0)
    if prior_placement_X is not None and len(placement_df) > 0:
        occupancy = prior_placement_downweight(
            X_prev,
            placement_df,
            lengthscale=occ_lengthscale,
            floor=occupancy_floor,
        )
    else:
        occupancy = prior_proximity_weights(
            X_prev,
            X_prev,
            lengthscale=prox_lengthscale,
            floor=proximity_floor,
        )
    modifiers = recency * occupancy
    if float(np.sum(modifiers)) <= 0.0:
        return TransferSurrogateResult(
            surrogate=baseline,
            transfer_used_this_round=False,
            transfer_weight_share=0.0,
            transfer_mae_delta=None,
            transfer_bad_rounds=transfer_bad_rounds,
            transfer_disabled=False,
            transfer_disabled_reason=None,
        )

    n_current = len(X_current)
    max_transfer_weight = n_current * weight_cap / max(1.0 - weight_cap, 1e-8)
    transfer_weights = similarity * modifiers
    transfer_weights = transfer_weights / max(float(np.sum(transfer_weights)), 1e-8)
    transfer_weights = transfer_weights * max_transfer_weight
    transfer_weight_share = float(
        np.sum(transfer_weights) / (np.sum(transfer_weights) + float(n_current))
    )

    base_mae = float(np.mean(np.abs(baseline.predict(X_current) - y_current)))

    X_train = pd.concat([X_current, X_prev], ignore_index=True)
    y_train = np.concatenate([y_current, y_prev], axis=0)
    sample_weight = np.concatenate(
        [np.ones(n_current, dtype=float), transfer_weights], axis=0
    )
    transfer_model = train_surrogate(
        X_train,
        y_train,
        surrogate=surrogate,
        n_estimators=n_estimators,
        random_state=random_state,
        sample_weight=sample_weight,
    )
    transfer_mae = float(np.mean(np.abs(transfer_model.predict(X_current) - y_current)))
    transfer_mae_delta = transfer_mae - base_mae
    bad_rounds = transfer_bad_rounds
    if transfer_mae_delta > mae_tolerance:
        bad_rounds += 1
    else:
        bad_rounds = 0

    if bad_rounds >= trust_patience:
        return TransferSurrogateResult(
            surrogate=baseline,
            transfer_used_this_round=False,
            transfer_weight_share=transfer_weight_share,
            transfer_mae_delta=transfer_mae_delta,
            transfer_bad_rounds=bad_rounds,
            transfer_disabled=True,
            transfer_disabled_reason="trust_degraded_on_current_step_residuals",
        )

    return TransferSurrogateResult(
        surrogate=transfer_model,
        transfer_used_this_round=True,
        transfer_weight_share=transfer_weight_share,
        transfer_mae_delta=transfer_mae_delta,
        transfer_bad_rounds=bad_rounds,
        transfer_disabled=False,
        transfer_disabled_reason=None,
    )


# ---------------------------------------------------------------------------
# Per-tree prediction with uncertainty
# ---------------------------------------------------------------------------


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, sigma)`` for minimisation.

    Tree ensembles: ``sigma`` is std dev across ``estimators_``. Ridge / HGB:
    ``sigma`` is residual RMSE with mild nearest-neighbour inflation in the
    pipeline's scaled feature space (see :func:`_attach_residual_uncertainty`).
    Plain linear models without attached residual stats still return σ=0; EI/PI
    then rank by ``-mu``.
    """
    regressor = model.named_steps["regressor"]
    if "scaler" in model.named_steps:
        X_eval = model.named_steps["scaler"].transform(X)
    else:
        X_eval = X

    if isinstance(regressor, EnsembleRegressor):
        return regressor.predict_with_uncertainty(X)
    if isinstance(regressor, GaussianProcessRegressor):
        mu, sigma = regressor.predict(X_eval, return_std=True)
        return np.asarray(mu, dtype=float).ravel(), np.asarray(
            sigma, dtype=float
        ).ravel()

    if hasattr(regressor, "estimators_"):
        X_tree = np.asarray(X_eval)
        tree_preds = np.array([t.predict(X_tree) for t in regressor.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
    else:
        mu = np.asarray(regressor.predict(X_eval)).ravel()
        sigma = _sigma_from_residual(regressor, np.asarray(X_eval, dtype=float), mu)

    return mu, sigma


# ---------------------------------------------------------------------------
# Acquisition functions for minimisation (lower E_ads is better)
# ---------------------------------------------------------------------------


def lcb_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 1.96,
) -> np.ndarray:
    """Lower confidence bound for minimisation: mu - kappa * sigma.

    Lower scores are better (more promising candidates).
    """
    return mu - kappa * sigma


def ei_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 1e-6,
) -> np.ndarray:
    """Expected Improvement for minimisation (minimize E_ads).

    EI = E[max(0, f_best - Y)] under Gaussian Y ~ N(mu, sigma^2). Higher EI is better.
    When ``sigma`` is (near) zero, ranks by ``-mu`` so the pool does not collapse
    to an arbitrary tied ordering of zeros.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    z = np.divide(imp, sigma, out=np.zeros_like(imp, dtype=float), where=sigma > 1e-9)
    return np.where(
        sigma > 1e-9,
        imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z),
        -mu,
    )


def pi_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 1e-6,
) -> np.ndarray:
    """Probability of Improvement for minimisation: P(Y < f_best - xi).

    Higher PI is better. When sigma is zero, ranks by ``-mu`` (same rationale as EI).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    z = np.divide(
        f_best - xi - mu,
        sigma,
        out=np.zeros_like(mu, dtype=float),
        where=sigma > 1e-9,
    )
    return np.where(sigma > 1e-9, stats.norm.cdf(z), -mu)


def _farthest_point_indices(
    features: pd.DataFrame | np.ndarray,
    n_pick: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Greedy farthest-point sampling in standardized feature space."""
    matrix = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    n_pool = matrix.shape[0]
    if n_pick >= n_pool:
        return list(range(n_pool))
    scaled = StandardScaler().fit_transform(matrix)
    first = int(rng.randint(n_pool))
    chosen = [first]
    min_dists = np.linalg.norm(scaled - scaled[first], axis=1)
    for _ in range(n_pick - 1):
        min_dists[chosen] = -1.0
        nxt = int(np.argmax(min_dists))
        chosen.append(nxt)
        min_dists = np.minimum(min_dists, np.linalg.norm(scaled - scaled[nxt], axis=1))
    return chosen


def _stratified_conformer_indices(
    features: pd.DataFrame,
    n_pick: int,
    rng: np.random.RandomState,
) -> list[int]:
    """Round-robin across conformer_index groups, then spread-fill remainder."""
    if "conformer_index" not in features.columns:
        return _farthest_point_indices(features, n_pick, rng)

    groups: dict[int, list[int]] = {}
    for i, value in enumerate(features["conformer_index"].astype(int)):
        groups.setdefault(int(value), []).append(i)
    for members in groups.values():
        rng.shuffle(members)
    keys = list(groups.keys())
    rng.shuffle(keys)

    chosen: list[int] = []
    while len(chosen) < n_pick:
        progressed = False
        for key in keys:
            if groups[key]:
                chosen.append(groups[key].pop())
                progressed = True
                if len(chosen) >= n_pick:
                    break
        if not progressed:
            break

    if len(chosen) < n_pick:
        remaining = [i for i in range(len(features)) if i not in chosen]
        if remaining:
            local = _farthest_point_indices(
                features.iloc[remaining],
                min(n_pick - len(chosen), len(remaining)),
                rng,
            )
            chosen.extend(remaining[i] for i in local)
    return chosen


def select_initial_bo_indices(
    candidate_features: pd.DataFrame,
    n_initial: int,
    *,
    sampling: InitialSamplingType = "spread_xyz",
    random_state: int = 0,
) -> list[int]:
    """Pick initial BO pool positions before any energy evaluations.

    Strategies:
    - ``random``: uniform without replacement
    - ``spread``: farthest-point on all geometry-aware features
    - ``spread_xyz``: farthest-point on absolute position (x, y, z) only
    - ``stratified``: round-robin across conformer_index, then spread-fill
    """
    if sampling not in BO_INITIAL_SAMPLING_OPTIONS:
        allowed = ", ".join(repr(item) for item in BO_INITIAL_SAMPLING_OPTIONS)
        raise ValueError(f"sampling must be one of {allowed}, got {sampling!r}")
    n_pool = len(candidate_features)
    n_pick = min(int(n_initial), n_pool)
    if n_pick <= 0:
        return []
    if n_pick >= n_pool:
        return list(range(n_pool))
    rng = np.random.RandomState(random_state)
    if sampling == "random":
        return rng.choice(n_pool, size=n_pick, replace=False).tolist()
    if sampling == "spread":
        return _farthest_point_indices(candidate_features, n_pick, rng)
    if sampling == "spread_xyz":
        position_cols = [c for c in ("x", "y", "z") if c in candidate_features.columns]
        subset = (
            candidate_features[position_cols] if position_cols else candidate_features
        )
        return _farthest_point_indices(subset, n_pick, rng)
    return _stratified_conformer_indices(candidate_features, n_pick, rng)


def select_candidates(
    scores: np.ndarray,
    batch_size: int,
    evaluated_indices: set[int] | None = None,
    *,
    higher_is_better: bool = False,
) -> list[int]:
    """Return up to *batch_size* best candidate indices by rank order.

    For minimisation objectives (default), ranks by ascending *scores*.
    For *higher_is_better* (e.g. EI, PI), ranks by descending *scores*.
    Indices in *evaluated_indices* are excluded.
    """
    s = np.asarray(scores, dtype=float).ravel()
    order = np.argsort(-s) if higher_is_better else np.argsort(s)
    selected: list[int] = []
    for idx in order:
        if evaluated_indices is not None and int(idx) in evaluated_indices:
            continue
        selected.append(int(idx))
        if len(selected) >= batch_size:
            break
    return selected


def select_candidates_batch_diverse(
    scores: np.ndarray,
    features: pd.DataFrame | np.ndarray,
    batch_size: int,
    evaluated_indices: set[int] | None = None,
    *,
    higher_is_better: bool = False,
) -> list[int]:
    """Greedy batch selection with soft local penalization in feature space.

    Picks the best remaining score, then down-weights (or up-penalizes for
    minimisation) candidates near the chosen point so a single batch does not
    collapse onto a tight cluster of near-duplicates.
    """
    s = np.asarray(scores, dtype=float).copy().ravel()
    matrix = (
        features.to_numpy(dtype=float)
        if isinstance(features, pd.DataFrame)
        else np.asarray(features, dtype=float)
    )
    n = len(s)
    if matrix.shape[0] != n:
        raise ValueError(
            f"features rows ({matrix.shape[0]}) must match scores length ({n})"
        )
    blocked: set[int] = set(evaluated_indices or ())
    available = [i for i in range(n) if i not in blocked]
    if not available or batch_size <= 0:
        return []
    if batch_size == 1 or len(available) == 1:
        return select_candidates(
            s,
            batch_size,
            evaluated_indices=blocked,
            higher_is_better=higher_is_better,
        )

    scaled = StandardScaler().fit_transform(matrix)
    # Lengthscale: median NN distance among available points.
    if len(available) >= 2:
        sub = scaled[available]
        d_nn = cdist(sub, sub)
        np.fill_diagonal(d_nn, np.inf)
        lengthscale = float(np.median(d_nn.min(axis=1)))
        lengthscale = max(lengthscale, _RESIDUAL_STD_FLOOR)
    else:
        lengthscale = 1.0
    finite = s[np.isfinite(s)]
    strength = float(np.std(finite)) if finite.size > 1 else 1.0
    strength = max(strength, 1e-3)

    chosen: list[int] = []
    remaining = set(available)
    working = s.copy()
    for _ in range(min(batch_size, len(available))):
        if not remaining:
            break
        cand = np.array(sorted(remaining), dtype=int)
        vals = working[cand]
        pick_local = int(np.argmax(vals) if higher_is_better else np.argmin(vals))
        pick = int(cand[pick_local])
        chosen.append(pick)
        remaining.remove(pick)
        if not remaining:
            break
        rem = np.array(sorted(remaining), dtype=int)
        dists = np.linalg.norm(scaled[rem] - scaled[pick], axis=1)
        near = np.exp(-0.5 * np.square(dists / lengthscale))
        if higher_is_better:
            working[rem] = working[rem] - strength * near
        else:
            working[rem] = working[rem] + strength * near
    return chosen


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def build_candidate_features(
    descriptors: list[PlacementDescriptor],
    molecule: str = "",
    smiles: str = "",
    surface_id: str = "",
    config: AdsorptionConfig | None = None,
) -> pd.DataFrame:
    """Extract feature matrix for a list of PlacementDescriptors."""
    rows = [
        extract_features(
            PlacementRecord.from_descriptor(
                d,
                molecule=molecule,
                smiles=smiles,
                surface_id=surface_id,
                config=config,
            )
        )
        for d in descriptors
    ]
    return pd.DataFrame(rows)


def build_spec_features_geometry_aware(
    specs: list[PlacementSpec],
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    smiles: str | None = None,
    molecule: str = "",
    surface_id: str = "",
    site_context: placement_generators.SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """Extract geometry-aware features from specs via resolved deterministic poses.

    Returns ``(features_df, valid_indices)`` where *valid_indices* maps each
    row in the DataFrame back to the position of the corresponding spec in
    *specs*.  Specs that cannot produce a valid placement are skipped; a single
    INFO line summarizes how many were skipped when that count is positive.

    When *materialization_cache* is provided, successful
    ``(adsorbate, descriptor)`` pairs are stored under ``placement_index`` for
    reuse by evaluate paths.
    """
    rows: list[dict[str, float]] = []
    valid_indices: list[int] = []
    if not conformers:
        raise ValueError(
            "build_spec_features_geometry_aware requires at least one conformer"
        )

    for i, spec in enumerate(specs):
        generated = placement_generators.generate_placement_from_spec(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
        )
        if generated is None:
            logger.debug(
                "Skipping spec placement_index=%d: no valid placement",
                spec.placement_index,
            )
            continue
        adsorbate, descriptor = generated
        if materialization_cache is not None:
            materialization_cache[int(descriptor.placement_index)] = (
                adsorbate.copy(),
                descriptor,
            )
        record = PlacementRecord.from_descriptor(
            descriptor,
            molecule=molecule,
            smiles=smiles or "",
            surface_id=surface_id,
            config=config,
        )
        rows.append(extract_features(record))
        valid_indices.append(i)

    n_skip = len(specs) - len(rows)
    if n_skip > 0:
        logger.info(
            "build_spec_features_geometry_aware: skipped %d/%d specs (no valid placement)",
            n_skip,
            len(specs),
        )
    return pd.DataFrame(rows), valid_indices


def score_and_select(
    model: Pipeline,
    candidate_features: pd.DataFrame,
    batch_size: int,
    kappa: float = 1.96,
    evaluated_indices: set[int] | None = None,
    acquisition: AcquisitionType = "lcb",
    f_best: float | None = None,
) -> list[int]:
    """Score candidates with the given acquisition and select the top batch.

    For ``acquisition="lcb"`` uses LCB (mu - kappa * sigma); lower is better.
    For ``acquisition="ei"`` and ``acquisition="pi"`` uses EI or PI; higher is better.
    ``f_best`` is required for EI and PI (current best observed value for minimisation).
    With near-zero ``sigma`` (unfitted linear models), EI/PI fall back to
    ranking by ``-mu`` so the pool does not collapse to an arbitrary tie.
    Batches use soft local penalization in feature space so picks are diverse.
    """
    mu, sigma = predict_with_uncertainty(model, candidate_features)
    if acquisition == "lcb":
        scores = lcb_scores(mu, sigma, kappa=kappa)
        return select_candidates_batch_diverse(
            scores,
            candidate_features,
            batch_size,
            evaluated_indices=evaluated_indices,
            higher_is_better=False,
        )
    if acquisition in ("ei", "pi"):
        if f_best is None:
            raise ValueError("f_best is required for EI and PI acquisition")
        if acquisition == "ei":
            scores = ei_scores(mu, sigma, f_best=f_best)
        else:
            scores = pi_scores(mu, sigma, f_best=f_best)
        return select_candidates_batch_diverse(
            scores,
            candidate_features,
            batch_size,
            evaluated_indices=evaluated_indices,
            higher_is_better=True,
        )
    raise ValueError(f"Unknown acquisition: {acquisition!r}")

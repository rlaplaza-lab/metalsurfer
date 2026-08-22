"""Surrogate training, uncertainty-aware prediction, and acquisition scoring for BO."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from ase import Atoms
from scipy import stats
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .._numeric_defaults import (
    ACQUISITION_SIGMA_FLOOR,
    ACQUISITION_XI_DEFAULT,
    DEFAULT_SEED,
    RESIDUAL_SIGMA_DISTANCE_TEMPER,
)
from ..config import (
    BO_INITIAL_SAMPLING_OPTIONS,
    BO_TRANSFER_CAPABLE_SURROGATES,
    AdsorptionConfig,
)
from ..models import PlacementDescriptor, PlacementSpec
from ..placement import generators as placement_generators
from ..placement._constants import _DISTANCE_ZERO_EPS
from ..placement.site_context import SiteContext
from .features import extract_features
from .regression import (
    TreeSurrogateKind,
    _build_estimator,
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
TransferCapableSurrogateType = Literal[
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "ensemble",
]
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
    """Characteristic length scale for BO GP: sqrt(number of features).

    Parameters
    ----------
    n_features
        Number of features.
    """
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
        """Instantiate the ensemble regressor.

        Parameters
        ----------
        member_surrogates
            Tuple of surrogate model identifiers.
        n_estimators
            Number of estimators per member.
        random_state
            Random seed for reproducibility.
        """
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
        """Fit each ensemble member.

        Parameters
        ----------
        X
            Feature matrix.
        y
            Target values.
        sample_weight
            Per-sample weights.

        Returns
        -------
        EnsembleRegressor
            Fitted self.
        """
        self.members_ = []
        for spec in self.member_surrogates:
            if spec == "ensemble":
                raise ValueError("EnsembleRegressor cannot nest another ensemble")
            weight = (
                sample_weight
                if spec in ("random_forest", "extra_trees", "ridge", "gradient_boost")
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
        """Predict mean values.

        Parameters
        ----------
        X
            Feature matrix.

        Returns
        -------
        np.ndarray
            Predicted means.
        """
        mu, _ = self.predict_with_uncertainty(X)
        return mu

    def predict_with_uncertainty(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty estimates.

        Parameters
        ----------
        X
            Feature matrix.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (mean, standard deviation) per sample.
        """
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
# Relative floor on sigma, as a fraction of the spread of the observed targets.
# Guards against a surrogate that interpolates its training set reporting
# sigma ~= 0, which collapses EI/PI to exactly zero for every candidate.
_RESIDUAL_STD_RELATIVE_FLOOR = 0.05
_RESIDUAL_OOF_MIN_SAMPLES = 4
_RESIDUAL_OOF_MAX_FOLDS = 5


def _format_residual_std(pipeline: Pipeline) -> str:
    """Render the attached residual std for logging, or "n/a" when skipped."""
    value = getattr(pipeline.named_steps["regressor"], "bo_residual_std_", None)
    return "n/a" if value is None else f"{float(value):.4f}"


def _out_of_fold_residual_std(
    pipeline: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y_arr: np.ndarray,
    *,
    random_state: int,
) -> float | None:
    """Return cross-validated residual RMSE, or None when not estimable.

    Fitted unweighted on clones: this estimates generalisation error, which is
    what sigma should represent.
    """
    n = int(y_arr.size)
    if n < _RESIDUAL_OOF_MIN_SAMPLES:
        return None
    n_splits = min(_RESIDUAL_OOF_MAX_FOLDS, n)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    try:
        oof = cross_val_predict(clone(pipeline), X, y_arr, cv=cv)
    except (ValueError, RuntimeError) as exc:
        logger.debug("Out-of-fold residual estimation failed (%s)", exc)
        return None
    resid = y_arr - np.asarray(oof, dtype=float).ravel()
    if not np.all(np.isfinite(resid)):
        return None
    return float(np.sqrt(np.mean(np.square(resid))))


def _attach_residual_uncertainty(
    pipeline: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    *,
    random_state: int = 42,
) -> None:
    """Store residual RMSE and scaled training features for distance-aware σ.

    Deterministic surrogates (ridge / HGB) have no epistemic σ from the
    estimator itself. Using residual RMSE plus nearest-neighbour distance in
    scaled feature space restores usable EI/PI/LCB without changing the mean
    predictor.

    The RMSE is estimated **out of fold**. An in-sample estimate is close to
    zero for an interpolating learner such as ``HistGradientBoostingRegressor``,
    which drives σ to the floor and makes EI/PI identically zero for every
    candidate -- i.e. silently turns BO into pool-ordered sampling.

    Features are standardised with a scaler stored on the regressor so that the
    nearest-neighbour distance is computed in the same space at predict time,
    and so it is not dominated by whichever feature happens to have the largest
    units (x/y in Ångström vs unit quaternion components).
    """
    regressor = pipeline.named_steps["regressor"]
    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)

    residual_std = _out_of_fold_residual_std(
        pipeline, X, y_arr, random_state=random_state
    )
    if residual_std is None:
        # Too few observations to cross-validate: fall back to the in-sample
        # estimate with a dof correction. Only reached for n < 4.
        resid = y_arr - np.asarray(pipeline.predict(X), dtype=float).ravel()
        n = int(resid.size)
        p = int(X_arr.shape[1]) if X_arr.ndim == 2 else 1
        dof = max(n - p - 1, 1)
        residual_std = float(np.sqrt(np.sum(np.square(resid)) / dof))

    spread = float(np.std(y_arr)) if y_arr.size > 1 else 0.0
    floor = max(_RESIDUAL_STD_FLOOR, _RESIDUAL_STD_RELATIVE_FLOOR * spread)
    regressor.bo_residual_std_ = max(residual_std, floor)

    sigma_scaler = StandardScaler().fit(X_arr)
    regressor.bo_sigma_scaler_ = sigma_scaler
    regressor.bo_X_train_scaled_ = np.asarray(
        sigma_scaler.transform(X_arr), dtype=float
    )
    if len(regressor.bo_X_train_scaled_) >= 2:
        nn = NearestNeighbors(n_neighbors=2).fit(regressor.bo_X_train_scaled_)
        nn_dist, _ = nn.kneighbors(regressor.bo_X_train_scaled_)
        lengthscale = float(np.median(nn_dist[:, 1]))
        regressor.bo_lengthscale_ = max(lengthscale, _RESIDUAL_STD_FLOOR)
    else:
        regressor.bo_lengthscale_ = 1.0


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
    # Evaluate in the same standardised space the training features were stored
    # in; comparing raw candidates against scaled training rows would make the
    # distance term meaningless.
    sigma_scaler = getattr(regressor, "bo_sigma_scaler_", None)
    if sigma_scaler is not None:
        X_e = np.asarray(sigma_scaler.transform(X_e), dtype=float)
    X_train_arr = np.asarray(X_train, dtype=float)
    d = cdist(X_e, X_train_arr).min(axis=1)
    lengthscale = getattr(regressor, "bo_lengthscale_", None)
    if lengthscale is None or not np.isfinite(lengthscale) or lengthscale <= 0:
        if len(X_train_arr) >= 2:
            nn = NearestNeighbors(n_neighbors=2).fit(X_train_arr)
            nn_dist, _ = nn.kneighbors(X_train_arr)
            lengthscale = float(np.median(nn_dist[:, 1]))
            lengthscale = max(lengthscale, _RESIDUAL_STD_FLOOR)
        else:
            lengthscale = 1.0
    else:
        lengthscale = float(lengthscale)
    # Mild distance tempering; cap prevents EI from ignoring the mean.
    sigma = base * (1.0 + RESIDUAL_SIGMA_DISTANCE_TEMPER * (d / lengthscale))
    return np.minimum(sigma, 2.0 * base)


def train_surrogate(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    surrogate: SurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    sample_weight: np.ndarray | None = None,
    attach_uncertainty: bool = True,
    **kwargs: Any,
) -> Pipeline:
    """Fit a surrogate on observed placement data.

    Tree ensembles (``random_forest``, ``extra_trees``) return a single-step
    ``Pipeline`` with a regressor only. ``ridge`` returns a ``scaler`` +
    ``regressor`` pipeline from :func:`regression._build_estimator`;
    ``gradient_boost`` returns a regressor-only pipeline (trees do not need
    feature scaling). Per-sample ``sample_weight`` is supported for tree
    ensembles, ``ridge``, and ``gradient_boost``.

    Parameters
    ----------
    X
        Feature matrix.
    y
        Target values.
    surrogate
        Surrogate model type to train.
    n_estimators
        Number of estimators for tree-based surrogates.
    random_state
        Random seed for reproducibility.
    sample_weight
        Optional per-sample weights.
    attach_uncertainty
        Whether to attach residual uncertainty for deterministic models.
    **kwargs
        Additional keyword arguments passed to the regressor builder.
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
        if attach_uncertainty:
            _attach_residual_uncertainty(pipeline, X, y, random_state=random_state)
        logger.info(
            "Trained ridge surrogate on %d samples (residual_std=%s)",
            len(np.asarray(y)),
            _format_residual_std(pipeline),
        )
        return pipeline
    if surrogate == "gradient_boost":
        # HistGradientBoostingRegressor supports sample_weight (incl. transfer).
        pipeline = _build_estimator(
            "gradient_boost", random_state=random_state, **kwargs
        )
        pipeline.fit(X, y, **_tree_pipeline_fit_kwargs(sample_weight))
        if attach_uncertainty:
            _attach_residual_uncertainty(pipeline, X, y, random_state=random_state)
        logger.info(
            "Trained gradient_boost surrogate on %d samples (residual_std=%s)",
            len(np.asarray(y)),
            _format_residual_std(pipeline),
        )
        return pipeline
    if surrogate == "gaussian_process":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is only supported for tree surrogates, ridge, and "
                f"gradient_boost, not {surrogate!r}"
            )
        n_features = int(X.shape[1])
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


def _min_feature_distances(
    X_prior: pd.DataFrame,
    X_ref: pd.DataFrame,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """Minimum Euclidean distance in standardized pose space from each prior row to X_ref.

    Discrete ``conformer_index`` is excluded from the kernel metric (it remains a
    surrogate training feature). Remaining columns are z-scored on the
    concatenated prior+ref matrix so Å positions and unit quaternions share a
    common scale. Transfer ``similarity_lengthscale`` / proximity lengthscales
    are therefore in standardized units.
    """
    if len(X_prior) == 0 or len(X_ref) == 0:
        return np.array([], dtype=float)
    cols = [c for c in X_ref.columns if c != "conformer_index"]
    p_arr = X_prior.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    r_arr = X_ref.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    combined = np.vstack([p_arr, r_arr])
    if combined.shape[0] >= 2 and combined.shape[1] > 0:
        scaled = StandardScaler().fit_transform(combined)
        p_arr = scaled[: len(p_arr)]
        r_arr = scaled[len(p_arr) :]
    dists: np.ndarray = cdist(p_arr, r_arr)
    if exclude_self:
        dists = np.where(dists <= _DISTANCE_ZERO_EPS, np.inf, dists)
    return np.min(dists, axis=1)


def _align_to_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Reindex ``df`` to ``ref``'s columns, padding missing features with 0.0."""
    return df.reindex(columns=ref.columns, fill_value=0.0)


def prior_similarity_to_current(
    X_prior: pd.DataFrame,
    X_current: pd.DataFrame,
    *,
    lengthscale: float,
) -> np.ndarray:
    """Similarity of each prior row to the nearest current-step placement.

    Distances use standardized pose features (see :func:`_min_feature_distances`);
    ``lengthscale`` is in those standardized units.

    Parameters
    ----------
    X_prior
        Prior feature matrix.
    X_current
        Current feature matrix.
    lengthscale
        Length scale for the exponential similarity kernel.
    """
    min_dist = _min_feature_distances(X_prior, X_current)
    return (
        np.exp(-min_dist / float(lengthscale))
        if len(min_dist)
        else np.array([], dtype=float)
    )


def prior_recency_weights(
    step_ages: np.ndarray | list[int],
    *,
    lengthscale: float,
) -> np.ndarray:
    """Exponential decay for older saturation-step observations (age 0 = most recent).

    Parameters
    ----------
    step_ages
        Ages of prior observations (0 = most recent).
    lengthscale
        Decay length scale.
    """
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
    """Reduce transfer weight for prior rows near executed placement sites.

    Parameters
    ----------
    X_prior
        Prior feature matrix.
    placement_X
        Feature rows of committed (on-slab) placements; all rows are used as
        occupancy anchors.
    lengthscale
        Length scale for the exponential distance kernel.
    floor
        Minimum downweight value.
    """
    if len(X_prior) == 0:
        return np.array([], dtype=float)
    if len(placement_X) == 0:
        return np.ones(len(X_prior), dtype=float)
    min_dist = _min_feature_distances(X_prior, placement_X)
    near = np.exp(-min_dist / float(lengthscale))
    return np.maximum(floor, 1.0 - near)


def prior_proximity_weights(
    X_prior: pd.DataFrame,
    X_anchor: pd.DataFrame,
    *,
    lengthscale: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Downweight prior observations near executed placement sites in feature space.

    Parameters
    ----------
    X_prior
        Prior feature matrix.
    X_anchor
        Anchor feature matrix (e.g. current placements).
    lengthscale
        Length scale for the exponential distance kernel.
    floor
        Minimum weight value.
    """
    if len(X_prior) == 0 or len(X_anchor) == 0:
        return np.array([], dtype=float)
    min_dist = _min_feature_distances(X_prior, X_anchor, exclude_self=True)
    proximity = np.exp(-min_dist / float(lengthscale))
    proximity = np.where(np.isfinite(min_dist), proximity, 1.0)
    return np.maximum(floor, proximity)


def cumulative_refit_training_set(
    X_prior: pd.DataFrame,
    y_prior: np.ndarray,
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    *,
    weight_cap: float,
    proximity_lengthscale: float,
    proximity_floor: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble the cumulative-refit training set as ``(X, y, sample_weight)``.

    Rows are ordered prior-first, then current. Current observations get weight
    1.0; prior observations are proximity-decayed and renormalised so their
    total mass is ``weight_cap`` of the combined mass.

    This returns the features, targets and weights together on purpose. The
    previous API returned only the weight vector, leaving the caller to
    concatenate ``X``/``y`` itself; because both orderings have the same length
    nothing raised when the two disagreed, and the weights were silently applied
    to the wrong rows (prior rows got 1.0 and current observations got the
    decayed prior weights, inverting the ``weight_cap`` guarantee).

    Parameters
    ----------
    X_prior
        Prior-step feature matrix.
    y_prior
        Prior-step target values.
    X_current
        Current-step feature matrix.
    y_current
        Current-step target values.
    weight_cap
        Fraction of total weight allocated to prior observations.
    proximity_lengthscale
        Length scale for proximity-based downweighting.
    proximity_floor
        Minimum proximity weight value.
    """
    if len(X_prior) != len(y_prior):
        raise ValueError(
            f"X_prior/y_prior length mismatch: {len(X_prior)} vs {len(y_prior)}"
        )
    if len(X_current) != len(y_current):
        raise ValueError(
            f"X_current/y_current length mismatch: {len(X_current)} vs {len(y_current)}"
        )

    n_current = len(X_current)
    n_prior = len(X_prior)
    current_weights = np.ones(n_current, dtype=float)
    if n_prior == 0:
        return X_current.reset_index(drop=True), np.asarray(y_current), current_weights

    prox = prior_proximity_weights(
        X_prior,
        X_current,
        lengthscale=proximity_lengthscale,
        floor=proximity_floor,
    )
    total_prox = float(np.sum(prox))
    prior_weights: np.ndarray = np.zeros(n_prior, dtype=float)
    if total_prox > 0.0:
        max_transfer_weight = n_current * weight_cap / max(1.0 - weight_cap, 1e-8)
        prior_weights = np.asarray(
            prox / max(total_prox, 1e-8) * max_transfer_weight, dtype=float
        )

    X_combined = pd.concat([X_prior, X_current], ignore_index=True)
    y_combined = np.concatenate([np.asarray(y_prior), np.asarray(y_current)])
    weights = np.concatenate([prior_weights, current_weights])
    return X_combined, y_combined, weights


_TRANSFER_GATE_MIN_SAMPLES = 4
_TRANSFER_GATE_FOLDS = 3


def _transfer_trust_gate(
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    X_prev: pd.DataFrame,
    y_prev: np.ndarray,
    transfer_weights: np.ndarray,
    *,
    fit_baseline: Callable[[], Any],
    surrogate: SurrogateType,
    n_estimators: int,
    random_state: int,
    mae_tolerance: float = 0.0,
) -> tuple[float, float, Pipeline | None, bool]:
    """Compare baseline vs transfer MAE; fit the full transfer model only if useful.

    Returns ``(base_mae, transfer_mae, transfer_model, out_of_sample)``.
    ``transfer_model`` is ``None`` when out-of-fold MAE already rejects transfer
    (``transfer_mae > base_mae + mae_tolerance``), so the caller can skip the
    full-data fit.

    *fit_baseline* is called lazily only for the small-n path or the in-sample
    exception fallback (OOF gating does not need a full-data baseline).
    """
    n_current = len(X_current)
    baseline: Any | None = None

    def _get_baseline() -> Any:
        nonlocal baseline
        if baseline is None:
            baseline = fit_baseline()
        return baseline

    def _fit_full_transfer() -> Pipeline:
        sample_weight = np.concatenate(
            [np.ones(n_current, dtype=float), transfer_weights], axis=0
        )
        return train_surrogate(
            pd.concat([X_current, X_prev], ignore_index=True),
            np.concatenate([y_current, y_prev], axis=0),
            surrogate=surrogate,
            n_estimators=n_estimators,
            random_state=random_state,
            sample_weight=sample_weight,
        )

    if n_current < _TRANSFER_GATE_MIN_SAMPLES:
        # Tiny current sets make in-sample MAE untrustworthy; skip transfer.
        base = _get_baseline()
        base_mae = float(np.mean(np.abs(base.predict(X_current) - y_current)))
        return base_mae, base_mae, None, False

    n_splits = min(_TRANSFER_GATE_FOLDS, n_current)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_pred = np.empty(n_current, dtype=float)
    transfer_pred = np.empty(n_current, dtype=float)
    try:
        for train_idx, test_idx in cv.split(np.arange(n_current)):
            X_tr = X_current.iloc[train_idx]
            y_tr = y_current[train_idx]
            X_te = X_current.iloc[test_idx]

            base_fold = train_surrogate(
                X_tr,
                y_tr,
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
                attach_uncertainty=False,
            )
            base_pred[test_idx] = np.asarray(base_fold.predict(X_te)).ravel()

            fold_weight = np.concatenate(
                [
                    np.ones(len(train_idx), dtype=float),
                    transfer_weights * (len(train_idx) / max(n_current, 1)),
                ],
                axis=0,
            )
            transfer_fold = train_surrogate(
                pd.concat([X_tr, X_prev], ignore_index=True),
                np.concatenate([y_tr, y_prev], axis=0),
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
                sample_weight=fold_weight,
                attach_uncertainty=False,
            )
            transfer_pred[test_idx] = np.asarray(transfer_fold.predict(X_te)).ravel()
    except (ValueError, RuntimeError) as exc:
        logger.debug(
            "Out-of-fold transfer trust gate failed (%s); using in-sample MAE", exc
        )
        transfer_model = _fit_full_transfer()
        base = _get_baseline()
        base_mae = float(np.mean(np.abs(base.predict(X_current) - y_current)))
        transfer_mae = float(
            np.mean(np.abs(transfer_model.predict(X_current) - y_current))
        )
        return base_mae, transfer_mae, transfer_model, False

    base_mae = float(np.mean(np.abs(base_pred - y_current)))
    transfer_mae = float(np.mean(np.abs(transfer_pred - y_current)))
    # Defer the expensive full-data fit when OOF already rejects transfer.
    if transfer_mae > base_mae + mae_tolerance:
        return base_mae, transfer_mae, None, True
    return base_mae, transfer_mae, _fit_full_transfer(), True


def _no_transfer(baseline: Any, transfer_bad_rounds: int) -> "TransferSurrogateResult":
    return TransferSurrogateResult(
        surrogate=baseline,
        transfer_used_this_round=False,
        transfer_weight_share=0.0,
        transfer_mae_delta=None,
        transfer_bad_rounds=transfer_bad_rounds,
        transfer_disabled=False,
        transfer_disabled_reason=None,
    )


def build_transfer_surrogate(
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    observed_X_prev: pd.DataFrame | list[dict[str, float]],
    observed_y_prev: np.ndarray | list[float],
    *,
    surrogate: TransferCapableSurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    weight_cap: float = 0.35,
    similarity_lengthscale: float = 1.0,
    min_similarity: float = 0.05,
    mae_tolerance: float = 0.0,
    transfer_bad_rounds: int = 0,
    trust_patience: int = 2,
    proximity_lengthscale: float | None = None,
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
    Only surrogates in ``BO_TRANSFER_CAPABLE_SURROGATES`` are accepted.

    Parameters
    ----------
    X_current
        Current-step feature matrix.
    y_current
        Current-step target values.
    observed_X_prev
        Prior-step observed features.
    observed_y_prev
        Prior-step observed targets.
    surrogate
        Surrogate model type to train.
    n_estimators
        Number of estimators for tree-based surrogates.
    random_state
        Random seed for reproducibility.
    weight_cap
        Fraction of total weight allocated to prior observations.
    similarity_lengthscale
        Length scale for feature-space similarity.
    min_similarity
        Minimum similarity threshold for transfer.
    mae_tolerance
        MAE delta tolerance for trust gating.
    transfer_bad_rounds
        Number of consecutive bad transfer rounds so far.
    trust_patience
        Maximum allowed consecutive bad rounds before disabling transfer.
    proximity_lengthscale
        Length scale for proximity weighting (also default for occupancy).
    prior_step_ages
        Ages of prior observations for recency decay.
    recency_lengthscale
        Length scale for recency decay.
    prior_placement_X
        Features of previously executed placements.
    occupancy_lengthscale
        Length scale for occupancy-based downweighting.
    occupancy_floor
        Minimum occupancy weight value.
    """
    if surrogate not in BO_TRANSFER_CAPABLE_SURROGATES:
        raise ValueError(
            "build_transfer_surrogate requires a transfer-capable surrogate "
            f"(one of {BO_TRANSFER_CAPABLE_SURROGATES}); got {surrogate!r}"
        )
    y_current = np.asarray(y_current, dtype=float)
    # Full-data baseline is only needed for early exits, small-n gating, gate
    # exceptions, or when transfer is rejected. Skip the eager fit when the
    # OOF gate can decide without it (n_current >= 4).
    baseline: Any | None = None

    def _fit_baseline() -> Any:
        nonlocal baseline
        if baseline is None:
            baseline = train_surrogate(
                X_current,
                y_current,
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
            )
        return baseline

    if len(observed_X_prev) == 0 or len(observed_y_prev) == 0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

    X_prev = (
        pd.DataFrame(observed_X_prev)
        if not isinstance(observed_X_prev, pd.DataFrame)
        else observed_X_prev.copy()
    )
    y_prev = np.asarray(observed_y_prev, dtype=float)
    _X_prev_raw_columns = set(X_prev.columns)
    X_prev = _align_to_columns(X_prev, X_current)
    _X_current_columns = set(X_current.columns)
    if _X_prev_raw_columns != _X_current_columns:
        logger.warning(
            "Transfer surrogate: prior feature columns {%s} differ from current {%s}; "
            "missing columns zero-padded",
            ", ".join(sorted(_X_prev_raw_columns - _X_current_columns)),
            ", ".join(sorted(_X_current_columns - _X_prev_raw_columns)),
        )
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
    # Use positional indexing for every filtered array so features, targets,
    # and metadata stay aligned. `X_prev` can carry a non-default index (e.g.
    # after reindex/concat), so `.iloc`/`np.asarray` is required instead of
    # `.loc[mask]`, which would align by label and silently desync rows.
    keep = np.flatnonzero(mask)
    X_prev = X_prev.iloc[keep].reset_index(drop=True)
    y_prev = y_prev[keep]
    similarity = similarity[keep]

    if len(X_prev) == 0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

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
        placement_df = _align_to_columns(placement_df, X_current)
    if prior_placement_X is not None and len(placement_df) > 0:
        occupancy = prior_placement_downweight(
            X_prev,
            placement_df,
            lengthscale=occ_lengthscale,
            floor=occupancy_floor,
        )
    else:
        # Invert proximity so clustered priors are down-weighted; do not pass
        # proximity weights through as occupancy (cumulative_refit upweights).
        if len(X_prev) <= 1:
            occupancy = np.ones(len(X_prev), dtype=float)
        else:
            prox = prior_proximity_weights(
                X_prev,
                X_prev,
                lengthscale=occ_lengthscale,
                floor=0.0,
            )
            occupancy = np.maximum(occupancy_floor, 1.0 - prox)
    modifiers = recency * occupancy
    if float(np.sum(modifiers)) <= 0.0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

    n_current = len(X_current)
    max_transfer_weight = n_current * weight_cap / max(1.0 - weight_cap, 1e-8)
    transfer_weights = similarity * modifiers
    transfer_weights = transfer_weights / max(float(np.sum(transfer_weights)), 1e-8)
    transfer_weights = transfer_weights * max_transfer_weight
    transfer_weight_share = float(
        np.sum(transfer_weights) / (np.sum(transfer_weights) + float(n_current))
    )

    base_mae, transfer_mae, transfer_model, gate_out_of_sample = _transfer_trust_gate(
        X_current,
        y_current,
        X_prev,
        y_prev,
        transfer_weights,
        fit_baseline=_fit_baseline,
        surrogate=surrogate,
        n_estimators=n_estimators,
        random_state=random_state,
        mae_tolerance=mae_tolerance,
    )
    transfer_mae_delta = transfer_mae - base_mae
    bad_rounds = transfer_bad_rounds
    if transfer_mae_delta > mae_tolerance:
        bad_rounds += 1
    else:
        bad_rounds = 0

    if bad_rounds >= trust_patience:
        return TransferSurrogateResult(
            surrogate=_fit_baseline(),
            transfer_used_this_round=False,
            transfer_weight_share=transfer_weight_share,
            transfer_mae_delta=transfer_mae_delta,
            transfer_bad_rounds=bad_rounds,
            transfer_disabled=True,
            transfer_disabled_reason=(
                "trust_degraded_on_current_step_residuals"
                if gate_out_of_sample
                else "trust_degraded_on_current_step_residuals_in_sample"
            ),
        )

    # OOF already rejected transfer: skip the full-data fit and use baseline
    # this round while still counting the bad round toward patience.
    if transfer_model is None:
        return TransferSurrogateResult(
            surrogate=_fit_baseline(),
            transfer_used_this_round=False,
            transfer_weight_share=transfer_weight_share,
            transfer_mae_delta=transfer_mae_delta,
            transfer_bad_rounds=bad_rounds,
            transfer_disabled=False,
            transfer_disabled_reason=None,
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

    Parameters
    ----------
    model
        Fitted sklearn Pipeline with a regressor step.
    X
        Feature matrix for prediction.
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
        sigma = _sigma_from_residual(regressor, np.asarray(X, dtype=float), mu)

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

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    kappa
        Exploration-exploitation trade-off parameter.
    """
    return mu - kappa * sigma


def ei_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = ACQUISITION_XI_DEFAULT,
) -> np.ndarray:
    """Compute expected improvement scores for minimisation.

    EI = E[max(0, f_best - Y)] under Gaussian Y ~ N(mu, sigma^2). Higher EI is better.
    When *every* ``sigma`` is (near) zero, ranks by ``-mu`` so the pool does not
    collapse to an arbitrary tied ordering of zeros. In a mixed-``sigma`` pool,
    zero-``sigma`` rows use the analytic limit ``max(f_best - mu - xi, 0)`` so
    they stay on the same scale as finite-``sigma`` EI.

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    f_best
        Best observed function value so far.
    xi
        Small jitter to encourage exploration.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    finite = sigma > ACQUISITION_SIGMA_FLOOR
    if not np.any(finite):
        return -mu
    z = np.divide(
        imp,
        sigma,
        out=np.zeros_like(imp, dtype=float),
        where=finite,
    )
    ei = imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    return np.where(finite, ei, np.maximum(imp, 0.0))


def pi_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = ACQUISITION_XI_DEFAULT,
) -> np.ndarray:
    """Probability of Improvement for minimisation: P(Y < f_best - xi).

    Higher PI is better. When every ``sigma`` is zero, ranks by ``-mu`` (same
    rationale as EI). In a mixed-``sigma`` pool, zero-``sigma`` rows use the
    analytic step ``1`` if ``mu < f_best - xi`` else ``0``.

    Parameters
    ----------
    mu
        Predicted mean values.
    sigma
        Predicted standard deviations.
    f_best
        Best observed function value so far.
    xi
        Small jitter to encourage exploration.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    finite = sigma > ACQUISITION_SIGMA_FLOOR
    if not np.any(finite):
        return -mu
    z = np.divide(
        f_best - xi - mu,
        sigma,
        out=np.zeros_like(mu, dtype=float),
        where=finite,
    )
    pi = stats.norm.cdf(z)
    degenerate = np.where(mu < f_best - xi, 1.0, 0.0)
    return np.clip(np.where(finite, pi, degenerate), 0.0, 1.0)


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
        chosen_set = set(chosen)
        remaining = [i for i in range(len(features)) if i not in chosen_set]
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
    random_state: int = DEFAULT_SEED,
) -> list[int]:
    """Pick initial BO pool positions before any energy evaluations.

    Strategies:
    - ``random``: uniform without replacement
    - ``spread``: farthest-point on all geometry-aware features
    - ``spread_xyz``: farthest-point on absolute position (x, y, z) only
    - ``stratified``: round-robin across conformer_index, then spread-fill

    Parameters
    ----------
    candidate_features
        DataFrame of candidate placement features.
    n_initial
        Number of initial samples to select.
    sampling
        Sampling strategy name.
    random_state
        Random seed for reproducibility.
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

    Parameters
    ----------
    scores
        Acquisition scores for each candidate.
    batch_size
        Number of candidates to select.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    higher_is_better
        If True, rank by descending scores.
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
    scaled_features: np.ndarray | None = None,
) -> list[int]:
    """Greedy batch selection with soft local penalization in feature space.

    Picks the best remaining score, then down-weights (or up-penalizes for
    minimisation) candidates near the chosen point so a single batch does not
    collapse onto a tight cluster of near-duplicates.

    Parameters
    ----------
    scores
        Acquisition scores for each candidate.
    features
        Feature matrix for diversity computation.
    batch_size
        Number of candidates to select.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    higher_is_better
        If True, rank by descending scores.
    scaled_features
        Optional pre-standardized feature matrix (same row order as *features*).
        When provided, skips re-fitting ``StandardScaler`` every acquisition round.
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

    if scaled_features is not None:
        scaled = np.asarray(scaled_features, dtype=float)
        if scaled.shape[0] != n:
            raise ValueError(
                f"scaled_features rows ({scaled.shape[0]}) must match scores length ({n})"
            )
    else:
        scaled = StandardScaler().fit_transform(matrix)
    # Lengthscale from the median nearest-neighbour separation, estimated with a
    # KDTree (k=2) instead of a full N x N cdist matrix — avoids the ~234 MB
    # allocation at the real pool size of 3840 while giving an identical value.
    avail_positions = scaled[available]
    if len(available) >= 2:
        tree = KDTree(avail_positions)
        nn_dist = np.asarray(tree.query(avail_positions, k=2)[0])[:, 1]
        lengthscale = float(np.median(nn_dist))
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
        dists = cdist(scaled[pick : pick + 1], scaled[rem])[0]
        near = np.exp(-0.5 * np.square(dists / lengthscale))
        if higher_is_better:
            working[rem] = working[rem] - strength * near
        else:
            working[rem] = working[rem] + strength * near
    return chosen


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def build_spec_features_geometry_aware(
    specs: list[PlacementSpec],
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    smiles: str | None = None,
    molecule: str = "",
    surface_id: str = "",
    site_context: SiteContext | None = None,
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

    Parameters
    ----------
    specs
        List of placement specifications.
    conformers
        List of conformer structures.
    slab
        Surface slab atoms.
    config
        Adsorption configuration.
    smiles
        Optional SMILES string for the molecule.
    molecule
        Molecule name.
    surface_id
        Surface identifier.
    site_context
        Optional site context for placement.
    slab_for_sites
        Optional alternate slab for site detection.
    materialization_cache
        Optional cache for materialized placements.
    """
    rows: list[dict[str, float]] = []
    valid_indices: list[int] = []
    if not conformers:
        raise ValueError(
            "build_spec_features_geometry_aware requires at least one conformer"
        )

    generated = placement_generators.generate_placements_from_specs(
        specs,
        conformers,
        slab,
        config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
        materialization_cache=materialization_cache,
    )
    for i, (spec, (result, _fail_reason)) in enumerate(
        zip(specs, generated, strict=True)
    ):
        if result is None:
            logger.debug(
                "Skipping spec placement_index=%d: no valid placement",
                spec.placement_index,
            )
            continue
        adsorbate, descriptor = result
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
            "Build_spec_features_geometry_aware: skipped %d/%d specs (no valid placement)",
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
    scaled_features: np.ndarray | None = None,
) -> list[int]:
    """Score candidates with the given acquisition and select the top batch.

    For ``acquisition="lcb"`` uses LCB (mu - kappa * sigma); lower is better.
    For ``acquisition="ei"`` and ``acquisition="pi"`` uses EI or PI; higher is better.
    ``f_best`` is required for EI and PI (current best observed value for minimisation).
    With near-zero ``sigma`` (unfitted linear models), EI/PI fall back to
    ranking by ``-mu`` so the pool does not collapse to an arbitrary tie.
    Batches use soft local penalization in feature space so picks are diverse.

    Parameters
    ----------
    model
        Fitted surrogate pipeline.
    candidate_features
        DataFrame of candidate placement features.
    batch_size
        Number of candidates to select.
    kappa
        Exploration parameter for LCB acquisition.
    evaluated_indices
        Set of already-evaluated indices to exclude.
    acquisition
        Acquisition function type ("lcb", "ei", or "pi").
    f_best
        Best observed value (required for EI and PI).
    scaled_features
        Optional pre-standardized pool matrix for diversity (avoids re-scaling
        every acquisition round).
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
            scaled_features=scaled_features,
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
            scaled_features=scaled_features,
        )
    raise ValueError(f"Unknown acquisition: {acquisition!r}")

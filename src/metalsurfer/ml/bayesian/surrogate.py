"""Surrogate training, ensembles, and uncertainty prediction."""

import logging
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..._numeric_defaults import RESIDUAL_SIGMA_DISTANCE_TEMPER
from ..regression import (
    TreeSurrogateKind,
    _build_estimator,
    tree_regressor_for_bayesian_surrogate,
)
from .types import DEFAULT_ENSEMBLE_MEMBERS, SurrogateType

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
    """Matern GP with marginal-likelihood-tuned length scale.

    Initialized at ``sqrt(n_features)`` in standardized feature space; the
    length scale is optimized within ``(1e-2, 1e2)`` during ``fit``.
    """
    length_scale = matern_length_scale_for_n_features(n_features)
    kernel = ConstantKernel(1.0, constant_value_bounds=(1e-2, 1e2)) * Matern(
        length_scale=length_scale,
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    )
    return GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,
        normalize_y=True,
        random_state=random_state,
        n_restarts_optimizer=2,
    )


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Average several BO surrogates; combine mean and disagreement as uncertainty.

    Each member reports ``sigma`` with different semantics (inter-tree std for
    forests, OOF residual RMSE for ridge/HGB, GP posterior std for
    ``gaussian_process``). :meth:`predict_with_uncertainty` combines them as
    ``sqrt(mean(sigma^2) + var(mu))`` — a heuristic exploration signal, not a
    calibrated predictive variance.
    """

    def __init__(
        self,
        member_surrogates: tuple[str, ...] = DEFAULT_ENSEMBLE_MEMBERS,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1,
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
        n_jobs
            Parallel workers forwarded to tree members and per-tree
            uncertainty prediction (joblib convention).
        """
        self.member_surrogates = member_surrogates
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = int(n_jobs)
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
                    n_jobs=self.n_jobs,
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

        Combines member ``(mu, sigma)`` pairs as
        ``sqrt(mean(sigma^2) + var(mu))``. Member sigmas are not harmonized to
        a single uncertainty semantics; the result is a heuristic, not a
        calibrated predictive variance.

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
            mu_i, sigma_i = predict_with_uncertainty(member, X, n_jobs=self.n_jobs)
            mus.append(np.asarray(mu_i, dtype=float).ravel())
            sigmas.append(np.asarray(sigma_i, dtype=float).ravel())
        mus_arr = np.vstack(mus)
        mu_ens = mus_arr.mean(axis=0)
        sigmas_arr = np.vstack(sigmas)
        aleatoric = np.mean(np.square(sigmas_arr), axis=0)
        epistemic = np.var(mus_arr, axis=0)
        sigma_ens = np.sqrt(np.maximum(aleatoric + epistemic, 0.0))
        return mu_ens, sigma_ens


def _tree_pipeline_fit_kwargs(
    sample_weight: np.ndarray | None,
) -> dict[str, Any]:
    if sample_weight is None:
        return {}
    return {"regressor__sample_weight": np.asarray(sample_weight, dtype=float)}


_RESIDUAL_STD_FLOOR = 1e-3

_RESIDUAL_STD_RELATIVE_FLOOR = 0.05

_RESIDUAL_OOF_MIN_SAMPLES = 4

_RESIDUAL_OOF_MAX_FOLDS = 3


def _median_nn_lengthscale(points: np.ndarray, *, use_kdtree: bool = False) -> float:
    """Median 1-NN separation; ``1.0`` if fewer than two rows."""
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return 1.0
    if use_kdtree:
        query = np.asarray(arr, dtype=np.float64)
        nn_dist = np.asarray(cast(Any, KDTree(query)).query(query, k=2)[0])[:, 1]
    else:
        nn_dist, _ = NearestNeighbors(n_neighbors=2).fit(arr).kneighbors(arr)
        nn_dist = nn_dist[:, 1]
    return max(float(np.median(nn_dist)), _RESIDUAL_STD_FLOOR)


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

    Only invoked by :func:`_attach_residual_uncertainty` when the cheap
    in-sample dof-corrected residual is at or below the sigma floor (the
    interpolating-learner case). Folds are capped at
    ``_RESIDUAL_OOF_MAX_FOLDS`` (3) to bound the extra fits per BO batch; no
    result cache is kept because the training set grows every round.
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

    Residual RMSE is estimated in-sample (dof-corrected) first. Out-of-fold
    cross-validation runs **only** when that in-sample value is at or below the
    sigma floor — the interpolating-learner case (e.g. ``HistGradientBoostingRegressor``)
    where in-sample RMSE ≈ 0 would collapse EI/PI to zero. Well-regularized
    ridge models skip the extra KFold fits.

    Features are standardised with a scaler stored on the regressor so that the
    nearest-neighbour distance is computed in the same space at predict time,
    and so it is not dominated by whichever feature happens to have the largest
    units (x/y in Ångström vs unit quaternion components).
    """
    regressor = pipeline.named_steps["regressor"]
    y_arr = np.asarray(y, dtype=float).ravel()
    X_arr = np.asarray(X, dtype=float)

    spread = float(np.std(y_arr)) if y_arr.size > 1 else 0.0
    floor = max(_RESIDUAL_STD_FLOOR, _RESIDUAL_STD_RELATIVE_FLOOR * spread)

    resid = y_arr - np.asarray(pipeline.predict(X), dtype=float).ravel()
    n = int(resid.size)
    p = int(X_arr.shape[1]) if X_arr.ndim == 2 else 1
    dof = max(n - p - 1, 1)
    in_sample_std = float(np.sqrt(np.sum(np.square(resid)) / dof))

    residual_std = in_sample_std
    if in_sample_std <= floor:
        oof_std = _out_of_fold_residual_std(
            pipeline, X, y_arr, random_state=random_state
        )
        if oof_std is not None:
            residual_std = oof_std

    regressor.bo_residual_std_ = max(residual_std, floor)

    sigma_scaler = StandardScaler().fit(X_arr)
    regressor.bo_sigma_scaler_ = sigma_scaler
    regressor.bo_X_train_scaled_ = np.asarray(
        sigma_scaler.transform(X_arr), dtype=float
    )
    regressor.bo_lengthscale_ = _median_nn_lengthscale(regressor.bo_X_train_scaled_)


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
        lengthscale = _median_nn_lengthscale(X_train_arr)
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
    n_jobs: int = -1,
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
    n_jobs
        CPU workers for tree-ensemble fitting (joblib convention; ignored by
        surrogates without native parallelism).
    """
    if surrogate in ("random_forest", "extra_trees"):
        tree_kind: TreeSurrogateKind = (
            "random_forest" if surrogate == "random_forest" else "extra_trees"
        )
        reg = tree_regressor_for_bayesian_surrogate(
            tree_kind,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
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
        pipeline = _build_estimator("ridge", random_state=random_state)
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
        pipeline = _build_estimator("gradient_boost", random_state=random_state)
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
        fitted_reg = pipeline.named_steps["regressor"]
        fitted_ls = float(fitted_reg.kernel_.get_params()["k2__length_scale"])
        logger.info(
            "Trained gaussian_process surrogate on %d samples "
            "(Matern length_scale=%.4f, init=sqrt(%d)=%.4f)",
            len(np.asarray(y)),
            fitted_ls,
            n_features,
            matern_length_scale_for_n_features(n_features),
        )
        return pipeline
    if surrogate == "ensemble":
        reg = EnsembleRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
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


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame | np.ndarray,
    *,
    n_jobs: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, sigma)`` for minimisation.

    Tree ensembles: ``sigma`` is std dev across ``estimators_``. Ridge / HGB:
    ``sigma`` is residual RMSE with mild nearest-neighbour inflation in the
    pipeline's scaled feature space (see :func:`_attach_residual_uncertainty`).
    GP: ``sigma`` is the posterior standard deviation. For
    :class:`EnsembleRegressor`, member sigmas are combined heuristically (see
    that class). Plain linear models without attached residual stats still
    return σ=0; EI/PI then rank by ``-mu``.

    Parameters
    ----------
    model
        Fitted sklearn Pipeline with a regressor step.
    X
        Feature matrix for prediction.
    n_jobs
        CPU workers for per-tree uncertainty prediction (joblib convention).
        ``None`` falls back to the estimator's own setting.
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
        return (
            np.asarray(mu, dtype=float).ravel(),
            np.asarray(sigma, dtype=float).ravel(),
        )

    if hasattr(regressor, "estimators_"):
        X_tree = np.asarray(X_eval)
        estimators = list(regressor.estimators_)
        # Individual tree predictions are single-threaded, so sklearn's own
        # n_jobs never kicks in on this code path. Tree traversal releases the
        # GIL, making a thread pool sufficient (no X pickling across processes).
        workers = (
            int(n_jobs) if n_jobs is not None else getattr(regressor, "n_jobs", -1)
        )
        if len(estimators) > 1 and workers != 1:
            tree_preds = np.asarray(
                joblib.Parallel(n_jobs=workers, backend="threading")(
                    joblib.delayed(t.predict)(X_tree) for t in estimators
                ),
                dtype=float,
            )
        else:
            tree_preds = np.asarray(
                [t.predict(X_tree) for t in estimators], dtype=float
            )
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
    else:
        mu = np.asarray(regressor.predict(X_eval)).ravel()
        sigma = _sigma_from_residual(regressor, np.asarray(X, dtype=float), mu)

    return mu, sigma

"""Sklearn estimators used by the Bayesian optimisation surrogate."""

from typing import Any, Literal

from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .._numeric_defaults import DEFAULT_SEED

ModelType = Literal["ridge", "gradient_boost"]
TreeSurrogateKind = Literal["random_forest", "extra_trees"]


def tree_regressor_for_bayesian_surrogate(
    kind: TreeSurrogateKind,
    *,
    n_estimators: int,
    random_state: int,
    **kwargs: Any,
) -> RandomForestRegressor | ExtraTreesRegressor:
    """Unscaled tree ensemble for BO surrogates (no ``StandardScaler`` in the pipeline).

    Parameters
    ----------
    kind
        Type of tree regressor to build.
    n_estimators
        Number of estimators in the ensemble.
    random_state
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments passed to the regressor.
    """
    params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "min_samples_leaf": kwargs.get("min_samples_leaf", 2),
        "max_depth": kwargs.get("max_depth"),
        "random_state": random_state,
        "n_jobs": -1,
    }
    if kind == "random_forest":
        return RandomForestRegressor(**params)
    return ExtraTreesRegressor(**params)


def _build_estimator(
    model_type: ModelType,
    random_state: int = DEFAULT_SEED,
    **kwargs: Any,
) -> Pipeline:
    """Build a scikit-learn pipeline for the selected regressor."""
    if model_type == "ridge":
        estimator = Ridge(alpha=kwargs.get("alpha", 1.0), random_state=random_state)
        return Pipeline([("scaler", StandardScaler()), ("regressor", estimator)])
    if model_type == "gradient_boost":
        estimator = HistGradientBoostingRegressor(
            max_iter=kwargs.get("max_iter", 300),
            max_depth=kwargs.get("max_depth", 6),
            learning_rate=kwargs.get("learning_rate", 0.1),
            min_samples_leaf=kwargs.get("min_samples_leaf", 5),
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    return Pipeline([("regressor", estimator)])

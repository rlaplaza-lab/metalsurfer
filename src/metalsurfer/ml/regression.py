"""Sklearn-based regressors: ``train_model``, ``evaluate_model``, ``grouped_cross_validate``.

Model types: ``ridge``, ``random_forest``, ``gradient_boost`` (HistGradientBoosting).
"""

import json
import logging
import os
import pickle
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .._numeric_defaults import DEFAULT_SEED

logger = logging.getLogger(__name__)

ModelType = Literal["ridge", "random_forest", "gradient_boost"]
TreeSurrogateKind = Literal["random_forest", "extra_trees"]


def tree_regressor_for_bayesian_surrogate(
    kind: TreeSurrogateKind,
    *,
    n_estimators: int,
    random_state: int,
    **kwargs: Any,
) -> RandomForestRegressor | ExtraTreesRegressor:
    """Unscaled tree ensemble for BO surrogates (no ``StandardScaler`` in the pipeline)."""
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
    if model_type == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=kwargs.get("n_estimators", 200),
            max_depth=kwargs.get("max_depth"),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "gradient_boost":
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


def train_model(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    model_type: ModelType = "gradient_boost",
    random_state: int = DEFAULT_SEED,
    **kwargs: Any,
) -> Pipeline:
    """Train a regression model on the full dataset; returns fitted scaler + regressor pipeline."""
    pipeline = _build_estimator(model_type, random_state=random_state, **kwargs)
    pipeline.fit(X, y)
    logger.info(
        "Trained %s model on %d samples with %d features",
        model_type,
        len(y),
        X.shape[1] if hasattr(X, "shape") else len(X[0]),
    )
    return pipeline


def evaluate_model(
    model: Pipeline,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics (MAE, RMSE, R², max_error, n_samples) on held-out data."""
    y_pred = model.predict(X)
    y_true = np.asarray(y)
    errors = y_true - y_pred

    metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "max_error": float(np.max(np.abs(errors))),
        "n_samples": len(y_true),
    }
    logger.info(
        "Evaluation: MAE=%.4f RMSE=%.4f R²=%.4f (n=%d)",
        metrics["mae"],
        metrics["rmse"],
        metrics["r2"],
        metrics["n_samples"],
    )
    return metrics


def grouped_cross_validate(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    model_type: ModelType = "gradient_boost",
    n_splits: int = 5,
    random_state: int = DEFAULT_SEED,
    **kwargs: Any,
) -> dict[str, Any]:
    """Grouped K-fold CV (by molecule/surface) to avoid leakage; returns fold_metrics and mean/std MAE, RMSE, R²."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    groups_arr = np.asarray(groups)
    if len(y_arr) < 2:
        raise ValueError("Need at least 2 samples for cross-validation")

    n_unique_groups = len(np.unique(groups_arr))
    if n_unique_groups < 2:
        raise ValueError(
            "Grouped cross-validation requires at least 2 unique groups; "
            f"got {n_unique_groups}"
        )
    actual_splits = min(n_splits, n_unique_groups)
    cv = GroupKFold(n_splits=actual_splits)
    split_iter = cv.split(X_arr, y_arr, groups_arr)

    fold_metrics: list[dict[str, float]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        pipeline = _build_estimator(model_type, random_state=random_state, **kwargs)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["fold"] = fold_idx
        fold_metrics.append(metrics)
        logger.info(
            "Fold %d/%d: MAE=%.4f RMSE=%.4f R²=%.4f (train=%d, test=%d)",
            fold_idx + 1,
            actual_splits,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
            len(train_idx),
            len(test_idx),
        )

    mae_vals = [m["mae"] for m in fold_metrics]
    rmse_vals = [m["rmse"] for m in fold_metrics]
    r2_vals = [m["r2"] for m in fold_metrics]

    result = {
        "fold_metrics": fold_metrics,
        "mean_mae": float(np.mean(mae_vals)),
        "mean_rmse": float(np.mean(rmse_vals)),
        "mean_r2": float(np.mean(r2_vals)),
        "std_mae": float(np.std(mae_vals)),
        "std_rmse": float(np.std(rmse_vals)),
        "std_r2": float(np.std(r2_vals)),
        "n_groups": n_unique_groups,
        "n_samples": len(y_arr),
    }

    logger.info(
        "CV summary (%d folds, %d groups): MAE=%.4f±%.4f  RMSE=%.4f±%.4f  R²=%.4f±%.4f",
        actual_splits,
        n_unique_groups,
        result["mean_mae"],
        result["std_mae"],
        result["mean_rmse"],
        result["std_rmse"],
        result["mean_r2"],
        result["std_r2"],
    )

    return result


def feature_importance(
    model: Pipeline,
    feature_names: list[str],
    X: pd.DataFrame | np.ndarray | None = None,
    y: pd.Series | np.ndarray | None = None,
    top_k: int = 15,
) -> pd.DataFrame:
    """Extract feature importance from a trained model.

    For RandomForest uses built-in ``feature_importances_``; for linear
    models uses coefficient magnitudes; for HistGradientBoosting, pass
    ``X`` and ``y`` so ``permutation_importance`` can be used. Otherwise
    importances are unavailable and an empty DataFrame is returned (with
    a warning).

    Returns a DataFrame sorted by importance (descending).
    """
    regressor = model.named_steps["regressor"]

    importances: np.ndarray | None = None
    if hasattr(regressor, "feature_importances_"):
        importances = np.asarray(regressor.feature_importances_)
    elif hasattr(regressor, "coef_"):
        importances = np.abs(np.asarray(regressor.coef_))
    elif X is not None and y is not None:
        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        importances = perm.importances_mean

    if importances is None:
        logger.warning("Cannot extract feature importances from %s", type(regressor))
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    if top_k > 0:
        df = df.head(top_k)

    return df


def save_model(
    model: Pipeline,
    output_dir: str,
    model_type: str,
    metrics: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
) -> str:
    """Save a trained model and its metadata to disk.

    Returns the path to the saved model file.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "binding_energy_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "model_type": model_type,
        "feature_names": feature_names or [],
    }
    if metrics is not None:
        metadata["metrics"] = metrics

    meta_path = os.path.join(output_dir, "binding_energy_model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Saved model to %s", model_path)
    return model_path


def load_model(model_dir: str) -> tuple[Pipeline, dict[str, Any]]:
    """Load a saved model and its metadata.

    Returns (model, metadata_dict).
    """
    model_path = os.path.join(model_dir, "binding_energy_model.pkl")
    meta_path = os.path.join(model_dir, "binding_energy_model_metadata.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    metadata: dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    logger.info(
        "Loaded %s model from %s", metadata.get("model_type", "unknown"), model_dir
    )
    return model, metadata

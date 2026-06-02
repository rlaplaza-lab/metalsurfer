"""Numeric features from :class:`PlacementRecord` (xyz, quaternion, conformer index) for sklearn."""

import logging

import numpy as np
import pandas as pd

from ..placement.geometry import normalize_quaternion
from .schema import PlacementRecord

logger = logging.getLogger(__name__)


FEATURE_NAMES = [
    "x",
    "y",
    "z",
    "conformer_index",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]


def _as_finite_float(value: float, field_name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed


def extract_features(record: PlacementRecord) -> dict[str, float]:
    """Extract numeric features from a single PlacementRecord.

    Returns a flat dictionary of feature_name -> value.
    """
    features: dict[str, float] = {
        "x": _as_finite_float(record.x_abs, "x_abs"),
        "y": _as_finite_float(record.y_abs, "y_abs"),
        "z": _as_finite_float(record.z_abs, "z_abs"),
        "conformer_index": _as_finite_float(record.conformer_index, "conformer_index"),
    }
    quat = normalize_quaternion(
        np.array(
            [
                float(record.quat_w),
                float(record.quat_x),
                float(record.quat_y),
                float(record.quat_z),
            ],
            dtype=float,
        )
    )
    features["quat_w"] = float(quat[0])
    features["quat_x"] = float(quat[1])
    features["quat_y"] = float(quat[2])
    features["quat_z"] = float(quat[3])

    return features


def extract_features_from_dataset(
    df: pd.DataFrame,
    target_column: str = "energy_adsorption",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a dataset DataFrame.

    Parameters
    ----------
    df : DataFrame
        Dataset loaded via :func:`load_dataset`.
    target_column : str
        Column name for the regression target.

    Returns
    -------
    X : DataFrame
        Feature matrix with named columns.
    y : Series
        Target values.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")

    working = df.copy()
    required_geometry_cols = ("x_abs", "y_abs", "z_abs", "conformer_index")
    missing = [col for col in required_geometry_cols if col not in working.columns]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            "Dataset must contain strict geometric feature columns: "
            f"{missing_csv} missing"
        )
    for col in required_geometry_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    if working[list(required_geometry_cols)].isna().any().any():
        raise ValueError(
            "Dataset contains missing/invalid values in strict geometric columns "
            "(x_abs, y_abs, z_abs, conformer_index)"
        )
    for col, default in (
        ("quat_w", 1.0),
        ("quat_x", 0.0),
        ("quat_y", 0.0),
        ("quat_z", 0.0),
    ):
        if col not in working.columns:
            working[col] = default

    quat_cols = (
        working[["quat_w", "quat_x", "quat_y", "quat_z"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna({"quat_w": 1.0, "quat_x": 0.0, "quat_y": 0.0, "quat_z": 0.0})
    )
    quat_values = quat_cols.to_numpy(dtype=float)
    norms = np.linalg.norm(quat_values, axis=1)
    zero_mask = norms < 1e-12
    norms[zero_mask] = 1.0
    quat_values = quat_values / norms[:, np.newaxis]
    neg_mask = quat_values[:, 0] < 0.0
    quat_values[neg_mask] *= -1.0
    quat_values[zero_mask] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    X = pd.DataFrame(
        {
            "x": working["x_abs"].astype(float),
            "y": working["y_abs"].astype(float),
            "z": working["z_abs"].astype(float),
            "conformer_index": working["conformer_index"].astype(float),
            "quat_w": quat_values[:, 0].astype(float),
            "quat_x": quat_values[:, 1].astype(float),
            "quat_y": quat_values[:, 2].astype(float),
            "quat_z": quat_values[:, 3].astype(float),
        }
    )
    y = df[target_column].copy()

    n_features = X.shape[1]
    logger.info("Extracted %d features from %d records", n_features, len(X))

    return X, y


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names produced by extract_features."""
    return FEATURE_NAMES.copy()

"""Numeric features from :class:`PlacementRecord` (xyz, quaternion, conformer index) for sklearn."""

import logging

import numpy as np
import pandas as pd

from ..placement.geometry import normalize_quaternion, normalize_quaternions
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


def _as_finite_float(value: float | None, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed


def extract_features(record: PlacementRecord) -> dict[str, float]:
    """Extract numeric features from a single PlacementRecord.

    Returns a flat dictionary of feature_name -> value.

    Parameters
    ----------
    record
        Placement record to extract features from.
    """
    features: dict[str, float] = {
        "x": _as_finite_float(record.descriptor.x_abs, "x_abs"),
        "y": _as_finite_float(record.descriptor.y_abs, "y_abs"),
        "z": _as_finite_float(record.descriptor.z_abs, "z_abs"),
        "conformer_index": _as_finite_float(
            record.descriptor.conformer_index, "conformer_index"
        ),
    }
    quat = normalize_quaternion(
        np.array(
            [
                _as_finite_float(record.descriptor.quat_w, "quat_w"),
                _as_finite_float(record.descriptor.quat_x, "quat_x"),
                _as_finite_float(record.descriptor.quat_y, "quat_y"),
                _as_finite_float(record.descriptor.quat_z, "quat_z"),
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
    if len(df) == 0:
        raise ValueError("Dataset is empty; cannot extract training features")

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
    geom_values = working[list(required_geometry_cols)].to_numpy(dtype=float)
    if not np.all(np.isfinite(geom_values)):
        raise ValueError(
            "Dataset contains non-finite values in strict geometric columns "
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
    if not np.all(np.isfinite(quat_values)):
        raise ValueError(
            "Dataset contains non-finite values in quaternion columns "
            "(quat_w, quat_x, quat_y, quat_z)"
        )
    quat_values = normalize_quaternions(quat_values)

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
    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.isna().any() or not np.all(np.isfinite(y.to_numpy(dtype=float))):
        raise ValueError(
            f"Dataset contains missing/non-finite values in target column "
            f"'{target_column}'"
        )

    n_features = X.shape[1]
    logger.info("Extracted %d features from %d records", n_features, len(X))

    return X, y


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names produced by extract_features."""
    return FEATURE_NAMES.copy()

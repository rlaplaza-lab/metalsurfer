"""Numeric features from :class:`PlacementRecord` for sklearn / Bayesian screening.

The feature vector is the initial-pose replay recipe: adsorbate COM
(``x``/``y``/``z`` from ``x_abs``/``y_abs``/``z_abs``), unit quaternion, and
conformer index. Those eight numbers fully describe a molecular initial pose
for regression / BO, independent of how the site was enumerated, and are
enough to rebuild a :class:`~metalsurfer.models.PlacementPose` via
:func:`placement_pose_from_features` then
:func:`~metalsurfer.placement.pose.generate_placement_from_pose`.

Site/orientation provenance (``site_index``, tilts, ``z_fraction``, …) and
variable-length ``fragment_positions`` are not features.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from ..models import PlacementPose
from ..placement.geometry import normalize_quaternion, normalize_quaternions
from .schema import PlacementRecord

logger = logging.getLogger(__name__)

# Sklearn / BO column names. Absolute Cartesian COM is stored on the descriptor
# as ``x_abs``/``y_abs``/``z_abs`` and aliased here for spread_xyz / kernels.
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
FEATURE_ABS_COLUMNS: dict[str, str] = {
    "x": "x_abs",
    "y": "y_abs",
    "z": "z_abs",
}
_QUAT_FEATURE_NAMES = ("quat_w", "quat_x", "quat_y", "quat_z")
# PlacementPose requires z_fraction; Cartesian replay uses z_abs instead.
_REPLAY_Z_FRACTION = 0.5


def _as_finite_float(value: float | None, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return parsed


def _finite_quaternion(
    w: float | None,
    x: float | None,
    y: float | None,
    z: float | None,
) -> np.ndarray:
    """Parse and normalize a quaternion from four components.

    ``None`` or non-finite components raise :class:`ValueError`.
    """
    components = [
        _as_finite_float(w, "quat_w"),
        _as_finite_float(x, "quat_x"),
        _as_finite_float(y, "quat_y"),
        _as_finite_float(z, "quat_z"),
    ]
    return normalize_quaternion(np.array(components, dtype=float))


def extract_features(record: PlacementRecord) -> dict[str, float]:
    """Extract the 8-D initial-pose feature block from a PlacementRecord.

    Parameters
    ----------
    record
        Placement record to extract features from.
    """
    descriptor = record.descriptor
    features: dict[str, float] = {
        name: _as_finite_float(getattr(descriptor, abs_name), abs_name)
        for name, abs_name in FEATURE_ABS_COLUMNS.items()
    }
    features["conformer_index"] = _as_finite_float(
        descriptor.conformer_index, "conformer_index"
    )
    quat = _finite_quaternion(
        descriptor.quat_w,
        descriptor.quat_x,
        descriptor.quat_y,
        descriptor.quat_z,
    )
    for i, name in enumerate(_QUAT_FEATURE_NAMES):
        features[name] = float(quat[i])
    return features


def placement_pose_from_features(
    row: Mapping[str, Any],
    *,
    placement_index: int = 0,
) -> PlacementPose:
    """Rebuild a :class:`~metalsurfer.models.PlacementPose` from FEATURE_NAMES.

    Only the 8-D feature block is required. Provenance fields that are not part
    of the BO vector (site index, tilts, …) are filled with replay placeholders
    (``site_index=-1``, default ``z_fraction``); molecular Cartesian replay uses
    COM + quaternion + conformer.

    Parameters
    ----------
    row
        Mapping with FEATURE_NAMES keys (e.g. ``extract_features`` output).
    placement_index
        Placement index stored on the pose.
    """
    quat = _finite_quaternion(
        row.get("quat_w"),
        row.get("quat_x"),
        row.get("quat_y"),
        row.get("quat_z"),
    )
    return PlacementPose(
        conformer_index=int(
            _as_finite_float(row.get("conformer_index"), "conformer_index")
        ),
        site_index=-1,
        site_type=None,
        placement_index=int(placement_index),
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
        x_abs=_as_finite_float(row.get("x"), "x"),
        y_abs=_as_finite_float(row.get("y"), "y"),
        z_fraction=_REPLAY_Z_FRACTION,
        z_abs=_as_finite_float(row.get("z"), "z"),
    )


def extract_features_from_dataset(
    df: pd.DataFrame,
    target_column: str = "energy_adsorption",
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target vector y from a dataset DataFrame.

    Every FEATURE_NAMES column must be finite (via ``x_abs``/``y_abs``/``z_abs``
    plus quaternion + conformer). Missing or non-finite geometry/quaternion
    values raise; no identity-quaternion fallback is applied.

    Parameters
    ----------
    df : DataFrame
        Dataset loaded via :func:`load_dataset`.
    target_column : str
        Column name for the regression target.

    Returns
    -------
    X : DataFrame
        Feature matrix with named columns (FEATURE_NAMES).
    y : Series
        Target values.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in dataset")
    if len(df) == 0:
        raise ValueError("Dataset is empty; cannot extract training features")

    working = df.copy()
    required_geometry_cols = tuple(FEATURE_ABS_COLUMNS.values()) + ("conformer_index",)
    missing = [col for col in required_geometry_cols if col not in working.columns]
    if missing:
        missing_csv = ", ".join(missing)
        raise ValueError(
            "Dataset must contain strict geometric feature columns: "
            f"{missing_csv} missing"
        )
    missing_quat = [col for col in _QUAT_FEATURE_NAMES if col not in working.columns]
    if missing_quat:
        missing_csv = ", ".join(missing_quat)
        raise ValueError(
            f"Dataset must contain quaternion feature columns: {missing_csv} missing"
        )

    strict_cols = list(required_geometry_cols) + list(_QUAT_FEATURE_NAMES)
    for col in strict_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    if working[strict_cols].isna().any().any():
        raise ValueError(
            "Dataset contains missing/invalid values in strict feature columns "
            f"({', '.join(strict_cols)})"
        )
    values = working[strict_cols].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Dataset contains non-finite values in strict feature columns "
            f"({', '.join(strict_cols)})"
        )

    quat_values = normalize_quaternions(
        working[list(_QUAT_FEATURE_NAMES)].to_numpy(dtype=float)
    )

    X = pd.DataFrame(
        {
            **{
                name: working[abs_name].astype(float)
                for name, abs_name in FEATURE_ABS_COLUMNS.items()
            },
            "conformer_index": working["conformer_index"].astype(float),
            "quat_w": quat_values[:, 0].astype(float),
            "quat_x": quat_values[:, 1].astype(float),
            "quat_y": quat_values[:, 2].astype(float),
            "quat_z": quat_values[:, 3].astype(float),
        },
        columns=FEATURE_NAMES,
    )
    y = pd.to_numeric(df[target_column], errors="coerce")
    if y.isna().any() or not np.all(np.isfinite(y.to_numpy(dtype=float))):
        raise ValueError(
            f"Dataset contains missing/non-finite values in target column "
            f"'{target_column}'"
        )

    logger.info("Extracted %d features from %d records", X.shape[1], len(X))
    return X, y


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names produced by extract_features."""
    return FEATURE_NAMES.copy()

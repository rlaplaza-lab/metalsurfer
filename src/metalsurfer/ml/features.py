"""Feature extraction from PlacementRecords for regression models.

Transforms raw placement descriptors into a numeric feature matrix
suitable for scikit-learn estimators.  Features are grouped into:

1. **Geometric** -- site coordinates, height, z_fraction
2. **Angular** -- sin/cos encodings of tilt, azimuth, and in-plane angles
3. **Categorical** -- one-hot encoded orientation_type, site_type, shape
4. **Identity** -- conformer index and optional surface/molecule embeddings

Angle encoding uses sin/cos pairs to preserve periodicity
(e.g. 0 deg and 360 deg map to the same point).
"""

import logging

import numpy as np
import pandas as pd

from .schema import PlacementRecord

logger = logging.getLogger(__name__)

# Categorical columns and their known levels
_ORIENTATION_TYPES = ["parallel", "EN-down", "vertical", "round"]
_SITE_TYPES = ["atop", "bridge", "hollow", "envelope", "none"]
_SHAPES = ["linear", "flat", "round"]


def _angle_features(deg: float) -> tuple[float, float]:
    """Encode an angle in degrees as (sin, cos) for periodicity."""
    rad = np.radians(deg)
    return float(np.sin(rad)), float(np.cos(rad))


def _one_hot(value: str | None, categories: list[str]) -> list[float]:
    """One-hot encode a categorical value."""
    val = str(value).lower() if value is not None else "none"
    return [1.0 if val == cat.lower() else 0.0 for cat in categories]


def extract_features(record: PlacementRecord) -> dict[str, float]:
    """Extract numeric features from a single PlacementRecord.

    Returns a flat dictionary of feature_name -> value.
    """
    features: dict[str, float] = {}

    # Geometric features
    features["x"] = record.x
    features["y"] = record.y
    features["z"] = record.z
    features["z_fraction"] = record.z_fraction
    features["site_index"] = float(record.site_index)
    features["conformer_index"] = float(record.conformer_index)
    features["face_flip"] = 1.0 if record.face_flip else 0.0

    # Angular features (sin/cos encoding for periodicity)
    sin_tilt, cos_tilt = _angle_features(record.tilt_deg)
    features["tilt_sin"] = sin_tilt
    features["tilt_cos"] = cos_tilt

    sin_az, cos_az = _angle_features(record.azimuth_deg)
    features["azimuth_sin"] = sin_az
    features["azimuth_cos"] = cos_az

    sin_aip, cos_aip = _angle_features(record.azimuth_in_plane_deg)
    features["azimuth_in_plane_sin"] = sin_aip
    features["azimuth_in_plane_cos"] = cos_aip

    # Categorical features (one-hot)
    for i, v in enumerate(_one_hot(record.orientation_type, _ORIENTATION_TYPES)):
        features[f"orient_{_ORIENTATION_TYPES[i]}"] = v

    for i, v in enumerate(_one_hot(record.site_type, _SITE_TYPES)):
        features[f"site_{_SITE_TYPES[i]}"] = v

    for i, v in enumerate(_one_hot(record.shape, _SHAPES)):
        features[f"shape_{_SHAPES[i]}"] = v

    # Derived geometric features
    features["xy_radius"] = float(np.sqrt(record.x**2 + record.y**2))
    features["height_above_surface"] = record.z

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

    feature_rows = []
    for _, row in df.iterrows():
        record = PlacementRecord.from_flat_dict(row.to_dict())
        feature_rows.append(extract_features(record))

    X = pd.DataFrame(feature_rows)
    y = df[target_column].copy()

    n_features = X.shape[1]
    logger.info("Extracted %d features from %d records", n_features, len(X))

    return X, y


def get_feature_names() -> list[str]:
    """Return the ordered list of feature names produced by extract_features."""
    from .schema import ComputationContext

    dummy = PlacementRecord(
        molecule="dummy",
        smiles="C",
        surface_id="test",
        placement_id=0,
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="atop",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        x=0.0,
        y=0.0,
        z=2.5,
        shape="round",
        context=ComputationContext(),
    )
    return list(extract_features(dummy).keys())

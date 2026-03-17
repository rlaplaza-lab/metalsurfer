"""Random Forest surrogate and acquisition functions for Bayesian placement selection.

Provides helpers to:
1. Train a lightweight RF surrogate on observed (features, energy) pairs.
2. Extract per-tree mean and standard deviation for uncertainty.
3. Score unevaluated candidates via LCB, EI, or PI acquisition for *minimisation*
   of binding energy.
4. Select the top-k candidates while excluding already-evaluated specs.
"""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from .features import extract_features
from .schema import ComputationContext, PlacementRecord

AcquisitionType = Literal["lcb", "ei", "pi"]
SurrogateType = Literal["random_forest", "extra_trees", "gradient_boost", "ridge"]

logger = logging.getLogger(__name__)


def _record_from_descriptor(
    descriptor: PlacementDescriptor,
    molecule: str = "",
    smiles: str = "",
    surface_id: str = "",
    config: AdsorptionConfig | None = None,
) -> PlacementRecord:
    """Build a lightweight PlacementRecord from a descriptor (zero energies)."""
    ctx = (
        ComputationContext.from_config(config)
        if config is not None
        else ComputationContext()
    )
    return PlacementRecord(
        molecule=molecule,
        smiles=smiles,
        surface_id=surface_id,
        placement_id=descriptor.placement_index,
        conformer_index=descriptor.conformer_index,
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
        z_fraction=descriptor.z_fraction,
        x=descriptor.x,
        y=descriptor.y,
        z=descriptor.z,
        shape=descriptor.shape,
        slab_indices=descriptor.slab_indices,
        context=ctx,
    )


def _record_from_spec(
    spec: PlacementSpec,
    molecule: str = "",
    smiles: str = "",
    surface_id: str = "",
    config: AdsorptionConfig | None = None,
) -> PlacementRecord:
    """Build a PlacementRecord from a spec with placeholder spatial values."""
    ctx = (
        ComputationContext.from_config(config)
        if config is not None
        else ComputationContext()
    )
    return PlacementRecord(
        molecule=molecule,
        smiles=smiles,
        surface_id=surface_id,
        placement_id=spec.placement_index,
        conformer_index=spec.conformer_index,
        orientation_type=spec.orientation_type,
        face_flip=spec.face_flip,
        en_atom_index=spec.en_atom_index,
        site_index=spec.site_index,
        site_type=spec.site_type,
        tilt_deg=spec.tilt_deg,
        azimuth_deg=spec.azimuth_deg,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
        z_fraction=spec.z_fraction,
        x=0.0,
        y=0.0,
        z=0.0,
        shape="round",
        context=ctx,
    )


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


def train_surrogate(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    surrogate: SurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    **kwargs: Any,
) -> Pipeline:
    """Fit a surrogate on observed placement data.

    Returns a scikit-learn Pipeline (scaler + regressor).
    """
    if surrogate == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            max_depth=kwargs.get("max_depth"),
            random_state=random_state,
            n_jobs=-1,
        )
    elif surrogate == "extra_trees":
        reg = ExtraTreesRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            max_depth=kwargs.get("max_depth"),
            random_state=random_state,
            n_jobs=-1,
        )
    elif surrogate in ("gradient_boost", "ridge"):
        # Reuse regression module pipelines for non-ensemble uncertainty models.
        from .regression import train_model

        model = train_model(
            X,
            y,
            model_type="gradient_boost" if surrogate == "gradient_boost" else "ridge",
            random_state=random_state,
            **kwargs,
        )
        logger.info("Trained %s surrogate on %d samples", surrogate, len(np.asarray(y)))
        return model
    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}")

    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
    pipeline.fit(X, y)
    logger.info(
        "Trained %s surrogate on %d samples (%d trees)",
        surrogate,
        len(np.asarray(y)),
        n_estimators,
    )
    return pipeline


# ---------------------------------------------------------------------------
# Per-tree prediction with uncertainty
# ---------------------------------------------------------------------------


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) predictions from per-tree outputs.

    Works with any Pipeline whose final step has ``estimators_``
    (RandomForest). Falls back to (predict, zeros) for other models.
    """
    regressor = model.named_steps["regressor"]
    scaler = model.named_steps["scaler"]
    X_scaled = scaler.transform(X)

    if hasattr(regressor, "estimators_"):
        tree_preds = np.array([t.predict(X_scaled) for t in regressor.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
    else:
        mu = np.asarray(model.predict(X)).ravel()
        sigma = np.zeros_like(mu)

    return mu, sigma


# ---------------------------------------------------------------------------
# Acquisition functions for minimisation (lower E_ads is better)
# ---------------------------------------------------------------------------


def ucb_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    kappa: float = 1.96,
) -> np.ndarray:
    """Lower-confidence-bound score for minimisation: mu - kappa * sigma.

    Lower scores are better (more promising candidates).
    """
    return mu - kappa * sigma


def ei_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 1e-6,
) -> np.ndarray:
    """Expected Improvement for minimisation.

    EI = E[max(0, f_best - Y)]. Higher EI is better.
    When sigma is zero or very small, returns max(0, f_best - mu).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    z = np.where(sigma > 1e-9, imp / sigma, 0.0)
    ei = np.where(
        sigma > 1e-9,
        imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z),
        np.maximum(0.0, imp + xi),
    )
    return ei


def pi_scores(
    mu: np.ndarray,
    sigma: np.ndarray,
    f_best: float,
    xi: float = 1e-6,
) -> np.ndarray:
    """Probability of Improvement for minimisation: P(Y < f_best - xi).

    Higher PI is better. When sigma is zero, returns 1 if mu < f_best - xi else 0.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    z = np.where(sigma > 1e-9, (f_best - xi - mu) / sigma, 0.0)
    pi = np.where(sigma > 1e-9, stats.norm.cdf(z), (mu < f_best - xi).astype(float))
    return pi


def select_candidates(
    scores: np.ndarray,
    batch_size: int,
    evaluated_indices: set[int] | None = None,
) -> list[int]:
    """Return the *batch_size* indices with the lowest acquisition score.

    Indices listed in *evaluated_indices* are excluded.
    """
    order = np.argsort(scores)
    selected: list[int] = []
    for idx in order:
        if evaluated_indices is not None and int(idx) in evaluated_indices:
            continue
        selected.append(int(idx))
        if len(selected) >= batch_size:
            break
    return selected


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
    rows = []
    for d in descriptors:
        record = _record_from_descriptor(
            d, molecule=molecule, smiles=smiles, surface_id=surface_id, config=config
        )
        rows.append(extract_features(record))
    return pd.DataFrame(rows)


def build_spec_features(
    specs: list[PlacementSpec],
    molecule: str = "",
    smiles: str = "",
    surface_id: str = "",
    config: AdsorptionConfig | None = None,
) -> pd.DataFrame:
    """Extract feature matrix for a list of PlacementSpecs (placeholder xy/z)."""
    rows = []
    for s in specs:
        record = _record_from_spec(
            s, molecule=molecule, smiles=smiles, surface_id=surface_id, config=config
        )
        rows.append(extract_features(record))
    return pd.DataFrame(rows)


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

    Returns indices into *candidate_features* (row positions).
    """
    mu, sigma = predict_with_uncertainty(model, candidate_features)
    if acquisition == "lcb":
        scores = ucb_scores(mu, sigma, kappa=kappa)
        return select_candidates(
            scores, batch_size, evaluated_indices=evaluated_indices
        )
    if acquisition in ("ei", "pi"):
        if f_best is None:
            raise ValueError("f_best is required for EI and PI acquisition")
        if acquisition == "ei":
            scores = ei_scores(mu, sigma, f_best=f_best)
        else:
            scores = pi_scores(mu, sigma, f_best=f_best)
        # For EI/PI higher is better: select_candidates expects lower-is-better, so negate
        return select_candidates(
            -scores, batch_size, evaluated_indices=evaluated_indices
        )
    raise ValueError(f"Unknown acquisition: {acquisition!r}")

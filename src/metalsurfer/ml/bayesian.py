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
from ase import Atoms
from scipy import stats
from sklearn.pipeline import Pipeline

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement import generators as placement_generators
from .features import extract_features
from .regression import train_model, tree_regressor_for_bayesian_surrogate
from .schema import PlacementRecord

AcquisitionType = Literal["lcb", "ei", "pi"]
SurrogateType = Literal["random_forest", "extra_trees", "gradient_boost", "ridge"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


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

    Returns a scikit-learn Pipeline (scaler + regressor).
    """
    if surrogate in ("random_forest", "extra_trees"):
        reg = tree_regressor_for_bayesian_surrogate(
            surrogate,
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs,
        )
    elif surrogate in ("gradient_boost", "ridge"):
        if sample_weight is not None:
            logger.debug(
                "sample_weight ignored for surrogate=%s (tree models only)",
                surrogate,
            )
        return train_model(
            X,
            y,
            model_type="gradient_boost" if surrogate == "gradient_boost" else "ridge",
            random_state=random_state,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown surrogate: {surrogate!r}")

    pipeline = Pipeline([("regressor", reg)])
    fit_kwargs: dict[str, Any] = {}
    if sample_weight is not None:
        fit_kwargs["regressor__sample_weight"] = np.asarray(sample_weight, dtype=float)
    pipeline.fit(X, y, **fit_kwargs)
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
    """Return ``(mean, sigma)`` where ``sigma`` is epistemic spread when available.

    For tree ensembles (``estimators_`` on the regressor), ``sigma`` is the
    standard deviation across trees. For other models (ridge, gradient boosting),
    epistemic uncertainty is not defined here: ``sigma`` is all zeros and only
    the point prediction ``mean`` should be used for BO scoring.
    """
    regressor = model.named_steps["regressor"]
    if "scaler" in model.named_steps:
        X_eval = model.named_steps["scaler"].transform(X)
    else:
        X_eval = X

    if hasattr(regressor, "estimators_"):
        # Tree estimators are trained on array-like internals; predict with ndarray
        # to avoid sklearn feature-name mismatch warnings.
        X_tree = np.asarray(X_eval)
        tree_preds = np.array([t.predict(X_tree) for t in regressor.estimators_])
        mu = tree_preds.mean(axis=0)
        sigma = tree_preds.std(axis=0)
    else:
        mu = np.asarray(regressor.predict(X_eval)).ravel()
        sigma = np.zeros_like(mu)

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
    """Expected Improvement for minimisation.

    EI = E[max(0, f_best - Y)]. Higher EI is better.
    When sigma is zero or very small, returns max(0, f_best - mu).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    z = np.divide(imp, sigma, out=np.zeros_like(imp, dtype=float), where=sigma > 1e-9)
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
    z = np.divide(
        f_best - xi - mu,
        sigma,
        out=np.zeros_like(mu, dtype=float),
        where=sigma > 1e-9,
    )
    pi = np.where(sigma > 1e-9, stats.norm.cdf(z), (mu < f_best - xi).astype(float))
    return pi


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
) -> tuple[pd.DataFrame, list[int]]:
    """Extract geometry-aware features from specs via resolved deterministic poses.

    Returns ``(features_df, valid_indices)`` where *valid_indices* maps each
    row in the DataFrame back to the position of the corresponding spec in
    *specs*.  Specs that cannot produce a valid placement are skipped; a single
    INFO line summarizes how many were skipped when that count is positive.
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
        )
        if generated is None:
            logger.debug(
                "Skipping spec placement_index=%d: no valid placement",
                spec.placement_index,
            )
            continue
        _adsorbate, descriptor = generated
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

    Returns indices into *candidate_features* (row positions).
    """
    mu, sigma = predict_with_uncertainty(model, candidate_features)
    if acquisition == "lcb":
        scores = lcb_scores(mu, sigma, kappa=kappa)
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
        return select_candidates(
            scores,
            batch_size,
            evaluated_indices=evaluated_indices,
            higher_is_better=True,
        )
    raise ValueError(f"Unknown acquisition: {acquisition!r}")

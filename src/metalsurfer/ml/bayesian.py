"""Surrogate training, uncertainty-aware prediction, and acquisition scoring for BO."""

import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from ase import Atoms
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from ..placement import generators as placement_generators
from .features import extract_features
from .regression import (
    TreeSurrogateKind,
    train_model,
    tree_regressor_for_bayesian_surrogate,
)
from .schema import PlacementRecord

AcquisitionType = Literal["lcb", "ei", "pi"]
SurrogateType = Literal[
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "gaussian_process",
    "ensemble",
]
TreeSurrogateType = Literal["random_forest", "extra_trees"]
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
    """Characteristic length scale for BO GP: sqrt(number of features)."""
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
        self.members_ = []
        for spec in self.member_surrogates:
            if spec == "ensemble":
                raise ValueError("EnsembleRegressor cannot nest another ensemble")
            weight = (
                sample_weight
                if spec in ("random_forest", "extra_trees")
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
        mu, _ = self.predict_with_uncertainty(X)
        return mu

    def predict_with_uncertainty(
        self, X: pd.DataFrame | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

    Tree ensembles (``random_forest``, ``extra_trees``) return a single-step
    ``Pipeline`` with a regressor only. ``gradient_boost`` and ``ridge`` return
    a ``scaler`` + ``regressor`` pipeline from :func:`regression.train_model`.
    Per-sample ``sample_weight`` is supported only for the tree ensembles.
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
    if surrogate in ("gradient_boost", "ridge"):
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is only supported for tree surrogates "
                f"(random_forest, extra_trees), not {surrogate!r}"
            )
        return train_model(
            X,
            y,
            model_type="gradient_boost" if surrogate == "gradient_boost" else "ridge",
            random_state=random_state,
            **kwargs,
        )
    if surrogate == "gaussian_process":
        if sample_weight is not None:
            raise ValueError(
                "sample_weight is only supported for tree surrogates "
                f"(random_forest, extra_trees), not {surrogate!r}"
            )
        n_features = int(X.shape[1]) if hasattr(X, "shape") else len(X[0])
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


# ---------------------------------------------------------------------------
# Per-tree prediction with uncertainty
# ---------------------------------------------------------------------------


def predict_with_uncertainty(
    model: Pipeline,
    X: pd.DataFrame | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, sigma)`` for minimisation.

    Tree ensembles: ``sigma`` is std dev across ``estimators_``. Ridge / HGB:
    epistemic uncertainty is not defined; ``sigma`` is all zeros (EI/PI then
    use the deterministic limits in :func:`ei_scores` / :func:`pi_scores`).
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
    """Expected Improvement for minimisation (minimize E_ads).

    EI = E[max(0, f_best - Y)] under Gaussian Y ~ N(mu, sigma^2). Higher EI is better.
    When ``sigma`` is (near) zero, uses ``max(0, f_best - mu)``.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    imp = f_best - mu - xi
    z = np.divide(imp, sigma, out=np.zeros_like(imp, dtype=float), where=sigma > 1e-9)
    return np.where(
        sigma > 1e-9,
        imp * stats.norm.cdf(z) + sigma * stats.norm.pdf(z),
        np.maximum(0.0, imp + xi),
    )


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
    return np.where(sigma > 1e-9, stats.norm.cdf(z), (mu < f_best - xi).astype(float))


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
    slab_for_sites: Atoms | None = None,
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
            slab_for_sites=slab_for_sites,
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
    With zero ``sigma`` (ridge / HGB), EI/PI use the deterministic branch in
    :func:`ei_scores` / :func:`pi_scores`. Returns row indices into *candidate_features*.
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

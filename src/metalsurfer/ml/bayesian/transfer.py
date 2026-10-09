"""Transfer-learning surrogates and prior weighting for BO."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...config import BO_TRANSFER_CAPABLE_SURROGATES
from ...placement._constants import _DISTANCE_ZERO_EPS
from .surrogate import train_surrogate
from .types import SurrogateType, TransferCapableSurrogateType

logger = logging.getLogger(__name__)


def _capped_prior_weights(
    raw: np.ndarray, *, n_current: int, weight_cap: float
) -> np.ndarray:
    """Scale *raw* so prior mass is ``weight_cap`` of prior + ``n_current``."""
    raw_arr = np.asarray(raw, dtype=float)
    total = float(np.sum(raw_arr))
    if total <= 0.0:
        return np.zeros(len(raw_arr), dtype=float)
    scale = n_current * weight_cap / max(1.0 - weight_cap, 1e-8)
    return raw_arr * (scale / total)


@dataclass
class TransferSurrogateResult:
    """Outcome of one BO round's transfer-augmented surrogate training."""

    surrogate: Pipeline
    transfer_used_this_round: bool
    transfer_weight_share: float
    transfer_mae_delta: float | None
    transfer_bad_rounds: int
    transfer_disabled: bool
    transfer_disabled_reason: str | None


def _min_feature_distances(
    X_prior: pd.DataFrame,
    X_ref: pd.DataFrame,
    *,
    exclude_self: bool = False,
) -> np.ndarray:
    """Minimum Euclidean distance in standardized pose space from each prior row to X_ref.

    Discrete ``conformer_index`` is excluded from the kernel metric (it remains a
    surrogate training feature). Remaining columns (Cartesian COM ``x``/``y``/``z``
    and unit quaternion) are z-scored on the concatenated prior+ref matrix so
    Å positions and orientation share a common scale. Transfer
    ``similarity_lengthscale`` / proximity lengthscales are therefore in
    standardized units.
    """
    if len(X_prior) == 0 or len(X_ref) == 0:
        return np.array([], dtype=float)
    cols = [c for c in X_ref.columns if c != "conformer_index"]
    p_arr = X_prior.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    r_arr = X_ref.reindex(columns=cols, fill_value=0.0).to_numpy(dtype=float)
    combined = np.vstack([p_arr, r_arr])
    if combined.shape[0] >= 2 and combined.shape[1] > 0:
        scaled = StandardScaler().fit_transform(combined)
        p_arr = scaled[: len(p_arr)]
        r_arr = scaled[len(p_arr) :]
    dists: np.ndarray = cdist(p_arr, r_arr)
    if exclude_self:
        dists = np.where(dists <= _DISTANCE_ZERO_EPS, np.inf, dists)
    return np.min(dists, axis=1)


def _align_to_columns(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Reindex ``df`` to ``ref``'s columns, padding missing features with 0.0."""
    return df.reindex(columns=ref.columns, fill_value=0.0)


def prior_similarity_to_current(
    X_prior: pd.DataFrame,
    X_current: pd.DataFrame,
    *,
    lengthscale: float,
) -> np.ndarray:
    """Similarity of each prior row to the nearest current-step placement.

    Distances use standardized pose features (see :func:`_min_feature_distances`);
    ``lengthscale`` is in those standardized units.

    Parameters
    ----------
    X_prior
        Prior feature matrix.
    X_current
        Current feature matrix.
    lengthscale
        Length scale for the exponential similarity kernel.
    """
    min_dist = _min_feature_distances(X_prior, X_current)
    return (
        np.exp(-min_dist / float(lengthscale))
        if len(min_dist)
        else np.array([], dtype=float)
    )


def prior_recency_weights(
    step_ages: np.ndarray | list[int],
    *,
    lengthscale: float,
) -> np.ndarray:
    """Exponential decay for older saturation-step observations (age 0 = most recent).

    Parameters
    ----------
    step_ages
        Ages of prior observations (0 = most recent).
    lengthscale
        Decay length scale.
    """
    ages = np.asarray(step_ages, dtype=float)
    if ages.size == 0:
        return np.array([], dtype=float)
    return np.exp(-ages / float(lengthscale))


def prior_proximity_weights(
    X_prior: pd.DataFrame,
    X_anchor: pd.DataFrame,
    *,
    lengthscale: float,
    floor: float = 0.0,
) -> np.ndarray:
    """Downweight prior observations near executed placement sites in feature space.

    Parameters
    ----------
    X_prior
        Prior feature matrix.
    X_anchor
        Anchor feature matrix (e.g. current placements).
    lengthscale
        Length scale for the exponential distance kernel.
    floor
        Minimum weight value.
    """
    if len(X_prior) == 0 or len(X_anchor) == 0:
        return np.array([], dtype=float)
    min_dist = _min_feature_distances(X_prior, X_anchor, exclude_self=True)
    proximity = np.exp(-min_dist / float(lengthscale))
    proximity = np.where(np.isfinite(min_dist), proximity, 1.0)
    return np.maximum(floor, proximity)


def cumulative_refit_training_set(
    X_prior: pd.DataFrame,
    y_prior: np.ndarray,
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    *,
    weight_cap: float,
    proximity_lengthscale: float,
    proximity_floor: float = 0.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Assemble the cumulative-refit training set as ``(X, y, sample_weight)``.

    Rows are ordered prior-first, then current. Current observations get weight
    1.0; prior observations are proximity-to-current, then renormalised so their
    total mass is ``weight_cap`` of the combined mass.

    This returns the features, targets and weights together on purpose. The
    previous API returned only the weight vector, leaving the caller to
    concatenate ``X``/``y`` itself; because both orderings have the same length
    nothing raised when the two disagreed, and the weights were silently applied
    to the wrong rows (prior rows got 1.0 and current observations got the
    decayed prior weights, inverting the ``weight_cap`` guarantee).

    Parameters
    ----------
    X_prior
        Prior-step feature matrix.
    y_prior
        Prior-step target values.
    X_current
        Current-step feature matrix.
    y_current
        Current-step target values.
    weight_cap
        Fraction of total weight allocated to prior observations.
    proximity_lengthscale
        Length scale for proximity-based weighting toward current observations.
    proximity_floor
        Minimum proximity weight value.
    """
    if len(X_prior) != len(y_prior):
        raise ValueError(
            f"X_prior/y_prior length mismatch: {len(X_prior)} vs {len(y_prior)}"
        )
    if len(X_current) != len(y_current):
        raise ValueError(
            f"X_current/y_current length mismatch: {len(X_current)} vs {len(y_current)}"
        )

    n_current = len(X_current)
    n_prior = len(X_prior)
    current_weights = np.ones(n_current, dtype=float)
    if n_prior == 0:
        return X_current.reset_index(drop=True), np.asarray(y_current), current_weights

    prox = prior_proximity_weights(
        X_prior,
        X_current,
        lengthscale=proximity_lengthscale,
        floor=proximity_floor,
    )
    total_mod = float(np.sum(prox))
    prior_weights: np.ndarray = np.zeros(n_prior, dtype=float)
    if total_mod > 0.0:
        prior_weights = _capped_prior_weights(
            prox, n_current=n_current, weight_cap=weight_cap
        )

    X_combined = pd.concat([X_prior, X_current], ignore_index=True)
    y_combined = np.concatenate([np.asarray(y_prior), np.asarray(y_current)])
    weights = np.concatenate([prior_weights, current_weights])
    return X_combined, y_combined, weights


_TRANSFER_GATE_MIN_SAMPLES = 4

_TRANSFER_GATE_FOLDS = 3


def _transfer_trust_gate(
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    X_prev: pd.DataFrame,
    y_prev: np.ndarray,
    transfer_weights: np.ndarray,
    *,
    fit_baseline: Callable[[], Any],
    surrogate: SurrogateType,
    n_estimators: int,
    random_state: int,
    mae_tolerance: float = 0.0,
    n_jobs: int = -1,
) -> tuple[float, float, Pipeline | None, bool]:
    """Compare baseline vs transfer MAE; fit the full transfer model only if useful.

    Returns ``(base_mae, transfer_mae, transfer_model, out_of_sample)``.
    ``transfer_model`` is ``None`` when out-of-fold MAE already rejects transfer
    (``transfer_mae > base_mae + mae_tolerance``), so the caller can skip the
    full-data fit.

    *fit_baseline* is called lazily only for the small-n path or the in-sample
    exception fallback (OOF gating does not need a full-data baseline).
    """
    n_current = len(X_current)
    baseline: Any | None = None

    def _get_baseline() -> Any:
        nonlocal baseline
        if baseline is None:
            baseline = fit_baseline()
        return baseline

    def _fit_full_transfer() -> Pipeline:
        sample_weight = np.concatenate(
            [np.ones(n_current, dtype=float), transfer_weights], axis=0
        )
        return train_surrogate(
            pd.concat([X_current, X_prev], ignore_index=True),
            np.concatenate([y_current, y_prev], axis=0),
            surrogate=surrogate,
            n_estimators=n_estimators,
            random_state=random_state,
            sample_weight=sample_weight,
            n_jobs=n_jobs,
        )

    if n_current < _TRANSFER_GATE_MIN_SAMPLES:
        # Tiny current sets make in-sample MAE untrustworthy; skip transfer.
        base = _get_baseline()
        base_mae = float(np.mean(np.abs(base.predict(X_current) - y_current)))
        return base_mae, base_mae, None, False

    n_splits = min(_TRANSFER_GATE_FOLDS, n_current)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    base_pred = np.empty(n_current, dtype=float)
    transfer_pred = np.empty(n_current, dtype=float)
    try:
        for train_idx, test_idx in cv.split(np.arange(n_current)):
            X_tr = X_current.iloc[train_idx]
            y_tr = y_current[train_idx]
            X_te = X_current.iloc[test_idx]

            base_fold = train_surrogate(
                X_tr,
                y_tr,
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
                attach_uncertainty=False,
                n_jobs=n_jobs,
            )
            base_pred[test_idx] = np.asarray(base_fold.predict(X_te)).ravel()

            fold_weight = np.concatenate(
                [
                    np.ones(len(train_idx), dtype=float),
                    transfer_weights * (len(train_idx) / max(n_current, 1)),
                ],
                axis=0,
            )
            transfer_fold = train_surrogate(
                pd.concat([X_tr, X_prev], ignore_index=True),
                np.concatenate([y_tr, y_prev], axis=0),
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
                sample_weight=fold_weight,
                attach_uncertainty=False,
                n_jobs=n_jobs,
            )
            transfer_pred[test_idx] = np.asarray(transfer_fold.predict(X_te)).ravel()
    except (ValueError, RuntimeError) as exc:
        logger.debug(
            "Out-of-fold transfer trust gate failed (%s); using in-sample MAE", exc
        )
        transfer_model = _fit_full_transfer()
        base = _get_baseline()
        base_mae = float(np.mean(np.abs(base.predict(X_current) - y_current)))
        transfer_mae = float(
            np.mean(np.abs(transfer_model.predict(X_current) - y_current))
        )
        return base_mae, transfer_mae, transfer_model, False

    base_mae = float(np.mean(np.abs(base_pred - y_current)))
    transfer_mae = float(np.mean(np.abs(transfer_pred - y_current)))
    # Defer the expensive full-data fit when OOF already rejects transfer.
    if transfer_mae > base_mae + mae_tolerance:
        return base_mae, transfer_mae, None, True
    return base_mae, transfer_mae, _fit_full_transfer(), True


def _no_transfer(baseline: Any, transfer_bad_rounds: int) -> "TransferSurrogateResult":
    return TransferSurrogateResult(
        surrogate=baseline,
        transfer_used_this_round=False,
        transfer_weight_share=0.0,
        transfer_mae_delta=None,
        transfer_bad_rounds=transfer_bad_rounds,
        transfer_disabled=False,
        transfer_disabled_reason=None,
    )


def build_transfer_surrogate(
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    observed_X_prev: pd.DataFrame | list[dict[str, float]],
    observed_y_prev: np.ndarray | list[float],
    *,
    surrogate: TransferCapableSurrogateType = "random_forest",
    n_estimators: int = 100,
    random_state: int = 42,
    weight_cap: float = 0.35,
    similarity_lengthscale: float = 1.0,
    min_similarity: float = 0.05,
    mae_tolerance: float = 0.0,
    transfer_bad_rounds: int = 0,
    trust_patience: int = 2,
    proximity_lengthscale: float | None = None,
    prior_step_ages: list[int] | None = None,
    recency_lengthscale: float | None = None,
    n_jobs: int = -1,
) -> TransferSurrogateResult:
    """Train baseline or transfer-weighted surrogate with production trust gating.

    Mirrors the transfer block in :func:`workflow.bayesian.process_molecule_bayesian`.
    Only surrogates in ``BO_TRANSFER_CAPABLE_SURROGATES`` are accepted.

    Parameters
    ----------
    X_current
        Current-step feature matrix.
    y_current
        Current-step target values.
    observed_X_prev
        Prior-step observed features.
    observed_y_prev
        Prior-step observed targets.
    surrogate
        Surrogate model type to train.
    n_estimators
        Number of estimators for tree-based surrogates.
    random_state
        Random seed for reproducibility.
    weight_cap
        Fraction of total weight allocated to prior observations.
    similarity_lengthscale
        Length scale for feature-space similarity.
    min_similarity
        Minimum similarity threshold for transfer.
    mae_tolerance
        MAE delta tolerance for trust gating.
    transfer_bad_rounds
        Number of consecutive bad transfer rounds so far.
    trust_patience
        Maximum allowed consecutive bad rounds before disabling transfer.
    proximity_lengthscale
        Unused alias kept for call-site compatibility; similarity drives gating.
    prior_step_ages
        Ages of prior observations for recency decay.
    recency_lengthscale
        Length scale for recency decay.
    n_jobs
        CPU workers forwarded to every surrogate fit (joblib convention).
    """
    if surrogate not in BO_TRANSFER_CAPABLE_SURROGATES:
        raise ValueError(
            "build_transfer_surrogate requires a transfer-capable surrogate "
            f"(one of {BO_TRANSFER_CAPABLE_SURROGATES}); got {surrogate!r}"
        )
    y_current = np.asarray(y_current, dtype=float)
    # Full-data baseline is only needed for early exits, small-n gating, gate
    # exceptions, or when transfer is rejected. Skip the eager fit when the
    # OOF gate can decide without it (n_current >= 4).
    baseline: Any | None = None

    def _fit_baseline() -> Any:
        nonlocal baseline
        if baseline is None:
            baseline = train_surrogate(
                X_current,
                y_current,
                surrogate=surrogate,
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs,
            )
        return baseline

    if len(observed_X_prev) == 0 or len(observed_y_prev) == 0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

    X_prev = (
        pd.DataFrame(observed_X_prev)
        if not isinstance(observed_X_prev, pd.DataFrame)
        else observed_X_prev.copy()
    )
    y_prev = np.asarray(observed_y_prev, dtype=float)
    _X_prev_raw_columns = set(X_prev.columns)
    X_prev = _align_to_columns(X_prev, X_current)
    _X_current_columns = set(X_current.columns)
    if _X_prev_raw_columns != _X_current_columns:
        logger.warning(
            "Transfer surrogate: prior feature columns {%s} differ from current {%s}; "
            "missing columns zero-padded",
            ", ".join(sorted(_X_prev_raw_columns - _X_current_columns)),
            ", ".join(sorted(_X_current_columns - _X_prev_raw_columns)),
        )
    # proximity_lengthscale is accepted for API stability; similarity gates rows.
    _ = proximity_lengthscale
    recency_ls = (
        float(similarity_lengthscale)
        if recency_lengthscale is None
        else float(recency_lengthscale)
    )
    step_ages_arr: np.ndarray | None = (
        None if prior_step_ages is None else np.asarray(prior_step_ages, dtype=int)
    )
    similarity = prior_similarity_to_current(
        X_prev,
        X_current,
        lengthscale=float(similarity_lengthscale),
    )
    mask = similarity >= min_similarity
    if step_ages_arr is not None and len(step_ages_arr) == len(mask):
        step_ages_arr = step_ages_arr[mask]
    # Use positional indexing for every filtered array so features, targets,
    # and metadata stay aligned. `X_prev` can carry a non-default index (e.g.
    # after reindex/concat), so `.iloc`/`np.asarray` is required instead of
    # `.loc[mask]`, which would align by label and silently desync rows.
    keep = np.flatnonzero(mask)
    X_prev = X_prev.iloc[keep].reset_index(drop=True)
    y_prev = y_prev[keep]
    similarity = similarity[keep]

    if len(X_prev) == 0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

    recency = (
        prior_recency_weights(step_ages_arr, lengthscale=recency_ls)
        if step_ages_arr is not None and len(step_ages_arr) == len(X_prev)
        else np.ones(len(X_prev), dtype=float)
    )
    modifiers = recency
    if float(np.sum(modifiers)) <= 0.0:
        return _no_transfer(_fit_baseline(), transfer_bad_rounds)

    n_current = len(X_current)
    transfer_weights = _capped_prior_weights(
        similarity * modifiers, n_current=n_current, weight_cap=weight_cap
    )
    transfer_weight_share = float(
        np.sum(transfer_weights) / (np.sum(transfer_weights) + float(n_current))
    )

    base_mae, transfer_mae, transfer_model, gate_out_of_sample = _transfer_trust_gate(
        X_current,
        y_current,
        X_prev,
        y_prev,
        transfer_weights,
        fit_baseline=_fit_baseline,
        surrogate=surrogate,
        n_estimators=n_estimators,
        random_state=random_state,
        mae_tolerance=mae_tolerance,
        n_jobs=n_jobs,
    )
    transfer_mae_delta = transfer_mae - base_mae
    bad_rounds = transfer_bad_rounds
    if transfer_mae_delta > mae_tolerance:
        bad_rounds += 1
    else:
        bad_rounds = 0

    if bad_rounds >= trust_patience:
        return TransferSurrogateResult(
            surrogate=_fit_baseline(),
            transfer_used_this_round=False,
            transfer_weight_share=transfer_weight_share,
            transfer_mae_delta=transfer_mae_delta,
            transfer_bad_rounds=bad_rounds,
            transfer_disabled=True,
            transfer_disabled_reason=(
                "trust_degraded_on_current_step_residuals"
                if gate_out_of_sample
                else "trust_degraded_on_current_step_residuals_in_sample"
            ),
        )

    # OOF already rejected transfer: skip the full-data fit and use baseline
    # this round while still counting the bad round toward patience.
    if transfer_model is None:
        return TransferSurrogateResult(
            surrogate=_fit_baseline(),
            transfer_used_this_round=False,
            transfer_weight_share=transfer_weight_share,
            transfer_mae_delta=transfer_mae_delta,
            transfer_bad_rounds=bad_rounds,
            transfer_disabled=False,
            transfer_disabled_reason=None,
        )

    return TransferSurrogateResult(
        surrogate=transfer_model,
        transfer_used_this_round=True,
        transfer_weight_share=transfer_weight_share,
        transfer_mae_delta=transfer_mae_delta,
        transfer_bad_rounds=bad_rounds,
        transfer_disabled=False,
        transfer_disabled_reason=None,
    )

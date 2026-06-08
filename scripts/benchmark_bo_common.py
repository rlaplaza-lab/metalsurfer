"""Shared utilities for offline BO benchmark scripts."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from metalsurfer.ml.bayesian import (
    AcquisitionType,
    SurrogateType,
    build_transfer_surrogate,
    score_and_select,
    train_surrogate,
)
from metalsurfer.ml.dataset import enrich_detailed_dataset_geometry
from metalsurfer.ml.features import extract_features_from_dataset
from metalsurfer.models import BOStepMemory

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
TOTAL_BUDGET = 100
INITIAL_RANDOM = 10
DEFAULT_SURFACE_TYPE = "co2_graphene"
DEFAULT_SMILES = "O=C=O"
DEFAULT_BO_SURROGATE: SurrogateType = "random_forest"
DEFAULT_BO_ACQUISITION: AcquisitionType = "lcb"
DEFAULT_BO_KAPPA = 1.0

REQUIRED_INPUT_COLUMNS = (
    "molecule",
    "placement_id",
    "conformer_index",
    "x_abs",
    "y_abs",
    "z_abs",
    "energy_adsorption",
)

DEFAULT_TRANSFER_KWARGS = {
    "weight_cap": 0.35,
    "similarity_lengthscale": 1.0,
    "min_similarity": 0.05,
    "mae_tolerance": 0.0,
    "trust_patience": 2,
    "min_step_observations": 5,
}


@dataclass(frozen=True)
class DefaultBOConfig:
    surrogate: SurrogateType = DEFAULT_BO_SURROGATE
    acquisition: AcquisitionType = DEFAULT_BO_ACQUISITION
    kappa: float = DEFAULT_BO_KAPPA
    initial_random: int = INITIAL_RANDOM
    batch_size: int = BATCH_SIZE
    total_budget: int = TOTAL_BUDGET


DEFAULT_BO_CONFIG = DefaultBOConfig()


def _validate_columns(df: pd.DataFrame, csv_path: str) -> None:
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.\nFile: {csv_path}")


def _assert_single_setup(df: pd.DataFrame, csv_path: str) -> None:
    if "surface_id" in df.columns and df["surface_id"].nunique(dropna=False) > 1:
        vals = sorted({str(v) for v in df["surface_id"].unique()})
        raise ValueError(
            "setup-specific: benchmark expects one surface_id per CSV; "
            f"found {len(vals)}: {vals}. File: {csv_path}"
        )
    if "smiles" in df.columns and df["smiles"].nunique(dropna=False) > 1:
        vals = sorted({str(v) for v in df["smiles"].unique()})
        raise ValueError(
            "setup-specific: benchmark expects one smiles string per CSV; "
            f"found {len(vals)}: {vals}. File: {csv_path}"
        )


def _assert_feature_energy_injective(X: pd.DataFrame, y: pd.Series) -> None:
    cols = list(X.columns)
    keys = X[cols].apply(tuple, axis=1)
    groups: dict[tuple, list[float]] = {}
    for k, val in zip(keys, y.astype(float).values, strict=True):
        groups.setdefault(k, []).append(float(val))
    conflicts: list[str] = []
    for k, energies in groups.items():
        if len({round(e, 8) for e in energies}) > 1:
            conflicts.append(f"{k}: {energies}")
    if conflicts:
        raise ValueError(
            "Example conflicts: duplicate feature rows with different energies — "
            + "; ".join(conflicts[:5])
        )


def _dedupe_features_by_best_energy(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, int]:
    keys = X.apply(tuple, axis=1)
    best_idx: list[int] = []
    n_dropped = 0
    for _key, idx in keys.groupby(keys).groups.items():
        indices = list(idx)
        if len(indices) > 1:
            n_dropped += len(indices) - 1
        best_idx.append(
            int(indices[int(np.argmin(y.iloc[indices].astype(float).values))])
        )
    return (
        X.iloc[best_idx].reset_index(drop=True),
        y.iloc[best_idx].reset_index(drop=True),
        n_dropped,
    )


def _prepare_xy(
    df: pd.DataFrame,
    *,
    csv_path: str,
    dedupe: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    _validate_columns(df, csv_path)
    _assert_single_setup(df, csv_path)
    X, y = extract_features_from_dataset(df, target_column="energy_adsorption")
    if not dedupe:
        return X, y, df
    try:
        _assert_feature_energy_injective(X, y)
    except ValueError:
        X, y, n_dropped = _dedupe_features_by_best_energy(X, y)
        logger.warning(
            "Duplicate feature rows with conflicting energies: deduped %d rows "
            "(kept best energy per key); pool size %d. File: %s",
            n_dropped,
            len(X),
            csv_path,
        )
        _assert_feature_energy_injective(X, y)
    return X, y, df


def _step_molecule_pattern(step: int) -> re.Pattern[str]:
    return re.compile(rf"_step_{step:03d}$", re.IGNORECASE)


def _filter_df_by_step(df: pd.DataFrame, step: int) -> pd.DataFrame:
    if "step" in df.columns:
        return df.loc[df["step"] == step].copy()
    pattern = _step_molecule_pattern(step)
    mask = df["molecule"].astype(str).str.contains(pattern)
    if not mask.any():
        raise ValueError(
            f"No rows for step {step}: need 'step' column or molecule suffix _step_{step:03d}"
        )
    return df.loc[mask].copy()


def list_available_steps(
    data_dir: str, source: Literal["auto", "detailed", "placements"] = "auto"
) -> list[int]:
    """Return sorted step indices present in benchmark CSVs."""
    detailed = os.path.join(data_dir, "adsorption_energies_detailed.csv")
    placements = os.path.join(data_dir, "saturation_placements_detailed.csv")
    if source == "auto":
        if os.path.isfile(placements):
            source = "placements"
        elif os.path.isfile(detailed):
            source = "detailed"
        else:
            raise FileNotFoundError(f"Missing benchmark CSV in {data_dir}")
    path = placements if source == "placements" else detailed
    df = pd.read_csv(path)
    if "step" in df.columns:
        return sorted(int(s) for s in df["step"].unique())
    steps: set[int] = set()
    for name in df["molecule"].astype(str):
        match = re.search(r"_step_(\d+)$", name, re.IGNORECASE)
        if match:
            steps.add(int(match.group(1)))
    if not steps:
        return [1]
    return sorted(steps)


def load_placement_pool(
    data_dir: str,
    *,
    step: int | None = None,
    source: Literal["auto", "detailed", "placements"] = "auto",
    surface_type: str = DEFAULT_SURFACE_TYPE,
    smiles: str = DEFAULT_SMILES,
    dedupe: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load X/y from benchmark CSVs, optionally filtered to one saturation step."""
    detailed_path = os.path.join(data_dir, "adsorption_energies_detailed.csv")
    placements_path = os.path.join(data_dir, "saturation_placements_detailed.csv")

    if source == "auto":
        if step is not None and os.path.isfile(placements_path):
            source = "placements"
        elif os.path.isfile(detailed_path):
            source = "detailed"
        elif os.path.isfile(placements_path):
            source = "placements"
        else:
            raise FileNotFoundError(f"Missing {detailed_path} and {placements_path}")
    else:
        if source == "detailed" and not os.path.isfile(detailed_path):
            raise FileNotFoundError(f"Missing {detailed_path}")
        if source == "placements" and not os.path.isfile(placements_path):
            raise FileNotFoundError(f"Missing {placements_path}")

    csv_path = detailed_path if source == "detailed" else placements_path
    df = pd.read_csv(csv_path)
    if step is not None:
        df = _filter_df_by_step(df, step)
    if source == "detailed":
        df = enrich_detailed_dataset_geometry(df, data_dir=data_dir)
    if "smiles" not in df.columns:
        df["smiles"] = smiles
    if "surface_id" not in df.columns:
        df["surface_id"] = surface_type
    return _prepare_xy(df, csv_path=csv_path, dedupe=dedupe)


def load_and_prepare_data(
    data_dir: str,
    surface_type: str = DEFAULT_SURFACE_TYPE,
    smiles: str = DEFAULT_SMILES,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Backward-compatible loader for full detailed CSV (no step filter)."""
    return load_placement_pool(
        data_dir,
        step=None,
        source="detailed",
        surface_type=surface_type,
        smiles=smiles,
    )


def run_random_search(
    X: pd.DataFrame,
    y: pd.Series,
    initial_random: int,
    batch_size: int,
    total_budget: int,
    seed: int,
) -> tuple[list[float], float]:
    """Pure random evaluation replay with the same budget schedule as BO."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y).ravel()
    evaluated: set[int] = set()
    best_after_eval: list[float] = []
    current_best = np.inf

    n_init = min(initial_random, total_budget, n)
    for i in rng.choice(n, size=n_init, replace=False).tolist():
        evaluated.add(i)
        current_best = min(current_best, float(y_arr[i]))
    total_eval = n_init
    best_after_eval.extend([current_best] * total_eval)

    while total_eval < total_budget:
        remaining = total_budget - total_eval
        batch = min(batch_size, remaining)
        uneval = [i for i in range(n) if i not in evaluated]
        if not uneval or batch <= 0:
            break
        take = min(batch, len(uneval))
        chosen = rng.choice(uneval, size=take, replace=False).tolist()
        for i in chosen:
            evaluated.add(i)
            current_best = min(current_best, float(y_arr[i]))
        total_eval += len(chosen)
        best_after_eval.extend([current_best] * len(chosen))

    return best_after_eval, current_best


def run_offline_bo(
    X: pd.DataFrame,
    y: pd.Series,
    initial_random: int,
    batch_size: int,
    total_budget: int,
    seed: int,
    surrogate: str = "random_forest",
    acquisition: str = "lcb",
    kappa: float = 1.96,
) -> tuple[list[float], float]:
    """One BO run with fixed batch size."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y).ravel()
    evaluated: set[int] = set()
    best_after_eval: list[float] = []
    current_best = np.inf

    n_init = min(initial_random, total_budget, n)
    for i in rng.choice(n, size=n_init, replace=False).tolist():
        evaluated.add(i)
        current_best = min(current_best, float(y_arr[i]))
    total_eval = n_init
    best_after_eval.extend([current_best] * total_eval)

    while total_eval < total_budget:
        remaining = total_budget - total_eval
        batch = min(batch_size, remaining)
        if batch <= 0:
            break

        obs_idx = sorted(evaluated)
        X_train = X.iloc[obs_idx]
        y_train = y_arr[obs_idx]

        if len(obs_idx) < 3:
            uneval = [i for i in range(n) if i not in evaluated]
            take = min(batch, len(uneval))
            chosen = rng.choice(uneval, size=take, replace=False).tolist()
        else:
            model = train_surrogate(
                X_train,
                y_train,
                surrogate=surrogate,  # type: ignore[arg-type]
                n_estimators=100,
                random_state=seed,
            )
            chosen = score_and_select(
                model,
                X,
                batch_size=batch,
                kappa=kappa,
                evaluated_indices=evaluated,
                acquisition=acquisition,  # type: ignore[arg-type]
                f_best=current_best if np.isfinite(current_best) else None,
            )
        for i in chosen:
            evaluated.add(i)
            current_best = min(current_best, float(y_arr[i]))
        total_eval += len(chosen)
        best_after_eval.extend([current_best] * len(chosen))

    return best_after_eval, current_best


def run_offline_bo_with_transfer(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_random: int,
    batch_size: int,
    total_budget: int,
    seed: int,
    prior_memory: BOStepMemory | None = None,
    transfer_enabled: bool = True,
    transfer_kwargs: dict | None = None,
) -> tuple[float, BOStepMemory, dict[str, float]]:
    """One step BO run; optionally reuse prior-step memory with production transfer."""
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y).ravel()
    evaluated: set[int] = set()
    current_best = np.inf
    transfer_kw = {**DEFAULT_TRANSFER_KWARGS, **(transfer_kwargs or {})}

    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    transfer_disabled = False
    transfer_bad_rounds = 0
    transfer_weight_shares: list[float] = []
    transfer_used_rounds = 0

    n_init = min(initial_random, total_budget, n)
    init_idx = rng.choice(n, size=n_init, replace=False).tolist()
    for i in init_idx:
        evaluated.add(i)
        current_best = min(current_best, float(y_arr[i]))
        observed_X_rows.append(X.iloc[i].to_dict())
        observed_y.append(float(y_arr[i]))

    while len(evaluated) < min(total_budget, n):
        remaining = total_budget - len(evaluated)
        batch = min(batch_size, remaining)
        if batch <= 0:
            break

        if len(evaluated) < 3:
            uneval = [i for i in range(n) if i not in evaluated]
            take = min(batch, len(uneval))
            if take <= 0:
                break
            chosen = rng.choice(uneval, size=take, replace=False).tolist()
            model = None
        else:
            X_current = pd.DataFrame(observed_X_rows)
            y_current = np.asarray(observed_y, dtype=float)
            model = None
            can_try = (
                transfer_enabled
                and prior_memory is not None
                and not transfer_disabled
                and len(X_current) >= transfer_kw["min_step_observations"]
                and len(prior_memory.observed_X_rows) > 0
            )
            if can_try:
                assert prior_memory is not None
                transfer_result = build_transfer_surrogate(
                    X_current,
                    y_current,
                    prior_memory.observed_X_rows,
                    prior_memory.observed_y,
                    surrogate="random_forest",
                    n_estimators=100,
                    random_state=seed,
                    weight_cap=transfer_kw["weight_cap"],
                    similarity_lengthscale=transfer_kw["similarity_lengthscale"],
                    min_similarity=transfer_kw["min_similarity"],
                    mae_tolerance=transfer_kw["mae_tolerance"],
                    transfer_bad_rounds=transfer_bad_rounds,
                    trust_patience=transfer_kw["trust_patience"],
                )
                transfer_bad_rounds = transfer_result.transfer_bad_rounds
                if transfer_result.transfer_used_this_round:
                    transfer_used_rounds += 1
                if transfer_result.transfer_disabled:
                    transfer_disabled = True
                if transfer_result.transfer_weight_share > 0:
                    transfer_weight_shares.append(transfer_result.transfer_weight_share)
                model = transfer_result.surrogate
            if model is None:
                model = train_surrogate(
                    X_current,
                    y_current,
                    surrogate="random_forest",
                    n_estimators=100,
                    random_state=seed,
                )
            chosen = score_and_select(
                model,
                X,
                batch_size=batch,
                kappa=DEFAULT_BO_KAPPA,
                evaluated_indices=evaluated,
                acquisition=DEFAULT_BO_ACQUISITION,
                f_best=current_best if np.isfinite(current_best) else None,
            )

        for i in chosen:
            evaluated.add(int(i))
            current_best = min(current_best, float(y_arr[i]))
            observed_X_rows.append(X.iloc[i].to_dict())
            observed_y.append(float(y_arr[i]))

    memory = BOStepMemory(
        observed_X_rows=observed_X_rows,
        observed_y=observed_y,
        best_energy=current_best if np.isfinite(current_best) else None,
    )
    info = {
        "transfer_used_fraction": float(
            transfer_used_rounds / max(1, total_budget // batch_size)
        ),
        "transfer_weight_share_mean": float(np.mean(transfer_weight_shares))
        if transfer_weight_shares
        else 0.0,
    }
    return current_best, memory, info


def paired_stats(baseline: list[float], treatment: list[float]) -> dict[str, float]:
    """Paired comparison (lower is better for E_ads)."""
    if len(baseline) != len(treatment) or not baseline:
        return {
            "mean_improvement": float("nan"),
            "win_rate": float("nan"),
            "p_value": float("nan"),
        }
    diffs = np.asarray(baseline, dtype=float) - np.asarray(treatment, dtype=float)
    win_rate = float(np.mean(diffs > 0))
    t_result = stats.ttest_rel(baseline, treatment)
    return {
        "mean_improvement": float(np.mean(diffs)),
        "win_rate": win_rate,
        "p_value": float(t_result.pvalue),
    }


def aggregate_curves(
    curves: list[list[float]], eval_points: list[int] | None = None
) -> pd.DataFrame:
    """Mean/std best-so-far at selected evaluation counts."""
    if eval_points is None:
        eval_points = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    rows: list[dict[str, float | int]] = []
    for ep in eval_points:
        vals = [float(c[ep - 1]) for c in curves if len(c) >= ep]
        if not vals:
            continue
        rows.append(
            {
                "eval_count": ep,
                "mean_best": float(np.mean(vals)),
                "std_best": float(np.std(vals)),
                "n_seeds": len(vals),
            }
        )
    return pd.DataFrame(rows)

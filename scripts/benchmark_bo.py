#!/usr/bin/env python3
"""Offline BO benchmark: replay production settings on placement pools, plot, report.

Single entry point for the bipyridine/Au(111) benchmark workflow:
  - Step 1: default BO vs random
  - Step 1 sweeps: surrogate, batch size, acquisition, initial sampling
  - Steps 2+: baseline BO vs transfer BO (production weighted transfer)
  - Publication figures and CSV artifacts

``bo_total_budget`` in :class:`~metalsurfer.AdsorptionConfig` counts acquisition
batches (default 18), not total evaluations. Offline replay uses
:func:`~metalsurfer.config.resolved_bo_eval_budget` and
:func:`~metalsurfer.config.bo_eval_schedule` so random and BO perform the same
number of pool lookups.

Usage:
  python scripts/benchmark_bo.py --report-dir bo_benchmark_report/
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from metalsurfer import configure_logging
from metalsurfer.config import (
    BO_INITIAL_SAMPLING_OPTIONS,
    AdsorptionConfig,
    bo_eval_schedule,
    resolved_bo_eval_budget,
)
from metalsurfer.ml.bayesian import (
    build_transfer_surrogate,
    score_and_select,
    select_initial_bo_indices,
    train_surrogate,
)
from metalsurfer.ml.features import extract_features_from_dataset, get_feature_names
from metalsurfer.models import BOStepMemory, windowed_bo_step_memories

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "examples/results_bipyridine_au111_defects_saturation_raw"
DEFAULT_SEEDS = 10
DEFAULT_REPORT_DIR = "bo_benchmark_report"

REQUIRED_COLUMNS = (
    "step",
    "molecule",
    "placement_id",
    "conformer_index",
    "x_abs",
    "y_abs",
    "z_abs",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "energy_adsorption",
)
_GEOMETRY_COLUMNS = (
    "conformer_index",
    "x_abs",
    "y_abs",
    "z_abs",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
)

ARCHITECTURES = (
    "random_forest",
    "extra_trees",
    "ridge",
    "gradient_boost",
    "gaussian_process",
    "ensemble",
)
ACQUISITIONS = ("lcb", "ei", "pi")
BATCH_SIZES = (5, 10, 20)
INIT_STRATEGIES = BO_INITIAL_SAMPLING_OPTIONS
INIT_LABELS: dict[str, str] = {
    "random": "Random",
    "spread": "Spread (full)",
    "spread_xyz": "Spread (xyz)",
    "stratified": "Stratified conformer",
}

EREL_YLABEL = r"$E_{\mathrm{rel}}$ (eV)"
Y_TICK_STEP = 0.2
STEP1_FIG_Y_MAX = 1.2

SERIES_CATALOG: dict[str, dict[str, Any]] = {
    "bo": {
        "label": "Default BO",
        "color": "#2166ac",
        "linewidth": 2.0,
        "marker": "o",
        "markersize": 3.5,
        "zorder": 4,
    },
    "random": {
        "label": "Random",
        "color": "#969696",
        "linewidth": 1.5,
        "marker": "s",
        "markersize": 3.0,
        "zorder": 2,
    },
    "transfer": {
        "label": "Default Transfer BO",
        "color": "#b2182b",
        "linewidth": 2.0,
        "marker": "D",
        "markersize": 3.0,
        "zorder": 5,
    },
}
SWEEP_PALETTE = (
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
    "#66c2a5",
    "#fc8d62",
)
_KEY_TO_SEMANTIC = {
    "random_search": "random",
    "random": "random",
    "baseline": "bo",
    "transfer": "transfer",
}

STALE_ARTIFACT_GLOBS = (
    "fig*_*.png",
    "transfer_step*_saturation.png",
    "screening*.csv",
    "transfer*.csv",
    "step1_*.csv",
    "report.md",
    "models*.csv",
    "ablation_*.csv",
)


def _cfg() -> AdsorptionConfig:
    return AdsorptionConfig()


def _replay_cfg() -> AdsorptionConfig:
    """Fixed offline-replay budget (10 init + 18 batches of 5 = 100 evals)."""
    return AdsorptionConfig(
        bo_initial_random=10,
        bo_batch_size=5,
        bo_total_budget=18,
    )


_REPLAY = _replay_cfg()


def _replay_cfg_for_init(sampling: str) -> AdsorptionConfig:
    """Replay budget with a specific initial-placement sampling strategy."""
    return AdsorptionConfig(
        bo_initial_random=_REPLAY.bo_initial_random,
        bo_batch_size=_REPLAY.bo_batch_size,
        bo_total_budget=_REPLAY.bo_total_budget,
        bo_initial_sampling=sampling,  # type: ignore[arg-type]
    )


def _replay_cfg_for_batch(batch_size: int) -> AdsorptionConfig:
    """Match default replay eval count while varying acquisition batch size."""
    init = int(_REPLAY.bo_initial_random)
    remaining = max(0, resolved_bo_eval_budget(_REPLAY) - init)
    return AdsorptionConfig(
        bo_initial_random=init,
        bo_batch_size=batch_size,
        bo_total_budget=remaining // batch_size,
    )


_CFG = _cfg()

SURROGATE = _CFG.bo_surrogate  # type: ignore[assignment]
ACQUISITION = _CFG.bo_acquisition  # type: ignore[assignment]
KAPPA = float(_CFG.bo_ucb_kappa)
EVAL_BUDGET = resolved_bo_eval_budget(_REPLAY)
TRANSFER_KWARGS = {
    "weight_cap": _CFG.bo_transfer_weight_cap,
    "similarity_lengthscale": _CFG.bo_transfer_similarity_lengthscale,
    "min_similarity": _CFG.bo_transfer_min_similarity,
    "mae_tolerance": _CFG.bo_transfer_mae_tolerance,
    "trust_patience": _CFG.bo_transfer_trust_patience,
    "min_step_observations": _CFG.bo_transfer_min_step_observations,
    "exploration_fraction": _CFG.bo_transfer_exploration_fraction,
    "proximity_lengthscale": _CFG.bo_transfer_proximity_lengthscale,
    "proximity_floor": _CFG.bo_transfer_proximity_floor,
    "recency_lengthscale": _CFG.bo_transfer_recency_lengthscale,
    "occupancy_lengthscale": _CFG.bo_transfer_occupancy_lengthscale,
    "occupancy_floor": _CFG.bo_transfer_occupancy_floor,
}
TRANSFER_WINDOW = _CFG.bo_transfer_prior_step_window


class SearchRunResult(NamedTuple):
    curve: list[float]
    final_best: float
    selected_energies: list[float]
    memory: BOStepMemory | None = None
    transfer_weight_share_mean: float = 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "legend.frameon": False,
        }
    )


def _config_key(surrogate: str, acquisition: str, kappa: float) -> str:
    suffix = f"_k{kappa}" if acquisition == "lcb" else ""
    return f"{surrogate}_{acquisition}{suffix}"


def _default_key() -> str:
    return _config_key(SURROGATE, ACQUISITION, KAPPA)


def _semantic_key(key: str, *, default_keys: frozenset[str] | None) -> str:
    if default_keys and key in default_keys:
        return "bo"
    return _KEY_TO_SEMANTIC.get(key, key)


def _series_style(
    key: str,
    *,
    default_keys: frozenset[str] | None,
    sweep_index: int = 0,
) -> dict[str, Any]:
    semantic = _semantic_key(key, default_keys=default_keys)
    if semantic in SERIES_CATALOG:
        return dict(SERIES_CATALOG[semantic])
    return {
        "label": key.replace("_", " "),
        "color": SWEEP_PALETTE[sweep_index % len(SWEEP_PALETTE)],
        "linewidth": 1.5,
        "marker": "o",
        "markersize": 3.0,
        "zorder": 3,
    }


def _y_top(ymax: float, y_max: float | None = None) -> float:
    if y_max is not None:
        return y_max
    raw = ymax if ymax > 0 else Y_TICK_STEP
    return max(Y_TICK_STEP, np.ceil(raw / Y_TICK_STEP) * Y_TICK_STEP)


def _plot_curves(
    curves_df: pd.DataFrame,
    oracle: float,
    series_order: list[str],
    *,
    out_path: str,
    series_col: str = "config",
    default_keys: frozenset[str] | None = None,
    label_overrides: dict[str, str] | None = None,
    x_max: int | None = None,
    y_max: float | None = None,
) -> None:
    if curves_df.empty:
        return
    rel = curves_df.copy()
    rel["mean_rel"] = rel["mean_best"] - oracle
    rel["std_rel"] = rel["std_best"]
    rel["_series"] = rel[series_col]
    if x_max is None:
        x_max = int(rel["eval_count"].max())

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ymax = 0.0
    sweep_idx = 0
    for key in series_order:
        grp = rel[rel["_series"] == key].sort_values("eval_count")
        if grp.empty:
            continue
        if (
            key not in (default_keys or ())
            and _semantic_key(key, default_keys=default_keys) not in SERIES_CATALOG
        ):
            style = _series_style(key, default_keys=default_keys, sweep_index=sweep_idx)
            sweep_idx += 1
        else:
            style = _series_style(key, default_keys=default_keys)
        if label_overrides and key in label_overrides:
            style["label"] = label_overrides[key]

        x = grp["eval_count"].to_numpy()
        y = grp["mean_rel"].to_numpy()
        y_lo = (grp["mean_rel"] - grp["std_rel"]).to_numpy()
        y_hi = (grp["mean_rel"] + grp["std_rel"]).to_numpy()
        ymax = max(ymax, float(np.nanmax(y_hi)))
        ax.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            marker=style["marker"],
            markersize=style["markersize"],
            zorder=style["zorder"],
            drawstyle="steps-post",
        )
        ax.fill_between(
            x, y_lo, y_hi, color=style["color"], alpha=0.12, step="post", linewidth=0
        )

    ax.set_xlabel("Evaluations")
    ax.set_ylabel(EREL_YLABEL)
    ax.set_xlim(0, x_max)
    tick = 20 if x_max > 50 else 10
    ax.set_xticks(list(range(0, x_max + 1, tick)))
    y_top = _y_top(ymax, y_max)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks(np.arange(0.0, y_top + Y_TICK_STEP * 0.01, Y_TICK_STEP))
    ax.legend(loc="upper right", handlelength=2.5)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _detailed_csv_path(data_dir: str) -> str:
    path = os.path.join(data_dir, "saturation_placements_detailed.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    return path


def _require_pool_columns(df: pd.DataFrame, *, path: str) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")


def _validate_pool_geometry(df: pd.DataFrame, *, path: str, step: int) -> None:
    _require_pool_columns(df, path=path)
    geometry = df[list(_GEOMETRY_COLUMNS)].apply(pd.to_numeric, errors="coerce")
    if geometry.isna().any().any():
        bad = geometry.columns[geometry.isna().any()].tolist()
        raise ValueError(
            f"Missing/invalid geometry in {path} step {step}: {', '.join(bad)}"
        )
    quat = geometry[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=float)
    norms = np.linalg.norm(quat, axis=1)
    if np.any(norms < 1e-12):
        n_bad = int(np.sum(norms < 1e-12))
        raise ValueError(
            f"Degenerate quaternion rows in {path} step {step}: {n_bad} placement(s)"
        )
    if df["energy_adsorption"].isna().any():
        raise ValueError(f"Missing energy_adsorption values in {path} step {step}")


def list_steps(data_dir: str) -> list[int]:
    path = _detailed_csv_path(data_dir)
    df = pd.read_csv(path)
    _require_pool_columns(df, path=path)
    return sorted(int(s) for s in df["step"].unique())


def load_pool(data_dir: str, *, step: int) -> tuple[pd.DataFrame, pd.Series]:
    path = _detailed_csv_path(data_dir)
    df = pd.read_csv(path)
    _require_pool_columns(df, path=path)
    df = df.loc[df["step"] == step].copy()
    if df.empty:
        raise ValueError(f"No rows for step {step} in {path}")
    _validate_pool_geometry(df, path=path, step=step)
    X, y = extract_features_from_dataset(df, target_column="energy_adsorption")
    expected = get_feature_names()
    if list(X.columns) != expected:
        raise ValueError(
            f"Feature columns {list(X.columns)} != production BO features {expected}"
        )
    return X.reset_index(drop=True), y.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _require_resolved_replay(config: AdsorptionConfig) -> None:
    if config.bo_initial_random is None or config.bo_batch_size is None:
        raise ValueError(
            "Offline replay requires resolved bo_initial_random and bo_batch_size"
        )


def _aggregate(
    curves: list[list[float]],
    config: AdsorptionConfig,
    *,
    eval_points: list[int] | None = None,
) -> pd.DataFrame:
    if eval_points is None:
        eval_points = bo_eval_schedule(config)
    rows = []
    for ep in eval_points:
        vals = [float(c[ep - 1]) for c in curves if len(c) >= ep]
        if vals:
            rows.append(
                {
                    "eval_count": ep,
                    "mean_best": float(np.mean(vals)),
                    "std_best": float(np.std(vals)),
                }
            )
    return pd.DataFrame(rows)


def _record_batch(
    indices: list[int],
    *,
    y_arr: np.ndarray,
    evaluated: set[int],
    selected: list[float],
    curve: list[float],
    best: float,
) -> float:
    for i in indices:
        evaluated.add(int(i))
        e = float(y_arr[i])
        selected.append(e)
        best = min(best, e)
    curve.extend([best] * len(indices))
    return best


def _run_replay(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    config: AdsorptionConfig,
    *,
    mode: str,
    surrogate: str = SURROGATE,
    acquisition: str = ACQUISITION,
    kappa: float = KAPPA,
    prior: BOStepMemory | None = None,
    anchor: BOStepMemory | None = None,
    transfer: bool = False,
    use_exploration: bool = False,
    transfer_kwargs: dict[str, float | int] | None = None,
) -> SearchRunResult:
    """Replay pool lookups with production batch-count semantics.

    Random and BO paths share the same initial-random batch and the same
    number of acquisition batches (``bo_total_budget``), each requesting up
    to ``bo_batch_size`` unevaluated placements.
    """
    _require_resolved_replay(config)
    xfer_kw = (
        transfer_kwargs
        if transfer_kwargs is not None
        else _transfer_kwargs_from_config(config)
    )
    rng = np.random.RandomState(seed)
    n = len(y)
    y_arr = np.asarray(y).ravel()
    evaluated: set[int] = set()
    curve: list[float] = []
    selected: list[float] = []
    best = np.inf
    best_X: dict[str, float] | None = None
    obs_X: list[dict[str, float]] = []
    obs_y: list[float] = []
    transfer_disabled = False
    transfer_bad_rounds = 0
    weight_shares: list[float] = []

    n_init = min(int(config.bo_initial_random), n)
    init_idx = select_initial_bo_indices(
        X,
        n_init,
        sampling=config.bo_initial_sampling,
        random_state=seed,
    )
    best = _record_batch(
        init_idx,
        y_arr=y_arr,
        evaluated=evaluated,
        selected=selected,
        curve=curve,
        best=best,
    )
    if mode in ("bo", "transfer"):
        for i in init_idx:
            row = X.iloc[i].to_dict()
            e = float(y_arr[i])
            obs_X.append(row)
            obs_y.append(e)
            if e <= best:
                best_X = dict(row)

    batches_run = 0
    while batches_run < int(config.bo_total_budget):
        uneval = [i for i in range(n) if i not in evaluated]
        if not uneval:
            break
        batch = min(int(config.bo_batch_size), len(uneval))

        if mode == "random" or len(evaluated) < 3:
            chosen = rng.choice(uneval, size=batch, replace=False).tolist()
        else:
            model = None
            if mode == "transfer":
                X_cur = pd.DataFrame(obs_X)
                y_cur = np.asarray(obs_y, dtype=float)
                use_prior = (
                    transfer
                    and prior is not None
                    and len(prior.observed_X_rows) > 0
                    and not transfer_disabled
                    and len(X_cur) >= int(xfer_kw["min_step_observations"])
                )
                if use_prior:
                    placement = (
                        anchor.best_X_row if anchor and anchor.best_X_row else None
                    )
                    tr = build_transfer_surrogate(
                        X_cur,
                        y_cur,
                        prior.observed_X_rows,
                        prior.observed_y,
                        surrogate=surrogate,  # type: ignore[arg-type]
                        n_estimators=100,
                        random_state=seed,
                        weight_cap=float(xfer_kw["weight_cap"]),
                        similarity_lengthscale=float(xfer_kw["similarity_lengthscale"]),
                        min_similarity=float(xfer_kw["min_similarity"]),
                        mae_tolerance=float(xfer_kw["mae_tolerance"]),
                        transfer_bad_rounds=transfer_bad_rounds,
                        trust_patience=int(xfer_kw["trust_patience"]),
                        proximity_lengthscale=float(xfer_kw["proximity_lengthscale"]),
                        proximity_floor=float(xfer_kw["proximity_floor"]),
                        prior_step_ages=prior.step_ages,
                        recency_lengthscale=float(xfer_kw["recency_lengthscale"]),
                        prior_placement_X=placement,
                        occupancy_lengthscale=float(xfer_kw["occupancy_lengthscale"]),
                        occupancy_floor=float(xfer_kw["occupancy_floor"]),
                    )
                    transfer_bad_rounds = tr.transfer_bad_rounds
                    if tr.transfer_disabled:
                        transfer_disabled = True
                    if tr.transfer_weight_share > 0:
                        weight_shares.append(tr.transfer_weight_share)
                    model = tr.surrogate
                if model is None:
                    model = train_surrogate(
                        X_cur,
                        y_cur,
                        surrogate=surrogate,  # type: ignore[arg-type]
                        n_estimators=100,
                        random_state=seed,
                    )
            else:
                obs = sorted(evaluated)
                model = train_surrogate(
                    X.iloc[obs],
                    y_arr[obs],
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
                f_best=best if np.isfinite(best) else None,
            )
            if use_exploration:
                chosen = _inject_exploration(
                    rng,
                    chosen,
                    batch_size=batch,
                    n_pool=n,
                    evaluated=evaluated,
                    exploration_fraction=float(xfer_kw["exploration_fraction"]),
                )

        if not chosen:
            break
        best = _record_batch(
            chosen,
            y_arr=y_arr,
            evaluated=evaluated,
            curve=curve,
            selected=selected,
            best=best,
        )
        if mode in ("bo", "transfer"):
            for i in chosen:
                row = X.iloc[i].to_dict()
                e = float(y_arr[i])
                obs_X.append(row)
                obs_y.append(e)
                if e <= best:
                    best_X = dict(row)
        batches_run += 1

    memory = None
    if mode == "transfer":
        memory = BOStepMemory(
            observed_X_rows=obs_X,
            observed_y=obs_y,
            best_energy=best if np.isfinite(best) else None,
            best_X_row=best_X,
        )
    share = float(np.mean(weight_shares)) if weight_shares else 0.0
    return SearchRunResult(
        curve,
        best,
        selected,
        memory=memory,
        transfer_weight_share_mean=share,
    )


def _paired_stats(baseline: list[float], treatment: list[float]) -> dict[str, float]:
    if len(baseline) != len(treatment) or not baseline:
        return {"mean_improvement": np.nan, "win_rate": np.nan, "p_value": np.nan}
    diffs = np.asarray(baseline) - np.asarray(treatment)
    t = stats.ttest_rel(baseline, treatment)
    return {
        "mean_improvement": float(np.mean(diffs)),
        "win_rate": float(np.mean(diffs > 0)),
        "p_value": float(t.pvalue),
    }


def _run_random(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    *,
    config: AdsorptionConfig = _REPLAY,
) -> SearchRunResult:
    return _run_replay(X, y, seed, config, mode="random")


def _run_bo(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    *,
    config: AdsorptionConfig = _REPLAY,
    surrogate: str = SURROGATE,
    acquisition: str = ACQUISITION,
    kappa: float = KAPPA,
) -> SearchRunResult:
    return _run_replay(
        X,
        y,
        seed,
        config,
        mode="bo",
        surrogate=surrogate,
        acquisition=acquisition,
        kappa=kappa,
    )


def _transfer_kwargs_from_config(config: AdsorptionConfig) -> dict[str, float | int]:
    return {
        "weight_cap": config.bo_transfer_weight_cap,
        "similarity_lengthscale": config.bo_transfer_similarity_lengthscale,
        "min_similarity": config.bo_transfer_min_similarity,
        "mae_tolerance": config.bo_transfer_mae_tolerance,
        "trust_patience": config.bo_transfer_trust_patience,
        "min_step_observations": config.bo_transfer_min_step_observations,
        "exploration_fraction": config.bo_transfer_exploration_fraction,
        "proximity_lengthscale": config.bo_transfer_proximity_lengthscale,
        "proximity_floor": config.bo_transfer_proximity_floor,
        "recency_lengthscale": config.bo_transfer_recency_lengthscale,
        "occupancy_lengthscale": config.bo_transfer_occupancy_lengthscale,
        "occupancy_floor": config.bo_transfer_occupancy_floor,
    }


def _inject_exploration(
    rng: np.random.RandomState,
    chosen: list[int],
    *,
    batch_size: int,
    n_pool: int,
    evaluated: set[int],
    exploration_fraction: float = TRANSFER_KWARGS["exploration_fraction"],
) -> list[int]:
    frac = float(exploration_fraction)
    if frac <= 0:
        return chosen
    explore_n = int(np.ceil(batch_size * frac))
    uneval = [i for i in range(n_pool) if i not in evaluated and i not in chosen]
    if not uneval or explore_n <= 0:
        return chosen
    picks = rng.choice(
        uneval, size=min(explore_n, len(uneval), len(chosen)), replace=False
    )
    kept = chosen[: batch_size - len(picks)]
    return (kept + picks.tolist())[:batch_size]


def _run_bo_transfer(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    *,
    prior: BOStepMemory | None,
    anchor: BOStepMemory | None,
    transfer: bool,
    config: AdsorptionConfig = _REPLAY,
    surrogate: str = SURROGATE,
    acquisition: str = ACQUISITION,
    kappa: float = KAPPA,
) -> tuple[SearchRunResult, BOStepMemory, float]:
    result = _run_replay(
        X,
        y,
        seed,
        config,
        mode="transfer",
        prior=prior,
        anchor=anchor,
        transfer=transfer,
        use_exploration=True,
        surrogate=surrogate,
        acquisition=acquisition,
        kappa=kappa,
        transfer_kwargs=_transfer_kwargs_from_config(config),
    )
    assert result.memory is not None
    return result, result.memory, result.transfer_weight_share_mean


# ---------------------------------------------------------------------------
# Benchmark runs
# ---------------------------------------------------------------------------


def _run_config_grid(
    X: pd.DataFrame,
    y: pd.Series,
    configs: list[tuple[str, str, float]],
    *,
    step: int,
    seeds: int,
    config: AdsorptionConfig = _REPLAY,
    eval_points: list[int] | None = None,
) -> pd.DataFrame:
    oracle = float(y.min())
    rows: list[dict[str, float | int | str]] = []
    for surrogate, acquisition, kappa in configs:
        key = _config_key(surrogate, acquisition, kappa)
        curves = []
        for seed in range(seeds):
            if surrogate == "random_search":
                curves.append(_run_random(X, y, seed, config=config).curve)
            else:
                curves.append(
                    _run_bo(
                        X,
                        y,
                        seed,
                        config=config,
                        surrogate=surrogate,
                        acquisition=acquisition,
                        kappa=kappa,
                    ).curve
                )
        for _, row in _aggregate(curves, config, eval_points=eval_points).iterrows():
            rows.append(
                {
                    "step": step,
                    "config": key,
                    "eval_count": int(row["eval_count"]),
                    "mean_best": float(row["mean_best"]),
                    "std_best": float(row["std_best"]),
                    "oracle_best": oracle,
                }
            )
        logger.info("Sweep %s (%d seeds)", key, seeds)
    return pd.DataFrame(rows)


def run_screening(data_dir: str, *, seeds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Default BO vs random on every available step."""
    steps = list_steps(data_dir)
    metrics_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for step in steps:
        X, y = load_pool(data_dir, step=step)
        oracle = float(y.min())
        logger.info(
            "Screening step %d: %d placements, oracle %.4f eV", step, len(X), oracle
        )

        rand_finals, rand_curves = [], []
        bo_finals, bo_curves = [], []
        for seed in range(seeds):
            r = _run_random(X, y, seed)
            b = _run_bo(X, y, seed)
            if len(r.selected_energies) != len(b.selected_energies):
                raise RuntimeError(
                    f"step {step} seed {seed}: random looked up "
                    f"{len(r.selected_energies)} placements, BO "
                    f"{len(b.selected_energies)}"
                )
            rand_finals.append(r.final_best)
            rand_curves.append(r.curve)
            bo_finals.append(b.final_best)
            bo_curves.append(b.curve)

        stats = _paired_stats(rand_finals, bo_finals)
        bo50 = [c[49] for c in bo_curves if len(c) >= 50]
        rand50 = [c[49] for c in rand_curves if len(c) >= 50]
        stats50 = _paired_stats(rand50, bo50) if len(bo50) == len(rand50) else stats

        metrics_rows.append(
            {
                "step": step,
                "oracle_best": oracle,
                "regret_at_100": float(np.mean(bo_finals)) - oracle,
                "vs_random_mean_improvement": stats["mean_improvement"],
                "vs_random_p_value": stats["p_value"],
                "vs_random_improvement_at_50": stats50["mean_improvement"],
                "vs_random_p_value_at_50": stats50["p_value"],
                "random_mean_final_best": float(np.mean(rand_finals)),
            }
        )

        for config, curves in (
            ("random_search", rand_curves),
            (_default_key(), bo_curves),
        ):
            for _, row in _aggregate(curves, _REPLAY).iterrows():
                curve_rows.append(
                    {
                        "step": step,
                        "config": config,
                        "eval_count": int(row["eval_count"]),
                        "mean_best": float(row["mean_best"]),
                        "std_best": float(row["std_best"]),
                    }
                )

    return pd.DataFrame(metrics_rows), pd.DataFrame(curve_rows)


def run_step1_sweeps(data_dir: str, *, seeds: int) -> dict[str, pd.DataFrame]:
    X, y = load_pool(data_dir, step=1)
    logger.info("Step 1 sweeps: %d placements, oracle %.4f eV", len(X), float(y.min()))

    arch_cfgs = [(s, ACQUISITION, KAPPA) for s in ARCHITECTURES]
    arch_df = _run_config_grid(X, y, arch_cfgs, step=1, seeds=seeds)

    acq_cfgs = [(SURROGATE, a, KAPPA) for a in ACQUISITIONS]
    acq_df = _run_config_grid(X, y, acq_cfgs, step=1, seeds=seeds)

    batch_rows: list[dict[str, float | int | str]] = []
    oracle = float(y.min())
    for batch_size in BATCH_SIZES:
        batch_cfg = _replay_cfg_for_batch(batch_size)
        budget = resolved_bo_eval_budget(batch_cfg)
        curves = [_run_bo(X, y, seed, config=batch_cfg).curve for seed in range(seeds)]
        key = f"batch_{batch_size}"
        for _, row in _aggregate(curves, batch_cfg).iterrows():
            batch_rows.append(
                {
                    "step": 1,
                    "config": key,
                    "eval_count": int(row["eval_count"]),
                    "mean_best": float(row["mean_best"]),
                    "std_best": float(row["std_best"]),
                    "oracle_best": oracle,
                    "batch_size": batch_size,
                    "total_budget": budget,
                }
            )
        logger.info("Batch sweep %s: budget=%d", key, budget)

    init_rows: list[dict[str, float | int | str]] = []
    for sampling in INIT_STRATEGIES:
        init_cfg = _replay_cfg_for_init(sampling)
        curves = [_run_bo(X, y, seed, config=init_cfg).curve for seed in range(seeds)]
        key = f"init_{sampling}"
        for _, row in _aggregate(curves, init_cfg).iterrows():
            init_rows.append(
                {
                    "step": 1,
                    "config": key,
                    "eval_count": int(row["eval_count"]),
                    "mean_best": float(row["mean_best"]),
                    "std_best": float(row["std_best"]),
                    "oracle_best": oracle,
                    "initial_sampling": sampling,
                }
            )
        logger.info("Init sweep %s", key)

    return {
        "architecture": arch_df,
        "acquisition": acq_df,
        "batch": pd.DataFrame(batch_rows),
        "init": pd.DataFrame(init_rows),
    }


def run_transfer(
    data_dir: str, *, seeds: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Baseline vs transfer BO across all saturation steps."""
    steps = list_steps(data_dir)
    transfer_steps = [s for s in steps if s >= 2]
    detail_rows: list[dict[str, float | int]] = []
    baseline_curves: dict[int, list[list[float]]] = {s: [] for s in transfer_steps}
    transfer_curves: dict[int, list[list[float]]] = {s: [] for s in transfer_steps}

    for seed in range(seeds):
        memories: list[BOStepMemory] = []
        prior_anchor: BOStepMemory | None = None
        for step in steps:
            X, y = load_pool(data_dir, step=step)
            oracle = float(y.min())
            rs = seed + step * 101
            prior = (
                windowed_bo_step_memories(memories, window=TRANSFER_WINDOW)
                if step >= 2
                else None
            )
            bl, bl_mem, _ = _run_bo_transfer(
                X, y, rs, prior=None, anchor=None, transfer=False
            )
            tr, tr_mem, share = _run_bo_transfer(
                X,
                y,
                rs,
                prior=prior,
                anchor=prior_anchor,
                transfer=step >= 2,
            )
            prior_anchor = tr_mem
            memories.append(tr_mem)
            if step in transfer_steps:
                detail_rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "oracle_best": oracle,
                        "baseline_best": bl.final_best,
                        "transfer_best": tr.final_best,
                        "transfer_weight_share_mean": share,
                    }
                )
                baseline_curves[step].append(bl.curve)
                transfer_curves[step].append(tr.curve)

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    curve_rows = []
    for step in transfer_steps:
        bl = detail_df.loc[detail_df["step"] == step, "baseline_best"].tolist()
        tr = detail_df.loc[detail_df["step"] == step, "transfer_best"].tolist()
        st = _paired_stats(bl, tr)
        oracle = float(detail_df.loc[detail_df["step"] == step, "oracle_best"].iloc[0])
        summary_rows.append(
            {
                "step": step,
                "mean_improvement": st["mean_improvement"],
                "win_rate": st["win_rate"],
                "p_value": st["p_value"],
                "baseline_regret_at_100": float(np.mean(bl)) - oracle,
                "transfer_regret_at_100": float(np.mean(tr)) - oracle,
            }
        )
        for variant, curves in (
            ("baseline", baseline_curves[step]),
            ("transfer", transfer_curves[step]),
        ):
            for _, row in _aggregate(curves, _REPLAY).iterrows():
                curve_rows.append(
                    {
                        "step": step,
                        "variant": variant,
                        "eval_count": int(row["eval_count"]),
                        "mean_best": float(row["mean_best"]),
                        "std_best": float(row["std_best"]),
                    }
                )
    return detail_df, pd.DataFrame(summary_rows), pd.DataFrame(curve_rows)


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------


def _ordered_keys(
    df: pd.DataFrame,
    col: str,
    *,
    default: str | None = None,
    priority: tuple[str, ...] = (),
) -> list[str]:
    present = list(df[col].unique())
    out: list[str] = [k for k in priority if k in present]
    if default and default in present and default not in out:
        out.insert(0, default)
    for k in sorted(present):
        if k not in out:
            out.append(k)
    return out


def write_figures(
    out_dir: str,
    *,
    screening_curves: pd.DataFrame,
    screening_report: pd.DataFrame,
    transfer_curves: pd.DataFrame,
    arch_curves: pd.DataFrame,
    batch_curves: pd.DataFrame,
    acq_curves: pd.DataFrame,
    init_curves: pd.DataFrame,
) -> None:
    default_key = _default_key()
    default_init = f"init_{_CFG.bo_initial_sampling}"
    default_keys = frozenset({default_key, "baseline", "batch_5", default_init})

    def oracle(step: int) -> float:
        row = screening_report.loc[screening_report["step"] == step, "oracle_best"]
        return float(row.iloc[0]) if not row.empty else 0.0

    _plot_curves(
        screening_curves[screening_curves["step"] == 1],
        oracle(1),
        _ordered_keys(
            screening_curves[screening_curves["step"] == 1],
            "config",
            default=default_key,
            priority=("random_search",),
        ),
        out_path=os.path.join(out_dir, "fig1_step1_bo_vs_random.png"),
        default_keys=default_keys,
        x_max=EVAL_BUDGET,
        y_max=STEP1_FIG_Y_MAX,
    )

    acq = ACQUISITION
    _plot_curves(
        arch_curves,
        oracle(1),
        _ordered_keys(arch_curves, "config", default=default_key),
        out_path=os.path.join(out_dir, "fig2_step1_architectures.png"),
        default_keys=default_keys,
        y_max=STEP1_FIG_Y_MAX,
        label_overrides={
            _config_key("random_forest", acq, KAPPA): "Random forest",
            _config_key("extra_trees", acq, KAPPA): "Extra trees",
            _config_key("gradient_boost", acq, KAPPA): "Gradient boost",
            _config_key("gaussian_process", acq, KAPPA): "Gaussian process",
            _config_key("ensemble", acq, KAPPA): "Ensemble",
        },
        x_max=EVAL_BUDGET,
    )

    _plot_curves(
        batch_curves,
        oracle(1),
        _ordered_keys(batch_curves, "config", default="batch_5"),
        out_path=os.path.join(out_dir, "fig3_step1_batch_size.png"),
        default_keys=default_keys,
        y_max=STEP1_FIG_Y_MAX,
        label_overrides={
            f"batch_{b}": f"Batch {b}"
            for b in BATCH_SIZES
            if b != int(_REPLAY.bo_batch_size)
        },
        x_max=EVAL_BUDGET,
    )

    sur = SURROGATE
    _plot_curves(
        acq_curves,
        oracle(1),
        _ordered_keys(
            acq_curves, "config", default=_config_key(sur, ACQUISITION, KAPPA)
        ),
        out_path=os.path.join(out_dir, "fig4_step1_acquisition.png"),
        default_keys=default_keys,
        y_max=STEP1_FIG_Y_MAX,
        label_overrides={
            _config_key(sur, "lcb", KAPPA): "LCB",
            _config_key(sur, "pi", KAPPA): "PI",
        },
        x_max=EVAL_BUDGET,
    )

    _plot_curves(
        init_curves,
        oracle(1),
        _ordered_keys(
            init_curves,
            "config",
            default=default_init,
            priority=tuple(f"init_{s}" for s in INIT_STRATEGIES),
        ),
        out_path=os.path.join(out_dir, "fig5_step1_initial_sampling.png"),
        default_keys=default_keys,
        y_max=STEP1_FIG_Y_MAX,
        label_overrides={f"init_{key}": INIT_LABELS[key] for key in INIT_STRATEGIES},
        x_max=EVAL_BUDGET,
    )

    for step in sorted(transfer_curves["step"].unique()):
        step = int(step)
        if step < 2:
            continue
        xfer = transfer_curves[transfer_curves["step"] == step]
        rand = screening_curves[
            (screening_curves["step"] == step)
            & (screening_curves["config"] == "random_search")
        ]
        combined = pd.concat(
            [rand.assign(variant="random"), xfer],
            ignore_index=True,
        )
        _plot_curves(
            combined,
            oracle(step),
            _ordered_keys(
                combined, "variant", priority=("random", "baseline", "transfer")
            ),
            out_path=os.path.join(out_dir, f"transfer_step{step}_saturation.png"),
            series_col="variant",
            default_keys=default_keys,
            x_max=EVAL_BUDGET,
        )


def _regret_at(
    curves: pd.DataFrame, oracle: float, key: str, ep: int, *, col: str
) -> float:
    row = curves[(curves[col] == key) & (curves["eval_count"] == ep)]
    return float(row["mean_best"].iloc[0] - oracle) if not row.empty else float("nan")


def _regret_final(curves: pd.DataFrame, oracle: float, key: str, *, col: str) -> float:
    grp = curves[curves[col] == key].sort_values("eval_count")
    return float(grp["mean_best"].iloc[-1] - oracle) if not grp.empty else float("nan")


def write_report(
    out_dir: str,
    *,
    data_dir: str,
    seeds: int,
    screening_report: pd.DataFrame,
    screening_curves: pd.DataFrame,
    transfer_curves: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    arch_curves: pd.DataFrame,
    batch_curves: pd.DataFrame,
    acq_curves: pd.DataFrame,
    init_curves: pd.DataFrame,
) -> None:
    cfg = _cfg()
    lines = [
        "# BO Benchmark Report",
        "",
        f"Data directory: `{data_dir}`",
        f"Seeds: {seeds}",
        f"Default BO: `{cfg.bo_surrogate}` / `{cfg.bo_acquisition}` (κ={cfg.bo_ucb_kappa}), "
        f"init=`{cfg.bo_initial_sampling}`",
        f"Transfer: `{cfg.bo_transfer_mode}`, window={cfg.bo_transfer_prior_step_window}, "
        f"recency_ls={cfg.bo_transfer_recency_lengthscale}, "
        f"occupancy_ls={cfg.bo_transfer_occupancy_lengthscale}",
        "",
    ]

    s1 = screening_report[screening_report["step"] == 1]
    if not s1.empty:
        r = s1.iloc[0]
        lines += [
            "## Step 1: default BO vs random",
            "",
            f"- Oracle E_ads: **{r['oracle_best']:.4f} eV**",
            f"- Default BO regret@100: **{r['regret_at_100']:.4f} eV**",
            f"- Random regret@100: **{r['random_mean_final_best'] - r['oracle_best']:.4f} eV**",
            f"- Improvement@50: **{r['vs_random_improvement_at_50']:.4f} eV** "
            f"(p={r['vs_random_p_value_at_50']:.4g})",
            f"- Final improvement: **{r['vs_random_mean_improvement']:.4f} eV** "
            f"(p={r['vs_random_p_value']:.4g})",
            "",
        ]

    oracle1 = float(s1.iloc[0]["oracle_best"]) if not s1.empty else float("nan")
    if np.isfinite(oracle1):
        lines.append("## Step 1 sweeps (regret@100)")
        lines.append("")
        dk = _default_key()
        for label, df in (
            ("architecture", arch_curves),
            ("batch", batch_curves),
            ("acquisition", acq_curves),
            ("initial sampling", init_curves),
        ):
            regrets = {
                k: _regret_final(df, oracle1, k, col="config")
                for k in df["config"].unique()
            }
            best = min(regrets, key=regrets.get)  # type: ignore[arg-type]
            lines.append(
                f"- Best {label}: **{best}** ({regrets[best]:.4f} eV); "
                f"default ({dk}): {regrets.get(dk, float('nan')):.4f} eV"
            )
        lines.append("")

    lines.append("## Transfer saturation (steps 2–end)")
    lines.append("")
    for step in sorted(transfer_summary["step"].unique()):
        step = int(step)
        trow = transfer_summary[transfer_summary["step"] == step].iloc[0]
        oracle = float(
            screening_report.loc[screening_report["step"] == step, "oracle_best"].iloc[
                0
            ]
        )
        lines += [
            f"### Step {step}",
            "",
            f"- Baseline regret@100: **{trow['baseline_regret_at_100']:.4f} eV**",
            f"- Transfer regret@100: **{trow['transfer_regret_at_100']:.4f} eV**",
            f"- Transfer vs baseline: **{trow['mean_improvement']:.4f} eV** "
            f"(p={trow['p_value']:.4g})",
        ]
        xfer = transfer_curves[transfer_curves["step"] == step]
        for ep in (20, 30, 50):
            b = _regret_at(xfer, oracle, "baseline", ep, col="variant")
            t = _regret_at(xfer, oracle, "transfer", ep, col="variant")
            if np.isfinite(b) and np.isfinite(t):
                lines.append(
                    f"- E_rel @{ep}: baseline **{b:.4f}**, transfer **{t:.4f}** (Δ={b - t:+.4f})"
                )
        rand = _regret_final(
            screening_curves[screening_curves["step"] == step],
            oracle,
            "random_search",
            col="config",
        )
        lines += [f"- Random regret@100: **{rand:.4f} eV**", ""]
        lines.append(f"- Figure: `transfer_step{step}_saturation.png`")
        lines.append("")

    lines += [
        "## Figures",
        "",
        "- `fig1_step1_bo_vs_random.png`",
        "- `fig2_step1_architectures.png`",
        "- `fig3_step1_batch_size.png`",
        "- `fig4_step1_acquisition.png`",
        "- `fig5_step1_initial_sampling.png`",
    ]
    for step in sorted(transfer_summary["step"].unique()):
        lines.append(f"- `transfer_step{int(step)}_saturation.png`")
    lines.append("")

    path = os.path.join(out_dir, "report.md")
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _clean_stale(out_dir: str) -> None:
    root = Path(out_dir)
    for pattern in STALE_ARTIFACT_GLOBS:
        for path in root.glob(pattern):
            path.unlink(missing_ok=True)


def run_benchmark(
    data_dir: str,
    *,
    seeds: int,
    report_dir: str,
) -> None:
    os.makedirs(report_dir, exist_ok=True)
    _clean_stale(report_dir)
    _apply_plot_style()

    logger.info("Screening: default BO vs random (%d seeds)", seeds)
    screening_report_df, screening_curves_df = run_screening(data_dir, seeds=seeds)
    screening_report_df.to_csv(
        os.path.join(report_dir, "screening_report.csv"), index=False
    )
    screening_curves_df.to_csv(
        os.path.join(report_dir, "screening_curves.csv"), index=False
    )

    logger.info("Transfer: baseline vs transfer BO (%d seeds)", seeds)
    transfer_df, transfer_summary_df, transfer_curves_df = run_transfer(
        data_dir, seeds=seeds
    )
    transfer_df.to_csv(os.path.join(report_dir, "transfer.csv"), index=False)
    transfer_summary_df.to_csv(
        os.path.join(report_dir, "transfer_summary.csv"), index=False
    )
    transfer_curves_df.to_csv(
        os.path.join(report_dir, "transfer_curves.csv"), index=False
    )

    logger.info("Step 1 sweeps (%d seeds)", seeds)
    sweeps = run_step1_sweeps(data_dir, seeds=seeds)
    arch_df = sweeps["architecture"]
    acq_df = sweeps["acquisition"]
    batch_df = sweeps["batch"]
    init_df = sweeps["init"]
    arch_df.to_csv(
        os.path.join(report_dir, "step1_architecture_curves.csv"), index=False
    )
    acq_df.to_csv(os.path.join(report_dir, "step1_acquisition_curves.csv"), index=False)
    batch_df.to_csv(os.path.join(report_dir, "step1_batch_curves.csv"), index=False)
    init_df.to_csv(os.path.join(report_dir, "step1_init_curves.csv"), index=False)

    write_figures(
        report_dir,
        screening_curves=screening_curves_df,
        screening_report=screening_report_df,
        transfer_curves=transfer_curves_df,
        arch_curves=arch_df,
        batch_curves=batch_df,
        acq_curves=acq_df,
        init_curves=init_df,
    )
    write_report(
        report_dir,
        data_dir=data_dir,
        seeds=seeds,
        screening_report=screening_report_df,
        screening_curves=screening_curves_df,
        transfer_curves=transfer_curves_df,
        transfer_summary=transfer_summary_df,
        arch_curves=arch_df,
        batch_curves=batch_df,
        acq_curves=acq_df,
        init_curves=init_df,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline BO benchmark on bipyridine/Au(111) placement pools"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory with saturation_placements_detailed.csv (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Number of random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--report-dir",
        default=DEFAULT_REPORT_DIR,
        help=f"Output directory for CSVs, figures, report (default: {DEFAULT_REPORT_DIR})",
    )
    args = parser.parse_args()
    if args.seeds < 1:
        logger.error("--seeds must be >= 1")
        return 1
    run_benchmark(args.data_dir, seeds=args.seeds, report_dir=args.report_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

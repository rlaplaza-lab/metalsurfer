#!/usr/bin/env python3
"""Offline BO benchmark: compare surrogate models and acquisition functions.

Loads real placement data from metalsurfer results directories, builds X/y,
and simulates the BO loop with fixed batch size 10.

Usage:
  python scripts/benchmark_bo_models.py \\
    --data-dir examples/results_bipyridine_au111_defects_saturation_raw \\
    --step 3 --seeds 30 --plot
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from benchmark_bo_common import (
    BATCH_SIZE,
    DEFAULT_BO_ACQUISITION,
    DEFAULT_BO_KAPPA,
    DEFAULT_BO_SURROGATE,
    DEFAULT_SMILES,
    DEFAULT_SURFACE_TYPE,
    INITIAL_RANDOM,
    TOTAL_BUDGET,
    aggregate_curves,
    list_available_steps,
    load_placement_pool,
    paired_stats,
    run_offline_bo,
    run_random_search,
)

from metalsurfer import configure_logging

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)

MODEL_CONFIGS: list[tuple[str, str, float]] = [
    ("random_forest", "lcb", 0.0),
    ("random_forest", "lcb", 1.0),
    ("random_forest", "lcb", 1.96),
    ("random_forest", "lcb", 2.5),
    ("random_forest", "ei", 1.96),
    ("random_forest", "pi", 1.96),
    ("extra_trees", "lcb", 1.0),
    ("extra_trees", "pi", 1.96),
    ("gaussian_process", "lcb", 1.0),
    ("gaussian_process", "lcb", 1.96),
    ("ensemble", "lcb", 1.0),
    ("ensemble", "lcb", 1.96),
    ("ensemble", "ei", 1.96),
    ("gradient_boost", "lcb", 0.0),
    ("ridge", "lcb", 0.0),
]


def _config_key(surrogate: str, acquisition: str, kappa: float) -> str:
    suffix = f"_k{kappa}" if acquisition == "lcb" else ""
    return f"{surrogate}_{acquisition}{suffix}"


def _is_default_config(surrogate: str, acquisition: str, kappa: float) -> bool:
    return (
        surrogate == DEFAULT_BO_SURROGATE
        and acquisition == DEFAULT_BO_ACQUISITION
        and kappa == DEFAULT_BO_KAPPA
    )


def _run_config(
    config: tuple[str, str, float],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seeds: int,
    initial_random: int,
    batch_size: int,
    total_budget: int,
) -> tuple[list[float], list[list[float]]]:
    surrogate, acquisition, kappa = config
    if surrogate == "random_search":
        runner = lambda seed: run_random_search(  # noqa: E731
            X, y, initial_random, batch_size, total_budget, seed
        )
    else:
        runner = lambda seed: run_offline_bo(  # noqa: E731
            X,
            y,
            initial_random,
            batch_size,
            total_budget,
            seed,
            surrogate=surrogate,
            acquisition=acquisition,
            kappa=kappa,
        )
    finals: list[float] = []
    curves: list[list[float]] = []
    for seed in range(seeds):
        curve, final_best = runner(seed)
        finals.append(final_best)
        curves.append(curve)
    return finals, curves


def _metrics_row(
    *,
    surrogate: str,
    acquisition: str,
    kappa: float,
    oracle_best: float,
    finals: list[float],
    curves: list[list[float]],
    seeds: int,
    random_finals: list[float] | None,
    step: int | None,
) -> dict[str, object]:
    best20 = [float(c[19]) for c in curves if len(c) >= 20]
    best50 = [float(c[49]) for c in curves if len(c) >= 50]
    best100 = [float(c[99]) for c in curves if len(c) >= 100]
    mean_best = float(np.mean(finals))
    std_best = float(np.std(finals))
    row: dict[str, object] = {
        "step": step,
        "surrogate": surrogate,
        "acquisition": acquisition,
        "kappa": kappa,
        "is_default": _is_default_config(surrogate, acquisition, kappa),
        "oracle_best": oracle_best,
        "mean_best_at_20": float(np.mean(best20)) if best20 else float("nan"),
        "mean_best_at_50": float(np.mean(best50)) if best50 else float("nan"),
        "mean_best_at_100": float(np.mean(best100)) if best100 else float("nan"),
        "mean_final_best": mean_best,
        "std_final_best": std_best,
        "regret_at_100": mean_best - oracle_best,
        "n_seeds": seeds,
    }
    if random_finals is not None:
        stats = paired_stats(random_finals, finals)
        row["vs_random_mean_improvement"] = stats["mean_improvement"]
        row["vs_random_win_rate"] = stats["win_rate"]
        row["vs_random_p_value"] = stats["p_value"]
    return row


def _write_plot(
    curve_by_config: dict[str, list[list[float]]],
    out_path: str,
    *,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    eval_points = list(range(10, 101, 10))
    for key, curves in curve_by_config.items():
        agg = aggregate_curves(curves, eval_points=eval_points)
        if agg.empty:
            continue
        label = key
        if key == _config_key(
            DEFAULT_BO_SURROGATE, DEFAULT_BO_ACQUISITION, DEFAULT_BO_KAPPA
        ):
            label += " [DEFAULT]"
        ax.plot(
            agg["eval_count"],
            agg["mean_best"],
            marker="o",
            label=label,
        )
        ax.fill_between(
            agg["eval_count"],
            agg["mean_best"] - agg["std_best"],
            agg["mean_best"] + agg["std_best"],
            alpha=0.15,
        )
    ax.set_xlabel("Evaluations")
    ax.set_ylabel("Mean best E_ads (eV)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote plot %s", out_path)


def _benchmark_pool(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seeds: int,
    step: int | None,
    configs: list[tuple[str, str, float]],
    include_random: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[list[float]]]]:
    oracle_best = float(np.min(y))
    random_finals: list[float] | None = None
    curve_by_config: dict[str, list[list[float]]] = {}
    if include_random:
        random_finals, random_curves = _run_config(
            ("random_search", "lcb", 0.0),
            X,
            y,
            seeds=seeds,
            initial_random=INITIAL_RANDOM,
            batch_size=BATCH_SIZE,
            total_budget=TOTAL_BUDGET,
        )
        curve_by_config["random_search"] = random_curves

    rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for surrogate, acquisition, kappa in configs:
        key = _config_key(surrogate, acquisition, kappa)
        finals, curves = _run_config(
            (surrogate, acquisition, kappa),
            X,
            y,
            seeds=seeds,
            initial_random=INITIAL_RANDOM,
            batch_size=BATCH_SIZE,
            total_budget=TOTAL_BUDGET,
        )
        curve_by_config[key] = curves
        rows.append(
            _metrics_row(
                surrogate=surrogate,
                acquisition=acquisition,
                kappa=kappa,
                oracle_best=oracle_best,
                finals=finals,
                curves=curves,
                seeds=seeds,
                random_finals=random_finals,
                step=step,
            )
        )
        agg = aggregate_curves(curves)
        for _, agg_row in agg.iterrows():
            curve_rows.append(
                {
                    "step": step,
                    "config": key,
                    "eval_count": int(agg_row["eval_count"]),
                    "mean_best": float(agg_row["mean_best"]),
                    "std_best": float(agg_row["std_best"]),
                }
            )
        default_tag = (
            " [DEFAULT]" if _is_default_config(surrogate, acquisition, kappa) else ""
        )
        logger.info(
            "%s%s: mean best E_ads = %.4f ± %.4f (regret@100 = %.4f)",
            key,
            default_tag,
            float(np.mean(finals)),
            float(np.std(finals)),
            float(np.mean(finals)) - oracle_best,
        )

    if include_random and random_finals is not None:
        random_mean = float(np.mean(random_finals))
        rows.insert(
            0,
            _metrics_row(
                surrogate="random_search",
                acquisition="none",
                kappa=float("nan"),
                oracle_best=oracle_best,
                finals=random_finals,
                curves=curve_by_config["random_search"],
                seeds=seeds,
                random_finals=None,
                step=step,
            ),
        )
        logger.info(
            "random_search: mean best E_ads = %.4f ± %.4f (regret@100 = %.4f)",
            random_mean,
            float(np.std(random_finals)),
            random_mean - oracle_best,
        )

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(
        by=["is_default", "mean_final_best"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out_df, pd.DataFrame(curve_rows), curve_by_config


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Offline BO benchmark (fixed batch size 10)"
    )
    ap.add_argument(
        "--data-dir",
        required=True,
        help="Results dir with adsorption_energies_detailed.csv",
    )
    ap.add_argument(
        "--surface-type",
        default=DEFAULT_SURFACE_TYPE,
        help="Surface type label for dataset columns (default: %(default)s)",
    )
    ap.add_argument(
        "--smiles",
        default=DEFAULT_SMILES,
        help="SMILES string for the adsorbate (default: %(default)s)",
    )
    ap.add_argument("--step", type=int, default=None, help="Saturation step pool")
    ap.add_argument(
        "--all-steps",
        action="store_true",
        help="Run benchmark on each saturation step and aggregate",
    )
    ap.add_argument(
        "--out", default="benchmark_bo_results.csv", help="Output metrics CSV"
    )
    ap.add_argument(
        "--curves-out",
        default=None,
        help="Output convergence curves CSV (default: <out>_curves.csv)",
    )
    ap.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Write convergence plot PNG next to --out",
    )
    ap.add_argument("--no-plot", action="store_true", help="Skip writing plot")
    ap.add_argument(
        "--default-only",
        action="store_true",
        help="Compare default BO vs random only (faster)",
    )
    args = ap.parse_args()

    if args.step is not None and args.all_steps:
        raise SystemExit("Use either --step or --all-steps, not both")

    configs = (
        [(DEFAULT_BO_SURROGATE, DEFAULT_BO_ACQUISITION, DEFAULT_BO_KAPPA)]
        if args.default_only
        else MODEL_CONFIGS
    )

    steps: list[int | None]
    if args.all_steps:
        steps = list_available_steps(args.data_dir)
    elif args.step is not None:
        steps = [args.step]
    else:
        steps = [None]

    all_metrics: list[pd.DataFrame] = []
    all_curves: list[pd.DataFrame] = []
    plot_curves: dict[str, list[list[float]]] = {}

    for step in steps:
        X, y, _ = load_placement_pool(
            args.data_dir,
            step=step,
            surface_type=args.surface_type,
            smiles=args.smiles,
        )
        step_label = f"step {step}" if step is not None else "full pool"
        logger.info(
            "Loaded %s: %d placements, %d features, oracle %.4f eV",
            step_label,
            len(X),
            X.shape[1],
            float(np.min(y)),
        )
        metrics_df, curves_df, curve_by_config = _benchmark_pool(
            X,
            y,
            seeds=args.seeds,
            step=step,
            configs=configs,
            include_random=True,
        )
        all_metrics.append(metrics_df)
        all_curves.append(curves_df)
        if len(steps) == 1:
            plot_curves = curve_by_config

    out_df = pd.concat(all_metrics, ignore_index=True)
    curves_out = args.curves_out or args.out.replace(".csv", "_curves.csv")
    curves_df = pd.concat(all_curves, ignore_index=True)

    default_rows = out_df[out_df["is_default"] == True]  # noqa: E712
    if not default_rows.empty:
        dr = default_rows.iloc[0]
        logger.info(
            "Default BO (%s/%s/kappa=%.1f): mean_final_best=%.4f, regret@100=%.4f",
            dr["surrogate"],
            dr["acquisition"],
            dr["kappa"],
            dr["mean_final_best"],
            dr["regret_at_100"],
        )
        if "vs_random_mean_improvement" in dr and pd.notna(
            dr["vs_random_mean_improvement"]
        ):
            logger.info(
                "Default vs random: improvement=%.4f eV, win_rate=%.2f, p=%.4g",
                dr["vs_random_mean_improvement"],
                dr["vs_random_win_rate"],
                dr["vs_random_p_value"],
            )

    out_df.to_csv(args.out, index=False)
    curves_df.to_csv(curves_out, index=False)
    logger.info("Wrote %s", args.out)
    logger.info("Wrote %s", curves_out)

    if args.plot and not args.no_plot and plot_curves:
        plot_path = args.out.replace(".csv", ".png")
        title = f"BO convergence ({args.data_dir})"
        if args.step is not None:
            title += f" step {args.step}"
        _write_plot(plot_curves, plot_path, title=title)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

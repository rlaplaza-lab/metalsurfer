#!/usr/bin/env python3
"""Offline benchmark: baseline per-step BO vs transfer-enabled BO.

This benchmark synthesizes saturation-like step shifts from a saved
``adsorption_energies_detailed.csv`` dataset and compares:
1) current-step-only BO (baseline)
2) BO with cross-step transfer memory and weighted carry-over
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from metalsurfer._logging import configure_logging
from metalsurfer.ml.bayesian import score_and_select, train_surrogate
from metalsurfer.ml.features import extract_features_from_dataset

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)


def _load_xy(data_dir: str) -> tuple[pd.DataFrame, np.ndarray]:
    csv_path = os.path.join(data_dir, "adsorption_energies_detailed.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    X, y = extract_features_from_dataset(df, target_column="energy_adsorption")
    return X.reset_index(drop=True), np.asarray(y, dtype=float)


def _simulate_steps(
    y: np.ndarray, n_steps: int, shift_per_step: float
) -> list[np.ndarray]:
    return [y + i * shift_per_step for i in range(n_steps)]


def _run_step_bo(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    total_budget: int,
    initial_random: int,
    batch_size: int,
    transfer_X: pd.DataFrame | None = None,
    transfer_y: np.ndarray | None = None,
    transfer_weight_cap: float = 0.35,
) -> tuple[float, pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(y)
    evaluated: set[int] = set()
    obs_idx = rng.choice(
        n, size=min(initial_random, total_budget, n), replace=False
    ).tolist()
    evaluated.update(obs_idx)

    while len(evaluated) < min(total_budget, n):
        obs = sorted(evaluated)
        X_cur = X.iloc[obs]
        y_cur = y[obs]
        remaining = min(batch_size, total_budget - len(evaluated))
        if remaining <= 0:
            break
        if len(obs) < 3:
            uneval = [i for i in range(n) if i not in evaluated]
            chosen = rng.choice(
                uneval, size=min(remaining, len(uneval)), replace=False
            ).tolist()
        else:
            if (
                transfer_X is not None
                and transfer_y is not None
                and len(transfer_X) > 0
            ):
                max_transfer_weight = (
                    len(X_cur)
                    * transfer_weight_cap
                    / max(1.0 - transfer_weight_cap, 1e-8)
                )
                transfer_weights = np.full(
                    len(transfer_X), max_transfer_weight / max(len(transfer_X), 1)
                )
                X_train = pd.concat([X_cur, transfer_X], ignore_index=True)
                y_train = np.concatenate([y_cur, transfer_y], axis=0)
                sample_weight = np.concatenate(
                    [np.ones(len(X_cur), dtype=float), transfer_weights], axis=0
                )
                model = train_surrogate(
                    X_train,
                    y_train,
                    surrogate="random_forest",
                    n_estimators=100,
                    random_state=seed,
                    sample_weight=sample_weight,
                )
            else:
                model = train_surrogate(
                    X_cur,
                    y_cur,
                    surrogate="random_forest",
                    n_estimators=100,
                    random_state=seed,
                )
            chosen = score_and_select(
                model,
                X,
                batch_size=remaining,
                kappa=1.0,
                evaluated_indices=evaluated,
                acquisition="lcb",
                f_best=float(np.min(y_cur)),
            )
        for idx in chosen:
            evaluated.add(int(idx))
    final_obs = sorted(evaluated)
    return float(np.min(y[final_obs])), X.iloc[final_obs].copy(), y[final_obs].copy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline BO vs transfer BO across pseudo-steps."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing adsorption_energies_detailed.csv",
    )
    parser.add_argument(
        "--steps", type=int, default=4, help="Number of pseudo saturation steps"
    )
    parser.add_argument(
        "--shift-per-step",
        type=float,
        default=0.2,
        help="Target shift applied each synthetic step (eV)",
    )
    parser.add_argument("--budget", type=int, default=100, help="BO budget per step")
    parser.add_argument(
        "--initial-random", type=int, default=10, help="Initial random evaluations"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size per BO round"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--out", default="benchmark_bo_transfer.csv", help="Output CSV")
    args = parser.parse_args()

    X, y_base = _load_xy(args.data_dir)
    y_steps = _simulate_steps(y_base, args.steps, args.shift_per_step)

    rows: list[dict[str, float | int]] = []
    for seed in range(args.seeds):
        prev_X: pd.DataFrame | None = None
        prev_y: np.ndarray | None = None
        for step_idx, y_step in enumerate(y_steps, start=1):
            round_seed = seed + step_idx * 101
            baseline_best, _, _ = _run_step_bo(
                X,
                y_step,
                seed=round_seed,
                total_budget=args.budget,
                initial_random=args.initial_random,
                batch_size=args.batch_size,
            )
            transfer_best, prev_X, prev_y = _run_step_bo(
                X,
                y_step,
                seed=round_seed,
                total_budget=args.budget,
                initial_random=args.initial_random,
                batch_size=args.batch_size,
                transfer_X=prev_X,
                transfer_y=prev_y,
            )
            rows.append(
                {
                    "seed": seed,
                    "step": step_idx,
                    "baseline_best": baseline_best,
                    "transfer_best": transfer_best,
                    "improvement_vs_baseline": baseline_best - transfer_best,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    logger.info("Wrote %s", args.out)
    summary = df.groupby("step")["improvement_vs_baseline"].mean().reset_index()
    logger.info("Mean improvement by step:\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

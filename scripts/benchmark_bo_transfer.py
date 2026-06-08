#!/usr/bin/env python3
"""Offline benchmark: baseline per-step BO vs transfer-enabled BO on real step pools."""

from __future__ import annotations

import argparse
import logging
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd
from benchmark_bo_common import (
    BATCH_SIZE,
    DEFAULT_SMILES,
    DEFAULT_SURFACE_TYPE,
    INITIAL_RANDOM,
    TOTAL_BUDGET,
    list_available_steps,
    load_placement_pool,
    paired_stats,
    run_offline_bo_with_transfer,
)

from metalsurfer import configure_logging
from metalsurfer.models import BOStepMemory

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)


def _parse_step_range(spec: str, available: list[int]) -> list[int]:
    if spec.lower() == "all":
        return [s for s in available if s >= 2]
    if "-" in spec:
        lo_s, hi_s = spec.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return [s for s in available if lo <= s <= hi]
    step = int(spec)
    return [step]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare baseline BO vs transfer BO across real saturation steps."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing saturation_placements_detailed.csv",
    )
    parser.add_argument(
        "--steps",
        default="2-9",
        help="Step range to benchmark (e.g. 2-9, all, or single step)",
    )
    parser.add_argument(
        "--budget", type=int, default=TOTAL_BUDGET, help="BO budget per step"
    )
    parser.add_argument(
        "--initial-random",
        type=int,
        default=INITIAL_RANDOM,
        help="Initial random evaluations",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size per BO round"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--surface-type",
        default=DEFAULT_SURFACE_TYPE,
        help="Surface type label when missing from CSV",
    )
    parser.add_argument(
        "--smiles",
        default=DEFAULT_SMILES,
        help="SMILES when missing from CSV",
    )
    parser.add_argument("--out", default="benchmark_bo_transfer.csv", help="Output CSV")
    args = parser.parse_args()

    available = list_available_steps(args.data_dir)
    step_list = _parse_step_range(args.steps, available)
    if not step_list:
        raise SystemExit(f"No steps matched {args.steps!r} in {available}")

    rows: list[dict[str, float | int]] = []
    baseline_by_step: dict[int, list[float]] = {s: [] for s in step_list}
    transfer_by_step: dict[int, list[float]] = {s: [] for s in step_list}

    for seed in range(args.seeds):
        prior_memory: BOStepMemory | None = None
        for step in sorted(available):
            X, y, _ = load_placement_pool(
                args.data_dir,
                step=step,
                surface_type=args.surface_type,
                smiles=args.smiles,
            )
            oracle = float(np.min(y))
            round_seed = seed + step * 101

            baseline_best, baseline_memory, _ = run_offline_bo_with_transfer(
                X,
                y,
                initial_random=args.initial_random,
                batch_size=args.batch_size,
                total_budget=args.budget,
                seed=round_seed,
                prior_memory=None,
                transfer_enabled=False,
            )
            transfer_best, transfer_memory, info = run_offline_bo_with_transfer(
                X,
                y,
                initial_random=args.initial_random,
                batch_size=args.batch_size,
                total_budget=args.budget,
                seed=round_seed,
                prior_memory=prior_memory,
                transfer_enabled=step >= 2,
            )
            prior_memory = baseline_memory

            if step in step_list:
                improvement = baseline_best - transfer_best
                rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "oracle_best": oracle,
                        "baseline_best": baseline_best,
                        "transfer_best": transfer_best,
                        "improvement_vs_baseline": improvement,
                        "transfer_weight_share_mean": info[
                            "transfer_weight_share_mean"
                        ],
                    }
                )
                baseline_by_step[step].append(baseline_best)
                transfer_by_step[step].append(transfer_best)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    logger.info("Wrote %s", args.out)

    summary_rows: list[dict[str, float | int]] = []
    for step in step_list:
        stats = paired_stats(baseline_by_step[step], transfer_by_step[step])
        summary_rows.append(
            {
                "step": step,
                "mean_improvement": stats["mean_improvement"],
                "win_rate": stats["win_rate"],
                "p_value": stats["p_value"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    logger.info(
        "Mean improvement by step (baseline - transfer, positive = transfer better):\n%s",
        summary.to_string(index=False),
    )
    total_improvement = float(summary["mean_improvement"].sum())
    logger.info("Cumulative mean improvement across steps: %.4f eV", total_improvement)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

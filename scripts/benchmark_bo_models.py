#!/usr/bin/env python3
"""Offline BO benchmark: compare surrogate models and acquisition functions.

Loads real placement data from ``results_*/adsorption_energies_detailed.csv``
written by metalsurfer (via ``save_summary_results`` / campaign workflows),
builds X/y, and simulates the BO loop with fixed batch size 10.

Expected CSV input (required columns for reproducible feature extraction):
  molecule, placement_id, conformer_index, face_flip,
  z_fraction, x_abs, y_abs, z_offset,
  quat_w, quat_x, quat_y, quat_z,
  energy_adsorption

The modern saved dataset also includes rich placement/context columns
(``x_abs``, ``y_abs``, ``z_offset``, ``shape``, ``model_name``, etc.), and this
benchmark uses them automatically when present.

Usage:
  python scripts/benchmark_bo_models.py --data-dir results_co2_graphene --out benchmark_bo_results.csv --seeds 5
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from metalsurfer._logging import configure_logging
from metalsurfer.ml.bayesian import (
    score_and_select,
    train_surrogate,
)
from metalsurfer.ml.features import extract_features_from_dataset

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)

BATCH_SIZE = 10
TOTAL_BUDGET = 100
INITIAL_RANDOM = 10  # so 10 batches of 10 = 100
DEFAULT_SURFACE_TYPE = "co2_graphene"
DEFAULT_SMILES = "O=C=O"
REQUIRED_INPUT_COLUMNS = (
    "molecule",
    "placement_id",
    "conformer_index",
    "face_flip",
    "z_fraction",
    "x_abs",
    "y_abs",
    "z_offset",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "energy_adsorption",
)


def _validate_columns(df: pd.DataFrame, csv_path: str) -> None:
    """Validate required columns are present."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.\nFile: {csv_path}")


def _assert_single_setup(df: pd.DataFrame, csv_path: str) -> None:
    """Offline BO benchmark assumes one adsorption setup per CSV (single surface / SMILES)."""
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
    """Require identical feature rows to share the same label energy."""
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


def load_and_prepare_data(
    data_dir: str,
    surface_type: str = DEFAULT_SURFACE_TYPE,
    smiles: str = DEFAULT_SMILES,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load metalsurfer detailed CSV, validate columns, return (X, y, df)."""
    csv_path = os.path.join(data_dir, "adsorption_energies_detailed.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    _validate_columns(df, csv_path)
    if "smiles" not in df.columns:
        df["smiles"] = smiles
    if "surface_id" not in df.columns:
        df["surface_id"] = surface_type
    _assert_single_setup(df, csv_path)
    X, y = extract_features_from_dataset(df, target_column="energy_adsorption")
    return X, y, df


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
    """One BO run with fixed batch size. Uses score_and_select for acquisition.

    Returns (best_after_each_eval, final_best).
    """
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
                surrogate=surrogate,
                n_estimators=100,
                random_state=seed,
            )
            chosen = score_and_select(
                model,
                X,
                batch_size=batch,
                kappa=kappa,
                evaluated_indices=evaluated,
                acquisition=acquisition,
                f_best=current_best if np.isfinite(current_best) else None,
            )
        for i in chosen:
            evaluated.add(i)
            current_best = min(current_best, float(y_arr[i]))
        total_eval += len(chosen)
        best_after_eval.extend([current_best] * len(chosen))

    # Should have exactly total_budget entries unless pool exhausted early
    return best_after_eval, current_best


def main():
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
    ap.add_argument(
        "--out", default="benchmark_bo_results.csv", help="Output metrics CSV"
    )
    ap.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    ap.add_argument("--no-plot", action="store_true", help="Skip writing plot")
    args = ap.parse_args()

    X, y, _ = load_and_prepare_data(
        args.data_dir, surface_type=args.surface_type, smiles=args.smiles
    )
    n = len(X)
    logger.info("Loaded %d placements, %d features", n, X.shape[1])

    configs: list[tuple[str, str, float]] = [
        ("rf", "lcb", 0.0),
        ("rf", "lcb", 1.0),
        ("rf", "lcb", 1.96),
        ("rf", "lcb", 2.5),
        ("rf", "ei", 1.96),
        ("rf", "pi", 1.96),
        ("extra_trees", "lcb", 1.0),
        ("extra_trees", "pi", 1.96),
        # mean-only surrogates (sigma=0 => EI/PI reduce to greedy)
        ("gradient_boost", "lcb", 0.0),
        ("ridge", "lcb", 0.0),
    ]
    rows = []
    for surrogate, acquisition, kappa in configs:
        key = f"{surrogate}_{acquisition}" + (
            f"_k{kappa}" if acquisition == "lcb" else ""
        )
        finals = []
        best20 = []
        best50 = []
        best100 = []
        for seed in range(args.seeds):
            best_curve, final_best = run_offline_bo(
                X,
                y,
                initial_random=INITIAL_RANDOM,
                batch_size=BATCH_SIZE,
                total_budget=TOTAL_BUDGET,
                seed=seed,
                surrogate=surrogate,
                acquisition=acquisition,
                kappa=kappa,
            )
            finals.append(final_best)
            if len(best_curve) >= 20:
                best20.append(float(best_curve[19]))
            if len(best_curve) >= 50:
                best50.append(float(best_curve[49]))
            if len(best_curve) >= 100:
                best100.append(float(best_curve[99]))
        mean_best = float(np.mean(finals))
        std_best = float(np.std(finals))
        rows.append(
            {
                "surrogate": surrogate,
                "acquisition": acquisition,
                "kappa": kappa,
                "mean_best_at_20": float(np.mean(best20)) if best20 else float("nan"),
                "mean_best_at_50": float(np.mean(best50)) if best50 else float("nan"),
                "mean_best_at_100": float(np.mean(best100))
                if best100
                else float("nan"),
                "mean_final_best": mean_best,
                "std_final_best": std_best,
                "n_seeds": args.seeds,
            }
        )
        logger.info("%s: mean best E_ads = %.4f ± %.4f", key, mean_best, std_best)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    logger.info("Wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

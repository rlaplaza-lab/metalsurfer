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
  python scripts/benchmark_bo_models.py --data-dir results_propane_pt111_ni --surface-type propane_pt111_ni --smiles CCC --out benchmark_bo_results_propane.csv
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
from metalsurfer.ml.schema import SCHEMA_VERSION

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
SETUP_IDENTITY_COLUMNS = ("molecule", "smiles", "surface_id")
CONTEXT_IDENTITY_COLUMNS = (
    "context_hash",
    "model_name",
    "fmax",
    "stage1_steps",
    "stage2_steps",
    "seed",
    "device",
    "placement_z_range",
    "placement_z_scale_by_covalent_radius",
    "min_initial_distance",
    "min_contact_ratio",
    "top_layer_tolerance",
    "relax_top_layer",
)


def _validate_input_dataframe(df: pd.DataFrame, csv_path: str) -> None:
    """Validate expected metalsurfer benchmark input schema."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Input dataset is not compatible with metalsurfer BO benchmark.\n"
            f"Missing required columns: {missing}\n"
            f"File: {csv_path}\n"
            "Expected source: results_*/adsorption_energies_detailed.csv "
            "written by metalsurfer save_summary_results/campaign APIs."
        )
    if "schema_version" in df.columns:
        versions = sorted(
            {str(v) for v in df["schema_version"].dropna().astype(str).unique()}
        )
        if versions and any(v != SCHEMA_VERSION for v in versions):
            logger.warning(
                "Dataset schema_version=%s differs from current=%s; "
                "benchmark will attempt best-effort compatibility.",
                ",".join(versions),
                SCHEMA_VERSION,
            )


def _validate_setup_specific_dataset(df: pd.DataFrame, csv_path: str) -> None:
    """Ensure BO benchmark data corresponds to exactly one setup."""
    mixed_cols: list[tuple[str, list[str]]] = []

    for col in SETUP_IDENTITY_COLUMNS:
        if col not in df.columns:
            continue
        values = [str(v) for v in df[col].dropna().astype(str).unique().tolist()]
        if len(values) > 1:
            mixed_cols.append((col, sorted(values)[:5]))

    present_context_cols = [c for c in CONTEXT_IDENTITY_COLUMNS if c in df.columns]
    for col in present_context_cols:
        values = [str(v) for v in df[col].dropna().astype(str).unique().tolist()]
        if len(values) > 1:
            mixed_cols.append((col, sorted(values)[:5]))

    if mixed_cols:
        details = ", ".join(
            f"{col}={vals}" + ("..." if len(vals) == 5 else "")
            for col, vals in mixed_cols
        )
        raise ValueError(
            "Input dataset mixes multiple setups but this BO benchmark is setup-specific. "
            "Split data so each run has a single adsorbate, surface, and relaxation context. "
            f"Mixed columns in {csv_path}: {details}"
        )

    if not present_context_cols:
        logger.warning(
            "Dataset has no context identity columns (%s). Setup-specific validation "
            "will only enforce molecule/smiles/surface_id.",
            ", ".join(CONTEXT_IDENTITY_COLUMNS),
        )


def _assert_feature_energy_injective(X: pd.DataFrame, y: pd.Series) -> None:
    """Ensure feature vectors map to a single adsorption energy."""
    if X.empty:
        raise ValueError("Input dataset has zero rows after feature extraction.")

    feature_cols = list(X.columns)
    check_df = X.copy()
    check_df["_energy_adsorption"] = pd.Series(y).to_numpy()
    grouped = check_df.groupby(feature_cols, dropna=False)["_energy_adsorption"]
    conflicts = grouped.nunique(dropna=False)
    n_conflicts = int((conflicts > 1).sum())
    if n_conflicts > 0:
        conflict_examples = []
        conflict_keys = list(conflicts[conflicts > 1].index[:3])
        for key in conflict_keys:
            if not isinstance(key, tuple):
                key = (key,)
            mask = (pd.Series(key, index=feature_cols) == X).all(axis=1)
            energies = sorted(
                {float(v) for v in check_df.loc[mask, "_energy_adsorption"]}
            )
            pairs = [
                f"{feature_cols[i]}={key[i]}" for i in range(min(3, len(feature_cols)))
            ]
            conflict_examples.append(f"[{', '.join(pairs)}] -> energies={energies}")
        detail_msg = "; ".join(conflict_examples)
        raise ValueError(
            "Input dataset violates injectivity for BO benchmark: "
            f"{n_conflicts} feature vector(s) map to multiple "
            "energy_adsorption values. Regenerate or deduplicate "
            "adsorption_energies_detailed.csv to ensure deterministic "
            f"feature->target mapping. Example conflicts: {detail_msg}"
        )


def load_and_prepare_data(
    data_dir: str,
    surface_type: str = DEFAULT_SURFACE_TYPE,
    smiles: str = DEFAULT_SMILES,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load metalsurfer detailed CSV, validate columns, return (X, y, df)."""
    csv_path = os.path.join(data_dir, "adsorption_energies_detailed.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Run a placement-generation script first to create {csv_path}"
        )
    df = pd.read_csv(csv_path)
    _validate_input_dataframe(df, csv_path)
    if "smiles" not in df.columns:
        logger.info(
            "Input has no 'smiles' column; using --smiles=%s for all rows.", smiles
        )
        df["smiles"] = smiles
    if "surface_id" not in df.columns:
        logger.info(
            "Input has no 'surface_id' column; using --surface-type=%s for all rows.",
            surface_type,
        )
        df["surface_id"] = surface_type
    _validate_setup_specific_dataset(df, csv_path)
    X, y = extract_features_from_dataset(df, target_column="energy_adsorption")
    _assert_feature_energy_injective(X, y)
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

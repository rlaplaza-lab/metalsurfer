#!/usr/bin/env python3
"""Focused 2026-10 BO / transfer setting sweeps for defaults + wave-100 figure.

Proxy track (init≈10, batch=5, budget=100): cold BO on step 1; transfer on
steps 2--7. Wave track: step-2 init/batch=100 for top proxy combos.

Writes under ``bo_benchmark_report/uma-s-1p2/sweeps_2026-10/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent
_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
os.chdir(_ROOT)

import benchmark_bo as bb

from metalsurfer import configure_logging
from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.models import BOStepMemory, windowed_bo_step_memories

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)
# Keep sweep progress readable; per-fit surrogate logs are too chatty.
logging.getLogger("metalsurfer.ml.bayesian").setLevel(logging.WARNING)
logging.getLogger("metalsurfer.ml.features").setLevel(logging.WARNING)

DEFAULT_DATA_DIR = bb.DEFAULT_DATA_DIR
DEFAULT_SEEDS = 10
OUT_DIR = Path("bo_benchmark_report/uma-s-1p2/sweeps_2026-10")

SURROGATES = ("gradient_boost", "extra_trees", "ensemble")
ACQUISITIONS = ("ei", "pi")
WEIGHT_CAPS = (0.25, 0.35, 0.50)
EXPLORE_FRACS = (0.0, 0.2)

WAVE_INIT = 100
WAVE_BATCH = 100


def _cfg_label(
    surrogate: str,
    acquisition: str,
    *,
    weight_cap: float | None = None,
    exploration_fraction: float | None = None,
) -> str:
    base = f"{surrogate}_{acquisition}"
    if weight_cap is None:
        return base
    return f"{base}_w{weight_cap:g}_e{exploration_fraction:g}"


def _proxy_config() -> AdsorptionConfig:
    return bb._REPLAY


def _wave_config() -> AdsorptionConfig:
    # Exhaust ~971-pool with waves of 100: init 100 + 9 batches of 100.
    return AdsorptionConfig(
        bo=BOConfig(initial_random=WAVE_INIT, batch_size=WAVE_BATCH, total_budget=9)
    )


def _xfer_kwargs(
    *,
    weight_cap: float,
    exploration_fraction: float,
) -> dict[str, float | int]:
    kw = dict(bb.TRANSFER_KWARGS)
    kw["weight_cap"] = float(weight_cap)
    kw["exploration_fraction"] = float(exploration_fraction)
    return kw


def _aurc50(curves: list[list[float]], oracle: float) -> float:
    return bb._mean_aurc(curves, oracle)


def run_proxy_cold_step1(data_dir: str, *, seeds: int, out_dir: Path) -> pd.DataFrame:
    """Step-1 cold BO grid: surrogate × acquisition."""
    out = out_dir / "proxy_step1_cold.csv"
    if out.is_file():
        logger.info("Cache hit %s", out)
        return pd.read_csv(out)

    X, y = bb.load_pool(data_dir, step=1)
    oracle = float(y.min())
    cfg = _proxy_config()
    rows: list[dict[str, Any]] = []
    for surrogate in SURROGATES:
        for acquisition in ACQUISITIONS:
            label = _cfg_label(surrogate, acquisition)
            logger.info("Proxy cold step-1 %s", label)
            curves = [
                bb._run_bo(
                    X,
                    y,
                    seed,
                    config=cfg,
                    surrogate=surrogate,
                    acquisition=acquisition,
                ).curve
                for seed in range(seeds)
            ]
            rows.append(
                {
                    "config": label,
                    "surrogate": surrogate,
                    "acquisition": acquisition,
                    "n_pool": len(X),
                    "oracle_best": oracle,
                    "aurc_50": _aurc50(curves, oracle),
                    "regret_at_20": float(
                        np.mean([bb._curve_at(c, 20) for c in curves]) - oracle
                    ),
                    "regret_at_30": float(
                        np.mean([bb._curve_at(c, 30) for c in curves]) - oracle
                    ),
                    "regret_at_50": float(
                        np.mean([bb._curve_at(c, 50) for c in curves]) - oracle
                    ),
                    "regret_at_100": float(
                        np.mean([bb._curve_at(c, 100) for c in curves]) - oracle
                    ),
                }
            )
            _append_curves(
                out_dir / "proxy_step1_cold_curves.csv",
                [
                    {
                        "config": label,
                        "eval_count": int(r["eval_count"]),
                        "mean_best": float(r["mean_best"]),
                        "std_best": float(r["std_best"]),
                        "oracle_best": oracle,
                    }
                    for _, r in bb._aggregate(curves, cfg).iterrows()
                ],
            )
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    logger.info("Wrote %s", out)
    return df


def _append_curves(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    frame = pd.DataFrame(rows)
    if path.is_file():
        prev = pd.read_csv(path)
        # drop overlapping configs then concat
        configs = set(frame["config"].unique())
        prev = prev[~prev["config"].isin(configs)]
        frame = pd.concat([prev, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def run_proxy_transfer(
    data_dir: str,
    *,
    seeds: int,
    out_dir: Path,
    surrogates: tuple[str, ...] = SURROGATES,
    acquisitions: tuple[str, ...] = ACQUISITIONS,
    weight_caps: tuple[float, ...] = WEIGHT_CAPS,
    explore_fracs: tuple[float, ...] = EXPLORE_FRACS,
) -> pd.DataFrame:
    """Transfer grid on steps 2--7 under the microbatch proxy schedule."""
    summary_path = out_dir / "proxy_transfer_summary.csv"
    curves_path = out_dir / "proxy_transfer_curves.csv"
    done: set[str] = set()
    if summary_path.is_file():
        existing = pd.read_csv(summary_path)
        done = set(existing["config"].astype(str))
        logger.info("Resuming transfer sweep; %d configs cached", len(done))
    else:
        existing = pd.DataFrame()

    steps = bb.list_steps(data_dir)
    transfer_steps = [s for s in steps if s >= 2]
    cfg = _proxy_config()
    new_summaries: list[dict[str, Any]] = []

    for surrogate in surrogates:
        for acquisition in acquisitions:
            for weight_cap in weight_caps:
                for explore in explore_fracs:
                    label = _cfg_label(
                        surrogate,
                        acquisition,
                        weight_cap=weight_cap,
                        exploration_fraction=explore,
                    )
                    if label in done:
                        continue
                    logger.info("Proxy transfer %s", label)
                    xfer_kw = _xfer_kwargs(
                        weight_cap=weight_cap, exploration_fraction=explore
                    )
                    baseline_curves: dict[int, list[list[float]]] = {
                        s: [] for s in transfer_steps
                    }
                    transfer_curves: dict[int, list[list[float]]] = {
                        s: [] for s in transfer_steps
                    }
                    oracles: dict[int, float] = {}
                    pool_sizes: dict[int, int] = {}

                    for seed in range(seeds):
                        memories: list[BOStepMemory] = []
                        for step in steps:
                            X, y = bb.load_pool(data_dir, step=step)
                            pool_sizes[step] = len(X)
                            oracle = float(y.min())
                            oracles[step] = oracle
                            rs = seed + step * 101
                            prior = (
                                windowed_bo_step_memories(
                                    memories, window=bb.TRANSFER_WINDOW
                                )
                                if step >= 2
                                else None
                            )
                            bl, _, _ = bb._run_bo_transfer(
                                X,
                                y,
                                rs,
                                prior=None,
                                transfer=False,
                                config=cfg,
                                surrogate=surrogate,
                                acquisition=acquisition,
                            )
                            tr, tr_mem, _share = bb._run_bo_transfer(
                                X,
                                y,
                                rs,
                                prior=prior,
                                transfer=step >= 2,
                                config=cfg,
                                surrogate=surrogate,
                                acquisition=acquisition,
                                transfer_kwargs=xfer_kw,
                            )
                            del _share
                            memories.append(tr_mem)
                            if step in transfer_steps:
                                baseline_curves[step].append(bl.curve)
                                transfer_curves[step].append(tr.curve)

                    curve_rows: list[dict[str, Any]] = []
                    for step in transfer_steps:
                        oracle = oracles[step]
                        bl_c = baseline_curves[step]
                        tr_c = transfer_curves[step]
                        bl_aurc = _aurc50(bl_c, oracle)
                        tr_aurc = _aurc50(tr_c, oracle)
                        row: dict[str, Any] = {
                            "config": label,
                            "surrogate": surrogate,
                            "acquisition": acquisition,
                            "weight_cap": weight_cap,
                            "exploration_fraction": explore,
                            "step": step,
                            "n_pool": pool_sizes[step],
                            "oracle_best": oracle,
                            "baseline_aurc_50": bl_aurc,
                            "transfer_aurc_50": tr_aurc,
                            "aurc_improvement": bl_aurc - tr_aurc,
                        }
                        for ep in (20, 30, 50, 100):
                            bl_ep = float(np.mean([bb._curve_at(c, ep) for c in bl_c]))
                            tr_ep = float(np.mean([bb._curve_at(c, ep) for c in tr_c]))
                            row[f"baseline_regret_at_{ep}"] = bl_ep - oracle
                            row[f"transfer_regret_at_{ep}"] = tr_ep - oracle
                            row[f"improvement_at_{ep}"] = (bl_ep - oracle) - (
                                tr_ep - oracle
                            )
                        new_summaries.append(row)
                        for variant, curves in (
                            ("baseline", bl_c),
                            ("transfer", tr_c),
                        ):
                            for _, agg in bb._aggregate(curves, cfg).iterrows():
                                curve_rows.append(
                                    {
                                        "config": label,
                                        "step": step,
                                        "variant": variant,
                                        "eval_count": int(agg["eval_count"]),
                                        "mean_best": float(agg["mean_best"]),
                                        "std_best": float(agg["std_best"]),
                                        "oracle_best": oracle,
                                    }
                                )
                    _append_curves(curves_path, curve_rows)
                    # persist after each config for resume
                    block = pd.DataFrame(
                        [r for r in new_summaries if r["config"] == label]
                    )
                    if summary_path.is_file():
                        prev = pd.read_csv(summary_path)
                        prev = prev[prev["config"] != label]
                        pd.concat([prev, block], ignore_index=True).to_csv(
                            summary_path, index=False
                        )
                    else:
                        block.to_csv(summary_path, index=False)
                    done.add(label)

    df = pd.read_csv(summary_path)
    logger.info("Wrote %s (%d rows)", summary_path, len(df))
    return df


def run_wave_step2(
    data_dir: str,
    *,
    seeds: int,
    out_dir: Path,
    combos: list[dict[str, Any]],
) -> pd.DataFrame:
    """Wave-100 cold BO / transfer / random on step 2 for selected combos."""
    out = out_dir / "wave_step2_curves.csv"
    summary_out = out_dir / "wave_step2_summary.csv"
    done: set[str] = set()
    if out.is_file():
        done = set(pd.read_csv(out)["config"].astype(str))

    X, y = bb.load_pool(data_dir, step=2)
    X1, y1 = bb.load_pool(data_dir, step=1)
    oracle = float(y.min())
    cfg = _wave_config()
    n = len(y)
    eval_grid = [min(WAVE_INIT, n)]
    while eval_grid[-1] < n:
        eval_grid.append(min(eval_grid[-1] + WAVE_BATCH, n))

    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    y1_arr = np.asarray(y1, dtype=float).ravel()
    best_i = int(np.argmin(y1_arr))
    prior_template = BOStepMemory(
        observed_X_rows=X1.to_dict(orient="records"),
        observed_y=y1_arr.tolist(),
        best_energy=float(y1_arr[best_i]),
        best_X_row=X1.iloc[best_i].to_dict(),
        step_ages=[0] * len(X1),
    )
    wave_max_evals = int(eval_grid[-1])

    for combo in combos:
        label = combo["config"]
        if label in done:
            logger.info("Wave cache hit %s", label)
            continue
        surrogate = combo["surrogate"]
        acquisition = combo["acquisition"]
        weight_cap = float(combo.get("weight_cap", 0.35))
        explore = float(combo.get("exploration_fraction", 0.2))
        xfer_kw = _xfer_kwargs(weight_cap=weight_cap, exploration_fraction=explore)
        logger.info("Wave step-2 %s", label)

        bo_curves = np.full((seeds, len(eval_grid)), np.nan)
        xfer_curves = np.full((seeds, len(eval_grid)), np.nan)
        rand_curves = np.full((seeds, len(eval_grid)), np.nan)

        for seed in range(seeds):
            prior = prior_template
            bl = bb._run_bo_transfer(
                X,
                y,
                seed,
                prior=None,
                transfer=False,
                config=cfg,
                surrogate=surrogate,
                acquisition=acquisition,
                max_evals=wave_max_evals,
            )[0]
            tr = bb._run_bo_transfer(
                X,
                y,
                seed + 1000,
                prior=prior,
                transfer=True,
                config=cfg,
                surrogate=surrogate,
                acquisition=acquisition,
                transfer_kwargs=xfer_kw,
                max_evals=wave_max_evals,
            )[0]
            rnd = bb._run_random(
                X, y, seed + 10_000, config=cfg, max_evals=wave_max_evals
            )

            for wi, n_eval in enumerate(eval_grid):
                bo_curves[seed, wi] = bb._curve_at(bl.curve, n_eval)
                xfer_curves[seed, wi] = bb._curve_at(tr.curve, n_eval)
                rand_curves[seed, wi] = bb._curve_at(rnd.curve, n_eval)

        xs = np.asarray(eval_grid, dtype=int)
        for variant, curves in (
            ("cold_bo", bo_curves),
            ("transfer_bo", xfer_curves),
            ("random_search", rand_curves),
        ):
            y_mean = np.nanmean(curves, axis=0)
            y_std = np.nanstd(curves, axis=0)
            for x, ym, ys in zip(xs, y_mean, y_std, strict=True):
                curve_rows.append(
                    {
                        "config": label,
                        "variant": variant,
                        "eval_count": int(x),
                        "mean_best": float(ym),
                        "std_best": float(ys),
                        "oracle_best": oracle,
                    }
                )
            # early-wave regrets
            i100 = int(np.where(xs == 100)[0][0]) if 100 in xs else 0
            i200 = int(np.where(xs == 200)[0][0]) if 200 in xs else min(1, len(xs) - 1)
            summary_rows.append(
                {
                    "config": label,
                    "variant": variant,
                    "surrogate": surrogate,
                    "acquisition": acquisition,
                    "weight_cap": weight_cap,
                    "exploration_fraction": explore,
                    "regret_at_100": float(y_mean[i100] - oracle),
                    "regret_at_200": float(y_mean[i200] - oracle),
                    "regret_final": float(y_mean[-1] - oracle),
                }
            )

        # flush incrementally
        block_c = pd.DataFrame([r for r in curve_rows if r["config"] == label])
        block_s = pd.DataFrame([r for r in summary_rows if r["config"] == label])
        if out.is_file():
            prev = pd.read_csv(out)
            prev = prev[prev["config"] != label]
            pd.concat([prev, block_c], ignore_index=True).to_csv(out, index=False)
        else:
            block_c.to_csv(out, index=False)
        if summary_out.is_file():
            prev = pd.read_csv(summary_out)
            prev = prev[prev["config"] != label]
            pd.concat([prev, block_s], ignore_index=True).to_csv(
                summary_out, index=False
            )
        else:
            block_s.to_csv(summary_out, index=False)

    return pd.read_csv(summary_out)


def analyze_and_decide(out_dir: Path) -> dict[str, Any]:
    """Apply the plan decision rule; write decision.json + report.md."""
    cold = pd.read_csv(out_dir / "proxy_step1_cold.csv")
    xfer = pd.read_csv(out_dir / "proxy_transfer_summary.csv")

    default_cold = "gradient_boost_ei"
    cold_default_aurc = float(
        cold.loc[cold["config"] == default_cold, "aurc_50"].iloc[0]
    )
    cold_best = cold.sort_values("aurc_50").iloc[0]
    cold_delta = cold_default_aurc - float(cold_best["aurc_50"])

    # Mean AURC improvement over steps 2--7 per config
    grp = (
        xfer.groupby("config", as_index=False)
        .agg(
            mean_aurc_improvement=("aurc_improvement", "mean"),
            mean_transfer_aurc=("transfer_aurc_50", "mean"),
            mean_baseline_aurc=("baseline_aurc_50", "mean"),
            mean_impr_20=("improvement_at_20", "mean"),
            n_steps=("step", "count"),
            surrogate=("surrogate", "first"),
            acquisition=("acquisition", "first"),
            weight_cap=("weight_cap", "first"),
            exploration_fraction=("exploration_fraction", "first"),
        )
        .sort_values("mean_aurc_improvement", ascending=False)
    )
    default_xfer_label = _cfg_label(
        "gradient_boost", "ei", weight_cap=0.35, exploration_fraction=0.2
    )
    if default_xfer_label not in set(grp["config"]):
        # fall back to any gb_ei_w0.35_e0.2
        matches = grp[grp["config"].str.startswith("gradient_boost_ei_w0.35")]
        default_xfer_label = (
            str(matches.iloc[0]["config"])
            if len(matches)
            else str(grp.iloc[-1]["config"])
        )
    default_row = grp.loc[grp["config"] == default_xfer_label].iloc[0]
    best_row = grp.iloc[0]
    xfer_delta = float(best_row["mean_aurc_improvement"]) - float(
        default_row["mean_aurc_improvement"]
    )

    # Count steps where best worsens regret@20 vs default transfer
    best_cfg = str(best_row["config"])
    def_cfg = str(default_row["config"])
    best_steps = xfer[xfer["config"] == best_cfg].set_index("step")
    def_steps = xfer[xfer["config"] == def_cfg].set_index("step")
    worsen = 0
    for step in best_steps.index:
        if step not in def_steps.index:
            continue
        if (
            float(best_steps.loc[step, "transfer_regret_at_20"])
            > float(def_steps.loc[step, "transfer_regret_at_20"]) + 1e-9
        ):
            worsen += 1

    adopt_cold = cold_delta >= 0.015
    adopt_xfer = xfer_delta >= 0.015 and worsen < 3

    # Wave ranking if present
    wave_pick = None
    wave_path = out_dir / "wave_step2_summary.csv"
    if wave_path.is_file():
        wave = pd.read_csv(wave_path)
        # prefer transfer variant; score = -(reg@100+reg@200) + gap vs cold/random
        piv = wave.pivot_table(
            index="config",
            columns="variant",
            values=["regret_at_100", "regret_at_200", "regret_final"],
        )
        scores = []
        for cfg in piv.index:
            try:
                r100_t = float(piv.loc[cfg, ("regret_at_100", "transfer_bo")])
                r200_t = float(piv.loc[cfg, ("regret_at_200", "transfer_bo")])
                r100_c = float(piv.loc[cfg, ("regret_at_100", "cold_bo")])
                r100_r = float(piv.loc[cfg, ("regret_at_100", "random_search")])
                rfin = float(piv.loc[cfg, ("regret_final", "transfer_bo")])
            except KeyError:
                continue
            gap_vs_cold = r100_c - r100_t
            gap_vs_rand = r100_r - r100_t
            score = gap_vs_cold + gap_vs_rand - 0.25 * (r100_t + r200_t) - 0.1 * rfin
            scores.append((score, cfg, r100_t, r200_t, gap_vs_cold, gap_vs_rand, rfin))
        scores.sort(reverse=True)
        if scores:
            wave_pick = {
                "config": scores[0][1],
                "score": scores[0][0],
                "regret_at_100": scores[0][2],
                "regret_at_200": scores[0][3],
                "gap_vs_cold_100": scores[0][4],
                "gap_vs_rand_100": scores[0][5],
                "regret_final": scores[0][6],
            }

    decision = {
        "cold_default": default_cold,
        "cold_default_aurc_50": cold_default_aurc,
        "cold_best": {
            "config": str(cold_best["config"]),
            "surrogate": str(cold_best["surrogate"]),
            "acquisition": str(cold_best["acquisition"]),
            "aurc_50": float(cold_best["aurc_50"]),
            "delta_vs_default": cold_delta,
        },
        "adopt_cold_default_change": adopt_cold,
        "transfer_default": def_cfg,
        "transfer_default_mean_aurc_improvement": float(
            default_row["mean_aurc_improvement"]
        ),
        "transfer_best": {
            "config": best_cfg,
            "surrogate": str(best_row["surrogate"]),
            "acquisition": str(best_row["acquisition"]),
            "weight_cap": float(best_row["weight_cap"]),
            "exploration_fraction": float(best_row["exploration_fraction"]),
            "mean_aurc_improvement": float(best_row["mean_aurc_improvement"]),
            "delta_vs_default": xfer_delta,
            "steps_worsened_regret20": worsen,
        },
        "adopt_transfer_default_change": adopt_xfer,
        "wave_pick": wave_pick,
        "proxy_transfer_ranking": grp.head(10).to_dict(orient="records"),
        "proxy_cold_ranking": cold.sort_values("aurc_50")
        .head(10)
        .to_dict(orient="records"),
    }
    (out_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n")

    lines = [
        "# BO / transfer sweeps 2026-10",
        "",
        "## Cold BO (step 1, proxy)",
        f"- Default `{default_cold}` AURC@50 = {cold_default_aurc:.4f}",
        f"- Best `{cold_best['config']}` AURC@50 = {float(cold_best['aurc_50']):.4f} "
        f"(Δ={cold_delta:+.4f})",
        f"- Adopt default change: **{adopt_cold}** (need Δ≥0.015)",
        "",
        "## Transfer (steps 2–7, proxy)",
        f"- Default `{def_cfg}` mean AURC improvement = "
        f"{float(default_row['mean_aurc_improvement']):.4f}",
        f"- Best `{best_cfg}` mean AURC improvement = "
        f"{float(best_row['mean_aurc_improvement']):.4f} (Δ={xfer_delta:+.4f})",
        f"- Steps with worse transfer regret@20 vs default: {worsen}",
        f"- Adopt default change: **{adopt_xfer}** (need Δ≥0.015 and worsen<3)",
        "",
    ]
    if wave_pick:
        lines += [
            "## Wave-100 step-2 pick (figure panel B)",
            f"- `{wave_pick['config']}` score={wave_pick['score']:.4f}",
            f"- regret@100={wave_pick['regret_at_100']:.4f}, "
            f"@200={wave_pick['regret_at_200']:.4f}, "
            f"final={wave_pick['regret_final']:.4f}",
            f"- gap vs cold@100={wave_pick['gap_vs_cold_100']:+.4f}, "
            f"vs random@100={wave_pick['gap_vs_rand_100']:+.4f}",
            "",
        ]
    lines.append("## Top proxy transfer configs")
    for _, r in grp.head(8).iterrows():
        lines.append(
            f"- `{r['config']}`: mean ΔAURC={r['mean_aurc_improvement']:+.4f}, "
            f"mean Δregret@20={r['mean_impr_20']:+.4f}"
        )
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")
    logger.info("Wrote decision.json and report.md")
    return decision


def _top_wave_combos(xfer_summary: pd.DataFrame, *, k: int = 4) -> list[dict[str, Any]]:
    grp = (
        xfer_summary.groupby("config", as_index=False)
        .agg(
            mean_aurc_improvement=("aurc_improvement", "mean"),
            surrogate=("surrogate", "first"),
            acquisition=("acquisition", "first"),
            weight_cap=("weight_cap", "first"),
            exploration_fraction=("exploration_fraction", "first"),
        )
        .sort_values("mean_aurc_improvement", ascending=False)
    )
    combos = []
    # Always include current package default and the proxy cold-BO winner (PI).
    for surrogate, acquisition in (("gradient_boost", "ei"), ("gradient_boost", "pi")):
        combos.append(
            {
                "config": _cfg_label(
                    surrogate,
                    acquisition,
                    weight_cap=0.35,
                    exploration_fraction=0.2,
                ),
                "surrogate": surrogate,
                "acquisition": acquisition,
                "weight_cap": 0.35,
                "exploration_fraction": 0.2,
            }
        )
    for _, r in grp.head(k).iterrows():
        c = {
            "config": str(r["config"]),
            "surrogate": str(r["surrogate"]),
            "acquisition": str(r["acquisition"]),
            "weight_cap": float(r["weight_cap"]),
            "exploration_fraction": float(r["exploration_fraction"]),
        }
        if c["config"] not in {x["config"] for x in combos}:
            combos.append(c)
    return combos[: k + 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--stage",
        choices=("all", "proxy-cold", "proxy-transfer", "wave", "analyze"),
        default="all",
    )
    # Faster first pass: only default transfer knobs × surrogate/acq, then expand.
    parser.add_argument(
        "--transfer-phase",
        choices=("surrogate-acq", "knobs", "full"),
        default="full",
        help="surrogate-acq: w=0.35,e=0.2 only; knobs: best 2×weight×explore; full: all",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("all", "proxy-cold"):
        run_proxy_cold_step1(args.data_dir, seeds=args.seeds, out_dir=out_dir)

    if args.stage in ("all", "proxy-transfer"):
        if args.transfer_phase == "surrogate-acq":
            run_proxy_transfer(
                args.data_dir,
                seeds=args.seeds,
                out_dir=out_dir,
                weight_caps=(0.35,),
                explore_fracs=(0.2,),
            )
        elif args.transfer_phase == "knobs":
            # Require surrogate-acq phase first
            summary = pd.read_csv(out_dir / "proxy_transfer_summary.csv")
            top = (
                summary.groupby("config", as_index=False)["aurc_improvement"]
                .mean()
                .sort_values("aurc_improvement", ascending=False)
                .head(2)
            )
            # parse surrogate/acq from those rows
            detail = summary[summary["config"].isin(top["config"])]
            sur_acq = detail[["surrogate", "acquisition"]].drop_duplicates()
            surrogates = tuple(sur_acq["surrogate"].unique())
            acquisitions = tuple(sur_acq["acquisition"].unique())
            run_proxy_transfer(
                args.data_dir,
                seeds=args.seeds,
                out_dir=out_dir,
                surrogates=surrogates,
                acquisitions=acquisitions,
                weight_caps=WEIGHT_CAPS,
                explore_fracs=EXPLORE_FRACS,
            )
        else:
            run_proxy_transfer(args.data_dir, seeds=args.seeds, out_dir=out_dir)

    if args.stage in ("all", "wave"):
        xfer = pd.read_csv(out_dir / "proxy_transfer_summary.csv")
        combos = _top_wave_combos(xfer, k=4)
        (out_dir / "wave_combos.json").write_text(json.dumps(combos, indent=2) + "\n")
        run_wave_step2(args.data_dir, seeds=args.seeds, out_dir=out_dir, combos=combos)

    if args.stage in ("all", "analyze"):
        analyze_and_decide(out_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Benchmark TorchSim optimization throughput for metalsurfer.

Goal
----
Provide a reproducible micro-benchmark that helps choose sensible defaults for
TorchSim optimization settings in ``AdsorptionConfig`` by comparing throughput
and convergence under realistic workload shapes.

Workloads
---------
1. **Isolated-molecule optimization** - small systems, many conformers.
2. **Slab+adsorbate optimization** - larger systems (ragged sizes), with
   frozen sub-surface constraints.  Closer to real screening workloads.

Each workload runs combinations of:
  - FIRE vs L-BFGS optimizer
  - Sequential vs autobatched
  - Different ``autobatcher_max_memory_padding`` values

Outputs
-------
Writes a CSV with timing, convergence, and GPU memory columns so default
settings can be data-driven.

Usage (GPU):
  python benchmarks/benchmark_torchsim.py

Usage (CPU, slower but no GPU needed):
  python benchmarks/benchmark_torchsim.py --device cpu

Pass --n-conformers, --n-placements and --n-steps to control workload size.
"""

import argparse
import csv
import logging
import os
import time

import numpy as np
from ase.build import fcc111, molecule

from metalsurfer._logging import configure_logging
from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import (
    clear_autobatcher_cache,
    optimize_adsorbate_slab_batched,
    optimize_isolated_molecules_batched,
    setup_single_model,
)

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)


def _gpu_mem_mb() -> float | None:
    """Return current GPU memory allocated in MB, or None if unavailable."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**2
    except Exception:
        pass
    return None


def _peak_gpu_mem_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return None


def _reset_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _make_conformers(n: int):
    """Build n small molecule Atoms objects suitable for optimisation."""
    templates = ["H2O", "CH4", "NH3", "H2"]
    conformers = []
    for i in range(n):
        mol = molecule(templates[i % len(templates)])
        mol.set_cell([10.0, 10.0, 10.0])
        mol.set_pbc([True, True, True])
        conformers.append(mol)
    return conformers


def _make_slab_adsorbate_systems(n: int):
    """Build n slab+adsorbate Atoms objects with ragged adsorbate sizes."""
    slab = fcc111("Cu", size=(3, 3, 3), vacuum=10.0)
    slab.set_pbc([True, True, True])

    # Mix rigid/simple adsorbates with flexible small organics to better mimic
    # real optimization difficulty (different atom counts + internal torsions).
    adsorbate_templates = [
        molecule("H2"),
        molecule("CO"),
        molecule("CO2"),
        molecule("H2O"),
        molecule("CH4"),
        molecule("NH3"),
        molecule("CH3OH"),  # methanol (small organic)
        molecule("CH3CHO"),  # acetaldehyde (small organic)
        molecule("CH3OCH3"),  # dimethyl ether (more flexible)
        molecule("CH3CH2OH"),  # ethanol (flexible)
    ]
    rng = np.random.default_rng(42)
    systems = []
    for i in range(n):
        ads = adsorbate_templates[i % len(adsorbate_templates)].copy()
        slab_z_max = slab.get_positions()[:, 2].max()
        xy_center = slab.get_cell()[:2, :2].sum(axis=0) / 2
        offset = np.array(
            [
                xy_center[0] + rng.uniform(-1, 1),
                xy_center[1] + rng.uniform(-1, 1),
                slab_z_max + 2.0 + rng.uniform(0, 0.5),
            ]
        )
        ads.set_positions(
            ads.get_positions() - ads.get_positions().mean(axis=0) + offset
        )
        combined = slab + ads
        combined.set_pbc([True, True, True])
        combined.set_cell(slab.get_cell())
        systems.append(combined)

    return systems, slab


def run_isolated_benchmark(
    ts_model,
    conformers,
    *,
    optimizer: str,
    sequential: bool,
    steps: int,
    padding: float,
    device: str,
) -> dict:
    """Run one isolated-molecule benchmark configuration and return metrics."""
    config = AdsorptionConfig(
        optimize_isolated_sequentially=sequential,
        ts_optimizer=optimizer,
        autobatcher_max_memory_padding=padding,
        device=device,
    )

    _reset_peak()
    mem_before = _gpu_mem_mb()
    t0 = time.perf_counter()

    results = optimize_isolated_molecules_batched(
        conformers,
        ts_model,
        fmax=0.05,
        steps=steps,
        config=config,
    )

    elapsed = time.perf_counter() - t0
    mem_after = _gpu_mem_mb()
    peak_mem = _peak_gpu_mem_mb()

    n_converged = sum(1 for r in results if r is not None)

    return {
        "scenario": "isolated",
        "optimizer": optimizer,
        "sequential": sequential,
        "padding": padding,
        "n_systems": len(conformers),
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
        "n_converged": n_converged,
        "gpu_mem_before_mb": round(mem_before, 1) if mem_before is not None else None,
        "gpu_mem_after_mb": round(mem_after, 1) if mem_after is not None else None,
        "gpu_peak_mb": round(peak_mem, 1) if peak_mem is not None else None,
    }


def run_slab_adsorbate_benchmark(
    ts_model,
    systems,
    slab,
    *,
    optimizer: str,
    steps: int,
    padding: float,
    device: str,
) -> dict:
    """Run one slab+adsorbate benchmark configuration and return metrics."""
    config = AdsorptionConfig(
        ts_optimizer=optimizer,
        autobatcher_max_memory_padding=padding,
        device=device,
        stage1_steps=steps,
        stage2_steps=0,
    )

    _reset_peak()
    mem_before = _gpu_mem_mb()
    t0 = time.perf_counter()

    results = optimize_adsorbate_slab_batched(
        systems,
        slab,
        ts_model,
        config=config,
    )

    elapsed = time.perf_counter() - t0
    mem_after = _gpu_mem_mb()
    peak_mem = _peak_gpu_mem_mb()

    n_converged = sum(1 for r in results if r is not None)

    return {
        "scenario": "slab_adsorbate",
        "optimizer": optimizer,
        "sequential": False,
        "padding": padding,
        "n_systems": len(systems),
        "steps": steps,
        "elapsed_s": round(elapsed, 3),
        "n_converged": n_converged,
        "gpu_mem_before_mb": round(mem_before, 1) if mem_before is not None else None,
        "gpu_mem_after_mb": round(mem_after, 1) if mem_after is not None else None,
        "gpu_peak_mb": round(peak_mem, 1) if peak_mem is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="torch device (default: cuda)")
    parser.add_argument("--n-conformers", type=int, default=8)
    parser.add_argument("--n-placements", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument(
        "--model",
        default="uma-s-1p1",
        help="Model name used by setup_single_model (default: %(default)s)",
    )
    parser.add_argument(
        "--out", default="benchmarks/torchsim_benchmark.csv", help="output CSV"
    )
    args = parser.parse_args()
    logger.info("Setting up shared model on %s...", args.device)
    _, ts_model = setup_single_model(args.model, args.device)
    conformers = _make_conformers(args.n_conformers)
    slab_systems, slab = _make_slab_adsorbate_systems(args.n_placements)

    # --- Isolated-molecule configurations ---
    iso_configs = []
    for optimizer in ("fire", "lbfgs"):
        for sequential in (True, False):
            for padding in (0.25, 0.5, 0.8):
                if sequential and padding != 0.5:
                    continue
                iso_configs.append(
                    {
                        "optimizer": optimizer,
                        "sequential": sequential,
                        "padding": padding,
                    }
                )

    rows = []
    total = len(iso_configs)
    for i, cfg in enumerate(iso_configs, 1):
        label = (
            f"isolated: {cfg['optimizer']}, "
            f"{'seq' if cfg['sequential'] else 'batched'}, "
            f"pad={cfg['padding']}"
        )
        logger.info("[%d/%d] %s", i, total, label)
        try:
            row = run_isolated_benchmark(
                ts_model,
                conformers,
                optimizer=cfg["optimizer"],
                sequential=cfg["sequential"],
                steps=args.n_steps,
                padding=cfg["padding"],
                device=args.device,
            )
            rows.append(row)
            logger.info(
                "  -> %.3fs, %d converged", row["elapsed_s"], row["n_converged"]
            )
        except Exception as exc:
            logger.warning("  -> FAILED: %s", exc)
            rows.append(
                {
                    "scenario": "isolated",
                    **cfg,
                    "n_systems": args.n_conformers,
                    "steps": args.n_steps,
                    "elapsed_s": None,
                    "n_converged": 0,
                    "error": str(exc),
                }
            )
        finally:
            clear_autobatcher_cache()

    # --- Slab+adsorbate configurations ---
    slab_configs = []
    for optimizer in ("fire", "lbfgs"):
        for padding in (0.25, 0.5, 0.8):
            slab_configs.append({"optimizer": optimizer, "padding": padding})

    total_slab = len(slab_configs)
    for i, cfg in enumerate(slab_configs, 1):
        label = f"slab+ads: {cfg['optimizer']}, pad={cfg['padding']}"
        logger.info("[%d/%d] %s", i, total_slab, label)
        try:
            row = run_slab_adsorbate_benchmark(
                ts_model,
                [s.copy() for s in slab_systems],
                slab,
                optimizer=cfg["optimizer"],
                steps=args.n_steps,
                padding=cfg["padding"],
                device=args.device,
            )
            rows.append(row)
            logger.info(
                "  -> %.3fs, %d converged", row["elapsed_s"], row["n_converged"]
            )
        except Exception as exc:
            logger.warning("  -> FAILED: %s", exc)
            rows.append(
                {
                    "scenario": "slab_adsorbate",
                    "optimizer": cfg["optimizer"],
                    "sequential": False,
                    "padding": cfg["padding"],
                    "n_systems": args.n_placements,
                    "steps": args.n_steps,
                    "elapsed_s": None,
                    "n_converged": 0,
                    "error": str(exc),
                }
            )
        finally:
            clear_autobatcher_cache()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    all_keys: set[str] = set()
    for r in rows:
        all_keys.update(r.keys())
    fieldnames = sorted(all_keys)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d rows to %s", len(rows), args.out)

    logger.info("\n--- Summary ---")
    for row in rows:
        elapsed = row.get("elapsed_s")
        peak = row.get("gpu_peak_mb")
        logger.info(
            "  [%s] %s %s pad=%.2f : %.3fs, peak=%s MB, converged=%s/%s",
            row.get("scenario", "?"),
            row.get("optimizer", "?"),
            "seq" if row.get("sequential") else "batched",
            row.get("padding", 0),
            elapsed if elapsed else float("nan"),
            peak if peak is not None else "N/A",
            row.get("n_converged", "?"),
            row.get("n_systems", "?"),
        )


if __name__ == "__main__":
    main()

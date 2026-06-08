#!/usr/bin/env python3
"""Run the full offline BO benchmark suite and print a summary report."""

from __future__ import annotations

import argparse
import io
import logging
import os
import subprocess
import sys
import tempfile

from metalsurfer import configure_logging

configure_logging(default_level="INFO")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def _run_script(script: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    cmd = [PYTHON, os.path.join(SCRIPT_DIR, script), *args]
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BO benchmark suite")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Results directory with benchmark CSVs",
    )
    parser.add_argument(
        "--seeds", type=int, default=10, help="Seeds for each benchmark"
    )
    parser.add_argument(
        "--surface-type",
        default="bipyridine_au111_defects_saturation_raw",
    )
    parser.add_argument(
        "--smiles",
        default="n1ccccc1-c2ccccn2",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="bo_bench_") as tmp:
        models_out = os.path.join(tmp, "models.csv")
        models_curves = os.path.join(tmp, "models_curves.csv")
        screening_out = os.path.join(tmp, "screening.csv")
        transfer_out = os.path.join(tmp, "transfer.csv")

        common = [
            "--data-dir",
            args.data_dir,
            "--surface-type",
            args.surface_type,
            "--smiles",
            args.smiles,
            "--seeds",
            str(args.seeds),
        ]

        results: list[tuple[str, subprocess.CompletedProcess[str]]] = []

        results.append(
            (
                "Model comparison (step 3)",
                _run_script(
                    "benchmark_bo_models.py",
                    [
                        *common,
                        "--step",
                        "3",
                        "--out",
                        models_out,
                        "--curves-out",
                        models_curves,
                        "--no-plot",
                    ],
                ),
            )
        )
        results.append(
            (
                "Screening vs random (all steps, default only)",
                _run_script(
                    "benchmark_bo_models.py",
                    [
                        *common,
                        "--all-steps",
                        "--default-only",
                        "--out",
                        screening_out,
                        "--no-plot",
                    ],
                ),
            )
        )
        results.append(
            (
                "Transfer (steps 2-9)",
                _run_script(
                    "benchmark_bo_transfer.py",
                    [
                        *common,
                        "--steps",
                        "2-9",
                        "--out",
                        transfer_out,
                    ],
                ),
            )
        )

        report = io.StringIO()
        report.write("# BO Benchmark Suite Report\n\n")
        report.write(f"Data directory: `{args.data_dir}`\n")
        report.write(f"Seeds per benchmark: {args.seeds}\n\n")

        failed = False
        for title, proc in results:
            report.write(f"## {title}\n\n")
            if proc.returncode != 0:
                failed = True
                report.write("**FAILED**\n\n")
                report.write("```\n")
                report.write(proc.stderr or proc.stdout or "unknown error")
                report.write("\n```\n\n")
                continue
            if proc.stdout:
                report.write("```\n")
                report.write(proc.stdout.strip())
                report.write("\n```\n\n")
            out_path = {
                "Model comparison (step 3)": models_out,
                "Screening vs random (all steps, default only)": screening_out,
                "Transfer (steps 2-9)": transfer_out,
            }[title]
            if os.path.isfile(out_path):
                import pandas as pd

                df = pd.read_csv(out_path)
                report.write("```\n")
                report.write(df.to_string(index=False))
                report.write("\n```\n\n")

        text = report.getvalue()
        print(text)
        if failed:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

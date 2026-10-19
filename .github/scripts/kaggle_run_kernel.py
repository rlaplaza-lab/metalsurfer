#!/usr/bin/env python3
"""Push a Kaggle GPU kernel, wait for completion, and report parsed logs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from kaggle_log_common import parse_kaggle_log, redact_text

# A newly pushed GPU kernel moves through queued/running before COMPLETE.
# Status that is already COMPLETE on the first polls is the previous run
# (a rejected push still exits 0 and leaves that status in place).
_ACTIVE_STATUS_MARKERS = ("queued", "running", "pending")
_STALE_COMPLETE_SECONDS = 300


def run_cmd(cmd: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if completed.stdout:
        print(
            redact_text(completed.stdout),
            end="" if completed.stdout.endswith("\n") else "\n",
        )
    if completed.stderr:
        print(
            redact_text(completed.stderr),
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode, cmd, output=completed.stdout, stderr=completed.stderr
        )
    return completed


def write_kernel_metadata(
    *,
    staging_dir: Path,
    slug: str,
    title: str,
    code_file: Path,
    dataset_sources: list[str],
    timeout_seconds: int,
    machine_shape: str,
) -> str:
    run_cmd(["kaggle", "kernels", "init", "-p", str(staging_dir)])
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    owner = str(metadata.get("id", slug)).split("/")[0]
    metadata["id"] = f"{owner}/{slug}"
    metadata["title"] = title
    metadata["code_file"] = str(code_file.resolve())
    metadata["language"] = "python"
    metadata["kernel_type"] = "script"
    metadata["is_private"] = True
    metadata["enable_gpu"] = True
    metadata["enable_tpu"] = False
    metadata["enable_internet"] = True
    metadata["dataset_sources"] = dataset_sources
    if machine_shape:
        metadata["machine_shape"] = machine_shape
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    push_cmd = ["kaggle", "kernels", "push", "-p", str(staging_dir)]
    if timeout_seconds > 0:
        push_cmd.extend(["--timeout", str(timeout_seconds)])
    pushed = run_cmd(push_cmd)
    return f"{pushed.stdout or ''}\n{pushed.stderr or ''}"


def push_was_rejected(output: str) -> bool:
    """True when the Kaggle CLI reported a push failure but exited 0."""
    return "kernel push error" in output.lower()


def advance_kernel_wait(
    status: str,
    *,
    saw_active: bool,
    stale_elapsed: int,
    fetch_seconds: int,
    stale_complete_seconds: int = _STALE_COMPLETE_SECONDS,
) -> tuple[str | None, bool, int]:
    """Advance one status poll.

    Returns ``(terminal, saw_active, stale_elapsed)``. ``terminal`` is
    ``complete``, ``error``, ``cancel``, or ``stale`` when the wait should
    stop, and ``None`` while the new run has not finished.
    """
    lowered = status.lower()
    active_now = any(marker in lowered for marker in _ACTIVE_STATUS_MARKERS)
    if active_now:
        saw_active = True
        stale_elapsed = 0
    if "error" in lowered and not active_now:
        return "error", saw_active, stale_elapsed
    if "cancel" in lowered:
        return "cancel", saw_active, stale_elapsed
    if "complete" in lowered:
        if saw_active:
            return "complete", True, 0
        stale_elapsed += max(fetch_seconds, 0)
        if stale_elapsed >= stale_complete_seconds:
            return "stale", False, stale_elapsed
        return None, False, stale_elapsed
    return None, saw_active, stale_elapsed


def download_kernel_log(slug: str, output_dir: Path) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(["kaggle", "kernels", "output", slug, "-p", str(output_dir)], check=False)
    log_files = sorted(output_dir.glob("*.log"))
    if not log_files:
        log_files = sorted(output_dir.rglob("*.log"))
    return log_files[0] if log_files else None


def wait_for_kernel(slug: str, *, fetch_seconds: int) -> str:
    saw_active = False
    stale_elapsed = 0
    while True:
        completed = run_cmd(["kaggle", "kernels", "status", slug], check=False)
        status = (completed.stdout or completed.stderr or "").strip()
        print(status)
        decision, saw_active, stale_elapsed = advance_kernel_wait(
            status,
            saw_active=saw_active,
            stale_elapsed=stale_elapsed,
            fetch_seconds=fetch_seconds,
        )
        if decision == "stale":
            print(
                "Kernel never left COMPLETE after push; refusing the previous log.",
                file=sys.stderr,
            )
            return decision
        if decision is not None:
            return decision
        if "complete" in status.lower():
            print(
                "Kernel status is still COMPLETE from the previous run; "
                "waiting for the new push to start."
            )
        time.sleep(fetch_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", default="metalsurfergpuci")
    parser.add_argument(
        "--slug-suffix",
        default="",
        help="Appended to --slug and --title (optional suite label)",
    )
    parser.add_argument("--title", default="MetalsurferGpuCI")
    parser.add_argument("--code-file", default=".github/scripts/kaggle_gpu_runner.py")
    parser.add_argument(
        "--dataset", action="append", default=["rlaplaza/metalsurfercisrc"]
    )
    parser.add_argument("--fetch-seconds", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--log-dir", default="/tmp/kaggle-kernel-log")
    parser.add_argument(
        "--machine-shape",
        default="NvidiaTeslaT4",
        help="Kaggle GPU type (T4 required; P100 sm_60 is incompatible with cu124 torch)",
    )
    args = parser.parse_args()

    suffix = str(args.slug_suffix or "").strip().strip("-")
    slug = f"{args.slug}-{suffix}" if suffix else args.slug
    title = f"{args.title}-{suffix}" if suffix else args.title

    code_file = Path(args.code_file)
    if not code_file.is_file():
        raise SystemExit(f"Kernel code file not found: {code_file}")

    staging = Path("/tmp/kaggle-kernel-push")
    staging.mkdir(parents=True, exist_ok=True)
    staged_code = staging / code_file.name
    staged_code.write_text(code_file.read_text(encoding="utf-8"), encoding="utf-8")

    push_output = write_kernel_metadata(
        staging_dir=staging,
        slug=slug,
        title=title,
        code_file=staged_code,
        dataset_sources=args.dataset,
        timeout_seconds=args.timeout_seconds,
        machine_shape=args.machine_shape,
    )
    if push_was_rejected(push_output):
        print(
            "Kaggle kernel push was rejected; "
            "not treating a previous kernel as this run.",
            file=sys.stderr,
        )
        return 1

    final_status = wait_for_kernel(slug, fetch_seconds=args.fetch_seconds)
    log_path = download_kernel_log(slug, Path(args.log_dir))
    if log_path is not None:
        print(f"::group::Kaggle kernel log ({log_path.name})")
        print(
            redact_text(
                parse_kaggle_log(log_path.read_text(encoding="utf-8", errors="replace"))
            )
        )
        print("::endgroup::")
    else:
        print("No Kaggle kernel .log file found", file=sys.stderr)

    if final_status == "complete":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

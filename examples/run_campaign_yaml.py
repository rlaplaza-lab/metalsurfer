#!/usr/bin/env python3
"""Run a metalsurfer campaign YAML document.

Requires: metalsurfer with MLIP stack (``pip install -e ".[mlip]"``).
Run from the project root so relative ``slab_file`` paths resolve.

Usage::

    python examples/run_campaign_yaml.py examples/ethene_ru_slab_binding_energy.yaml
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from metalsurfer import configure_logging, load_campaign_yaml, run_campaign


def _resolve_device(requested: str) -> str:
    if requested != "cuda":
        return requested
    try:
        import torch
    except ImportError:
        print("torch not installed; falling back to device=cpu", file=sys.stderr)
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    print(
        "CUDA not available; falling back to device=cpu "
        "(set config.device explicitly in YAML to silence this).",
        file=sys.stderr,
    )
    return "cpu"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "yaml_path",
        type=Path,
        help="Path to a campaign YAML file",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip molecules already listed in result CSVs (default: always recompute)",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default=None,
        help="Override config.device (default: YAML value, with CUDA→CPU fallback)",
    )
    args = parser.parse_args(argv)

    configure_logging(default_level="INFO")
    document = load_campaign_yaml(args.yaml_path)
    device = _resolve_device(args.device or document.config.device)
    if device != document.config.device:
        document = replace(document, config=replace(document.config, device=device))

    result = run_campaign(document, skip_existing=args.skip_existing)

    print()
    if hasattr(result, "format_summary"):
        print(
            result.format_summary(
                title=f"Campaign summary ({document.campaign})",
                results_dir=document.results_dir,
            )
        )
    elif hasattr(result, "format_completion"):
        print(
            result.format_completion(
                label=f"Campaign summary ({document.campaign})",
                results_dir=document.results_dir,
            )
        )
    else:
        print(f"Campaign finished: {document.campaign} -> {document.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

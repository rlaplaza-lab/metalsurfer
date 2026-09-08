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

from metalsurfer import (
    BindingCampaignResult,
    SaturationCampaignResult,
    configure_logging,
    load_campaign_yaml,
    run_campaign,
)

# Best-E_ads bands by YAML stem (uma-s-1p2 + oc25 QC). Exclusive ceiling /
# inclusive floor around campaign best (binding) or first committed step best
# (saturation). Observed values are stable across repeated local GPU runs.
_BINDING_BEST_E_ADS_CEILING: dict[str, float] = {
    "ethene_ru_slab_binding_energy": 0.35,  # obs ≈ +0.244 eV
    "h2_ru_slab_binding_energy": -0.05,  # obs ≈ −0.183 eV
    "co2_mof_binding_energy": -0.10,  # obs ≈ −0.21 eV
    "water_cu111_adsorption_bo": -0.20,  # obs ≈ −0.373 eV
}
_BINDING_BEST_E_ADS_FLOOR: dict[str, float] = {
    "ethene_ru_slab_binding_energy": 0.10,
    "h2_ru_slab_binding_energy": -0.35,
    "co2_mof_binding_energy": -0.40,
    "water_cu111_adsorption_bo": -0.55,
}
_SATURATION_STEP1_BEST_E_ADS_CEILING: dict[str, float] = {
    "ethane_cu_saturation": -0.40,  # obs ≈ −0.513 eV
}
_SATURATION_STEP1_BEST_E_ADS_FLOOR: dict[str, float] = {
    "ethane_cu_saturation": -0.70,
}


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


def _validate_best_e_ads_lock(yaml_stem: str, result: object) -> None:
    """Exit non-zero when a known demo leaves its QC best-E_ads band."""
    if isinstance(result, BindingCampaignResult):
        ceiling = _BINDING_BEST_E_ADS_CEILING.get(yaml_stem)
        floor = _BINDING_BEST_E_ADS_FLOOR.get(yaml_stem)
        if ceiling is None:
            return
        if not result.molecule_summaries:
            print("No molecule summaries produced.", file=sys.stderr)
            raise SystemExit(1)
        best = result.molecule_summaries[0].best_adsorption_energy
        if best is None or best >= ceiling:
            print(
                f"Best E_ads regression lock failed for {yaml_stem}: "
                f"expected < {ceiling:.2f} eV, got {best}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if floor is not None and best < floor:
            print(
                f"Best E_ads floor lock failed for {yaml_stem}: "
                f"expected >= {floor:.2f} eV, got {best}.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        return

    if isinstance(result, SaturationCampaignResult):
        ceiling = _SATURATION_STEP1_BEST_E_ADS_CEILING.get(yaml_stem)
        floor = _SATURATION_STEP1_BEST_E_ADS_FLOOR.get(yaml_stem)
        if ceiling is None:
            return
        if not result.runs:
            print("No saturation runs produced.", file=sys.stderr)
            raise SystemExit(1)
        run = result.runs[0]
        first_bound = next((s for s in run.steps if s.n_added > 0), None)
        if first_bound is None:
            print("No committed saturation step found.", file=sys.stderr)
            raise SystemExit(1)
        best = min(u.energy_adsorption for u in first_bound.committed())
        if best >= ceiling:
            print(
                f"First-step best E_ads regression lock failed for {yaml_stem}: "
                f"expected < {ceiling:.2f} eV, got {best:.4f} eV.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        if floor is not None and best < floor:
            print(
                f"First-step best E_ads floor lock failed for {yaml_stem}: "
                f"expected >= {floor:.2f} eV, got {best:.4f} eV.",
                file=sys.stderr,
            )
            raise SystemExit(1)


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

    _validate_best_e_ads_lock(args.yaml_path.stem, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Canonical user-facing strings for CLI entry points (single place for wording)."""

from __future__ import annotations

from pathlib import Path

from ..models import MoleculeCampaignSummary


def format_results_saved_line(results_dir: str) -> str:
    """Return a canonical results output line."""
    return f"Results saved to {Path(results_dir).as_posix()}/ (XYZ, POSCAR, CSV)"


def format_screening_complete(total_configurations: int) -> str:
    """Return a canonical screening completion line."""
    return f"Screening complete: {total_configurations} total configurations"


def format_saturation_complete(
    *,
    label: str,
    n_molecules_at_saturation: int,
    total_steps: int,
    results_dir: str,
) -> str:
    """Return a canonical multi-line saturation completion summary."""
    return "\n".join(
        [
            f"{label} complete:",
            f"  Molecules at saturation: {n_molecules_at_saturation}",
            f"  Total steps: {total_steps}",
            f"  {format_results_saved_line(results_dir)}",
        ]
    )


def format_binding_summary(
    *,
    title: str,
    molecule_summaries: list[MoleculeCampaignSummary],
    results_dir: str,
) -> str:
    """Return a canonical multi-line binding summary block."""
    lines = [
        "=" * 60,
        title,
        "=" * 60,
        "(E_ads = E(slab+molecule) - E(slab) - E(molecule); negative = favorable)",
        "",
    ]
    for item in molecule_summaries:
        if item.best_adsorption_energy is None:
            lines.append(f"  {item.molecule:12s}: (no valid placements)")
            continue
        lines.append(
            f"  {item.molecule:12s}: {item.best_adsorption_energy:+.4f} eV  "
            f"({item.n_valid_placements} valid placements)"
        )
    lines.append("")
    lines.append(format_results_saved_line(results_dir))
    return "\n".join(lines)

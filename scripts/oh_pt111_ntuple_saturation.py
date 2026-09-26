#!/usr/bin/env python3
"""OH n-tuplet saturation on Pt(111), in the spirit of CO/Pt(111) coverage series.

Gunasooriya and Saeys (ACS Catal. 2018, 8, 10225) rank ordered CO cells from
isolated molecules to high coverage by the energy of the whole structure,
including lateral interactions. Metalsurfer cannot enumerate those supercells;
this script shows the part it can do on one fixed FairChem Pt(111) cell:

- isolated OH site preference (atop / bridge / hollow, pre-relax labels)
- sequential coverage growth (differential E_ads, one OH per step)
- n-tuplet commits (``saturation_molecules_per_step=3``): several winners
  packed and relaxed as one composite so lateral interactions enter E_ads

OH is the hydroxyl radical (SMILES ``[OH]``). Energies are versus gas-phase OH,
not an electrochemical cycle. No pH / activity scan: for a single adsorbate
that only shifts Ω uniformly and does not change site order or the
sequential-versus-tuplet comparison at fixed coverage.

Requires: ``pip install -e ".[mlip]"``. Run from the project root::

    python scripts/oh_pt111_ntuple_saturation.py
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from metalsurfer import (
    AdsorptionConfig,
    SaturationRunResult,
    configure_logging,
    results_dir_for,
    run_saturation,
)
from metalsurfer.models import ScreeningResult
from metalsurfer.surface_prep import prepare_substrate

# FairChem mp-126 Pt(111) primitive: 6 layers × 9 surface atoms.
N_SURFACE_ATOMS = 9
MOLECULES = [("[OH]", "OH")]
SITE_ORDER = ("atop", "bridge", "hollow")


def make_base_config() -> AdsorptionConfig:
    """Shared saturation knobs for the isolated / sequential / n-tuplet runs."""
    return AdsorptionConfig(
        num_conformers=1,
        num_placements=250,
    )


def coverage_ml(n_oh: int) -> float:
    """Monolayer fraction relative to the 9-atom Pt(111) top layer."""
    return n_oh / float(N_SURFACE_ATOMS)


def site_type_of(result: ScreeningResult) -> str:
    """Pre-relaxation site label from the placement descriptor."""
    site = result.placement_descriptor.site_type
    return site if site is not None else "unknown"


def _validate_single_run(result: object, *, label: str) -> SaturationRunResult:
    if not isinstance(result, SaturationRunResult):
        raise SystemExit(
            f"{label}: expected SaturationRunResult, got {type(result).__name__}."
        )
    return result


def print_isolated_site_summary(run: SaturationRunResult) -> None:
    """Best E_ads among all valid placements, grouped by initial site type."""
    if not run.steps:
        print("Isolated: no steps recorded.")
        return
    pool = run.steps[0].all_results
    if not pool:
        print("Isolated: no valid placements.")
        return

    by_site: dict[str, list[ScreeningResult]] = defaultdict(list)
    for item in pool:
        by_site[site_type_of(item)].append(item)

    print()
    print("Isolated OH: best E_ads by initial site type (pre-relax label)")
    print(f"{'site':>10}  {'E_ads / eV':>12}  {'n_valid':>8}")
    print("-" * 34)
    ordered = list(SITE_ORDER) + sorted(s for s in by_site if s not in SITE_ORDER)
    for site in ordered:
        group = by_site.get(site) or []
        if not group:
            print(f"{site:>10}  {'-':>12}  {0:>8d}")
            continue
        best = min(group, key=lambda r: r.energy_adsorption)
        print(f"{site:>10}  {best.energy_adsorption:>+12.4f}  {len(group):>8d}")
    overall = min(pool, key=lambda r: r.energy_adsorption)
    print(
        f"{'overall':>10}  {overall.energy_adsorption:>+12.4f}  "
        f"{len(pool):>8d}  ({site_type_of(overall)})"
    )


def print_coverage_table(label: str, run: SaturationRunResult) -> None:
    """Print committed coverage steps: n_OH, θ, E_ads, initial site types."""
    print()
    print(f"{label}: coverage steps")
    header = (
        f"{'step':>4}  {'n_OH':>4}  {'θ / ML':>7}  {'E_ads / eV':>12}  "
        f"{'n_added':>7}  sites"
    )
    print(header)
    print("-" * len(header))
    for step_result in run.steps:
        committed = step_result.committed()
        if not committed:
            print(
                f"{step_result.step:>4}  {'-':>4}  {'-':>7}  "
                f"{'unbound':>12}  {0:>7d}  -"
            )
            continue
        n_oh = step_result.n_molecules_on_slab + len(committed)
        e_ads = committed[0].energy_adsorption
        sites = ",".join(site_type_of(unit) for unit in committed)
        print(
            f"{step_result.step:>4}  {n_oh:>4d}  {coverage_ml(n_oh):>7.3f}  "
            f"{e_ads:>+12.4f}  {len(committed):>7d}  {sites}"
        )
    print(
        f"  at saturation: {run.n_molecules_at_saturation} OH "
        f"(θ = {coverage_ml(run.n_molecules_at_saturation):.3f} ML)"
    )


def sequential_integral_to_n(run: SaturationRunResult, n_oh: int) -> float | None:
    """Sum of differential E_ads for the first *n_oh* sequential commits."""
    energies: list[float] = []
    for step_result in run.steps:
        committed = step_result.committed()
        if not committed:
            continue
        energies.append(committed[0].energy_adsorption)
        if len(energies) >= n_oh:
            return sum(energies[:n_oh])
    return None


def first_tuplet_energy_at_n(run: SaturationRunResult, n_oh: int) -> float | None:
    """Tuplet E_ads from the first step that reaches exactly *n_oh* OH."""
    for step_result in run.steps:
        committed = step_result.committed()
        if not committed:
            continue
        n_after = step_result.n_molecules_on_slab + len(committed)
        if n_after == n_oh and len(committed) == n_oh:
            return committed[0].energy_adsorption
    return None


def print_one_third_ml_comparison(
    sequential: SaturationRunResult,
    ntuple3: SaturationRunResult,
) -> None:
    """Integral of first three sequential steps vs 3-tuplet E_ads at 1/3 ML."""
    print()
    print("1/3 ML comparison (3 OH on 9 Pt)")
    seq_sum = sequential_integral_to_n(sequential, 3)
    tuplet = first_tuplet_energy_at_n(ntuple3, 3)
    if seq_sum is None:
        print("  sequential: did not commit 3 OH; cannot form integral E_ads.")
    else:
        print(f"  sequential integral (sum of 3 differentials): {seq_sum:+.4f} eV")
    if tuplet is None:
        print("  n-tuplet: did not commit a 3-OH composite on step 1.")
    else:
        print(f"  3-tuplet composite E_ads:                    {tuplet:+.4f} eV")
    if seq_sum is not None and tuplet is not None:
        print(
            f"  difference (tuplet − sequential integral):  {tuplet - seq_sum:+.4f} eV"
        )
        print(
            "  (same coverage; tuplet is one joint relaxation, sequential is "
            "greedy differentials)"
        )


def run_campaign(
    *,
    slab,
    config: AdsorptionConfig,
    surface_type: str,
    label: str,
) -> SaturationRunResult | None:
    results_dir = results_dir_for(surface_type).as_posix()
    campaign = run_saturation(
        slab=slab,
        molecules=MOLECULES,
        config=config,
        surface_type=surface_type,
    )
    print()
    if not campaign.runs:
        print(f"{label}: no saturation results.")
        if campaign.failure_summary:
            print(campaign.format_failure_summary())
        return None
    print(
        campaign.format_completion(
            label=label,
            results_dir=results_dir,
        )
    )
    return _validate_single_run(campaign.runs[0], label=label)


def main() -> int:
    configure_logging(default_level="INFO")
    base = make_base_config()
    prep_dir = results_dir_for("oh_pt111_prep").as_posix()

    slab = prepare_substrate(
        bulk_id="mp-126",
        miller_indices=(1, 1, 1),
        supercell=(1, 1, 1),
        config=base,
        results_dir=prep_dir,
        relax_top_layer=True,
    )

    isolated = run_campaign(
        slab=slab,
        config=replace(
            base,
            saturation_molecules_per_step=1,
            saturation_max_steps=1,
            saturation_save_all_placements=True,
            export_placement_provenance=True,
        ),
        surface_type="oh_pt111_isolated",
        label="Isolated OH on Pt(111)",
    )
    if isolated is not None:
        print_isolated_site_summary(isolated)

    sequential = run_campaign(
        slab=slab,
        config=replace(
            base,
            saturation_molecules_per_step=1,
            saturation_max_steps=5,
            saturation_save_all_placements=False,
            export_placement_provenance=True,
        ),
        surface_type="oh_pt111_sequential",
        label="Sequential OH saturation on Pt(111)",
    )
    if sequential is not None:
        print_coverage_table("Sequential", sequential)

    ntuple3 = run_campaign(
        slab=slab,
        config=replace(
            base,
            saturation_molecules_per_step=3,
            saturation_max_steps=2,
            saturation_save_all_placements=False,
            export_placement_provenance=True,
        ),
        surface_type="oh_pt111_ntuple3",
        label="n-tuplet (n=3) OH saturation on Pt(111)",
    )
    if ntuple3 is not None:
        print_coverage_table("n-tuplet n=3", ntuple3)

    if sequential is not None and ntuple3 is not None:
        print_one_third_ml_comparison(sequential, ntuple3)

    print()
    print(
        "Note: initial site types are pre-relax placement labels; inspect "
        "xyz under results_oh_pt111_*/ for post-relax geometry. The 3-tuplet "
        "is a greedy packing + joint relaxation, not an asserted (√3×√3) cell."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

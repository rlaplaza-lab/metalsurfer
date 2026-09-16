#!/usr/bin/env python3
"""Undergraduate tutorial: pH-dependent saturation of a Pt₄ cluster.

We start from a four-atom platinum cluster, then let water and hydroxide
compete for adsorption sites from neutral to basic pH. Metalsurfer has no pH
or Pourbaix setting. Instead we encode pH as reservoir *activities* ``a_i``,
which change the ranking energy used to pick a winner and to stop:

    Ω = E_ads − k_B T ln(a_i p / p°)

``E_ads`` is still the electronic adsorption energy versus each species' own
isolated-molecule reference. This is a reservoir-abundance model, not a full
electrochemical cycle (no electrode potential; OH / H₂O are not on a shared
Pourbaix reference).

Activities used here (same order as the molecule list):

- water:      a = 1                 (pure liquid water)
- hydroxide:  a = 10**(pH - 14)     (from Kw and a(OH-) ≈ 10**(pH-14))

Cluster geometry
----------------

By default the script builds a regular Pt₄ tetrahedron in code. You can also
place any ASE-readable ``.xyz`` file in this directory (``scripts/tutorials/``)
and point ``CLUSTER_XYZ`` at it, or pass ``--cluster my_pt4.xyz`` on the
command line. See ``pt4_tetrahedron.xyz`` for the default geometry on disk.

Requires: ``pip install -e ".[mlip]"``. Runs on CPU by default.

Run from the project root::

    python scripts/tutorials/water_oh_pt4_saturation.py
    python scripts/tutorials/water_oh_pt4_saturation.py --cluster pt4_tetrahedron.xyz
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import replace
from pathlib import Path

import ase.io
from ase import Atoms

from metalsurfer import (
    AdsorptionConfig,
    MultiMolSaturationRunResult,
    configure_logging,
    results_dir_for,
    run_saturation,
)
from metalsurfer.surface_prep import prepare_substrate

# ---------------------------------------------------------------------------
# Tutorial knobs (edit these first)
# ---------------------------------------------------------------------------

# Directory containing this script and optional cluster .xyz files.
TUTORIAL_DIR = Path(__file__).resolve().parent

# ``None`` → build the regular tetrahedron below in code.
# ``"pt4_tetrahedron.xyz"`` (or any other filename here) → load with ASE.
CLUSTER_XYZ: str | None = None

# Students can set DEVICE = "cuda" if they have a GPU. This tutorial is sized
# to finish in a moderate time on CPU.
DEVICE = "cpu"

# Boltzmann constant and SATP pressure used in Ω (same values as metalsurfer).
K_B_EV_PER_K = 8.617e-5
STANDARD_PRESSURE_BAR = 1.0
TEMPERATURE_K = 298.15

# Bulk Pt–Pt nearest-neighbor distance (Å) for the built-in tetrahedron.
PT_PT_ANG = 2.775

# Non-periodic box side length (Å) for built-in clusters and loaded .xyz files
# that arrive without a cell.
VACUUM_BOX_ANG = 20.0

# SMILES + names. ``saturation_activities`` must follow this same order.
MOLECULES: list[tuple[str, str]] = [
    ("O", "water"),
    ("[OH-]", "hydroxide"),
]
MOLECULE_NAMES = tuple(name for _, name in MOLECULES)

# Neutral to strongly basic reservoirs (pH 7–14).
PH_VALUES: tuple[float, ...] = (7.0, 10.0, 14.0)


def build_pt4_tetrahedron() -> Atoms:
    """Return a regular Pt₄ tetrahedron in a large non-periodic box.

    The four vertices of a cube ``(±s, ±s, ±s)`` with an even number of minus
    signs form a regular tetrahedron. Choosing ``s`` so that the edge length
    equals ``PT_PT_ANG`` gives a physically reasonable Pt–Pt distance.
    """
    # Distance between (s, s, s) and (s, -s, -s) is s * sqrt(8).
    scale = PT_PT_ANG / math.sqrt(8.0)
    positions = [
        [scale, scale, scale],
        [scale, -scale, -scale],
        [-scale, scale, -scale],
        [-scale, -scale, scale],
    ]
    cluster = Atoms(
        "Pt4",
        positions=positions,
        cell=[VACUUM_BOX_ANG, VACUUM_BOX_ANG, VACUUM_BOX_ANG],
        pbc=False,
    )
    cluster.center()
    return cluster


def _normalize_nanoparticle_cluster(atoms: Atoms) -> Atoms:
    """Ensure a loaded cluster has a vacuum box and no periodic boundaries."""
    cluster = atoms.copy()
    if cluster.cell.rank < 3 or cluster.cell.volume < 1.0:
        cluster.set_cell([VACUUM_BOX_ANG, VACUUM_BOX_ANG, VACUUM_BOX_ANG])
    cluster.pbc = False
    cluster.center()
    return cluster


def _validate_pt4_cluster(atoms: Atoms, *, source: str) -> None:
    """Exit with a clear message unless *atoms* is exactly four Pt atoms."""
    symbols = atoms.get_chemical_symbols()
    if len(symbols) != 4:
        print(
            f"{source}: expected exactly 4 atoms for Pt₄, got {len(symbols)}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if set(symbols) != {"Pt"}:
        print(
            f"{source}: expected four Pt atoms, got {symbols}.",
            file=sys.stderr,
        )
        raise SystemExit(1)


def load_pt4_cluster(xyz_name: str | None) -> Atoms:
    """Build the default tetrahedron or read ``TUTORIAL_DIR / xyz_name``."""
    if xyz_name is None:
        return build_pt4_tetrahedron()

    path = TUTORIAL_DIR / xyz_name
    if not path.is_file():
        print(
            f"Cluster file not found: {path}\n"
            f"Place a .xyz in {TUTORIAL_DIR} or set CLUSTER_XYZ = None "
            "to use the built-in tetrahedron.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    loaded = ase.io.read(path)
    if isinstance(loaded, list):
        if len(loaded) != 1:
            print(
                f"{path}: expected one structure, found {len(loaded)} frames.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        loaded = loaded[0]
    if not isinstance(loaded, Atoms):
        print(f"{path}: ASE read did not return an Atoms object.", file=sys.stderr)
        raise SystemExit(1)

    cluster = _normalize_nanoparticle_cluster(loaded)
    _validate_pt4_cluster(cluster, source=str(path))
    return cluster


def activities_for_ph(ph: float) -> tuple[float, float]:
    """Map pH onto (a_water, a_hydroxide).

    Water stays at unit activity. Hydroxide follows a(OH-) ≈ 10**(pH-14).
    Activities must be strictly positive (metalsurfer rejects zeros).
    """
    a_water = 1.0
    a_hydroxide = 10.0 ** (ph - 14.0)
    return (a_water, a_hydroxide)


def ranking_energy(e_ads: float, activity: float) -> float:
    """Ω = E_ads − k_B T ln(a p / p°) at SATP (p = p° = 1 bar, so ln a)."""
    return e_ads - K_B_EV_PER_K * TEMPERATURE_K * math.log(activity)


def make_config(ph: float) -> AdsorptionConfig:
    """Return an ``AdsorptionConfig`` sized for CPU competitive saturation."""
    return AdsorptionConfig(
        # --- Substrate type -------------------------------------------------
        # Tells placement code we have a finite cluster, not a periodic slab.
        material_type="nanoparticle",
        # --- MLIP model -----------------------------------------------------
        # FairChem UMA checkpoint + matching task head. Pair must stay matched:
        # uma-s-1p2 ↔ oc25 (library default when installed), uma-s-1p1 ↔ oc20.
        model_name="uma-s-1p1",
        task_name="oc20",
        # Torch device for energies/forces. This tutorial defaults to CPU.
        device=DEVICE,
        # --- Reproducibility & search budget --------------------------------
        # Fixed seed for conformer and placement sampling.
        seed=42,
        # One RDKit geometry per adsorbate is enough for this tiny demo.
        num_conformers=1,
        # How many initial placements to relax per molecule per step.
        num_placements=12,
        # --- Relaxation lengths (keep stage2 ≥ ~50 on CPU or filters reject) -
        stage1_steps=50,
        stage2_steps=80,
        # Steps when relaxing isolated water/OH reference conformers.
        reference_optimization_steps=50,
        # Batched isolated opts target CUDA; sequential is safer on CPU.
        optimize_isolated_sequentially=True,
        # --- Substrate prep (before saturation) -----------------------------
        # Ionic MLIP relaxation of the hand-built or loaded Pt₄ geometry.
        # Use "none" only if your .xyz is already equilibrated and must not move.
        slab_relaxation_mode="ionic_only",
        # Post-relax force gate (eV/Å). Slightly looser than default 0.05 so
        # short CPU runs still keep chemisorbed poses.
        max_force_convergence=0.15,
        # --- Competitive saturation -----------------------------------------
        # Water and OH⁻ compete each step; lowest Ω wins (see saturation_activities).
        multi_molecule_saturation=True,
        # Commit one adsorbate per step (easier to read than n-tuplet mode).
        saturation_molecules_per_step=1,
        # Hard cap on coverage steps for this demo.
        saturation_max_steps=2,
        # Skip writing every placement to disk (faster, less clutter).
        saturation_save_all_placements=False,
        # --- Reservoir ranking Ω = E_ads − k_B T ln(a p / p°) --------------
        # Temperature and pressure in the Ω formula (SATP defaults in library).
        saturation_temperature=TEMPERATURE_K,
        saturation_pressure=STANDARD_PRESSURE_BAR,
        # (a_water, a_hydroxide) for this pH; order matches MOLECULES above.
        saturation_activities=activities_for_ph(ph),
        # --- Validation (tiny clusters often trip connectivity guards) --------
        skip_topology_check=True,
        saturation_discard_topology_rearrangements=False,
    )


def surface_type_for_ph(ph: float) -> str:
    """Results directory label: results_water_oh_pt4_ph{N}/."""
    return f"water_oh_pt4_ph{int(ph)}"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Water/OH competitive saturation on Pt₄ vs pH (activities)."
    )
    parser.add_argument(
        "--cluster",
        metavar="FILE",
        default=None,
        help=(
            "Pt₄ cluster .xyz in scripts/tutorials/ "
            f"(default: CLUSTER_XYZ={CLUSTER_XYZ!r}, or built-in tetrahedron)"
        ),
    )
    return parser.parse_args(argv)


def _validate_run(
    result: object,
    *,
    ph: float,
) -> MultiMolSaturationRunResult:
    """Return the competitive run, or exit with a short student-facing error."""
    if not isinstance(result, MultiMolSaturationRunResult):
        print(
            f"pH {ph:g}: expected competitive multi-molecule saturation, "
            f"got {type(result).__name__}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    counts_total = sum(result.molecule_counts.values())
    if counts_total != result.n_molecules_at_saturation:
        print(
            f"pH {ph:g}: molecule_counts {result.molecule_counts} do not sum "
            f"to n_molecules_at_saturation={result.n_molecules_at_saturation}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return result


def _print_ph_run(
    result: MultiMolSaturationRunResult,
    *,
    ph: float,
    activities: tuple[float, float],
) -> None:
    activity_by_name = dict(zip(MOLECULE_NAMES, activities, strict=True))
    print()
    print(f"pH {ph:g}  activities {activity_by_name}")
    for step_result in result.steps:
        committed = step_result.committed()
        if not committed:
            print(
                f"  step {step_result.step:>2d} | on cluster: "
                f"{step_result.n_molecules_on_slab:>2d} | unbound final "
                f"(Ω ≥ 0 or no valid placement)"
            )
        else:
            unit = committed[0]
            activity = activity_by_name[unit.molecule]
            omega = ranking_energy(unit.energy_adsorption, activity)
            print(
                f"  step {step_result.step:>2d} | on cluster: "
                f"{step_result.n_molecules_on_slab:>2d} | "
                f"{unit.molecule}: E_ads={unit.energy_adsorption:+.3f} eV  "
                f"Ω={omega:+.3f} eV"
            )
        for name in MOLECULE_NAMES:
            pool = step_result.per_molecule_results.get(name) or []
            if not pool:
                print(f"    {name:>10}: no valid placements")
                continue
            best = min(pool, key=lambda item: item.energy_adsorption)
            omega = ranking_energy(best.energy_adsorption, activity_by_name[name])
            print(
                f"    {name:>10}: E_ads={best.energy_adsorption:+.3f} eV  "
                f"Ω={omega:+.3f} eV"
            )
    print(f"  coverage at saturation: {result.molecule_counts}")


def _print_coverage_table(
    coverage_by_ph: list[tuple[float, dict[str, int]]],
) -> None:
    names = list(MOLECULE_NAMES)
    header = f"{'pH':>6}  " + "  ".join(f"{name:>10}" for name in names)
    print()
    print("Coverage vs pH (molecules committed on Pt₄)")
    print(header)
    print("-" * len(header))
    for ph, counts in coverage_by_ph:
        cells = "  ".join(f"{counts.get(name, 0):>10d}" for name in names)
        print(f"{ph:>6g}  {cells}")
    if all(sum(counts.values()) == 0 for _, counts in coverage_by_ph):
        print()
        print(
            "Coverage is zero: electronic E_ads stayed positive or water had "
            "no valid placements, so Ω ≥ 0 and nothing committed. Compare Ω "
            "across pH in the per-step lines above — that is the activity effect."
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cluster_xyz = args.cluster if args.cluster is not None else CLUSTER_XYZ

    configure_logging(default_level="INFO")
    logger = logging.getLogger(__name__)

    cluster_source = cluster_xyz or "built-in tetrahedron"
    logger.info("Loading Pt₄ cluster from %s", cluster_source)

    # Prep once with ionic MLIP relaxation. On a 4-atom tetrahedron every Pt is
    # "surface", so relax_top_layer=True would leave nothing frozen and prep
    # would re-freeze the whole cluster — clear FixAtoms so the cluster can
    # respond to chemisorption (campaigns allow a fully mobile substrate).
    prep_config = make_config(PH_VALUES[0])
    prep_dir = str(results_dir_for("water_oh_pt4"))
    cluster = prepare_substrate(
        slab=load_pt4_cluster(cluster_xyz),
        config=prep_config,
        results_dir=prep_dir,
    )
    cluster.atoms.set_constraint()
    logger.info(
        "Prepared Pt₄ cluster (%d atoms, fully mobile) in %s",
        len(cluster.atoms),
        prep_dir,
    )

    coverage_by_ph: list[tuple[float, dict[str, int]]] = []
    for ph in PH_VALUES:
        config = replace(prep_config, saturation_activities=activities_for_ph(ph))
        surface_type = surface_type_for_ph(ph)
        campaign = run_saturation(
            slab=cluster,
            molecules=MOLECULES,
            config=config,
            surface_type=surface_type,
            skip_existing=False,
        )
        if not campaign.runs:
            print(f"pH {ph:g}: no saturation runs produced.", file=sys.stderr)
            return 1
        result = _validate_run(campaign.runs[0], ph=ph)
        _print_ph_run(result, ph=ph, activities=activities_for_ph(ph))
        coverage_by_ph.append((ph, dict(result.molecule_counts)))
        print(f"  results: {results_dir_for(surface_type)}")

    _print_coverage_table(coverage_by_ph)
    print()
    print(
        "Remember: Ω, not raw E_ads, decides who binds and when saturation "
        "stops. Stored CSV energies remain electronic E_ads."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

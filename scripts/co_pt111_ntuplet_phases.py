#!/usr/bin/env python3
"""Independent high-n joint searches for CO/Pt(111) ordered coverages.

Asks whether n-tuplet mode (exact-n joint configs, metalsurfer >= 0.9.4) can
recover the coverage-dependent bridge:top sequence of Gunasooriya & Saeys
(ACS Catal. 2018) without seeding the literature registries.

Each job is one coverage, one cell, one step:

* (√3×√3)×3   — 9 surface Pt, n=3, θ=1/3, target all atop
* c(4×2)      — 8 surface Pt, n=4, θ=1/2, target B:T=1:1
* c(√3×5)rect — 10 surface Pt, n=6, θ=0.6, target B:T=1:2
* c(√3×3)rect — 6 surface Pt, n=4, θ=2/3, target B:T=1:3

Optional ``--large-cell`` repeats the same θ/n points on an 8×6 orthogonal
cell (48 surface Pt) that is not the LEED box. Submit that only after the
matched-cell gap to the literature ratio is small (~0.1 eV/CO).

Do not run this on a laptop. From the metalsurfer project root (conda env with
``uma-s-1p2`` / CUDA), the default entry point runs all four primary jobs and
writes the aggregated summary::

    python scripts/co_pt111_ntuplet_phases.py

Optional overrides::

    python scripts/co_pt111_ntuplet_phases.py --job c42
    python scripts/co_pt111_ntuplet_phases.py --large-cell
    python scripts/co_pt111_ntuplet_phases.py --summarize
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import fcc111, make_supercell
from ase.io import write

from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    results_dir_for,
    run_saturation,
)
from metalsurfer.models import ScreeningResult
from metalsurfer.surface_prep import prepare_substrate

RESULTS_ROOT = Path("results_co_pt111_ntuplet")
NUM_JOINT_CONFIGS = 400
CO_SMILES = "[C-]#[O+]"
CO_NAME = "CO"
TOP_K_HOLLOW = 20
VACUUM = 12.0
N_LAYERS = 4
A_PT = 3.967  # Å; near FairChem / experimental Pt lattice
SITE_CUTOFF = 2.5


@dataclass(frozen=True)
class PhaseJob:
    """One independent one-step n-tuplet search at fixed coverage."""

    key: str
    surface_type: str
    theta_ml: float
    n_surface: int
    n_co: int
    description: str
    expected_atop: int
    expected_bridge: int
    expected_hollow: int
    expected_ratio: str
    builder_name: str


PRIMARY_JOBS: tuple[PhaseJob, ...] = (
    PhaseJob(
        key="sqrt3_x3",
        surface_type="co_pt111_ntuplet_sqrt3_x3",
        theta_ml=1.0 / 3.0,
        n_surface=9,
        n_co=3,
        description="(√3×√3)×3, all atop",
        expected_atop=3,
        expected_bridge=0,
        expected_hollow=0,
        expected_ratio="atop×3",
        builder_name="sqrt3_x3",
    ),
    PhaseJob(
        key="c42",
        surface_type="co_pt111_ntuplet_c42",
        theta_ml=0.5,
        n_surface=8,
        n_co=4,
        description="c(4×2), B:T=1:1",
        expected_atop=2,
        expected_bridge=2,
        expected_hollow=0,
        expected_ratio="B:T=2:2",
        builder_name="c42",
    ),
    PhaseJob(
        key="csqrt3x5",
        surface_type="co_pt111_ntuplet_csqrt3x5",
        theta_ml=0.6,
        n_surface=10,
        n_co=6,
        description="c(√3×5)rect, B:T=1:2",
        expected_atop=4,
        expected_bridge=2,
        expected_hollow=0,
        expected_ratio="B:T=2:4",
        builder_name="csqrt3x5",
    ),
    PhaseJob(
        key="csqrt3x3",
        surface_type="co_pt111_ntuplet_csqrt3x3",
        theta_ml=2.0 / 3.0,
        n_surface=6,
        n_co=4,
        description="c(√3×3)rect, B:T=1:3",
        expected_atop=3,
        expected_bridge=1,
        expected_hollow=0,
        expected_ratio="B:T=1:3",
        builder_name="csqrt3x3",
    ),
)

# Same coverages on a non-LEED orthogonal cell (submit only if matched-cell gap
# to the literature ratio is small). n_co chosen so B:T ratios stay exact.
LARGE_N_SURFACE = 48  # 8×6 fcc111 orthogonal
LARGE_JOBS: tuple[PhaseJob, ...] = (
    PhaseJob(
        key="large_sqrt3_x3",
        surface_type="co_pt111_ntuplet_large_sqrt3_x3",
        theta_ml=16 / LARGE_N_SURFACE,
        n_surface=LARGE_N_SURFACE,
        n_co=16,
        description="8×6 orth., θ=1/3, all atop",
        expected_atop=16,
        expected_bridge=0,
        expected_hollow=0,
        expected_ratio="atop×16",
        builder_name="large_8x6",
    ),
    PhaseJob(
        key="large_c42",
        surface_type="co_pt111_ntuplet_large_c42",
        theta_ml=24 / LARGE_N_SURFACE,
        n_surface=LARGE_N_SURFACE,
        n_co=24,
        description="8×6 orth., θ=1/2, B:T=1:1",
        expected_atop=12,
        expected_bridge=12,
        expected_hollow=0,
        expected_ratio="B:T=12:12",
        builder_name="large_8x6",
    ),
    PhaseJob(
        key="large_csqrt3x5",
        surface_type="co_pt111_ntuplet_large_csqrt3x5",
        theta_ml=30 / LARGE_N_SURFACE,
        n_surface=LARGE_N_SURFACE,
        n_co=30,
        description="8×6 orth., θ=0.625 (~0.6), B:T=1:2",
        expected_atop=20,
        expected_bridge=10,
        expected_hollow=0,
        expected_ratio="B:T=10:20",
        builder_name="large_8x6",
    ),
    PhaseJob(
        key="large_csqrt3x3",
        surface_type="co_pt111_ntuplet_large_csqrt3x3",
        theta_ml=32 / LARGE_N_SURFACE,
        n_surface=LARGE_N_SURFACE,
        n_co=32,
        description="8×6 orth., θ=2/3, B:T=1:3",
        expected_atop=24,
        expected_bridge=8,
        expected_hollow=0,
        expected_ratio="B:T=8:24",
        builder_name="large_8x6",
    ),
)

JOBS_BY_KEY: dict[str, PhaseJob] = {
    **{j.key: j for j in PRIMARY_JOBS},
    **{j.key: j for j in LARGE_JOBS},
}


def _primitive_pt111() -> Atoms:
    return fcc111("Pt", size=(1, 1, N_LAYERS), a=A_PT, vacuum=VACUUM, periodic=True)


def build_sqrt3() -> Atoms:
    """(√3×√3)R30° cell: 3 surface Pt."""
    return make_supercell(_primitive_pt111(), [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])


def build_sqrt3_x3() -> Atoms:
    """Three copies of (√3×√3)R30°: 9 surface Pt."""
    return build_sqrt3().repeat((3, 1, 1))


def build_c42() -> Atoms:
    """c(4×2) orthogonal cell: 8 surface Pt."""
    return fcc111(
        "Pt",
        size=(4, 2, N_LAYERS),
        a=A_PT,
        vacuum=VACUUM,
        orthogonal=True,
        periodic=True,
    )


def build_csqrt3x5() -> Atoms:
    """c(√3×5)rect: 10 surface Pt (5×2 orthogonal)."""
    return fcc111(
        "Pt",
        size=(5, 2, N_LAYERS),
        a=A_PT,
        vacuum=VACUUM,
        orthogonal=True,
        periodic=True,
    )


def build_csqrt3x3() -> Atoms:
    """c(√3×3)rect: 6 surface Pt (3×2 orthogonal)."""
    return fcc111(
        "Pt",
        size=(3, 2, N_LAYERS),
        a=A_PT,
        vacuum=VACUUM,
        orthogonal=True,
        periodic=True,
    )


def build_large_8x6() -> Atoms:
    """Orthogonal 8×6 Pt(111) cell: 48 surface Pt."""
    return fcc111(
        "Pt",
        size=(8, 6, N_LAYERS),
        a=A_PT,
        vacuum=VACUUM,
        orthogonal=True,
        periodic=True,
    )


def _top_layer_pts(atoms: Atoms) -> np.ndarray:
    symbols = np.array(atoms.get_chemical_symbols())
    pt = np.where(symbols == "Pt")[0]
    z = atoms.positions[pt, 2]
    zmax = z.max()
    return pt[z > zmax - 0.5]


def _n_surface(atoms: Atoms) -> int:
    return int(len(_top_layer_pts(atoms)))


def _cell2d(cell: np.ndarray) -> np.ndarray:
    """2×2 in-plane cell with ASE row-vector lattice vectors."""
    return np.asarray(cell[:2, :2], dtype=float)


def _mic_delta(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image cartesian displacement b − a in the surface plane."""
    cell2 = _cell2d(cell)
    dfrac = (b - a) @ np.linalg.inv(cell2)
    dfrac -= np.round(dfrac)
    return dfrac @ cell2


def _mic_delta3(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image cartesian displacement b − a (PBC in xy only)."""
    delta = np.asarray(b - a, dtype=float).copy()
    delta[:2] = _mic_delta(a[:2], b[:2], cell)
    return delta


def _classify_sites(atoms: Atoms, cutoff: float = SITE_CUTOFF) -> tuple[int, int, int]:
    """Count C atoms by Pt coordination: 1=atop, 2=bridge, >=3=hollow.

    Distances use in-plane MIC. For very small cells the Pt lattice is
    expanded (adsorbates kept once) before coordination is measured, so a
    hollow's three neighbours are distinct atoms.
    """
    symbols = np.array(atoms.get_chemical_symbols())
    pt_mask = symbols == "Pt"
    ads = atoms[~pt_mask]
    slab = atoms[pt_mask]
    if _n_surface(atoms) < 8:
        slab = slab.repeat((2, 2, 1))
    work = slab + ads
    work.cell = slab.cell
    work.set_pbc(atoms.pbc)
    symbols = np.array(work.get_chemical_symbols())
    c_idx = np.where(symbols == "C")[0]
    pt_idx = np.where(symbols == "Pt")[0]
    pt_pos = work.positions[pt_idx]
    cell = work.cell.array
    n_atop = n_bridge = n_hollow = 0
    for ci in c_idx:
        cpos = work.positions[ci]
        d = np.array([np.linalg.norm(_mic_delta3(cpos, p, cell)) for p in pt_pos])
        n = int(np.sum(d < cutoff))
        if n <= 1:
            n_atop += 1
        elif n == 2:
            n_bridge += 1
        else:
            n_hollow += 1
    return n_atop, n_bridge, n_hollow


def _bridge_top_string(n_atop: int, n_bridge: int, n_hollow: int) -> str:
    bits: list[str] = []
    if n_bridge and n_atop:
        bits.append(f"B:T={n_bridge}:{n_atop}")
    elif n_atop:
        bits.append(f"atop×{n_atop}")
    elif n_bridge:
        bits.append(f"bridge×{n_bridge}")
    if n_hollow:
        bits.append(f"hollow×{n_hollow}")
    return "+".join(bits) if bits else "?"


def _build_slab(builder_name: str) -> Atoms:
    builders = {
        "sqrt3_x3": build_sqrt3_x3,
        "c42": build_c42,
        "csqrt3x5": build_csqrt3x5,
        "csqrt3x3": build_csqrt3x3,
        "large_8x6": build_large_8x6,
    }
    if builder_name not in builders:
        raise KeyError(f"unknown builder {builder_name!r}")
    atoms = builders[builder_name]()
    atoms.set_pbc([True, True, False])
    return atoms


def _group_packs(
    all_results: list[ScreeningResult], n: int
) -> list[list[ScreeningResult]]:
    """Chunk flattened n-tuplet rows into joint packs of length *n*."""
    if n <= 0:
        return []
    if len(all_results) % n != 0:
        raise ValueError(
            f"flattened n-tuplet results length ({len(all_results)}) is not "
            f"divisible by n={n}; expected complete joint packs only"
        )
    return [all_results[i : i + n] for i in range(0, len(all_results), n)]


def _pack_atoms(pack: list[ScreeningResult]) -> Atoms:
    """Full composite geometry (every unit stores the shared relaxed structure)."""
    return pack[0].atoms.copy()


def _pack_energy_per_co(pack: list[ScreeningResult]) -> float:
    return float(pack[0].energy_adsorption)


def _matches_literature(
    n_atop: int,
    n_bridge: int,
    n_hollow: int,
    job: PhaseJob,
) -> bool:
    return (n_atop, n_bridge, n_hollow) == (
        job.expected_atop,
        job.expected_bridge,
        job.expected_hollow,
    )


def _job_outdir(job: PhaseJob) -> Path:
    return RESULTS_ROOT / job.key


def _analyze_packs(
    job: PhaseJob,
    packs: list[list[ScreeningResult]],
    *,
    num_placements: int = NUM_JOINT_CONFIGS,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Per-pack rows plus one summary row for this coverage."""
    ranked: list[tuple[float, list[ScreeningResult], int, int, int, str]] = []
    for pack in packs:
        atoms = _pack_atoms(pack)
        n_a, n_b, n_h = _classify_sites(atoms)
        e_per = _pack_energy_per_co(pack)
        ranked.append((e_per, pack, n_a, n_b, n_h, _bridge_top_string(n_a, n_b, n_h)))
    ranked.sort(key=lambda t: t[0])

    pack_rows: list[dict[str, object]] = []
    for rank, (e_per, _pack, n_a, n_b, n_h, ratio) in enumerate(ranked, start=1):
        pack_rows.append(
            {
                "job": job.key,
                "rank": rank,
                "theta_ml": f"{job.theta_ml:.6f}",
                "n_surface": job.n_surface,
                "n_co": job.n_co,
                "e_ads_per_co_eV": f"{e_per:.8f}",
                "n_atop": n_a,
                "n_bridge": n_b,
                "n_hollow": n_h,
                "bridge_top_ratio": ratio,
                "matches_literature": int(_matches_literature(n_a, n_b, n_h, job)),
                "hollow_frac": f"{(n_h / job.n_co):.6f}",
            }
        )

    if not ranked:
        summary = {
            "job": job.key,
            "surface_type": job.surface_type,
            "description": job.description,
            "theta_ml": f"{job.theta_ml:.6f}",
            "n_surface": job.n_surface,
            "n_co": job.n_co,
            "n_joint_configs_requested": num_placements,
            "n_valid_packs": 0,
            "expected_ratio": job.expected_ratio,
            "winner_e_ads_per_co_eV": "",
            "winner_n_atop": "",
            "winner_n_bridge": "",
            "winner_n_hollow": "",
            "winner_ratio": "",
            "lit_found": 0,
            "lit_e_ads_per_co_eV": "",
            "lit_gap_eV": "",
            "top20_mean_hollow_frac": "",
        }
        return pack_rows, summary

    w_e, w_pack, w_a, w_b, w_h, w_ratio = ranked[0]
    lit = next(
        (
            (e, pack, a, b, h, ratio)
            for e, pack, a, b, h, ratio in ranked
            if _matches_literature(a, b, h, job)
        ),
        None,
    )
    top = ranked[: min(TOP_K_HOLLOW, len(ranked))]
    mean_hollow = float(np.mean([h / job.n_co for _, _, _, _, h, _ in top]))

    summary = {
        "job": job.key,
        "surface_type": job.surface_type,
        "description": job.description,
        "theta_ml": f"{job.theta_ml:.6f}",
        "n_surface": job.n_surface,
        "n_co": job.n_co,
        "n_joint_configs_requested": num_placements,
        "n_valid_packs": len(ranked),
        "expected_ratio": job.expected_ratio,
        "winner_e_ads_per_co_eV": f"{w_e:.8f}",
        "winner_n_atop": w_a,
        "winner_n_bridge": w_b,
        "winner_n_hollow": w_h,
        "winner_ratio": w_ratio,
        "lit_found": int(lit is not None),
        "lit_e_ads_per_co_eV": f"{lit[0]:.8f}" if lit else "",
        "lit_gap_eV": f"{(lit[0] - w_e):.8f}" if lit else "",
        "top20_mean_hollow_frac": f"{mean_hollow:.6f}",
    }

    out = _job_outdir(job)
    xyz_dir = out / "xyz"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    write(xyz_dir / "best_pack.xyz", _pack_atoms(w_pack))
    if lit is not None:
        write(xyz_dir / "best_literature_ratio_pack.xyz", _pack_atoms(lit[1]))

    return pack_rows, summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_job(
    job: PhaseJob, *, num_placements: int = NUM_JOINT_CONFIGS
) -> dict[str, object]:
    """Run one independent one-step n-tuplet search and write analysis CSVs."""
    configure_logging(default_level="INFO")
    out = _job_outdir(job)
    out.mkdir(parents=True, exist_ok=True)

    slab_atoms = _build_slab(job.builder_name)
    n_surf = _n_surface(slab_atoms)
    if n_surf != job.n_surface:
        raise RuntimeError(
            f"{job.key}: surface Pt {n_surf} != expected {job.n_surface}"
        )
    if abs(job.n_co / job.n_surface - job.theta_ml) > 1e-6:
        raise RuntimeError(
            f"{job.key}: n_co/n_surface = {job.n_co / job.n_surface} "
            f"!= theta {job.theta_ml}"
        )
    site_sum = job.expected_atop + job.expected_bridge + job.expected_hollow
    if site_sum != job.n_co:
        raise RuntimeError(
            f"{job.key}: expected site counts sum to {site_sum}, not n_co={job.n_co}"
        )

    # Explicit num_placements skips workload autotune (and the n-tuplet
    # floor-divide of that autotuned budget).
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=num_placements,
        saturation_molecules_per_step=job.n_co,
        saturation_max_steps=1,
        saturation_save_all_placements=True,
        slab_relaxation_mode="none",
        min_pbc_image_separation=2.0,
        stage1_steps=50,
        stage2_steps=100,
        seed=42,
    )

    results_dir = str(results_dir_for(job.surface_type))
    slab = prepare_substrate(
        slab=slab_atoms,
        config=config,
        results_dir=results_dir,
        slab_relaxation_mode="none",
        relax_top_layer=False,
    )

    print(
        f"\n=== {job.key}: {job.description} | n={job.n_co} | "
        f"{num_placements} joint configs ==="
    )
    campaign = run_saturation(
        slab=slab,
        molecules=[(CO_SMILES, CO_NAME)],
        config=config,
        surface_type=job.surface_type,
        skip_existing=False,
    )
    if not campaign.runs:
        raise RuntimeError(f"{job.key}: no saturation runs produced")
    run = campaign.runs[0]
    if not run.steps:
        raise RuntimeError(f"{job.key}: empty step list")
    step = run.steps[0]
    packs = _group_packs(step.all_results, job.n_co)
    pack_rows, summary = _analyze_packs(job, packs, num_placements=num_placements)

    _write_csv(out / "packs_detailed.csv", pack_rows)
    _write_csv(out / "summary.csv", [summary])
    meta = {
        "job": job.key,
        "surface_type": job.surface_type,
        "model": config.model_name,
        "task": config.task_name,
        "a_pt": A_PT,
        "n_layers": N_LAYERS,
        "n_co": job.n_co,
        "num_placements": num_placements,
        "n_valid_packs": summary["n_valid_packs"],
        "metalsurfer_results_dir": results_dir,
        "protocol": "one-step exact-n joint tuplet on bare slab",
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"  valid packs: {summary['n_valid_packs']} | "
        f"winner {summary['winner_ratio']} "
        f"{summary['winner_e_ads_per_co_eV']} eV/CO | "
        f"lit_found={summary['lit_found']} gap={summary['lit_gap_eV']}"
    )
    print(f"  wrote {out / 'summary.csv'}")
    return summary


def summarize(jobs: tuple[PhaseJob, ...] | list[PhaseJob]) -> Path:
    """Aggregate per-job summary.csv files into one campaign CSV."""
    rows: list[dict[str, object]] = []
    for job in jobs:
        path = _job_outdir(job) / "summary.csv"
        if not path.is_file():
            print(f"missing {path}; skip")
            continue
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    out = RESULTS_ROOT / "phase_discovery_summary.csv"
    _write_csv(out, rows)
    print(f"wrote {out} ({len(rows)} rows)")
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--job",
        choices=sorted(JOBS_BY_KEY),
        help="Run only one coverage cell (default: all four primary cells).",
    )
    group.add_argument(
        "--large-cell",
        action="store_true",
        help="Run optional 8×6 non-LEED jobs (only after matched-cell gap is small).",
    )
    group.add_argument(
        "--summarize",
        action="store_true",
        help="Aggregate existing per-job summary.csv into phase_discovery_summary.csv.",
    )
    parser.add_argument(
        "--num-placements",
        type=int,
        default=NUM_JOINT_CONFIGS,
        help=f"Joint configs per cell (default {NUM_JOINT_CONFIGS}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize([*PRIMARY_JOBS, *LARGE_JOBS])
        return 0

    if args.job:
        jobs = [JOBS_BY_KEY[args.job]]
    elif args.large_cell:
        jobs = list(LARGE_JOBS)
    else:
        jobs = list(PRIMARY_JOBS)

    for job in jobs:
        run_job(job, num_placements=args.num_placements)

    # Refresh the campaign CSV from whatever this invocation ran, plus any
    # other finished job dirs already on disk.
    summarize([*PRIMARY_JOBS, *LARGE_JOBS])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A/B adaptive_grid vs topology/voronoi on example substrates (+ optional e2e).

Internal plugin is not on AdsorptionConfig; this script temporarily widens
SITE_GENERATOR_OPTIONS for campaign runs. Site-generation timing always runs;
pass --e2e for slim binding demos (GPU / MLIP).
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import numpy as np
from ase.cluster import Icosahedron
from ase.io import read
from scipy.spatial import KDTree

# Ensure local src is preferred when not using an editable install.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import metalsurfer.config as config_mod
from metalsurfer import (
    AdsorptionConfig,
    configure_logging,
    results_dir_for,
    run_adsorption,
)
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _SITE_CONTEXT_CACHE_LOCK,
)
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.surface_prep import prepare_substrate


def _enable_adaptive_grid_config() -> None:
    """Allow AdsorptionConfig(site_generator='adaptive_grid') for this process."""
    if "adaptive_grid" not in config_mod.SITE_GENERATOR_OPTIONS:
        config_mod.SITE_GENERATOR_OPTIONS = (
            *config_mod.SITE_GENERATOR_OPTIONS,
            "adaptive_grid",
        )
    config_mod._SITE_GENERATOR_ALLOWED_MATERIALS["adaptive_grid"] = frozenset(
        {"slab", "nanoparticle", "porous"}
    )


def _clear_site_cache() -> None:
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()


def _pt13():
    atoms = Icosahedron("Pt", noshells=2)
    atoms.set_cell([30.0, 30.0, 30.0])
    atoms.center()
    atoms.pbc = False
    return atoms


def _bench_sites(
    atoms, material_type: str, plugin: str, repeats: int = 3, adsorbate=None
):
    times = []
    sites = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        sites = get_unified_sites(
            atoms,
            material_type=material_type,
            site_generator=plugin,
            adsorbate=adsorbate,
        )
        times.append(time.perf_counter() - t0)
    assert sites is not None
    xyz = np.asarray([s.xyz for s in sites], dtype=float)
    types = {s.site_type: 0 for s in sites}
    for s in sites:
        types[s.site_type] += 1
    nn = np.asarray(
        [s.nn_distance for s in sites if s.nn_distance is not None], dtype=float
    )
    return {
        "n": len(sites),
        "t_mean": statistics.mean(times),
        "t_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "xyz": xyz,
        "types": types,
        "nn_med": float(np.median(nn)) if len(nn) else float("nan"),
    }


def _overlap(ref_xyz: np.ndarray, other_xyz: np.ndarray, tol: float) -> float:
    if len(ref_xyz) == 0 or len(other_xyz) == 0:
        return 0.0
    d, _ = KDTree(other_xyz).query(ref_xyz, k=1)
    return float(np.mean(np.asarray(d, dtype=float) <= tol))


def run_site_ab() -> None:
    print("=" * 72)
    print("Site-generation A/B (get_unified_sites)")
    print("=" * 72)

    cases = []

    # Ru(0001)-like slab via prepare (ionic prep is slow); use ASE Ru(0001) cut.
    from ase.build import hcp0001

    slab = hcp0001("Ru", size=(3, 3, 3), vacuum=12.0)
    slab.pbc = True
    cases.append(("Ru(0001) 3x3x3", "slab", "topology", slab, 0.75))

    cases.append(("Pt13 icosahedron", "nanoparticle", "topology", _pt13(), 0.75))

    cif = os.path.join(_ROOT, "examples", "mof_structures", "RUBTAK01.cif")
    mof = read(cif)
    cases.append(("RUBTAK01 MOF", "porous", "voronoi", mof, 1.5))

    # Optional adsorbate for spacing on slab.
    from ase import Atoms

    h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])

    rows = []
    for label, mat, baseline, atoms, tol in cases:
        print(f"\n--- {label} ({mat}) ---")
        base = _bench_sites(atoms, mat, baseline)
        grid = _bench_sites(atoms, mat, "adaptive_grid")
        grid_h2 = None
        if mat == "slab":
            grid_h2 = _bench_sites(atoms, mat, "adaptive_grid", adsorbate=h2)

        # Porous: reverse coverage (grid near voronoi); else baseline near grid.
        if mat == "porous":
            hit = _overlap(grid["xyz"], base["xyz"], tol)
            hit_label = f"grid→{baseline}@{tol}Å"
        else:
            hit = _overlap(base["xyz"], grid["xyz"], tol)
            hit_label = f"{baseline}→grid@{tol}Å"

        speed = base["t_mean"] / grid["t_mean"] if grid["t_mean"] > 0 else float("inf")
        print(
            f"  {baseline:12s}  n={base['n']:4d}  "
            f"t={base['t_mean'] * 1e3:7.1f}±{base['t_std'] * 1e3:5.1f} ms  "
            f"nn_med={base['nn_med']:.2f}  types={base['types']}"
        )
        print(
            f"  {'adaptive_grid':12s}  n={grid['n']:4d}  "
            f"t={grid['t_mean'] * 1e3:7.1f}±{grid['t_std'] * 1e3:5.1f} ms  "
            f"nn_med={grid['nn_med']:.2f}  types={grid['types']}"
        )
        if grid_h2 is not None:
            print(
                f"  {'adap+H2':12s}  n={grid_h2['n']:4d}  "
                f"t={grid_h2['t_mean'] * 1e3:7.1f}±{grid_h2['t_std'] * 1e3:5.1f} ms"
            )
        print(f"  overlap {hit_label}: {hit * 100:.1f}%")
        print(
            f"  speed vs baseline: {speed:.2f}x ({'faster' if speed > 1 else 'slower'} adaptive)"
        )
        rows.append((label, mat, baseline, base, grid, hit, speed))

    print("\n" + "=" * 72)
    print(
        f"{'system':22s} {'base':>6s} {'grid':>6s} {'t_base':>8s} {'t_grid':>8s} {'speed':>6s} {'overlap':>8s}"
    )
    for label, _mat, _baseline, base, grid, hit, speed in rows:
        print(
            f"{label:22s} {base['n']:6d} {grid['n']:6d} "
            f"{base['t_mean'] * 1e3:7.1f}ms {grid['t_mean'] * 1e3:7.1f}ms "
            f"{speed:5.2f}x {hit * 100:6.1f}%"
        )


def _run_campaign(
    name: str,
    config: AdsorptionConfig,
    slab,
    molecules,
    surface_type: str,
    system_name: str,
):
    _clear_site_cache()
    results_dir = str(results_dir_for(f"{surface_type}_{config.site_generator}"))
    t0 = time.perf_counter()
    campaign = run_adsorption(
        slab=slab,
        molecules=molecules,
        config=config,
        surface_type=f"{surface_type}_{config.site_generator}",
        system_name=system_name,
        skip_existing=False,
    )
    elapsed = time.perf_counter() - t0
    best = None
    n_valid = 0
    if campaign.molecule_summaries:
        s = campaign.molecule_summaries[0]
        best = s.best_adsorption_energy
        n_valid = s.n_valid_placements
    elif campaign.run_results and campaign.run_results[0].results:
        results = campaign.run_results[0].results
        n_valid = len(results)
        best = min(r.energy_adsorption for r in results)
    return {
        "name": name,
        "plugin": config.site_generator,
        "elapsed_s": elapsed,
        "n_valid": n_valid,
        "best_eads": best,
        "results_dir": results_dir,
    }


def run_e2e(num_placements: int) -> None:
    print("\n" + "=" * 72)
    print(f"End-to-end demos (num_placements={num_placements})")
    print("=" * 72)
    _enable_adaptive_grid_config()
    configure_logging(default_level="WARNING")

    results = []

    # --- Ru H2 slab ---
    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="slab",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            enable_dissociative_placement=True,
            skip_topology_check=True,
            stage2_steps=200,
            slab_relaxation_mode="none",
        )
        print(f"\n[H2/Ru] preparing substrate + run site_generator={plugin!r} ...")
        slab = prepare_substrate(
            bulk_id="mp-33",
            miller_indices=(0, 0, 1),
            supercell=(2, 2, 1),
            config=cfg,
            results_dir=str(results_dir_for(f"ab_h2_ru_{plugin}")),
        )
        results.append(
            _run_campaign(
                "H2/Ru(0001)",
                cfg,
                slab,
                [("[H][H]", "H2")],
                "ab_h2_ru",
                "Ru_0001",
            )
        )

    # --- Pt13 H2 ---
    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="nanoparticle",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            slab_relaxation_mode="none",
            enable_dissociative_placement=True,
            skip_topology_check=True,
            stage2_steps=200,
        )
        print(f"\n[H2/Pt13] run site_generator={plugin!r} ...")
        cluster = prepare_substrate(
            slab=_pt13(),
            config=cfg,
            results_dir=str(results_dir_for(f"ab_h2_pt13_{plugin}")),
        )
        results.append(
            _run_campaign(
                "H2/Pt13",
                cfg,
                cluster,
                [("[H][H]", "H2")],
                "ab_h2_pt13",
                "Pt_13",
            )
        )

    # --- CO2 MOF ---
    cif = os.path.join(_ROOT, "examples", "mof_structures", "RUBTAK01.cif")
    mof_atoms = read(cif)
    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="porous",
            site_generator=plugin,
            slab_relaxation_mode="none",
            seed=42,
            num_conformers=1,
            num_placements=max(3, num_placements // 2),
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            stage2_steps=200,
        )
        print(f"\n[CO2/MOF] run site_generator={plugin!r} ...")
        mof = prepare_substrate(
            slab=mof_atoms,
            config=cfg,
            results_dir=str(results_dir_for(f"ab_co2_mof_{plugin}")),
            align=False,
        )
        results.append(
            _run_campaign(
                "CO2/MOF",
                cfg,
                mof,
                [("O=C=O", "CO2")],
                "ab_co2_mof",
                "MOF_cell",
            )
        )

    print("\n" + "=" * 72)
    print(
        f"{'system':14s} {'plugin':14s} {'t_s':>8s} {'n_valid':>8s} {'best_Eads':>10s}"
    )
    for r in results:
        e = r["best_eads"]
        e_s = f"{e:10.4f}" if e is not None and np.isfinite(e) else f"{'n/a':>10s}"
        print(
            f"{r['name']:14s} {r['plugin']:14s} {r['elapsed_s']:8.1f} "
            f"{r['n_valid']:8d} {e_s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run slim binding demos (GPU/MLIP) for auto vs adaptive_grid",
    )
    parser.add_argument(
        "--num-placements",
        type=int,
        default=6,
        help="Placements per e2e campaign (default 6)",
    )
    args = parser.parse_args()
    run_site_ab()
    if args.e2e:
        run_e2e(args.num_placements)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

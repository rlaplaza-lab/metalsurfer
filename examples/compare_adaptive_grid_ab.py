#!/usr/bin/env python3
"""A/B adaptive_grid vs topology/voronoi (auto) on example substrates (+ optional e2e).

Site-generation timing and e2e demos honour ``--n-jobs`` (default ``1``;
prefer ``--n-jobs 1`` on small GPUs to avoid thread/CUDA contention).

Default conclusions (keep ``site_generator="auto"``; adaptive_grid opt-in only):
topology/Voronoi stay the production defaults with system-specific heuristics;
adaptive_grid is one PBC/clearance path for every material, slower to build,
and did not beat ``auto`` on best E_ads for H₂/Ru, CO₂/MOF, or slim
camphor/Cu(111) BO (marginal wins on H₂/Pt₁₃ / ethene/Ru₅₅ in slim e2e only).
Keep ``adaptive_grid_spacing=0.70``, refine ``0``, NMS framework scale ``0.25``,
and ``voronoi_site_enrichment=True``.

Run (conda env metalsurfer)::

  conda activate metalsurfer
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python examples/compare_adaptive_grid_ab.py --n-jobs 1
  python examples/compare_adaptive_grid_ab.py --e2e --num-placements 6 --n-jobs 1
  # Re-run GPU demos without repeating the CPU catalog sweep:
  python examples/compare_adaptive_grid_ab.py --e2e --skip-site-ab --num-placements 6 --n-jobs 1
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.cluster import Icosahedron
from ase.io import read
from scipy.spatial import KDTree

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from metalsurfer import (
    AdsorptionConfig,
    BOConfig,
    configure_logging,
    results_dir_for,
    run_adsorption,
    run_adsorption_bo,
)
from metalsurfer.placement._parallel import resolve_materialize_workers
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _SITE_CONTEXT_CACHE_LOCK,
)
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.surface_prep import prepare_substrate

_DEFAULT_N_JOBS = 1
_RU_FCC_LATTICE_CONSTANT = 2.71 * (2.0**0.5)


def _clear_site_cache() -> None:
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()


def _pt13():
    atoms = Icosahedron("Pt", noshells=2)
    atoms.set_cell([30.0, 30.0, 30.0])
    atoms.center()
    atoms.pbc = False
    return atoms


def _ru55():
    atoms = Icosahedron("Ru", noshells=3, latticeconstant=_RU_FCC_LATTICE_CONSTANT)
    atoms.set_cell([40.0, 40.0, 40.0])
    atoms.center()
    atoms.pbc = False
    return atoms


def _tilted_ru_slab():
    from ase.build import hcp0001

    slab = hcp0001("Ru", size=(3, 3, 3), vacuum=12.0)
    angle = np.deg2rad(35.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    slab.set_cell(np.asarray(slab.get_cell(), dtype=float) @ R.T, scale_atoms=True)
    slab.pbc = True
    return slab


def _stepped_ru_slab():
    from ase.build import hcp0001

    slab = hcp0001("Ru", size=(4, 4, 4), vacuum=12.0)
    pos = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    # Drop half the top layer along a to create a step.
    z = pos[:, 2]
    z_max = float(np.max(z))
    top = z > z_max - 0.8
    frac_a = (pos @ np.linalg.inv(cell).T)[:, 0]
    keep = ~(top & (frac_a > 0.5))
    slab = slab[keep]
    slab.set_pbc([True, True, False])
    return slab


def _camphor_cu111_slab():
    """Load paper Cu(111) slab if cached; else build a modest Cu(111) fallback."""
    xyz = Path(_ROOT) / "examples" / "camphor_cu111" / "dft_reference_slab.xyz"
    if xyz.exists():
        atoms = read(str(xyz))
        atoms.pbc = True
        return atoms
    from ase.build import fcc111

    slab = fcc111("Cu", size=(6, 4, 4), vacuum=12.0, orthogonal=True)
    slab.pbc = True
    return slab


def _bench_sites(
    atoms,
    material_type: str,
    plugin: str,
    repeats: int = 3,
    adaptive_grid_spacing: float | None = None,
    adaptive_grid_refine_levels: int = 0,
    adaptive_grid_nms_framework_scale: float | None = None,
    enrich: bool = True,
    n_jobs: int = _DEFAULT_N_JOBS,
):
    times = []
    sites = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        sites = get_unified_sites(
            atoms,
            material_type=material_type,
            site_generator=plugin,
            adaptive_grid_spacing=adaptive_grid_spacing,
            adaptive_grid_refine_levels=adaptive_grid_refine_levels,
            adaptive_grid_nms_framework_scale=adaptive_grid_nms_framework_scale,
            enrich=enrich,
            n_jobs=n_jobs,
        )
        times.append(time.perf_counter() - t0)
    assert sites is not None
    xyz = np.asarray([s.xyz for s in sites], dtype=float)
    types: dict[str, int] = {}
    for s in sites:
        types[s.site_type] = types.get(s.site_type, 0) + 1
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


def run_site_ab(n_jobs: int = _DEFAULT_N_JOBS) -> None:
    n_workers = resolve_materialize_workers(n_jobs)
    print("=" * 72)
    print(
        f"Site-generation A/B (get_unified_sites, n_jobs={n_jobs} → {n_workers} workers)"
    )
    print("=" * 72)

    from ase.build import hcp0001

    cases = []
    slab = hcp0001("Ru", size=(3, 3, 3), vacuum=12.0)
    slab.pbc = True
    cases.append(("Ru(0001) 3x3x3", "slab", "topology", slab, 0.75))
    cases.append(("Ru(0001) tilted", "slab", "topology", _tilted_ru_slab(), 0.75))
    cases.append(("Ru(0001) stepped", "slab", "topology", _stepped_ru_slab(), 0.75))
    cases.append(("Pt13 icosahedron", "nanoparticle", "topology", _pt13(), 0.75))
    cases.append(("Ru55 icosahedron", "nanoparticle", "topology", _ru55(), 0.75))
    cases.append(("camphor Cu(111)", "slab", "topology", _camphor_cu111_slab(), 0.75))

    cif = os.path.join(_ROOT, "examples", "mof_structures", "RUBTAK01.cif")
    mof = read(cif)
    cases.append(("RUBTAK01 MOF", "porous", "voronoi", mof, 1.5))

    rows = []
    for label, mat, baseline, atoms, tol in cases:
        print(f"\n--- {label} ({mat}) ---")
        base = _bench_sites(atoms, mat, baseline, n_jobs=n_jobs)
        # Default coarse adaptive_grid_spacing from AdsorptionConfig (0.70 Å).
        grid = _bench_sites(atoms, mat, "adaptive_grid", n_jobs=n_jobs)
        grid_dense = None
        # On metals the NN merge floor often dominates ``1.5 * h``, so a finer
        # absolute spacing need not increase catalog size. Refine levels do.
        if mat == "slab" and "tilted" not in label and "stepped" not in label:
            grid_dense = _bench_sites(
                atoms,
                mat,
                "adaptive_grid",
                adaptive_grid_spacing=0.70,
                adaptive_grid_refine_levels=1,
                n_jobs=n_jobs,
            )
        elif mat == "porous":
            grid_dense = _bench_sites(
                atoms,
                mat,
                "adaptive_grid",
                adaptive_grid_spacing=0.50,
                adaptive_grid_refine_levels=0,
                n_jobs=n_jobs,
            )

        if mat == "porous":
            hit = _overlap(grid["xyz"], base["xyz"], tol)
            hit_label = f"grid→{baseline}@{tol}Å"
            # adaptive_grid is wall-near shells (not free-volume pores); overlap
            # with Voronoi centres is optional context, not a pass/fail gate.
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
        if grid_dense is not None:
            denser_label = "adap+refine1" if mat == "slab" else "adap@0.50Å"
            print(
                f"  {denser_label:12s}  n={grid_dense['n']:4d}  "
                f"t={grid_dense['t_mean'] * 1e3:7.1f}±{grid_dense['t_std'] * 1e3:5.1f} ms"
            )
        if mat == "porous":
            vor_plain = _bench_sites(atoms, mat, "voronoi", enrich=False, n_jobs=n_jobs)
            print(
                f"  {'voronoi-noen':12s}  n={vor_plain['n']:4d}  "
                f"t={vor_plain['t_mean'] * 1e3:7.1f}±{vor_plain['t_std'] * 1e3:5.1f} ms"
            )
        print(f"  overlap {hit_label}: {hit * 100:.1f}%")
        print(
            f"  speed vs baseline: {speed:.2f}x "
            f"({'faster' if speed > 1 else 'slower'} adaptive)"
        )
        rows.append((label, mat, baseline, base, grid, hit, speed))

    print("\n" + "=" * 72)
    print(
        f"{'system':22s} {'base':>6s} {'grid':>6s} {'t_base':>8s} "
        f"{'t_grid':>8s} {'speed':>6s} {'overlap':>8s}"
    )
    for label, _mat, _baseline, base, grid, hit, speed in rows:
        print(
            f"{label:22s} {base['n']:6d} {grid['n']:6d} "
            f"{base['t_mean'] * 1e3:7.1f}ms {grid['t_mean'] * 1e3:7.1f}ms "
            f"{speed:5.2f}x {hit * 100:6.1f}%"
        )

    # Knob sweep on flat Ru + MOF (spacing / refine / NMS floor).
    print("\n" + "=" * 72)
    print("Adaptive-grid knob sweep (Ru flat + RUBTAK01)")
    print("=" * 72)
    ru = hcp0001("Ru", size=(3, 3, 3), vacuum=12.0)
    ru.pbc = True
    for spacing in (0.50, 0.70):
        for refine in (0, 1):
            for nms in (0.25, 0.50, 0.65):
                r = _bench_sites(
                    ru,
                    "slab",
                    "adaptive_grid",
                    adaptive_grid_spacing=spacing,
                    adaptive_grid_refine_levels=refine,
                    adaptive_grid_nms_framework_scale=nms,
                    n_jobs=n_jobs,
                    repeats=2,
                )
                print(
                    f"  Ru spacing={spacing:.2f} refine={refine} nms={nms:.2f} "
                    f"→ n={r['n']:4d}  t={r['t_mean'] * 1e3:6.1f} ms"
                )
    for spacing in (0.50, 0.70):
        r = _bench_sites(
            mof,
            "porous",
            "adaptive_grid",
            adaptive_grid_spacing=spacing,
            adaptive_grid_refine_levels=0,
            n_jobs=n_jobs,
            repeats=2,
        )
        print(
            f"  MOF spacing={spacing:.2f} refine=0 "
            f"→ n={r['n']:4d}  t={r['t_mean'] * 1e3:6.1f} ms"
        )

    # Auto-plugin knobs that matter for production defaults.
    print("\n" + "=" * 72)
    print("Auto-plugin knob sweep (Voronoi enrich + rough-slab topology)")
    print("=" * 72)
    for enrich in (True, False):
        r = _bench_sites(
            mof,
            "porous",
            "voronoi",
            enrich=enrich,
            n_jobs=n_jobs,
            repeats=2,
        )
        label = "enrich" if enrich else "no-enrich"
        print(
            f"  MOF voronoi {label:9s} → n={r['n']:4d}  "
            f"t={r['t_mean'] * 1e3:6.1f} ms  types={r['types']}"
        )
    for label, atoms in (
        ("Ru tilted", _tilted_ru_slab()),
        ("Ru stepped", _stepped_ru_slab()),
    ):
        for enrich in (True, False):
            r = _bench_sites(
                atoms,
                "slab",
                "topology",
                enrich=enrich,
                n_jobs=n_jobs,
                repeats=2,
            )
            tag = "enrich" if enrich else "no-enrich"
            print(
                f"  {label:10s} topology {tag:9s} → n={r['n']:4d}  "
                f"t={r['t_mean'] * 1e3:6.1f} ms  types={r['types']}"
            )


def _eads_stats(energies: list[float]) -> dict[str, float]:
    arr = np.asarray(energies, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            k: float("nan")
            for k in ("min", "p25", "median", "mean", "p75", "max", "std")
        }
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def _ascii_hist(energies: list[float], bins: int = 8) -> str:
    arr = np.asarray([e for e in energies if np.isfinite(e)], dtype=float)
    if len(arr) == 0:
        return "(no finite E_ads)"
    counts, edges = np.histogram(arr, bins=bins)
    peak = max(int(counts.max()), 1)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(20 * c / peak))
        lines.append(f"  [{edges[i]:7.3f},{edges[i + 1]:7.3f}) {bar} ({c})")
    return "\n".join(lines)


def _as_atoms(slab) -> Atoms:
    if isinstance(slab, Atoms):
        return slab
    return slab.atoms


def _fresh_results_dir(name: str) -> str:
    """Return *name*'s results path after deleting any previous run artifacts."""
    path = Path(results_dir_for(name))
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _run_campaign(
    name: str,
    config: AdsorptionConfig,
    slab,
    molecules,
    surface_type: str,
    system_name: str,
    *,
    bo: bool = False,
):
    _clear_site_cache()
    atoms = _as_atoms(slab)
    n_sites = len(
        get_unified_sites(
            atoms,
            material_type=config.material_type,
            site_generator=config.site_generator,
            adaptive_grid_spacing=float(config.adaptive_grid_spacing),
            adaptive_grid_refine_levels=int(config.adaptive_grid_refine_levels),
            adaptive_grid_nms_framework_scale=float(
                config.adaptive_grid_nms_framework_scale
            ),
            n_jobs=config.n_jobs,
        )
    )
    results_dir = _fresh_results_dir(f"{surface_type}_{config.site_generator}")
    t0 = time.perf_counter()
    if bo:
        campaign = run_adsorption_bo(
            slab=slab,
            molecules=molecules,
            config=config,
            surface_type=f"{surface_type}_{config.site_generator}",
            system_name=system_name,
            skip_existing=False,
        )
    else:
        campaign = run_adsorption(
            slab=slab,
            molecules=molecules,
            config=config,
            surface_type=f"{surface_type}_{config.site_generator}",
            system_name=system_name,
            skip_existing=False,
        )
    elapsed = time.perf_counter() - t0
    energies: list[float] = []
    if campaign.run_results:
        for run in campaign.run_results:
            for r in run.results:
                energies.append(float(r.energy_adsorption))
    n_valid = len(energies)
    best = min(energies) if energies else None
    if campaign.molecule_summaries and best is None:
        s = campaign.molecule_summaries[0]
        best = s.best_adsorption_energy
        n_valid = s.n_valid_placements
    stats = _eads_stats(energies)
    return {
        "name": name,
        "plugin": config.site_generator,
        "elapsed_s": elapsed,
        "n_sites": n_sites,
        "n_valid": n_valid,
        "best_eads": best,
        "energies": energies,
        "stats": stats,
        "results_dir": results_dir,
    }


def run_e2e(
    num_placements: int,
    csv_path: Path | None = None,
    *,
    n_jobs: int = 1,
) -> None:
    print("\n" + "=" * 72)
    print(f"End-to-end demos (num_placements={num_placements}, n_jobs={n_jobs})")
    print("=" * 72)
    configure_logging(default_level="WARNING")

    results = []

    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="slab",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            n_jobs=n_jobs,
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
            results_dir=_fresh_results_dir(f"ab_h2_ru_{plugin}"),
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

    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="nanoparticle",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            n_jobs=n_jobs,
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
            results_dir=_fresh_results_dir(f"ab_h2_pt13_{plugin}"),
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
            n_jobs=n_jobs,
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            stage2_steps=200,
        )
        print(f"\n[CO2/MOF] run site_generator={plugin!r} ...")
        mof = prepare_substrate(
            slab=mof_atoms,
            config=cfg,
            results_dir=_fresh_results_dir(f"ab_co2_mof_{plugin}"),
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

    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="nanoparticle",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            num_placements=num_placements,
            n_jobs=n_jobs,
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            slab_relaxation_mode="none",
            stage2_steps=200,
        )
        print(f"\n[ethene/Ru55] run site_generator={plugin!r} ...")
        cluster = prepare_substrate(
            slab=_ru55(),
            config=cfg,
            results_dir=_fresh_results_dir(f"ab_ethene_ru55_{plugin}"),
        )
        results.append(
            _run_campaign(
                "ethene/Ru55",
                cfg,
                cluster,
                [("C=C", "ethene")],
                "ab_ethene_ru55",
                "Ru_55",
            )
        )

    # Slim camphor BO (reduced budget — full 25-batch twice is too heavy).
    for plugin in ("auto", "adaptive_grid"):
        cfg = AdsorptionConfig(
            material_type="slab",
            site_generator=plugin,
            seed=42,
            num_conformers=1,
            n_jobs=n_jobs,
            autobatcher_max_memory_padding=0.8,
            autobatcher_max_memory_scaler=500,
            autobatcher_max_atoms_to_try=5000,
            slab_relaxation_mode="none",
            placement_z_range=(2.0, 3.5),
            placement_z_scale_by_covalent_radius=False,
            binding_distance_threshold=5.0,
            top_layer_tolerance=2.1,
            stage2_steps=200,
            bo=BOConfig(total_budget=8, acquisition="ei"),
        )
        print(f"\n[camphor/Cu111 BO] run site_generator={plugin!r} ...")
        try:
            from camphor_cu111_binding_energy import (  # type: ignore[import-not-found]
                CAMPHOR_SMILES,
                prepare_campaign_slab,
            )

            slab = prepare_campaign_slab(
                cfg, results_directory=_fresh_results_dir(f"ab_camphor_{plugin}")
            )
        except Exception as exc:
            print(f"  skipping camphor (could not load paper slab: {exc})")
            continue
        results.append(
            _run_campaign(
                "camphor/Cu111",
                cfg,
                slab,
                [(CAMPHOR_SMILES, "camphor")],
                "ab_camphor",
                "Cu_111",
                bo=True,
            )
        )

    print("\n" + "=" * 72)
    print(
        f"{'system':14s} {'plugin':14s} {'t_s':>8s} {'n_sites':>8s} "
        f"{'n_valid':>8s} {'best':>9s} {'med':>9s} {'mean':>9s} {'std':>8s}"
    )
    for r in results:
        e = r["best_eads"]
        st = r["stats"]
        e_s = f"{e:9.4f}" if e is not None and np.isfinite(e) else f"{'n/a':>9s}"
        print(
            f"{r['name']:14s} {r['plugin']:14s} {r['elapsed_s']:8.1f} "
            f"{r['n_sites']:8d} {r['n_valid']:8d} {e_s} "
            f"{st['median']:9.4f} {st['mean']:9.4f} {st['std']:8.4f}"
        )
        print(f"  E_ads histogram ({r['name']} / {r['plugin']}):")
        print(_ascii_hist(r["energies"]))

    out = csv_path or Path(results_dir_for("adaptive_grid_ab")) / "eads.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "system",
                "plugin",
                "elapsed_s",
                "n_sites",
                "n_valid",
                "best_eads",
                "eads_min",
                "eads_p25",
                "eads_median",
                "eads_mean",
                "eads_p75",
                "eads_max",
                "eads_std",
            ],
        )
        writer.writeheader()
        for r in results:
            st = r["stats"]
            writer.writerow(
                {
                    "system": r["name"],
                    "plugin": r["plugin"],
                    "elapsed_s": f"{r['elapsed_s']:.3f}",
                    "n_sites": r["n_sites"],
                    "n_valid": r["n_valid"],
                    "best_eads": r["best_eads"],
                    "eads_min": st["min"],
                    "eads_p25": st["p25"],
                    "eads_median": st["median"],
                    "eads_mean": st["mean"],
                    "eads_p75": st["p75"],
                    "eads_max": st["max"],
                    "eads_std": st["std"],
                }
            )
    print(f"\nWrote E_ads summary CSV → {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run slim binding demos (GPU/MLIP) for auto vs adaptive_grid",
    )
    parser.add_argument(
        "--skip-site-ab",
        action="store_true",
        help="Skip the CPU site-catalog A/B (useful when re-running --e2e only)",
    )
    parser.add_argument(
        "--num-placements",
        type=int,
        default=12,
        help="Placements per e2e campaign (default 12)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=_DEFAULT_N_JOBS,
        help="Joblib-style n_jobs for site enumeration (default 1)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path for e2e E_ads summary CSV "
        "(default: results_adaptive_grid_ab/eads.csv)",
    )
    args = parser.parse_args()
    if not args.skip_site_ab:
        run_site_ab(n_jobs=args.n_jobs)
    if args.e2e:
        # Allow importing sibling example helpers (camphor slab loader).
        sys.path.insert(0, os.path.join(_ROOT, "examples"))
        run_e2e(args.num_placements, csv_path=args.csv, n_jobs=args.n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

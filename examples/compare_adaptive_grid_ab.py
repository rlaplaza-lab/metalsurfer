#!/usr/bin/env python3
"""A/B site generators: auto vs adaptive_grid vs rolling_probe (+ optional e2e).

Not part of the official example runner. Production defaults and A/B conclusions:
docs/guides/configuration.rst (site-generator section).

Run (conda env metalsurfer)::

  conda activate metalsurfer
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  python examples/compare_adaptive_grid_ab.py --n-jobs 1
  python examples/compare_adaptive_grid_ab.py --e2e --skip-site-ab --n-jobs 1 \\
      --csv results_adaptive_grid_ab/eads_gpu_full.csv \\
      --baseline-csv results_adaptive_grid_ab/eads_baseline_v1.csv
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


def _load_baseline_best(path: Path | None) -> dict[tuple[str, str], float]:
    """Map (system, plugin) → best_eads from a prior CSV (empty if missing)."""
    out: dict[tuple[str, str], float] = {}
    if path is None or not path.is_file():
        return out
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                best = float(row["best_eads"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(best):
                out[(row["system"], row["plugin"])] = best
    return out


def _e2e_common_kwargs(
    *,
    plugin: str,
    material_type: str,
    n_jobs: int,
    num_placements: int | None,
    fmax: float,
    stage2_steps: int,
    memory_padding: float,
) -> dict:
    """Shared AdsorptionConfig fields for GPU-full single-batch e2e demos."""
    return {
        "material_type": material_type,
        "site_generator": plugin,
        "seed": 42,
        "num_conformers": 1,
        "num_placements": num_placements,  # None → GPU autotune (one full batch)
        "n_jobs": n_jobs,
        "fmax": float(fmax),
        "stage2_steps": int(stage2_steps),
        "slab_relaxation_mode": "none",
        # Fill the GPU; do not pin an artificial scaler / atom cap.
        "autobatcher_max_memory_padding": float(memory_padding),
        "autobatcher_max_memory_scaler": None,
        "autobatcher_max_atoms_to_try": None,
    }


_E2E_PLUGINS = ("auto", "adaptive_grid", "rolling_probe")


def run_e2e(
    num_placements: int | None = None,
    csv_path: Path | None = None,
    *,
    n_jobs: int = 1,
    fmax: float = 0.05,
    stage2_steps: int = 200,
    bo_budget: int = 5,
    memory_padding: float = 0.8,
    baseline_csv: Path | None = None,
) -> None:
    """GPU binding demos: auto vs adaptive_grid vs rolling_probe.

    When *num_placements* is ``None``, workload autotune sets it to the probed
    single-batch GPU capacity. Relaxations use *fmax* / *stage2_steps*.
    Camphor BO uses *bo_budget* acquisition rounds with auto initial/batch sizes.
    """
    place_label = "auto(GPU)" if num_placements is None else str(num_placements)
    print("\n" + "=" * 72)
    print(
        f"End-to-end demos (num_placements={place_label}, fmax={fmax}, "
        f"stage2_steps={stage2_steps}, bo_budget={bo_budget}, "
        f"memory_padding={memory_padding}, n_jobs={n_jobs})"
    )
    print("=" * 72)
    configure_logging(default_level="WARNING")
    baseline = _load_baseline_best(baseline_csv)
    if baseline:
        print(f"Loaded {len(baseline)} baseline best-E_ads rows from {baseline_csv}")
    else:
        print("No baseline CSV loaded (regression checks skipped).")

    results = []

    for plugin in _E2E_PLUGINS:
        cfg = AdsorptionConfig(
            **_e2e_common_kwargs(
                plugin=plugin,
                material_type="slab",
                n_jobs=n_jobs,
                num_placements=num_placements,
                fmax=fmax,
                stage2_steps=stage2_steps,
                memory_padding=memory_padding,
            ),
            enable_dissociative_placement=True,
            skip_topology_check=True,
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

    for plugin in _E2E_PLUGINS:
        cfg = AdsorptionConfig(
            **_e2e_common_kwargs(
                plugin=plugin,
                material_type="nanoparticle",
                n_jobs=n_jobs,
                num_placements=num_placements,
                fmax=fmax,
                stage2_steps=stage2_steps,
                memory_padding=memory_padding,
            ),
            enable_dissociative_placement=True,
            skip_topology_check=True,
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
    for plugin in _E2E_PLUGINS:
        cfg = AdsorptionConfig(
            **_e2e_common_kwargs(
                plugin=plugin,
                material_type="porous",
                n_jobs=n_jobs,
                num_placements=num_placements,
                fmax=fmax,
                stage2_steps=stage2_steps,
                memory_padding=memory_padding,
            ),
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

    for plugin in _E2E_PLUGINS:
        cfg = AdsorptionConfig(
            **_e2e_common_kwargs(
                plugin=plugin,
                material_type="nanoparticle",
                n_jobs=n_jobs,
                num_placements=num_placements,
                fmax=fmax,
                stage2_steps=stage2_steps,
                memory_padding=memory_padding,
            ),
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

    # Camphor BO: fewer acquisition rounds, GPU-sized initial/batch.
    for plugin in _E2E_PLUGINS:
        cfg = AdsorptionConfig(
            **_e2e_common_kwargs(
                plugin=plugin,
                material_type="slab",
                n_jobs=n_jobs,
                num_placements=num_placements,
                fmax=fmax,
                stage2_steps=stage2_steps,
                memory_padding=memory_padding,
            ),
            placement_z_range=(2.0, 3.5),
            placement_z_scale_by_covalent_radius=False,
            binding_distance_threshold=5.0,
            top_layer_tolerance=2.1,
            bo=BOConfig(
                total_budget=int(bo_budget),
                acquisition="ei",
                # None → autotune to one full GPU batch each.
                initial_random=None,
                batch_size=None,
            ),
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
        f"{'n_valid':>8s} {'best':>9s} {'med':>9s} {'base':>9s} {'Δbest':>9s}"
    )
    regressions: list[str] = []
    for r in results:
        e = r["best_eads"]
        st = r["stats"]
        e_s = f"{e:9.4f}" if e is not None and np.isfinite(e) else f"{'n/a':>9s}"
        base = baseline.get((r["name"], r["plugin"]))
        if base is not None and np.isfinite(base):
            base_s = f"{base:9.4f}"
            if e is not None and np.isfinite(e):
                delta = float(e) - float(base)
                # More positive Δ = worse (less bound). Flag if worse by >1 meV.
                delta_s = f"{delta:+9.4f}"
                if delta > 0.001:
                    regressions.append(
                        f"{r['name']}/{r['plugin']}: best {e:.4f} vs baseline "
                        f"{base:.4f} (Δ={delta:+.4f} eV)"
                    )
            else:
                delta_s = f"{'n/a':>9s}"
        else:
            base_s = f"{'n/a':>9s}"
            delta_s = f"{'n/a':>9s}"
        print(
            f"{r['name']:14s} {r['plugin']:14s} {r['elapsed_s']:8.1f} "
            f"{r['n_sites']:8d} {r['n_valid']:8d} {e_s} "
            f"{st['median']:9.4f} {base_s} {delta_s}"
        )
        print(f"  E_ads histogram ({r['name']} / {r['plugin']}):")
        print(_ascii_hist(r["energies"]))

    if regressions:
        print("\n" + "=" * 72)
        print(f"REGRESSIONS vs baseline ({len(regressions)}):")
        for line in regressions:
            print(f"  ! {line}")
    elif baseline:
        print("\nNo best-E_ads regressions vs baseline (> +1 meV).")

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
                "baseline_best",
                "delta_best",
            ],
        )
        writer.writeheader()
        for r in results:
            st = r["stats"]
            base = baseline.get((r["name"], r["plugin"]))
            e = r["best_eads"]
            delta = (
                float(e) - float(base)
                if (
                    e is not None
                    and base is not None
                    and np.isfinite(e)
                    and np.isfinite(base)
                )
                else ""
            )
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
                    "baseline_best": base if base is not None else "",
                    "delta_best": delta,
                }
            )
    print(f"\nWrote E_ads summary CSV → {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run binding demos (GPU/MLIP) for auto vs adaptive_grid vs rolling_probe",
    )
    parser.add_argument(
        "--skip-site-ab",
        action="store_true",
        help="Skip the CPU site-catalog A/B (useful when re-running --e2e only)",
    )
    parser.add_argument(
        "--num-placements",
        type=int,
        default=None,
        help="Placements per e2e campaign (default: None = GPU autotune / one full batch)",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence for placement relaxations (default 0.05)",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=200,
        help="Max optimizer steps for placement relaxations (default 200)",
    )
    parser.add_argument(
        "--bo-budget",
        type=int,
        default=5,
        help="BO acquisition rounds for camphor (default 5); initial/batch autotuned",
    )
    parser.add_argument(
        "--memory-padding",
        type=float,
        default=0.8,
        help="Autobatcher memory padding fraction (default 0.8)",
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
        help="Output path for e2e E_ads summary CSV "
        "(default: results_adaptive_grid_ab/eads.csv)",
    )
    parser.add_argument(
        "--baseline-csv",
        type=Path,
        default=Path(_ROOT) / "results_adaptive_grid_ab" / "eads_side_by_side.csv",
        help="Prior E_ads CSV for best-energy regression checks",
    )
    args = parser.parse_args()
    if not args.skip_site_ab:
        run_site_ab(n_jobs=args.n_jobs)
    if args.e2e:
        # Allow importing sibling example helpers (camphor slab loader).
        sys.path.insert(0, os.path.join(_ROOT, "examples"))
        run_e2e(
            args.num_placements,
            csv_path=args.csv,
            n_jobs=args.n_jobs,
            fmax=args.fmax,
            stage2_steps=args.stage2_steps,
            bo_budget=args.bo_budget,
            memory_padding=args.memory_padding,
            baseline_csv=args.baseline_csv,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

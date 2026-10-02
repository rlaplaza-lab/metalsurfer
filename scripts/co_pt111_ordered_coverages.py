#!/usr/bin/env python3
"""CO on Pt(111) ordered coverages in the spirit of Gunasooriya & Saeys (ACS Catal. 2018).

Builds the literature cells, places CO carbon-down on the stated sites, and reports
whole-adlayer E_ads per CO with uma-s-1p2/oc25 (slab frozen). Two adsorbate
relaxations are recorded:

* free — CO fully free, Pt frozen (may leave the published sites)
* constrained — each C atom FixedLine along the surface normal (site-preserving
  height/tilt relaxation); O free; Pt frozen

Also evaluates same-cell alternatives (√3 bridge/hollow, c(4×2) all-atop,
c(√3×3)rect all-atop) under the constrained protocol, and optionally screens
unconstrained placements on (√3×√3) and c(4×2).

Run from the metalsurfer project root (conda env metalsurfer)::

    python scripts/co_pt111_ordered_coverages.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111, make_supercell
from ase.constraints import FixAtoms, FixedLine
from ase.io import write
from ase.optimize import LBFGS

from metalsurfer import AdsorptionConfig, configure_logging, run_adsorption
from metalsurfer.optimization import setup_single_model
from metalsurfer.surface_prep import apply_surface_constraints, create_slab_from_atoms

RESULTS = Path("results_co_pt111_ordered")
XYZ_DIR = RESULTS / "xyz"
VACUUM = 12.0
N_LAYERS = 4
A_PT = 3.967  # Å; near FairChem / experimental Pt lattice
SITE_CUTOFF = 2.5


@dataclass(frozen=True)
class OrderedCase:
    label: str
    theta_ml: float
    n_surface: int
    n_co: int
    description: str
    bridge_top_expected: str
    expected_atop: int
    expected_bridge: int
    expected_hollow: int = 0


def _primitive_pt111() -> Atoms:
    return fcc111("Pt", size=(1, 1, N_LAYERS), a=A_PT, vacuum=VACUUM, periodic=True)


def _add_co_at(slab: Atoms, xy: tuple[float, float], height: float = 1.85) -> Atoms:
    """Place one upright CO (C down) at cartesian (x, y)."""
    out = slab.copy()
    add_adsorbate(out, "C", height=height, position=xy)
    c_idx = len(out) - 1
    o_pos = out.positions[c_idx] + np.array([0.0, 0.0, 1.14])
    out.append("O")
    out.positions[-1] = o_pos
    return out


def _top_layer_pts(atoms: Atoms) -> np.ndarray:
    symbols = np.array(atoms.get_chemical_symbols())
    pt = np.where(symbols == "Pt")[0]
    z = atoms.positions[pt, 2]
    zmax = z.max()
    return pt[z > zmax - 0.5]


def _sorted_top_xy(atoms: Atoms) -> np.ndarray:
    tops = _top_layer_pts(atoms)
    pos = atoms.positions[tops][:, :2].copy()
    order = np.lexsort((pos[:, 1], pos[:, 0]))
    return pos[order]


def _cell2d(cell: np.ndarray) -> np.ndarray:
    """2×2 in-plane cell with ASE row-vector lattice vectors."""
    return np.asarray(cell[:2, :2], dtype=float)


def _mic_delta(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Minimum-image cartesian displacement b − a in the surface plane."""
    cell2 = _cell2d(cell)
    # ASE: cart = frac @ cell2  →  frac = cart @ inv(cell2)
    dfrac = (b - a) @ np.linalg.inv(cell2)
    dfrac -= np.round(dfrac)
    return dfrac @ cell2


def _nn_pairs(xy: np.ndarray, cell: np.ndarray) -> list[tuple[int, int, float]]:
    """Nearest-neighbor pairs under 2D minimum-image convention."""
    n = len(xy)
    pairs: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(_mic_delta(xy[i], xy[j], cell)))
            pairs.append((i, j, dist))
    pairs.sort(key=lambda t: t[2])
    return pairs


def _bridge_midpoint(xy: np.ndarray, cell: np.ndarray, i: int, j: int) -> tuple[float, float]:
    mid = xy[i] + 0.5 * _mic_delta(xy[i], xy[j], cell)
    return float(mid[0]), float(mid[1])


def _primitive_site_xy(site: str) -> tuple[float, float]:
    """Cartesian (x, y) of an ASE fcc111 adsorption site on the primitive cell."""
    ref = fcc111("Pt", size=(1, 1, N_LAYERS), a=A_PT, vacuum=VACUUM, periodic=True)
    info = ref.info["adsorbate_info"]
    frac = np.asarray(info["sites"][site], dtype=float)
    xy = frac @ np.asarray(info["cell"], dtype=float)
    return float(xy[0]), float(xy[1])


def _site_xy(atoms: Atoms, kind: str, which: int = 0) -> tuple[float, float]:
    """Cartesian (x, y) for atop / bridge / hollow on the top Pt layer.

    Sites are taken from ASE's primitive fcc111 adsorbate dictionary and
    translated onto the *which*-th top-layer atom (sorted by x then y).
    """
    xy = _sorted_top_xy(atoms)
    origin = xy[which % len(xy)]
    ase_name = {"atop": "ontop", "bridge": "bridge", "hollow": "fcc"}[kind]
    sx, sy = _primitive_site_xy(ase_name)
    return float(origin[0] + sx), float(origin[1] + sy)


def build_p3x3() -> Atoms:
    return fcc111("Pt", size=(3, 3, N_LAYERS), a=A_PT, vacuum=VACUUM, periodic=True)


def build_sqrt3() -> Atoms:
    """(√3×√3)R30° cell: 3 surface Pt."""
    prim = _primitive_pt111()
    return make_supercell(prim, [[2, 1, 0], [-1, 1, 0], [0, 0, 1]])


def build_c42() -> Atoms:
    """c(4×2) orthogonal cell: 8 surface Pt."""
    return fcc111("Pt", size=(4, 2, N_LAYERS), a=A_PT, vacuum=VACUUM, orthogonal=True, periodic=True)


def build_csqrt3x5() -> Atoms:
    """c(√3×5)rect: 10 surface Pt (5×2 orthogonal)."""
    return fcc111("Pt", size=(5, 2, N_LAYERS), a=A_PT, vacuum=VACUUM, orthogonal=True, periodic=True)


def build_csqrt3x3() -> Atoms:
    """c(√3×3)rect: 6 surface Pt (3×2 orthogonal)."""
    return fcc111("Pt", size=(3, 2, N_LAYERS), a=A_PT, vacuum=VACUUM, orthogonal=True, periodic=True)


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


def _verify_registry(label: str, atoms: Atoms, case: OrderedCase) -> None:
    """Assert initial placement matches the literature site counts."""
    n_a, n_b, n_h = _classify_sites(atoms)
    n_surf = _n_surface(atoms)
    if n_surf != case.n_surface:
        raise RuntimeError(f"{label}: surface Pt {n_surf} != expected {case.n_surface}")
    if (n_a, n_b, n_h) != (case.expected_atop, case.expected_bridge, case.expected_hollow):
        raise RuntimeError(
            f"{label}: sites atop/bridge/hollow = {n_a}/{n_b}/{n_h}, "
            f"expected {case.expected_atop}/{case.expected_bridge}/{case.expected_hollow}"
        )
    print(f"  registry OK: {_bridge_top_string(n_a, n_b, n_h)} on {n_surf} surface Pt")


def _single_point(calculator, atoms: Atoms) -> float:
    atoms = atoms.copy()
    atoms.calc = calculator
    calculator.calculate(atoms, properties=["energy"])
    return float(calculator.results["energy"])


def _relax_adsorbates_free(calculator, atoms: Atoms, fmax: float = 0.05, steps: int = 200) -> Atoms:
    """Freeze all Pt; relax CO freely."""
    out = atoms.copy()
    symbols = np.array(out.get_chemical_symbols())
    frozen = [i for i, s in enumerate(symbols) if s == "Pt"]
    out.set_constraint(FixAtoms(indices=frozen))
    out.calc = calculator
    opt = LBFGS(out, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return out


def _relax_adsorbates_constrained(
    calculator, atoms: Atoms, fmax: float = 0.05, steps: int = 200
) -> Atoms:
    """Freeze Pt; pin each C to a FixedLine along z; leave O free."""
    out = atoms.copy()
    symbols = np.array(out.get_chemical_symbols())
    constraints = [FixAtoms(indices=[i for i, s in enumerate(symbols) if s == "Pt"])]
    for i, s in enumerate(symbols):
        if s == "C":
            constraints.append(FixedLine(i, direction=[0.0, 0.0, 1.0]))
    out.set_constraint(constraints)
    out.calc = calculator
    opt = LBFGS(out, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return out


def _n_surface(atoms: Atoms) -> int:
    return int(len(_top_layer_pts(atoms)))


def _place_from_sites(
    slab: Atoms,
    sites: list[tuple[str, tuple[float, float], float]],
) -> Atoms:
    out = slab.copy()
    for _kind, xy, h in sites:
        out = _add_co_at(out, xy, height=h)
    return out


def place_ordered(label: str) -> Atoms:
    """Literature-facing registries with verified B:T ratios."""
    if label == "p3x3_atop":
        slab = build_p3x3()
        return _add_co_at(slab, _site_xy(slab, "atop", 0))
    if label == "p3x3_bridge":
        slab = build_p3x3()
        return _add_co_at(slab, _site_xy(slab, "bridge", 0), height=1.55)
    if label == "p3x3_hollow":
        slab = build_p3x3()
        return _add_co_at(slab, _site_xy(slab, "hollow", 0), height=1.35)
    if label == "sqrt3_atop":
        slab = build_sqrt3()
        return _add_co_at(slab, _site_xy(slab, "atop", 0))
    if label == "sqrt3_bridge":
        slab = build_sqrt3()
        return _add_co_at(slab, _site_xy(slab, "bridge", 0), height=1.55)
    if label == "sqrt3_hollow":
        slab = build_sqrt3()
        return _add_co_at(slab, _site_xy(slab, "hollow", 0), height=1.35)
    if label == "c42_2top2bridge":
        # Classic c(4×2)-4CO: equal top/bridge (B:T = 2:2).
        # On a (4×2) orthogonal cell, place atop on rows 0 and 2 of the sorted
        # top layer (indices 0 and 4), and bridges between (1,2) and (5,6).
        slab = build_c42()
        xy = _sorted_top_xy(slab)
        cell = slab.cell.array
        assert len(xy) == 8, len(xy)
        sites = [
            ("atop", (float(xy[0, 0]), float(xy[0, 1])), 1.85),
            ("atop", (float(xy[4, 0]), float(xy[4, 1])), 1.85),
            ("bridge", _bridge_midpoint(xy, cell, 1, 2), 1.55),
            ("bridge", _bridge_midpoint(xy, cell, 5, 6), 1.55),
        ]
        return _place_from_sites(slab, sites)
    if label == "c42_all_atop":
        slab = build_c42()
        xy = _sorted_top_xy(slab)
        # Four evenly spaced atop sites on the 8-atom cell.
        sites = [
            ("atop", (float(xy[i, 0]), float(xy[i, 1])), 1.85) for i in (0, 2, 4, 6)
        ]
        return _place_from_sites(slab, sites)
    if label == "csqrt3x5_4top2bridge":
        # c(√3×5)rect-6CO with B:T = 1:2 (2 bridge + 4 atop) on 10 surface Pt.
        slab = build_csqrt3x5()
        xy = _sorted_top_xy(slab)
        cell = slab.cell.array
        assert len(xy) == 10, len(xy)
        # Atop on every other atom along the long axis; bridges fill the gaps.
        sites = [
            ("atop", (float(xy[i, 0]), float(xy[i, 1])), 1.85) for i in (0, 2, 5, 7)
        ]
        sites.append(("bridge", _bridge_midpoint(xy, cell, 1, 3), 1.55))
        sites.append(("bridge", _bridge_midpoint(xy, cell, 6, 8), 1.55))
        return _place_from_sites(slab, sites)
    if label == "csqrt3x3_3top1bridge":
        # c(√3×3)rect-4CO with B:T = 1:3 on 6 surface Pt.
        slab = build_csqrt3x3()
        xy = _sorted_top_xy(slab)
        cell = slab.cell.array
        assert len(xy) == 6, len(xy)
        sites = [
            ("atop", (float(xy[i, 0]), float(xy[i, 1])), 1.85) for i in (0, 2, 4)
        ]
        sites.append(("bridge", _bridge_midpoint(xy, cell, 1, 3), 1.55))
        return _place_from_sites(slab, sites)
    if label == "csqrt3x3_all_atop":
        slab = build_csqrt3x3()
        xy = _sorted_top_xy(slab)
        # Four atop on a 6-atom cell (matches the free-relaxation collapse).
        sites = [
            ("atop", (float(xy[i, 0]), float(xy[i, 1])), 1.85) for i in (0, 1, 3, 4)
        ]
        return _place_from_sites(slab, sites)
    raise KeyError(label)


# Literature series shown in the main-text figure (constrained + free).
LITERATURE_CASES = [
    OrderedCase("p3x3_atop", 1 / 9, 9, 1, "p(3×3) atop", "atop×1", 1, 0),
    OrderedCase("p3x3_bridge", 1 / 9, 9, 1, "p(3×3) bridge", "bridge×1", 0, 1),
    OrderedCase("p3x3_hollow", 1 / 9, 9, 1, "p(3×3) hollow", "hollow×1", 0, 0, 1),
    OrderedCase("sqrt3_atop", 1 / 3, 3, 1, "(√3×√3)R30° atop", "atop×1", 1, 0),
    OrderedCase("c42_2top2bridge", 0.5, 8, 4, "c(4×2) 2 atop + 2 bridge", "B:T=2:2", 2, 2),
    OrderedCase(
        "csqrt3x5_4top2bridge", 0.6, 10, 6, "c(√3×5)rect 4 atop + 2 bridge", "B:T=2:4", 4, 2
    ),
    OrderedCase(
        "csqrt3x3_3top1bridge", 2 / 3, 6, 4, "c(√3×3)rect 3 atop + 1 bridge", "B:T=1:3", 3, 1
    ),
]

# Same-cell alternatives evaluated under constrained relaxation only.
ALTERNATIVE_CASES = [
    OrderedCase("sqrt3_bridge", 1 / 3, 3, 1, "(√3×√3)R30° bridge", "bridge×1", 0, 1),
    OrderedCase("sqrt3_hollow", 1 / 3, 3, 1, "(√3×√3)R30° hollow", "hollow×1", 0, 0, 1),
    OrderedCase("c42_all_atop", 0.5, 8, 4, "c(4×2) 4 atop", "atop×4", 4, 0),
    OrderedCase("csqrt3x3_all_atop", 2 / 3, 6, 4, "c(√3×3)rect 4 atop", "atop×4", 4, 0),
]


def _row(
    series: str,
    case: OrderedCase,
    e_ads_total: float,
    n_a: int,
    n_b: int,
    n_h: int,
) -> dict[str, object]:
    e_per = e_ads_total / case.n_co
    return {
        "series": series,
        "label": case.label,
        "description": case.description,
        "theta_ml": f"{case.theta_ml:.6f}",
        "n_surface": case.n_surface,
        "n_co": case.n_co,
        "e_ads_total_eV": f"{e_ads_total:.8f}",
        "e_ads_per_co_eV": f"{e_per:.8f}",
        "bridge_top_ratio": _bridge_top_string(n_a, n_b, n_h),
        "n_atop": n_a,
        "n_bridge": n_b,
        "n_hollow": n_h,
        "expected_ratio": case.bridge_top_expected,
    }


def _slab_energy_cache(calculator, adslab0: Atoms, n_surface: int, cache: dict[str, float]) -> float:
    symbols = np.array(adslab0.get_chemical_symbols())
    slab = adslab0[symbols == "Pt"]
    slab.set_pbc(True)
    slab.cell = adslab0.cell
    key = f"{n_surface}"
    if key not in cache:
        cache[key] = _single_point(calculator, slab)
        print(f"  E_slab ({n_surface} surface Pt) = {cache[key]:.6f} eV")
    return cache[key]


def run_seeded_relaxations(calculator) -> list[dict[str, object]]:
    XYZ_DIR.mkdir(parents=True, exist_ok=True)
    slab_energy: dict[str, float] = {}
    co = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.14]], cell=[20, 20, 20], pbc=True)
    e_co = _single_point(calculator, co)
    print(f"E_CO (gas) = {e_co:.6f} eV")

    rows: list[dict[str, object]] = []

    for case in LITERATURE_CASES:
        print(f"\n=== {case.label}: {case.description} ===")
        adslab0 = place_ordered(case.label)
        _verify_registry(case.label, adslab0, case)
        e_slab = _slab_energy_cache(calculator, adslab0, case.n_surface, slab_energy)
        write(XYZ_DIR / f"{case.label}_initial.xyz", adslab0)

        # Free adsorbate relaxation (may leave the site).
        free = _relax_adsorbates_free(calculator, adslab0)
        write(XYZ_DIR / f"{case.label}_relaxed.xyz", free)
        e_free = _single_point(calculator, free) - e_slab - case.n_co * e_co
        n_a, n_b, n_h = _classify_sites(free)
        print(
            f"  free:         E_ads/CO = {e_free / case.n_co:+.4f} eV   "
            f"sites {_bridge_top_string(n_a, n_b, n_h)}"
        )
        rows.append(_row("ordered", case, e_free, n_a, n_b, n_h))

        # Site-preserving height relaxation.
        held = _relax_adsorbates_constrained(calculator, adslab0)
        write(XYZ_DIR / f"{case.label}_constrained.xyz", held)
        e_held = _single_point(calculator, held) - e_slab - case.n_co * e_co
        n_a, n_b, n_h = _classify_sites(held)
        print(
            f"  constrained:  E_ads/CO = {e_held / case.n_co:+.4f} eV   "
            f"sites {_bridge_top_string(n_a, n_b, n_h)}"
        )
        if (n_a, n_b, n_h) != (case.expected_atop, case.expected_bridge, case.expected_hollow):
            raise RuntimeError(
                f"{case.label} constrained relaxation left the registry: "
                f"{n_a}/{n_b}/{n_h}"
            )
        rows.append(_row("constrained", case, e_held, n_a, n_b, n_h))

    for case in ALTERNATIVE_CASES:
        print(f"\n=== alt {case.label}: {case.description} ===")
        adslab0 = place_ordered(case.label)
        _verify_registry(case.label, adslab0, case)
        e_slab = _slab_energy_cache(calculator, adslab0, case.n_surface, slab_energy)
        write(XYZ_DIR / f"{case.label}_initial.xyz", adslab0)
        held = _relax_adsorbates_constrained(calculator, adslab0)
        write(XYZ_DIR / f"{case.label}_constrained.xyz", held)
        e_held = _single_point(calculator, held) - e_slab - case.n_co * e_co
        n_a, n_b, n_h = _classify_sites(held)
        print(
            f"  constrained:  E_ads/CO = {e_held / case.n_co:+.4f} eV   "
            f"sites {_bridge_top_string(n_a, n_b, n_h)}"
        )
        if (n_a, n_b, n_h) != (case.expected_atop, case.expected_bridge, case.expected_hollow):
            raise RuntimeError(
                f"{case.label} constrained relaxation left the registry: "
                f"{n_a}/{n_b}/{n_h}"
            )
        rows.append(_row("alternative", case, e_held, n_a, n_b, n_h))

    return rows


def run_unbiased_screens() -> list[dict[str, object]]:
    """Modest run_adsorption on √3 and c(4×2) cells (cells expanded for PBC sizing)."""
    rows: list[dict[str, object]] = []
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=50,
        stage1_steps=30,
        stage2_steps=60,
        slab_relaxation_mode="none",
        min_pbc_image_separation=2.0,
    )
    builders = (
        ("sqrt3_screen", lambda: build_sqrt3().repeat((2, 2, 1))),
        ("c42_screen", lambda: build_c42().repeat((1, 2, 1))),
    )
    for label, builder in builders:
        print(f"\n=== unbiased screen {label} ===")
        slab_atoms = builder()
        slab_atoms.set_pbc([True, True, False])
        slab_atoms = apply_surface_constraints(slab_atoms, relax_top_layer=False)
        container = create_slab_from_atoms(slab_atoms, align=True)
        campaign = run_adsorption(
            slab=container,
            molecules=[("[C-]#[O+]", "CO")],
            config=config,
            surface_type=f"co_pt111_{label}",
        )
        if not campaign.run_results:
            print(f"  no results for {label}")
            continue
        run = campaign.run_results[0]
        best = min(run.results, key=lambda r: r.energy_adsorption) if run.results else None
        if best is None:
            print(f"  empty pool for {label}")
            continue
        print(f"  best E_ads = {best.energy_adsorption:+.4f} eV  (n={len(run.results)})")
        rows.append(
            {
                "series": "screen",
                "label": label,
                "description": f"unbiased {label}",
                "theta_ml": "",
                "n_surface": _n_surface(slab_atoms),
                "n_co": 1,
                "e_ads_total_eV": f"{best.energy_adsorption:.8f}",
                "e_ads_per_co_eV": f"{best.energy_adsorption:.8f}",
                "bridge_top_ratio": "",
                "n_atop": "",
                "n_bridge": "",
                "n_hollow": "",
                "expected_ratio": "search",
            }
        )
    return rows


def main() -> int:
    configure_logging(default_level="INFO")
    RESULTS.mkdir(parents=True, exist_ok=True)
    calculator, _model = setup_single_model("uma-s-1p2", "cuda", task_name="oc25")
    rows = run_seeded_relaxations(calculator)
    # Reuse existing unbiased screens if present; skip re-running them.
    old_csv = RESULTS / "ordered_coverages.csv"
    if old_csv.is_file():
        with old_csv.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("series") == "screen":
                    rows.append(row)
                    print(f"kept prior screen row: {row['label']}")
    else:
        try:
            rows.extend(run_unbiased_screens())
        except Exception as exc:  # noqa: BLE001 — keep ordered results if screen fails
            print(f"unbiased screens failed: {exc}")

    out_csv = RESULTS / "ordered_coverages.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        "model": "uma-s-1p2",
        "task": "oc25",
        "a_pt": A_PT,
        "n_layers": N_LAYERS,
        "protocols": ["ordered=free", "constrained=FixedLine_C_z", "alternative=constrained"],
    }
    (RESULTS / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

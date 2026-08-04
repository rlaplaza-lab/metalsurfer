"""Tests for spglib-backed symmetry analysis (periodic slabs and clusters)."""

import numpy as np
import pytest
import spglib
from ase import Atoms
from ase.build import bulk, fcc111, graphene

from metalsurfer.placement import get_symmetry_aware_sites, get_unified_sites
from metalsurfer.surface_prep import create_slab_from_atoms
from metalsurfer.symmetry import SymmetryAnalysisError, SymmetryAnalyzer

from .conftest import make_nanoparticle, make_slab


def _symop_roundtrip_preserves_structure(
    cell: np.ndarray,
    frac: np.ndarray,
    numbers: np.ndarray,
    symprec: float,
) -> None:
    """Every spglib symmetry operation is a permutation of atoms (MIC, same species)."""
    ds = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=symprec)
    assert ds is not None
    for R, t in zip(ds.rotations, ds.translations, strict=True):
        Rf = np.asarray(R, dtype=float)
        tf = np.asarray(t, dtype=float)
        frac_new = frac @ Rf.T + tf
        frac_new %= 1.0
        n = len(frac)
        used: set[int] = set()
        for i in range(n):
            matched = False
            for j in range(n):
                if j in used or numbers[i] != numbers[j]:
                    continue
                d = frac_new[i] - frac[j]
                d -= np.round(d)
                sep = d @ cell
                if float(np.linalg.norm(sep)) < symprec * 50:
                    used.add(j)
                    matched = True
                    break
            assert matched, f"symop did not match atom {i}"
        assert len(used) == n


def _raw_indices_matching_orbit(
    equiv_xy: list,
    raw_sites: list,
    tol: float = 1e-4,
) -> list[int]:
    """Indices of raw sites whose xy lies on an equivalent position in this orbit."""
    idxs: list[int] = []
    for k, s in enumerate(raw_sites):
        sxy = np.asarray(s.xy, dtype=float)
        for xy in equiv_xy:
            if np.linalg.norm(sxy - np.asarray(xy, dtype=float)) < tol:
                idxs.append(k)
                break
    return sorted(set(idxs))


def _assert_orbit_pairwise_symops(
    analyzer: SymmetryAnalyzer,
    raw_sites: list,
    planar: bool,
) -> None:
    """Each orbit from ``analyze_site_symmetry`` is pairwise connected by one symop."""
    frac_ops = analyzer._frac_ops_from_dataset()
    cart_pts = [analyzer._site_3d_cart(s) for s in raw_sites]
    site_types = [str(s.site_type) for s in raw_sites]
    grouped = analyzer.analyze_site_symmetry(raw_sites, planar=planar)
    assert sum(int(g.symmetry_multiplicity or 0) for g in grouped) == len(raw_sites)
    for g in grouped:
        idxs = _raw_indices_matching_orbit(
            list(g.symmetry_equivalent_sites or ()), raw_sites
        )
        assert len(idxs) == g.symmetry_multiplicity
        for ii, i in enumerate(idxs):
            for j in idxs[ii + 1 :]:
                assert analyzer._site_pair_connected_by_ops(
                    i, j, cart_pts, frac_ops, planar, site_types=site_types
                )


def _assert_op_count_matches_spglib(analyzer: SymmetryAnalyzer, symprec: float) -> None:
    cell, frac, numbers = analyzer.get_spglib_cell_tuple()
    ds = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=symprec)
    assert ds is not None
    assert len(analyzer.detect_symmetry_operations()) == len(ds.rotations)


def test_bulk_cu_space_group_and_symop_count():
    """FCC primitive Cu has Fm-3m (225) and 48 symmetry operations."""
    atoms = bulk("Cu", "fcc", a=3.6)
    an = SymmetryAnalyzer(atoms, symmetry_tolerance=0.01)
    info = an.get_symmetry_info()
    assert info["spacegroup_number"] == 225
    assert info["international_symbol"] == "Fm-3m"
    assert info["symmetry_mode"] == "periodic"
    assert len(an.detect_symmetry_operations()) == 48


def test_hexagonal_cell_not_p1():
    """Off-diagonal in-plane cell (graphene net) yields non-trivial symmetry."""
    atoms = graphene(vacuum=10.0)
    an = SymmetryAnalyzer(atoms, symmetry_tolerance=0.1)
    info = an.get_symmetry_info()
    assert info["spacegroup_number"] is not None
    assert int(info["spacegroup_number"]) != 1
    assert len(an.detect_symmetry_operations()) > 1


def test_spglib_symop_roundtrip_bulk_cu():
    """Symmetry operations permute atoms with MIC (FCC Cu)."""
    atoms = bulk("Cu", "fcc", a=3.6)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    inv = np.linalg.inv(cell)
    frac = atoms.get_positions() @ inv.T
    numbers = np.array(atoms.get_atomic_numbers(), dtype=int)
    _symop_roundtrip_preserves_structure(cell, frac, numbers, symprec=1e-4)


def test_fcc111_slab_consistent_symmetry():
    """ASE fcc111 slab may be P1 (asymmetric surfaces); result is deterministic."""
    slab = fcc111("Al", size=(2, 2, 3), vacuum=7.0)
    a1 = SymmetryAnalyzer(slab, symmetry_tolerance=0.1)
    a2 = SymmetryAnalyzer(slab, symmetry_tolerance=0.1)
    assert (
        a1.get_symmetry_info()["spacegroup_number"]
        == a2.get_symmetry_info()["spacegroup_number"]
    )
    assert len(a1.detect_symmetry_operations()) == len(a2.detect_symmetry_operations())


def test_equivalent_atoms_bulk_cu():
    """Single-atom primitive cell: one Wyckoff class."""
    atoms = bulk("Cu", "fcc", a=3.6)
    an = SymmetryAnalyzer(atoms, 0.01)
    groups = an.find_equivalent_atoms()
    assert len(groups) == 1
    assert groups[0] == [0]


def test_get_symmetry_aware_sites_multiplicity_partition():
    """Every raw site lies in exactly one orbit; multiplicities sum to raw count."""
    slab = make_slab(nx=2, ny=2)
    raw = get_unified_sites(slab, material_type="slab")
    assert raw is not None and len(raw) >= 1
    sym = get_symmetry_aware_sites(slab)
    assert len(sym) >= 1
    assert sum((s.symmetry_multiplicity or 0) for s in sym) == len(raw)
    assert len(sym) <= len(raw)


def test_cluster_methane_td_operation_count():
    """Tetrahedral CH4 in a box: 24 symmetry operations (Td order)."""
    a = 1.09
    methane = Atoms(
        "CH4",
        positions=[
            [0, 0, 0],
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a],
        ],
    )
    methane.set_cell([20, 20, 20])
    methane.set_pbc([False, False, False])
    an = SymmetryAnalyzer(methane, symmetry_tolerance=0.15, mode="cluster")
    assert an.get_symmetry_info()["symmetry_mode"] == "cluster"
    assert len(an.detect_symmetry_operations()) == 24


def test_cluster_octahedron_oh_operation_count():
    """Octahedral Ni6 (vertices): Oh order 48 in cubic box."""
    pos = []
    for s in (1, -1):
        pos.extend([[s, 0, 0], [0, s, 0], [0, 0, s]])
    atoms = Atoms("Ni6", positions=pos)
    atoms.set_cell([20, 20, 20])
    atoms.set_pbc([False, False, False])
    an = SymmetryAnalyzer(atoms, symmetry_tolerance=0.2, mode="cluster")
    assert len(an.detect_symmetry_operations()) == 48


def test_detect_symmetry_breaking_distorted_cu():
    """Breaking FCC primitive Cu symmetry is detected."""
    ref = bulk("Cu", "fcc", a=3.6)
    cur = ref.copy()
    cur.positions[0] += 0.5
    an = SymmetryAnalyzer(cur, symmetry_tolerance=0.01)
    assert an.detect_symmetry_breaking(ref)


def test_detect_symmetry_breaking_identical_false():
    """Identical structures are not flagged as broken."""
    ref = bulk("Cu", "fcc", a=3.6)
    an = SymmetryAnalyzer(ref.copy(), symmetry_tolerance=0.01)
    assert not an.detect_symmetry_breaking(ref)


def test_get_symmetry_aware_sites_nanoparticle_envelope():
    """Non-periodic cluster: envelope sites with cluster symmetry analyzer."""
    atoms = make_nanoparticle()  # Au₁₃ icosahedral
    raw = get_unified_sites(
        atoms, material_type="nanoparticle", top_layer_tolerance=2.0
    )
    assert raw and len(raw) >= 1
    sites = get_symmetry_aware_sites(
        atoms,
        top_layer_tolerance=2.0,
        symmetry_tolerance=0.3,
        material_type="nanoparticle",
        raw_sites=raw,
    )
    assert len(sites) >= 1
    # Orbit multiplicities must partition the raw site list exactly.
    assert sum(int(s.symmetry_multiplicity or 0) for s in sites) == len(raw)
    assert all(int(s.symmetry_multiplicity or 0) >= 1 for s in sites)


def test_cube_nanoparticle_symmetry_reduces_redundant_sites_deterministically():
    """Icosahedral Au₁₃ should collapse redundant sites consistently."""
    atoms = make_nanoparticle()  # Au₁₃ icosahedral
    raw = get_unified_sites(
        atoms, material_type="nanoparticle", top_layer_tolerance=2.0
    )

    sites1 = get_symmetry_aware_sites(
        atoms,
        top_layer_tolerance=2.0,
        symmetry_tolerance=0.1,
        material_type="nanoparticle",
        raw_sites=raw,
    )
    sites2 = get_symmetry_aware_sites(
        atoms,
        top_layer_tolerance=2.0,
        symmetry_tolerance=0.1,
        material_type="nanoparticle",
        raw_sites=raw,
    )

    assert len(sites1) == len(sites2)
    assert len(sites1) >= 1
    assert sum(int(s.symmetry_multiplicity or 0) for s in sites1) == len(raw)
    assert len(sites1) < len(raw), "symmetry should collapse redundant envelope sites"
    assert any(int(s.symmetry_multiplicity or 0) > 1 for s in sites1)
    for s1, s2 in zip(sites1, sites2, strict=True):
        np.testing.assert_allclose(np.asarray(s1.xy), np.asarray(s2.xy), atol=1e-8)
        assert int(s1.symmetry_multiplicity or 0) == int(s2.symmetry_multiplicity or 0)


def test_symmetry_info_includes_hall_and_mode():
    """Extended info dict includes Hall symbol and mode when spglib succeeds."""
    atoms = bulk("Cu", "fcc", a=3.6)
    info = SymmetryAnalyzer(atoms, 0.01).get_symmetry_info()
    assert "hall_symbol" in info
    assert info["hall_symbol"] is not None
    assert info["symmetry_mode"] == "periodic"


def test_spglib_returns_none_raises(monkeypatch):
    """spglib returning None raises SymmetryAnalysisError (no identity fallback)."""
    atoms = bulk("Cu", "fcc", a=3.6)

    def _none(*_a, **_kw):
        return None

    monkeypatch.setattr(spglib, "get_symmetry_dataset", _none)
    an = SymmetryAnalyzer(atoms, symmetry_tolerance=0.01)
    with pytest.raises(SymmetryAnalysisError, match="returned None"):
        an.detect_symmetry_operations()


def test_cluster_symop_roundtrip_ni6_and_ch4():
    """Cluster cell tuples: every symop permutes atoms (MIC) like bulk."""
    pos = []
    for s in (1, -1):
        pos.extend([[s, 0, 0], [0, s, 0], [0, 0, s]])
    ni6 = Atoms("Ni6", positions=pos)
    ni6.set_cell([20, 20, 20])
    ni6.set_pbc([False, False, False])
    an_ni = SymmetryAnalyzer(ni6, symmetry_tolerance=0.2, mode="cluster")
    c_ni, f_ni, z_ni = an_ni.get_spglib_cell_tuple()
    _symop_roundtrip_preserves_structure(c_ni, f_ni, z_ni, symprec=0.2)

    a = 1.09
    methane = Atoms(
        "CH4",
        positions=[
            [0, 0, 0],
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a],
        ],
    )
    methane.set_cell([20, 20, 20])
    methane.set_pbc([False, False, False])
    an_ch4 = SymmetryAnalyzer(methane, symmetry_tolerance=0.15, mode="cluster")
    c_m, f_m, z_m = an_ch4.get_spglib_cell_tuple()
    _symop_roundtrip_preserves_structure(c_m, f_m, z_m, symprec=0.15)


def test_cluster_op_count_matches_spglib_reference():
    """Operation count matches ``spglib`` on the same tuple (Ni6, CH4, square Cu4)."""
    pos = []
    for s in (1, -1):
        pos.extend([[s, 0, 0], [0, s, 0], [0, 0, s]])
    ni6 = Atoms("Ni6", positions=pos)
    ni6.set_cell([20, 20, 20])
    ni6.set_pbc([False, False, False])
    _assert_op_count_matches_spglib(
        SymmetryAnalyzer(ni6, symmetry_tolerance=0.2, mode="cluster"),
        symprec=0.2,
    )

    a = 1.09
    methane = Atoms(
        "CH4",
        positions=[
            [0, 0, 0],
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a],
        ],
    )
    methane.set_cell([20, 20, 20])
    methane.set_pbc([False, False, False])
    _assert_op_count_matches_spglib(
        SymmetryAnalyzer(methane, symmetry_tolerance=0.15, mode="cluster"),
        symprec=0.15,
    )

    d = 2.0
    square = Atoms(
        "Cu4",
        positions=[[0, 0, 0], [d, 0, 0], [d, d, 0], [0, d, 0]],
    )
    square.set_cell([22, 22, 22])
    square.set_pbc([False, False, False])
    _assert_op_count_matches_spglib(
        SymmetryAnalyzer(square, symmetry_tolerance=0.12, mode="cluster"),
        symprec=0.12,
    )


def test_fcc111_pt_slab_symmetry_reduces_sites_and_verifies_orbits():
    """High-symmetry periodic slab: fewer unique sites; orbit pairwise check."""
    slab = create_slab_from_atoms(
        fcc111("Pt", size=(2, 2, 3), vacuum=7.0, orthogonal=True)
    ).atoms
    raw = get_unified_sites(slab, material_type="slab")
    assert raw is not None and len(raw) >= 1
    sym = get_symmetry_aware_sites(slab, symmetry_tolerance=0.15)
    assert len(sym) >= 1
    assert len(sym) <= len(raw)
    assert sum((s.symmetry_multiplicity or 0) for s in sym) == len(raw)

    an = SymmetryAnalyzer(slab, symmetry_tolerance=0.15)
    _assert_orbit_pairwise_symops(an, raw, planar=True)


def test_make_slab_orbit_soundness():
    """Synthetic FCC-like slab: pairwise orbit property."""
    slab = make_slab(nx=2, ny=2)
    raw = get_unified_sites(slab, material_type="slab")
    assert raw is not None and len(raw) >= 1
    an = SymmetryAnalyzer(slab, symmetry_tolerance=0.1)
    _assert_orbit_pairwise_symops(an, raw, planar=True)


def test_angle_tolerance_passes_to_spglib():
    """Optional ``angle_tolerance`` is accepted and analysis succeeds."""
    atoms = bulk("Cu", "fcc", a=3.6)
    an = SymmetryAnalyzer(
        atoms,
        symmetry_tolerance=0.01,
        angle_tolerance=5.0,
    )
    assert an.get_symmetry_info()["spacegroup_number"] == 225


def test_get_symmetry_aware_sites_precomputed_raw_matches_internal_fetch():
    """Passing *raw_sites* should match calling :func:`get_unified_sites` internally."""
    slab = make_slab()
    raw = get_unified_sites(slab, material_type="slab")
    assert len(raw) >= 1
    with_pre = get_symmetry_aware_sites(
        slab, material_type="slab", raw_sites=raw, symmetry_tolerance=0.1
    )
    without = get_symmetry_aware_sites(
        slab, material_type="slab", symmetry_tolerance=0.1
    )
    assert with_pre is not None and without is not None
    assert len(with_pre) == len(without)


def test_analyze_site_symmetry_keeps_distinct_site_types_separate():
    """Identical xyz with different site_type must not share an orbit."""
    from metalsurfer.placement.site_types import site_from_dict

    slab = make_slab(nx=2, ny=2)
    an = SymmetryAnalyzer(slab, symmetry_tolerance=0.1)
    xyz = [1.0, 1.0, 5.0]
    sites = [
        site_from_dict(
            {
                "xy": xyz[:2],
                "z": xyz[2],
                "xyz": xyz,
                "site_type": "atop",
                "slab_indices": (0,),
                "normal": [0.0, 0.0, 1.0],
                "material_type": "slab",
                "site_source": "test",
                "env_fingerprint": (("Cu",), "atop"),
            }
        ),
        site_from_dict(
            {
                "xy": xyz[:2],
                "z": xyz[2],
                "xyz": xyz,
                "site_type": "hollow",
                "slab_indices": (0, 1, 2),
                "normal": [0.0, 0.0, 1.0],
                "material_type": "slab",
                "site_source": "test",
                "env_fingerprint": (("Cu", "Cu", "Cu"), "hollow"),
            }
        ),
    ]
    grouped = an.analyze_site_symmetry(sites, planar=True)
    assert len(grouped) == 2
    assert {g.site_type for g in grouped} == {"atop", "hollow"}
    assert all(int(g.symmetry_multiplicity or 0) == 1 for g in grouped)


def test_planar_separation_distance_uses_slab_normal_not_cartesian_xy():
    """Planar mode drops the a×b normal component, not Cartesian z."""
    # Orthorhombic cell rotated so the slab normal is not +z.
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    atoms = make_slab(nx=2, ny=2)
    cell = tilt @ np.asarray(atoms.get_cell(), dtype=float)
    atoms.set_cell(cell)
    pos = atoms.get_positions()
    pos[:] = (tilt @ pos.T).T
    atoms.set_positions(pos)

    an = SymmetryAnalyzer(atoms, symmetry_tolerance=0.1)
    n = an._slab_normal()
    # Pure normal separation: planar distance must be ~0; Cartesian [:2] is not.
    sep = 1.5 * n
    assert an._separation_distance(sep, planar=True) < 1e-9
    assert float(np.linalg.norm(sep[:2])) > 0.5
    # Pure in-plane separation along a: planar distance equals |sep|.
    sep_ab = np.asarray(an._lattice[0], dtype=float)
    sep_ab = sep_ab / np.linalg.norm(sep_ab)
    assert an._separation_distance(sep_ab, planar=True) == pytest.approx(1.0, abs=1e-9)


def test_tilted_slab_symmetry_multiplicities_partition_raw():
    """Tilted slab (normal ≠ +z) still partitions raw sites into orbits."""
    slab = make_slab(nx=2, ny=2)
    cell = np.array(slab.get_cell(), dtype=float)
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell[:3] = tilt @ cell[:3]
    slab.set_cell(cell)
    pos = slab.get_positions()
    pos[:] = (tilt @ pos.T).T
    slab.set_positions(pos)

    raw = get_unified_sites(slab, material_type="slab")
    assert raw is not None and len(raw) >= 1
    sym = get_symmetry_aware_sites(slab, symmetry_tolerance=0.15)
    assert len(sym) >= 1
    assert len(sym) <= len(raw)
    assert sum(int(s.symmetry_multiplicity or 0) for s in sym) == len(raw)
    # analyze_site_symmetry re-verifies pairwise symop connectivity (raises on failure).
    an = SymmetryAnalyzer(slab, symmetry_tolerance=0.15)
    regrouped = an.analyze_site_symmetry(raw, planar=True)
    assert sum(int(g.symmetry_multiplicity or 0) for g in regrouped) == len(raw)

"""Regression tests for the Tier 2 correctness fixes.

Each test pins a defect that was silent: it produced no exception and no failing
test, but changed scientific output (site geometry, site labels, orbit counts,
saturation bookkeeping, substrate constraints). Keep these specific to the
failure mode rather than to the surrounding implementation.
"""

import dataclasses
import logging
import time
from collections import Counter

import numpy as np
import pytest
from ase.build import fcc100, fcc111, hcp0001
from ase.cluster import Octahedron

from metalsurfer.placement.site_coords import (
    _slab_normal,
    top_layer_mask_by_normal,
)
from metalsurfer.placement.site_enumeration import (
    get_symmetry_aware_sites,
    get_unified_sites,
)
from metalsurfer.symmetry import SymmetryAnalyzer

# ---------------------------------------------------------------------------
# #1 slab site normals are the exact a×b surface normal
# ---------------------------------------------------------------------------


def _rotated(atoms, angle_deg: float):
    rotated = atoms.copy()
    rotated.rotate(angle_deg, "x", rotate_cell=True)
    return rotated


@pytest.mark.parametrize(
    "slab_factory",
    [
        lambda: fcc111("Pt", (3, 3, 4), vacuum=10.0),
        lambda: fcc100("Cu", (3, 3, 4), vacuum=10.0),
        lambda: hcp0001("Ru", (3, 3, 4), vacuum=10.0),
    ],
    ids=["fcc111", "fcc100", "hcp0001"],
)
def test_slab_site_normals_are_exactly_the_slab_normal(slab_factory):
    """A k-nearest centroid tilted slab normals by up to 56°; use a×b instead."""
    slab = slab_factory()
    cell = np.asarray(slab.get_cell(), dtype=float)
    expected = _slab_normal(cell)

    sites = get_unified_sites(slab, material_type="slab")
    assert sites

    normals = np.array([s.normal for s in sites], dtype=float)
    assert np.allclose(normals, expected, atol=1e-12)
    # Unrotated ASE slabs have the surface normal along +z.
    assert np.allclose(normals[:, 2], 1.0, atol=1e-12)


def test_rotated_slab_site_normals_follow_the_rotated_cell():
    """Normals track the cell, not Cartesian z, when the slab is rotated."""
    slab = _rotated(fcc111("Pt", (3, 3, 4), vacuum=10.0), 30.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    expected = _slab_normal(cell)
    assert abs(float(expected[2])) < 0.99  # genuinely tilted

    sites = get_unified_sites(slab, material_type="slab")
    assert sites
    normals = np.array([s.normal for s in sites], dtype=float)
    assert np.allclose(normals, expected, atol=1e-12)


@pytest.mark.parametrize("size", [3, 4, 5])
def test_hollow_count_is_twice_the_top_layer_on_fcc111(size):
    """Accessibility gating is periodic: no hollow is dropped at an a/b boundary."""
    slab = fcc111("Pt", (size, size, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))
    assert n_top == size * size

    sites = get_unified_sites(slab, material_type="slab")
    n_hollow = sum(1 for s in sites if s.site_source == "topology_hollow")
    assert n_hollow == 2 * n_top


# ---------------------------------------------------------------------------
# #2 cross-boundary hollows keep their hollow label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slab_factory",
    [
        lambda: fcc111("Pt", (3, 3, 4), vacuum=10.0),
        lambda: fcc111("Pt", (4, 4, 4), vacuum=10.0),
        lambda: fcc111("Pt", (5, 5, 4), vacuum=10.0),
        lambda: hcp0001("Ru", (3, 3, 4), vacuum=10.0),
    ],
    ids=["fcc111-3", "fcc111-4", "fcc111-5", "hcp0001-3"],
)
def test_site_type_matches_topology_source_on_close_packed_slabs(slab_factory):
    """Every generated site keeps the label its generator intended."""
    slab = slab_factory()
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))

    sites = get_unified_sites(slab, material_type="slab")
    assert sites
    mismatches = [
        (s.site_type, s.site_source)
        for s in sites
        if s.site_type != s.site_source.removeprefix("topology_")
    ]
    assert mismatches == []
    assert Counter(s.site_type for s in sites) == Counter(
        {"atop": n_top, "bridge": 3 * n_top, "hollow": 2 * n_top}
    )


def test_expanded_classification_index_carries_hollow_candidates():
    """The ±1 a/b index must emit hollows, not only cross-boundary bridges."""
    from scipy.spatial import Delaunay

    from metalsurfer.placement.site_coords import _project_to_slab_plane
    from metalsurfer.placement.site_voronoi import (
        _build_delaunay_classification_index,
    )

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    top_idx = np.nonzero(top_layer_mask_by_normal(positions, cell, 0.5))[0]
    top_2d = _project_to_slab_plane(positions[top_idx], cell)
    tri = Delaunay(top_2d)

    _, types_primary, _ = _build_delaunay_classification_index(top_2d, top_idx, tri)
    _, types_pbc, _ = _build_delaunay_classification_index(
        top_2d,
        top_idx,
        tri,
        cell=cell,
        pbc=np.array([True, True, False]),
    )
    counts_primary = Counter(types_primary)
    counts_pbc = Counter(types_pbc)
    assert counts_pbc["hollow"] > counts_primary["hollow"]
    assert counts_pbc["bridge"] > counts_primary["bridge"]


# ---------------------------------------------------------------------------
# #3 the Voronoi pass on planar slabs is skipped explicitly, not swallowed
# ---------------------------------------------------------------------------


def test_planar_slab_skips_voronoi_and_says_so(caplog, monkeypatch):
    """Qhull used to raise QH6154 on the coplanar top layer and be swallowed."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    def _fail(*_args, **_kwargs):
        raise AssertionError("_voronoi_sites must not run on a planar slab")

    monkeypatch.setattr(site_enumeration, "_voronoi_sites", _fail)

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.INFO, logger="metalsurfer.placement.site_enumeration"):
        sites = site_enumeration.get_unified_sites(slab, material_type="slab")

    assert sites
    assert any(
        "skipping Voronoi vertex generation" in record.getMessage()
        for record in caplog.records
    )


def test_non_planar_slab_still_runs_voronoi():
    """Only coplanar top layers skip the Voronoi pass."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = slab.get_positions()
    positions[-1, 2] += 1.5  # adatom-like bump breaks coplanarity
    slab.set_positions(positions)

    cell = np.asarray(slab.get_cell(), dtype=float)
    assert not site_enumeration._top_layer_is_planar_from_arrays(
        np.asarray(slab.get_positions(), dtype=float), cell, 0.5
    )


def test_slab_only_enrichment_flag_warns(caplog):
    """voronoi_site_enrichment is porous/nanoparticle-only; slabs get a warning."""
    from metalsurfer.config import AdsorptionConfig
    from metalsurfer.placement.site_context import (
        _get_unique_sites_for_specs,
        clear_site_caches,
    )

    clear_site_caches()
    config = AdsorptionConfig(material_type="slab", voronoi_site_enrichment=False)
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.WARNING, logger="metalsurfer.placement.site_context"):
        _get_unique_sites_for_specs(slab, config)
    clear_site_caches()

    assert any(
        "voronoi_site_enrichment=False has no effect" in record.getMessage()
        for record in caplog.records
    )


# ---------------------------------------------------------------------------
# #4 cluster symmetry is translation invariant and does not fold antipodes
# ---------------------------------------------------------------------------

_TRANSLATIONS = [
    np.zeros(3),
    np.array([3.7, 0.0, 0.0]),
    np.array([1.234, 2.345, 3.456]),
    np.array([-10.0, 0.0, 0.0]),
    np.array([100.0, -50.0, 7.0]),
]


def test_cluster_site_orbits_are_translation_invariant():
    """Orbits used to be 5/7/9/7 depending on where the cluster sat in space."""
    cluster = Octahedron("Au", 3, cutoff=1)
    sites = get_unified_sites(cluster, material_type="nanoparticle")
    assert sites

    signatures = []
    for offset in _TRANSLATIONS:
        moved = cluster.copy()
        moved.set_positions(moved.get_positions() + offset)
        moved_sites = [
            dataclasses.replace(s, xyz=np.asarray(s.xyz, dtype=float) + offset)
            for s in sites
        ]
        analyzer = SymmetryAnalyzer(moved, mode="cluster")
        orbits = analyzer.analyze_site_symmetry(moved_sites)
        signatures.append(tuple(sorted(o.symmetry_multiplicity for o in orbits)))

    assert len(set(signatures)) == 1, signatures
    assert len(signatures[0]) == 5
    assert signatures[0] == (1, 1, 2, 2, 3)


def test_cluster_symmetry_does_not_over_merge_antipodal_sites():
    """Unconditional MIC folding in a padded box merged opposite faces."""
    cluster = Octahedron("Au", 5, cutoff=2)
    assert len(cluster) == 55
    sites = get_unified_sites(cluster, material_type="nanoparticle")
    analyzer = SymmetryAnalyzer(cluster, mode="cluster")
    orbits = analyzer.analyze_site_symmetry(sites)
    assert len(orbits) > 15


def test_periodic_slab_reduction_is_unaffected_by_the_cluster_fix():
    """Periodic mode still wraps and folds on all three spglib lattice axes."""
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    raw = get_unified_sites(slab, material_type="slab")
    reduced = get_symmetry_aware_sites(slab, material_type="slab", raw_sites=raw)
    assert sorted(s.symmetry_multiplicity for s in reduced) == [9, 18, 27]


# ---------------------------------------------------------------------------
# #5 the vectorized orbit search is orbit-identical and fast
# ---------------------------------------------------------------------------


def _reference_orbits(analyzer, sites, planar):
    """Original O(n²·m) scalar triple loop, kept here as the oracle."""
    sorted_sites = sorted(sites, key=analyzer._site_sort_key)
    frac_ops = analyzer._frac_ops_from_dataset()
    n = len(sorted_sites)
    cart = [analyzer._site_3d_cart(s) for s in sorted_sites]
    types = [str(s.site_type) for s in sorted_sites]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    frac = [analyzer._cart_to_frac(p) for p in cart]
    for i in range(n):
        for R, t in frac_ops:
            moved = analyzer._wrap_frac(analyzer._apply_frac_symop(frac[i], R, t))
            for j in range(n):
                if i == j or types[i] != types[j]:
                    continue
                d_frac = analyzer._mic_frac_delta(moved, frac[j])
                sep = analyzer._cart_sep_from_frac_delta(d_frac)
                if analyzer._separation_distance(sep, planar) < (
                    analyzer.symmetry_tolerance
                ):
                    union(i, j)

    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(find(i), []).append(i)
    return sorted(tuple(sorted(v)) for v in roots.values())


def test_vectorized_orbits_match_the_scalar_reference_on_a_slab():
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    sites = get_unified_sites(slab, material_type="slab")
    analyzer = SymmetryAnalyzer(slab, mode="auto")

    reference = _reference_orbits(analyzer, sites, True)
    orbits = analyzer.analyze_site_symmetry(sites, planar=True)

    assert sorted(len(o) for o in reference) == sorted(
        o.symmetry_multiplicity for o in orbits
    )
    sorted_sites = sorted(sites, key=analyzer._site_sort_key)
    ref_sets = {
        frozenset(tuple(np.round(sorted_sites[i].xy, 6)) for i in o) for o in reference
    }
    new_sets = {
        frozenset(tuple(np.round(xy, 6)) for xy in o.symmetry_equivalent_sites)
        for o in orbits
    }
    assert ref_sets == new_sets


def test_vectorized_orbits_match_the_scalar_reference_on_a_cluster():
    cluster = Octahedron("Au", 3, cutoff=1)
    sites = get_unified_sites(cluster, material_type="nanoparticle")
    analyzer = SymmetryAnalyzer(cluster, mode="cluster")

    reference = _reference_orbits(analyzer, sites, False)
    orbits = analyzer.analyze_site_symmetry(sites, planar=False)
    assert sorted(len(o) for o in reference) == sorted(
        o.symmetry_multiplicity for o in orbits
    )


def test_symmetry_reduction_of_a_4x4_slab_is_fast():
    """3.97 s (3×3) / 20.9 s (4×4) before vectorization."""
    slab = fcc111("Pt", (4, 4, 4), vacuum=10.0)
    raw = get_unified_sites(slab, material_type="slab")
    analyzer = SymmetryAnalyzer(slab, mode="auto")
    analyzer._frac_ops_from_dataset()  # exclude one-off spglib cost

    start = time.perf_counter()
    orbits = analyzer.analyze_site_symmetry(raw, planar=True)
    elapsed = time.perf_counter() - start

    assert sorted(o.symmetry_multiplicity for o in orbits) == [16, 32, 48]
    assert elapsed < 1.0, f"analyze_site_symmetry took {elapsed:.2f}s"

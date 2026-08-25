"""Delaunay/local-normal site classification and symmetry reduction."""

import dataclasses
import time
from collections import Counter

import numpy as np
import pytest
from ase.build import fcc111, hcp0001
from ase.cluster import Octahedron
from scipy.spatial import Delaunay, KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement import (
    get_symmetry_aware_sites,
    get_unified_sites,
)
from metalsurfer.placement._constants import _NORMAL_K_NEIGHBOURS
from metalsurfer.placement.site_classify import _compute_local_normals_batch
from metalsurfer.placement.site_context import _get_unique_sites_for_specs
from metalsurfer.placement.site_coords import top_layer_mask_by_normal
from metalsurfer.symmetry import SymmetryAnalyzer

from ..conftest import (
    make_nanoparticle,
    make_slab,
)
from ._helpers import (
    _GOLDEN_SLAB_SITE_TYPE_MULTISET,
    _site_ordering_key,
)

_TRANSLATIONS = [
    np.zeros(3),
    np.array([3.7, 0.0, 0.0]),
    np.array([1.234, 2.345, 3.456]),
    np.array([-10.0, 0.0, 0.0]),
    np.array([100.0, -50.0, 7.0]),
]


def _reference_orbits(analyzer, sites, planar):
    """Brute-force orbit oracle; MIC distances via ASE find_mic (not analyzer MIC helpers)."""
    from ase.geometry import find_mic

    from metalsurfer._geom_pbc import frac_to_cart

    sorted_sites = sorted(sites, key=analyzer._site_sort_key)
    frac_ops = analyzer._frac_ops_from_dataset()
    n = len(sorted_sites)
    cart = np.asarray([analyzer._site_3d_cart(s) for s in sorted_sites], dtype=float)
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

    cell = np.asarray(analyzer._lattice, dtype=float)
    pbc = list(analyzer._symmetry_pbc())
    frac = [analyzer._cart_to_frac(p) for p in cart]
    tol = float(analyzer.symmetry_tolerance)
    n_hat = analyzer._slab_normal() if planar else None
    cluster_com = getattr(analyzer, "_cluster_com", None)
    cluster_half = getattr(analyzer, "_cluster_half", None)
    for i in range(n):
        for R, t in frac_ops:
            moved_frac = analyzer._wrap_frac(analyzer._apply_frac_symop(frac[i], R, t))
            moved_cart = frac_to_cart(moved_frac, cell)
            # Undo cluster COM/half-box shift applied by _cart_to_frac.
            if (
                getattr(analyzer, "_mode", None) == "cluster"
                and cluster_com is not None
                and cluster_half is not None
            ):
                moved_cart = moved_cart - cluster_half + cluster_com
            for j in range(n):
                if i == j or types[i] != types[j]:
                    continue
                delta = cart[j] - moved_cart
                mic_vec, _ = find_mic(delta.reshape(1, 3), cell, pbc=pbc)
                sep = np.asarray(mic_vec[0], dtype=float)
                if n_hat is not None:
                    sep = sep - float(np.dot(sep, n_hat)) * n_hat
                if float(np.linalg.norm(sep)) < tol:
                    union(i, j)

    roots: dict[int, list[int]] = {}
    for i in range(n):
        roots.setdefault(find(i), []).append(i)
    return sorted(tuple(sorted(v)) for v in roots.values())


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
        if s.site_source.startswith("topology_")
        and s.site_type != s.site_source.removeprefix("topology_")
    ]
    assert mismatches == []
    assert Counter(s.site_type for s in sites) == Counter(
        {"atop": n_top, "bridge": 3 * n_top, "hollow": 2 * n_top}
    )


def test_expanded_classification_index_carries_hollow_candidates():
    """The ±1 a/b index must emit hollows, not only cross-boundary bridges."""

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
    """Vectorization regression guard: pre-vectorization ~21 s on 4×4 Pt(111)."""
    slab = fcc111("Pt", (4, 4, 4), vacuum=10.0)
    raw = get_unified_sites(slab, material_type="slab")
    analyzer = SymmetryAnalyzer(slab, mode="auto")
    analyzer._frac_ops_from_dataset()  # exclude one-off spglib cost

    start = time.perf_counter()
    orbits = analyzer.analyze_site_symmetry(raw, planar=True)
    elapsed = time.perf_counter() - start

    assert sorted(o.symmetry_multiplicity for o in orbits) == [16, 32, 48]
    # Baseline ≈0.2 s locally; 5 s leaves ~25x headroom for loaded CI while
    # still failing on the ~21 s pre-vectorization behaviour this guards against.
    assert elapsed < 5.0, f"analyze_site_symmetry took {elapsed:.2f}s"


def test_build_site_records_classifies_boundary_bridge_via_expanded_index():
    """Production classify path labels a cross-boundary bridge without an upgrade pass."""

    from metalsurfer.placement.site_classify import (
        _build_site_records,
        _DelaunayClassifyInputs,
    )
    from metalsurfer.placement.site_coords import _project_to_slab_plane
    from metalsurfer.placement.site_voronoi import _build_delaunay_classification_index

    positions = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc_on = np.array([True, True, False], dtype=bool)
    pbc_off = np.array([False, False, False], dtype=bool)
    top_idx = np.arange(4, dtype=int)
    top_2d = _project_to_slab_plane(positions, cell)
    tri = Delaunay(top_2d)
    primary = _build_delaunay_classification_index(top_2d, top_idx, tri)
    expanded = _build_delaunay_classification_index(
        top_2d, top_idx, tri, cell=cell, pbc=pbc_on
    )
    _, expanded_types, expanded_indices = expanded
    assert "bridge" in expanded_types
    assert frozenset((0, 1)) in {
        frozenset(idx)
        for typ, idx in zip(expanded_types, expanded_indices, strict=True)
        if typ == "bridge"
    }
    vertex = np.array([[0.0, 1.0, 0.5]], dtype=float)
    nn_dists = np.array([1.0], dtype=float)
    local_tree = KDTree(positions)
    symbols = ["Cu", "Cu", "Cu", "Cu"]

    with_pbc = _build_site_records(
        vertex,
        nn_dists,
        positions,
        symbols,
        local_tree,
        "slab",
        pore_threshold=2.5,
        cell=cell,
        pbc=pbc_on,
        delaunay=_DelaunayClassifyInputs(top_2d, top_idx, expanded),
    )
    primary_only = _build_site_records(
        vertex,
        nn_dists,
        positions,
        symbols,
        local_tree,
        "slab",
        pore_threshold=2.5,
        cell=cell,
        pbc=pbc_off,
        delaunay=_DelaunayClassifyInputs(top_2d, top_idx, primary),
    )
    assert with_pbc[0].site_type == "bridge"
    assert frozenset(with_pbc[0].slab_indices) == frozenset((0, 1))
    assert primary_only[0].site_type == "atop"


def test_get_unified_sites_labels_pbc_edge_bridge_on_production_path():
    """Hot path on a real slab: boundary sites primary-atop are labelled bridge."""

    from metalsurfer.placement.site_classify import (
        _build_site_records,
        _DelaunayClassifyInputs,
    )
    from metalsurfer.placement.site_coords import (
        _cart_to_frac,
        _project_to_slab_plane,
        top_layer_mask_by_normal,
    )
    from metalsurfer.placement.site_voronoi import _build_delaunay_classification_index

    slab = make_slab()
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(slab.get_pbc(), dtype=bool)
    top_idx = np.nonzero(top_layer_mask_by_normal(positions, cell, 0.5))[0]
    top_2d = _project_to_slab_plane(positions[top_idx], cell)
    tri = Delaunay(top_2d)
    primary = _build_delaunay_classification_index(top_2d, top_idx, tri)
    delaunay_primary = _DelaunayClassifyInputs(top_2d, top_idx, primary)
    local_tree = KDTree(positions)
    symbols = list(slab.get_chemical_symbols())

    sites = get_unified_sites(
        slab,
        material_type="slab",
        site_classification_method="delaunay",
    )
    assert sites

    boundary_sites = []
    for site in sites:
        frac = _cart_to_frac(np.asarray(site.xyz, dtype=float).reshape(1, 3), cell)[0]
        if (
            float(frac[0]) < 0.2
            or float(frac[0]) > 0.8
            or float(frac[1]) < 0.2
            or float(frac[1]) > 0.8
        ):
            boundary_sites.append(site)

    primary_records = _build_site_records(
        np.asarray([s.xyz for s in boundary_sites], dtype=float),
        np.asarray([s.nn_distance for s in boundary_sites], dtype=float),
        positions,
        symbols,
        local_tree,
        "slab",
        pore_threshold=2.5,
        cell=cell,
        pbc=pbc,
        delaunay=delaunay_primary,
    )
    upgraded = [
        site
        for site, rec in zip(boundary_sites, primary_records, strict=True)
        if rec.site_type == "atop" and site.site_type == "bridge"
    ]
    assert upgraded, (
        "expected get_unified_sites to label near-boundary primary-atop as bridge"
    )
    assert (
        sum(1 for s in sites if s.site_type == "hollow")
        == _GOLDEN_SLAB_SITE_TYPE_MULTISET["hollow"]
    )


def test_compute_local_normals_batch_points_outward_from_surface_centroid():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    vertex = np.array([1.0, 1.0, 2.0])
    k = min(_NORMAL_K_NEIGHBOURS, len(positions))
    _, idx = KDTree(positions).query(vertex.reshape(1, 3), k=k)
    normals = _compute_local_normals_batch(
        vertex.reshape(1, 3), positions, np.asarray(idx)
    )
    normal = normals[0]
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12)
    # Vertex sits above the centre of a flat square: normal is exactly +z.
    assert float(normal[2]) > 1.0 - 1e-9


def test_symmetry_aware_sites_are_consistent_with_core_sites_on_slab():
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(
        material_type="slab", symmetry_tolerance=0.1, site_equivalence_tolerance=0.05
    )

    _site_ctx = _get_unique_sites_for_specs(slab, config)
    core_sites, use_sites = _site_ctx.sites, _site_ctx.use_sites
    assert use_sites and len(core_sites) > 0

    reduced = get_symmetry_aware_sites(
        slab,
        top_layer_tolerance=config.top_layer_tolerance,
        symmetry_tolerance=config.symmetry_tolerance,
        material_type="slab",
    )
    assert 0 < len(reduced) <= len(core_sites)


def test_delaunay_classification_on_slab():
    """Delaunay method should produce valid site types for a simple slab."""
    slab = make_slab()
    sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="delaunay"
    )
    assert len(sites) > 0
    valid_types = {"atop", "bridge", "hollow"}
    for s in sites:
        assert s.site_type in valid_types, f"Bad type: {s.site_type}"


def test_delaunay_fallback_for_nanoparticle():
    """Delaunay classification should fall back to distance_ratio for NPs."""
    np_sites_dr = get_unified_sites(
        make_nanoparticle(),
        material_type="nanoparticle",
        site_classification_method="distance_ratio",
    )
    np_sites_del = get_unified_sites(
        make_nanoparticle(),
        material_type="nanoparticle",
        site_classification_method="delaunay",
    )
    # Both should produce the same result (fallback to distance_ratio)
    assert len(np_sites_del) == len(np_sites_dr)


def test_site_classification_auto_matches_delaunay_on_slab():
    slab = make_slab()
    auto_sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="auto"
    )
    del_sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="delaunay"
    )
    assert [s.site_type for s in auto_sites] == [s.site_type for s in del_sites]


def test_get_unified_sites_ordering_is_deterministic():
    slab = make_slab()
    first = get_unified_sites(slab, material_type="slab")
    second = get_unified_sites(slab, material_type="slab")
    assert [_site_ordering_key(s) for s in first] == [
        _site_ordering_key(s) for s in second
    ]


def test_unique_sites_cache_hit_and_miss():
    slab_a = make_slab(nx=4)
    slab_b = make_slab(nx=5)
    config = AdsorptionConfig(material_type="slab")
    ctx_a1 = _get_unique_sites_for_specs(slab_a, config)
    ctx_a2 = _get_unique_sites_for_specs(slab_a, config)
    ctx_b = _get_unique_sites_for_specs(slab_b, config)
    assert ctx_a1 is ctx_a2
    assert ctx_a1 is not ctx_b


def test_delaunay_bridge_fallback_uses_top_layer_not_bulk():
    """1.3: a bridge→hollow reclassification must reference surface atoms.

    The fallback used the full (bulk-inclusive) ``local_tree``; here a vertex
    whose three nearest bulk atoms differ from its three nearest top-layer atoms
    must resolve to top-layer indices only.
    """
    from metalsurfer.placement.site_classify import (
        _ClassificationContext,
        _classify_delaunay_vertices_batch,
        _DelaunayClassifyInputs,
    )

    # Top-layer atoms (indices 0,1,2) sit far from the vertex; bulk atoms
    # (indices 3,4) sit much closer, so the full-tree nearest-3 includes bulk.
    positions = np.array(
        [
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.1, 0.0, -1.0],
            [0.0, 0.1, -1.0],
        ],
        dtype=float,
    )
    top_atom_indices = np.array([0, 1, 2], dtype=int)

    vertex_2d = np.array([[0.0, 0.15]], dtype=float)
    vertices = np.array([[0.0, 0.15, 0.1]], dtype=float)

    cand_xy = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=float)
    cand_types = ["bridge", "atop", "atop", "atop"]
    cand_indices = [(1,), (0,), (1,), (2,)]
    cand_tree = KDTree(cand_xy)

    delaunay = _DelaunayClassifyInputs(
        top_positions_2d=np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0]], dtype=float),
        top_atom_indices=top_atom_indices,
        class_index=(cand_xy, cand_types, cand_indices),
    )
    ctx = _ClassificationContext(
        vertex_2d=vertex_2d,
        normals=np.array([[0.0, 0.0, 1.0]], dtype=float),
        pbc=np.array([True, True, False]),
        class_dists=None,
        class_idx=None,
        delaunay=delaunay,
        char_len=1e-6,  # tiny so any off-bridge vertex exceeds bridge_cut
        cand_tree=cand_tree,
    )
    local_tree = KDTree(positions)

    result = _classify_delaunay_vertices_batch(ctx, vertices, positions, local_tree)
    site_type, site_indices = result[0]
    assert site_type == "hollow"
    # The reclassified hollow must reference only top-layer atoms.
    assert set(site_indices).issubset(set(top_atom_indices.tolist()))
    assert len(site_indices) == 3


def test_build_classification_context_builds_images_once_porous(monkeypatch):
    """4.1: porous 3D-periodic contexts build the periodic-image KDTree once.

    The normals path and the Voronoi classifier must share a single
    periodic-image build, and the merged nearest-k distances must match the
    previous (separate) classification build.
    """
    from metalsurfer.placement import site_classify as sc
    from metalsurfer.placement._constants import _SITE_CLASSIFICATION_NEIGHBOURS

    rng = np.random.default_rng(0)
    cell = np.diag([5.0, 5.0, 5.0])
    pbc = np.array([True, True, True])
    positions = rng.uniform(0, 5, size=(40, 3))
    local_tree = KDTree(positions)
    vertices = positions[:5] + 0.01

    calls: list[int] = []
    orig = sc._build_periodic_images

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(sc, "_build_periodic_images", counting)

    ctx = sc._build_classification_context(
        vertices,
        positions,
        local_tree,
        material_type="porous",
        cell=cell,
        pbc=pbc,
        delaunay=None,
    )

    # Single shared build (normals reuses it instead of rebuilding).
    assert len(calls) == 1
    assert ctx.class_dists is not None
    assert ctx.class_idx is not None
    assert np.all(ctx.class_idx >= 0) and np.all(ctx.class_idx < len(positions))
    assert np.isfinite(ctx.normals).all()

    # Reference: original separate classification build using k_class margin.
    k_class = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(positions))
    d_ref, _ = local_tree.query(vertices, k=k_class)
    d_ref = np.atleast_2d(np.asarray(d_ref, dtype=float))
    margin_ref = float(np.max(d_ref[:, -1]))
    images_ref = orig(positions, cell, pbc, margin=margin_ref)
    dists_ref, _ = KDTree(images_ref).query(vertices, k=min(k_class, len(images_ref)))
    class_dists_ref = np.atleast_2d(np.asarray(dists_ref, dtype=float))
    assert np.allclose(ctx.class_dists, class_dists_ref)

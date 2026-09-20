"""Unified site generation, clustering and enumeration."""

import math

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc100, fcc111, hcp0001
from scipy.spatial import KDTree

from metalsurfer.placement import (
    check_initial_placement_distance,
    get_unified_sites,
)
from metalsurfer.placement.geometry import (
    detect_vdw_overlaps,
)
from metalsurfer.placement.site_coords import (
    _slab_normal,
    top_layer_mask_by_normal,
)
from metalsurfer.placement.site_enumeration import (
    _cluster_equivalent_sites,
)

from ..conftest import (
    adsorption_config_factory,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    water_conformers,
)
from ._helpers import (
    _generate_placements,
    _make_site,
    _tilted_make_slab,
)


def test_get_unified_sites_rejects_empty_atoms():
    with pytest.raises(ValueError, match="at least one atom"):
        get_unified_sites(Atoms(), material_type="slab")


def test_get_unified_sites_slab_nanoparticle_porous_have_expected_metadata():
    slab_sites = get_unified_sites(make_slab(), material_type="slab")
    np_sites = get_unified_sites(make_nanoparticle(), material_type="nanoparticle")
    porous_sites = get_unified_sites(make_porous_framework(), material_type="porous")

    assert len(slab_sites) > 0
    assert len(np_sites) > 0
    assert len(porous_sites) > 0

    for sites, mat in (
        (slab_sites, "slab"),
        (np_sites, "nanoparticle"),
        (porous_sites, "porous"),
    ):
        for site in sites:
            assert site.material_type == mat
            # Updated to accept both old "voronoi" and new topology-based sources
            assert site.site_source in (
                "voronoi",
                "topology_atop",
                "topology_bridge",
                "topology_hollow",
                "atop_injected",
            )
            assert site.nn_distance is not None
            assert np.asarray(site.xyz).shape == (3,)
            assert np.linalg.norm(np.asarray(site.normal)) > 0.5


def test_site_enumeration_exports_wrap_cartesian_for_atop_injection():
    """Atop injection under PBC uses _wrap_cartesian from site_coords."""
    from metalsurfer.placement import site_enumeration as enum_mod
    from metalsurfer.placement.site_coords import _wrap_cartesian as wrap_ref

    assert enum_mod._wrap_cartesian is wrap_ref
    slab = make_slab()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(slab.get_pbc(), dtype=bool)
    pts = slab.get_positions()[:1] + np.array([[0.1, 0.1, 0.5]])
    wrapped = enum_mod._wrap_cartesian(pts, cell, pbc)
    assert wrapped.shape == pts.shape
    assert len(get_unified_sites(slab, material_type="slab")) > 0


def test_get_unified_sites_slab_atop_injection_wraps_under_pbc(monkeypatch):
    """Atop injection must call _wrap_cartesian and emit atop_injected sites."""
    from metalsurfer.placement import site_enumeration as enum_mod
    from metalsurfer.placement.site_plugins import topology_slab as topo_mod
    from metalsurfer.placement.site_voronoi import _generate_slab_topology_sites

    real_topo = _generate_slab_topology_sites
    real_wrap = enum_mod._wrap_cartesian
    wrap_calls: list[int] = []

    def _topo_without_atop(*args, **kwargs):
        result = real_topo(*args, **kwargs)
        # Topology returns (verts, dists, sources, atom_indices, tri, exp...).
        verts, dists, sources, atoms = result[0], result[1], result[2], result[3]
        rest = result[4:]
        keep = [i for i, src in enumerate(sources) if src != "topology_atop"]
        if not keep:
            empty = (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                [],
                [],
            )
            return (*empty, *rest) if rest else empty
        idx = np.asarray(keep, dtype=int)
        trimmed = (
            verts[idx],
            dists[idx],
            [sources[i] for i in keep],
            [atoms[i] for i in keep],
        )
        return (*trimmed, *rest) if rest else trimmed

    def _counting_wrap(points, cell, pbc):
        wrap_calls.append(len(np.asarray(points)))
        return real_wrap(points, cell, pbc)

    monkeypatch.setattr(topo_mod, "_generate_slab_topology_sites", _topo_without_atop)
    monkeypatch.setattr(enum_mod, "_wrap_cartesian", _counting_wrap)
    slab = make_slab()
    assert bool(np.any(slab.get_pbc()))
    sites = get_unified_sites(slab, material_type="slab")
    assert len(sites) > 0
    assert wrap_calls, "_wrap_cartesian must run on the atop-injection path"
    assert any(str(s.site_source) == "atop_injected" for s in sites)


def test_cluster_equivalent_sites_reduces_or_keeps_sites_per_material():
    slab = make_slab()
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    slab_raw = get_unified_sites(slab, material_type="slab")
    np_raw = get_unified_sites(nanoparticle, material_type="nanoparticle")
    porous_raw = get_unified_sites(porous, material_type="porous")

    slab_unique = _cluster_equivalent_sites(
        slab_raw, np.asarray(slab.get_cell()), tolerance=0.05
    )
    np_unique = _cluster_equivalent_sites(
        np_raw, np.asarray(nanoparticle.get_cell()), tolerance=0.05
    )
    porous_unique = _cluster_equivalent_sites(
        porous_raw, np.asarray(porous.get_cell()), tolerance=0.05
    )

    assert 0 < len(slab_unique) <= len(slab_raw)
    assert 0 < len(np_unique) <= len(np_raw)
    assert 0 < len(porous_unique) <= len(porous_raw)


@pytest.mark.parametrize(
    "sites,expected_count",
    [
        (
            [
                _make_site(
                    [1.0, 1.0, 5.0],
                    site_type="atop",
                    material_type="slab",
                ),
                _make_site(
                    [1.0, 1.0, 6.0],
                    site_type="atop",
                    material_type="slab",
                ),
            ],
            2,
        ),
        (
            [
                _make_site(
                    [1.0, 1.0, 5.0],
                    site_type="atop",
                    material_type="slab",
                    env_fingerprint=(("Ni",), (), 0),
                ),
                _make_site(
                    [1.0, 1.0, 5.0],
                    site_type="atop",
                    material_type="slab",
                    env_fingerprint=(("Pt",), (), 0),
                ),
            ],
            2,
        ),
        (
            [
                _make_site(
                    [1.0, 1.0, 5.0],
                    site_type="atop",
                    material_type="slab",
                    env_fingerprint=(("Ru",), (), 0),
                ),
                _make_site(
                    [1.001, 1.001, 5.0],
                    site_type="atop",
                    material_type="slab",
                    env_fingerprint=(("Ru",), (), 0),
                ),
            ],
            1,
        ),
    ],
)
def test_cluster_equivalent_sites_case_matrix(sites, expected_count):
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    unique = _cluster_equivalent_sites(sites, cell, tolerance=0.05)
    assert len(unique) == expected_count


def test_slab_enumeration_and_generation_have_high_success_and_site_coverage():
    slab = make_slab()
    config = adsorption_config_factory(
        material_type="slab",
        num_placements=50,
        # Scaled covalent z puts (1.5, 2.0) in the physical contact band (~3–4 Å).
        placement_z_range=(1.5, 2.0),
        reject_vdw_overlaps=True,
    )
    results = _generate_placements(
        water_conformers(), slab, config, smiles="O", n_desired=50
    )

    min_ok = max(35, int(math.ceil(0.8 * 50)))
    assert len(results) >= min_ok, (
        f"slab water generation yield too low: {len(results)}/50 (need >= {min_ok})"
    )
    visited_sites = {spec.site_index for spec, _, _ in results}
    assert len(visited_sites) >= 2
    for _spec, adsorbate, _descriptor in results:
        ok, dist, reason = check_initial_placement_distance(
            adsorbate,
            slab,
            reject_vdw_overlaps=True,
            material_type="slab",
        )
        assert ok, f"Successful placement must pass contact gates: {reason}"
        # Lower floor is gated by `assert ok`; only the slack upper tail is checked.
        assert dist <= _descriptor.z_offset + 0.2, (
            f"Adsorbate–surface distance should be physical, got {dist:.3f}"
        )
        overlaps, _ = detect_vdw_overlaps(adsorbate, slab, material_type="slab")
        assert len(overlaps) == 0, "Successful placement must not have VDW clashes"


def test_is_top_layer_planar_true_for_three_coplanar_atoms():
    from metalsurfer.placement.site_plugins.helpers import (
        is_top_layer_planar as _is_top_layer_planar,
    )

    atoms = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 5.0], [2.5, 0.0, 5.0], [1.25, 2.2, 5.0]],
        cell=[5.0, 5.0, 20.0],
        pbc=[True, True, False],
    )
    assert _is_top_layer_planar(atoms, top_layer_tolerance=0.5) is True


def test_get_unified_sites_uses_material_aware_pbc_not_atoms_ttt():
    """TTT atoms with material_type=slab must still enumerate as TTF slab sites."""
    slab = make_slab()
    ttf = get_unified_sites(slab, material_type="slab")
    ttt = slab.copy()
    ttt.set_pbc([True, True, True])
    sites_ttt = get_unified_sites(ttt, material_type="slab")
    assert len(sites_ttt) == len(ttf)
    assert {s.site_type for s in sites_ttt} == {s.site_type for s in ttf}


def test_get_symmetry_aware_sites_mode_follows_material_type_not_atoms_pbc(
    monkeypatch,
):
    """SymmetryAnalyzer mode comes from material_type, not atoms.get_pbc()."""
    from metalsurfer.placement import get_symmetry_aware_sites
    from metalsurfer.placement import site_enumeration as se
    from metalsurfer.symmetry import SymmetryAnalyzer

    captured: dict[str, str] = {}
    real_analyzer = SymmetryAnalyzer

    def _capturing(*args, **kwargs):
        analyzer = real_analyzer(*args, **kwargs)
        captured["mode"] = analyzer._mode
        return analyzer

    monkeypatch.setattr(se, "SymmetryAnalyzer", _capturing)

    slab = make_slab(nx=2, ny=2)
    raw = get_unified_sites(slab, material_type="slab")
    fff = slab.copy()
    fff.set_pbc([False, False, False])
    get_symmetry_aware_sites(
        fff, material_type="slab", raw_sites=raw, symmetry_tolerance=0.15
    )
    assert captured["mode"] == "periodic"

    np_atoms = make_nanoparticle()
    np_atoms.set_pbc([True, True, True])
    np_raw = get_unified_sites(np_atoms, material_type="nanoparticle")
    assert np_raw
    get_symmetry_aware_sites(np_atoms, material_type="nanoparticle", raw_sites=np_raw)
    assert captured["mode"] == "cluster"


def test_topology_bridges_keep_distinct_pbc_midpoints():
    """Same atom-pair interior vs boundary bridges must both survive generation."""
    from metalsurfer.placement.site_voronoi import _generate_slab_topology_sites

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
    pbc = np.array([True, True, False], dtype=bool)
    top_idx = np.arange(4, dtype=int)
    from metalsurfer.placement.site_plugins.helpers import (
        periodic_accessibility_tree as _periodic_accessibility_tree,
    )

    access_tree = _periodic_accessibility_tree(positions, cell, pbc, max_distance=5.0)
    verts, _dists, sources, _tri, *_rest = _generate_slab_topology_sites(
        positions,
        cell,
        pbc,
        top_idx,
        access_tree,
        site_height=0.5,
        probe_radius=1.0,
        max_distance=5.0,
    )
    bridge_xy = [
        (round(float(v[0]), 3), round(float(v[1]), 3))
        for v, src in zip(verts, sources, strict=True)
        if src == "topology_bridge"
    ]
    # Interior midpoints around (2,1)/(1,2) and near-boundary midpoints near x/y≈0.
    assert (2.0, 1.0) in bridge_xy or any(
        abs(x - 2.0) < 0.05 and abs(y - 1.0) < 0.05 for x, y in bridge_xy
    )
    assert any(abs(x) < 0.15 or abs(x - 4.0) < 0.15 for x, _y in bridge_xy) or any(
        abs(y) < 0.15 or abs(y - 4.0) < 0.15 for _x, y in bridge_xy
    )


def test_cluster_equivalent_sites_cartesian_tolerance_scales_with_cell():
    """0.05 Å tolerance merges sub-0.05 Cartesian duplicates regardless of cell size."""
    site_a = _make_site(
        [1.0, 1.0, 5.0],
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Ru",), (), 0),
    )
    site_b = _make_site(
        [1.04, 1.0, 5.0],
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Ru",), (), 0),
    )
    for a_len in (8.1, 16.2):
        cell = np.array([[a_len, 0.0, 0.0], [0.0, a_len, 0.0], [0.0, 0.0, 20.0]])
        unique = _cluster_equivalent_sites([site_a, site_b], cell, tolerance=0.05)
        assert len(unique) == 1


def test_cluster_equivalent_sites_tilted_slab_uses_in_plane_distance():
    """Clustering must use slab-plane distance, not Cartesian xy.

    Along the tilted b vector, Cartesian ``[:2]`` under-reports separation, so a
    tolerance between cart_xy and plane distance merges under the old metric and
    keeps sites distinct under the plane metric.
    """
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell = tilt @ np.diag([8.0, 8.0, 20.0])
    n = np.cross(cell[0], cell[1])
    n = n / np.linalg.norm(n)
    along_b = cell[1] / np.linalg.norm(cell[1])

    base = np.array([2.0, 2.0, 5.0], dtype=float)
    # sep=0.5 Å along b → cart_xy≈0.285, plane=0.5; tol=0.35 discriminates.
    other = base + 0.5 * along_b
    assert np.linalg.norm((other - base)[:2]) < 0.35
    assert np.linalg.norm((other - base) - np.dot(other - base, n) * n) > 0.35

    site_a = _make_site(
        base.copy(),
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    site_b = _make_site(
        other.copy(),
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    unique = _cluster_equivalent_sites(
        [site_a, site_b], cell, tolerance=0.35, z_abs_tolerance=0.2
    )
    assert len(unique) == 2

    # Same height, truly close in-plane → still merge.
    along_a = cell[0] / np.linalg.norm(cell[0])
    near = base + 0.05 * along_a
    site_near = _make_site(
        near.copy(),
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    unique_near = _cluster_equivalent_sites(
        [site_a, site_near], cell, tolerance=0.35, z_abs_tolerance=0.2
    )
    assert len(unique_near) == 1


def test_top_layer_mask_unchanged_for_bulk_slab():
    from metalsurfer.placement.site_coords import (
        _height_along_slab_normal,
        top_layer_mask_by_normal,
    )

    slab = make_slab()
    positions = slab.get_positions()
    cell = np.array(slab.get_cell())
    tol = 0.5
    heights = _height_along_slab_normal(positions, cell)
    legacy = heights >= (float(np.max(heights)) - tol)
    layered = top_layer_mask_by_normal(positions, cell, tol)
    assert np.array_equal(legacy, layered)


def test_top_layer_mask_derived_tol_excludes_subsurface_fcc():
    """Derived tol must not mask an entire multi-layer FCC-like slab."""
    from metalsurfer.placement.site_coords import (
        _derive_top_layer_tolerance,
        _height_along_slab_normal,
        top_layer_mask_by_normal,
    )

    positions = []
    for iz in range(4):
        for ix in range(4):
            for iy in range(4):
                positions.append([ix * 2.55, iy * 2.55, iz * 2.1])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[10.2, 0.0, 0.0], [0.0, 10.2, 0.0], [0.0, 0.0, 25.0]])
    symbols = ["Cu"] * len(positions)
    tol = _derive_top_layer_tolerance(symbols)
    assert tol <= 1.2
    mask = top_layer_mask_by_normal(positions, cell, tol)
    heights = _height_along_slab_normal(positions, cell)
    h_max = float(np.max(heights))
    assert mask.sum() == 16
    assert np.all(heights[mask] >= h_max - tol - 1e-9)
    assert not np.any(heights[mask] < h_max - 1.5)


def test_top_layer_mask_includes_step_terrace_for_reconstructed_surface():
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    positions = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * 2.7, iy * 2.7, 5.4])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 5.0])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 2.7])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[8.1, 0.0, 0.0], [0.0, 8.1, 0.0], [0.0, 0.0, 20.0]])
    mask = top_layer_mask_by_normal(positions, cell, 0.5)
    assert mask.sum() == 12  # 9 top + 3 step; exclude bulk at 2.7
    assert np.any(positions[mask, 2] < 5.2)
    assert not np.any(np.isclose(positions[mask, 2], 2.7))


def test_top_layer_mask_includes_step_just_outside_tol():
    """Terrace just below the primary band is included via gap rule."""
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    positions = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * 2.7, iy * 2.7, 5.4])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 4.8])  # Δh = 0.6 > tol=0.5
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 2.7])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[8.1, 0.0, 0.0], [0.0, 8.1, 0.0], [0.0, 0.0, 20.0]])
    mask = top_layer_mask_by_normal(positions, cell, 0.5)
    assert mask.sum() == 12
    assert np.any(np.isclose(positions[mask, 2], 4.8))
    assert not np.any(np.isclose(positions[mask, 2], 2.7))


def test_top_layer_mask_empty_positions():
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    mask = top_layer_mask_by_normal(
        np.empty((0, 3)),
        np.eye(3) * 10.0,
        0.5,
    )
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_hollow_order_metadata_on_slab():
    """Slab hollow sites should carry hollow_order metadata when classified as hollow."""
    sites = get_unified_sites(make_slab(), material_type="slab")
    hollow_sites = [s for s in sites if s.site_type == "hollow"]
    assert len(hollow_sites) > 0
    for site in hollow_sites:
        order = site.hollow_order
        assert order is not None, f"hollow site {site.xyz} missing hollow_order"
        assert order in (3, 4), f"unexpected hollow_order {order} at {site.xyz}"


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


def test_fcc111_site_type_ratios_and_coordination_numbers():
    """fcc111 site-type counts and CN (coordination = len(slab_indices)).

    The library labels sites only atop/bridge/hollow (no separate fcc/hcp tag),
    so we assert the verifiable decomposition: hollow count is 2× the top-layer
    count, bridge is 3× (triangular lattice), and atop is 1×. Each site's
    coordination number is the number of nearest substrate atoms in
    ``slab_indices`` (atop=1, bridge=2, hollow=3).
    """
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))
    sites = get_unified_sites(slab, material_type="slab")

    by_type = {
        t: [s for s in sites if s.site_type == t] for t in ("atop", "bridge", "hollow")
    }
    assert len(by_type["atop"]) == n_top
    assert len(by_type["hollow"]) == 2 * n_top
    assert len(by_type["bridge"]) == 3 * n_top

    for s in sites:
        if s.site_type == "atop":
            assert len(s.slab_indices) == 1
        elif s.site_type == "bridge":
            assert len(s.slab_indices) == 2
        elif s.site_type == "hollow":
            assert len(s.slab_indices) == 3


def test_fcc100_site_type_ratios_and_coordination_numbers():
    """fcc100 keeps atop/bridge/hollow with CN 1/2/3.

    The library reports fcc100 hollows as 3-fold (``hollow_order == 3``), not
    4-fold: each square is two Delaunay triangles, so hollow count is
    ``4 * n_top`` under topology-owned typing (two triangles × two
    periodic images of the square lattice). We assert the count ratios and
    the per-type coordination numbers exposed via ``slab_indices``.
    """
    slab = fcc100("Cu", (3, 3, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))
    sites = get_unified_sites(slab, material_type="slab")

    by_type = {
        t: [s for s in sites if s.site_type == t] for t in ("atop", "bridge", "hollow")
    }
    assert len(by_type["atop"]) == n_top
    assert len(by_type["bridge"]) == 3 * n_top
    assert len(by_type["hollow"]) == 4 * n_top

    for s in sites:
        if s.site_type == "atop":
            assert len(s.slab_indices) == 1
        elif s.site_type == "bridge":
            assert len(s.slab_indices) == 2
        elif s.site_type == "hollow":
            assert len(s.slab_indices) == 3


def test_atop_injection_runs_when_voronoi_empty_nanoparticle(monkeypatch):
    """NP path does not need Voronoi; topology alone yields atops."""
    import metalsurfer.placement.site_voronoi as voronoi_mod

    def _fail(*_a, **_k):
        raise AssertionError("Voronoi must not run for nanoparticles")

    monkeypatch.setattr(voronoi_mod, "_voronoi_sites", _fail)
    struct = make_nanoparticle()
    sites = get_unified_sites(struct, material_type="nanoparticle", enrich=False)
    assert len(sites) > 0
    assert any(s.site_type == "atop" for s in sites)
    assert any(s.site_source == "topology_atop" for s in sites)


def test_atop_injection_safety_net_when_np_topology_empty(monkeypatch):
    """When hull topology fails, metal–metal atop injection still runs."""
    from metalsurfer.placement.site_np import _NPTopologyResult
    from metalsurfer.placement.site_plugins import topology_np as np_mod

    monkeypatch.setattr(
        np_mod,
        "_generate_nanoparticle_topology_sites",
        lambda *a, **k: _NPTopologyResult(
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            [],
            [],
        ),
    )
    struct = make_nanoparticle()
    sites = get_unified_sites(struct, material_type="nanoparticle", enrich=False)
    assert len(sites) > 0
    assert any(s.site_source == "atop_injected" for s in sites)
    assert any(s.site_type == "atop" for s in sites)


def test_issue6_ni55_and_ni13_expose_atop_bridge_hollow():
    """Issue #6: NPs must expose atop/bridge/hollow outside the convex hull."""
    from collections import Counter

    from ase.cluster import Icosahedron, Octahedron
    from scipy.spatial import ConvexHull

    from metalsurfer.placement._constants import (
        _ATOP_INJECTION_HEIGHT_FACTOR,
        _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
    )
    from metalsurfer.placement.site_coords import _mean_covalent_radius
    from metalsurfer.placement.site_np import _outside_convex_hull_mask
    from metalsurfer.placement.site_plugins.helpers import (
        median_nn_or_fallback as _median_nn_or_fallback,
    )

    for name, atoms in (
        ("Ni55", Octahedron("Ni", 5, 2)),
        ("Ni13", Icosahedron("Ni", 2)),
    ):
        sites = get_unified_sites(atoms, material_type="nanoparticle")
        counts = Counter(s.site_type for s in sites)
        assert counts.get("atop", 0) > 0, name
        assert counts.get("bridge", 0) > 0, name
        assert counts.get("hollow", 0) > 0, name
        assert any(s.site_source == "topology_atop" for s in sites), name

        pos = np.asarray(atoms.get_positions(), dtype=float)
        hull = ConvexHull(pos)
        xyz = np.asarray([s.xyz for s in sites], dtype=float)
        assert np.all(_outside_convex_hull_mask(xyz, hull)), name

        metal_nn = _median_nn_or_fallback(
            np.empty(0, dtype=float),
            reference_positions=pos,
            cell=np.asarray(atoms.get_cell(), dtype=float),
            pbc=np.array([False, False, False], dtype=bool),
        )
        height = _ATOP_INJECTION_HEIGHT_FACTOR * metal_nn
        probe = _VORONOI_PROBE_RADIUS_COVALENT_SCALE * _mean_covalent_radius(
            list(atoms.get_chemical_symbols())
        )
        assert height >= probe, f"{name}: height={height:.3f} probe={probe:.3f}"

        atop_nn = [float(s.nn_distance) for s in sites if s.site_type == "atop"]
        assert atop_nn
        assert abs(float(np.median(atop_nn)) - height) < 0.35, name

        # Site.normal must match the support-atom lift direction (not a tilted
        # k-NN centroid), otherwise pose slides laterally off the topology site.
        for s in sites:
            idx = [int(i) for i in s.slab_indices if 0 <= int(i) < len(pos)]
            assert idx, name
            lift = np.asarray(s.xyz, dtype=float) - np.mean(pos[idx], axis=0)
            nrm = float(np.linalg.norm(lift))
            assert nrm > 1e-8, name
            lift_hat = lift / nrm
            n_hat = np.asarray(s.normal, dtype=float)
            n_hat = n_hat / float(np.linalg.norm(n_hat))
            assert float(np.dot(n_hat, lift_hat)) > 0.999, name


def test_issue6_nonempty_voronoi_must_not_zero_out_np_atops(monkeypatch):
    """NP enumeration must not call Voronoi (topology-only path)."""
    from ase.cluster import Octahedron

    import metalsurfer.placement.site_voronoi as voronoi_mod

    def _fail(*_a, **_k):
        raise AssertionError("Voronoi must not run for nanoparticles")

    monkeypatch.setattr(voronoi_mod, "_voronoi_sites", _fail)
    sites = get_unified_sites(Octahedron("Ni", 5, 2), material_type="nanoparticle")
    assert any(
        s.site_type == "atop" and s.site_source == "topology_atop" for s in sites
    )
    assert not any(s.site_source == "atop_injected" for s in sites)


def test_nanoparticle_asymmetric_cluster_still_has_typed_sites():
    """Lopsided convex NPs use hull-facet normals, not COM-radial only."""
    from collections import Counter

    from ase.cluster import Octahedron
    from scipy.spatial import ConvexHull

    from metalsurfer.placement.site_np import _outside_convex_hull_mask

    atoms = Octahedron("Ni", 5, 2)
    pos = atoms.get_positions()
    # Stretch one axis and shift a corner atom so the particle is asymmetric.
    pos[:, 0] *= 1.35
    pos[np.argmax(pos[:, 0]), 0] += 1.2
    atoms.set_positions(pos)

    sites = get_unified_sites(atoms, material_type="nanoparticle")
    counts = Counter(s.site_type for s in sites)
    assert counts.get("atop", 0) > 0
    assert counts.get("bridge", 0) > 0
    assert counts.get("hollow", 0) > 0
    hull = ConvexHull(atoms.get_positions())
    xyz = np.asarray([s.xyz for s in sites], dtype=float)
    assert np.all(_outside_convex_hull_mask(xyz, hull))
    # Atop normals should agree with the local hull outward direction.
    for s in sites:
        if s.site_type != "atop":
            continue
        assert float(np.linalg.norm(s.normal)) > 0.5
        assert float(np.dot(s.normal, s.xyz - atoms.get_positions().mean(0))) > 0.0


def test_issue6_ni111_slab_counts_unchanged():
    """Issue #6 control: Ni(111) 4×4 still yields textbook 16/48/32."""
    from collections import Counter

    slab = fcc111("Ni", size=(4, 4, 4), vacuum=12.0)
    sites = get_unified_sites(slab, material_type="slab")
    counts = Counter(s.site_type for s in sites)
    assert counts.get("atop", 0) == 16
    assert counts.get("bridge", 0) == 48
    assert counts.get("hollow", 0) == 32


def test_atop_injection_runs_when_voronoi_and_topology_empty_slab(monkeypatch):
    """1.1: planar slab with no Voronoi vertices and no topology still gets atop."""
    from metalsurfer.placement.site_plugins import topology_slab as topo_mod

    monkeypatch.setattr(
        topo_mod,
        "_voronoi_sites",
        lambda *a, **k: (
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
        ),
    )
    monkeypatch.setattr(
        topo_mod,
        "_generate_slab_topology_sites",
        lambda *a, **k: (
            np.empty((0, 3), dtype=float),
            np.empty((0,), dtype=float),
            [],
            [],
            None,
            None,
            None,
            None,
        ),
    )
    slab = make_slab()
    sites = get_unified_sites(slab, material_type="slab", enrich=False)
    assert len(sites) > 0
    assert any(s.site_source == "atop_injected" for s in sites)


def test_cluster_equivalent_sites_orders_by_slab_normal_height():
    """1.2: representative ordering follows slab-normal height, never Cartesian z.

    For a tilted slab the surface normal is not Cartesian-z; the ordered
    representatives must be keyed on height along the slab normal (as computed by
    the fixed ``_slab_coord``), reproduced here independently.
    """
    from metalsurfer._geom_pbc import height_along_slab_normal
    from metalsurfer.placement.site_coords import _slab_plane_projectors
    from metalsurfer.placement.site_enumeration import _cluster_equivalent_sites

    slab = _tilted_make_slab()
    sites = get_unified_sites(slab, material_type="slab")
    cell = np.asarray(slab.get_cell(), dtype=float)
    clustered = _cluster_equivalent_sites(sites, cell, tolerance=0.5)
    assert len(clustered) > 0

    pinv_ab_T, _ = _slab_plane_projectors(cell)

    def slab_key(s):
        xyz = np.asarray(s.xyz, dtype=float)
        frac = xyz @ pinv_ab_T
        frac = frac - np.floor(frac)
        h = float(height_along_slab_normal(xyz.reshape(1, 3), cell)[0])
        return (float(frac[0]), float(frac[1]), h, str(s.site_type))

    # The output must already be in slab-normal-height order.
    assert clustered == sorted(clustered, key=slab_key)


def test_median_nn_or_fallback_elongated_cell_avoids_self_image():
    """k=len(offsets)+1 must recover true NN; naive k=2 leaks self-images."""
    from ase.geometry import find_mic

    from metalsurfer.placement._constants import (
        _ATOP_INJECTION_HEIGHT_FACTOR,
        _SURFACE_COVALENT_RADIUS_FALLBACK,
        _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    )
    from metalsurfer.placement.site_plugins.helpers import (
        median_nn_or_fallback as _median_nn_or_fallback,
    )

    def reference_median_nn(points, cell, pbc):
        nn = []
        for i in range(len(points)):
            _, dists = find_mic(points - points[i], cell, pbc=pbc)
            dists = np.asarray(dists, dtype=float)
            dists[i] = np.inf
            nn.append(float(np.min(dists)))
        return float(np.median(nn))

    pbc = np.array([True, True, False], dtype=bool)
    elong_cell = np.diag([3.0, 20.0, 30.0])
    elong_points = np.array([[0.0, 0.0, 5.0], [0.0, 10.0, 5.0]], dtype=float)
    got = _median_nn_or_fallback(
        np.empty(0, dtype=float),
        reference_positions=elong_points,
        cell=elong_cell,
        pbc=pbc,
    )
    expected = reference_median_nn(elong_points, elong_cell, pbc)
    assert abs(got - expected) < 1e-9
    assert abs(got - 10.0) < 1e-9, f"self-image leaked into median NN: {got}"
    fallback = _VORONOI_MAX_DISTANCE_COVALENT_SCALE * _SURFACE_COVALENT_RADIUS_FALLBACK
    assert abs(got - fallback) > 1.0
    assert abs(_ATOP_INJECTION_HEIGHT_FACTOR * got - 8.0) < 1e-9


def test_cluster_equivalent_sites_anisotropic_slab_metric_bound():
    """Pairs that pass dxy/dz but exceed 1.5*tol Cartesian must still merge."""
    from metalsurfer.placement.site_enumeration import _cluster_equivalent_sites

    cell = np.diag([10.0, 10.0, 20.0])
    tol = 0.05
    z_tol = 0.5
    site_a = _make_site(
        [1.0, 1.0, 5.0],
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    site_b = _make_site(
        [1.04, 1.0, 5.10],
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    cart = float(np.linalg.norm(np.asarray(site_b.xyz) - np.asarray(site_a.xyz)))
    assert cart > 1.5 * tol
    assert cart < float(np.hypot(tol, z_tol))
    unique = _cluster_equivalent_sites(
        [site_a, site_b], cell, tolerance=tol, z_abs_tolerance=z_tol
    )
    assert len(unique) == 1

    site_far = _make_site(
        [1.04, 1.0, 5.60],
        site_type="atop",
        material_type="slab",
        env_fingerprint=(("Cu",), (), 0),
    )
    cart_far = float(np.linalg.norm(np.asarray(site_far.xyz) - np.asarray(site_a.xyz)))
    assert cart_far > float(np.hypot(tol, z_tol))
    unique_far = _cluster_equivalent_sites(
        [site_a, site_far], cell, tolerance=tol, z_abs_tolerance=z_tol
    )
    assert len(unique_far) == 2


def test_inject_atop_pbc_boundary_duplicate_merged_by_final_dedup():
    """PBC-aware merge alone collapses boundary-duplicate atop injections."""
    from metalsurfer.placement._constants import _ATOP_INJECTION_HEIGHT_FACTOR
    from metalsurfer.placement.site_enumeration import _inject_atop_sites
    from metalsurfer.placement.site_plugins.helpers import (
        periodic_accessibility_tree as _periodic_accessibility_tree,
    )

    positions = np.array(
        [
            [0.05, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.05, 3.0, 0.0],
            [2.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc = np.array([True, True, False], dtype=bool)
    median_nn = 2.0
    site_z = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
    existing = np.array([[3.98, 1.0, site_z]], dtype=float)
    existing_dists = np.array([site_z], dtype=float)
    existing_sources = ["voronoi"]
    access = _periodic_accessibility_tree(positions, cell, pbc, max_distance=5.0)
    verts, _dists, _sources, _atoms, _normals, _clearances = _inject_atop_sites(
        existing,
        existing_dists,
        existing_sources,
        positions=positions,
        cell=cell,
        pbc=pbc,
        material_type="slab",
        local_tree=KDTree(positions),
        accessibility_tree=access,
        median_nn=median_nn,
        slab_top_atom_indices=np.arange(4, dtype=int),
        has_topology_atop=False,
        probe_radius=0.5,
        max_site_distance=5.0,
    )
    near_atom0 = [
        v
        for v in verts
        if abs(float(v[1]) - 1.0) < 0.05
        and (abs(float(v[0]) % 4.0) < 0.15 or abs(float(v[0]) % 4.0 - 4.0) < 0.15)
    ]
    assert len(near_atom0) == 1


def test_merge_dedup_freezes_existing_unique_sites():
    """An injected midpoint near two old sites must not collapse them.

    Global union-find on the combined set would transitively merge A≈M≈B into
    one representative; freeze-existing merge keeps both A and B and drops M.
    """
    from metalsurfer.placement._constants import _VORONOI_DEDUP_TOLERANCE
    from metalsurfer.placement.site_plugins.helpers import (
        merge_dedup_site_arrays as _merge_dedup_site_arrays,
    )

    tol = _VORONOI_DEDUP_TOLERANCE
    # A and B farther than tol; M within tol of both.
    existing = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.5 * tol, 0.0, 1.0],
        ],
        dtype=float,
    )
    assert float(np.linalg.norm(existing[1] - existing[0])) > tol
    midpoint = 0.5 * (existing[0] + existing[1])
    assert float(np.linalg.norm(midpoint - existing[0])) < tol
    assert float(np.linalg.norm(midpoint - existing[1])) < tol

    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False], dtype=bool)
    verts, dists, sources, atoms, _normals, _clearances = _merge_dedup_site_arrays(
        existing,
        np.array([1.0, 1.0], dtype=float),
        ["voronoi", "voronoi"],
        midpoint.reshape(1, 3),
        np.array([1.0], dtype=float),
        ["atop_injected"],
        cell=cell,
        pbc=pbc,
        atom_indices=[(0,), (1,)],
        new_atom_indices=[(2,)],
    )
    assert len(verts) == 2
    np.testing.assert_allclose(verts, existing)
    assert sources == ["voronoi", "voronoi"]
    assert atoms == [(0,), (1,)]
    assert len(dists) == 2


def test_topology_boundary_candidate_retained_with_accessibility_tree():
    """PBC-aware accessibility_tree keeps a boundary candidate a plain tree drops."""
    from metalsurfer.placement.site_plugins.helpers import (
        periodic_accessibility_tree as _periodic_accessibility_tree,
    )
    from metalsurfer.placement.site_voronoi import _generate_slab_topology_sites

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc = np.array([True, True, False], dtype=bool)
    top_idx = np.arange(2, dtype=int)
    site_height = 0.5
    probe_radius = 0.3
    max_distance = 1.0

    plain = KDTree(positions)
    access = _periodic_accessibility_tree(positions, cell, pbc, max_distance)

    boundary_pt = np.array([[3.95, 0.0, site_height]], dtype=float)
    d_plain = float(np.asarray(plain.query(boundary_pt, k=1)[0]).ravel()[0])
    d_access = float(np.asarray(access.query(boundary_pt, k=1)[0]).ravel()[0])
    assert d_plain > max_distance
    assert probe_radius <= d_access <= max_distance

    verts_plain, _, sources_plain, *_ = _generate_slab_topology_sites(
        positions,
        cell,
        pbc,
        top_idx,
        plain,
        site_height=site_height,
        probe_radius=probe_radius,
        max_distance=max_distance,
    )
    verts_access, _, sources_access, *_ = _generate_slab_topology_sites(
        positions,
        cell,
        pbc,
        top_idx,
        access,
        site_height=site_height,
        probe_radius=probe_radius,
        max_distance=max_distance,
    )
    assert len(verts_access) >= len(verts_plain)
    assert sum(s == "topology_atop" for s in sources_access) >= sum(
        s == "topology_atop" for s in sources_plain
    )
    assert any(
        abs(float(v[0]) % 4.0) < 0.2 or abs(float(v[0]) % 4.0 - 4.0) < 0.2
        for v in verts_access
    )

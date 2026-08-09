"""Unified site generation, clustering and enumeration."""


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
from metalsurfer.placement.site_types import site_from_dict

from ..conftest import (
    adsorption_config_factory,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    water_conformers,
)
from ._helpers import (
    _generate_placements,
)


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

    real_topo = enum_mod._generate_slab_topology_sites
    real_wrap = enum_mod._wrap_cartesian
    wrap_calls: list[int] = []

    def _topo_without_atop(*args, **kwargs):
        result = real_topo(*args, **kwargs)
        # Topology may return (verts, dists, sources) or (+ triangulation).
        if len(result) == 4:
            verts, dists, sources, primary_tri = result
        else:
            verts, dists, sources = result
            primary_tri = None
        keep = [i for i, src in enumerate(sources) if src != "topology_atop"]
        if not keep:
            empty = (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                [],
            )
            return empty if primary_tri is None else (*empty, primary_tri)
        idx = np.asarray(keep, dtype=int)
        trimmed = (verts[idx], dists[idx], [sources[i] for i in keep])
        return trimmed if primary_tri is None else (*trimmed, primary_tri)

    def _counting_wrap(points, cell, pbc):
        wrap_calls.append(len(np.asarray(points)))
        return real_wrap(points, cell, pbc)

    monkeypatch.setattr(enum_mod, "_generate_slab_topology_sites", _topo_without_atop)
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
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 6.0,
                        "xyz": np.array([1.0, 1.0, 6.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                    }
                ),
            ],
            2,
        ),
        (
            [
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ni",), "atop"),
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Pt",), "atop"),
                    }
                ),
            ],
            2,
        ),
        (
            [
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ru",), "atop"),
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.001, 1.001]),
                        "z": 5.0,
                        "xyz": np.array([1.001, 1.001, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ru",), "atop"),
                    }
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

    assert len(results) >= 50
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
        assert 1.2 <= dist <= 4.0, (
            f"Adsorbate–surface distance should be physical (1.2–4.0 Å), got {dist:.3f}"
        )
        overlaps, _ = detect_vdw_overlaps(adsorbate, slab, material_type="slab")
        assert len(overlaps) == 0, "Successful placement must not have VDW clashes"

def test_is_top_layer_planar_true_for_three_coplanar_atoms():
    from metalsurfer.placement.site_enumeration import _is_top_layer_planar

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
    local_tree = KDTree(positions)
    verts, _dists, sources, _tri = _generate_slab_topology_sites(
        positions,
        cell,
        pbc,
        top_idx,
        local_tree,
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
    site_a = site_from_dict(
        {
            "xy": np.array([1.0, 1.0]),
            "z": 5.0,
            "xyz": np.array([1.0, 1.0, 5.0]),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Ru",), "atop"),
        }
    )
    site_b = site_from_dict(
        {
            "xy": np.array([1.04, 1.0]),
            "z": 5.0,
            "xyz": np.array([1.04, 1.0, 5.0]),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Ru",), "atop"),
        }
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

    site_a = site_from_dict(
        {
            "xy": base[:2],
            "z": float(base[2]),
            "xyz": base.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
    )
    site_b = site_from_dict(
        {
            "xy": other[:2],
            "z": float(other[2]),
            "xyz": other.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
    )
    unique = _cluster_equivalent_sites(
        [site_a, site_b], cell, tolerance=0.35, z_abs_tolerance=0.2
    )
    assert len(unique) == 2

    # Same height, truly close in-plane → still merge.
    along_a = cell[0] / np.linalg.norm(cell[0])
    near = base + 0.05 * along_a
    site_near = site_from_dict(
        {
            "xy": near[:2],
            "z": float(near[2]),
            "xyz": near.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
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
        assert order is None or order in (3, 4)


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


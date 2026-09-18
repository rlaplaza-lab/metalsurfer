"""Adaptive-grid site generator: spacing, accessibility, PBC, and A/B overlap."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from scipy.spatial import KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.generators import _topology_first_site_indices
from metalsurfer.placement.site_adaptive_grid import (
    _fractional_voxel_seeds,
    _nms,
    _shell_offsets,
    adaptive_grid_characteristic_length,
    adaptive_grid_spacing,
    adsorbate_contact_distance,
    generate_adaptive_grid_sites,
    min_adsorbate_grid_scale,
)
from metalsurfer.placement.site_coords import (
    _derive_voronoi_distance_window,
    _frac_to_cart,
)
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.placement.site_np import _outside_convex_hull_mask, _try_convex_hull
from metalsurfer.placement.site_types import Site

from ..conftest import make_nanoparticle, make_porous_framework, make_slab


def _window(
    atoms: Atoms, material_type: str
) -> tuple[float, float, np.ndarray, np.ndarray]:
    pos = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc(material_type), dtype=bool)
    probe, maxd = _derive_voronoi_distance_window(
        pos, list(atoms.get_chemical_symbols()), pbc, cell
    )
    return probe, maxd, cell, pbc


def test_adaptive_grid_not_in_adsorption_config():
    with pytest.raises(ValueError, match="site_generator"):
        AdsorptionConfig(site_generator="adaptive_grid")


def test_adsorbate_spacing_scales_with_size():
    probe, _, _, _ = _window(make_slab(), "slab")
    co = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    assert adaptive_grid_characteristic_length(probe) == pytest.approx(probe)
    assert (
        adaptive_grid_spacing(probe, co).initial_spacing
        < adaptive_grid_spacing(probe, molecule("C6H6")).initial_spacing
    )
    # Bulky adsorbates are no longer capped by probe (coarser than probe-only).
    assert adaptive_grid_characteristic_length(probe, molecule("C6H6")) > probe


def test_min_adsorbate_grid_scale_uses_smallest():
    probe = 1.2
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    bz = molecule("C6H6")
    L_h2 = adaptive_grid_characteristic_length(probe, h2)
    L_bz = adaptive_grid_characteristic_length(probe, bz)
    assert L_h2 < L_bz
    assert min_adsorbate_grid_scale(probe, [bz, h2]) == pytest.approx(L_h2)
    assert min_adsorbate_grid_scale(probe, []) == pytest.approx(probe)


def _site(xyz, site_type, source, nn):
    return Site(
        xyz=np.asarray(xyz, dtype=float),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type=site_type,
        slab_indices=(),
        material_type="slab",
        site_source=source,
        env_fingerprint=(),
        nn_distance=nn,
    )


def test_site_ranking_prefers_nn_near_adsorbate_contact():
    preferred = adsorbate_contact_distance(
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        ["Ru"],
    )
    assert preferred is not None and preferred > 0.0
    sites = [
        _site((0.0, 0.0, 0.0), "hollow", "adaptive_grid", preferred + 1.5),
        _site((1.0, 0.0, 0.0), "bridge", "adaptive_grid", preferred),
        _site((2.0, 0.0, 0.0), "atop", "atop_injected", preferred + 0.2),
    ]
    order = _topology_first_site_indices(sites, [0, 1, 2], preferred_nn=preferred)
    # atop_injected preferred first; among adaptive_grid, closer nn wins.
    assert order[0] == 2
    assert order[1] == 1
    assert order[2] == 0


def test_grid_in_accessibility_window_and_np_outside_hull():
    slab = make_slab()
    probe, maxd, cell, pbc = _window(slab, "slab")
    verts, nn, _, _ = generate_adaptive_grid_sites(
        slab.get_positions(),
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        top_layer_tolerance=1.0,
        n_jobs=1,
    )
    assert len(verts) > 0
    assert np.all((nn >= probe - 1e-9) & (nn <= maxd + 1e-9))

    np_atoms = make_nanoparticle()
    probe, maxd, cell, pbc = _window(np_atoms, "nanoparticle")
    verts, _, _, _ = generate_adaptive_grid_sites(
        np_atoms.get_positions(),
        cell,
        pbc,
        material_type="nanoparticle",
        probe_radius=probe,
        max_site_distance=maxd,
        top_layer_tolerance=1.0,
        n_jobs=1,
    )
    hull = _try_convex_hull(np_atoms.get_positions())
    assert hull is not None and len(verts) > 0
    assert np.all(_outside_convex_hull_mask(verts, hull))


def test_shell_offsets_skip_inner_ball():
    frame = np.eye(3)
    probe, max_d, h = 1.2, 3.0, 0.5
    offsets = _shell_offsets(max_d, h, frame, r_min=probe)
    assert len(offsets) > 0
    r = np.linalg.norm(offsets, axis=1)
    assert np.all(r > probe - 1e-9)
    assert np.all(r <= max_d + 1e-9)


def test_nms_pbc_keeps_higher_score_across_boundary():
    """Periodic image twins: higher score wins (not min index)."""
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    # Two points 0.3 Å apart across the a-boundary.
    vertices = np.array(
        [
            [0.1, 5.0, 10.0],
            [9.8, 5.0, 10.0],  # MIC distance ≈ 0.3 Å
            [5.0, 5.0, 10.0],  # far away, kept
        ],
        dtype=float,
    )
    nn = np.array([1.0, 1.0, 1.0], dtype=float)
    # Point 1 (index 1) has higher score than point 0.
    scores = np.array([-1.0, 0.0, -0.5], dtype=float)
    kept, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc)
    assert len(kept) == 2
    # Winner near the boundary should be the high-score twin at x≈9.8.
    near_boundary = kept[np.abs(kept[:, 0] - 5.0) > 1.0]
    assert len(near_boundary) == 1
    assert float(near_boundary[0, 0]) == pytest.approx(9.8, abs=1e-6)


def test_fractional_voxel_seeds_merge_wrapped_and_skewed():
    # Skewed cell: two atoms with nearby fractional coords share one voxel.
    cell = np.array([[6.0, 0.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 8.0]], dtype=float)
    pbc = np.array([True, True, True])
    # frac ≈ (0.05, 0.5, 0.5) and (0.08, 0.5, 0.5) → same voxel at seed_voxel=2.
    positions = _frac_to_cart(
        np.array([[0.05, 0.5, 0.5], [0.08, 0.5, 0.5], [0.7, 0.2, 0.3]], dtype=float),
        cell,
    )
    idx = _fractional_voxel_seeds(
        positions, cell, pbc, seed_voxel=2.0, idx=np.arange(3)
    )
    assert len(idx) == 2
    assert 2 in set(idx.tolist())

    # seed_voxel spanning the full a-period collapses all a-wrapped atoms.
    ortho = np.diag([4.0, 4.0, 4.0])
    pts = np.array([[0.1, 2.0, 2.0], [3.9, 2.0, 2.0]], dtype=float)
    collapsed = _fractional_voxel_seeds(
        pts, ortho, pbc, seed_voxel=4.0, idx=np.arange(2)
    )
    assert len(collapsed) == 1


@pytest.mark.parametrize(
    ("material_type", "factory", "baseline"),
    [
        ("slab", make_slab, "topology"),
        ("nanoparticle", make_nanoparticle, "topology"),
        ("porous", make_porous_framework, "voronoi"),
    ],
)
def test_unified_sites_and_baseline_overlap(material_type, factory, baseline):
    atoms = factory()
    grid = get_unified_sites(
        atoms,
        material_type=material_type,
        site_generator="adaptive_grid",
        n_jobs=1,
    )
    assert len(grid) > 0
    assert {s.site_source for s in grid} <= {"adaptive_grid", "atop_injected"}

    base = get_unified_sites(
        atoms, material_type=material_type, site_generator=baseline
    )
    assert len(base) > 0
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    grid_xyz = np.asarray([s.xyz for s in grid], dtype=float)
    # Same coverage metric for every material: baseline sites near a grid point.
    dists, _ = KDTree(grid_xyz).query(base_xyz, k=1)
    tol = 1.5 if material_type == "porous" else 0.75
    assert float(np.mean(np.asarray(dists, dtype=float) <= tol)) >= 0.25


def test_tilted_slab_still_overlaps_topology():
    """Rotate the slab so the surface normal is not lab-z; overlap bar still holds."""
    slab = make_slab(nx=3, ny=3, n_layers=3)
    # 40° rotation about x tilts a×b away from lab z while keeping slab PBC.
    angle = np.deg2rad(40.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)
    slab.set_cell(np.asarray(slab.get_cell(), dtype=float) @ R.T, scale_atoms=True)
    slab.set_pbc([True, True, False])

    grid = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    base = get_unified_sites(slab, material_type="slab", site_generator="topology")
    assert len(grid) > 0 and len(base) > 0
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    grid_xyz = np.asarray([s.xyz for s in grid], dtype=float)
    dists, _ = KDTree(grid_xyz).query(base_xyz, k=1)
    assert float(np.mean(np.asarray(dists, dtype=float) <= 0.75)) >= 0.25


@pytest.mark.parametrize(
    ("material_type", "factory"),
    [
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
        ("porous", make_porous_framework),
    ],
)
def test_n_jobs_serial_matches_parallel(material_type, factory):
    atoms = factory()
    serial = get_unified_sites(
        atoms,
        material_type=material_type,
        site_generator="adaptive_grid",
        n_jobs=1,
    )
    parallel = get_unified_sites(
        atoms,
        material_type=material_type,
        site_generator="adaptive_grid",
        n_jobs=4,
    )
    assert len(serial) == len(parallel)
    s_xyz = np.asarray([s.xyz for s in serial], dtype=float)
    p_xyz = np.asarray([s.xyz for s in parallel], dtype=float)
    np.testing.assert_allclose(s_xyz, p_xyz, atol=1e-9)


def test_porous_sites_prefer_near_atom_shell_not_pore_centres():
    atoms = make_porous_framework()
    probe, maxd, cell, pbc = _window(atoms, "porous")
    verts, nn, _, _ = generate_adaptive_grid_sites(
        atoms.get_positions(),
        cell,
        pbc,
        material_type="porous",
        probe_radius=probe,
        max_site_distance=maxd,
        top_layer_tolerance=1.0,
        n_jobs=1,
    )
    assert len(verts) > 0
    # Shell target sits near probe / metal-height scale, not at max_d (pore centres).
    assert float(np.median(nn)) < 0.5 * (probe + maxd)
    sites = get_unified_sites(
        atoms, material_type="porous", site_generator="adaptive_grid", n_jobs=1
    )
    types = {s.site_type for s in sites}
    assert types - {"pore"}  # must include near-framework typed sites

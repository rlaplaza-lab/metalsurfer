"""Adaptive-grid site generator: exposure, PBC, shared scale, and A/B overlap."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from scipy.spatial import KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement._constants import (
    _ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE,
    _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE,
    _ADAPTIVE_GRID_WORK_BUDGET,
)
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.generators import (
    _topology_first_site_indices,
    enumerate_placement_specs,
    generate_placements_from_specs,
)
from metalsurfer.placement.site_adaptive_grid import (
    _exposure_mask,
    _fractional_voxel_seeds,
    _framework_median_nn,
    _nms,
    _shell_offsets,
    _work_chunk_bounds,
    adaptive_grid_characteristic_length,
    adaptive_grid_spacing,
    generate_adaptive_grid_sites,
    min_adsorbate_grid_scale,
)
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _SITE_CONTEXT_CACHE_LOCK,
    resolve_site_context_for_sampling,
)
from metalsurfer.placement.site_coords import (
    _derive_voronoi_distance_window,
    _frac_to_cart,
    _slab_normal,
)
from metalsurfer.placement.site_enumeration import (
    get_hollow_sites_for_adatoms,
    get_unified_sites,
)
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


def test_adaptive_grid_accepted_on_adsorption_config():
    cfg = AdsorptionConfig(site_generator="adaptive_grid", material_type="slab")
    assert cfg.site_generator == "adaptive_grid"
    for mat in ("slab", "nanoparticle", "porous"):
        AdsorptionConfig(site_generator="adaptive_grid", material_type=mat)


def test_adaptive_grid_spacing_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="characteristic length"):
        adaptive_grid_spacing(1.2, grid_spacing_scale=0.0)
    with pytest.raises(ValueError, match="characteristic length"):
        adaptive_grid_spacing(1.2, grid_spacing_scale=float("nan"))


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


def test_site_ranking_prefers_topology_then_clearance():
    sites = [
        _site((0.0, 0.0, 0.0), "hollow", "adaptive_grid", 2.0),
        _site((1.0, 0.0, 0.0), "bridge", "adaptive_grid", 1.5),
        _site((2.0, 0.0, 0.0), "atop", "atop_injected", 1.8),
    ]
    clearances = np.array([1.0, 5.0, 2.0], dtype=float)
    order = _topology_first_site_indices(sites, [0, 1, 2], clearances=clearances)
    assert order[0] == 2
    assert order[1] == 1
    assert order[2] == 0


def test_grid_in_accessibility_window():
    slab = make_slab()
    probe, maxd, cell, pbc = _window(slab, "slab")
    verts, nn, _, _ = generate_adaptive_grid_sites(
        slab.get_positions(),
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
    )
    assert len(verts) > 0
    assert np.all((nn >= probe - 1e-9) & (nn <= maxd + 1e-9))


def test_shell_offsets_skip_inner_ball():
    probe, max_d, h = 1.2, 3.0, 0.5
    offsets = _shell_offsets(max_d, h, r_min=probe)
    assert len(offsets) > 0
    r = np.linalg.norm(offsets, axis=1)
    assert np.all(r > probe - 1e-9)
    assert np.all(r <= max_d + 1e-9)


def test_exposure_rejects_buried_keeps_surface():
    """Synthetic: interstitial between two atoms is buried; outer point exposed."""
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    tree = KDTree(positions)
    # Midpoint interstitial — stepping away from nearest still hits the other atom.
    buried = np.array([[1.25, 0.0, 0.0]], dtype=float)
    nn_b, idx_b = tree.query(buried, k=1)
    nearest_b = positions[np.atleast_1d(idx_b)]
    keep_b = _exposure_mask(
        buried,
        np.atleast_1d(nn_b).astype(float),
        nearest_b,
        tree,
        material_type="nanoparticle",
        cell=np.eye(3),
    )
    assert not bool(keep_b[0])

    # Outer point above first atom — stepping away increases nn.
    surface = np.array([[0.0, 0.0, 1.5]], dtype=float)
    nn_s, idx_s = tree.query(surface, k=1)
    nearest_s = positions[np.atleast_1d(idx_s)]
    keep_s = _exposure_mask(
        surface,
        np.atleast_1d(nn_s).astype(float),
        nearest_s,
        tree,
        material_type="nanoparticle",
        cell=np.eye(3),
    )
    assert bool(keep_s[0])


def test_concave_np_pocket_keeps_sites():
    """Missing corner atom leaves a dent; exposure still finds nearby sites."""
    # 2x2x2 cube missing one corner → concave pocket.
    pts = [
        [0.0, 0.0, 0.0],
        [2.5, 0.0, 0.0],
        [0.0, 2.5, 0.0],
        [2.5, 2.5, 0.0],
        [0.0, 0.0, 2.5],
        [2.5, 0.0, 2.5],
        [0.0, 2.5, 2.5],
        # Intentionally omit the +x+y+z corner to leave a concave pocket.
    ]
    atoms = Atoms("Pt7", positions=pts)
    atoms.set_cell([20.0, 20.0, 20.0])
    atoms.center()
    atoms.pbc = False
    probe, maxd, cell, pbc = _window(atoms, "nanoparticle")
    verts, _, _, _ = generate_adaptive_grid_sites(
        atoms.get_positions(),
        cell,
        pbc,
        material_type="nanoparticle",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
    )
    assert len(verts) > 0
    # At least one site near the missing-corner pocket (lab coords after center).
    com = np.mean(atoms.get_positions(), axis=0)
    # Pocket direction is roughly (+x,+y,+z) from the cube center.
    pocket_dir = np.array([1.0, 1.0, 1.0])
    pocket_dir /= np.linalg.norm(pocket_dir)
    rel = verts - com
    proj = rel @ pocket_dir
    assert float(np.max(proj)) > 0.5


def test_stepped_slab_has_lower_terrace_not_bottom():
    """Two-terrace slab: lower terrace sites present; bottom face absent."""
    from ase.build import fcc111

    slab = fcc111("Cu", size=(4, 4, 4), vacuum=10.0, orthogonal=True)
    # Carve a step: remove half of the top layer.
    pos = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_hat = _slab_normal(cell)
    heights = pos @ n_hat
    h_max = float(np.max(heights))
    top = heights > h_max - 0.5
    # Drop atoms with large a-fraction in the top layer.
    frac = pos @ np.linalg.inv(cell).T
    drop = top & (frac[:, 0] > 0.5)
    keep = ~drop
    slab = slab[keep]
    slab.set_pbc([True, True, False])

    probe, maxd, cell, pbc = _window(slab, "slab")
    verts, nn, _, _ = generate_adaptive_grid_sites(
        slab.get_positions(),
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
    )
    assert len(verts) > 0
    n_hat = _slab_normal(cell)
    v_h = verts @ n_hat
    atom_h = slab.get_positions() @ n_hat
    h_min_atoms = float(np.min(atom_h))
    # No sites below the bottom atom plane (half-space filter).
    assert float(np.min(v_h)) >= h_min_atoms - 0.1
    # Sites span more than a single terrace height band.
    assert float(np.max(v_h) - np.min(v_h)) > 0.5


def test_nms_pbc_keeps_higher_score_across_boundary():
    """Periodic image twins: higher score wins (not min index)."""
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    vertices = np.array(
        [
            [0.1, 5.0, 10.0],
            [9.8, 5.0, 10.0],
            [5.0, 5.0, 10.0],
        ],
        dtype=float,
    )
    nn = np.array([1.0, 1.0, 1.0], dtype=float)
    scores = np.array([-1.0, 0.0, -0.5], dtype=float)
    kept, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc)
    assert len(kept) == 2
    near_boundary = kept[np.abs(kept[:, 0] - 5.0) > 1.0]
    assert len(near_boundary) == 1
    assert float(near_boundary[0, 0]) == pytest.approx(9.8, abs=1e-6)


def test_adaptive_grid_retains_bridge_and_hollow_on_slab():
    """Same-class NMS + within-type dedup keep bridges/hollows near topology density."""
    slab = make_slab(nx=3, ny=3, n_layers=3)
    sites = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    topo = get_unified_sites(
        slab, material_type="slab", site_generator="topology", n_jobs=1
    )
    types = {s.site_type for s in sites}
    assert "hollow" in types
    assert "bridge" in types
    assert len(sites) <= max(2 * len(topo), 100)
    assert len(sites) >= max(len(topo) // 4, 8)
    xyz = np.asarray([s.xyz for s in sites if s.site_type == "hollow"], dtype=float)
    if len(xyz) >= 2:
        d, _ = KDTree(xyz).query(xyz, k=2)
        assert float(np.median(d[:, 1])) >= 0.9


def test_fractional_voxel_seeds_merge_wrapped_and_skewed():
    cell = np.array([[6.0, 0.0, 0.0], [2.0, 5.0, 0.0], [0.0, 0.0, 8.0]], dtype=float)
    pbc = np.array([True, True, True])
    positions = _frac_to_cart(
        np.array([[0.05, 0.5, 0.5], [0.08, 0.5, 0.5], [0.7, 0.2, 0.3]], dtype=float),
        cell,
    )
    idx = _fractional_voxel_seeds(
        positions, cell, pbc, seed_voxel=2.0, idx=np.arange(3)
    )
    assert len(idx) == 2
    assert 2 in set(idx.tolist())


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
    assert {s.site_source for s in grid} <= {"adaptive_grid"}
    assert all(s.slab_indices for s in grid)
    assert all(s.env_fingerprint for s in grid)

    base = get_unified_sites(
        atoms, material_type=material_type, site_generator=baseline
    )
    assert len(base) > 0
    # Overlap vs topology/voronoi is measured on the NMS cloud (pre-orbit), since
    # orbit representatives need not sit near every baseline site.
    probe, maxd, cell, pbc = _window(atoms, material_type)
    verts, _, _, _ = generate_adaptive_grid_sites(
        atoms.get_positions(),
        cell,
        pbc,
        material_type=material_type,
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
    )
    assert len(verts) > 0
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    dists, _ = KDTree(verts).query(base_xyz, k=1)
    # Framework NN floors coarsen NP shells relative to topology; allow a wider
    # match radius / lower hit rate than slabs.
    if material_type == "porous":
        tol, frac_min = 1.5, 0.25
    elif material_type == "nanoparticle":
        tol, frac_min = 1.0, 0.15
    else:
        tol, frac_min = 0.75, 0.25
    assert float(np.mean(np.asarray(dists, dtype=float) <= tol)) >= frac_min


def test_tilted_slab_still_overlaps_topology():
    slab = make_slab(nx=3, ny=3, n_layers=3)
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
    probe, maxd, cell, pbc = _window(slab, "slab")
    verts, _, _, _ = generate_adaptive_grid_sites(
        slab.get_positions(),
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
    )
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    dists, _ = KDTree(verts).query(base_xyz, k=1)
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
        n_jobs=1,
    )
    assert len(verts) > 0
    assert float(np.median(nn)) < 0.5 * (probe + maxd)
    sites = get_unified_sites(
        atoms, material_type="porous", site_generator="adaptive_grid", n_jobs=1
    )
    types = {s.site_type for s in sites}
    assert types - {"pore"}


def test_adaptive_grid_site_context_uses_shared_symmetry_path():
    """adaptive_grid goes through the same spglib path as topology/voronoi."""
    slab = make_slab(nx=3, ny=3, n_layers=3)
    raw = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    assert len(raw) > 0

    cfg = AdsorptionConfig(
        material_type="slab",
        site_generator="adaptive_grid",
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()
    ctx = resolve_site_context_for_sampling(slab, cfg, symmetry_broken=False)
    assert ctx.use_sites
    assert ctx.source == "symmetry_aware" or len(ctx.sites) <= len(raw)
    assert len(ctx.sites) <= len(raw)
    assert all(s.slab_indices for s in ctx.sites)


def test_shared_min_scale_catalog_denser_than_bulky_only():
    slab = make_slab()
    probe, _, _, _ = _window(slab, "slab")
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    bz = molecule("C6H6")
    scale_min = min_adsorbate_grid_scale(probe, [h2, bz])
    scale_bz = adaptive_grid_characteristic_length(probe, bz)
    assert scale_min < scale_bz
    spacing_min = adaptive_grid_spacing(probe, grid_spacing_scale=scale_min)
    spacing_bz = adaptive_grid_spacing(probe, grid_spacing_scale=scale_bz)
    assert spacing_min.initial_spacing <= spacing_bz.initial_spacing
    assert spacing_min.merge_radius <= spacing_bz.merge_radius
    dense = get_unified_sites(
        slab,
        material_type="slab",
        site_generator="adaptive_grid",
        grid_spacing_scale=scale_min,
        n_jobs=1,
    )
    coarse = get_unified_sites(
        slab,
        material_type="slab",
        site_generator="adaptive_grid",
        grid_spacing_scale=scale_bz,
        n_jobs=1,
    )
    assert len(dense) > 0 and len(coarse) > 0


def test_porous_adaptive_specs_include_non_pore():
    atoms = make_porous_framework()
    ads = Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]])
    cfg = AdsorptionConfig(
        material_type="porous",
        site_generator="adaptive_grid",
        num_placements=24,
        num_conformers=1,
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    specs = enumerate_placement_specs([ads], atoms, cfg, smiles="O=C=O", n_desired=24)
    assert specs
    sites = get_unified_sites(
        atoms, material_type="porous", site_generator="adaptive_grid", n_jobs=1
    )
    non_pore = {i for i, s in enumerate(sites) if s.site_type != "pore"}
    used = {int(sp.site_index) for sp in specs if sp.site_index is not None}
    # Adaptive catalogs must not be pore-capped away.
    if non_pore:
        assert used & non_pore


def test_adaptive_grid_floors_density_against_framework_nn():
    """Tiny adsorbate scales must not densify below framework-aware NMS floors."""
    cluster = make_nanoparticle()
    pos = cluster.get_positions()
    cell = np.asarray(cluster.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc("nanoparticle"), dtype=bool)
    median_nn = _framework_median_nn(pos, cell, pbc)
    assert median_nn > 0.0
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    scale = min_adsorbate_grid_scale(None, [h2])
    unfloored = adaptive_grid_spacing(1.2, grid_spacing_scale=scale)
    floored = adaptive_grid_spacing(
        1.2, grid_spacing_scale=scale, framework_median_nn=median_nn
    )
    length_floor = max(scale, _ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE * median_nn)
    merge_floor = _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE * median_nn
    assert floored.characteristic_length >= length_floor - 1e-12
    assert floored.merge_radius >= merge_floor - 1e-12
    assert floored.merge_radius >= unfloored.merge_radius
    sites = get_unified_sites(
        cluster,
        material_type="nanoparticle",
        site_generator="adaptive_grid",
        n_jobs=1,
        grid_spacing_scale=scale,
    )
    assert len(sites) > 0
    verts, _, spacing_out, _ = generate_adaptive_grid_sites(
        pos,
        cell,
        pbc,
        material_type="nanoparticle",
        probe_radius=1.2,
        max_site_distance=3.5,
        grid_spacing_scale=scale,
        n_jobs=1,
    )
    assert len(verts) > 0
    assert spacing_out.merge_radius >= merge_floor - 1e-12


@pytest.mark.parametrize(
    ("material_type", "factory", "ads"),
    [
        (
            "slab",
            make_slab,
            Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
        ),
        (
            "nanoparticle",
            make_nanoparticle,
            Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
        ),
        (
            "porous",
            make_porous_framework,
            Atoms("CO2", positions=[[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]),
        ),
    ],
)
def test_adaptive_grid_placement_materialization(material_type, factory, ads):
    atoms = factory()
    cfg = AdsorptionConfig(
        material_type=material_type,
        site_generator="adaptive_grid",
        num_placements=8,
        num_conformers=1,
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    specs = enumerate_placement_specs([ads], atoms, cfg, smiles="C=O", n_desired=8)
    assert specs
    results = generate_placements_from_specs(specs, [ads], atoms, cfg, smiles="C=O")
    ok = [pair for pair, reason in results if pair is not None]
    assert ok
    for placed, _desc in ok:
        pos = np.asarray(placed.get_positions(), dtype=float)
        assert pos.ndim == 2 and len(pos) == len(ads)
        assert np.all(np.isfinite(pos))
    sites = get_unified_sites(
        atoms, material_type=material_type, site_generator="adaptive_grid", n_jobs=1
    )
    assert all(s.slab_indices for s in sites)


def test_adaptive_grid_dissociative_hollows():
    for material_type, factory in (
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
    ):
        atoms = factory()
        hollows = get_hollow_sites_for_adatoms(
            atoms,
            material_type=material_type,
            site_generator="adaptive_grid",
        )
        assert any(s.site_type == "hollow" for s in hollows)


def test_work_chunks_cover_all_atoms_without_dropping_seeds():
    """Large fixtures still seed every atom; chunks stay under the work budget."""
    slab = make_slab(nx=4, ny=4, n_layers=3)
    n_atoms = len(slab)
    assert n_atoms > 20
    n_offsets = 50_000
    bounds = _work_chunk_bounds(n_atoms, n_offsets)
    assert bounds
    covered = 0
    for lo, hi in bounds:
        assert (hi - lo) * n_offsets <= _ADAPTIVE_GRID_WORK_BUDGET
        covered += hi - lo
    assert covered == n_atoms
    sites = get_unified_sites(
        slab,
        material_type="slab",
        site_generator="adaptive_grid",
        n_jobs=1,
    )
    assert len(sites) > 0

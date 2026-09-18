"""Adaptive-grid site generator: spacing, exposure, NMS, and a lean slab smoke path."""

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
)
from metalsurfer.placement.site_enumeration import (
    get_hollow_sites_for_adatoms,
    get_unified_sites,
)
from metalsurfer.placement.site_types import Site

from ..conftest import make_nanoparticle, make_slab


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
    spacing_min = adaptive_grid_spacing(probe, grid_spacing_scale=L_h2)
    spacing_bz = adaptive_grid_spacing(probe, grid_spacing_scale=L_bz)
    assert spacing_min.initial_spacing <= spacing_bz.initial_spacing
    assert spacing_min.merge_radius <= spacing_bz.merge_radius


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
    offsets = _shell_offsets(3.0, 0.5, r_min=1.2)
    assert len(offsets) > 0
    r = np.linalg.norm(offsets, axis=1)
    assert np.all(r > 1.2 - 1e-9)
    assert np.all(r <= 3.0 + 1e-9)


def test_exposure_rejects_buried_keeps_surface():
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    tree = KDTree(positions)
    buried = np.array([[1.25, 0.0, 0.0]], dtype=float)
    nn_b, idx_b = tree.query(buried, k=1)
    keep_b = _exposure_mask(
        buried,
        np.atleast_1d(nn_b).astype(float),
        positions[np.atleast_1d(idx_b)],
        tree,
        material_type="nanoparticle",
        cell=np.eye(3),
    )
    assert not bool(keep_b[0])

    surface = np.array([[0.0, 0.0, 1.5]], dtype=float)
    nn_s, idx_s = tree.query(surface, k=1)
    keep_s = _exposure_mask(
        surface,
        np.atleast_1d(nn_s).astype(float),
        positions[np.atleast_1d(idx_s)],
        tree,
        material_type="nanoparticle",
        cell=np.eye(3),
    )
    assert bool(keep_s[0])


def test_nms_pbc_keeps_higher_score_across_boundary():
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    vertices = np.array(
        [[0.1, 5.0, 10.0], [9.8, 5.0, 10.0], [5.0, 5.0, 10.0]],
        dtype=float,
    )
    nn = np.array([1.0, 1.0, 1.0], dtype=float)
    scores = np.array([-1.0, 0.0, -0.5], dtype=float)
    kept, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc)
    assert len(kept) == 2
    near_boundary = kept[np.abs(kept[:, 0] - 5.0) > 1.0]
    assert len(near_boundary) == 1
    assert float(near_boundary[0, 0]) == pytest.approx(9.8, abs=1e-6)


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


def test_work_chunks_cover_all_atoms_without_dropping_seeds():
    n_atoms, n_offsets = 48, 50_000
    bounds = _work_chunk_bounds(n_atoms, n_offsets)
    assert bounds
    covered = 0
    for lo, hi in bounds:
        assert (hi - lo) * n_offsets <= _ADAPTIVE_GRID_WORK_BUDGET
        covered += hi - lo
    assert covered == n_atoms


def test_adaptive_grid_floors_density_against_framework_nn():
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
    assert floored.characteristic_length >= (
        max(scale, _ADAPTIVE_GRID_LENGTH_FRAMEWORK_SCALE * median_nn) - 1e-12
    )
    assert (
        floored.merge_radius >= _ADAPTIVE_GRID_NMS_FRAMEWORK_SCALE * median_nn - 1e-12
    )
    assert floored.merge_radius >= unfloored.merge_radius
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
    assert spacing_out.merge_radius >= floored.merge_radius - 1e-12


def test_unified_sites_overlap_topology_on_slab():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    grid = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    assert len(grid) > 0
    assert {s.site_source for s in grid} <= {"adaptive_grid"}
    assert all(s.slab_indices and s.env_fingerprint for s in grid)
    types = {s.site_type for s in grid}
    assert "hollow" in types and "bridge" in types
    base = get_unified_sites(slab, material_type="slab", site_generator="topology")
    assert len(grid) <= max(2 * len(base), 100)
    assert len(grid) >= max(len(base) // 4, 8)
    grid_xyz = np.asarray([s.xyz for s in grid], dtype=float)
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    dists, _ = KDTree(grid_xyz).query(base_xyz, k=1)
    assert float(np.mean(np.asarray(dists, dtype=float) <= 0.75)) >= 0.25
    hollow = grid_xyz[[s.site_type == "hollow" for s in grid]]
    if len(hollow) >= 2:
        d, _ = KDTree(hollow).query(hollow, k=2)
        assert float(np.median(d[:, 1])) >= 0.9


def test_adaptive_grid_site_context_uses_shared_symmetry_path():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    cfg = AdsorptionConfig(
        material_type="slab",
        site_generator="adaptive_grid",
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()
    ctx = resolve_site_context_for_sampling(slab, cfg, symmetry_broken=False)
    assert ctx.use_sites and len(ctx.sites) > 0
    assert all(s.slab_indices for s in ctx.sites)


def test_adaptive_grid_placement_materialization_on_slab():
    ads = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    atoms = make_slab()
    cfg = AdsorptionConfig(
        material_type="slab",
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


def test_adaptive_grid_dissociative_hollows_on_slab():
    hollows = get_hollow_sites_for_adatoms(
        make_slab(),
        material_type="slab",
        site_generator="adaptive_grid",
    )
    assert any(s.site_type == "hollow" for s in hollows)

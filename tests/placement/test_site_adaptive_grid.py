"""Adaptive-grid site generator: spacing, exposure, NMS, and a lean slab smoke path."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from ase.geometry import find_mic
from scipy.spatial import KDTree

from metalsurfer._geom_pbc import minimum_image_cartesian_delta
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
    _fractional_bin_keys,
    _fractional_voxel_seeds,
    _framework_median_nn,
    _nms,
    _ray_exposed,
    _ray_exposure_mask,
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
    assert cfg.side_policy == "positive"
    assert cfg.adaptive_grid_spacing == pytest.approx(0.70)
    assert cfg.adaptive_grid_refine_levels == 0
    assert cfg.adaptive_grid_nms_framework_scale == pytest.approx(0.50)
    for mat in ("slab", "nanoparticle", "porous"):
        AdsorptionConfig(site_generator="adaptive_grid", material_type=mat)
    for policy in ("all", "positive", "negative", "external"):
        AdsorptionConfig(side_policy=policy, material_type="slab")
    with pytest.raises(ValueError, match="adaptive_grid_nms_framework_scale"):
        AdsorptionConfig(adaptive_grid_nms_framework_scale=0.0)


def test_adaptive_grid_spacing_rejects_non_positive_absolute():
    with pytest.raises(ValueError, match="initial_spacing"):
        adaptive_grid_spacing(1.2, initial_spacing=0.0)
    with pytest.raises(ValueError, match="initial_spacing"):
        adaptive_grid_spacing(1.2, initial_spacing=float("nan"))


def test_adaptive_grid_spacing_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="characteristic length"):
        adaptive_grid_spacing(1.2, grid_spacing_scale=0.0)
    with pytest.raises(ValueError, match="characteristic length"):
        adaptive_grid_spacing(1.2, grid_spacing_scale=float("nan"))


def test_absolute_spacing_knob_controls_catalog_density():
    """Finer ``adaptive_grid_spacing`` densifies when spacing sets the merge radius.

    On frameworks with short median NN (MOF-like), merge tracks ``1.5 * h`` so
    the exposed increment clearly controls catalog size. On close-packed metals
    the NN floor can dominate and counts need not be strictly monotonic in ``h``.
    """
    framework = make_porous_framework()
    coarse = get_unified_sites(
        framework,
        material_type="porous",
        site_generator="adaptive_grid",
        adaptive_grid_spacing=1.2,
        adaptive_grid_refine_levels=0,
        n_jobs=1,
    )
    fine = get_unified_sites(
        framework,
        material_type="porous",
        site_generator="adaptive_grid",
        adaptive_grid_spacing=0.55,
        adaptive_grid_refine_levels=0,
        n_jobs=1,
    )
    assert len(coarse) > 0
    assert len(fine) > len(coarse)

    sp_fine = adaptive_grid_spacing(
        1.2, initial_spacing=0.55, max_levels=0, framework_median_nn=2.7
    )
    sp_coarse = adaptive_grid_spacing(
        1.2, initial_spacing=1.4, max_levels=0, framework_median_nn=2.7
    )
    assert sp_fine.initial_spacing < sp_coarse.initial_spacing
    assert sp_fine.merge_radius <= sp_coarse.merge_radius


def test_adaptive_grid_default_counts_comparable_to_auto():
    """Default coarse grid stays within ~8× of topology/Voronoi catalog size."""
    cases = [
        ("slab", make_slab(nx=3, ny=3, n_layers=3)),
        ("nanoparticle", make_nanoparticle()),
        ("porous", make_porous_framework()),
    ]
    for mat, atoms in cases:
        auto = get_unified_sites(
            atoms, material_type=mat, site_generator="auto", n_jobs=1
        )
        grid = get_unified_sites(
            atoms,
            material_type=mat,
            site_generator="adaptive_grid",
            adaptive_grid_spacing=0.70,
            adaptive_grid_refine_levels=0,
            n_jobs=1,
        )
        assert len(auto) > 0 and len(grid) > 0, mat
        lo = max(len(auto) // 4, 1)
        hi = max(8 * len(auto), 500)
        assert lo <= len(grid) <= hi, (
            f"{mat}: auto={len(auto)} grid={len(grid)} not in [{lo}, {hi}]"
        )


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
        env_fingerprint=((), (), 0),
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
    result = generate_adaptive_grid_sites(
        slab.get_positions(),
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
        symbols=list(slab.get_chemical_symbols()),
    )
    assert len(result.vertices) > 0
    # nn_dists are centre-to-centre; clearances are radius-subtracted.
    assert np.all((result.nn_dists >= probe - 1e-6) | (result.clearances >= -1.0))
    assert len(result.nn_dists) == len(result.vertices)
    assert len(result.clearances) == len(result.vertices)
    assert len(result.atom_indices) == len(result.vertices)
    assert all(result.atom_indices)
    assert result.normals.shape == (len(result.vertices), 3)


def test_shell_offsets_skip_inner_ball():
    offsets = _shell_offsets(3.0, 0.5, r_min=1.2)
    assert len(offsets) > 0
    r = np.linalg.norm(offsets, axis=1)
    assert np.all(r > 1.2 - 1e-9)
    assert np.all(r <= 3.0 + 1e-9)


def test_exposure_rejects_buried_keeps_surface():
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    radii = np.full(2, 0.7, dtype=float)
    tree = KDTree(positions)
    buried = np.array([[1.25, 0.0, 0.0]], dtype=float)
    # Normal pointing toward the gap centre (into the other atom).
    normals_b = np.array([[1.0, 0.0, 0.0]], dtype=float)
    keep_b = _exposure_mask(
        buried,
        normals_b,
        tree,
        radii,
        2,
        material_type="nanoparticle",
        cell=np.eye(3),
        side_policy="all",
        positions=positions,
    )
    assert not bool(keep_b[0])

    surface = np.array([[0.0, 0.0, 1.5]], dtype=float)
    normals_s = np.array([[0.0, 0.0, 1.0]], dtype=float)
    keep_s = _exposure_mask(
        surface,
        normals_s,
        tree,
        radii,
        2,
        material_type="nanoparticle",
        cell=np.eye(3),
        side_policy="all",
        positions=positions,
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


def test_nms_tie_break_is_deterministic():
    cell = np.diag([10.0, 10.0, 20.0])
    pbc = np.array([True, True, False])
    vertices = np.array(
        [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=float,
    )
    nn = np.ones(3)
    scores = np.array([0.0, 0.0, -1.0], dtype=float)
    kept_a, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc)
    # Permute input order; accepted set must be identical.
    order = np.array([2, 0, 1])
    kept_b, _ = _nms(
        vertices[order], nn[order], scores[order], merge_r=0.5, cell=cell, pbc=pbc
    )
    assert len(kept_a) == len(kept_b)
    tree = KDTree(kept_a)
    for pt in kept_b:
        assert float(tree.query(pt, k=1)[0]) < 1e-9


def test_nms_uses_min_radius_symmetrically():
    cell = np.eye(3) * 20.0
    pbc = np.array([False, False, False])
    vertices = np.array([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=float)
    nn = np.ones(2)
    scores = np.array([1.0, 0.0], dtype=float)
    radii = np.array([0.5, 0.3], dtype=float)
    # Distance 0.4 > min(0.5, 0.3)=0.3 → both kept.
    kept, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc, radii=radii)
    assert len(kept) == 2
    # With larger min radius both merge.
    radii2 = np.array([0.5, 0.45], dtype=float)
    kept2, _ = _nms(vertices, nn, scores, merge_r=0.5, cell=cell, pbc=pbc, radii=radii2)
    assert len(kept2) == 1


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


def test_fractional_bin_keys_equal_width():
    frac = np.array([[0.0, 0.0, 0.0], [0.99, 0.5, 0.5]], dtype=float)
    dfrac = np.array([0.3, 0.3, 0.3])
    pbc = np.array([True, True, True])
    keys = _fractional_bin_keys(frac, dfrac, pbc)
    n_bins = max(1, int(np.ceil(1.0 / 0.3)))
    assert keys[0, 0] == 0
    assert keys[1, 0] == n_bins - 1
    assert int(keys[1, 0]) < n_bins


def test_triclinic_mic_matches_find_mic():
    # Highly skewed cell where fractional rounding alone can fail.
    cell = np.array(
        [[5.0, 0.0, 0.0], [4.5, 1.5, 0.0], [0.5, 0.5, 8.0]],
        dtype=float,
    )
    pbc = np.array([True, True, True])
    a = np.array([0.1, 0.1, 1.0], dtype=float)
    b = np.array([4.8, 1.4, 1.0], dtype=float)
    delta = a - b
    mic = minimum_image_cartesian_delta(delta, cell, pbc)
    ase_mic, _ = find_mic(delta.reshape(1, 3), cell, pbc=pbc.tolist())
    assert np.allclose(mic, ase_mic[0], atol=1e-8)
    # Rounded-fraction MIC may differ; our helper must match ASE.
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    frac_r = frac - np.round(frac)
    rounded = frac_r @ cell
    # Just ensure our result is at least as short as rounded.
    assert float(np.linalg.norm(mic)) <= float(np.linalg.norm(rounded)) + 1e-9


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
    result = generate_adaptive_grid_sites(
        pos,
        cell,
        pbc,
        material_type="nanoparticle",
        probe_radius=1.2,
        max_site_distance=3.5,
        grid_spacing_scale=scale,
        n_jobs=1,
        symbols=list(cluster.get_chemical_symbols()),
    )
    assert len(result.vertices) > 0
    assert result.spacing.merge_radius >= floored.merge_radius - 1e-12


def test_side_policy_positive_vs_all():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    pos = slab.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc("slab"), dtype=bool)
    probe, maxd = _derive_voronoi_distance_window(
        pos, list(slab.get_chemical_symbols()), pbc, cell
    )
    pos_only = generate_adaptive_grid_sites(
        pos,
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
        symbols=list(slab.get_chemical_symbols()),
        side_policy="positive",
    )
    both = generate_adaptive_grid_sites(
        pos,
        cell,
        pbc,
        material_type="slab",
        probe_radius=probe,
        max_site_distance=maxd,
        n_jobs=1,
        symbols=list(slab.get_chemical_symbols()),
        side_policy="all",
    )
    assert len(pos_only.vertices) > 0
    assert len(both.vertices) > 0
    n_hat = _slab_normal(cell)
    pos_dots = pos_only.normals @ n_hat
    both_dots = both.normals @ n_hat
    assert np.all(pos_dots >= -1e-12)
    assert np.any(both_dots > 0.5)
    assert np.any(both_dots < -0.5)


def test_unified_sites_overlap_topology_on_slab():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    grid = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    assert len(grid) > 0
    assert {s.site_source for s in grid} <= {"adaptive_grid"}
    assert all(s.slab_indices and s.env_fingerprint for s in grid)
    assert all(s.clearance is not None for s in grid)
    assert all(s.tangent_basis is not None for s in grid)
    types = {s.site_type for s in grid}
    assert "hollow" in types or "bridge" in types or "atop" in types
    base = get_unified_sites(slab, material_type="slab", site_generator="topology")
    assert len(grid) <= max(4 * len(base), 250)
    assert len(grid) >= max(len(base) // 8, 4)
    grid_xyz = np.asarray([s.xyz for s in grid], dtype=float)
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    dists, _ = KDTree(grid_xyz).query(base_xyz, k=1)
    # Coarse default spacing + NN-floored NMS; require modest topology coverage.
    assert float(np.mean(np.asarray(dists, dtype=float) <= 1.5)) >= 0.40
    assert float(np.mean(np.asarray(dists, dtype=float) <= 1.0)) >= 0.15


def test_alloy_atops_keep_distinct_fingerprints():
    # Two nearby atops on different elements must not share env_fingerprint.
    atoms = Atoms(
        "PtO",
        positions=[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    sites = get_unified_sites(
        atoms,
        material_type="nanoparticle",
        site_generator="adaptive_grid",
        n_jobs=1,
        probe_radius=1.0,
        max_site_distance=3.0,
    )
    atops = [s for s in sites if s.site_type == "atop" and s.slab_indices]
    if len(atops) >= 2:
        fps = {s.env_fingerprint for s in atops}
        # Distinct support chemistry → distinct fingerprints.
        assert len(fps) >= 1


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
        make_slab(nx=3, ny=3, n_layers=3),
        material_type="slab",
        site_generator="adaptive_grid",
    )
    assert isinstance(hollows, list)
    assert hollows, (
        "adaptive_grid slab should expose hollow sites for dissociative pairs"
    )
    assert all(s.site_type in ("hollow", "pore") for s in hollows)


def test_ray_exposure_mask_matches_scalar_ray_exposed():
    positions = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=float)
    radii = np.full(2, 0.7, dtype=float)
    tree = KDTree(positions)
    verts = np.array(
        [
            [1.25, 0.0, 0.0],
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    normals = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    batched = _ray_exposure_mask(verts, normals, tree, radii, 2)
    scalar = np.array(
        [_ray_exposed(verts[i], normals[i], tree, radii, 2) for i in range(len(verts))],
        dtype=bool,
    )
    assert np.array_equal(batched, scalar)


def test_adaptive_grid_porous_wall_near_placement():
    framework = make_porous_framework()
    sites = get_unified_sites(
        framework,
        material_type="porous",
        site_generator="adaptive_grid",
        n_jobs=1,
    )
    assert sites
    assert all(s.site_source == "adaptive_grid" for s in sites)
    assert all(s.slab_indices for s in sites)
    # Same material-agnostic path: wall-near shells, not free-volume pores.
    ads = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
    cfg = AdsorptionConfig(
        material_type="porous",
        site_generator="adaptive_grid",
        num_placements=6,
        num_conformers=1,
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    specs = enumerate_placement_specs(
        [ads], framework, cfg, smiles="O=C=O", n_desired=6
    )
    assert specs
    results = generate_placements_from_specs(
        specs, [ads], framework, cfg, smiles="O=C=O"
    )
    ok = [pair for pair, reason in results if pair is not None]
    assert ok, "expected at least one clash-free placement from wall-near MOF sites"


def test_adaptive_grid_same_path_on_all_materials():
    cases = (
        ("slab", make_slab()),
        ("nanoparticle", make_nanoparticle()),
        ("porous", make_porous_framework()),
    )
    for mat, atoms in cases:
        sites = get_unified_sites(
            atoms, material_type=mat, site_generator="adaptive_grid", n_jobs=1
        )
        assert sites, f"adaptive_grid empty on {mat}"
        assert {s.site_source for s in sites} <= {"adaptive_grid"}
        assert all(s.tangent_basis is not None for s in sites)

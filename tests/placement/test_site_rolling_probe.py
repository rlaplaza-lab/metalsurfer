"""Rolling-probe site generator: contacts, exposure, and lean smoke paths."""

import numpy as np
import pytest
from ase import Atoms
from scipy.spatial import KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement.generators import (
    enumerate_placement_specs,
    generate_placements_from_specs,
)
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _SITE_CONTEXT_CACHE_LOCK,
    resolve_site_context_for_sampling,
)
from metalsurfer.placement.site_enumeration import (
    get_hollow_sites_for_adatoms,
    get_unified_sites,
)
from metalsurfer.placement.site_rolling_probe import (
    _fibonacci_sphere,
    _three_sphere_intersections,
    _two_sphere_circle_samples,
)

from ..conftest import make_nanoparticle, make_porous_framework, make_slab


def test_rolling_probe_accepted_on_adsorption_config():
    cfg = AdsorptionConfig(site_generator="rolling_probe", material_type="slab")
    assert cfg.site_generator == "rolling_probe"
    assert cfg.side_policy == "positive"
    for mat in ("slab", "nanoparticle", "porous"):
        AdsorptionConfig(site_generator="rolling_probe", material_type=mat)


def test_fibonacci_sphere_unit_norm():
    pts = _fibonacci_sphere(12)
    assert pts.shape == (12, 3)
    assert np.allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-9)


def test_three_sphere_intersection_equilateral():
    c1 = np.array([0.0, 0.0, 0.0])
    c2 = np.array([2.0, 0.0, 0.0])
    c3 = np.array([1.0, np.sqrt(3.0), 0.0])
    sols = _three_sphere_intersections(c1, 1.5, c2, 1.5, c3, 1.5)
    assert len(sols) == 2
    for p in sols:
        assert abs(float(np.linalg.norm(p - c1)) - 1.5) < 1e-6
        assert abs(float(np.linalg.norm(p - c2)) - 1.5) < 1e-6
        assert abs(float(np.linalg.norm(p - c3)) - 1.5) < 1e-6


def test_two_sphere_circle_samples_on_radius():
    samples = _two_sphere_circle_samples(
        np.array([0.0, 0.0, 0.0]), 1.5, np.array([2.0, 0.0, 0.0]), 1.5, 8
    )
    assert len(samples) == 8
    for p in samples:
        assert abs(float(np.linalg.norm(p - [0.0, 0.0, 0.0])) - 1.5) < 1e-6
        assert abs(float(np.linalg.norm(p - [2.0, 0.0, 0.0])) - 1.5) < 1e-6


def test_unified_sites_overlap_topology_on_slab():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    grid = get_unified_sites(
        slab, material_type="slab", site_generator="rolling_probe", n_jobs=1
    )
    assert len(grid) > 0
    assert {s.site_source for s in grid} <= {"rolling_probe"}
    assert all(s.slab_indices and s.env_fingerprint for s in grid)
    assert all(s.clearance is not None and s.tangent_basis is not None for s in grid)
    assert {s.site_type for s in grid} & {"hollow", "bridge", "atop"}
    base = get_unified_sites(slab, material_type="slab", site_generator="topology")
    assert max(len(base) // 8, 2) <= len(grid) <= max(4 * len(base), 250)
    grid_xyz = np.asarray([s.xyz for s in grid], dtype=float)
    base_xyz = np.asarray([s.xyz for s in base], dtype=float)
    dists, _ = KDTree(grid_xyz).query(base_xyz, k=1)
    assert float(np.mean(np.asarray(dists, dtype=float) <= 1.5)) >= 0.50


def test_side_policy_splits_slab_faces():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    pos = get_unified_sites(
        slab,
        material_type="slab",
        site_generator="rolling_probe",
        side_policy="positive",
        n_jobs=1,
    )
    neg = get_unified_sites(
        slab,
        material_type="slab",
        site_generator="rolling_probe",
        side_policy="negative",
        n_jobs=1,
    )
    assert pos and neg
    assert np.mean([s.xyz[2] for s in pos]) > np.mean([s.xyz[2] for s in neg])


def test_rolling_probe_has_bridges_and_hollows_on_slab():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    sites = get_unified_sites(
        slab, material_type="slab", site_generator="rolling_probe", n_jobs=1
    )
    types = {s.site_type for s in sites}
    assert types & {"bridge", "hollow"}
    adap = get_unified_sites(
        slab, material_type="slab", site_generator="adaptive_grid", n_jobs=1
    )
    assert max(len(adap) // 3, 8) <= len(sites) <= max(4 * len(adap), 200)


def test_porous_wall_near_not_pore_centres():
    """Pore-wall contacts exist; nn stays wall-near (not Voronoi pore centres)."""
    framework = make_porous_framework()
    rp = get_unified_sites(
        framework,
        material_type="porous",
        site_generator="rolling_probe",
        n_jobs=1,
    )
    assert rp
    assert {s.site_type for s in rp} & {"atop", "bridge", "hollow"}
    assert any(s.site_type in ("bridge", "hollow") for s in rp)
    assert all(s.site_source == "rolling_probe" and s.slab_indices for s in rp)
    vor = get_unified_sites(
        framework, material_type="porous", site_generator="voronoi", n_jobs=1
    )
    pore_nn = [float(s.nn_distance or 0.0) for s in vor if s.site_type == "pore"]
    rp_nn = float(np.median([float(s.nn_distance or 0.0) for s in rp]))
    if pore_nn:
        assert rp_nn < float(np.median(pore_nn)) + 0.5

    ads = Atoms("CO2", positions=[[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])
    cfg = AdsorptionConfig(
        material_type="porous",
        site_generator="rolling_probe",
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
    assert any(pair is not None for pair, _reason in results)


def test_nanoparticle_sites_are_external():
    atoms = make_nanoparticle()
    sites = get_unified_sites(
        atoms, material_type="nanoparticle", site_generator="rolling_probe", n_jobs=1
    )
    assert sites
    com = np.mean(atoms.get_positions(), dtype=float, axis=0)
    r_hull = float(np.max(np.linalg.norm(atoms.get_positions() - com, axis=1)))
    for s in sites:
        r = float(np.linalg.norm(np.asarray(s.xyz, dtype=float) - com))
        assert r > 0.5 * r_hull


def test_rolling_probe_placement_materialization_on_slab():
    ads = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]])
    atoms = make_slab()
    cfg = AdsorptionConfig(
        material_type="slab",
        site_generator="rolling_probe",
        num_placements=8,
        num_conformers=1,
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    specs = enumerate_placement_specs([ads], atoms, cfg, smiles="C=O", n_desired=8)
    assert specs
    results = generate_placements_from_specs(specs, [ads], atoms, cfg, smiles="C=O")
    ok = [pair for pair, _reason in results if pair is not None]
    assert ok
    for placed, _desc in ok:
        pos = np.asarray(placed.get_positions(), dtype=float)
        assert pos.shape == (len(ads), 3)
        assert np.all(np.isfinite(pos))


def test_rolling_probe_dissociative_hollows_on_slab():
    hollows = get_hollow_sites_for_adatoms(
        make_slab(nx=3, ny=3, n_layers=3),
        material_type="slab",
        site_generator="rolling_probe",
    )
    assert hollows
    assert all(s.site_type in ("hollow", "pore") for s in hollows)


@pytest.mark.parametrize(
    ("material_type", "factory"),
    [
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
        ("porous", make_porous_framework),
    ],
)
def test_rolling_probe_same_path_on_all_materials(material_type, factory):
    sites = get_unified_sites(
        factory(),
        material_type=material_type,
        site_generator="rolling_probe",
        n_jobs=1,
    )
    assert sites
    assert {s.site_source for s in sites} <= {"rolling_probe"}
    assert all(s.tangent_basis is not None for s in sites)


def test_rolling_probe_site_context_uses_shared_symmetry_path():
    slab = make_slab(nx=3, ny=3, n_layers=3)
    cfg = AdsorptionConfig(
        material_type="slab",
        site_generator="rolling_probe",
        n_jobs=1,
        slab_relaxation_mode="none",
    )
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()
    ctx = resolve_site_context_for_sampling(slab, cfg, symmetry_broken=False)
    assert ctx.use_sites and len(ctx.sites) > 0
    assert all(s.slab_indices for s in ctx.sites)

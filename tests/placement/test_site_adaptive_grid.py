"""Adaptive-grid site generator: spacing, accessibility, and A/B overlap."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule
from scipy.spatial import KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.site_adaptive_grid import (
    adaptive_grid_characteristic_length,
    adaptive_grid_spacing,
    generate_adaptive_grid_sites,
)
from metalsurfer.placement.site_coords import _derive_voronoi_distance_window
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.placement.site_np import _outside_convex_hull_mask, _try_convex_hull

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
    )
    hull = _try_convex_hull(np_atoms.get_positions())
    assert hull is not None and len(verts) > 0
    assert np.all(_outside_convex_hull_mask(verts, hull))


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
        atoms, material_type=material_type, site_generator="adaptive_grid"
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
    )
    assert len(verts) > 0
    # Shell target sits near probe / metal-height scale, not at max_d (pore centres).
    assert float(np.median(nn)) < 0.5 * (probe + maxd)
    sites = get_unified_sites(
        atoms, material_type="porous", site_generator="adaptive_grid"
    )
    types = {s.site_type for s in sites}
    assert types - {"pore"}  # must include near-framework typed sites

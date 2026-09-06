"""Unit tests for Packmol-style overlap penalty and rigid-body clash descent."""

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement import geometry as geom
from metalsurfer.placement.clash import (
    atom_radii_for_symbols,
    overlap_penalty,
    resolve_rigid_clash,
)
from metalsurfer.placement.geometry import compute_surface_site_frame

from ..conftest import make_slab, make_water


def test_overlap_penalty_zero_when_clear():
    moving = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    fixed = np.array([[5.0, 0.0, 0.0]], dtype=float)
    r_m = np.array([0.7, 0.3], dtype=float)
    r_f = np.array([0.7], dtype=float)
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]
    f = overlap_penalty(moving, r_m, fixed, r_f, cell=cell, pbc=pbc, min_separation=1.5)
    assert f == 0.0


def test_overlap_penalty_positive_when_overlapping():
    moving = np.array([[0.0, 0.0, 0.0]], dtype=float)
    fixed = np.array([[0.5, 0.0, 0.0]], dtype=float)
    r_m = np.array([0.7], dtype=float)
    r_f = np.array([0.7], dtype=float)
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]
    f = overlap_penalty(moving, r_m, fixed, r_f, cell=cell, pbc=pbc)
    assert f > 0.0


def test_overlap_penalty_c1_smooth_across_threshold():
    """Finite-difference of f wrt pair distance is continuous at d = r_i+r_j."""
    r_sum = 1.4
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]
    r_m = np.array([0.7], dtype=float)
    r_f = np.array([0.7], dtype=float)
    eps = 1e-5

    def f_at(d: float) -> float:
        return overlap_penalty(
            np.array([[0.0, 0.0, 0.0]]),
            r_m,
            np.array([[d, 0.0, 0.0]]),
            r_f,
            cell=cell,
            pbc=pbc,
        )

    # Just inside and just outside the threshold: left/right derivatives of
    # f = [max(0, r^2 - d^2)]^2 should both approach 0 at d = r_sum.
    d0 = r_sum
    left = (f_at(d0) - f_at(d0 - eps)) / eps
    right = (f_at(d0 + eps) - f_at(d0)) / eps
    assert abs(left) < 1e-3
    assert abs(right) < 1e-3
    assert f_at(d0) == 0.0
    assert f_at(d0 - 0.1) > 0.0


def test_resolve_rigid_clash_separates_near_overlap():
    config = AdsorptionConfig(
        material_type="slab",
        seed=0,
        placement_x_range=(-1.5, 1.5),
        placement_y_range=(-1.5, 1.5),
        min_adsorbate_separation=1.5,
        placement_clash_descent=True,
    )
    water = make_water()
    water_pos = water.get_positions().copy()
    com = np.mean(water_pos, axis=0)
    water_pos -= com
    # Fixed water shifted +1.1 Å in x (O–O ~1.1 Å with covalent floor → clash).
    fixed_atoms = water.copy()
    fixed_atoms.set_positions(water_pos + np.array([1.1, 0.0, 0.0]))
    moving = water.copy()
    moving.set_positions(water_pos)

    fixed_pos = fixed_atoms.get_positions()
    fixed_radii = atom_radii_for_symbols(
        list(fixed_atoms.get_chemical_symbols()),
        min_separation=config.min_adsorbate_separation,
    )
    frame = compute_surface_site_frame(np.array([0.0, 0.0, 1.0]))
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]

    new_pos, az, ok = resolve_rigid_clash(
        moving,
        fixed_pos,
        fixed_radii,
        origin=np.zeros(3),
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        include_substrate_min_sep=True,
    )
    assert ok
    assert az is not None
    moving.set_positions(new_pos)
    min_d = geom.calculate_min_distance(
        new_pos, fixed_pos, cell, use_pbc=False, pbc=pbc
    )
    assert min_d >= config.min_adsorbate_separation - 1e-3


def test_resolve_rigid_clash_fails_when_stacked():
    config = AdsorptionConfig(
        material_type="slab",
        seed=0,
        placement_x_range=(-0.5, 0.5),
        placement_y_range=(-0.5, 0.5),
        min_adsorbate_separation=1.5,
    )
    a = Atoms("O", positions=[[0.0, 0.0, 0.0]])
    fixed = np.array([[0.1, 0.0, 0.0]], dtype=float)
    fixed_radii = np.array([0.66], dtype=float)
    frame = compute_surface_site_frame(np.array([0.0, 0.0, 1.0]))
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]
    _, _, ok = resolve_rigid_clash(
        a,
        fixed,
        fixed_radii,
        origin=np.zeros(3),
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        include_substrate_min_sep=True,
    )
    assert not ok


def test_resolve_rigid_clash_deterministic():
    config = AdsorptionConfig(
        material_type="slab",
        seed=0,
        placement_x_range=(-1.5, 1.5),
        placement_y_range=(-1.5, 1.5),
        min_adsorbate_separation=1.5,
    )
    water = make_water()
    pos = water.get_positions().copy()
    pos -= np.mean(pos, axis=0)
    fixed = pos + np.array([1.2, 0.0, 0.0])
    moving = water.copy()
    moving.set_positions(pos)
    fixed_radii = atom_radii_for_symbols(
        list(water.get_chemical_symbols()),
        min_separation=config.min_adsorbate_separation,
    )
    frame = compute_surface_site_frame(np.array([0.0, 0.0, 1.0]))
    cell = np.eye(3) * 20.0
    pbc = [False, False, False]
    kwargs = dict(
        fixed_pos=fixed,
        fixed_radii=fixed_radii,
        origin=np.zeros(3),
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        include_substrate_min_sep=True,
    )
    p1, a1, ok1 = resolve_rigid_clash(moving, **kwargs)
    p2, a2, ok2 = resolve_rigid_clash(moving, **kwargs)
    assert ok1 == ok2
    assert a1 == a2
    np.testing.assert_allclose(p1, p2, atol=0.0)


def test_mol_slab_pairwise_mic_returns_vectors():
    slab = make_slab()
    mol = make_water()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = [True, True, False]
    vecs, dists = geom._mol_slab_pairwise_mic(
        mol.get_positions(), slab.get_positions(), cell, pbc
    )
    assert vecs.shape == (len(mol), len(slab), 3)
    assert dists.shape == (len(mol), len(slab))
    np.testing.assert_allclose(np.linalg.norm(vecs, axis=2), dists, atol=1e-9)

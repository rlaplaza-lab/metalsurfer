"""Tests for placement diversity, correctness, and success rate."""

import random

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement import (
    _classify_molecule_shape,
    _cluster_equivalent_sites,
    _compute_site_z_base,
    _get_site_surface_radii,
    _is_flat_aromatic_with_en,
    _random_rotation_matrix,
    _sample_xy_in_cell,
    calculate_min_distance,
    check_initial_placement_distance,
    classify_adsorbate_orientation,
    generate_conformer_placement,
    get_adsorption_sites,
    get_envelope_placement_sites,
    get_hollow_sites_for_adatoms,
    is_surface_planar,
)
from tests.optional_deps import cuda_available, has_mlip_stack

from .conftest import make_ethanol, make_slab, make_water

# ---------------------------------------------------------------------------
# shape classification
# ---------------------------------------------------------------------------


def test_classify_molecule_shape_linear():
    """H2 and CO2 are linear (I1/I3 << 1)."""
    h2 = Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]])
    shape, _, _ = _classify_molecule_shape(h2.get_positions())
    assert shape == "linear"

    co2 = Atoms(
        "CO2",
        positions=[[0, 0, 0], [-1.16, 0, 0], [1.16, 0, 0]],
    )
    shape, _, _ = _classify_molecule_shape(co2.get_positions())
    assert shape == "linear"


def test_classify_molecule_shape_flat():
    """Benzene-like planar ring is flat."""
    # Hexagon in xy plane
    r = 1.4
    positions = [
        [r * np.cos(i * np.pi / 3), r * np.sin(i * np.pi / 3), 0.0] for i in range(6)
    ]
    benzene = Atoms("C6H6", positions=positions + [[0, 0, 0]] * 6)  # simplified
    shape, _, _ = _classify_molecule_shape(benzene.get_positions()[:6])
    assert shape == "flat"


def test_classify_molecule_shape_round():
    """Methane (tetrahedral) is round."""
    a = 1.09
    methane = Atoms(
        "CH4",
        positions=[
            [0, 0, 0],
            [a, a, a],
            [-a, -a, a],
            [-a, a, -a],
            [a, -a, -a],
        ],
    )
    shape, _, _ = _classify_molecule_shape(methane.get_positions())
    assert shape == "round"


def test_is_flat_aromatic_with_en():
    """Vanillin has aromatic ring and O atoms."""
    assert _is_flat_aromatic_with_en("c1(C=O)cc(OC)c(O)cc1") is True
    assert _is_flat_aromatic_with_en("c1ccccc1") is False  # no EN
    assert _is_flat_aromatic_with_en("CCO") is False  # not aromatic


def test_classify_adsorbate_orientation_parallel_vs_en_down():
    """Six-membered ring parallel to surface = 'parallel'; tilted = 'EN-down'.

    Uses inertia plane normal (eigenvecs[:, 2] for flat molecules). A ring in
    the xy plane has plane normal along z → parallel. A ring tilted 60° has
    plane normal tilted → EN-down.
    """
    r = 1.4
    # Benzene-like hexagon in xy plane (ring parallel to surface)
    hex_xy = np.array(
        [[r * np.cos(i * np.pi / 3), r * np.sin(i * np.pi / 3), 0.0] for i in range(6)]
    )
    parallel_atoms = Atoms("C6", positions=hex_xy)
    # Combined slab (2 Ni) + adsorbate for classify_adsorbate_orientation
    slab = Atoms("Ni2", positions=[[0, 0, 0], [1, 1, 0]])
    combined_parallel = slab + parallel_atoms
    assert classify_adsorbate_orientation(combined_parallel, slab_size=2) == "parallel"

    # Tilt ring 60° around x (EN-down: ring tilted, would have O down in vanillin)
    c, s = np.cos(np.radians(60)), np.sin(np.radians(60))
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    hex_tilted = (Rx @ hex_xy.T).T
    tilted_atoms = Atoms("C6", positions=hex_tilted)
    combined_tilted = slab + tilted_atoms
    assert classify_adsorbate_orientation(combined_tilted, slab_size=2) == "EN-down"


# ---------------------------------------------------------------------------
# rotation matrix tests
# ---------------------------------------------------------------------------


def test_rotation_matrix_is_orthogonal():
    """Every random rotation must be a proper rotation (det=+1, R^T R=I)."""
    rng = random.Random(123)
    for _ in range(50):
        rot = _random_rotation_matrix(rng)
        assert rot.shape == (3, 3)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_rotation_matrices_are_diverse():
    """Different seeds should produce different rotation matrices."""
    mats = []
    for seed in range(20):
        rng = random.Random(seed)
        mats.append(_random_rotation_matrix(rng))
    # pairwise Frobenius distances should not all be zero
    dists = []
    for i in range(len(mats)):
        for j in range(i + 1, len(mats)):
            dists.append(np.linalg.norm(mats[i] - mats[j]))
    assert max(dists) > 0.5, "rotation matrices are too similar"


# ---------------------------------------------------------------------------
# cell-aware sampling tests
# ---------------------------------------------------------------------------


def test_sample_xy_in_cell_stays_inside():
    """Sampled points should lie inside the parallelogram spanned by a,b."""
    cell = np.array([[10.0, 0.0, 0.0], [3.0, 9.0, 0.0], [0.0, 0.0, 20.0]])
    rng = random.Random(0)
    for _ in range(200):
        x, y = _sample_xy_in_cell(cell, rng)
        pos = np.array([x, y])
        ab = cell[:2, :2]
        frac = np.linalg.solve(ab.T, pos)
        assert -0.01 <= frac[0] <= 1.01
        assert -0.01 <= frac[1] <= 1.01


def test_sample_xy_covers_cell():
    """Points should spread across the full cell, not cluster in one corner."""
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    rng = random.Random(7)
    xs, ys = [], []
    for _ in range(500):
        x, y = _sample_xy_in_cell(cell, rng)
        xs.append(x)
        ys.append(y)
    # check that the range spans at least 70% of the cell
    assert max(xs) - min(xs) > 7.0
    assert max(ys) - min(ys) > 7.0


# ---------------------------------------------------------------------------
# min distance
# ---------------------------------------------------------------------------


def test_calculate_min_distance_non_pbc():
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[3.0, 4.0, 0.0]])
    assert np.isclose(calculate_min_distance(p1, p2, use_pbc=False), 5.0)


def test_calculate_min_distance_with_pbc():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.5, 9.5, 5.0]])
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True)
    assert d < 2.0, f"PBC distance should wrap around, got {d}"


# ---------------------------------------------------------------------------
# initial placement distance
# ---------------------------------------------------------------------------


def test_initial_placement_valid():
    slab = make_slab()
    mol = make_water()
    surface_z = max(slab.get_positions()[:, 2])
    pos = mol.get_positions().copy()
    pos[:, 2] += surface_z + 2.5
    pos[:, 0] += 3.0
    pos[:, 1] += 3.0
    mol.set_positions(pos)
    mol.set_cell(slab.get_cell())
    mol.set_pbc(slab.get_pbc())
    ok, dist = check_initial_placement_distance(mol, slab)
    assert ok, f"Should be a valid placement at 2.5 A above surface, dist={dist}"


def test_initial_placement_too_close():
    slab = make_slab()
    mol = make_water()
    surface_z = max(slab.get_positions()[:, 2])
    pos = mol.get_positions().copy()
    pos[:, 2] += surface_z + 0.3
    mol.set_positions(pos)
    mol.set_cell(slab.get_cell())
    mol.set_pbc(slab.get_pbc())
    ok, dist = check_initial_placement_distance(mol, slab)
    assert not ok, f"Placement 0.3 A above surface should be rejected, dist={dist}"


# ---------------------------------------------------------------------------
# full placement generation: diversity + success rate
# ---------------------------------------------------------------------------


def test_placements_are_diverse():
    """Different placement_ids should produce spatially distinct placements."""
    slab = make_slab()
    water = make_water()
    conformers = [water]
    energies = [0.0]

    config = AdsorptionConfig(
        num_placements=30,
        placement_z_range=(2.0, 3.0),
    )

    placed = []
    for pid in range(30):
        result = generate_conformer_placement(
            conformers, energies, slab, pid, config=config
        )
        if result is not None:
            placed.append(result.get_positions().copy())

    assert len(placed) >= 20, (
        f"Expected at least 20/30 successful placements, got {len(placed)}"
    )

    # check that xy centres span a significant portion of the cell
    centres = np.array([np.mean(p, axis=0) for p in placed])
    x_range = centres[:, 0].max() - centres[:, 0].min()
    y_range = centres[:, 1].max() - centres[:, 1].min()
    cell_x = slab.get_cell()[0, 0]
    cell_y = slab.get_cell()[1, 1]

    assert x_range > 0.3 * cell_x, (
        f"x-spread {x_range:.1f} should cover >30% of cell ({cell_x:.1f})"
    )
    assert y_range > 0.3 * cell_y, (
        f"y-spread {y_range:.1f} should cover >30% of cell ({cell_y:.1f})"
    )


def test_placements_have_rotational_diversity():
    """Different placements should show diverse molecular orientations."""
    slab = make_slab()
    ethanol = make_ethanol()
    conformers = [ethanol]
    energies = [0.0]

    config = AdsorptionConfig(
        num_placements=20,
        placement_z_range=(2.0, 3.5),
    )

    orientation_vectors = []
    for pid in range(20):
        result = generate_conformer_placement(
            conformers, energies, slab, pid, config=config
        )
        if result is not None:
            pos = result.get_positions()
            centred = pos - np.mean(pos, axis=0)
            # use the direction to the farthest atom as an orientation proxy
            dists = np.linalg.norm(centred, axis=1)
            farthest = centred[np.argmax(dists)]
            farthest /= np.linalg.norm(farthest)
            orientation_vectors.append(farthest)

    assert len(orientation_vectors) >= 10

    # pairwise angles between orientation vectors
    angles = []
    for i in range(len(orientation_vectors)):
        for j in range(i + 1, len(orientation_vectors)):
            dot = np.clip(np.dot(orientation_vectors[i], orientation_vectors[j]), -1, 1)
            angles.append(np.degrees(np.arccos(abs(dot))))

    assert max(angles) > 30.0, (
        f"Max angular spread is {max(angles):.1f}°, need >30° for diversity"
    )


def test_success_rate_water():
    """Water (tiny molecule) should have a very high success rate."""
    slab = make_slab()
    water = make_water()
    conformers = [water]
    energies = [0.0]
    config = AdsorptionConfig(placement_z_range=(2.0, 3.0))

    n_ok = sum(
        1
        for pid in range(50)
        if generate_conformer_placement(conformers, energies, slab, pid, config=config)
        is not None
    )
    assert n_ok >= 45, f"Water placement success rate too low: {n_ok}/50"


def test_success_rate_larger_molecule():
    """Ethanol-like molecule should still place >60% of the time."""
    slab = make_slab()
    mol = make_ethanol()
    conformers = [mol]
    energies = [0.0]
    config = AdsorptionConfig(placement_z_range=(2.0, 3.5))

    n_ok = sum(
        1
        for pid in range(50)
        if generate_conformer_placement(conformers, energies, slab, pid, config=config)
        is not None
    )
    assert n_ok >= 30, f"Ethanol placement success rate too low: {n_ok}/50"


def test_no_conformers_returns_none():
    slab = make_slab()
    result = generate_conformer_placement([], [], slab, 0)
    assert result is None


def test_placement_above_surface():
    """All placed atoms must be above the slab's top layer."""
    slab = make_slab()
    water = make_water()
    surface_z = max(slab.get_positions()[:, 2])
    config = AdsorptionConfig(placement_z_range=(2.0, 3.0))

    for pid in range(20):
        result = generate_conformer_placement([water], [0.0], slab, pid, config=config)
        if result is not None:
            min_z = result.get_positions()[:, 2].min()
            assert min_z > surface_z - 0.5, (
                f"Placement {pid}: atom at z={min_z:.2f} is below "
                f"surface z={surface_z:.2f}"
            )


# ---------------------------------------------------------------------------
# MIC / skewed-cell / boundary-crossing tests
# ---------------------------------------------------------------------------


def test_calculate_min_distance_skewed_cell():
    """MIC distance must be correct for a non-orthogonal cell.

    For cell [[10, 2, 0], [0, 10, 0], [0, 0, 20]] with PBC in x,y,
    the MIC vector for (0.5, 0.5) -> (9.8, 0.5) wraps to ~(0.7, 2.0, 0)
    with length ~2.12 A.  The naive Cartesian distance is 9.3 A, so the
    MIC result must be drastically shorter.
    """
    cell = np.array([[10.0, 2.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.8, 0.5, 5.0]])
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True)
    naive = float(np.linalg.norm(p1 - p2))
    assert d < naive * 0.5, (
        f"Skewed-cell MIC distance ({d:.2f}) should be much shorter "
        f"than naive Cartesian ({naive:.2f})"
    )


def test_calculate_min_distance_molecule_crosses_boundary():
    """A molecule straddling the periodic boundary should give short distances."""
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    mol_pos = np.array([[0.3, 5.0, 5.0], [9.7, 5.0, 5.0]])
    slab_pos = np.array([[5.0, 5.0, 0.0]])
    d = calculate_min_distance(mol_pos, slab_pos, cell=cell, use_pbc=True)
    direct_d = np.linalg.norm(mol_pos[0] - slab_pos[0])
    assert d < direct_d, "MIC distance should be shorter than naive Cartesian distance"


def test_check_initial_placement_skewed_cell():
    """check_initial_placement_distance must work with a skewed cell."""
    cell = np.array([[10.0, 3.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    slab = Atoms(
        "Ru4",
        positions=[[0, 0, 0], [2.5, 0, 0], [0, 2.5, 0], [2.5, 2.5, 0]],
        cell=cell,
        pbc=[True, True, True],
    )
    mol = make_water()
    pos = mol.get_positions().copy()
    pos[:, 2] += 2.5
    pos[:, 0] += 1.0
    mol.set_positions(pos)
    mol.set_cell(cell)
    mol.set_pbc([True, True, True])
    ok, dist = check_initial_placement_distance(mol, slab)
    assert ok, f"Valid placement on skewed cell failed, dist={dist}"


# ---------------------------------------------------------------------------
# site detection and equivalence
# ---------------------------------------------------------------------------


def test_get_adsorption_sites_returns_sites():
    """get_adsorption_sites returns atop, bridge, hollow for periodic slab."""
    slab = make_slab()
    sites = get_adsorption_sites(slab)
    assert sites is not None
    assert len(sites) >= 3
    types = {s["site_type"] for s in sites}
    assert "atop" in types
    assert "bridge" in types
    assert "hollow" in types
    for s in sites:
        assert "xy" in s
        assert "z" in s
        assert "slab_indices" in s
        xy = np.asarray(s["xy"])
        assert xy.shape == (2,)


def test_get_site_surface_radii():
    """_get_site_surface_radii returns mean covalent radius of site atoms."""
    slab = make_slab()
    sites = get_adsorption_sites(slab)
    assert sites is not None
    r = _get_site_surface_radii(slab, sites[0])
    assert r is not None
    assert 1.0 < r < 2.0  # Ru covalent radius ~1.25
    r_top = _get_site_surface_radii(slab, None)
    assert r_top is not None


def test_compute_site_z_base_scaled():
    """placement_z_scale_by_covalent_radius adjusts z range by surface atoms."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=True,
    )
    site = get_adsorption_sites(slab)[0]
    z_lo, z_hi = _compute_site_z_base(config, slab, site, ["O", "H", "H"])
    assert z_lo >= 1.5
    assert z_hi <= 4.0
    assert z_lo < z_hi


def test_compute_site_z_base_unscaled():
    """placement_z_scale_by_covalent_radius=False uses fixed range."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=False,
    )
    z_lo, z_hi = _compute_site_z_base(config, slab, None, ["O", "H", "H"])
    assert z_lo == 2.0
    assert z_hi == 3.0


def test_get_adsorption_sites_non_periodic_returns_none():
    """get_adsorption_sites returns None for non-periodic or invalid slab."""
    slab = Atoms("Ru4", positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    slab.set_cell([5, 5, 10])
    slab.set_pbc([False, False, False])
    assert get_adsorption_sites(slab) is None


def test_cluster_equivalent_sites_reduces_count():
    """_cluster_equivalent_sites groups symmetry-equivalent sites."""
    slab = make_slab(nx=2, ny=2)
    raw = get_adsorption_sites(slab)
    assert raw is not None
    cell = slab.get_cell()
    unique = _cluster_equivalent_sites(raw, cell, tolerance=0.05)
    assert len(unique) <= len(raw)
    assert len(unique) >= 1


def test_get_hollow_sites_for_adatoms():
    """get_hollow_sites_for_adatoms returns xy positions for adatom placement."""
    slab = make_slab()
    sites = get_hollow_sites_for_adatoms(slab)
    assert len(sites) >= 1
    for xy in sites:
        assert np.asarray(xy).shape == (2,)


# ---------------------------------------------------------------------------
# planarity and envelope placement
# ---------------------------------------------------------------------------


def test_is_surface_planar_clean_slab():
    """Planar Ru slab is classified as planar."""
    slab = make_slab()
    assert is_surface_planar(slab) is True


def test_is_surface_planar_with_adatoms():
    """Slab with H adatoms above surface is non-planar."""
    slab = make_slab()
    z_max = float(np.max(slab.get_positions()[:, 2]))
    # Add H adatoms above the top layer
    h_positions = [[2.0, 2.0, z_max + 1.5], [5.0, 5.0, z_max + 1.2]]
    slab_with_h = slab + Atoms("H2", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    assert is_surface_planar(slab_with_h) is False


def test_is_surface_planar_too_few_atoms():
    """Slab with < 3 top atoms returns False."""
    slab = Atoms("Ru2", positions=[[0, 0, 0], [1, 1, 0]])
    slab.set_cell([5, 5, 10])
    slab.set_pbc([True, True, True])
    assert is_surface_planar(slab) is False


def test_is_surface_planar_non_periodic():
    """Non-periodic slab returns False."""
    slab = make_slab()
    slab.set_pbc([False, False, False])
    assert is_surface_planar(slab) is False


def test_get_envelope_placement_sites_returns_sites():
    """get_envelope_placement_sites returns envelope sites on slab with adatoms."""
    slab = make_slab()
    z_max = float(np.max(slab.get_positions()[:, 2]))
    # Use tolerance so top metal + H are in top layer (need >= 3 atoms)
    h_positions = [
        [2.0, 2.0, z_max + 1.5],
        [5.0, 2.0, z_max + 1.2],
        [2.0, 5.0, z_max + 1.3],
    ]
    slab_with_h = slab + Atoms("H3", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    sites = get_envelope_placement_sites(slab_with_h, top_layer_tolerance=2.0)
    assert sites is not None
    assert len(sites) >= 1
    for s in sites:
        assert s["site_type"] == "envelope"
        assert "xy" in s
        assert "z" in s
        xy = np.asarray(s["xy"])
        assert xy.shape == (2,)


def test_get_envelope_placement_sites_planar_fallback():
    """Planar slab can still use envelope (sanity check)."""
    slab = make_slab()
    sites = get_envelope_placement_sites(slab)
    assert sites is not None
    assert len(sites) >= 1


def test_envelope_sites_use_slab_indices_for_covalent_radius():
    """Envelope sites use slab_indices for site-local covalent radius scaling."""
    slab = make_slab()
    sites = get_envelope_placement_sites(slab)
    assert sites is not None
    assert len(sites) >= 1
    for s in sites:
        assert "slab_indices" in s
        assert len(s["slab_indices"]) >= 1
        r = _get_site_surface_radii(slab, s)
        assert r is not None
        assert 0.3 < r < 2.5  # H ~0.31, Ru ~1.25, Ni ~1.24
    # With H adatoms, mean radius can differ (facet may include H)
    z_max = float(np.max(slab.get_positions()[:, 2]))
    h_positions = [[2.0, 2.0, z_max + 1.5], [5.0, 2.0, z_max + 1.2]]
    slab_with_h = slab + Atoms("H2", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    sites_h = get_envelope_placement_sites(slab_with_h, top_layer_tolerance=2.0)
    assert sites_h is not None
    r_with_h = _get_site_surface_radii(slab_with_h, sites_h[0])
    assert r_with_h is not None
    assert 0.3 < r_with_h < 2.5


def test_placement_mode_envelope():
    """placement_mode=envelope produces placements from envelope sites."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_mode="envelope",
        placement_z_range=(2.0, 3.0),
    )
    results = [
        generate_conformer_placement([make_water()], [0.0], slab, pid, config=config)
        for pid in range(10)
    ]
    assert sum(1 for r in results if r is not None) >= 5


def test_placement_auto_uses_envelope_for_non_planar():
    """placement_mode=auto uses envelope when slab has adatoms (non-planar)."""
    slab = make_slab()
    z_max = float(np.max(slab.get_positions()[:, 2]))
    h_positions = [
        [2.0, 2.0, z_max + 1.5],
        [5.0, 2.0, z_max + 1.2],
        [2.0, 5.0, z_max + 1.3],
    ]
    slab_with_h = slab + Atoms("H3", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    config = AdsorptionConfig(
        placement_mode="auto",
        placement_z_range=(2.0, 3.0),
        top_layer_tolerance=2.0,  # Include top metal + H in top layer
    )
    results = [
        generate_conformer_placement(
            [make_water()], [0.0], slab_with_h, pid, config=config
        )
        for pid in range(10)
    ]
    assert sum(1 for r in results if r is not None) >= 5


def test_envelope_placement_sites_reproducible():
    """get_envelope_placement_sites returns identical site order across calls."""
    slab = make_slab()
    z_max = float(np.max(slab.get_positions()[:, 2]))
    h_positions = [
        [2.0, 2.0, z_max + 1.5],
        [5.0, 2.0, z_max + 1.2],
        [2.0, 5.0, z_max + 1.3],
    ]
    slab_with_h = slab + Atoms("H3", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    sites1 = get_envelope_placement_sites(slab_with_h, top_layer_tolerance=2.0)
    sites2 = get_envelope_placement_sites(slab_with_h, top_layer_tolerance=2.0)
    assert sites1 is not None and sites2 is not None
    assert len(sites1) == len(sites2)
    for s1, s2 in zip(sites1, sites2, strict=True):
        np.testing.assert_array_almost_equal(s1["xy"], s2["xy"])
        assert s1["z"] == s2["z"]
        assert s1["slab_indices"] == s2["slab_indices"]


def test_generate_placement_from_spec_deterministic():
    """generate_placement_from_spec yields identical placement for same spec and slab."""
    from metalsurfer.placement import (
        enumerate_placement_specs,
        generate_placement_from_spec,
    )

    slab = make_slab()
    config = AdsorptionConfig(
        placement_mode="envelope",
        placement_z_range=(2.0, 3.0),
    )
    conformers = [make_water()]
    specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=5)
    site_specs = [s for s in specs if s.site_index >= 0]
    if not site_specs:
        pytest.skip("No site-based specs for envelope mode")
    spec = site_specs[0]
    result1 = generate_placement_from_spec(spec, conformers, slab, config, smiles="O")
    result2 = generate_placement_from_spec(spec, conformers, slab, config, smiles="O")
    assert result1 is not None and result2 is not None
    ads1, desc1 = result1
    ads2, desc2 = result2
    np.testing.assert_array_almost_equal(ads1.get_positions(), ads2.get_positions())
    assert desc1.x == desc2.x and desc1.y == desc2.y and desc1.z == desc2.z
    assert desc1.site_index == desc2.site_index


# ---------------------------------------------------------------------------
# placement mode and distance bounds
# ---------------------------------------------------------------------------


def test_placement_mode_random():
    """placement_mode=random uses random xy sampling."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_mode="random",
        placement_z_range=(2.0, 3.0),
    )
    results = [
        generate_conformer_placement([make_water()], [0.0], slab, pid, config=config)
        for pid in range(10)
    ]
    assert sum(1 for r in results if r is not None) >= 8


def test_placement_mode_sites():
    """placement_mode=sites uses symmetry-unique sites."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
    )
    results = [
        generate_conformer_placement([make_water()], [0.0], slab, pid, config=config)
        for pid in range(10)
    ]
    assert sum(1 for r in results if r is not None) >= 8


def test_check_initial_placement_max_distance():
    """max_initial_distance rejects placements too far from surface."""
    slab = make_slab()
    mol = make_water()
    surface_z = max(slab.get_positions()[:, 2])
    pos = mol.get_positions().copy()
    pos[:, 2] += surface_z + 5.0
    pos[:, 0] += 3.0
    pos[:, 1] += 3.0
    mol.set_positions(pos)
    mol.set_cell(slab.get_cell())
    mol.set_pbc(slab.get_pbc())
    ok, dist = check_initial_placement_distance(mol, slab, max_initial_distance=3.5)
    assert not ok, (
        f"Placement 5 A above surface should be rejected when max=3.5, dist={dist}"
    )


def test_check_initial_placement_min_contact_ratio():
    """min_contact_ratio tunes covalent-radius-based lower bound.

    Looser ratio (0.5) accepts placements that stricter ratio (0.8) may reject.
    A valid placement at 2.0 A above surface should pass with min_contact_ratio=0.5.
    """
    slab = make_slab()
    mol = make_water()
    surface_z = max(slab.get_positions()[:, 2])
    pos = mol.get_positions().copy()
    pos[:, 2] += surface_z + 2.0
    pos[:, 0] += 3.0
    pos[:, 1] += 3.0
    mol.set_positions(pos)
    mol.set_cell(slab.get_cell())
    mol.set_pbc(slab.get_pbc())
    ok_05, dist_05 = check_initial_placement_distance(mol, slab, min_contact_ratio=0.5)
    assert ok_05, (
        f"Placement at 2.0 A above surface should pass with min_contact_ratio=0.5, "
        f"got dist={dist_05}"
    )
    # Stricter ratio may reject; at least verify we get a consistent distance
    ok_08, dist_08 = check_initial_placement_distance(mol, slab, min_contact_ratio=0.8)
    assert dist_05 is not None and dist_08 is not None


def test_dissociative_placement_when_skip_topology_check():
    """When skip_topology_check=True, H2 gets dissociative placements at different sites."""
    slab = make_slab()
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])  # H-H ~0.74 A
    config = AdsorptionConfig(
        skip_topology_check=True,
        placement_z_range=(2.0, 3.0),
        placement_mode="auto",
    )
    placed = []
    for pid in range(20):
        result = generate_conformer_placement([h2], [0.0], slab, pid, config=config)
        if result is not None:
            placed.append(result)

    assert len(placed) >= 10, f"Expected >= 10 H2 placements, got {len(placed)}"
    # Dissociative placements have H atoms at different xy (different hollow sites)
    xy_seps = [
        np.linalg.norm(p.get_positions()[0, :2] - p.get_positions()[1, :2])
        for p in placed
    ]
    # At least some should be dissociative (atoms at different sites, sep > 1 A)
    n_dissociative = sum(1 for s in xy_seps if s > 1.0)
    assert n_dissociative >= 1, (
        f"Expected some dissociative placements (xy sep > 1 A), got max sep {max(xy_seps):.2f}"
    )


def test_placement_with_smiles_flat_aromatic():
    """Placement with smiles enables flat-aromatic strategy (parallel + EN-down)."""
    from metalsurfer.conformers import create_conformers_from_smiles

    slab = make_slab()
    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1", config=AdsorptionConfig(num_conformers=3)
    )
    if result is None:
        pytest.skip("RDKit required for vanillin conformers")
    conformers, energies = result
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
        num_placements=20,
    )
    placed = []
    for pid in range(20):
        result = generate_conformer_placement(
            conformers,
            energies,
            slab,
            pid,
            config=config,
            smiles="c1(C=O)cc(OC)c(O)cc1",
        )
        if result is not None:
            placed.append(result)
    assert len(placed) >= 10, (
        f"Expected >= 10 vanillin placements with smiles, got {len(placed)}"
    )


def test_flat_aromatic_placement_random_mode():
    """Flat aromatic molecules get shape-specific (parallel + EN-down) in random mode."""
    from metalsurfer.conformers import create_conformers_from_smiles

    slab = make_slab()
    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1", config=AdsorptionConfig(num_conformers=3)
    )
    if result is None:
        pytest.skip("RDKit required for vanillin conformers")
    conformers, energies = result
    config = AdsorptionConfig(
        placement_mode="random",
        placement_z_range=(2.0, 3.0),
        num_placements=20,
    )
    placed = []
    for pid in range(20):
        result = generate_conformer_placement(
            conformers,
            energies,
            slab,
            pid,
            config=config,
            smiles="c1(C=O)cc(OC)c(O)cc1",
        )
        if result is not None:
            placed.append(result)
    assert len(placed) >= 10, (
        f"Expected >= 10 vanillin placements in random mode with shape-specific logic, "
        f"got {len(placed)}"
    )


def test_flat_aromatic_explores_both_parallel_and_en_down():
    """Default strategy explores both horizontal (parallel) and EN-down orientations."""
    from metalsurfer.conformers import create_conformers_from_smiles

    slab = make_slab()
    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1", config=AdsorptionConfig(num_conformers=3)
    )
    if result is None:
        pytest.skip("RDKit required for vanillin conformers")
    conformers, energies = result
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
        flat_aromatic_parallel_fraction=0.5,
        num_placements=24,
    )
    placed = []
    for pid in range(24):
        p = generate_conformer_placement(
            conformers,
            energies,
            slab,
            pid,
            config=config,
            smiles="c1(C=O)cc(OC)c(O)cc1",
        )
        if p is not None:
            placed.append(p)
    assert len(placed) >= 12, f"Expected >= 12 placements, got {len(placed)}"
    orientations = [classify_adsorbate_orientation(p, slab_size=0) for p in placed]
    n_parallel = sum(1 for o in orientations if o == "parallel")
    n_en_down = sum(1 for o in orientations if o == "EN-down")
    assert n_parallel >= 1, (
        f"Default strategy should explore horizontal (parallel) placements, "
        f"got {n_parallel} parallel, {n_en_down} EN-down"
    )
    assert n_en_down >= 1, (
        f"Default strategy should explore EN-down placements, "
        f"got {n_parallel} parallel, {n_en_down} EN-down"
    )


def test_placement_z_scale_disabled_backward_compatible():
    """placement_z_scale_by_covalent_radius=False uses fixed z range."""
    slab = make_slab()
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=False,
    )
    results = [
        generate_conformer_placement([make_water()], [0.0], slab, pid, config=config)
        for pid in range(10)
    ]
    assert sum(1 for r in results if r is not None) >= 8


def test_no_dissociative_placement_without_skip_topology_check():
    """H2 with skip_topology_check=False uses normal placement only (no pre-dissociated)."""
    slab = make_slab()
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    config = AdsorptionConfig(
        skip_topology_check=False,  # Normal mode: no dissociative placement
        placement_z_range=(2.0, 3.0),
        placement_mode="auto",
    )
    placed = []
    for pid in range(20):
        result = generate_conformer_placement([h2], [0.0], slab, pid, config=config)
        if result is not None:
            placed.append(result)

    assert len(placed) >= 10
    # All placements should be molecular (H2 intact): xy separation ~0.74 A
    xy_seps = [
        np.linalg.norm(p.get_positions()[0, :2] - p.get_positions()[1, :2])
        for p in placed
    ]
    # Molecular H2 has H-H ~0.74 A; no dissociative (sep > 1.5 A)
    n_dissociative = sum(1 for s in xy_seps if s > 1.5)
    assert n_dissociative == 0, (
        f"With skip_topology_check=False, expected no dissociative placements, "
        f"got {n_dissociative} with sep > 1.5 A"
    )


def test_enumerate_placement_specs_returns_specs():
    """enumerate_placement_specs returns list of PlacementSpec."""
    from metalsurfer.placement import enumerate_placement_specs

    slab = make_slab()
    conformers = [make_water()]
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
        num_placements=10,
    )
    specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=10)
    assert len(specs) <= 10
    assert len(specs) >= 1
    for s in specs:
        assert s.conformer_index >= 0
        assert s.orientation_type in ("parallel", "EN-down", "vertical", "round")
        assert s.placement_index >= 0


def test_generate_placement_from_spec_returns_descriptor():
    """generate_placement_from_spec returns (Atoms, PlacementDescriptor)."""
    from metalsurfer.placement import (
        enumerate_placement_specs,
        generate_placement_from_spec,
    )

    slab = make_slab()
    conformers = [make_water()]
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
    )
    specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=5)
    placed_count = 0
    for spec in specs:
        result = generate_placement_from_spec(
            spec, conformers, slab, config, smiles="O"
        )
        if result is not None:
            adsorbate, descriptor = result
            assert descriptor.placement_index == spec.placement_index
            assert descriptor.x is not None
            assert descriptor.y is not None
            assert descriptor.z is not None
            assert descriptor.shape in ("linear", "flat", "round")
            placed_count += 1
    assert placed_count >= 1


def test_generate_placement_from_descriptor_reproduces():
    """generate_placement_from_descriptor reproduces placement."""
    from metalsurfer.placement import (
        enumerate_placement_specs,
        generate_placement_from_descriptor,
        generate_placement_from_spec,
    )

    slab = make_slab()
    conformers = [make_water()]
    config = AdsorptionConfig(
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
    )
    specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=3)
    for spec in specs:
        result = generate_placement_from_spec(
            spec, conformers, slab, config, smiles="O"
        )
        if result is not None:
            adsorbate, descriptor = result
            reproduced = generate_placement_from_descriptor(
                descriptor, conformers, slab, config
            )
            assert reproduced is not None
            np.testing.assert_allclose(
                adsorbate.get_positions(),
                reproduced.get_positions(),
                atol=1e-6,
            )
            break


def test_generate_placement_from_descriptor_envelope_reproduces():
    """generate_placement_from_descriptor reproduces envelope placements (site-local z)."""
    from metalsurfer.placement import (
        enumerate_placement_specs,
        generate_placement_from_descriptor,
        generate_placement_from_spec,
    )

    slab = make_slab()
    z_max = float(np.max(slab.get_positions()[:, 2]))
    h_positions = [
        [2.0, 2.0, z_max + 1.5],
        [5.0, 2.0, z_max + 1.2],
        [2.0, 5.0, z_max + 1.3],
    ]
    slab_with_h = slab + Atoms("H3", positions=h_positions)
    slab_with_h.set_cell(slab.get_cell())
    slab_with_h.set_pbc(slab.get_pbc())
    conformers = [make_water()]
    config = AdsorptionConfig(
        placement_mode="envelope",
        placement_z_range=(2.0, 3.0),
        top_layer_tolerance=2.0,
    )
    specs = enumerate_placement_specs(conformers, slab_with_h, config, "O", n_desired=5)
    site_specs = [s for s in specs if s.site_index >= 0 and s.site_type == "envelope"]
    if not site_specs:
        pytest.skip("No envelope site specs")
    for spec in site_specs:
        result = generate_placement_from_spec(
            spec, conformers, slab_with_h, config, smiles="O"
        )
        if result is not None:
            adsorbate, descriptor = result
            reproduced = generate_placement_from_descriptor(
                descriptor, conformers, slab_with_h, config
            )
            assert reproduced is not None
            np.testing.assert_allclose(
                adsorbate.get_positions(),
                reproduced.get_positions(),
                atol=1e-6,
                err_msg="Envelope placement must reproduce with site-local z",
            )
            break


@pytest.mark.slow
@pytest.mark.mlip
@pytest.mark.gpu
@pytest.mark.no_fork  # CUDA incompatible with pytest-forked
@pytest.mark.skipif(not has_mlip_stack, reason="MLIP stack not installed")
@pytest.mark.skipif(not cuda_available, reason="CUDA GPU required")
def test_same_placement_spec_same_final_energy():
    """Same placement spec yields same final energy (within numerical error)."""

    from metalsurfer.optimization import setup_single_model
    from metalsurfer.surfaces import create_slab_from_bulk
    from metalsurfer.workflow import (
        calculate_reference_energies,
        process_molecule,
    )

    slab = create_slab_from_bulk(
        bulk_id="mp-23",
        miller_indices=(1, 1, 1),
        supercell=(2, 2, 1),
        results_dir="results_test_water_spec",
    )
    config = AdsorptionConfig(
        seed=42,
        num_conformers=1,
        num_placements=4,
        device="cuda",
        placement_mode="sites",
        placement_z_range=(2.0, 3.0),
        auto_resize_slab=False,
        autobatcher_max_memory_padding=0.5,
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab, calculator, ["water"], ["O"], ts_model=ts_model, config=config
    )
    results1 = process_molecule(
        "O",
        "water",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="test_spec",
    )
    results2 = process_molecule(
        "O",
        "water",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="test_spec",
    )
    assert results1 is not None and results2 is not None
    energies1 = {r.placement_id: r.energy_adsorption for r in results1}
    energies2 = {r.placement_id: r.energy_adsorption for r in results2}
    common = set(energies1) & set(energies2)
    assert len(common) >= 1, "Expected overlapping placement_ids from two runs"
    for pid in common:
        # Batched GPU relaxation can differ slightly between identical runs.
        np.testing.assert_allclose(
            energies1[pid],
            energies2[pid],
            atol=5e-3,
            rtol=0.15,
            err_msg=f"placement_id={pid} energy mismatch",
        )

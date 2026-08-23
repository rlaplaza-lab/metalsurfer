"""Geometry invariants: distances, material typing, contact quality."""

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111, molecule

from metalsurfer._numeric_defaults import (
    CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from metalsurfer.placement import (
    calculate_min_distance,
    check_initial_placement_distance,
    material_aware_pbc,
)
from metalsurfer.placement.geometry import (
    _classify_molecule_shape,
    calculate_contact_quality,
    check_adsorbate_separation,
    check_initial_contact_quality,
    detect_vdw_overlaps,
)

from ..conftest import (
    make_h2,
    make_slab,
    make_water,
    place_adsorbate_above_slab,
)


def test_classify_molecule_shape_linear_flat_round():
    shape_h2, _, _ = _classify_molecule_shape(make_h2().get_positions())
    assert shape_h2 == "linear"

    shape_flat, _, _ = _classify_molecule_shape(
        np.array(
            [
                [1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0.0]
                for i in range(6)
            ]
        )
    )
    assert shape_flat == "flat"

    shape_ch4, _, _ = _classify_molecule_shape(
        Atoms(
            "CH4",
            positions=[
                [0, 0, 0],
                [1.09, 1.09, 1.09],
                [-1.09, -1.09, 1.09],
                [-1.09, 1.09, -1.09],
                [1.09, -1.09, -1.09],
            ],
        ).get_positions()
    )
    assert shape_ch4 == "round"


def test_calculate_min_distance_mic_wraps_periodic_boundary():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.5, 9.5, 5.0]])
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True, pbc=[True, True, False])
    # Minimum image of (0.5,0.5)↔(9.5,9.5) in a 10×10 cell is √(1²+1²)=√2
    assert d == pytest.approx(np.sqrt(2.0), abs=1e-9)


def test_calculate_min_distance_requires_explicit_pbc_for_periodic_cell():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.5, 9.5, 5.0]])
    with pytest.raises(ValueError, match="pbc must be provided"):
        calculate_min_distance(p1, p2, cell=cell, use_pbc=True)


def test_initial_placement_distance_accepts_and_rejects_expected_heights():
    slab = make_slab()
    water = make_water()
    surface_z = float(np.max(slab.get_positions()[:, 2]))

    near = water.copy()
    p_near = near.get_positions().copy()
    p_near[:, 2] += surface_z + 0.3
    near.set_positions(p_near)
    near.set_cell(slab.get_cell())
    near.set_pbc(slab.get_pbc())
    ok_near, _, reason_near = check_initial_placement_distance(
        near, slab, material_type="slab"
    )
    assert not ok_near
    assert reason_near == "too_close"

    valid = water.copy()
    p_valid = valid.get_positions().copy()
    p_valid[:, 2] += surface_z + 2.4
    p_valid[:, 0] += 2.0
    p_valid[:, 1] += 2.0
    valid.set_positions(p_valid)
    valid.set_cell(slab.get_cell())
    valid.set_pbc(slab.get_pbc())
    ok_valid, _, reason_valid = check_initial_placement_distance(
        valid, slab, material_type="slab"
    )
    assert ok_valid
    assert reason_valid is None


def test_check_initial_placement_distance_too_far_reason():
    slab = make_slab()
    water = make_water()
    surface_z = float(np.max(slab.get_positions()[:, 2]))
    far = water.copy()
    p = far.get_positions().copy()
    p[:, 2] += surface_z + 8.0
    far.set_positions(p)
    far.set_cell(slab.get_cell())
    far.set_pbc(slab.get_pbc())
    ok, _, reason = check_initial_placement_distance(
        far, slab, max_initial_distance=3.0, material_type="slab"
    )
    assert not ok
    assert reason == "too_far"


def test_min_contact_ratio_default_is_covalent_binding_boundary():
    """Default min_contact_ratio rejects covalent overlap and accepts physisorption."""
    from ase.data import atomic_numbers, covalent_radii

    from metalsurfer._numeric_defaults import MIN_CONTACT_RATIO_DEFAULT

    slab = make_slab(n_layers=1, symbol="Ru")
    # Place a single O atom directly above a Ru atom and scan the contact ratio.
    ru_pos = slab.get_positions()[0]
    r_o = float(covalent_radii[atomic_numbers["O"]])
    r_ru = float(covalent_radii[atomic_numbers["Ru"]])
    covalent_sum = r_o + r_ru
    surface_xy = ru_pos.copy()

    too_close = Atoms("O", positions=[surface_xy + np.array([0.0, 0.0, 0.0])])
    p = too_close.get_positions().copy()
    p[0, 2] = ru_pos[2] + covalent_sum * (MIN_CONTACT_RATIO_DEFAULT - 0.05)
    too_close.set_positions(p)
    too_close.set_cell(slab.get_cell())
    too_close.set_pbc(slab.get_pbc())
    ok_close, min_close, reason_close = check_initial_placement_distance(
        too_close,
        slab,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type="slab",
    )
    assert not ok_close
    assert reason_close == "too_close"
    assert min_close < covalent_sum * MIN_CONTACT_RATIO_DEFAULT

    ok_far_side = Atoms("O", positions=[surface_xy + np.array([0.0, 0.0, 0.0])])
    p_ok = ok_far_side.get_positions().copy()
    p_ok[0, 2] = ru_pos[2] + covalent_sum * (MIN_CONTACT_RATIO_DEFAULT + 0.05)
    ok_far_side.set_positions(p_ok)
    ok_far_side.set_cell(slab.get_cell())
    ok_far_side.set_pbc(slab.get_pbc())
    ok_pass, min_pass, reason_pass = check_initial_placement_distance(
        ok_far_side,
        slab,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type="slab",
    )
    assert ok_pass
    assert reason_pass is None
    assert min_pass > covalent_sum * MIN_CONTACT_RATIO_DEFAULT


def test_check_initial_placement_distance_gates_every_covalent_pair():
    """A second pair can violate covalent floors while the global min clears min_distance."""
    from ase.data import atomic_numbers, covalent_radii

    # Pt slab atom + H (small) and Ge (large) adsorbate atoms.
    slab = Atoms(
        "Pt",
        positions=[[0.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 20.0],
        pbc=[True, True, False],
    )
    r_pt = float(covalent_radii[atomic_numbers["Pt"]])
    r_h = float(covalent_radii[atomic_numbers["H"]])
    r_ge = float(covalent_radii[atomic_numbers["Ge"]])
    ratio = 0.8
    min_distance = 1.5
    # H–Pt at 1.52 Å: above flat min_distance, and above H covalent floor.
    h_z = 1.52
    assert h_z >= min_distance
    assert h_z >= (r_h + r_pt) * ratio
    # Ge–Pt at 1.70 Å: above flat min_distance but below Ge covalent floor.
    ge_z = 1.70
    assert ge_z >= min_distance
    assert ge_z < (r_ge + r_pt) * ratio

    mol = Atoms(
        "HGe",
        positions=[[0.0, 0.0, h_z], [0.0, 0.0, ge_z]],
        cell=slab.get_cell(),
        pbc=slab.get_pbc(),
    )
    ok, _, reason = check_initial_placement_distance(
        mol,
        slab,
        min_distance=min_distance,
        min_contact_ratio=ratio,
        material_type="slab",
    )
    assert not ok
    assert reason == "too_close"


def test_vdw_overlap_detection_accepts_good_contact():
    """VDW overlap detection should accept placements with adequate separation."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=3.5, x_shift=2.0, y_shift=2.0
    )

    overlaps, min_dist = detect_vdw_overlaps(water, slab, material_type="slab")
    assert len(overlaps) == 0, "Should not detect overlaps for well-separated water"
    assert min_dist > 3.0


def test_calculate_contact_quality_detects_good_contact():
    """calculate_contact_quality should correctly identify contact atoms."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.0, x_shift=2.0, y_shift=2.0
    )

    metrics = calculate_contact_quality(
        water, slab, contact_distance_threshold=2.5, material_type="slab"
    )

    assert metrics["num_contacting_atoms"] > 0, "Should have contacting atoms"
    assert metrics["contact_distance"] < 2.8
    assert metrics["contact_ratio"] > 0.0, "Should have contact ratio"


def test_adsorbate_separation_accepts_well_separated():
    """check_adsorbate_separation should accept well-separated adsorbates."""
    slab = make_slab()
    water = make_water().copy()

    # Place water
    pos = water.get_positions()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.0
    pos[:, 0] += 5.0
    pos[:, 1] += 5.0
    water.set_positions(pos)
    water.set_cell(slab.get_cell())
    water.set_pbc(slab.get_pbc())

    # Pre-adsorbed positions far away
    pre_ads = np.array([[0.0, 0.0, 5.0]])

    ok, dist = check_adsorbate_separation(
        water,
        pre_ads,
        min_separation=2.0,
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
    )
    assert ok, "Should accept well-separated adsorbates"
    assert dist > 6.0


def test_adsorbate_separation_rejects_close_atoms():
    """check_adsorbate_separation should reject too-close adsorbates."""
    slab = make_slab()
    water = make_water().copy()

    # Place water
    pos = water.get_positions()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.0
    water.set_positions(pos)
    water.set_cell(slab.get_cell())
    water.set_pbc(slab.get_pbc())

    # Pre-adsorbed positions very close
    pre_ads = np.array([[0.0, 0.0, 5.0]])

    ok, dist = check_adsorbate_separation(
        water,
        pre_ads,
        min_separation=5.0,
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
    )
    assert not ok, "Should reject too-close adsorbates"


def test_check_initial_placement_distance_empty_geometry():
    slab = make_slab()
    empty = Atoms()
    empty.set_cell(slab.get_cell())
    empty.set_pbc(slab.get_pbc())
    ok, dist, reason = check_initial_placement_distance(
        empty, slab, material_type="slab"
    )
    assert not ok
    assert dist == float("inf")
    assert reason == "empty_geometry"


def test_min_distance_floor_rejects_close_o_cu():
    """Explicit min_distance=5 Å must reject O–Cu at ~1.5 Å even if covalent ratio allows."""
    from ase.data import atomic_numbers, covalent_radii

    slab = make_slab(n_layers=1, symbol="Cu")
    cu = slab.get_positions()[0]
    r_o = float(covalent_radii[atomic_numbers["O"]])
    r_cu = float(covalent_radii[atomic_numbers["Cu"]])
    # Within covalent-ratio acceptance but far below a 5 Å floor.
    height = 1.5
    assert height < 5.0
    assert height > (r_o + r_cu) * 0.5
    oxygen = Atoms("O", positions=[cu + np.array([0.0, 0.0, height])])
    oxygen.set_cell(slab.get_cell())
    oxygen.set_pbc(slab.get_pbc())
    ok, dist, reason = check_initial_placement_distance(
        oxygen,
        slab,
        min_distance=5.0,
        min_contact_ratio=0.5,
        material_type="slab",
    )
    assert not ok
    assert reason == "too_close"
    assert dist == pytest.approx(height, abs=1e-9)


def test_check_adsorbate_separation_requires_cell_when_pbc_requested():
    mol = make_water()
    pre = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="cell"):
        check_adsorbate_separation(mol, pre, pbc=[True, True, False])


def test_check_adsorbate_separation_requires_pbc_when_cell_periodic():
    slab = make_slab()
    mol = make_water()
    pre = np.array([[0.0, 0.0, 5.0]])
    with pytest.raises(ValueError, match="pbc must be provided"):
        check_adsorbate_separation(mol, pre, cell=slab.get_cell(), pbc=None)


def test_check_adsorbate_separation_explicit_false_pbc_uses_nonperiodic():
    slab = make_slab()
    mol = make_water()
    pre = np.array([[0.0, 0.0, 5.0]])
    ok, dist = check_adsorbate_separation(
        mol, pre, cell=slab.get_cell(), pbc=[False, False, False]
    )
    assert ok
    expected = calculate_min_distance(mol.get_positions(), pre, use_pbc=False)
    assert dist == pytest.approx(expected)


def test_calculate_min_distance_left_handed_cell_uses_abs_det():
    p1 = np.array([[0.1, 0.1, 0.0]])
    p2 = np.array([[9.9, 0.1, 0.0]])
    # Left-handed cell (det < 0) with the same |a|,|b| as a 10×10 slab.
    cell = np.array([[10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 15.0]])
    assert float(np.linalg.det(cell)) < 0.0
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True, pbc=[True, True, False])
    assert d == pytest.approx(0.2, abs=1e-9)


def test_strict_contact_gate_accepts_physical_heights_and_rejects_liftoff():
    """Regression: the default was 0.8 A, a *ratio* value in a distance field.

    ``contact_distance`` is an absolute interatomic distance bounded below at
    ~1.7 A by ``check_initial_placement_distance``, so the admissible window was
    empty and ``strict_initial_placement=True`` rejected every placement with
    the misleading reason ``contact_distance_too_large``.
    """
    assert CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM > (
        MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    ), "the strict-contact window must be non-empty"

    slab = fcc111("Pt", (3, 3, 3), vacuum=10.0)
    top_z = float(slab.get_positions()[:, 2].max())
    anchor = slab.get_positions()[-1]

    def _co_at(height):
        co = molecule("CO")
        pos = co.get_positions()
        co.translate([anchor[0], anchor[1], top_z + height - pos[:, 2].min()])
        return co

    for height in (1.8, 2.0, 2.5):
        ok, reason = check_initial_contact_quality(
            _co_at(height), slab, strict_initial_placement=True
        )
        assert ok, f"physical height {height} A rejected: {reason}"

    ok_far, reason_far = check_initial_contact_quality(
        _co_at(8.0), slab, strict_initial_placement=True
    )
    assert not ok_far
    assert reason_far == "contact_distance_too_large"

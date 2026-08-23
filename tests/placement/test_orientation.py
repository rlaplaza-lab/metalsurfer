"""Adsorbate orientation strategies and parallel-fraction estimation."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement import (
    enumerate_placement_specs,
)
from metalsurfer.placement.orientation import (
    _estimate_parallel_fraction,
    _is_flat_aromatic_with_en,
)

from ..conftest import (
    make_slab,
)


def test_flat_aromatic_detection_requires_ring_and_en_atoms():
    assert _is_flat_aromatic_with_en("c1(C=O)cc(OC)c(O)cc1") is True
    assert _is_flat_aromatic_with_en("c1ccccc1") is False
    assert _is_flat_aromatic_with_en("CCO") is False


def test_flat_aromatic_specs_include_parallel_and_en_down_when_applicable():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=24,
        placement_z_range=(2.0, 3.0),
        flat_aromatic_parallel_fraction=0.5,
    )
    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1",
        config=AdsorptionConfig(num_conformers=3),
    )
    if result is None:
        pytest.skip("RDKit required")
    conformers, _ = result

    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        "c1(C=O)cc(OC)c(O)cc1",
        n_desired=24,
    )
    kinds = {spec.orientation_type for spec in specs}
    assert "parallel" in kinds
    assert "EN-down" in kinds


@pytest.mark.parametrize(
    "symbols, smiles, expected",
    [
        (["C"] * 6 + ["H"] * 6, "c1ccccc1", 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, "c1ccncc1", 0.3),
        (["C"] * 6 + ["O"] + ["H"] * 6, "c1ccccc1O", 0.3),
        (["C"] * 6 + ["N", "O"] + ["H"] * 6, "c1ccc(O)c(N)c1", 0.5),
        (["C"] * 6 + ["H"] * 6, None, 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, None, 0.3),
        (["C"] * 4 + ["N", "O"] + ["H"] * 4, None, 0.3),
        (["C"] * 8 + ["N", "O"] + ["H"] * 8, None, 0.5),
    ],
)
def test_estimate_parallel_fraction(symbols, smiles, expected):
    frac = _estimate_parallel_fraction(symbols, smiles=smiles)
    assert frac == expected


def test_principal_axis_rotation_flat_hexagon_stays_near_flat():
    from metalsurfer.placement.geometry import _principal_axis_rotation

    hex_pos = np.array(
        [
            [1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0.0]
            for i in range(6)
        ],
        dtype=float,
    )
    hex_pos -= hex_pos.mean(axis=0)
    rotated, _score, _R = _principal_axis_rotation(hex_pos, np.array([0.0, 0.0, 1.0]))
    # Plane normal ≈ z → z-span stays small (near-flat).
    assert float(np.ptp(rotated[:, 2])) < 0.28


@pytest.mark.parametrize("tilt_deg", [0.0, 10.0, 15.0])
def test_surface_aligned_rotation_flips_binder_pointing_up(tilt_deg):
    """Binder near +normal must rotate to point toward the surface (−normal)."""
    from metalsurfer.placement.geometry import _surface_aligned_rotation

    normal = np.array([0.0, 0.0, 1.0])
    angle = np.deg2rad(tilt_deg)
    # C at origin, O tilted slightly from +z (away from surface).
    pos = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=float,
    )
    out, _R = _surface_aligned_rotation(pos, normal, symbols=["C", "O"])
    com = out.mean(axis=0)
    binder_dir = out[1] - com
    binder_dir /= np.linalg.norm(binder_dir)
    assert float(np.dot(binder_dir, normal)) < -0.95


def test_surface_aligned_rotation_noop_when_already_pointing_down():
    from metalsurfer.placement.geometry import _surface_aligned_rotation

    normal = np.array([0.0, 0.0, 1.0])
    pos = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0]], dtype=float)
    out, _R = _surface_aligned_rotation(pos, normal, symbols=["C", "O"])
    com = out.mean(axis=0)
    binder_dir = out[1] - com
    binder_dir /= np.linalg.norm(binder_dir)
    assert float(np.dot(binder_dir, normal)) < -0.95


def test_composed_rotation_reproduces_sequential():
    """R_base @ R_tilt @ canonical must equal the sequentially built rotated_pos."""
    from metalsurfer.placement.geometry import (
        _flat_orientation_from_principal_axis,
        _rotation_with_tilt,
    )

    normal = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(7, 3))
    pos -= pos.mean(axis=0)
    base_pos, R_base = _flat_orientation_from_principal_axis(
        pos, normal, azimuth_in_plane_deg=37.0, face_flip=True
    )
    rotated, R_tilt = _rotation_with_tilt(
        base_pos, normal, tilt_deg=15.0, azimuth_deg=22.0
    )
    R_total = R_tilt @ R_base
    composed = (R_total @ np.asarray(pos, dtype=float).T).T
    assert np.allclose(composed, rotated, atol=1e-10)

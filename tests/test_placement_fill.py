"""Unit tests for the placement fill engine's discrete helpers."""

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.workflow.placement_fill import (
    _clamp_target_to_capacity,
    _request_count,
    _yield_floor,
    placement_cell_key,
)

from .conftest import make_slab, make_water
from .placement._helpers import _round_atop_placement_spec


def test_placement_cell_key_excludes_continuous_pose_params():
    a = _round_atop_placement_spec(
        tilt_deg=10.0, azimuth_deg=20.0, z_fraction=0.1, azimuth_in_plane_deg=5.0
    )
    b = _round_atop_placement_spec(
        tilt_deg=80.0, azimuth_deg=200.0, z_fraction=0.9, azimuth_in_plane_deg=355.0
    )
    # Same discrete neighborhood: continuous pose differences are ignored.
    assert placement_cell_key(a) == placement_cell_key(b)

    c = _round_atop_placement_spec(site_index=99)
    assert placement_cell_key(a) != placement_cell_key(c)


@pytest.mark.parametrize(
    "oversample_max, expected",
    [
        (1.0, 1.0),
        (2.0, 0.5),
        (4.0, 0.25),
        (10.0, 0.1),
        (0.1, 1.0),  # values below 1 are clamped so the floor never exceeds 1
    ],
)
def test_yield_floor_is_inverse_of_oversample(oversample_max, expected):
    assert _yield_floor(oversample_max) == pytest.approx(expected)


@pytest.mark.parametrize(
    "remaining, yield_est, oversample_max, expected",
    [
        (0, 0.5, 4.0, 0),  # no deficit -> no request
        (10, 1.0, 4.0, 10),  # perfect yield: exactly the deficit
        (10, 0.5, 8.0, 20),  # 50% yield: double request
        (10, 0.01, 2.0, 20),  # estimate below floor is floored to 1/2
        (6, 0.05, 3.0, 18),  # terrible yield capped at remaining * oversample
    ],
)
def test_request_count(remaining, yield_est, oversample_max, expected):
    assert (
        _request_count(
            remaining=remaining,
            yield_est=yield_est,
            oversample_max=oversample_max,
        )
        == expected
    )


def test_clamp_target_to_capacity_disabled_passthrough():
    slab = make_slab(nx=1, ny=1)
    config = AdsorptionConfig(
        material_type="slab",
        placement_fill_clamp_to_capacity=False,
    )
    n = _clamp_target_to_capacity(
        n_target=10_000,
        conformers=[],
        slab_for_sites=slab,
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=slab,
    )
    assert n == 10_000


def test_clamp_target_to_capacity_caps_at_enumerable_capacity(monkeypatch):
    """An unreachable target must be clamped to the estimated spec capacity."""
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()

    monkeypatch.setattr(
        fill_mod,
        "estimate_molecule_complexity",
        lambda conformers, *a, **k: float(len(conformers)) * 21.0,
    )
    config = AdsorptionConfig(material_type="slab")
    kwargs = dict(
        conformers=[water, water],  # capacity 42
        slab_for_sites=slab,
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=slab,
    )

    assert _clamp_target_to_capacity(n_target=10_000, **kwargs) == 42
    # A target within capacity is kept unchanged.
    assert _clamp_target_to_capacity(n_target=42, **kwargs) == 42

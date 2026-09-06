"""Unit tests for the one-shot placement fill helpers."""

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.workflow.placement_fill import (
    _clamp_target_to_capacity,
    _pool_request_count,
)

from .conftest import make_slab, make_water


@pytest.mark.parametrize(
    "n_target, oversample_max, capacity, expected",
    [
        (0, 6.0, None, 0),
        (10, 1.0, None, 10),
        (10, 4.0, None, 40),
        (10, 6.0, 25, 25),
        (10, 6.0, 100, 60),
        (10, 6.0, 0, 0),
    ],
)
def test_pool_request_count(n_target, oversample_max, capacity, expected):
    assert _pool_request_count(n_target, oversample_max, capacity=capacity) == expected


def test_clamp_target_to_capacity_disabled_passthrough():
    slab = make_slab(nx=1, ny=1)
    config = AdsorptionConfig(
        material_type="slab",
        placement_fill_clamp_to_capacity=False,
    )
    assert (
        _clamp_target_to_capacity(
            n_target=10_000,
            conformers=[],
            slab_for_sites=slab,
            config=config,
            smiles="O",
            site_context=None,
            slab_atoms=slab,
        )
        == 10_000
    )


def test_clamp_target_to_capacity_caps_at_enumerable_capacity(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    monkeypatch.setattr(
        fill_mod,
        "estimate_placement_capacity",
        lambda conformers, *a, **k: float(len(conformers)) * 21.0,
    )
    kwargs = dict(
        conformers=[water, water],
        slab_for_sites=slab,
        config=AdsorptionConfig(material_type="slab"),
        smiles="O",
        site_context=None,
        slab_atoms=slab,
    )
    assert _clamp_target_to_capacity(n_target=10_000, **kwargs) == 42
    assert _clamp_target_to_capacity(n_target=42, **kwargs) == 42
    # Precomputed capacity skips a second estimate.
    assert _clamp_target_to_capacity(n_target=10_000, capacity=7, **kwargs) == 7

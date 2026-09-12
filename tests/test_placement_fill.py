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
        "estimate_placement_spec_capacity",
        lambda conformers, *a, **k: len(conformers) * 21,
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


def test_materialize_specs_uses_supplied_capacity_without_estimating(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    estimator_called = False

    def fail_if_estimated(*_args, **_kwargs):
        nonlocal estimator_called
        estimator_called = True
        raise AssertionError("capacity was estimated")

    def materialize_all(*, specs, **_kwargs):
        count = len(specs)
        return (
            [water.copy() for _ in range(count)],
            list(range(count)),
            [],
            [],
        )

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", fail_if_estimated)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", materialize_all)

    result = fill_mod.materialize_specs(
        specs=[object() for _ in range(10)],
        n_target=10,
        conformers=[water],
        slab_atoms=slab,
        calculator=None,
        config=AdsorptionConfig(material_type="slab"),
        smiles="O",
        site_context=None,
        capacity=7,
    )

    assert not estimator_called
    assert len(result.combined) == 7
    assert result.placement_ids == list(range(7))
    assert result.n_attempts == 1


def test_materialize_specs_without_capacity_estimates_capacity(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    calls = []

    def estimate_capacity(*_args, **_kwargs):
        calls.append(1)
        return 3

    def materialize_all(*, specs, **_kwargs):
        count = len(specs)
        return (
            [water.copy() for _ in range(count)],
            list(range(count)),
            [],
            [],
        )

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", estimate_capacity)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", materialize_all)

    result = fill_mod.materialize_specs(
        specs=[object() for _ in range(10)],
        n_target=10,
        conformers=[water],
        slab_atoms=slab,
        calculator=None,
        config=AdsorptionConfig(material_type="slab"),
        smiles="O",
        site_context=None,
    )

    assert calls == [1]
    assert len(result.combined) == 3
    assert result.n_attempts == 1


def test_materialize_specs_disabled_clamping_ignores_capacity(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    estimator_called = False

    def fail_if_estimated(*_args, **_kwargs):
        nonlocal estimator_called
        estimator_called = True
        raise AssertionError("capacity was estimated")

    def materialize_all(*, specs, **_kwargs):
        count = len(specs)
        return (
            [water.copy() for _ in range(count)],
            list(range(count)),
            [],
            [],
        )

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", fail_if_estimated)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", materialize_all)

    result = fill_mod.materialize_specs(
        specs=[object() for _ in range(10)],
        n_target=10,
        conformers=[water],
        slab_atoms=slab,
        calculator=None,
        config=AdsorptionConfig(
            material_type="slab",
            placement_fill_clamp_to_capacity=False,
        ),
        smiles="O",
        site_context=None,
        capacity=7,
    )

    assert not estimator_called
    assert len(result.combined) == 10
    assert result.n_attempts == 1


@pytest.mark.parametrize("capacity", [0, -1])
def test_materialize_specs_nonpositive_capacity_returns_empty(monkeypatch, capacity):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    estimator_called = False
    materializer_called = False

    def fail_if_estimated(*_args, **_kwargs):
        nonlocal estimator_called
        estimator_called = True
        raise AssertionError("capacity was estimated")

    def fail_if_materialized(**_kwargs):
        nonlocal materializer_called
        materializer_called = True
        raise AssertionError("materialization ran")

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", fail_if_estimated)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fail_if_materialized)

    result = fill_mod.materialize_specs(
        specs=[object()],
        n_target=10,
        conformers=[water],
        slab_atoms=slab,
        calculator=None,
        config=AdsorptionConfig(material_type="slab"),
        smiles="O",
        site_context=None,
        capacity=capacity,
    )

    assert not estimator_called
    assert not materializer_called
    assert result.combined == []
    assert result.placement_ids == []
    assert result.descriptors == []
    assert result.failures == []
    assert result.n_attempts == 0


@pytest.mark.parametrize("estimated_capacity", [0, -1])
def test_materialize_specs_nonpositive_estimated_capacity_returns_empty(
    monkeypatch, estimated_capacity
):
    from metalsurfer.workflow import placement_fill as fill_mod

    slab = make_slab()
    water = make_water()
    calls = []
    materializer_called = False

    def estimate_capacity(*_args, **_kwargs):
        calls.append(1)
        return estimated_capacity

    def fail_if_materialized(**_kwargs):
        nonlocal materializer_called
        materializer_called = True
        raise AssertionError("materialization ran")

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", estimate_capacity)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fail_if_materialized)

    result = fill_mod.materialize_specs(
        specs=[object()],
        n_target=10,
        conformers=[water],
        slab_atoms=slab,
        calculator=None,
        config=AdsorptionConfig(material_type="slab"),
        smiles="O",
        site_context=None,
    )

    assert calls == [1]
    assert not materializer_called
    assert result.combined == []
    assert result.n_attempts == 0

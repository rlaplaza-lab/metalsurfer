"""Workload autotuning: parallel-capacity-driven placement/BO budget resolution."""

from dataclasses import replace

import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.workflow import shared as shared_mod
from metalsurfer.workflow.shared import (
    needs_workload_autotune,
    resolve_workload_config,
)

_REPRESENTATIVE = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])


@pytest.mark.parametrize(
    "num_placements,initial_random,batch_size,bo,expected",
    [
        # Nothing pinned: always needs autotuning.
        (None, None, None, False, True),
        (None, None, None, True, True),
        # Placement count fixed, no BO: fully resolved.
        (12, None, None, False, False),
        # Placement count fixed but BO fields missing.
        (12, None, None, True, True),
        (12, 8, None, True, True),
        (12, None, 4, True, True),
        # Fully resolved BO workload.
        (12, 8, 4, True, False),
    ],
)
def test_needs_workload_autotune_truth_table(
    num_placements, initial_random, batch_size, bo, expected
):
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=num_placements,
        bo=BOConfig(initial_random=initial_random, batch_size=batch_size),
    )
    assert needs_workload_autotune(config, bo=bo) is expected


def test_resolve_workload_config_fills_all_fields_from_probed_capacity(monkeypatch):
    capacity_calls = []

    def _fake_capacity(ts_model, representative_atoms, config, *, frozen_indices):
        capacity_calls.append((len(representative_atoms), tuple(frozen_indices)))
        return 7

    monkeypatch.setattr(
        shared_mod, "estimate_parallel_relaxation_capacity", _fake_capacity
    )
    config = AdsorptionConfig(material_type="slab")  # all three fields None

    resolved = resolve_workload_config(
        config,
        ts_model=object(),
        representative_atoms=_REPRESENTATIVE,
        frozen_indices=[0],
        bo_enabled=True,
    )

    assert len(capacity_calls) == 1
    assert resolved.num_placements == 7
    assert resolved.bo.initial_random == 7
    assert resolved.bo.batch_size == 7
    # The input config must stay untouched (functional update).
    assert config.num_placements is None
    assert config.bo.initial_random is None


def test_resolve_workload_config_fills_only_missing_fields(monkeypatch):
    monkeypatch.setattr(
        shared_mod,
        "estimate_parallel_relaxation_capacity",
        lambda *a, **k: 9,
    )
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=3,
        bo=BOConfig(initial_random=5, batch_size=None),
    )
    resolved = resolve_workload_config(
        config,
        ts_model=object(),
        representative_atoms=_REPRESENTATIVE,
        frozen_indices=[],
        bo_enabled=True,
    )
    assert resolved.num_placements == 3  # kept
    assert resolved.bo.initial_random == 5  # kept
    assert resolved.bo.batch_size == 9  # filled


def test_resolve_workload_config_noop_returns_input_unchanged():
    config = AdsorptionConfig(material_type="slab", num_placements=4)
    resolved = resolve_workload_config(
        config,
        ts_model=object(),
        representative_atoms=_REPRESENTATIVE,
        frozen_indices=[],
        bo_enabled=False,
    )
    assert resolved is config


def test_resolve_saturation_step_workload_config_resolves_before_split(monkeypatch):
    """The step wrapper must plumb site/freeze context into the capacity probe."""
    sentinel_site_context = object()
    recorded: dict[str, object] = {}

    def _fake_site_context(slab_for_sites, config, *, symmetry_broken):
        recorded["symmetry_broken"] = symmetry_broken
        return sentinel_site_context

    def _fake_representative(
        conformers, slab_atoms, slab_for_sites, config, smiles, *, site_context
    ):
        recorded["site_context_is_sentinel"] = site_context is sentinel_site_context
        recorded["smiles"] = smiles
        return _REPRESENTATIVE

    monkeypatch.setattr(
        shared_mod, "resolve_site_context_for_sampling", _fake_site_context
    )
    monkeypatch.setattr(
        shared_mod, "build_representative_relaxation_atoms", _fake_representative
    )
    monkeypatch.setattr(
        shared_mod,
        "estimate_parallel_relaxation_capacity",
        lambda *a, **k: 11,
    )

    water = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 0.9, 0.0]])
    resolved = shared_mod.resolve_saturation_step_workload_config(
        replace(AdsorptionConfig(material_type="slab"), num_placements=None),
        ts_model=object(),
        conformers=[water],
        slab_atoms=water,  # plumbing is fully mocked; identity suffices
        slab_for_sites=water,
        smiles="O",
        base_slab_for_frozen=None,
        symmetry_broken=True,
        bo_enabled=False,
    )

    assert recorded["symmetry_broken"] is True
    assert recorded["smiles"] == "O"
    assert recorded["site_context_is_sentinel"] is True
    assert resolved.num_placements == 11

"""Unit tests for joint n-tuplet screening helpers."""

import math
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import ReferenceEnergies
from metalsurfer.reporting import PlacementFailure
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow.composite import (
    assemble_joint_config_groups,
    assemble_quota_joint_configs,
)
from metalsurfer.workflow.joint_tuplet import (
    JointTupletScreenOutcome,
    _pool_size_for_joint_screen,
    screen_joint_tuplet_homogeneous,
)

from .conftest import (
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)


@pytest.mark.parametrize(
    ("n", "num_placements", "oversample"),
    [
        (2, 4, 2.0),
        (3, 1, 1.5),
        (1, 1, 1.0),
    ],
)
def test_pool_size_matches_one_species_slot_formula(n, num_placements, oversample):
    config = AdsorptionConfig(
        saturation_molecules_per_step=n,
        num_placements=num_placements,
        placement_retry_oversample_max=oversample,
    )
    expected = max(1, int(math.ceil(num_placements * n * oversample)))
    assert _pool_size_for_joint_screen(config) == expected
    slots = n * num_placements
    assert max(1, int(math.ceil(slots * oversample))) == expected


def test_pool_size_floor_when_product_below_one():
    config = cast(
        AdsorptionConfig,
        SimpleNamespace(
            num_placements=1,
            saturation_molecules_per_step=2,
            placement_retry_oversample_max=0.1,
        ),
    )
    assert _pool_size_for_joint_screen(config) == 1


def test_homogeneous_delegates_assembly_seed_and_flags(monkeypatch):
    slab = make_slab()
    config = AdsorptionConfig(
        seed=17,
        num_placements=3,
        saturation_molecules_per_step=2,
    )
    ref = ReferenceEnergies(
        slab_energy=0.0,
        molecule_energies={"water": -14.0},
    )
    captured: dict = {}

    def _fake_multi(**kwargs):
        captured.update(kwargs)
        return JointTupletScreenOutcome(valid_configs=[], flat_results=[])

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.screen_joint_tuplet_multi",
        _fake_multi,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.site_context_for_sampling",
        lambda *_a, **_k: object(),
    )

    screen_joint_tuplet_homogeneous(
        smiles="O",
        molecule_name="water",
        current_slab=SlabContainer(slab),
        calculator=None,
        ref_step=ref,
        ts_model=None,
        config=config,
        base_slab=slab,
        E_slab=0.0,
        symmetry_broken=False,
        conformers=[make_water()],
        conformer_energies=None,
        site_context=None,
        debug_sites_step=None,
    )
    assert captured["assembly_seed"] == 17
    assert captured["empty_pool_failure"] is True
    assert captured["assemble"] == "exact"

    captured.clear()
    screen_joint_tuplet_homogeneous(
        smiles="O",
        molecule_name="water",
        current_slab=SlabContainer(slab),
        calculator=None,
        ref_step=ref,
        ts_model=None,
        config=config,
        base_slab=slab,
        E_slab=0.0,
        symmetry_broken=False,
        conformers=[make_water()],
        conformer_energies=None,
        site_context=None,
        debug_sites_step=4,
    )
    assert captured["assembly_seed"] == 17 + 4


def test_homogeneous_empty_pool_returns_placement_failure(monkeypatch):
    slab = make_slab()
    config = AdsorptionConfig(
        seed=1,
        num_placements=5,
        saturation_molecules_per_step=2,
    )
    ref = ReferenceEnergies(
        slab_energy=0.0,
        molecule_energies={"water": -14.0},
    )
    relax_calls: list[object] = []

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.site_context_for_sampling",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet._materialize_pose_pool",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet._relax_groups",
        lambda **_k: relax_calls.append(True),
    )

    out = screen_joint_tuplet_homogeneous(
        smiles="O",
        molecule_name="water",
        current_slab=SlabContainer(slab),
        calculator=None,
        ref_step=ref,
        ts_model=None,
        config=config,
        base_slab=slab,
        E_slab=0.0,
        symmetry_broken=False,
        conformers=[make_water()],
        conformer_energies=None,
        site_context=None,
    )
    assert out.valid_configs == []
    assert isinstance(out.failure_summary, PlacementFailure)
    assert out.failure_summary.n_placements_attempted == 5
    assert out.failure_summary.n_initial_placements == 0
    assert relax_calls == []


def test_exact_and_quota_assemblers_diverge_when_quota_retries_skipped_pose(
    monkeypatch,
):
    """Quota restarts its scan each slot; exact keeps a forward scan per group."""
    slab = make_slab()
    poses = []
    for i in range(4):
        combined = place_molecule_on_slab(
            slab, make_water(), z_offset=3.0, x_shift=2.0 + i, y_shift=5.0
        )
        poses.append(
            make_screening_result(
                molecule="water",
                placement_id=i,
                energy_adsorption=-1.0,
                atoms=combined,
                slab_size=len(slab),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=i),
            )
        )
    config = AdsorptionConfig(material_type="slab", placement_clash_descent=True)

    def _fake_pack(winners, _slab_atoms, _config, *, n_substrate=None):
        ids = tuple(w.placement_id for w in winners)
        if len(ids) == 1:
            return list(winners)
        if ids == (0, 1):
            return None
        if ids in {(0, 2), (0, 2, 1), (0, 2, 3)}:
            return list(winners)
        return None

    monkeypatch.setattr(
        "metalsurfer.workflow.composite.pack_exact_tuplet",
        _fake_pack,
    )

    class _IdentityRng:
        def permutation(self, n: int) -> np.ndarray:
            return np.arange(n)

    rng = cast(np.random.Generator, _IdentityRng())
    exact_groups = assemble_joint_config_groups(
        poses,
        n_per_config=3,
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
    )
    quota_groups = assemble_quota_joint_configs(
        {"water": poses},
        quotas={"water": 3},
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
    )
    assert [[r.placement_id for r in g] for g in exact_groups] == [[0, 2, 3]]
    assert [[r.placement_id for r in g] for g in quota_groups] == [[0, 2, 1]]


def test_homogeneous_missing_molecule_energy_raises_before_multi(monkeypatch):
    slab = make_slab()
    config = AdsorptionConfig(num_placements=2, saturation_molecules_per_step=2)
    ref = ReferenceEnergies(slab_energy=0.0, molecule_energies={})
    called = {"multi": False}

    def _fake_multi(**_k):
        called["multi"] = True
        raise AssertionError("multi must not run when E_mol is missing")

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.screen_joint_tuplet_multi",
        _fake_multi,
    )
    with pytest.raises(ValueError, match="missing reference energy"):
        screen_joint_tuplet_homogeneous(
            smiles="O",
            molecule_name="water",
            current_slab=SlabContainer(slab),
            calculator=None,
            ref_step=ref,
            ts_model=None,
            config=config,
            base_slab=slab,
            E_slab=0.0,
            symmetry_broken=False,
            conformers=[make_water()],
            conformer_energies=None,
            site_context=None,
        )
    assert called["multi"] is False

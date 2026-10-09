"""Unit tests for joint n-tuplet screening helpers."""

import math
from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.models import ReferenceEnergies
from metalsurfer.reporting import PlacementFailure
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow import joint_tuplet as joint_tuplet_mod
from metalsurfer.workflow.composite import (
    assemble_joint_config_groups,
    assemble_quota_joint_configs,
)
from metalsurfer.workflow.joint_tuplet import (
    JointTupletScreenOutcome,
    _assemble_bo_joint_groups_from_specs,
    _composition_budgets,
    _composition_funding_order,
    enumerate_tuplet_compositions,
    process_joint_tuplet_bayesian,
    screen_joint_tuplet_homogeneous,
)
from metalsurfer.workflow.shared import MoleculeScreeningContext

from .conftest import (
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)


def test_homogeneous_pool_size_uses_slot_oversample(monkeypatch):
    """One-species multi path sizes the pose pool as ceil(n * configs * oversample)."""
    slab = make_slab()
    config = AdsorptionConfig(
        seed=1,
        num_placements=4,
        saturation_molecules_per_step=2,
        placement_retry_oversample_max=2.0,
    )
    ref = ReferenceEnergies(slab_energy=0.0, molecule_energies={"water": -14.0})
    captured: dict = {}

    def _fake_materialize(**kwargs):
        captured["pool_size"] = kwargs["pool_size"]
        return []

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.site_context_for_sampling",
        lambda *_a, **_k: object(),
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet._materialize_pose_pool",
        _fake_materialize,
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
    )
    assert captured["pool_size"] == max(1, int(math.ceil(4 * 2 * 2.0)))


def test_composition_funding_order_pures_before_mixtures():
    molecules = ["A", "B", "C"]
    compositions = enumerate_tuplet_compositions(molecules, 2)
    order = _composition_funding_order(compositions, molecules)
    ordered = [compositions[i] for i in order]
    pures = ordered[:3]
    assert pures == [
        {"A": 2, "B": 0, "C": 0},
        {"A": 0, "B": 2, "C": 0},
        {"A": 0, "B": 0, "C": 2},
    ]
    assert all(sum(1 for v in c.values() if int(v) > 0) > 1 for c in ordered[3:])


def test_composition_budgets_short_funds_pures_first():
    molecules = ["A", "B", "C"]
    compositions = enumerate_tuplet_compositions(molecules, 2)
    budgets = _composition_budgets(compositions, molecules, n_configs=3)
    funded = [compositions[i] for i, share in enumerate(budgets) if share > 0]
    assert len(funded) == 3
    assert all(sum(1 for v in c.values() if int(v) > 0) == 1 for c in funded)
    assert sum(budgets) == 3


def test_composition_budgets_equal_when_enough_slots():
    molecules = ["water", "OH"]
    compositions = enumerate_tuplet_compositions(molecules, 2)
    budgets = _composition_budgets(compositions, molecules, n_configs=6)
    assert budgets == [2, 2, 2]
    assert sum(budgets) == 6


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


def test_bo_assemble_releases_incomplete_companions(monkeypatch):
    """Failed packs must not swallow later anchors from the same batch."""
    slab = make_slab()
    config = AdsorptionConfig(saturation_molecules_per_step=2)
    stubs = {
        i: make_screening_result(
            molecule="water",
            placement_id=i,
            energy_adsorption=0.0,
            atoms=place_molecule_on_slab(
                slab, make_water(), z_offset=3.0, x_shift=float(i), y_shift=0.0
            ),
            slab_size=len(slab),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=i),
        )
        for i in range(4)
    }

    def _fake_materialize(*, specs, **_k):
        pid = int(specs[0].placement_index)
        stub = stubs[pid]
        return MagicMock(
            combined=[stub.atoms],
            placement_ids=[pid],
            descriptors=[stub.placement_descriptor],
        )

    def _fake_pack(winners, _slab, _config, *, n_substrate=None):
        ids = tuple(w.placement_id for w in winners)
        if len(ids) == 1:
            return list(winners)
        # Anchor 0 cannot pair with anyone; anchor 1 pairs with 2.
        if ids[0] == 0:
            return None
        if ids == (1, 2):
            return list(winners)
        return None

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.materialize_specs",
        _fake_materialize,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.pack_exact_tuplet",
        _fake_pack,
    )

    class _NoShuffleRng:
        def shuffle(self, x):
            return None

    specs = [MagicMock(placement_index=i) for i in range(4)]
    groups, pool_sets = _assemble_bo_joint_groups_from_specs(
        [0, 1],
        all_specs=specs,
        valid_spec_indices=list(range(4)),
        materialization_cache={},
        conformers=[make_water()],
        slab=SlabContainer(slab),
        calculator=None,
        config=config,
        smiles="O",
        molecule_name="water",
        site_context=MagicMock(),
        slab_for_sites=slab,
        E_slab=0.0,
        E_mol=-14.0,
        n_per_config=2,
        rng=cast(np.random.RandomState, _NoShuffleRng()),
        n_substrate=len(slab),
    )
    assert len(groups) == 1
    assert [r.placement_id for r in groups[0]] == [1, 2]
    assert pool_sets == [{1, 2}]


def test_bo_assemble_respects_blocked_companions(monkeypatch):
    slab = make_slab()
    config = AdsorptionConfig(saturation_molecules_per_step=2)
    stubs = {
        i: make_screening_result(
            molecule="water",
            placement_id=i,
            energy_adsorption=0.0,
            atoms=place_molecule_on_slab(
                slab, make_water(), z_offset=3.0, x_shift=float(i), y_shift=0.0
            ),
            slab_size=len(slab),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=i),
        )
        for i in range(3)
    }

    def _fake_materialize(*, specs, **_k):
        pid = int(specs[0].placement_index)
        stub = stubs[pid]
        return MagicMock(
            combined=[stub.atoms],
            placement_ids=[pid],
            descriptors=[stub.placement_descriptor],
        )

    def _fake_pack(winners, _slab, _config, *, n_substrate=None):
        return list(winners)

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.materialize_specs",
        _fake_materialize,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.pack_exact_tuplet",
        _fake_pack,
    )

    class _NoShuffleRng:
        def shuffle(self, x):
            return None

    specs = [MagicMock(placement_index=i) for i in range(3)]
    groups, pool_sets = _assemble_bo_joint_groups_from_specs(
        [0],
        all_specs=specs,
        valid_spec_indices=list(range(3)),
        materialization_cache={},
        conformers=[make_water()],
        slab=SlabContainer(slab),
        calculator=None,
        config=config,
        smiles="O",
        molecule_name="water",
        site_context=MagicMock(),
        slab_for_sites=slab,
        E_slab=0.0,
        E_mol=-14.0,
        n_per_config=2,
        rng=cast(np.random.RandomState, _NoShuffleRng()),
        n_substrate=len(slab),
        blocked={1},
    )
    assert len(groups) == 1
    assert [r.placement_id for r in groups[0]] == [0, 2]
    assert pool_sets == [{0, 2}]


def test_process_joint_tuplet_bayesian_uses_ctx_conformer_energies(monkeypatch):
    """Prep-resolved conformer energies must reach enumeration."""
    slab = make_slab()
    config = AdsorptionConfig(
        num_placements=2,
        saturation_molecules_per_step=2,
        bo=BOConfig(initial_random=1, batch_size=1, total_budget=1),
    )
    captured: dict = {}

    ctx = MoleculeScreeningContext(
        slab=SlabContainer(slab),
        slab_for_sites=slab,
        effective_base_slab_for_frozen=slab,
        conformers=[make_water()],
        site_context=MagicMock(),
        config=config,
        E_slab=0.0,
        E_mol=-14.0,
        t_conformers=0.0,
        conformer_energies=[0.12],
    )

    monkeypatch.setattr(
        joint_tuplet_mod,
        "_prepare_molecule_screening",
        lambda **_k: (ctx, None),
    )
    monkeypatch.setattr(
        joint_tuplet_mod,
        "estimate_placement_spec_capacity",
        lambda *_a, **_k: 4,
    )

    def _fake_enumerate(*_a, **kwargs):
        captured["conformer_energies"] = kwargs.get("conformer_energies")
        return []

    monkeypatch.setattr(joint_tuplet_mod, "enumerate_placement_specs", _fake_enumerate)

    out = process_joint_tuplet_bayesian(
        "O",
        "water",
        SlabContainer(slab),
        calculator=None,
        reference_energies=ReferenceEnergies(
            slab_energy=0.0, molecule_energies={"water": -14.0}
        ),
        config=config,
        conformers=[make_water()],
        conformer_energies=None,
    )
    assert captured["conformer_energies"] == [0.12]
    assert out.valid_configs == []

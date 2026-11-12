"""Unit tests for joint n-tuplet screening helpers."""

import math
from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.filters import (
    lateral_contact_shell,
    min_interadsorbate_covalent_ratio,
)
from metalsurfer.models import ReferenceEnergies
from metalsurfer.placement.geometry import occupied_neighbor_counts
from metalsurfer.reporting import PlacementFailure
from metalsurfer.surface_prep import Substrate
from metalsurfer.workflow import composite as composite_mod
from metalsurfer.workflow import joint_tuplet as joint_tuplet_mod
from metalsurfer.workflow.composite import (
    _site_positions,
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
    make_h2,
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
        current_slab=Substrate(slab, material_type="slab"),
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

    screen_joint_tuplet_homogeneous(
        smiles="O",
        molecule_name="water",
        current_slab=Substrate(slab, material_type="slab"),
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
    assert captured["symmetry_broken"] is False
    assert captured["site_context"] is None

    captured.clear()
    screen_joint_tuplet_homogeneous(
        smiles="O",
        molecule_name="water",
        current_slab=Substrate(slab, material_type="slab"),
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
        current_slab=Substrate(slab, material_type="slab"),
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


def test_trimer_assembly_continues_after_a_short_seed(monkeypatch) -> None:
    """A seed that can take only one partner does not abandon the queue."""
    slab = make_slab(nx=8)
    cell = np.asarray(slab.get_cell(), dtype=float)
    config = AdsorptionConfig(placement_clash_descent=False)
    y_shift = float(cell[1, 1]) / 2.0
    step = 4.0
    poses = []
    for placement_id in range(6):
        poses.append(
            make_screening_result(
                molecule="h2",
                placement_id=placement_id,
                energy_adsorption=-1.0,
                atoms=place_molecule_on_slab(
                    slab,
                    make_h2(),
                    x_shift=2.0 + placement_id * step,
                    y_shift=y_shift,
                ),
                slab_size=len(slab),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(
                    placement_id=placement_id
                ),
            )
        )
    real_accept = composite_mod._accept_packed_pose

    def _accept(group, index, **kwargs):
        seed = group[0].placement_id if group else None
        pose = kwargs["pose_at"](index)
        if pose is None:
            return None
        if seed == 0 and (len(group) >= 2 or pose.placement_id != 1):
            return None
        if seed == 1:
            return None
        return real_accept(group, index, **kwargs)

    monkeypatch.setattr(composite_mod, "_accept_packed_pose", _accept)
    rng = np.random.default_rng(0)
    groups = assemble_joint_config_groups(
        poses,
        n_per_config=3,
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    assert len(groups) == 1
    assert len(groups[0]) == 3
    assert groups[0][0].placement_id not in {0, 1}

    quota_groups = assemble_quota_joint_configs(
        {"h2": poses},
        quotas={"h2": 3},
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    assert len(quota_groups) == 1
    assert [pose.placement_id for pose in quota_groups[0]][0] not in {0, 1}


def test_assemblers_do_not_replace_a_clashing_neighbor() -> None:
    """A clashing neighbor is skipped. A non-neighbor partner stays legal."""
    slab = make_slab(nx=8)
    cell = np.asarray(slab.get_cell(), dtype=float)
    probe = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    other = Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    r_sum = 1.0 / min_interadsorbate_covalent_ratio(
        probe, other, material_type="slab", cell=cell
    )
    centered = make_h2().get_positions()
    half_extent = float(np.max(centered[:, 0] - centered.mean(axis=0)[0]))
    config = AdsorptionConfig(placement_clash_descent=False)
    inner, _outer = lateral_contact_shell(config.connectivity_multiplier)
    clash = _h2_com_gap(inner / 2.0, r_sum=r_sum, half_extent=half_extent)
    step = 4.0
    y_shift = float(cell[1, 1]) / 2.0
    x0 = 2.0
    poses = []
    for placement_id, x_shift in enumerate(
        (x0, x0 + clash, x0 + clash + step, x0 + clash + 2.0 * step)
    ):
        poses.append(
            make_screening_result(
                molecule="h2",
                placement_id=placement_id,
                energy_adsorption=-1.0,
                atoms=place_molecule_on_slab(
                    slab, make_h2(), x_shift=x_shift, y_shift=y_shift
                ),
                slab_size=len(slab),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(
                    placement_id=placement_id
                ),
            )
        )

    rng = np.random.default_rng(0)
    exact_groups = assemble_joint_config_groups(
        poses,
        n_per_config=2,
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    quota_groups = assemble_quota_joint_configs(
        {"h2": poses},
        quotas={"h2": 2},
        n_configs=1,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    exact_ids = [[r.placement_id for r in group] for group in exact_groups]
    quota_ids = [[r.placement_id for r in group] for group in quota_groups]
    assert exact_ids == quota_ids
    assert exact_ids == [[0, 3]]


def test_full_catalog_draws_the_open_shell_before_a_neighbor():
    """Two graph steps stay two steps only while the intermediate site exists.

    Poses sit on a line of hollows at the outer lateral shell, so packing
    does not reject them. The far site is the open bin when the waypoint
    between it and the anchor is in the catalog, and a neighbor when it is not.
    """
    slab = make_slab(nx=8)
    cell = np.asarray(slab.get_cell(), dtype=float)
    config = AdsorptionConfig(placement_clash_descent=False)
    probe = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    other = Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    r_sum = 1.0 / min_interadsorbate_covalent_ratio(
        probe, other, material_type="slab", cell=cell
    )
    centered = make_h2().get_positions()
    half_extent = float(np.max(centered[:, 0] - centered[:, 0].mean()))
    _inner, outer = lateral_contact_shell(config.connectivity_multiplier)
    gap = _h2_com_gap(outer, r_sum=r_sum, half_extent=half_extent)
    y_shift = float(cell[1, 1]) / 2.0
    hops = (0, 1, 3)
    poses = [
        make_screening_result(
            molecule="h2",
            placement_id=placement_id,
            energy_adsorption=-1.0,
            atoms=place_molecule_on_slab(
                slab,
                make_h2(),
                x_shift=half_extent + placement_id * gap,
                y_shift=y_shift,
            ),
            slab_size=len(slab),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(
                placement_id=placement_id,
                site_index=hops[placement_id],
                site_type="hollow",
            ),
        )
        for placement_id in range(len(hops))
    ]
    n_line = hops[-1] + 1
    positions = {index: np.array([float(index), 0.0, 0.0]) for index in range(n_line)}
    rng = np.random.default_rng(0)

    def _sites(site_positions: dict[int, np.ndarray]) -> list[int]:
        groups = assemble_joint_config_groups(
            poses,
            n_per_config=len(hops) - 1,
            n_configs=1,
            slab_atoms=slab,
            config=config,
            rng=rng,
            n_substrate=len(slab),
            site_positions=site_positions,
            site_types={index: "hollow" for index in site_positions},
            material_type="slab",
        )
        assert len(groups) == 1
        return [int(pose.placement_descriptor.site_index) for pose in groups[0]]

    assert _sites(positions) == [hops[0], hops[-1]]
    assert _sites({index: positions[index] for index in hops}) == [hops[0], hops[1]]


def _h2_contact_layout(slab: Atoms, config: AdsorptionConfig):
    cell = np.asarray(slab.get_cell(), dtype=float)
    probe = Atoms("H", positions=[[0.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    other = Atoms("H", positions=[[1.0, 0.0, 0.0]], cell=cell, pbc=slab.get_pbc())
    r_sum = 1.0 / min_interadsorbate_covalent_ratio(
        probe, other, material_type="slab", cell=cell
    )
    centered = make_h2().get_positions()
    half_extent = float(np.max(centered[:, 0] - centered[:, 0].mean()))
    inner, outer = lateral_contact_shell(config.connectivity_multiplier)
    gap = _h2_com_gap((inner + outer) / 2.0, r_sum=r_sum, half_extent=half_extent)
    return cell, half_extent, gap


def _h2_com_gap(target_ratio: float, *, r_sum: float, half_extent: float) -> float:
    """COM separation whose facing H–H gap has intermolecular ratio *target_ratio*."""
    return target_ratio * r_sum + 2.0 * half_extent


def _partner_occupancy_count(
    group: list,
    positions: dict[int, np.ndarray],
    *,
    cell: np.ndarray,
    pbc: list[bool],
) -> int:
    """Occupied-neighbor count of the second member on the full site graph."""
    origin = int(group[0].placement_descriptor.site_index)
    other = int(group[1].placement_descriptor.site_index)
    return int(occupied_neighbor_counts(positions, {origin}, cell, pbc).get(other, 0))


def test_assemblers_span_close_and_far_partners() -> None:
    """Screened groups include a zero count and a positive count."""
    slab = make_slab(nx=8)
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = list(slab.get_pbc())
    config = AdsorptionConfig(placement_clash_descent=False)
    y_shift = float(cell[1, 1]) / 2.0
    step = 3.0
    x0 = 2.0
    poses = []
    for placement_id in range(5):
        poses.append(
            make_screening_result(
                molecule="h2",
                placement_id=placement_id,
                energy_adsorption=-1.0,
                atoms=place_molecule_on_slab(
                    slab,
                    make_h2(),
                    x_shift=x0 + placement_id * step,
                    y_shift=y_shift,
                ),
                slab_size=len(slab),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(
                    placement_id=placement_id
                ),
            )
        )

    rng = np.random.default_rng(0)
    exact_groups = assemble_joint_config_groups(
        poses,
        n_per_config=2,
        n_configs=2,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    quota_groups = assemble_quota_joint_configs(
        {"h2": poses},
        quotas={"h2": 2},
        n_configs=2,
        slab_atoms=slab,
        config=config,
        rng=rng,
        n_substrate=len(slab),
        material_type="slab",
    )
    exact_ids = [[r.placement_id for r in group] for group in exact_groups]
    quota_ids = [[r.placement_id for r in group] for group in quota_groups]
    assert exact_ids == quota_ids
    assert len(exact_groups) == 2
    positions = _site_positions(poses)
    counts = [
        _partner_occupancy_count(group, positions, cell=cell, pbc=pbc)
        for group in exact_groups
    ]
    assert 0 in counts
    assert any(count > 0 for count in counts)


def test_bo_assemble_second_group_takes_a_separated_site(monkeypatch):
    """Bayesian groups continue the occupancy-count round-robin."""
    slab = make_slab(nx=8)
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = list(slab.get_pbc())
    config = AdsorptionConfig(
        placement_clash_descent=False,
        saturation_molecules_per_step=2,
    )
    y_shift = float(cell[1, 1]) / 2.0
    stubs = {
        placement_id: make_screening_result(
            molecule="h2",
            placement_id=placement_id,
            energy_adsorption=-1.0,
            atoms=place_molecule_on_slab(
                slab,
                make_h2(),
                x_shift=2.0 + placement_id * 3.0,
                y_shift=y_shift,
            ),
            slab_size=len(slab),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=placement_id),
        )
        for placement_id in range(5)
    }

    def _fake_materialize(*, specs, **_k):
        pid = int(specs[0].placement_index)
        stub = stubs[pid]
        return MagicMock(
            combined=[stub.atoms],
            placement_ids=[pid],
            descriptors=[stub.placement_descriptor],
        )

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.materialize_specs",
        _fake_materialize,
    )
    specs = [MagicMock(placement_index=i) for i in range(len(stubs))]
    groups, _pool_sets = _assemble_bo_joint_groups_from_specs(
        [0, 2],
        all_specs=specs,
        valid_spec_indices=list(range(len(stubs))),
        materialization_cache={},
        conformers=[make_water()],
        slab=Substrate(slab, material_type="slab"),
        calculator=None,
        config=config,
        smiles="O",
        molecule_name="h2",
        site_context=None,
        slab_for_sites=slab,
        E_slab=0.0,
        E_mol=-1.0,
        n_per_config=2,
        n_substrate=len(slab),
    )
    assert len(groups) == 2
    positions = _site_positions(list(stubs.values()))
    counts = [
        _partner_occupancy_count(group, positions, cell=cell, pbc=pbc)
        for group in groups
    ]
    assert 0 in counts
    assert any(count > 0 for count in counts)


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
            current_slab=Substrate(slab, material_type="slab"),
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
    slab = make_slab(nx=8)
    config = AdsorptionConfig(saturation_molecules_per_step=2)
    cell, half_extent, gap = _h2_contact_layout(slab, config)
    y_shift = float(cell[1, 1]) / 2.0
    stubs = {
        i: make_screening_result(
            molecule="h2",
            placement_id=i,
            energy_adsorption=0.0,
            atoms=place_molecule_on_slab(
                slab, make_h2(), x_shift=half_extent + i * gap, y_shift=y_shift
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

    def _fake_pack(winners, _slab, _config, *, material_type="slab", n_substrate=None):
        del material_type
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
        "metalsurfer.workflow.composite.pack_exact_tuplet",
        _fake_pack,
    )

    specs = [MagicMock(placement_index=i) for i in range(4)]
    groups, pool_sets = _assemble_bo_joint_groups_from_specs(
        [0, 1],
        all_specs=specs,
        valid_spec_indices=list(range(4)),
        materialization_cache={},
        conformers=[make_water()],
        slab=Substrate(slab, material_type="slab"),
        calculator=None,
        config=config,
        smiles="O",
        molecule_name="water",
        site_context=None,
        slab_for_sites=slab,
        E_slab=0.0,
        E_mol=-14.0,
        n_per_config=2,
        n_substrate=len(slab),
    )
    assert len(groups) == 1
    assert [r.placement_id for r in groups[0]] == [1, 2]
    assert pool_sets == [{1, 2}]


def test_bo_assemble_respects_blocked_companions(monkeypatch):
    slab = make_slab(nx=8)
    config = AdsorptionConfig(saturation_molecules_per_step=2)
    cell, half_extent, gap = _h2_contact_layout(slab, config)
    y0 = float(cell[1, 1]) / 2.0
    x0 = float(cell[0, 0]) / 2.0
    shifts = (x0, x0 + gap, x0 - gap)
    stubs = {
        i: make_screening_result(
            molecule="h2",
            placement_id=i,
            energy_adsorption=0.0,
            atoms=place_molecule_on_slab(
                slab, make_h2(), x_shift=half_extent + shifts[i], y_shift=y0
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

    def _fake_pack(winners, _slab, _config, *, material_type="slab", n_substrate=None):
        del material_type
        return list(winners)

    monkeypatch.setattr(
        "metalsurfer.workflow.joint_tuplet.materialize_specs",
        _fake_materialize,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.composite.pack_exact_tuplet",
        _fake_pack,
    )

    specs = [MagicMock(placement_index=i) for i in range(3)]
    groups, pool_sets = _assemble_bo_joint_groups_from_specs(
        [0],
        all_specs=specs,
        valid_spec_indices=list(range(3)),
        materialization_cache={},
        conformers=[make_water()],
        slab=Substrate(slab, material_type="slab"),
        calculator=None,
        config=config,
        smiles="O",
        molecule_name="water",
        site_context=None,
        slab_for_sites=slab,
        E_slab=0.0,
        E_mol=-14.0,
        n_per_config=2,
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
        bo=BOConfig(initial_random=1, batch_size=1, total_batches=1),
    )
    captured: dict = {}

    ctx = MoleculeScreeningContext(
        slab=Substrate(slab, material_type="slab"),
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

    monkeypatch.setattr(
        "metalsurfer.workflow.bayesian.enumerate_placement_specs",
        _fake_enumerate,
    )

    out = process_joint_tuplet_bayesian(
        "O",
        "water",
        Substrate(slab, material_type="slab"),
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

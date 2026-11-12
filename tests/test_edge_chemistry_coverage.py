"""Edge motifs, read from materialized placement snapshots.

Regenerate the snapshots from the repository root with
``PYTHONPATH=. python tests/edge_placement_snapshots.py``.
These tests do not build conformers or place molecules. Pair and trimer
checks assemble the joint configs that budget would screen.
"""

from __future__ import annotations

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import lateral_contact_shell, min_interadsorbate_covalent_ratio
from metalsurfer.models import ScreeningResult
from metalsurfer.placement.geometry import occupied_neighbor_counts
from metalsurfer.workflow.composite import assemble_joint_config_groups
from tests.edge_chemistry_helpers import (
    carboxyl_down,
    has_hbond_contact,
    p_nearer_au_than_carbons,
)
from tests.edge_placement_snapshots import (
    frame_atoms,
    joint_pose_pool_size,
    load_snapshot,
    pose_atoms,
    screening_result,
)


def _placements(snapshot: dict[str, object]) -> list[dict[str, object]]:
    placements = snapshot["placements"]
    assert isinstance(placements, list)
    return placements


def test_pme3_au4_first_placements_point_phosphorus_at_gold() -> None:
    """Among the stored Au4 placements, some put P nearer Au than C."""
    snapshot = load_snapshot("pme3_au4_100")
    slab = frame_atoms(snapshot["slab"])
    au_pos = np.asarray(slab.get_positions(), dtype=float)
    hits = sum(
        p_nearer_au_than_carbons(pose_atoms(snapshot, pose), au_pos)
        for pose in _placements(snapshot)
    )
    assert hits == 95


def test_pyridone_sequential_first_placements_can_hbond() -> None:
    """Stored step-2 pyridone poses include an NH···O contact and a neighbor site."""
    snapshot = load_snapshot("pyridone_step2_100")
    anchor = frame_atoms(snapshot["anchor"])
    cell = np.asarray(anchor.get_cell(), dtype=float)
    placements = _placements(snapshot)
    hits = sum(
        has_hbond_contact(anchor, pose_atoms(snapshot, pose), cell=cell)
        for pose in placements
    )
    assert hits == 4
    catalog = snapshot["catalog"]
    assert isinstance(catalog, dict)
    explored = {int(pose["descriptor"]["site_index"]) for pose in placements}
    assert len(explored & set(catalog["positive_sites"])) == 51
    assert len(explored & set(catalog["open_atops"])) == 19


def _catalog_positions(snapshot: dict[str, object]) -> dict[int, np.ndarray]:
    raw = snapshot["catalog_positions"]
    assert isinstance(raw, list)
    return {index: np.asarray(xyz, dtype=float) for index, xyz in enumerate(raw)}


def _joint_groups(snapshot: dict[str, object]) -> list[list[ScreeningResult]]:
    """Groups ``assemble_joint_config_groups`` would hand to relaxation."""
    n_per = int(snapshot["n_per_config"])
    n_configs = int(snapshot["n_configs"])
    placements = _placements(snapshot)
    assert len(placements) == joint_pose_pool_size(n_per, n_configs)
    poses = [screening_result(snapshot, pose) for pose in placements]
    slab = frame_atoms(snapshot["slab"])
    config = AdsorptionConfig(
        seed=42,
        device="cpu",
        slab_relaxation_mode="none",
        num_conformers=4,
        num_placements=len(placements),
        saturation_molecules_per_step=n_per,
    )
    groups = assemble_joint_config_groups(
        poses,
        n_per_config=n_per,
        n_configs=n_configs,
        slab_atoms=slab,
        config=config,
        rng=np.random.default_rng(config.seed),
        n_substrate=len(slab),
        site_positions=_catalog_positions(snapshot),
        material_type="slab",
    )
    assert len(groups) == n_configs
    return groups


def _shell_hbond_edges(
    group: list[ScreeningResult],
    cell: np.ndarray,
    inner: float,
    outer: float,
) -> int:
    mols = [pose.atoms[pose.slab_size :] for pose in group]
    hits = 0
    for index, left in enumerate(mols):
        for right in mols[index + 1 :]:
            ratio = min_interadsorbate_covalent_ratio(
                left, right, material_type="slab", cell=cell
            )
            if inner < ratio <= outer and has_hbond_contact(left, right, cell=cell):
                hits += 1
    return hits


def _later_member_counts(
    group: list[ScreeningResult],
    positions: dict[int, np.ndarray],
    cell: np.ndarray,
    pbc: list[bool],
) -> list[int]:
    sites = [int(pose.placement_descriptor.site_index) for pose in group]
    counts = occupied_neighbor_counts(positions, {sites[0]}, cell, pbc)
    return [int(counts.get(site, 0)) for site in sites[1:]]


def _screened_hbond_edges(snapshot: dict[str, object]) -> list[int]:
    """In-shell NH···O contacts in the groups this snapshot's budget screens."""
    slab = frame_atoms(snapshot["slab"])
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = [bool(flag) for flag in slab.get_pbc()]
    positions = _catalog_positions(snapshot)
    groups = _joint_groups(snapshot)
    counts = [
        count
        for group in groups
        for count in _later_member_counts(group, positions, cell, pbc)
    ]
    assert 0 in counts
    assert any(count > 0 for count in counts)
    inner, outer = lateral_contact_shell(AdsorptionConfig().connectivity_multiplier)
    return [_shell_hbond_edges(group, cell, inner, outer) for group in groups]


@pytest.mark.slow
def test_pyridone_pair_budget_assembles_an_hbond() -> None:
    """Pair configs from the full-lattice pool include an NH···O contact."""
    edges = _screened_hbond_edges(load_snapshot("pyridone_pair"))
    assert edges.count(1) == 2


@pytest.mark.slow
def test_pyridone_trimer_budget_assembles_a_connected_hbond() -> None:
    """Trimer configs include one pack with two NH···O contacts."""
    edges = _screened_hbond_edges(load_snapshot("pyridone_trimer"))
    assert edges.count(2) == 1
    assert edges.count(1) == 2


def test_oleic_first_placements_point_carboxyl_down() -> None:
    """Some stored oleic placements point a carboxyl oxygen at the metal."""
    snapshot = load_snapshot("oleic_100")
    slab = frame_atoms(snapshot["slab"])
    slab_pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    hits = sum(
        carboxyl_down(pose_atoms(snapshot, pose), slab_pos, cell)
        for pose in _placements(snapshot)
    )
    assert hits == 93

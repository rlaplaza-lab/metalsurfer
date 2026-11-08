"""Short-budget edge placements, read from materialized snapshots.

These checks replace the saturation smokes. They do not relax or call a
calculator. Regenerate from the repository root with
``PYTHONPATH=. python tests/edge_placement_snapshots.py``.
"""

from __future__ import annotations

import numpy as np

from metalsurfer.config import AdsorptionConfig
from tests.edge_chemistry_helpers import (
    _OLEIC_FORMULA,
    _PME3_FORMULA,
    _PYRIDONE_FORMULA,
    assert_intact,
    carboxyl_down,
    p_nearer_au_than_carbons,
)
from tests.edge_placement_snapshots import frame_atoms, load_snapshot, pose_atoms


def _config(material_type: str) -> AdsorptionConfig:
    return AdsorptionConfig(
        material_type=material_type,
        seed=42,
        device="cpu",
        slab_relaxation_mode="none",
    )


def _placements(snapshot: dict[str, object]) -> list[dict[str, object]]:
    placements = snapshot["placements"]
    assert isinstance(placements, list)
    return placements


def _assert_neighbor_snapshot(snapshot: dict[str, object]) -> None:
    catalog = snapshot["catalog"]
    assert isinstance(catalog, dict)
    explored = {int(pose["descriptor"]["site_index"]) for pose in _placements(snapshot)}
    assert explored & set(catalog["positive_sites"])
    open_atops = set(catalog["open_atops"])
    if open_atops:
        assert explored & open_atops


def test_pme3_au4_short_draw_keeps_every_site_type() -> None:
    """Eight Au4 placements cover atop, bridge, and hollow, with P toward gold."""
    snapshot = load_snapshot("pme3_au4_8")
    slab = frame_atoms(snapshot["slab"])
    config = _config("nanoparticle")
    poses = _placements(snapshot)
    explored_types = {str(pose["descriptor"]["site_type"]) for pose in poses}
    assert {"atop", "bridge", "hollow"} <= explored_types
    au_pos = np.asarray(slab.get_positions(), dtype=float)
    assert any(
        p_nearer_au_than_carbons(pose_atoms(snapshot, pose), au_pos) for pose in poses
    )
    for pose in poses:
        assert_intact(pose_atoms(snapshot, pose), _PME3_FORMULA, config)


def test_pyridone_step2_short_draw_explores_a_neighbor_site() -> None:
    """Sixteen covered pyridone poses include a neighbor count and an open atop.

    The NH···O geometry is checked on the 100-placement snapshot.
    """
    snapshot = load_snapshot("pyridone_step2_16")
    config = _config("slab")
    poses = _placements(snapshot)
    for pose in poses:
        assert_intact(pose_atoms(snapshot, pose), _PYRIDONE_FORMULA, config)
    _assert_neighbor_snapshot(snapshot)


def test_pyridone_pool_short_draw_reaches_contact_height() -> None:
    """Sixteen bare pyridone poses stay intact and include the contact height.

    The packed NH···O dimer is the pair-budget assembly check.
    """
    snapshot = load_snapshot("pyridone_pool_16")
    config = _config("slab")
    poses = _placements(snapshot)
    for pose in poses:
        assert_intact(pose_atoms(snapshot, pose), _PYRIDONE_FORMULA, config)
    assert any(float(pose["descriptor"]["z_fraction"]) <= 0.5 + 1e-9 for pose in poses)


def test_oleic_step2_short_draw_has_a_carboxyl_down_neighbor() -> None:
    """One of the sixteen covered oleic poses points the carboxyl at a neighbor site."""
    snapshot = load_snapshot("oleic_step2_16")
    config = _config("slab")
    slab = frame_atoms(snapshot["slab"])
    slab_pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    catalog = snapshot["catalog"]
    assert isinstance(catalog, dict)
    positive = {int(site) for site in catalog["positive_sites"]}
    poses = _placements(snapshot)
    assert any(
        carboxyl_down(pose_atoms(snapshot, pose), slab_pos, cell)
        and int(pose["descriptor"]["site_index"]) in positive
        for pose in poses
    )
    for pose in poses:
        assert_intact(pose_atoms(snapshot, pose), _OLEIC_FORMULA, config)
    _assert_neighbor_snapshot(snapshot)

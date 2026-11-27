"""Unit tests for consolidation helpers introduced in the cleanup pass."""

import logging

import numpy as np
from ase import Atoms

from metalsurfer._logging import log_prefix_context
from metalsurfer.optimization import _deps
from metalsurfer.placement._cache_key import BoundedFifoCache
from metalsurfer.placement.pose.context import (
    _PoseBatchCache,
    pose_cache_cell,
    pose_cache_positions,
)
from metalsurfer.result_paths import (
    FINAL_SATURATED_SLAB_XYZ,
    conformer_adsorbate_xyz_name,
    conformer_dir_name,
    conformer_poscar_relpath,
    conformer_xyz_name,
    step_placements_dirname,
)


def test_missing_torch_sim_message():
    err = _deps.missing_torch_sim("test_feature")
    assert err.dependency == "torch-sim-atomistic"
    assert "pip install torch-sim-atomistic" in str(err)


def test_result_paths_leaf_names():
    assert conformer_xyz_name(7) == "conformer_007.xyz"
    assert conformer_adsorbate_xyz_name(7) == "conformer_007_adsorbate.xyz"
    assert conformer_dir_name(7) == "conformer_007"
    assert conformer_poscar_relpath(7).as_posix() == "conformer_007/POSCAR"
    assert step_placements_dirname(12) == "step_012_placements"
    assert FINAL_SATURATED_SLAB_XYZ == "final_saturated_slab.xyz"


def test_bounded_fifo_cache_evicts_oldest():
    cache: BoundedFifoCache[int] = BoundedFifoCache(2)
    cache.store("a", 1)
    cache.store("b", 2)
    cache.store("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_pose_cache_helpers():
    slab = Atoms("Cu", positions=[[0, 0, 0]], cell=[[4, 0, 0], [0, 4, 0], [0, 0, 20]])
    assert np.allclose(pose_cache_cell(None, slab), slab.get_cell())
    assert np.allclose(pose_cache_positions(None, slab), slab.get_positions())
    cell = np.eye(3) * 5.0
    positions = np.array([[1.0, 2.0, 3.0]])
    cache = _PoseBatchCache(cell=cell, positions=positions)
    assert pose_cache_cell(cache, slab) is cell
    assert pose_cache_positions(cache, slab) is positions


def test_log_prefix_context_sets_phase(caplog):
    caplog.set_level(logging.INFO)
    logger = logging.getLogger("metalsurfer.test_log_prefix_ctx")
    with log_prefix_context("Saturation for water | step 2 | "):
        logger.info("slab saturated at step %d", 2)
    prefixes = [getattr(r, "ctx_prefix", "") for r in caplog.records]
    assert any("phase=Saturation for water" in p for p in prefixes)

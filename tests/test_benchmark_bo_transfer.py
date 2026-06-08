"""Tests for offline BO transfer benchmark utilities."""

import sys
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd

from metalsurfer.ml.bayesian import build_transfer_surrogate
from metalsurfer.models import BOStepMemory


def _load_common_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_bo_common.py"
    )
    spec = util.spec_from_file_location("benchmark_bo_common", script_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules["benchmark_bo_common"] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_pool(
    n: int, energy_offset: float = 0.0
) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "x": float(i) * 0.1,
                "y": float(i % 3) * 0.2,
                "z": 1.0,
                "conformer_index": float(i % 2),
                "quat_w": 1.0,
                "quat_x": 0.0,
                "quat_y": 0.0,
                "quat_z": 0.0,
                "energy": -1.0 - 0.05 * i + energy_offset,
            }
        )
    df = pd.DataFrame(rows)
    X = df.drop(columns=["energy"])
    y = df["energy"]
    return X, y


def test_build_transfer_surrogate_uses_prior_when_similar():
    X, y = _synthetic_pool(20)
    X_prev = X.iloc[:10].copy()
    y_prev = (y.iloc[:10] - 0.5).to_numpy()
    result = build_transfer_surrogate(
        X.iloc[:8],
        y.iloc[:8].to_numpy(),
        X_prev,
        y_prev,
        weight_cap=0.35,
        similarity_lengthscale=1.0,
        min_similarity=0.0,
        mae_tolerance=1.0,
    )
    assert result.surrogate is not None
    assert result.transfer_weight_share > 0.0


def test_build_transfer_surrogate_trust_gate_falls_back_to_baseline():
    X, y = _synthetic_pool(20)
    X_prev = X.iloc[:10].copy()
    y_prev = np.full(10, 100.0)
    bad_rounds = 0
    disabled = False
    for _ in range(3):
        result = build_transfer_surrogate(
            X.iloc[:8],
            y.iloc[:8].to_numpy(),
            X_prev,
            y_prev,
            weight_cap=0.35,
            similarity_lengthscale=1.0,
            min_similarity=0.0,
            mae_tolerance=0.0,
            transfer_bad_rounds=bad_rounds,
            trust_patience=2,
        )
        bad_rounds = result.transfer_bad_rounds
        disabled = result.transfer_disabled
    assert disabled is True
    assert result.transfer_disabled_reason == "trust_degraded_on_current_step_residuals"


def test_run_offline_bo_with_transfer_completes_two_steps():
    mod = _load_common_module()
    X1, y1 = _synthetic_pool(30, energy_offset=0.0)
    X2, y2 = _synthetic_pool(30, energy_offset=0.3)

    baseline1, memory1, _ = mod.run_offline_bo_with_transfer(
        X1,
        y1,
        initial_random=5,
        batch_size=5,
        total_budget=15,
        seed=1,
        transfer_enabled=False,
    )
    transfer2, _, info = mod.run_offline_bo_with_transfer(
        X2,
        y2,
        initial_random=5,
        batch_size=5,
        total_budget=15,
        seed=1,
        prior_memory=memory1,
        transfer_enabled=True,
    )
    assert np.isfinite(baseline1)
    assert np.isfinite(transfer2)
    assert isinstance(memory1, BOStepMemory)
    assert len(memory1.observed_X_rows) > 0
    assert info["transfer_weight_share_mean"] >= 0.0

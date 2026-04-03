"""Tests for BO benchmark setup-specific validation and injectivity checks."""

from importlib import util
from pathlib import Path

import pandas as pd
import pytest


def _load_benchmark_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "benchmark_bo_models.py"
    )
    spec = util.spec_from_file_location("benchmark_bo_models", script_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _base_row() -> dict[str, object]:
    return {
        "molecule": "co2",
        "placement_id": 0,
        "conformer_index": 0,
        "orientation_type": "round",
        "face_flip": False,
        "site_index": 0,
        "site_type": "atop",
        "tilt_deg": 0.0,
        "azimuth_deg": 0.0,
        "azimuth_in_plane_deg": 0.0,
        "z_fraction": 0.5,
        "x_abs": 0.0,
        "y_abs": 0.0,
        "z_abs": 2.5,
        "z_offset": 2.5,
        "quat_w": 1.0,
        "quat_x": 0.0,
        "quat_y": 0.0,
        "quat_z": 0.0,
        "shape": "round",
        "energy_adsorption": -0.8,
        "smiles": "O=C=O",
        "surface_id": "co2_graphene",
        "context_hash": "abc123",
    }


def test_load_and_prepare_data_rejects_mixed_setup(tmp_path):
    mod = _load_benchmark_module()
    row1 = _base_row()
    row2 = _base_row()
    row2["placement_id"] = 1
    row2["x_abs"] = 1.0
    row2["surface_id"] = "other_surface"

    df = pd.DataFrame([row1, row2])
    df.to_csv(tmp_path / "adsorption_energies_detailed.csv", index=False)

    with pytest.raises(ValueError, match="setup-specific"):
        mod.load_and_prepare_data(str(tmp_path))


def test_load_and_prepare_data_accepts_single_setup(tmp_path):
    mod = _load_benchmark_module()
    row1 = _base_row()
    row2 = _base_row()
    row2["placement_id"] = 1
    row2["x_abs"] = 1.0
    row2["energy_adsorption"] = -0.7

    df = pd.DataFrame([row1, row2])
    df.to_csv(tmp_path / "adsorption_energies_detailed.csv", index=False)

    X, y, loaded = mod.load_and_prepare_data(str(tmp_path))
    assert len(X) == 2
    assert len(y) == 2
    assert len(loaded) == 2


def test_feature_energy_injective_allows_duplicates_with_same_energy():
    mod = _load_benchmark_module()
    X = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 1.0]})
    y = pd.Series([-1.0, -1.0])

    mod._assert_feature_energy_injective(X, y)


def test_feature_energy_injective_reports_conflicts():
    mod = _load_benchmark_module()
    X = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 1.0]})
    y = pd.Series([-1.0, -0.9])

    with pytest.raises(ValueError, match="Example conflicts"):
        mod._assert_feature_energy_injective(X, y)

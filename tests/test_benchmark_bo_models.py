"""Tests for BO benchmark setup-specific validation and injectivity checks."""

import sys
from importlib import util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _load_script_module(name: str, script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = util.spec_from_file_location(name, script_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_benchmark_module():
    return _load_script_module("benchmark_bo_models", "benchmark_bo_models.py")


def _load_common_module():
    return _load_script_module("benchmark_bo_common", "benchmark_bo_common.py")


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
    mod = _load_common_module()
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
    mod = _load_common_module()
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
    mod = _load_common_module()
    X = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 1.0]})
    y = pd.Series([-1.0, -1.0])

    mod._assert_feature_energy_injective(X, y)


def test_feature_energy_injective_reports_conflicts():
    mod = _load_common_module()
    X = pd.DataFrame({"x": [0.0, 0.0], "y": [1.0, 1.0]})
    y = pd.Series([-1.0, -0.9])

    with pytest.raises(ValueError, match="Example conflicts"):
        mod._assert_feature_energy_injective(X, y)


def _hpc_schema_row(**overrides: object) -> dict[str, object]:
    """Row shape from older HPC exports (geometry + tilt, no quat columns)."""
    row: dict[str, object] = {
        "molecule": "bipyridine",
        "placement_id": 0,
        "conformer_index": 0,
        "orientation_type": "flat",
        "face_flip": False,
        "site_index": 0,
        "site_type": "atop",
        "tilt_deg": 15.0,
        "azimuth_deg": 45.0,
        "azimuth_in_plane_deg": 0.0,
        "x_abs": 1.0,
        "y_abs": 2.0,
        "z_offset": 2.5,
        "z_abs": 12.5,
        "energy_adsorption": -0.5,
    }
    row.update(overrides)
    return row


def test_load_and_prepare_data_accepts_hpc_schema_without_quats(tmp_path):
    mod = _load_common_module()
    df = pd.DataFrame(
        [
            _hpc_schema_row(),
            _hpc_schema_row(
                placement_id=1,
                x_abs=3.0,
                energy_adsorption=-0.6,
            ),
        ]
    )
    df.to_csv(tmp_path / "adsorption_energies_detailed.csv", index=False)

    X, y, loaded = mod.load_and_prepare_data(
        str(tmp_path),
        surface_type="bipyridine_au111_defects_saturation_raw",
        smiles="n1ccccc1-c2ccccn2",
    )
    assert len(X) == 2
    assert len(y) == 2
    assert len(loaded) == 2
    assert set(X.columns) == {
        "x",
        "y",
        "z",
        "conformer_index",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    }


def test_enrich_detailed_dataset_geometry_from_ml_dataset(tmp_path):
    from metalsurfer.ml.dataset import enrich_detailed_dataset_geometry

    detailed = pd.DataFrame(
        [
            _hpc_schema_row(placement_id=0, tilt_deg=10.0),
            _hpc_schema_row(placement_id=1, tilt_deg=20.0, x_abs=3.0),
        ]
    )
    ml = pd.DataFrame(
        [
            {
                "placement_id": 0,
                "conformer_index": 0,
                "tilt_deg": 10.0,
                "azimuth_deg": 45.0,
                "azimuth_in_plane_deg": 0.0,
                "face_flip": False,
                "quat_w": 0.8,
                "quat_x": 0.1,
                "quat_y": 0.2,
                "quat_z": 0.3,
                "z_fraction": 0.25,
            }
        ]
    )
    ml.to_csv(tmp_path / "ml_dataset.csv", index=False)

    enriched = enrich_detailed_dataset_geometry(detailed, data_dir=tmp_path)
    assert enriched.loc[0, "quat_w"] == pytest.approx(0.8)
    assert enriched.loc[0, "z_fraction"] == pytest.approx(0.25)
    assert enriched.loc[1, "quat_w"] == pytest.approx(1.0)


def test_load_placement_pool_filters_by_step_column(tmp_path):
    mod = _load_common_module()
    rows = []
    for step in (1, 2):
        for pid in range(3):
            row = _base_row()
            row.update(
                {
                    "step": step,
                    "placement_id": pid,
                    "x_abs": float(pid + step),
                    "energy_adsorption": -0.5 - 0.1 * pid - 0.01 * step,
                }
            )
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        tmp_path / "saturation_placements_detailed.csv", index=False
    )
    X, y, df = mod.load_placement_pool(str(tmp_path), step=2)
    assert len(df) == 3
    assert len(X) == 3


def test_run_random_search_and_bo_smoke(tmp_path):
    mod = _load_common_module()
    rows = []
    for pid in range(15):
        row = _base_row()
        row.update(
            {
                "placement_id": pid,
                "x_abs": float(pid),
                "energy_adsorption": -0.2 - 0.05 * pid,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        tmp_path / "adsorption_energies_detailed.csv", index=False
    )
    X, y, _ = mod.load_placement_pool(str(tmp_path))
    _, random_best = mod.run_random_search(X, y, 5, 5, 10, seed=0)
    _, bo_best = mod.run_offline_bo(X, y, 5, 5, 10, seed=0)
    assert np.isfinite(random_best)
    assert np.isfinite(bo_best)

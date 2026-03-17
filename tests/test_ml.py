"""Tests for the metalsurfer.ml binding energy regression pipeline."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from metalsurfer.ml.dataset import DatasetLogger, load_dataset, merge_datasets
from metalsurfer.ml.features import (
    extract_features,
    extract_features_from_dataset,
    get_feature_names,
)
from metalsurfer.ml.predict import BindingEnergyPredictor, PredictionResult
from metalsurfer.ml.regression import (
    evaluate_model,
    feature_importance,
    grouped_cross_validate,
    load_model,
    save_model,
    train_model,
)
from metalsurfer.ml.reproduce import record_to_config, record_to_placement_descriptor
from metalsurfer.ml.schema import ComputationContext, PlacementRecord
from metalsurfer.models import ScreeningResult


def _make_record(
    i: int = 0,
    molecule: str = "ethanol",
    smiles: str = "CCO",
    energy: float = -0.85,
) -> PlacementRecord:
    return PlacementRecord(
        molecule=molecule,
        smiles=smiles,
        surface_id="Cu_fcc111",
        placement_id=i,
        conformer_index=i % 3,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=i % 5,
        site_type="atop",
        tilt_deg=15.0,
        azimuth_deg=45.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        x=float(1.0 + i * 0.1),
        y=float(2.0 - i * 0.1),
        z=2.5,
        shape="round",
        energy_adsorption=energy,
        energy_adslab=-150.0 + energy,
        energy_slab=-145.0,
        energy_adsorbate=-5.0,
        distance=2.3,
        context=ComputationContext(),
    )


def _make_synthetic_dataset(n: int = 80) -> list[PlacementRecord]:
    rng = np.random.RandomState(42)
    records = []
    mols = ["ethanol", "methanol", "water", "CO"]
    smiles_map = {"ethanol": "CCO", "methanol": "CO", "water": "O", "CO": "[C-]#[O+]"}
    for i in range(n):
        mol = mols[i % len(mols)]
        z = float(rng.uniform(2, 3))
        tilt = float(rng.choice([0, 15, 30, 45, 60, 90]))
        e_ads = -0.5 * z + 0.01 * tilt + float(rng.normal(0, 0.1))
        records.append(
            PlacementRecord(
                molecule=mol,
                smiles=smiles_map[mol],
                surface_id="Cu_fcc111",
                placement_id=i,
                conformer_index=i % 3,
                orientation_type=str(
                    rng.choice(["parallel", "EN-down", "vertical", "round"])
                ),
                face_flip=False,
                en_atom_index=None,
                site_index=i % 5,
                site_type=str(rng.choice(["atop", "bridge", "hollow", "envelope"])),
                tilt_deg=tilt,
                azimuth_deg=float(rng.choice([0, 45, 90, 135, 180, 225, 270, 315])),
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                x=float(rng.uniform(-4, 4)),
                y=float(rng.uniform(-4, 4)),
                z=z,
                shape=str(rng.choice(["linear", "flat", "round"])),
                energy_adsorption=e_ads,
                energy_adslab=float(-150 + e_ads),
                energy_slab=-145.0,
                energy_adsorbate=-5.0,
                distance=2.3,
                context=ComputationContext(),
            )
        )
    return records


# ── Schema tests ──


class TestComputationContext:
    def test_from_config(self):
        from metalsurfer.config import AdsorptionConfig

        config = AdsorptionConfig(model_name="test-model", fmax=0.03, seed=123)
        ctx = ComputationContext.from_config(config)
        assert ctx.model_name == "test-model"
        assert ctx.fmax == 0.03
        assert ctx.seed == 123

    def test_settings_hash_deterministic(self):
        ctx1 = ComputationContext()
        ctx2 = ComputationContext()
        assert ctx1.settings_hash() == ctx2.settings_hash()

    def test_settings_hash_changes(self):
        ctx1 = ComputationContext(model_name="a")
        ctx2 = ComputationContext(model_name="b")
        assert ctx1.settings_hash() != ctx2.settings_hash()

    def test_to_dict_roundtrip(self):
        ctx = ComputationContext(fmax=0.02, seed=99)
        d = ctx.to_dict()
        assert d["fmax"] == 0.02
        assert d["seed"] == 99
        assert isinstance(d["placement_z_range"], list)


class TestPlacementRecord:
    def test_record_hash_deterministic(self):
        r1 = _make_record(0)
        r2 = _make_record(0)
        assert r1.record_hash() == r2.record_hash()

    def test_record_hash_changes_with_position(self):
        r1 = _make_record(0)
        r2 = _make_record(1)
        assert r1.record_hash() != r2.record_hash()

    def test_to_flat_dict_keys(self):
        r = _make_record()
        flat = r.to_flat_dict()
        assert "record_hash" in flat
        assert "molecule" in flat
        assert "energy_adsorption" in flat
        assert "context_hash" in flat
        assert "model_name" in flat

    def test_flat_dict_roundtrip(self):
        r = _make_record(42, energy=-1.23)
        flat = r.to_flat_dict()
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.molecule == r.molecule
        assert r2.placement_id == r.placement_id
        assert abs(r2.energy_adsorption - r.energy_adsorption) < 1e-10
        assert r2.tilt_deg == r.tilt_deg
        assert r2.context.model_name == r.context.model_name

    def test_from_screening_result_returns_record_with_descriptor(self):
        from metalsurfer.models import PlacementDescriptor

        descriptor = PlacementDescriptor(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=0,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=0,
            x=0.0,
            y=0.0,
            z=2.5,
            shape="round",
            slab_indices=None,
        )
        result = ScreeningResult(
            molecule="test",
            placement_id=0,
            energy_adslab=-150.0,
            energy_slab=-145.0,
            energy_adsorbate=-5.0,
            energy_adsorption=-0.5,
            atoms=Atoms("H"),
            distance=2.0,
            placement_descriptor=descriptor,
        )
        record = PlacementRecord.from_screening_result(
            result, smiles="C", surface_id="test"
        )
        assert record is not None
        assert record.molecule == "test"
        assert record.placement_id == 0
        assert record.energy_adsorption == -0.5


# ── Dataset tests ──


class TestDatasetLogger:
    def test_flush_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(_make_record(0))
            ds.add_record(_make_record(1))
            path = ds.flush()
            assert os.path.exists(path)
            df = pd.read_csv(path)
            assert len(df) == 2

    def test_flush_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir)
            ds1.add_record(_make_record(0))
            ds1.flush()

            ds2 = DatasetLogger(tmpdir)
            ds2.add_record(_make_record(1))
            ds2.flush()

            df = pd.read_csv(ds2.csv_path)
            assert len(df) == 2

    def test_flush_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir)
            ds1.add_record(_make_record(0))
            ds1.flush()

            ds2 = DatasetLogger(tmpdir)
            ds2.add_record(_make_record(0))  # same record
            ds2.flush()

            df = pd.read_csv(ds2.csv_path)
            assert len(df) == 1

    def test_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir, surface_id="test")
            ds.add_record(_make_record())
            ds.flush()
            assert os.path.exists(ds.metadata_path)


class TestLoadDataset:
    def test_load_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(_make_record(0))
            ds.flush()
            df = load_dataset(tmpdir)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1

    def test_load_as_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(_make_record(0))
            ds.flush()
            records = load_dataset(tmpdir, as_records=True)
            assert isinstance(records, list)
            assert isinstance(records[0], PlacementRecord)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_dataset("/nonexistent/path")


class TestMergeDatasets:
    def test_merge_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = os.path.join(tmpdir, "a")
            dir2 = os.path.join(tmpdir, "b")
            ds1 = DatasetLogger(dir1)
            ds1.add_record(_make_record(0))
            ds1.add_record(_make_record(1))
            ds1.flush()
            ds2 = DatasetLogger(dir2)
            ds2.add_record(_make_record(1))  # duplicate
            ds2.add_record(_make_record(2))
            ds2.flush()

            merged = merge_datasets(dir1, dir2)
            assert len(merged) == 3


# ── Feature tests ──


class TestFeatureExtraction:
    def test_feature_count(self):
        r = _make_record()
        features = extract_features(r)
        assert len(features) == 27

    def test_angle_encoding(self):
        r = _make_record()
        r.tilt_deg = 0.0
        features = extract_features(r)
        assert abs(features["tilt_sin"]) < 1e-6
        assert abs(features["tilt_cos"] - 1.0) < 1e-6

    def test_one_hot_orientation(self):
        r = _make_record()
        r.orientation_type = "parallel"
        features = extract_features(r)
        assert features["orient_parallel"] == 1.0
        assert features["orient_EN-down"] == 0.0

    def test_extract_from_dataset(self):
        records = _make_synthetic_dataset(20)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            X, y = extract_features_from_dataset(df)
            assert X.shape[0] == 20
            assert len(y) == 20

    def test_feature_names_consistent(self):
        names = get_feature_names()
        r = _make_record()
        features = extract_features(r)
        assert list(features.keys()) == names


# ── Regression tests ──


class TestRegression:
    @pytest.fixture()
    def dataset(self):
        records = _make_synthetic_dataset(80)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            X, y = extract_features_from_dataset(df)
            yield X, y, df

    def test_train_ridge(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="ridge")
        metrics = evaluate_model(model, X, y)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"]

    def test_train_random_forest(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="random_forest")
        metrics = evaluate_model(model, X, y)
        assert metrics["r2"] > 0

    def test_train_gradient_boost(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="gradient_boost")
        metrics = evaluate_model(model, X, y)
        assert metrics["r2"] > 0.5

    def test_grouped_cv(self, dataset):
        X, y, df = dataset
        result = grouped_cross_validate(
            X, y, groups=df["molecule"], model_type="ridge", n_splits=4
        )
        assert "mean_mae" in result
        assert "fold_metrics" in result
        assert len(result["fold_metrics"]) == 4

    def test_feature_importance_rf(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="random_forest")
        fi = feature_importance(model, list(X.columns), top_k=5)
        assert len(fi) == 5
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_permutation(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="gradient_boost")
        fi = feature_importance(model, list(X.columns), X=X, y=y, top_k=5)
        assert len(fi) == 5

    def test_save_load_model(self, dataset):
        X, y, _ = dataset
        model = train_model(X, y, model_type="ridge")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, "ridge", feature_names=list(X.columns))
            loaded, meta = load_model(tmpdir)
            assert meta["model_type"] == "ridge"
            y_pred = loaded.predict(X)
            assert len(y_pred) == len(y)

    def test_invalid_model_type(self, dataset):
        X, y, _ = dataset
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X, y, model_type="invalid")


# ── Prediction tests ──


class TestPredictor:
    @pytest.fixture()
    def predictor(self):
        records = _make_synthetic_dataset(80)
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            X, y = extract_features_from_dataset(df)
            model = train_model(X, y, model_type="gradient_boost")
            metadata = {
                "model_type": "gradient_boost",
                "feature_names": list(X.columns),
            }
            yield BindingEnergyPredictor(model, metadata=metadata), records

    def test_predict_single(self, predictor):
        pred, records = predictor
        result = pred.predict_record(records[0])
        assert isinstance(result, PredictionResult)
        assert isinstance(result.energy, float)

    def test_predict_batch(self, predictor):
        pred, records = predictor
        results = pred.predict_batch(records[:10])
        assert len(results) == 10

    def test_rank_placements(self, predictor):
        pred, records = predictor
        ranked = pred.rank_placements(records[:20], top_k=5)
        assert len(ranked) == 5
        energies = [p.energy for _, p in ranked]
        assert energies == sorted(energies)

    def test_predict_descriptor(self, predictor):
        pred, records = predictor
        descriptor = record_to_placement_descriptor(records[0])
        result = pred.predict_descriptor(descriptor, molecule="ethanol", smiles="CCO")
        assert isinstance(result.energy, float)


# ── Reproduce tests ──


class TestReproduce:
    def test_record_to_descriptor(self):
        r = _make_record()
        d = record_to_placement_descriptor(r)
        assert d.conformer_index == r.conformer_index
        assert d.tilt_deg == r.tilt_deg
        assert d.x == r.x

    def test_record_to_config(self):
        r = _make_record()
        config = record_to_config(r)
        assert config.model_name == r.context.model_name
        assert config.fmax == r.context.fmax
        assert config.seed == r.context.seed

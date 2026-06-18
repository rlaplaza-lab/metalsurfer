"""Tests for the metalsurfer.ml binding energy regression pipeline."""

import os
import tempfile
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from numpy.testing import assert_allclose
from scipy import stats

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.bayesian import ei_scores, lcb_scores, pi_scores
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
from metalsurfer.models import PlacementDescriptor, ScreeningResult
from tests.factories import make_random_placement_records


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
        x_abs=float(1.0 + i * 0.1),
        y_abs=float(2.0 - i * 0.1),
        z_offset=2.5,
        surface_ref_z_abs=10.0,
        z_abs=12.5,
        shape="round",
        energy_adsorption=energy,
        energy_adslab=-150.0 + energy,
        energy_slab=-145.0,
        energy_adsorbate=-5.0,
        distance=2.3,
        context=ComputationContext(),
    )


@dataclass
class _MLRegressionData:
    X: pd.DataFrame
    y: pd.Series
    df: pd.DataFrame
    records: list[PlacementRecord]


@pytest.fixture(scope="module")
def ml_regression_data(tmp_path_factory) -> _MLRegressionData:
    tmpdir = tmp_path_factory.mktemp("ml_regression_dataset")
    records = make_random_placement_records(80, variant="ml")
    ds = DatasetLogger(str(tmpdir))
    for r in records:
        ds.add_record(r)
    ds.flush()
    df = load_dataset(str(tmpdir))
    X, y = extract_features_from_dataset(df)
    return _MLRegressionData(X=X, y=y, df=df, records=records)


# ── Schema tests ──


class TestComputationContext:
    def test_from_config(self):
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
        r.converged = False
        r.failure_stage = "validation"
        r.failure_reason = "desorbed"
        r.is_penalty_label = True
        r.label_source = "bo_failure_penalty"
        flat = r.to_flat_dict()
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.molecule == r.molecule
        assert r2.placement_id == r.placement_id
        assert abs(r2.energy_adsorption - r.energy_adsorption) < 1e-10
        assert r2.tilt_deg == r.tilt_deg
        assert r2.context.model_name == r.context.model_name
        assert r2.converged is False
        assert r2.failure_stage == "validation"
        assert r2.failure_reason == "desorbed"
        assert r2.is_penalty_label is True
        assert r2.label_source == "bo_failure_penalty"

    def test_from_flat_dict_parses_string_bools(self):
        r = _make_record(3)
        flat = r.to_flat_dict()
        flat["face_flip"] = "False"
        flat["converged"] = "0"
        flat["is_penalty_label"] = "True"
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.face_flip is False
        assert r2.converged is False
        assert r2.is_penalty_label is True

    def test_flat_dict_roundtrip_preserves_context_fields(self):
        r = _make_record(5)
        r.context = ComputationContext(
            model_name="m",
            fmax=0.02,
            stage1_steps=12,
            stage2_steps=34,
            device="cpu",
            seed=9,
            placement_z_range=(1.8, 2.9),
            placement_z_scale_by_covalent_radius=False,
            min_initial_distance=1.2,
            min_contact_ratio=0.7,
            top_layer_tolerance=0.4,
        )
        r2 = PlacementRecord.from_flat_dict(r.to_flat_dict())
        assert r2.context.device == "cpu"
        assert r2.context.placement_z_range == (1.8, 2.9)
        assert r2.context.placement_z_scale_by_covalent_radius is False

    def test_quaternion_is_canonicalized(self):
        base = _make_record(0)
        kwargs = {**base.__dict__}
        kwargs.update({"quat_w": -2.0, "quat_x": 0.0, "quat_y": 0.0, "quat_z": 0.0})
        r = PlacementRecord(**kwargs)
        assert r.quat_w == 1.0
        assert r.quat_x == 0.0
        assert r.quat_y == 0.0
        assert r.quat_z == 0.0

    def test_from_screening_result_returns_record_with_descriptor(self):
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
            z_offset=2.5,
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
            slab_size=0,
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
        assert len(features) == 8
        assert "height_above_surface" not in features
        assert "xy_radius" not in features
        assert "shape_round" not in features
        assert "face_flip" not in features
        assert "z_fraction" not in features

    def test_quaternion_features(self):
        r = _make_record()
        r.quat_w = 2.0
        r.quat_x = 2.0
        r.quat_y = 2.0
        r.quat_z = 2.0
        features = extract_features(r)
        assert abs(features["quat_w"] - 0.5) < 1e-8
        assert abs(features["quat_x"] - 0.5) < 1e-8
        assert abs(features["quat_y"] - 0.5) < 1e-8
        assert abs(features["quat_z"] - 0.5) < 1e-8

    def test_quaternion_sign_invariance(self):
        r1 = _make_record()
        r2 = _make_record()
        r1.quat_w, r1.quat_x, r1.quat_y, r1.quat_z = 0.5, 0.5, 0.5, 0.5
        r2.quat_w, r2.quat_x, r2.quat_y, r2.quat_z = -0.5, -0.5, -0.5, -0.5
        f1 = extract_features(r1)
        f2 = extract_features(r2)
        assert f1 == f2

    def test_rotation_and_categorical_independence(self):
        r = _make_record()
        r.orientation_type = "parallel"
        r.site_type = "bridge"
        features = extract_features(r)
        assert "orient_parallel" not in features
        assert "site_bridge" not in features
        assert "tilt_sin" not in features
        assert "azimuth_sin" not in features
        assert "azimuth_in_plane_sin" not in features

    def test_face_flip_not_encoded_in_features(self):
        r1 = _make_record()
        r2 = _make_record()
        r1.face_flip = False
        r2.face_flip = True
        assert extract_features(r1) == extract_features(r2)

    def test_z_fraction_not_encoded_in_features(self):
        r1 = _make_record()
        r2 = _make_record()
        r1.z_fraction = 0.1
        r2.z_fraction = 0.9
        assert extract_features(r1) == extract_features(r2)

    def test_extract_from_dataset(self):
        records = make_random_placement_records(20, variant="ml")
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            X, y = extract_features_from_dataset(df)
            assert X.shape[0] == 20
            assert X.shape[1] == 8
            assert len(y) == 20
            assert "face_flip" not in X.columns
            assert "z_fraction" not in X.columns

    def test_extract_features_uses_absolute_geometry_only(self):
        r = _make_record()
        r.x = 99.0
        r.y = 88.0
        r.z_offset = 77.0
        r.x_abs = 1.25
        r.y_abs = -2.5
        r.z_abs = 3.75
        features = extract_features(r)
        assert features["x"] == 1.25
        assert features["y"] == -2.5
        assert features["z"] == 3.75

    def test_extract_from_dataset_requires_absolute_geometry_columns(self):
        records = make_random_placement_records(4, variant="ml")
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            df = df.drop(columns=["x_abs", "y_abs", "z_abs"])
            with pytest.raises(ValueError, match="strict geometric feature columns"):
                extract_features_from_dataset(df)

    def test_feature_names_consistent(self):
        names = get_feature_names()
        r = _make_record()
        features = extract_features(r)
        assert list(features.keys()) == names


# ── Regression tests ──


class TestRegression:
    def test_train_ridge(self, ml_regression_data):
        X, y, _ = ml_regression_data.X, ml_regression_data.y, ml_regression_data.df
        model = train_model(X, y, model_type="ridge")
        metrics = evaluate_model(model, X, y)
        assert metrics["mae"] >= 0
        assert metrics["rmse"] >= metrics["mae"]

    def test_train_random_forest(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        model = train_model(X, y, model_type="random_forest")
        metrics = evaluate_model(model, X, y)
        assert metrics["r2"] > 0

    def test_train_gradient_boost(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        model = train_model(X, y, model_type="gradient_boost")
        metrics = evaluate_model(model, X, y)
        assert -1.0 <= metrics["r2"] <= 1.0

    def test_grouped_cv(self, ml_regression_data):
        X, y, df = ml_regression_data.X, ml_regression_data.y, ml_regression_data.df
        result = grouped_cross_validate(
            X, y, groups=df["molecule"], model_type="ridge", n_splits=4
        )
        assert "mean_mae" in result
        assert "fold_metrics" in result
        assert len(result["fold_metrics"]) == 4

    def test_grouped_cv_raises_for_single_sample(self):
        X = pd.DataFrame({"x": [0.1], "y": [0.2]})
        y = pd.Series([-0.3])
        groups = pd.Series(["g1"])
        with pytest.raises(ValueError, match="at least 2 samples"):
            grouped_cross_validate(X, y, groups=groups, model_type="ridge", n_splits=4)

    def test_grouped_cv_raises_for_single_group(self):
        X = pd.DataFrame({"x": [0.1, 0.2], "y": [0.2, 0.3]})
        y = pd.Series([-0.3, -0.4])
        groups = pd.Series(["g1", "g1"])
        with pytest.raises(ValueError, match="at least 2 unique groups"):
            grouped_cross_validate(X, y, groups=groups, model_type="ridge", n_splits=4)

    def test_feature_importance_rf(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        model = train_model(X, y, model_type="random_forest")
        fi = feature_importance(model, list(X.columns), top_k=5)
        assert len(fi) == 5
        assert "feature" in fi.columns
        assert "importance" in fi.columns

    def test_feature_importance_permutation(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        model = train_model(X, y, model_type="gradient_boost")
        fi = feature_importance(model, list(X.columns), X=X, y=y, top_k=5)
        assert len(fi) == 5

    def test_save_load_model(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        model = train_model(X, y, model_type="ridge")
        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(model, tmpdir, "ridge", feature_names=list(X.columns))
            loaded, meta = load_model(tmpdir)
            assert meta["model_type"] == "ridge"
            y_pred = loaded.predict(X)
            assert len(y_pred) == len(y)

    def test_invalid_model_type(self, ml_regression_data):
        X, y, _ = (
            ml_regression_data.X,
            ml_regression_data.y,
            ml_regression_data.df,
        )
        with pytest.raises(ValueError, match="Unknown model_type"):
            train_model(X, y, model_type="invalid")


# ── Prediction tests ──


class TestPredictor:
    @pytest.fixture(scope="module")
    def trained_predictor(self, ml_regression_data):
        X, y = ml_regression_data.X, ml_regression_data.y
        model = train_model(X, y, model_type="gradient_boost")
        metadata = {
            "model_type": "gradient_boost",
            "feature_names": list(X.columns),
        }
        return BindingEnergyPredictor(
            model, metadata=metadata
        ), ml_regression_data.records

    def test_predict_single(self, trained_predictor):
        pred, records = trained_predictor
        result = pred.predict_record(records[0])
        assert isinstance(result, PredictionResult)
        assert isinstance(result.energy, float)

    def test_predict_batch(self, trained_predictor):
        pred, records = trained_predictor
        results = pred.predict_batch(records[:10])
        assert len(results) == 10

    def test_rank_placements(self, trained_predictor):
        pred, records = trained_predictor
        ranked = pred.rank_placements(records[:20], top_k=5)
        assert len(ranked) == 5
        energies = [p.energy for _, p in ranked]
        assert energies == sorted(energies)

    def test_predict_descriptor(self, trained_predictor):
        pred, records = trained_predictor
        descriptor = record_to_placement_descriptor(records[0])
        result = pred.predict_descriptor(descriptor, molecule="ethanol", smiles="CCO")
        assert isinstance(result.energy, float)


# ── Reproduce tests ──


class TestReproduce:
    def test_record_to_descriptor(self):
        r = _make_record()
        r.x_abs = 4.1
        r.y_abs = -1.2
        r.z_offset = 2.8
        r.surface_ref_z_abs = 10.0
        r.z_abs = 12.8
        r.site_source = "adsorption_sites"
        r.site_reference_frame = "local_site"
        r.site_xy_frac_a = 0.25
        r.site_xy_frac_b = 0.75
        d = record_to_placement_descriptor(r)
        assert d.conformer_index == r.conformer_index
        assert d.tilt_deg == r.tilt_deg
        assert d.x == r.x
        assert d.x_abs == r.x_abs
        assert d.y_abs == r.y_abs
        assert d.z_offset == r.z_offset
        assert d.surface_ref_z_abs == r.surface_ref_z_abs
        assert d.z_abs == r.z_abs
        assert d.site_source == r.site_source
        assert d.site_reference_frame == r.site_reference_frame
        assert d.site_xy_frac_a == r.site_xy_frac_a
        assert d.site_xy_frac_b == r.site_xy_frac_b

    def test_record_to_config(self):
        r = _make_record()
        cfg = AdsorptionConfig(
            symmetry_tolerance=0.2,
            site_equivalence_tolerance=0.06,
            hollow_site_dedup_tolerance=0.3,
            planar_z_variance_threshold=0.04,
        )
        r.context = ComputationContext.from_config(cfg)
        config = record_to_config(r)
        assert config.model_name == r.context.model_name
        assert config.fmax == r.context.fmax
        assert config.seed == r.context.seed
        assert config.symmetry_tolerance == r.context.symmetry_tolerance
        assert config.site_equivalence_tolerance == r.context.site_equivalence_tolerance
        assert (
            config.hollow_site_dedup_tolerance == r.context.hollow_site_dedup_tolerance
        )
        assert (
            config.planar_z_variance_threshold == r.context.planar_z_variance_threshold
        )

    def test_record_to_descriptor_requires_finite_geometry(self):
        r = _make_record()
        r.z_abs = float("nan")
        with pytest.raises(
            ValueError, match="missing finite deterministic geometry fields"
        ):
            record_to_placement_descriptor(r)


class TestAcquisitionMinimization:
    """LCB / EI / PI for minimisation (lower binding energy is better)."""

    def test_lcb_scores(self):
        mu = np.array([0.0, 1.0])
        sig = np.array([1.0, 2.0])
        out = lcb_scores(mu, sig, kappa=1.0)
        assert_allclose(out, mu - sig)

    def test_ei_scores_sigma_zero_no_improvement(self):
        mu = np.array([-1.0])
        sig = np.array([0.0])
        out = ei_scores(mu, sig, f_best=-3.0, xi=1e-6)
        assert_allclose(out, [0.0], atol=1e-5)

    def test_ei_scores_sigma_zero_improvement(self):
        mu = np.array([-3.0])
        sig = np.array([0.0])
        f_best = -1.0
        out = ei_scores(mu, sig, f_best=f_best, xi=1e-6)
        assert_allclose(out, np.array([max(0.0, f_best - float(mu[0]))]), rtol=1e-5)

    def test_pi_scores_sigma_zero(self):
        mu = np.array([-2.0, 0.0])
        sig = np.array([0.0, 0.0])
        f_best = -1.0
        xi = 1e-6
        out = pi_scores(mu, sig, f_best=f_best, xi=xi)
        assert out[0] == 1.0
        assert out[1] == 0.0

    def test_ei_matches_analytic_normal(self):
        mu = np.array([0.5])
        sig = np.array([1.0])
        f_best = 0.0
        xi = 0.0
        imp = f_best - mu - xi
        z = imp / sig
        expected = imp * stats.norm.cdf(z) + sig * stats.norm.pdf(z)
        out = ei_scores(mu, sig, f_best=f_best, xi=xi)
        assert_allclose(out, expected)

"""Tests for metalsurfer.ml dataset, features, schema, and surrogate builders."""

import json
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from numpy.testing import assert_allclose
from scipy import stats

from metalsurfer import _numeric_defaults as numeric_defaults
from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml import (
    ComputationContext as PublicComputationContext,
)
from metalsurfer.ml import (
    DatasetLogger as PublicDatasetLogger,
)
from metalsurfer.ml import (
    PlacementRecord as PublicPlacementRecord,
)
from metalsurfer.ml import (
    extract_features as public_extract_features,
)
from metalsurfer.ml import (
    load_dataset as public_load_dataset,
)
from metalsurfer.ml.bayesian import ei_scores, lcb_scores, pi_scores
from metalsurfer.ml.dataset import DatasetLogger, load_dataset
from metalsurfer.ml.features import (
    extract_features,
    extract_features_from_dataset,
    get_feature_names,
)
from metalsurfer.ml.regression import _build_estimator
from metalsurfer.ml.schema import SCHEMA_VERSION, ComputationContext, PlacementRecord
from metalsurfer.models import PlacementDescriptor, ScreeningResult
from tests.factories import make_placement_record, make_random_placement_records


def test_schema_version_is_3_0():
    assert SCHEMA_VERSION == "3.0"


def test_ml_package_exports_expanded_surface():
    """Public ml package re-exports dataset/schema/features helpers."""
    assert PublicComputationContext is ComputationContext
    assert PublicPlacementRecord is PlacementRecord
    assert PublicDatasetLogger is DatasetLogger
    assert public_load_dataset is load_dataset
    assert public_extract_features is extract_features


def test_computation_context_defaults_match_numeric_defaults():
    ctx = ComputationContext()
    assert (
        ctx.min_initial_distance
        == numeric_defaults.MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    )
    assert ctx.min_contact_ratio == numeric_defaults.MIN_CONTACT_RATIO_DEFAULT
    assert ctx.symmetry_tolerance == numeric_defaults.DEFAULT_SYMMETRY_TOLERANCE
    assert (
        ctx.site_equivalence_tolerance
        == numeric_defaults.DEFAULT_SITE_EQUIVALENCE_TOLERANCE
    )
    assert (
        ctx.hollow_site_dedup_tolerance
        == numeric_defaults.DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
    )
    assert (
        ctx.planar_z_variance_threshold
        == numeric_defaults.DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
    )


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
        r1 = make_placement_record(0)
        r2 = make_placement_record(0)
        assert r1.record_hash() == r2.record_hash()

    def test_record_hash_changes_with_position(self):
        r1 = make_placement_record(0)
        r2 = make_placement_record(1)
        assert r1.record_hash() != r2.record_hash()

    def test_to_flat_dict_keys(self):
        r = make_placement_record()
        flat = r.to_flat_dict()
        assert "record_hash" in flat
        assert "molecule" in flat
        assert "energy_adsorption" in flat
        assert "context_hash" in flat
        assert "x_abs" in flat
        assert "quat_w" in flat
        assert "model_name" not in flat
        assert "tilt_deg" not in flat
        assert "initial_tilt_deg" not in flat
        assert "orientation_type" not in flat

    def test_to_flat_dict_rich_provenance_keys(self):
        r = make_placement_record()
        flat = r.to_flat_dict(include_provenance=True)
        assert "initial_tilt_deg" in flat
        assert "initial_orientation_type" in flat
        assert "initial_site_type" in flat
        assert "ctx_model_name" in flat
        assert "model_name" not in flat
        assert flat["initial_tilt_deg"] == r.descriptor.tilt_deg

    def test_flat_dict_roundtrip(self):
        r = make_placement_record(42, energy=-1.23)
        r.converged = False
        r.failure_stage = "validation"
        r.failure_reason = "desorbed"
        r.is_penalty_label = True
        r.label_source = "bo_failure_penalty"
        flat = r.to_flat_dict(include_provenance=True)
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.molecule == r.molecule
        assert r2.placement_id == r.placement_id
        assert abs(r2.energy_adsorption - r.energy_adsorption) < 1e-10
        assert r2.descriptor.tilt_deg == r.descriptor.tilt_deg
        assert r2.context.model_name == r.context.model_name
        assert r2.converged is False
        assert r2.failure_stage == "validation"
        assert r2.failure_reason == "desorbed"
        assert r2.is_penalty_label is True
        assert r2.label_source == "bo_failure_penalty"

    def test_lean_flat_dict_roundtrip_preserves_features(self):
        r = make_placement_record(7, energy=-0.4)
        flat = r.to_flat_dict(include_provenance=False)
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.descriptor.x_abs == r.descriptor.x_abs
        assert r2.descriptor.y_abs == r.descriptor.y_abs
        assert r2.descriptor.z_abs == r.descriptor.z_abs
        assert r2.descriptor.conformer_index == r.descriptor.conformer_index
        assert abs(r2.energy_adsorption - r.energy_adsorption) < 1e-10
        # Provenance absent → defaults
        assert r2.descriptor.tilt_deg == 0.0
        assert r2.descriptor.site_index == -1

    def test_from_flat_dict_ignores_unprefixed_provenance_columns(self):
        r = make_placement_record(3)
        flat = r.to_flat_dict(include_provenance=False)
        flat["tilt_deg"] = 15.0
        flat["orientation_type"] = "round"
        flat["site_index"] = 2
        flat["site_type"] = "atop"
        flat["azimuth_deg"] = 45.0
        flat["azimuth_in_plane_deg"] = 0.0
        flat["face_flip"] = False
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.descriptor.tilt_deg == 0.0
        assert r2.descriptor.site_index == -1
        assert r2.descriptor.site_type is None

    def test_from_flat_dict_parses_string_bools(self):
        r = make_placement_record(3)
        flat = r.to_flat_dict(include_provenance=True)
        flat["initial_face_flip"] = "False"
        flat["converged"] = "0"
        flat["is_penalty_label"] = "True"
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.descriptor.face_flip is False
        assert r2.converged is False
        assert r2.is_penalty_label is True

    def test_flat_dict_roundtrip_preserves_context_fields(self):
        r = make_placement_record(5)
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
        r2 = PlacementRecord.from_flat_dict(r.to_flat_dict(include_provenance=True))
        assert r2.context.device == "cpu"
        assert r2.context.placement_z_range == (1.8, 2.9)
        assert r2.context.placement_z_scale_by_covalent_radius is False
        assert r2.context.model_name == "m"

    def test_quaternion_is_canonicalized(self):
        base = make_placement_record(0)
        descriptor = PlacementDescriptor(
            **{
                **base.descriptor.__dict__,
                "quat_w": -2.0,
                "quat_x": 0.0,
                "quat_y": 0.0,
                "quat_z": 0.0,
            }
        )
        r = PlacementRecord(
            molecule=base.molecule,
            smiles=base.smiles,
            surface_id=base.surface_id,
            placement_id=base.placement_id,
            descriptor=descriptor,
            context=base.context,
        )
        assert r.descriptor.quat_w == 1.0
        assert r.descriptor.quat_x == 0.0
        assert r.descriptor.quat_y == 0.0
        assert r.descriptor.quat_z == 0.0

    def test_quaternion_w_zero_antipodal_canonicalized(self):
        base = make_placement_record(0)
        common = {
            **base.descriptor.__dict__,
            "quat_w": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
        }
        r1 = PlacementRecord(
            molecule=base.molecule,
            smiles=base.smiles,
            surface_id=base.surface_id,
            placement_id=0,
            descriptor=PlacementDescriptor(**{**common, "quat_x": 1.0}),
            context=base.context,
        )
        r2 = PlacementRecord(
            molecule=base.molecule,
            smiles=base.smiles,
            surface_id=base.surface_id,
            placement_id=0,
            descriptor=PlacementDescriptor(**{**common, "quat_x": -1.0}),
            context=base.context,
        )
        assert r1.descriptor.quat_w == pytest.approx(0.0)
        assert r1.descriptor.quat_x == pytest.approx(1.0)
        assert r2.descriptor.quat_w == pytest.approx(0.0)
        assert r2.descriptor.quat_x == pytest.approx(1.0)
        assert extract_features(r1) == extract_features(r2)
        assert r1.record_hash() == r2.record_hash()

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
        assert record.energy_adsorption == pytest.approx(-0.5)


# ── Dataset tests ──


class TestDatasetLogger:
    def test_flush_metadata_counts_duplicate_csv_rows(self):
        """total_records should count CSV rows even when hashes are duplicated on disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(make_placement_record(0))
            ds.flush()
            # Corrupt on-disk CSV by duplicating the data row (same record_hash).
            path = ds.csv_path
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # header + 1 row
            with open(path, "a") as f:
                f.write(lines[1])
            with open(path) as f:
                assert sum(1 for _ in f) - 1 == 2

            ds2 = DatasetLogger(tmpdir)
            ds2.add_record(make_placement_record(1))
            ds2.flush()
            with open(ds2.metadata_path) as f:
                meta = json.load(f)
            # Unique hashes would report 2; row count must report 3.
            assert meta["total_records"] == 3
            assert len(pd.read_csv(ds2.csv_path)) == 3

    def test_flush_creates_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(make_placement_record(0))
            ds.add_record(make_placement_record(1))
            path = ds.flush()
            assert os.path.exists(path)
            df = pd.read_csv(path)
            assert len(df) == 2

    def test_flush_appends(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir)
            ds1.add_record(make_placement_record(0))
            ds1.flush()

            ds2 = DatasetLogger(tmpdir)
            ds2.add_record(make_placement_record(1))
            ds2.flush()

            df = pd.read_csv(ds2.csv_path)
            assert len(df) == 2

    def test_flush_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir)
            ds1.add_record(make_placement_record(0))
            ds1.flush()

            ds2 = DatasetLogger(tmpdir)
            ds2.add_record(make_placement_record(0))  # same record
            ds2.flush()

            df = pd.read_csv(ds2.csv_path)
            assert len(df) == 1

    def test_metadata_written(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir, surface_id="test")
            ds.add_record(make_placement_record())
            ds.flush()
            assert os.path.exists(ds.metadata_path)
            with open(ds.metadata_path) as f:
                meta = json.load(f)
            assert meta["schema_version"] == SCHEMA_VERSION
            assert meta["export_placement_provenance"] is False

    def test_flush_lean_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir, config=AdsorptionConfig())
            ds.add_record(make_placement_record(0))
            ds.flush()
            df = pd.read_csv(ds.csv_path)
            assert "x_abs" in df.columns
            assert "energy_adsorption" in df.columns
            assert "initial_tilt_deg" not in df.columns
            assert "ctx_model_name" not in df.columns

    def test_flush_rich_when_provenance_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = AdsorptionConfig(export_placement_provenance=True)
            ds = DatasetLogger(tmpdir, config=cfg)
            ds.add_record(make_placement_record(0))
            ds.flush()
            df = pd.read_csv(ds.csv_path)
            assert "initial_tilt_deg" in df.columns
            assert "initial_site_type" in df.columns
            assert "ctx_model_name" in df.columns
            with open(ds.metadata_path) as f:
                meta = json.load(f)
            assert meta["export_placement_provenance"] is True

    def test_flush_rejects_provenance_schema_mismatch(self):
        """Lean↔provenance append must abort before corrupting the CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            lean = DatasetLogger(tmpdir, config=AdsorptionConfig())
            lean.add_record(make_placement_record(0))
            lean.flush()
            with open(lean.csv_path, encoding="utf-8") as f:
                before = f.read()

            rich = DatasetLogger(
                tmpdir, config=AdsorptionConfig(export_placement_provenance=True)
            )
            rich.add_record(make_placement_record(1))
            with pytest.raises(ValueError, match="column schema mismatch"):
                rich.flush()
            with open(lean.csv_path, encoding="utf-8") as f:
                assert f.read() == before

            # Opposite direction: provenance file, lean append.
            with tempfile.TemporaryDirectory() as tmpdir2:
                rich2 = DatasetLogger(
                    tmpdir2, config=AdsorptionConfig(export_placement_provenance=True)
                )
                rich2.add_record(make_placement_record(0))
                rich2.flush()
                with open(rich2.csv_path, encoding="utf-8") as f:
                    before2 = f.read()

                lean2 = DatasetLogger(tmpdir2, config=AdsorptionConfig())
                lean2.add_record(make_placement_record(1))
                with pytest.raises(ValueError, match="column schema mismatch"):
                    lean2.flush()
                with open(rich2.csv_path, encoding="utf-8") as f:
                    assert f.read() == before2

    def test_flush_rejects_mixed_context_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir, config=AdsorptionConfig(model_name="uma-s-1p2"))
            ds1.add_record(make_placement_record(0))
            ds1.flush()

            ds2 = DatasetLogger(tmpdir, config=AdsorptionConfig(model_name="uma-s-1p1"))
            ds2.add_record(make_placement_record(1))
            with pytest.raises(ValueError, match="computation context mismatch"):
                ds2.flush()

    def test_flush_allow_mixed_context(self, caplog):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds1 = DatasetLogger(tmpdir, config=AdsorptionConfig(model_name="uma-s-1p2"))
            ds1.add_record(make_placement_record(0))
            ds1.flush()

            ds2 = DatasetLogger(
                tmpdir,
                config=AdsorptionConfig(model_name="uma-s-1p1"),
                allow_mixed_context=True,
            )
            ds2.add_record(make_placement_record(1))
            with caplog.at_level(logging.WARNING, logger="metalsurfer.ml.dataset"):
                ds2.flush()
            assert "mixed computation context" in caplog.text
            df = pd.read_csv(ds2.csv_path)
            assert len(df) == 2


class TestLoadDataset:
    def test_load_from_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(make_placement_record(0))
            ds.flush()
            df = load_dataset(tmpdir)
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1

    def test_load_as_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            ds.add_record(make_placement_record(0))
            ds.flush()
            records = load_dataset(tmpdir, as_records=True)
            assert isinstance(records, list)
            assert isinstance(records[0], PlacementRecord)

    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError, match="Dataset not found:"):
            load_dataset("/nonexistent/path")


# ── Feature tests ──


class TestFeatureExtraction:
    def test_feature_count(self):
        r = make_placement_record()
        features = extract_features(r)
        assert len(features) == 8
        assert "height_above_surface" not in features
        assert "xy_radius" not in features
        assert "shape_round" not in features
        assert "face_flip" not in features
        assert "z_fraction" not in features

    def test_quaternion_features(self):
        r = make_placement_record()
        r.descriptor.quat_w = 2.0
        r.descriptor.quat_x = 2.0
        r.descriptor.quat_y = 2.0
        r.descriptor.quat_z = 2.0
        features = extract_features(r)
        assert abs(features["quat_w"] - 0.5) < 1e-8
        assert abs(features["quat_x"] - 0.5) < 1e-8
        assert abs(features["quat_y"] - 0.5) < 1e-8
        assert abs(features["quat_z"] - 0.5) < 1e-8

    def test_quaternion_sign_invariance(self):
        r1 = make_placement_record()
        r2 = make_placement_record()
        (
            r1.descriptor.quat_w,
            r1.descriptor.quat_x,
            r1.descriptor.quat_y,
            r1.descriptor.quat_z,
        ) = (
            0.5,
            0.5,
            0.5,
            0.5,
        )
        (
            r2.descriptor.quat_w,
            r2.descriptor.quat_x,
            r2.descriptor.quat_y,
            r2.descriptor.quat_z,
        ) = (
            -0.5,
            -0.5,
            -0.5,
            -0.5,
        )
        f1 = extract_features(r1)
        f2 = extract_features(r2)
        assert f1 == f2

    def test_rotation_and_categorical_independence(self):
        r = make_placement_record()
        r.descriptor.orientation_type = "parallel"
        r.descriptor.site_type = "bridge"
        features = extract_features(r)
        assert "orient_parallel" not in features
        assert "site_bridge" not in features
        assert "tilt_sin" not in features
        assert "azimuth_sin" not in features
        assert "azimuth_in_plane_sin" not in features

    def test_face_flip_not_encoded_in_features(self):
        r1 = make_placement_record()
        r2 = make_placement_record()
        r1.descriptor.face_flip = False
        r2.descriptor.face_flip = True
        assert extract_features(r1) == extract_features(r2)

    def test_z_fraction_not_encoded_in_features(self):
        r1 = make_placement_record()
        r2 = make_placement_record()
        r1.descriptor.z_fraction = 0.1
        r2.descriptor.z_fraction = 0.9
        assert extract_features(r1) == extract_features(r2)

    @pytest.mark.xfail(
        strict=False,
        reason=(
            "extract_features includes the absolute Cartesian coordinates "
            "x/x_abs and y/y_abs, so it is NOT invariant under a 2D lattice "
            "translation or an SO(2) surface rotation of the pose. This is a "
            "missing pose-relative capability (TODO/bug report: derive features "
            "from pose-relative descriptors only, not absolute in-plane x/y)."
        ),
    )
    def test_feature_translation_rotation_invariance(self):
        """Features must be unchanged by a 2D translation or SO(2) rotation.

        A physically pose-relative feature vector should depend only on the
        placement's relationship to the surface (height, orientation, site type),
        not on the absolute in-plane Cartesian position.
        """
        from metalsurfer.ml.features import extract_features

        r0 = make_placement_record()
        f0 = extract_features(r0)

        # 2D lattice translation of the adsorbate.
        r1 = make_placement_record()
        r1.descriptor.x_abs += 2.0
        r1.descriptor.y_abs += 1.0
        f1 = extract_features(r1)
        assert f1 == f0

        # SO(2) rotation about the surface normal: rotate the in-plane position.
        r2 = make_placement_record()
        x, y = float(r0.descriptor.x_abs), float(r0.descriptor.y_abs)
        r2.descriptor.x_abs = -y
        r2.descriptor.y_abs = x
        f2 = extract_features(r2)
        assert f2 == f0

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
        r = make_placement_record()
        r.descriptor.x = 99.0
        r.descriptor.y = 88.0
        r.descriptor.z_offset = 77.0
        r.descriptor.x_abs = 1.25
        r.descriptor.y_abs = -2.5
        r.descriptor.z_abs = 3.75
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

    def test_extract_from_dataset_rejects_empty_or_nonfinite_targets(self):
        records = make_random_placement_records(3, variant="ml")
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            empty = df.iloc[0:0].copy()
            with pytest.raises(ValueError, match="empty"):
                extract_features_from_dataset(empty)
            bad = df.copy()
            bad.loc[0, "energy_adsorption"] = float("nan")
            with pytest.raises(ValueError, match="non-finite"):
                extract_features_from_dataset(bad)
            bad2 = df.copy()
            bad2.loc[1, "energy_adsorption"] = float("inf")
            with pytest.raises(ValueError, match="non-finite"):
                extract_features_from_dataset(bad2)

    def test_extract_from_dataset_warns_on_nan_quaternion(self, caplog):
        records = make_random_placement_records(3, variant="ml")
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = DatasetLogger(tmpdir)
            for r in records:
                ds.add_record(r)
            ds.flush()
            df = load_dataset(tmpdir)
            df.loc[0, "quat_w"] = float("nan")
            with caplog.at_level(logging.WARNING, logger="metalsurfer.ml.features"):
                X, _ = extract_features_from_dataset(df)
            assert "identity default" in caplog.text
            assert X.loc[0, "quat_w"] == pytest.approx(1.0)

    def test_extract_features_raises_on_none_quaternion(self):
        r = make_placement_record()
        r.descriptor.quat_w = None
        with pytest.raises(ValueError, match="quat_w must be finite"):
            extract_features(r)

    def test_feature_names_consistent(self):
        names = get_feature_names()
        r = make_placement_record()
        features = extract_features(r)
        assert list(features.keys()) == names


# ── Estimator construction ──


class TestBuildEstimator:
    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            _build_estimator("invalid")


# ── Record replay tests ──


class TestRecordReplay:
    def test_record_to_descriptor(self):
        r = make_placement_record()
        r.descriptor.x_abs = 4.1
        r.descriptor.y_abs = -1.2
        r.descriptor.z_offset = 2.8
        r.descriptor.surface_ref_z_abs = 10.0
        r.descriptor.z_abs = 12.8
        r.descriptor.site_source = "adsorption_sites"
        r.descriptor.site_reference_frame = "local_site"
        r.descriptor.site_xy_frac_a = 0.25
        r.descriptor.site_xy_frac_b = 0.75
        d = r.to_placement_descriptor()
        assert d.conformer_index == r.descriptor.conformer_index
        assert d.tilt_deg == r.descriptor.tilt_deg
        assert d.x == r.descriptor.x
        assert d.x_abs == r.descriptor.x_abs
        assert d.y_abs == r.descriptor.y_abs
        assert d.z_offset == r.descriptor.z_offset
        assert d.surface_ref_z_abs == r.descriptor.surface_ref_z_abs
        assert d.z_abs == r.descriptor.z_abs
        assert d.site_source == r.descriptor.site_source
        assert d.site_reference_frame == r.descriptor.site_reference_frame
        assert d.site_xy_frac_a == r.descriptor.site_xy_frac_a
        assert d.site_xy_frac_b == r.descriptor.site_xy_frac_b
        assert d.fragment_positions is None

    def test_record_to_descriptor_preserves_fragment_positions(self):
        r = make_placement_record()
        fragments = ((1.0, 2.0, 3.0), (1.5, 2.5, 3.5))
        r.descriptor.fragment_positions = fragments
        d = r.to_placement_descriptor()
        assert d.fragment_positions == fragments
        flat = r.to_flat_dict(include_provenance=True)
        r2 = PlacementRecord.from_flat_dict(flat)
        assert r2.descriptor.fragment_positions == fragments
        lean = r.to_flat_dict(include_provenance=False)
        assert "initial_fragment_positions" not in lean
        assert "fragment_positions" not in lean

    def test_record_to_config(self):
        r = make_placement_record()
        cfg = AdsorptionConfig(
            symmetry_tolerance=0.2,
            site_equivalence_tolerance=0.06,
            hollow_site_dedup_tolerance=0.3,
            planar_z_variance_threshold=0.04,
        )
        r.context = ComputationContext.from_config(cfg)
        config = r.to_config()
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
        r = make_placement_record()
        r.descriptor.z_abs = float("nan")
        with pytest.raises(
            ValueError, match="missing finite deterministic geometry fields"
        ):
            r.to_placement_descriptor()


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
        out = ei_scores(
            mu, sig, f_best=-3.0, xi=numeric_defaults.ACQUISITION_XI_DEFAULT
        )
        # Near-zero sigma ranks by -mu (avoid pool collapse).
        assert_allclose(out, [1.0], atol=1e-5)

    def test_ei_scores_sigma_zero_improvement(self):
        mu = np.array([-3.0])
        sig = np.array([0.0])
        f_best = -1.0
        out = ei_scores(
            mu, sig, f_best=f_best, xi=numeric_defaults.ACQUISITION_XI_DEFAULT
        )
        assert_allclose(out, np.array([-float(mu[0])]), rtol=1e-5)

    def test_pi_scores_sigma_zero(self):
        mu = np.array([-2.0, 0.0])
        sig = np.array([0.0, 0.0])
        f_best = -1.0
        xi = numeric_defaults.ACQUISITION_XI_DEFAULT
        out = pi_scores(mu, sig, f_best=f_best, xi=xi)
        assert_allclose(out, -mu)
        assert out[0] > out[1]

    def test_ei_scores_sigma_zero_ranks_by_negative_mu(self):
        mu = np.array([1.0, 2.0, -0.5])
        sig = np.array([0.0, 0.0, 0.0])
        out = ei_scores(mu, sig, f_best=0.0, xi=numeric_defaults.ACQUISITION_XI_DEFAULT)
        assert_allclose(out, -mu)
        assert int(np.argmax(out)) == 2

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

    def test_mixed_sigma_pi_stays_in_unit_interval(self):
        mu = np.array([-1.5, -0.2, -2.0, -0.8])
        sig = np.array([0.0, 0.3, 0.0, 0.25])
        f_best = -1.0
        xi = 0.0
        out = pi_scores(mu, sig, f_best=f_best, xi=xi)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_mixed_sigma_worse_zero_sigma_does_not_beat_better_uncertain(self):
        # Zero-σ μ=-0.1 (worse) vs uncertain μ=-1.8 (better); finite-σ should win.
        mu = np.array([-0.1, -1.8])
        sig = np.array([0.0, 0.4])
        f_best = -1.0
        xi = 0.0
        ei = ei_scores(mu, sig, f_best=f_best, xi=xi)
        pi = pi_scores(mu, sig, f_best=f_best, xi=xi)
        assert int(np.argmax(ei)) == 1
        assert int(np.argmax(pi)) == 1
        assert ei[0] == pytest.approx(0.0)
        assert pi[0] == pytest.approx(0.0)


def test_from_flat_dict_rejects_corrupt_payload():
    with pytest.raises((KeyError, TypeError, ValueError)):
        PlacementRecord.from_flat_dict({"schema_version": "not-a-real-record"})

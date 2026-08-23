"""Tests for Bayesian optimisation placement selection pipeline."""

import logging

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metalsurfer._numeric_defaults import ACQUISITION_SIGMA_FLOOR
from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.ml.bayesian import (
    EnsembleRegressor,
    build_spec_features_geometry_aware,
    build_transfer_surrogate,
    cumulative_refit_training_set,
    ei_scores,
    lcb_scores,
    matern_length_scale_for_n_features,
    predict_with_uncertainty,
    prior_placement_downweight,
    prior_proximity_weights,
    prior_recency_weights,
    prior_similarity_to_current,
    score_and_select,
    select_candidates,
    select_candidates_batch_diverse,
    select_initial_bo_indices,
    train_surrogate,
)
from metalsurfer.ml.features import extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.models import (
    BOStepMemory,
    PlacementDescriptor,
    PlacementSpec,
    windowed_bo_step_memories,
)
from metalsurfer.placement import (
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
)
from tests.factories import make_random_placement_records

from ._logging_helpers import CaptureHandler
from .conftest import gpu_mlip_test, make_placement_descriptor, make_slab, make_water


def _make_synthetic_training_data(n: int = 40):
    records = make_random_placement_records(n, variant="bayesian")
    rows = [extract_features(r) for r in records]
    X = pd.DataFrame(rows)
    y = pd.Series([r.energy_adsorption for r in records])
    return X, y


def _placement_spec(idx: int = 0) -> PlacementSpec:
    return PlacementSpec(
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=idx % 5,
        site_type="atop",
        tilt_deg=float(idx * 15 % 90),
        azimuth_deg=float(idx * 45 % 360),
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=idx,
    )


def _bayesian_descriptor(idx: int) -> PlacementDescriptor:
    return make_placement_descriptor(
        placement_id=idx,
        site_index=idx % 5,
        tilt_deg=float(idx * 15 % 90),
        azimuth_deg=float(idx * 45 % 360),
        x=float(1.0 + idx * 0.1),
        y=float(2.0 - idx * 0.1),
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
    )


# ---------------------------------------------------------------------------
# Surrogate training
# ---------------------------------------------------------------------------


class TestSurrogate:
    def test_train_and_predict(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, n_estimators=20)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (40,)
        assert sigma.shape == (40,)
        assert np.all(sigma >= ACQUISITION_SIGMA_FLOOR)
        assert not np.any(np.isnan(mu))

    def test_seed_reproducibility(self):
        X, y = _make_synthetic_training_data(40)
        m1 = train_surrogate(X, y, n_estimators=20, random_state=123)
        m2 = train_surrogate(X, y, n_estimators=20, random_state=123)
        mu1, _ = predict_with_uncertainty(m1, X)
        mu2, _ = predict_with_uncertainty(m2, X)
        np.testing.assert_array_equal(mu1, mu2)

    def test_predict_with_uncertainty_fallback_no_estimators(self):
        """Fallback path when regressor has no estimators_ (e.g. LinearRegression)."""
        X, y = _make_synthetic_training_data(20)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        pipeline.fit(X, y)
        mu, sigma = predict_with_uncertainty(pipeline, X)
        assert mu.shape == (20,)
        assert sigma.shape == (20,)
        assert np.all(np.isclose(sigma, 0.0))

    def test_train_surrogate_rejects_sample_weight_for_non_weighted(self):
        X, y = _make_synthetic_training_data(20)
        w = np.ones(20, dtype=float)
        with pytest.raises(ValueError, match="sample_weight"):
            train_surrogate(X, y, surrogate="gaussian_process", sample_weight=w)

    @pytest.mark.parametrize(
        "surrogate, kwargs, expect_sigma_gt0, n_samples",
        [
            (None, {"n_estimators": 10}, False, 30),
            ("ridge", {}, False, 20),
            ("gradient_boost", {}, True, 20),
            ("ensemble", {"n_estimators": 5}, False, 20),
        ],
    )
    def test_train_surrogate_accepts_sample_weights(
        self, surrogate, kwargs, expect_sigma_gt0, n_samples
    ):
        X, y = _make_synthetic_training_data(n_samples)
        weights = np.ones(n_samples, dtype=float)
        weights[:5] = 2.0
        train_kwargs = dict(kwargs)
        if surrogate is not None:
            train_kwargs["surrogate"] = surrogate
        model = train_surrogate(X, y, sample_weight=weights, **train_kwargs)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (n_samples,)
        assert sigma.shape == (n_samples,)
        if expect_sigma_gt0:
            assert np.all(sigma > 0)

    def test_gaussian_process_matern_length_scale(self):
        X, y = _make_synthetic_training_data(25)
        n_features = X.shape[1]
        init_scale = matern_length_scale_for_n_features(n_features)
        assert init_scale == pytest.approx(np.sqrt(n_features))
        model = train_surrogate(X, y, surrogate="gaussian_process", random_state=7)
        reg = model.named_steps["regressor"]
        kernel = reg.kernel_
        params = kernel.get_params()
        assert params["k2__length_scale_bounds"] != "fixed"
        fitted_ls = float(params["k2__length_scale"])
        assert np.isfinite(fitted_ls)
        assert 1e-2 <= fitted_ls <= 1e2

    def test_gaussian_process_predict_with_uncertainty(self):
        X, y = _make_synthetic_training_data(25)
        model = train_surrogate(X, y, surrogate="gaussian_process", random_state=7)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (25,)
        assert sigma.shape == (25,)
        assert np.all(sigma >= 0)
        assert np.any(sigma > 0)

    def test_ridge_skips_oof_when_in_sample_above_floor(self, monkeypatch):
        """Ridge with usable in-sample residual should not run KFold OOF."""
        from metalsurfer.ml import bayesian as bayesian_mod

        called = {"oof": False}

        def _fake_oof(*args, **kwargs):
            called["oof"] = True
            return 999.0

        monkeypatch.setattr(bayesian_mod, "_out_of_fold_residual_std", _fake_oof)
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, surrogate="ridge", random_state=0)
        assert not called["oof"]
        reg = model.named_steps["regressor"]
        assert getattr(reg, "bo_residual_std_", None) is not None
        assert reg.bo_residual_std_ > 0

        resid = y.to_numpy(dtype=float) - model.predict(X)
        dof = max(len(y) - X.shape[1] - 1, 1)
        expected = float(np.sqrt(np.sum(np.square(resid)) / dof))
        floor = max(1e-3, 0.05 * float(np.std(y.to_numpy(dtype=float))))
        assert reg.bo_residual_std_ == pytest.approx(max(expected, floor))

    def test_interpolating_regressor_still_runs_oof(self, monkeypatch):
        """An interpolating learner has in-sample RMSE ~= 0, so OOF must kick in."""
        from sklearn.tree import DecisionTreeRegressor

        from metalsurfer.ml import bayesian as bayesian_mod

        X, y = _make_synthetic_training_data(40)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("regressor", DecisionTreeRegressor(random_state=0)),
            ]
        )
        pipeline.fit(X, y)

        real_oof = bayesian_mod._out_of_fold_residual_std
        called = {"oof": False}

        def _spy(*args, **kwargs):
            called["oof"] = True
            return real_oof(*args, **kwargs)

        monkeypatch.setattr(bayesian_mod, "_out_of_fold_residual_std", _spy)
        bayesian_mod._attach_residual_uncertainty(pipeline, X, y, random_state=0)

        reg = pipeline.named_steps["regressor"]
        floor = max(1e-3, 0.05 * float(np.std(y.to_numpy(dtype=float))))
        assert called["oof"]
        assert reg.bo_residual_std_ >= floor

    def test_ensemble_trains_multiple_members(self):
        X, y = _make_synthetic_training_data(30)
        model = train_surrogate(
            X, y, surrogate="ensemble", n_estimators=10, random_state=3
        )
        reg = model.named_steps["regressor"]
        assert isinstance(reg, EnsembleRegressor)
        assert len(reg.members_) == len(reg.member_surrogates)

    def test_ensemble_predict_with_uncertainty(self):
        X, y = _make_synthetic_training_data(30)
        model = train_surrogate(
            X, y, surrogate="ensemble", n_estimators=10, random_state=3
        )
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (30,)
        assert sigma.shape == (30,)
        assert np.all(sigma >= 0)
        assert np.any(sigma > 0)


# ---------------------------------------------------------------------------
# LCB acquisition (lower confidence bound for minimisation)
# ---------------------------------------------------------------------------


class TestAcquisition:
    def test_ridge_predict_with_uncertainty_is_positive(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, surrogate="ridge", random_state=0)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (40,)
        assert sigma.shape == (40,)
        assert np.all(sigma > 0)
        # Far-from-train points should not be less uncertain than in-sample.
        X_far = X.copy()
        X_far.iloc[:, :] = X_far.to_numpy() + 50.0
        _, sigma_far = predict_with_uncertainty(model, X_far)
        assert float(np.mean(sigma_far)) >= float(np.mean(sigma))

    def test_ridge_in_sample_sigma_uses_correct_feature_space(self):
        """In-sample residual NN distance should be ~0 (not double-scaled)."""
        from scipy.spatial.distance import cdist

        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, surrogate="ridge", random_state=0)
        regressor = model.named_steps["regressor"]
        X_train = regressor.bo_X_train_scaled_
        X_raw = np.asarray(X, dtype=float)
        X_eval_scaled = regressor.bo_sigma_scaler_.transform(X_raw)
        d_in_sample = cdist(X_eval_scaled, X_train).min(axis=1)
        assert float(np.max(d_in_sample)) < 1e-6

    def test_lcb_scores_shape(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        scores = lcb_scores(mu, sigma, kappa=1.0)
        assert scores.shape == (3,)
        np.testing.assert_array_almost_equal(scores, [0.5, 1.5, 2.5])

    def test_lcb_lower_is_better(self):
        mu = np.array([-2.0, 0.0, 1.0])
        sigma = np.array([1.0, 0.1, 0.1])
        scores = lcb_scores(mu, sigma, kappa=2.0)
        assert np.argmin(scores) == 0

    def test_select_excludes_evaluated(self):
        scores = np.array([5.0, 1.0, 3.0, 0.5, 2.0])
        selected = select_candidates(scores, batch_size=2, evaluated_indices={1, 3})
        assert 1 not in selected
        assert 3 not in selected
        assert len(selected) == 2
        assert selected[0] == 4  # next best after excluding {1,3}

    def test_select_respects_batch_size(self):
        scores = np.arange(10, dtype=float)
        selected = select_candidates(scores, batch_size=3)
        assert len(selected) == 3
        assert selected == [0, 1, 2]

    def test_select_fewer_than_batch_when_exhausted(self):
        scores = np.array([1.0, 2.0])
        selected = select_candidates(scores, batch_size=5, evaluated_indices={0})
        assert len(selected) == 1

    def test_no_duplicate_selections(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(50)
        selected = select_candidates(scores, batch_size=10, evaluated_indices=set())
        assert len(selected) == len(set(selected))


class TestInitialSampling:
    def test_spread_returns_unique_indices(self):
        X = pd.DataFrame(
            {
                "x": np.linspace(0.0, 10.0, 20),
                "y": np.zeros(20),
                "z": np.zeros(20),
                "conformer_index": np.arange(20),
                "quat_w": np.ones(20),
                "quat_x": np.zeros(20),
                "quat_y": np.zeros(20),
                "quat_z": np.zeros(20),
            }
        )
        picked = select_initial_bo_indices(X, 5, sampling="spread", random_state=7)
        assert len(picked) == 5
        assert len(set(picked)) == 5

    def test_spread_covers_endpoints_in_1d(self):
        X = pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, 11),
                "y": np.zeros(11),
                "z": np.zeros(11),
                "conformer_index": np.zeros(11),
                "quat_w": np.ones(11),
                "quat_x": np.zeros(11),
                "quat_y": np.zeros(11),
                "quat_z": np.zeros(11),
            }
        )
        picked = select_initial_bo_indices(X, 3, sampling="spread", random_state=0)
        assert 0 in picked
        assert 10 in picked

    def test_random_matches_numpy_choice(self):
        X, _ = _make_synthetic_training_data(30)
        expected = np.random.RandomState(11).choice(len(X), size=4, replace=False)
        picked = select_initial_bo_indices(X, 4, sampling="random", random_state=11)
        np.testing.assert_array_equal(picked, expected)

    def test_spread_xyz_uses_position_only(self):
        X = pd.DataFrame(
            {
                "x": np.linspace(0.0, 1.0, 8),
                "y": np.zeros(8),
                "z": np.zeros(8),
                "conformer_index": np.arange(8),
                "quat_w": np.ones(8),
                "quat_x": np.zeros(8),
                "quat_y": np.zeros(8),
                "quat_z": np.zeros(8),
            }
        )
        picked = select_initial_bo_indices(X, 3, sampling="spread_xyz", random_state=0)
        assert len(picked) == 3
        assert 0 in picked
        assert 7 in picked

    def test_stratified_covers_conformers(self):
        X = pd.DataFrame(
            {
                "x": np.zeros(12),
                "y": np.zeros(12),
                "z": np.zeros(12),
                "conformer_index": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
                "quat_w": np.ones(12),
                "quat_x": np.zeros(12),
                "quat_y": np.zeros(12),
                "quat_z": np.zeros(12),
            }
        )
        picked = select_initial_bo_indices(X, 4, sampling="stratified", random_state=3)
        conformers = X.iloc[picked]["conformer_index"].astype(int).tolist()
        assert len(set(conformers)) == 4


# ---------------------------------------------------------------------------
# Feature building helpers
# ---------------------------------------------------------------------------


class TestFeatureBuilding:
    def test_build_spec_features_geometry_aware_varies_with_site(self):
        slab = make_slab(nx=2, ny=2, n_layers=3)
        conformers = [make_water()]
        config = AdsorptionConfig(num_conformers=1, num_placements=12)
        specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=12)
        assert len(specs) >= 4
        X, valid_indices = build_spec_features_geometry_aware(
            specs,
            conformers,
            slab,
            config,
            smiles="O",
            molecule="water",
            surface_id="test_surface",
        )
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == len(valid_indices)
        assert X.shape[0] > 0
        # Ensure candidate geometry is not collapsed to a constant placeholder.
        assert X[["x", "y", "z"]].drop_duplicates().shape[0] > 1

    def test_build_spec_features_fills_materialization_cache(self):
        slab = make_slab(nx=2, ny=2, n_layers=3)
        conformers = [make_water()]
        config = AdsorptionConfig(num_conformers=1, num_placements=8)
        specs = enumerate_placement_specs(conformers, slab, config, "O", n_desired=8)
        cache: dict = {}
        X, valid_indices = build_spec_features_geometry_aware(
            specs,
            conformers,
            slab,
            config,
            smiles="O",
            materialization_cache=cache,
        )
        assert X.shape[0] == len(valid_indices)
        assert len(cache) == X.shape[0]
        for pid, (ads, desc) in cache.items():
            assert pid == desc.placement_index
            assert len(ads) > 0

    def test_materialize_backfill_fills_batch_target(self, monkeypatch):
        """Primary materialization misses are replaced from backfill specs."""
        from metalsurfer.workflow import placement_fill as fill_mod
        from metalsurfer.workflow.shared import PlacementFailureEvent

        primary = [_placement_spec(0), _placement_spec(1)]
        backfill = [_placement_spec(2), _placement_spec(3)]
        fail_ids = {0}

        def fake_materialize(**kwargs):
            combined = []
            ids = []
            descs = []
            failures = []
            for spec in kwargs["specs"]:
                if spec.placement_index in fail_ids:
                    failures.append(
                        PlacementFailureEvent(
                            placement_id=spec.placement_index,
                            stage="generation",
                            reason="too_close",
                            descriptor=None,
                        )
                    )
                    continue
                desc = make_placement_descriptor(placement_id=spec.placement_index)
                combined.append(Atoms("H"))
                ids.append(spec.placement_index)
                descs.append(desc)
            return combined, ids, descs, failures

        monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)
        result = fill_mod.materialize_specs_filling_target(
            primary_specs=primary,
            backfill_specs=backfill,
            n_target=2,
            conformers=[make_water()],
            slab_atoms=make_slab(),
            calculator=None,
            config=AdsorptionConfig(
                num_placements=2,
                placement_retry_oversample_max=1.0,
            ),
            smiles="O",
            site_context=None,
        )
        assert len(result.combined) == 2
        assert set(result.placement_ids) == {1, 2}

    def test_record_from_descriptor_roundtrip(self):
        d = _bayesian_descriptor(7)
        record = PlacementRecord.from_descriptor(d, molecule="mol", smiles="C")
        assert record.placement_id == 7
        assert record.descriptor.tilt_deg == d.tilt_deg
        assert record.descriptor.x == d.x
        assert record.descriptor.quat_w == 1.0
        assert record.descriptor.quat_x == 0.0

    def test_record_from_spec_roundtrip(self):
        s = _placement_spec(3)
        record = PlacementRecord.from_spec(s, molecule="mol", smiles="C")
        assert record.placement_id == 3
        assert record.descriptor.tilt_deg == s.tilt_deg
        assert record.descriptor.quat_w == 1.0
        assert record.descriptor.quat_z == 0.0

    def test_enumerated_pool_capacity_matches_estimate(self):
        conformer = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        conformer.set_cell([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 20.0]])
        conformer.set_pbc([True, True, True])
        slab = Atoms(
            "Cu6",
            positions=[
                [0.0, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.0, 2.5, 0.0],
                [2.5, 2.5, 0.0],
                [1.25, 1.25, 0.0],
                [3.75, 1.25, 0.0],
            ],
            cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]],
            pbc=[True, True, True],
        )
        config = AdsorptionConfig(num_conformers=1, num_placements=10)
        capacity = estimate_placement_spec_capacity([conformer], slab, config, "CO")
        specs = enumerate_placement_specs(
            [conformer], slab, config, "CO", n_desired=capacity
        )
        assert capacity > 0
        assert len(specs) == capacity


# ---------------------------------------------------------------------------
# score_and_select integration
# ---------------------------------------------------------------------------


class TestScoreAndSelect:
    def test_end_to_end_selection(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, n_estimators=20)
        selected = score_and_select(
            model, X, batch_size=5, kappa=1.0, evaluated_indices={0, 1, 2}
        )
        assert len(selected) == 5
        assert 0 not in selected
        assert 1 not in selected
        assert 2 not in selected
        assert len(set(selected)) == 5

    def test_kappa_zero_is_greedy(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, n_estimators=20)
        greedy = score_and_select(model, X, batch_size=5, kappa=0.0)
        mu, _ = predict_with_uncertainty(model, X)
        # First pick is pure greedy; later picks use local penalization for diversity.
        assert greedy[0] == int(np.argmin(mu))
        assert len(greedy) == 5
        assert len(set(greedy)) == 5

    def test_batch_diverse_differs_from_pure_topk(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, n_estimators=20, random_state=0)
        mu, sigma = predict_with_uncertainty(model, X)
        scores = lcb_scores(mu, sigma, kappa=1.0)
        topk = select_candidates(scores, batch_size=5)
        diverse = select_candidates_batch_diverse(
            scores, X, batch_size=5, higher_is_better=False
        )
        assert diverse[0] == topk[0]
        assert len(diverse) == 5
        assert len(set(diverse)) == 5

    def test_high_kappa_favours_uncertain(self):
        X, y = _make_synthetic_training_data(40)
        model = train_surrogate(X, y, n_estimators=20)
        conservative = score_and_select(model, X, batch_size=5, kappa=0.0)
        exploratory = score_and_select(model, X, batch_size=5, kappa=10.0)
        assert conservative != exploratory

    def test_ei_and_pi_acquisition_require_f_best(self):
        X, y = _make_synthetic_training_data(20)
        model = train_surrogate(X, y, n_estimators=10)
        with pytest.raises(ValueError, match="f_best"):
            score_and_select(model, X, batch_size=3, acquisition="ei", f_best=None)
        selected_ei = score_and_select(
            model, X, batch_size=3, acquisition="ei", f_best=float(y.min())
        )
        selected_pi = score_and_select(
            model, X, batch_size=3, acquisition="pi", f_best=float(y.min())
        )
        assert len(selected_ei) == 3
        assert len(selected_pi) == 3


# ---------------------------------------------------------------------------
# Transfer learning smoke (production ml.bayesian path)
# ---------------------------------------------------------------------------


class TestTransferSmoke:
    @pytest.mark.parametrize(
        "surrogate",
        [None, "ridge", "gradient_boost"],
    )
    def test_build_transfer_surrogate_smoke(self, surrogate):
        X, y = _make_synthetic_training_data(20)
        X_prev = X.iloc[:10].copy()
        y_prev = (y.iloc[:10] - 0.5).to_numpy()
        kwargs = {}
        if surrogate is not None:
            kwargs["surrogate"] = surrogate
        result = build_transfer_surrogate(
            X.iloc[:8],
            y.iloc[:8].to_numpy(),
            X_prev,
            y_prev,
            weight_cap=0.35,
            similarity_lengthscale=1.0,
            min_similarity=0.0,
            mae_tolerance=1.0,
            **kwargs,
        )
        assert result.surrogate is not None
        assert result.transfer_weight_share > 0.0

    def test_build_transfer_surrogate_warns_on_schema_mismatch(self, caplog):
        X, y = _make_synthetic_training_data(20)
        X_current = X.iloc[:8].copy()
        y_current = y.iloc[:8].to_numpy()
        # Prior has an extra column and is missing one relative to current.
        X_prev = X.iloc[:10].copy()
        X_prev = X_prev.drop(columns=[X_prev.columns[0]])
        X_prev["extra_prior_feature"] = np.arange(len(X_prev), dtype=float)

        # Capture directly on the module logger so the assertion is robust to
        # logger-propagation / caplog-handler quirks across environments.
        _module_logger = logging.getLogger("metalsurfer.ml.bayesian")
        _captured: list[logging.LogRecord] = []
        _handler = CaptureHandler(_captured)
        _module_logger.addHandler(_handler)
        try:
            result = build_transfer_surrogate(
                X_current,
                y_current,
                X_prev,
                y.iloc[:10].to_numpy(),
                weight_cap=0.35,
                similarity_lengthscale=1.0,
                min_similarity=0.0,
                mae_tolerance=1.0,
            )
        finally:
            _module_logger.removeHandler(_handler)
        assert result.surrogate is not None
        assert result.transfer_weight_share > 0.0
        assert any("prior feature columns" in r.message for r in _captured)

    def test_build_transfer_surrogate_rejects_gaussian_process(self):
        X, y = _make_synthetic_training_data(20)
        with pytest.raises(ValueError, match="transfer-capable"):
            build_transfer_surrogate(
                X.iloc[:8],
                y.iloc[:8].to_numpy(),
                X.iloc[:10],
                y.iloc[:10].to_numpy(),
                surrogate="gaussian_process",  # type: ignore[arg-type]
            )

    def test_windowed_bo_step_memories_keeps_recent_steps(self):
        memories = [
            BOStepMemory(observed_X_rows=[{"x": float(i)}], observed_y=[float(i)])
            for i in range(4)
        ]
        windowed = windowed_bo_step_memories(memories, window=2)
        assert windowed is not None
        assert len(windowed.observed_X_rows) == 2
        assert windowed.observed_X_rows[0]["x"] == 2.0
        assert windowed.observed_X_rows[1]["x"] == 3.0
        assert windowed.step_ages == [1, 0]

    def test_prior_recency_weights_decay_with_age(self):
        weights = prior_recency_weights([0, 1, 2], lengthscale=1.0)
        assert weights[0] > weights[1] > weights[2]

    def test_prior_placement_downweight_prefers_far_sites(self):
        priors = pd.DataFrame(
            [
                {"x": 0.05, "y": 0.0, "z": 0.0},
                {"x": 5.0, "y": 0.0, "z": 0.0},
                {"x": 2.0, "y": 1.0, "z": 0.0},
            ]
        )
        placement = pd.DataFrame([{"x": 0.0, "y": 0.0, "z": 0.0}])
        weights = prior_placement_downweight(
            priors, placement, lengthscale=1.0, floor=0.0
        )
        assert weights[0] < weights[1]

    def test_prior_placement_downweight_uses_all_committed_rows(self):
        placements = pd.DataFrame(
            [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 10.0, "y": 0.0, "z": 0.0},
            ]
        )
        probes = pd.DataFrame(
            [
                {"x": 0.05, "y": 0.0, "z": 0.0},
                {"x": 10.05, "y": 0.0, "z": 0.0},
                {"x": 5.0, "y": 5.0, "z": 0.0},
            ]
        )
        weights = prior_placement_downweight(
            probes, placements, lengthscale=1.0, floor=0.0
        )
        assert weights[0] < weights[2]
        assert weights[1] < weights[2]

    def test_prior_similarity_to_current_prefers_nearby(self):
        X, _ = _make_synthetic_training_data(5)
        current = X.iloc[[0, 1]].copy()
        near = X.iloc[[0]].copy()
        near.iloc[0, 0] = float(X.iloc[0, 0]) + 0.05
        far = X.iloc[[4]].copy()
        far.iloc[0, 0] = float(X.iloc[0, 0]) + 5.0
        near_s = prior_similarity_to_current(near, current, lengthscale=0.5)
        far_s = prior_similarity_to_current(far, current, lengthscale=0.5)
        assert near_s[0] > far_s[0]

    def test_prior_proximity_weights_smoke(self):
        X, _ = _make_synthetic_training_data(5)
        near = X.iloc[[0]].copy()
        near.iloc[0, 0] = float(X.iloc[0, 0]) + 0.05
        far = X.iloc[[4]].copy()
        far.iloc[0, 0] = float(X.iloc[0, 0]) + 5.0
        near_w = prior_proximity_weights(near, X.iloc[[0]], lengthscale=0.02, floor=0.0)
        far_w = prior_proximity_weights(far, X.iloc[:1], lengthscale=10.0, floor=0.0)
        assert near_w[0] < far_w[0]

    def test_occupancy_fallback_downweights_clustered_priors(self):
        """Fallback occupancy is 1 - proximity(exclude_self), matching build_transfer."""
        clustered = pd.DataFrame(
            [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.1, "y": 0.0, "z": 0.0},
                {"x": 10.0, "y": 0.0, "z": 0.0},
            ]
        )
        # prior_placement_X is None → invert proximity to other prior rows.
        occupancy = np.maximum(
            0.0,
            1.0
            - prior_proximity_weights(clustered, clustered, lengthscale=1.0, floor=0.0),
        )
        assert occupancy[0] < occupancy[2]
        assert occupancy[1] < occupancy[2]
        assert occupancy[2] > occupancy[0] + 0.3

    def test_transfer_similarity_ignores_conformer_index_vs_translation(self):
        """Same pose, Δconformer must not look as far as a multi-Å translation."""
        base = {
            "x": 0.0,
            "y": 0.0,
            "z": 2.0,
            "conformer_index": 0.0,
            "quat_w": 1.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": 0.0,
        }
        current = pd.DataFrame([base])
        same_pose_diff_conf = pd.DataFrame([{**base, "conformer_index": 5.0}])
        translated = pd.DataFrame([{**base, "x": 5.0}])
        sim_conf = prior_similarity_to_current(
            same_pose_diff_conf, current, lengthscale=4.0
        )
        sim_trans = prior_similarity_to_current(translated, current, lengthscale=4.0)
        assert sim_conf[0] > sim_trans[0]
        assert sim_conf[0] > 0.9


class TestTransferTolerance:
    """QC #8 — ``mae_tolerance`` gating (default widened 0.0 -> 0.05 eV).

    A tiny positive MAE delta must not accrue a "bad round" once the tolerance
    covers it, but the same delta still increments under the old 0.0 tolerance.
    """

    def _slow_prior_data(self):
        rng = np.random.default_rng(0)
        n = 40
        x = np.linspace(0.0, 1.0, n)
        y_current = x + 0.01 * rng.standard_normal(n)
        # Mildly degraded prior: same linear trend but offset by a constant.
        # Out-of-fold this makes the transferred model marginally (but
        # consistently) worse than the current-only baseline -> a small positive
        # MAE delta, used to exercise the tolerance gate below.
        y_prior = x + 0.01 * rng.standard_normal(n) + 0.08
        X_current = pd.DataFrame({"f": x})
        X_prior = pd.DataFrame({"f": x})
        return X_current, y_current, X_prior, y_prior

    def test_config_default_tolerance_is_0_05(self):
        assert AdsorptionConfig().bo.transfer.mae_tolerance == 0.05

    def test_tiny_positive_delta_tolerated_by_widened_tolerance(self):
        Xc, yc, Xp, yp = self._slow_prior_data()
        strict = build_transfer_surrogate(
            Xc, yc, Xp, yp, mae_tolerance=0.0, random_state=42
        )
        tolerant = build_transfer_surrogate(
            Xc, yc, Xp, yp, mae_tolerance=0.05, random_state=42
        )
        # The prior genuinely makes the transferred model a touch worse.
        assert strict.transfer_mae_delta > 0.0
        # Old strict tolerance (0.0) counts any positive delta as a bad round.
        assert strict.transfer_bad_rounds == 1
        # Widened tolerance (0.05 eV) absorbs this tiny delta -> no bad round.
        assert tolerant.transfer_mae_delta < 0.05
        assert tolerant.transfer_bad_rounds == 0
        assert tolerant.transfer_disabled is False

    def test_large_delta_still_increments_under_widened_tolerance(self):
        rng = np.random.default_rng(0)
        n = 40
        x = np.linspace(0.0, 1.0, n)
        y_current = x + 0.01 * rng.standard_normal(n)
        # Opposite trend: transfer clearly degrades the model.
        y_prior = (1.0 - x) + 0.05 * rng.standard_normal(n)
        Xc = pd.DataFrame({"f": x})
        Xp = pd.DataFrame({"f": x})
        result = build_transfer_surrogate(
            Xc, y_current, Xp, y_prior, mae_tolerance=0.05, random_state=42
        )
        assert result.transfer_mae_delta > 0.05
        assert result.transfer_bad_rounds == 1
        assert result.transfer_used_this_round is False
        assert result.transfer_disabled is False


# ---------------------------------------------------------------------------
# Integration: BO two generations on surface with defects/doping
# ---------------------------------------------------------------------------


@gpu_mlip_test
def test_bayesian_two_generations_on_defect_surface(tmp_path):
    """BO smoke test for two generations on an adatom-defect surface.

    Generation 1: bo.initial_random placements at random.
    Generation 2: bo.batch_size placements selected by acquisition (default EI;
    falls back to LCB until a finite best E_ads exists).
    """
    from metalsurfer.optimization import setup_single_model
    from metalsurfer.surface_prep import SlabContainer, deposit_adatoms
    from metalsurfer.workflow import (
        calculate_reference_energies,
        process_molecule_bayesian,
    )

    base = SlabContainer(make_slab(nx=4, ny=4, n_layers=3))
    slab = deposit_adatoms(
        base,
        "Sn",
        coverage_fraction=0.2,
        seed=42,
        results_dir=str(tmp_path),
        relaxation_mode="none",
    )

    n_placements = 8
    config = AdsorptionConfig(
        bo=BOConfig(
            initial_random=n_placements,
            batch_size=n_placements,
            total_budget=2,
        ),
        seed=42,
        num_conformers=2,
        num_placements=3 * n_placements,
        device="cuda",
        # Tiny defect slab: allow near-contact survivors so BO can finish two rounds.
        skip_desorption_check=True,
        min_interatomic_distance=0.45,
        stage1_steps=75,
        stage2_steps=200,
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab,
        calculator,
        ["water"],
        ["O"],
        ts_model=ts_model,
        config=config,
    )
    outcome = process_molecule_bayesian(
        "O",
        "water",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="defect_test",
    )
    assert outcome is not None, "BO pipeline should return an outcome"
    results = outcome.results
    assert len(results) >= 1, (
        "BO two generations on defect/doped surface should yield at least one valid "
        "binding energy result"
    )
    for r in results:
        assert hasattr(r, "energy_adsorption") and hasattr(r, "placement_id")
        assert isinstance(r.energy_adsorption, (int, float))
        assert np.isfinite(r.energy_adsorption)
        assert -5.0 <= r.energy_adsorption < 2.0, (
            f"E_ads should be in a physical binding window [-5, 2) eV, got {r.energy_adsorption}"
        )
        assert np.isfinite(r.distance) and r.distance > 0.5, (
            f"Adsorbate–surface distance should be finite and >0.5 Å, got {r.distance}"
        )
    assert min(r.energy_adsorption for r in results) < 0.0, (
        "Best E_ads should be favorable (negative) on this defect smoke"
    )
    assert any(1.0 <= float(r.distance) <= 4.5 for r in results), (
        "At least one survivor should sit in a typical binding distance window"
    )
    assert len(set(r.placement_id for r in results)) == len(results), (
        "No duplicate placement_id in results"
    )


# ---------------------------------------------------------------------------
# Migrated regression tests (from tests/test_tier1_regressions.py)
# ---------------------------------------------------------------------------


def _feature_frame(values):
    return pd.DataFrame({"x": np.asarray(values, dtype=float)})


_BO_FEATURES = [
    "x",
    "y",
    "z",
    "conformer_index",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
]


def _bo_training_data(n=24, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(_BO_FEATURES))), columns=_BO_FEATURES)
    y = 1.5 * X["x"] - 0.7 * X["z"] + rng.normal(scale=0.1, size=n)
    return X, np.asarray(y, dtype=float)


def test_cumulative_refit_training_set_aligns_weights_with_rows():
    """Regression: weights were built current-first, rows prior-first.

    Both orderings have the same length so nothing raised; prior rows silently
    received weight 1.0 and the current observations received the decayed prior
    weights, inverting the ``weight_cap`` guarantee.
    """
    X_prior = _feature_frame([0.0, 1.0, 2.0, 3.0])
    y_prior = np.array([0.0, 1.0, 2.0, 3.0])
    X_current = _feature_frame([10.0, 11.0])
    y_current = np.array([10.0, 11.0])

    X, y, w = cumulative_refit_training_set(
        X_prior,
        y_prior,
        X_current,
        y_current,
        weight_cap=0.35,
        proximity_lengthscale=1.0,
    )

    assert X["x"].tolist() == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]
    assert y.tolist() == [0.0, 1.0, 2.0, 3.0, 10.0, 11.0]
    # Current observations are the anchor and must carry full weight.
    np.testing.assert_allclose(w[-2:], 1.0)
    # Prior rows are decayed, and their total mass honours weight_cap.
    assert np.all(w[:4] < 1.0)
    assert float(w[:4].sum() / w.sum()) == pytest.approx(0.35, abs=1e-6)


def test_cumulative_refit_training_set_rejects_length_mismatch():
    with pytest.raises(ValueError, match="X_prior/y_prior length mismatch"):
        cumulative_refit_training_set(
            _feature_frame([0.0, 1.0]),
            np.array([0.0]),
            _feature_frame([2.0]),
            np.array([2.0]),
            weight_cap=0.35,
            proximity_lengthscale=1.0,
        )


@pytest.mark.parametrize("surrogate", ["gradient_boost", "ridge"])
def test_expected_improvement_is_not_identically_zero(surrogate):
    """Regression: sigma came from *in-sample* residuals of an interpolator.

    ``HistGradientBoostingRegressor`` essentially interpolates its training
    rows, so residual RMSE collapsed to the 1e-3 floor and EI evaluated to
    exactly 0.0 for every candidate. Acquisition then fell through to
    ``np.argmax`` over a constant array, i.e. pool-ordered sampling rather than
    Bayesian optimisation.
    """
    X, y = _bo_training_data()
    rng = np.random.default_rng(7)
    X_pool = pd.DataFrame(
        rng.normal(size=(200, len(_BO_FEATURES))), columns=_BO_FEATURES
    )

    model = train_surrogate(X, y, surrogate=surrogate)
    mu, sigma = predict_with_uncertainty(model, X_pool)
    ei = ei_scores(mu, sigma, float(np.min(y)))

    assert np.all(sigma > 0.0)
    assert float(np.max(ei)) > 0.0
    assert int(np.count_nonzero(ei)) > 1, "EI must discriminate between candidates"


def test_residual_sigma_uses_a_scaler_shared_with_predict_time():
    """The NN-distance term must be computed in one consistent feature space."""
    X, y = _bo_training_data()
    model = train_surrogate(X, y, surrogate="gradient_boost")
    regressor = model.named_steps["regressor"]

    assert hasattr(regressor, "bo_sigma_scaler_")
    scaled = regressor.bo_X_train_scaled_
    # Standardised: zero mean, unit variance per column.
    np.testing.assert_allclose(scaled.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(scaled.std(axis=0), 1.0, atol=1e-8)


def test_transfer_gate_rejects_a_misleading_prior():
    """Regression: the gate compared in-sample MAE, so it measured capacity.

    With the default ``gradient_boost`` surrogate both models interpolated the
    current step, the delta came out at ~1e-4 (far below ``mae_tolerance``), and
    a prior encoding the opposite relationship was trusted indefinitely.
    """
    from metalsurfer.ml.bayesian import build_transfer_surrogate

    rng_c = np.random.default_rng(2)
    X_current = pd.DataFrame(
        rng_c.normal(size=(20, len(_BO_FEATURES))), columns=_BO_FEATURES
    )
    y_current = np.asarray(X_current["x"] - 0.4 * X_current["y"], dtype=float)

    rng_p = np.random.default_rng(3)
    X_prior = pd.DataFrame(
        rng_p.normal(size=(30, len(_BO_FEATURES))), columns=_BO_FEATURES
    )
    # Deliberately contradicts the current-step relationship.
    y_prior = np.asarray(-3.0 * X_prior["x"] + 5.0, dtype=float)

    result = build_transfer_surrogate(
        X_current,
        y_current,
        X_prior,
        y_prior,
        surrogate="gradient_boost",
        min_similarity=0.0,
        mae_tolerance=0.05,
        trust_patience=1,
    )
    assert result.transfer_mae_delta > 0.05
    assert result.transfer_disabled
    assert not result.transfer_used_this_round


def test_transfer_gate_accepts_a_consistent_prior():
    from metalsurfer.ml.bayesian import build_transfer_surrogate

    def _sample(seed):
        rng = np.random.default_rng(seed)
        X = pd.DataFrame(rng.normal(size=(20, len(_BO_FEATURES))), columns=_BO_FEATURES)
        y = np.asarray(X["x"] - 0.4 * X["y"], dtype=float)
        return X, y

    X_current, y_current = _sample(2)
    X_prior, y_prior = _sample(7)

    result = build_transfer_surrogate(
        X_current,
        y_current,
        X_prior,
        y_prior,
        surrogate="gradient_boost",
        min_similarity=0.0,
        mae_tolerance=0.05,
        trust_patience=1,
    )
    assert not result.transfer_disabled
    assert result.transfer_used_this_round


def test_bo_rng_seed_decorrelates_by_slab_atom_count():
    """Coverage growth must change the exploration RNG stream (L5)."""
    seed = 42
    rng_a = np.random.RandomState(int(seed + 1_000_003 * 36) % (2**31))
    rng_b = np.random.RandomState(int(seed + 1_000_003 * 39) % (2**31))
    draw_a = rng_a.choice([0, 1, 2, 3, 4], size=2, replace=False).tolist()
    draw_b = rng_b.choice([0, 1, 2, 3, 4], size=2, replace=False).tolist()
    assert draw_a != draw_b


def test_cumulative_refit_transfer_weight_share_from_weights():
    """Prior weight fraction matches transfer_weight_share formula (L6)."""
    X_prior = _feature_frame([0.0, 1.0, 2.0, 3.0])
    y_prior = np.array([0.0, 1.0, 2.0, 3.0])
    X_current = _feature_frame([10.0, 11.0])
    y_current = np.array([10.0, 11.0])
    _, _, refit_weights = cumulative_refit_training_set(
        X_prior,
        y_prior,
        X_current,
        y_current,
        weight_cap=0.35,
        proximity_lengthscale=1.0,
    )
    n_prior = len(X_prior)
    share = float(np.sum(refit_weights[:n_prior]) / np.sum(refit_weights))
    assert share == pytest.approx(0.35, abs=1e-6)

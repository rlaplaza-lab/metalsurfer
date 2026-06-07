"""Tests for Bayesian optimisation placement selection pipeline."""

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.bayesian import (
    EnsembleRegressor,
    build_candidate_features,
    build_spec_features_geometry_aware,
    ei_scores,
    lcb_scores,
    matern_length_scale_for_n_features,
    predict_with_uncertainty,
    score_and_select,
    select_candidates,
    train_surrogate,
)
from metalsurfer.ml.features import extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.models import PlacementDescriptor, PlacementSpec
from metalsurfer.placement import (
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
)
from tests.factories import make_random_placement_records
from tests.optional_deps import cuda_available, has_mlip_stack

from .conftest import make_placement_descriptor, make_slab, make_water


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
        assert np.all(sigma >= 0)
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
        assert np.all(sigma == 0.0)

    def test_train_surrogate_accepts_sample_weights(self):
        X, y = _make_synthetic_training_data(30)
        weights = np.ones(len(y), dtype=float)
        weights[:5] = 2.0
        model = train_surrogate(X, y, n_estimators=10, sample_weight=weights)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (30,)
        assert sigma.shape == (30,)

    def test_train_surrogate_rejects_sample_weight_for_non_tree(self):
        X, y = _make_synthetic_training_data(20)
        w = np.ones(20, dtype=float)
        for sur in ("ridge", "gradient_boost", "gaussian_process"):
            with pytest.raises(ValueError, match="sample_weight"):
                train_surrogate(X, y, surrogate=sur, sample_weight=w)

    def test_ensemble_accepts_sample_weight_for_tree_members(self):
        X, y = _make_synthetic_training_data(20)
        w = np.ones(20, dtype=float)
        model = train_surrogate(
            X, y, surrogate="ensemble", n_estimators=5, sample_weight=w
        )
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (20,)
        assert sigma.shape == (20,)

    def test_gaussian_process_matern_length_scale(self):
        X, y = _make_synthetic_training_data(25)
        n_features = X.shape[1]
        expected = matern_length_scale_for_n_features(n_features)
        assert expected == pytest.approx(np.sqrt(n_features))
        model = train_surrogate(X, y, surrogate="gaussian_process", random_state=7)
        reg = model.named_steps["regressor"]
        kernel = reg.kernel_
        assert float(kernel.k2.length_scale) == pytest.approx(expected)

    def test_gaussian_process_predict_with_uncertainty(self):
        X, y = _make_synthetic_training_data(25)
        model = train_surrogate(X, y, surrogate="gaussian_process", random_state=7)
        mu, sigma = predict_with_uncertainty(model, X)
        assert mu.shape == (25,)
        assert sigma.shape == (25,)
        assert np.all(sigma >= 0)
        assert np.any(sigma > 0)

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
    def test_ei_scores_zero_sigma_is_deterministic_improvement(self):
        """When sigma=0, EI reduces to max(0, f_best - mu) (see ei_scores)."""
        mu = np.array([1.0, 2.0, -0.5])
        sigma = np.array([0.0, 0.0, 0.0])
        f_best = 0.0
        ei = ei_scores(mu, sigma, f_best=f_best, xi=1e-6)
        expected = np.maximum(0.0, f_best - mu)
        np.testing.assert_allclose(ei, expected, rtol=0, atol=1e-5)

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


# ---------------------------------------------------------------------------
# Feature building helpers
# ---------------------------------------------------------------------------


class TestFeatureBuilding:
    def test_build_candidate_features(self):
        descriptors = [_bayesian_descriptor(i) for i in range(5)]
        X = build_candidate_features(descriptors, molecule="test", smiles="C")
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == 5
        assert X.shape[1] > 0

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

    def test_record_from_descriptor_roundtrip(self):
        d = _bayesian_descriptor(7)
        record = PlacementRecord.from_descriptor(d, molecule="mol", smiles="C")
        assert record.placement_id == 7
        assert record.tilt_deg == d.tilt_deg
        assert record.x == d.x
        assert record.quat_w == 1.0
        assert record.quat_x == 0.0

    def test_record_from_spec_roundtrip(self):
        s = _placement_spec(3)
        record = PlacementRecord.from_spec(s, molecule="mol", smiles="C")
        assert record.placement_id == 3
        assert record.tilt_deg == s.tilt_deg
        assert record.quat_w == 1.0
        assert record.quat_z == 0.0

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
        expected_order = np.argsort(mu)[:5].tolist()
        assert greedy == expected_order

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
# Integration: BO two generations on surface with defects/doping
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.mlip
@pytest.mark.gpu
@pytest.mark.no_fork
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
@pytest.mark.skipif(
    not cuda_available,
    reason="CUDA GPU required; run in conda env metalsurfer with GPU",
)
def test_bayesian_two_generations_on_defect_surface(tmp_path):
    """BO smoke test for two generations on an adatom-defect surface.

    Generation 1: bo_initial_random placements at random.
    Generation 2: bo_batch_size placements selected by UCB acquisition.
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
    )

    n_placements = 3
    config = AdsorptionConfig(
        bo_enabled=True,
        bo_initial_random=n_placements,
        bo_batch_size=n_placements,
        bo_total_budget=2 * n_placements,
        seed=42,
        num_conformers=2,
        num_placements=2 * n_placements,
        device="cuda",
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
    results = process_molecule_bayesian(
        "O",
        "water",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="defect_test",
    )
    assert results is not None, "BO pipeline should return a list (possibly empty)"
    assert len(results) >= 1, (
        "BO two generations on defect/doped surface should yield at least one valid "
        "binding energy result"
    )
    for r in results:
        assert hasattr(r, "energy_adsorption") and hasattr(r, "placement_id")
        assert isinstance(r.energy_adsorption, (int, float))
    assert len(set(r.placement_id for r in results)) == len(results), (
        "No duplicate placement_id in results"
    )

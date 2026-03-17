"""Tests for Bayesian optimisation placement selection pipeline."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.bayesian import (
    _record_from_descriptor,
    _record_from_spec,
    build_candidate_features,
    build_spec_features,
    predict_with_uncertainty,
    score_and_select,
    select_candidates,
    train_surrogate,
    ucb_scores,
)
from metalsurfer.ml.features import extract_features
from metalsurfer.ml.schema import ComputationContext, PlacementRecord
from metalsurfer.models import PlacementDescriptor, PlacementSpec
from tests.optional_deps import cuda_available, has_mlip_stack


def _dummy_descriptor(idx: int = 0) -> PlacementDescriptor:
    return PlacementDescriptor(
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
        x=float(1.0 + idx * 0.1),
        y=float(2.0 - idx * 0.1),
        z=2.5,
        shape="round",
    )


def _dummy_spec(idx: int = 0) -> PlacementSpec:
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


def _make_synthetic_training_data(n: int = 40):
    """Build a synthetic (X, y) dataset from PlacementRecords."""
    rng = np.random.RandomState(42)
    records = []
    for i in range(n):
        z = float(rng.uniform(2, 3))
        tilt = float(rng.choice([0, 15, 30, 45, 60, 90]))
        e_ads = -0.5 * z + 0.01 * tilt + float(rng.normal(0, 0.1))
        records.append(
            PlacementRecord(
                molecule="test",
                smiles="C",
                surface_id="test",
                placement_id=i,
                conformer_index=i % 3,
                orientation_type=str(
                    rng.choice(["parallel", "EN-down", "vertical", "round"])
                ),
                face_flip=False,
                en_atom_index=None,
                site_index=i % 5,
                site_type=str(rng.choice(["atop", "bridge", "hollow"])),
                tilt_deg=tilt,
                azimuth_deg=float(rng.choice([0, 45, 90, 135, 180])),
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                x=float(rng.uniform(-4, 4)),
                y=float(rng.uniform(-4, 4)),
                z=z,
                shape=str(rng.choice(["linear", "flat", "round"])),
                energy_adsorption=e_ads,
                context=ComputationContext(),
            )
        )
    rows = [extract_features(r) for r in records]
    X = pd.DataFrame(rows)
    y = pd.Series([r.energy_adsorption for r in records])
    return X, y


# ---------------------------------------------------------------------------
# BO config validation
# ---------------------------------------------------------------------------


class TestBOConfig:
    def test_defaults_backward_compatible(self):
        c = AdsorptionConfig()
        assert c.bo_enabled is False
        assert c.bo_initial_random == 10
        assert c.bo_batch_size == 10
        assert c.bo_total_budget == 100
        assert c.bo_ucb_kappa == 1.0
        assert c.bo_acquisition == "lcb"
        assert c.bo_surrogate == "random_forest"
        assert c.bo_candidate_pool_size is None

    def test_enabled_valid(self):
        c = AdsorptionConfig(
            bo_enabled=True,
            bo_initial_random=10,
            bo_batch_size=5,
            bo_total_budget=50,
            bo_ucb_kappa=2.0,
        )
        assert c.bo_total_budget == 50

    def test_initial_exceeds_budget_raises(self):
        with pytest.raises(ValueError, match="bo_initial_random"):
            AdsorptionConfig(
                bo_enabled=True, bo_initial_random=200, bo_total_budget=100
            )

    def test_negative_kappa_raises(self):
        with pytest.raises(ValueError, match="bo_ucb_kappa"):
            AdsorptionConfig(bo_enabled=True, bo_ucb_kappa=-1.0)

        with pytest.raises(ValueError, match="bo_acquisition"):
            AdsorptionConfig(bo_enabled=True, bo_acquisition="invalid")

        with pytest.raises(ValueError, match="bo_surrogate"):
            AdsorptionConfig(bo_enabled=True, bo_surrogate="invalid")

    def test_candidate_pool_size_validation(self):
        c = AdsorptionConfig(bo_enabled=True, bo_candidate_pool_size=500)
        assert c.bo_candidate_pool_size == 500
        with pytest.raises(ValueError):
            AdsorptionConfig(bo_enabled=True, bo_candidate_pool_size=0)


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

    def test_seed_reproducibility(self):
        X, y = _make_synthetic_training_data(40)
        m1 = train_surrogate(X, y, n_estimators=20, random_state=123)
        m2 = train_surrogate(X, y, n_estimators=20, random_state=123)
        mu1, _ = predict_with_uncertainty(m1, X)
        mu2, _ = predict_with_uncertainty(m2, X)
        np.testing.assert_array_equal(mu1, mu2)

    def test_predict_with_dataframe_columns(self):
        X, y = _make_synthetic_training_data(20)
        model = train_surrogate(X, y, n_estimators=10)
        mu, sigma = predict_with_uncertainty(model, X)
        assert not np.any(np.isnan(mu))

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


# ---------------------------------------------------------------------------
# UCB acquisition
# ---------------------------------------------------------------------------


class TestAcquisition:
    def test_ucb_scores_shape(self):
        mu = np.array([1.0, 2.0, 3.0])
        sigma = np.array([0.5, 0.5, 0.5])
        scores = ucb_scores(mu, sigma, kappa=1.0)
        assert scores.shape == (3,)
        np.testing.assert_array_almost_equal(scores, [0.5, 1.5, 2.5])

    def test_ucb_lower_is_better(self):
        mu = np.array([-2.0, 0.0, 1.0])
        sigma = np.array([1.0, 0.1, 0.1])
        scores = ucb_scores(mu, sigma, kappa=2.0)
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
        descriptors = [_dummy_descriptor(i) for i in range(5)]
        X = build_candidate_features(descriptors, molecule="test", smiles="C")
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == 5
        assert X.shape[1] > 0

    def test_build_spec_features(self):
        specs = [_dummy_spec(i) for i in range(5)]
        X = build_spec_features(specs, molecule="test", smiles="C")
        assert isinstance(X, pd.DataFrame)
        assert X.shape[0] == 5

    def test_record_from_descriptor_roundtrip(self):
        d = _dummy_descriptor(7)
        record = _record_from_descriptor(d, molecule="mol", smiles="C")
        assert record.placement_id == 7
        assert record.tilt_deg == d.tilt_deg
        assert record.x == d.x

    def test_record_from_spec_roundtrip(self):
        s = _dummy_spec(3)
        record = _record_from_spec(s, molecule="mol", smiles="C")
        assert record.placement_id == 3
        assert record.tilt_deg == s.tilt_deg


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
# Budget accounting
# ---------------------------------------------------------------------------


class TestBudgetAccounting:
    def test_initial_plus_batches_within_budget(self):
        config = AdsorptionConfig(
            bo_enabled=True,
            bo_initial_random=10,
            bo_batch_size=5,
            bo_total_budget=30,
        )
        total = config.bo_initial_random
        remaining = config.bo_total_budget - total
        batches = 0
        while remaining > 0:
            batch = min(config.bo_batch_size, remaining)
            total += batch
            remaining -= batch
            batches += 1
        assert total <= config.bo_total_budget
        assert total == config.bo_total_budget

    def test_initial_equals_budget_no_bo_iterations(self):
        config = AdsorptionConfig(
            bo_enabled=True,
            bo_initial_random=50,
            bo_batch_size=10,
            bo_total_budget=50,
        )
        remaining = config.bo_total_budget - config.bo_initial_random
        assert remaining == 0


# ---------------------------------------------------------------------------
# Best-energy monotonicity
# ---------------------------------------------------------------------------


class TestBestEnergyTracking:
    def test_best_is_monotonically_nonincreasing(self):
        rng = np.random.RandomState(42)
        energies_over_time = rng.randn(50)
        best_so_far = float("inf")
        history = []
        for e in energies_over_time:
            best_so_far = min(best_so_far, e)
            history.append(best_so_far)
        for i in range(1, len(history)):
            assert history[i] <= history[i - 1]


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
@pytest.mark.parametrize("surface_kind", ["adatom_defect", "alloy_doped"])
def test_bayesian_two_generations_on_defect_surface(surface_kind, tmp_path):
    """BO runs two generations of n_placements on a surface with defects/doping.

    Generation 1: bo_initial_random placements at random.
    Generation 2: bo_batch_size placements selected by UCB acquisition.
    Surfaces: adatom_defect (Sn on Ru) or alloy_doped (RuCu).
    """
    from metalsurfer.optimization import setup_single_model
    from metalsurfer.surfaces import SlabContainer, deposit_adatoms, substitute_alloy
    from metalsurfer.workflow import (
        calculate_reference_energies,
        process_molecule_bayesian,
    )

    from .conftest import make_slab

    base = SlabContainer(make_slab(nx=4, ny=4, n_layers=3))
    if surface_kind == "adatom_defect":
        slab = deposit_adatoms(
            base,
            "Sn",
            coverage_fraction=0.2,
            seed=42,
            results_dir=str(tmp_path),
        )
    else:
        slab = substitute_alloy(
            base,
            "Ru",
            "Cu",
            guest_fraction=0.25,
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

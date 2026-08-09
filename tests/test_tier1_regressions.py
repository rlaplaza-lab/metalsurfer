"""Regression tests for the Tier 1 correctness fixes.

Each test below pins a bug that was silent: it produced no exception, no failing
test and no log line, but changed scientific output, destroyed results, or
disabled a documented safety check. Keep these tests specific to the failure
mode rather than to the surrounding implementation.
"""

import json
import logging
import warnings

import numpy as np
import pandas as pd
import pytest
from ase.build import fcc111, molecule

from metalsurfer._logging import configure_logging
from metalsurfer._numeric_defaults import (
    CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from metalsurfer._utils import cell_has_volume
from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.bayesian import (
    cumulative_refit_training_set,
    ei_scores,
    predict_with_uncertainty,
    train_surrogate,
)
from metalsurfer.placement.geometry import check_initial_contact_quality

# ---------------------------------------------------------------------------
# #1 batched forces are a per-atom tensor, not per-system
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for a torch tensor supporting .detach().cpu().numpy()."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array


class _FakeBatch:
    def __init__(self, forces, system_idx):
        self.forces = _FakeTensor(forces)
        self.system_idx = _FakeTensor(system_idx)


def test_split_forces_by_system_returns_per_system_arrays():
    """``forces`` is an atom attribute; indexing it by system index is wrong.

    Regression: ``forces_list[i]`` returned atom *i*'s ``(3,)`` force vector, so
    the adsorbate force-convergence check downstream sliced an empty array and
    never fired.
    """
    from metalsurfer.optimization._optimize import _split_forces_by_system

    # Two systems: 3 atoms and 5 atoms.
    atom_counts = [3, 5]
    forces = np.arange(8 * 3, dtype=float).reshape(8, 3)
    system_idx = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    batch = _FakeBatch(forces, system_idx)

    per_system = _split_forces_by_system(batch, 2, atom_counts)

    assert per_system is not None
    assert [f.shape for f in per_system] == [(3, 3), (5, 3)]
    np.testing.assert_allclose(per_system[0], forces[:3])
    np.testing.assert_allclose(per_system[1], forces[3:])
    # The pre-fix behaviour would have produced shape (3,) for system 1.
    assert per_system[1].shape != (3,)


def test_split_forces_by_system_raises_on_atom_count_mismatch():
    from metalsurfer.optimization._optimize import _split_forces_by_system

    batch = _FakeBatch(np.zeros((8, 3)), np.array([0, 0, 0, 1, 1, 1, 1, 1]))
    with pytest.raises(RuntimeError, match="could not be split per system"):
        _split_forces_by_system(batch, 2, [4, 4])


# ---------------------------------------------------------------------------
# #2 incremental writes must not destroy previously saved molecules
# ---------------------------------------------------------------------------


def test_merge_preserves_molecules_absent_from_the_new_run(tmp_path):
    """Regression: ``skip_existing`` read the same CSV that was truncated.

    Run 1 wrote {A, B}; run 2 skipped A and B (already present) and then
    overwrote the file with only {C}, permanently losing A and B.
    """
    from metalsurfer.io_results import _merge_preserving_existing_molecules

    path = tmp_path / "adsorption_energies_detailed.csv"
    pd.DataFrame(
        [{"molecule": "A", "E_ads": -1.0}, {"molecule": "B", "E_ads": -2.0}]
    ).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path, pd.DataFrame([{"molecule": "C", "E_ads": -3.0}])
    )
    assert set(merged["molecule"]) == {"A", "B", "C"}


def test_merge_replaces_rows_for_recomputed_molecules(tmp_path):
    from metalsurfer.io_results import _merge_preserving_existing_molecules

    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [{"molecule": "A", "E_ads": -1.0}, {"molecule": "B", "E_ads": -2.0}]
    ).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path, pd.DataFrame([{"molecule": "A", "E_ads": -9.0}])
    )
    assert set(merged["molecule"]) == {"A", "B"}
    assert merged.loc[merged["molecule"] == "A", "E_ads"].tolist() == [-9.0]
    assert merged.loc[merged["molecule"] == "B", "E_ads"].tolist() == [-2.0]


def test_merge_falls_back_when_existing_file_is_corrupt(tmp_path):
    from metalsurfer.io_results import _merge_preserving_existing_molecules

    path = tmp_path / "summary.csv"
    path.write_text("not,a,valid\ncsv\x00row\n")
    new = pd.DataFrame([{"molecule": "C", "E_ads": -3.0}])
    merged = _merge_preserving_existing_molecules(path, new)
    assert set(merged["molecule"]) == {"C"}


# ---------------------------------------------------------------------------
# #3 configure_logging must not retarget a caller's FileHandler
# ---------------------------------------------------------------------------


def test_configure_logging_preserves_file_handlers(tmp_path, monkeypatch):
    """Regression: ``FileHandler`` subclasses ``StreamHandler``.

    The isinstance sweep pointed the caller's file handler at stdout, silently
    killing file logging and dropping the open file object without closing it.
    """
    monkeypatch.setenv("METALSURFER_FORCE_STDOUT_LOGS", "1")
    log_path = tmp_path / "run.log"
    file_handler = logging.FileHandler(log_path)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(file_handler)
    try:
        root.setLevel(logging.INFO)
        logging.getLogger("metalsurfer.test").info("before configure")
        configure_logging()
        logging.getLogger("metalsurfer.test").info("after configure")
        file_handler.flush()

        assert file_handler.stream.name == str(log_path)
        contents = log_path.read_text()
        assert "before configure" in contents
        assert "after configure" in contents
    finally:
        root.removeHandler(file_handler)
        file_handler.close()
        root.setLevel(previous_level)


# ---------------------------------------------------------------------------
# #4 the strict contact gate must accept physical adsorption heights
# ---------------------------------------------------------------------------


def test_strict_contact_gate_accepts_physical_heights_and_rejects_liftoff():
    """Regression: the default was 0.8 A, a *ratio* value in a distance field.

    ``contact_distance`` is an absolute interatomic distance bounded below at
    ~1.7 A by ``check_initial_placement_distance``, so the admissible window was
    empty and ``strict_initial_placement=True`` rejected every placement with
    the misleading reason ``contact_distance_too_large``.
    """
    assert CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM > (
        MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    ), "the strict-contact window must be non-empty"

    slab = fcc111("Pt", (3, 3, 3), vacuum=10.0)
    top_z = float(slab.get_positions()[:, 2].max())
    anchor = slab.get_positions()[-1]

    def _co_at(height):
        co = molecule("CO")
        pos = co.get_positions()
        co.translate([anchor[0], anchor[1], top_z + height - pos[:, 2].min()])
        return co

    for height in (1.8, 2.0, 2.5):
        ok, reason = check_initial_contact_quality(
            _co_at(height), slab, strict_initial_placement=True
        )
        assert ok, f"physical height {height} A rejected: {reason}"

    ok_far, reason_far = check_initial_contact_quality(
        _co_at(8.0), slab, strict_initial_placement=True
    )
    assert not ok_far
    assert reason_far == "contact_distance_too_large"


# ---------------------------------------------------------------------------
# #5 cumulative-refit weights must align with the rows they weight
# ---------------------------------------------------------------------------


def _feature_frame(values):
    return pd.DataFrame({"x": np.asarray(values, dtype=float)})


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


# ---------------------------------------------------------------------------
# #6 sigma must be estimated out of sample so EI is not identically zero
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# #7 the two conformer-selection knobs are no-ops and must say so
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("conformer_sampling", "boltzmann"), ("boltzmann_temperature", 500.0)],
)
def test_deprecated_conformer_knobs_warn(field_name, value):
    with pytest.warns(DeprecationWarning, match="no longer affect"):
        AdsorptionConfig(**{field_name: value})


def test_default_config_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        AdsorptionConfig()


# ---------------------------------------------------------------------------
# #9 the autobatcher accessor must always return a 3-tuple
# ---------------------------------------------------------------------------


def test_get_inflight_autobatcher_returns_triple_when_unavailable(monkeypatch):
    """Regression: a bare ``None`` broke both unpacking call sites.

    ``optimize_isolated_molecules_batched`` does ``...[0]`` and
    ``optimize_adsorbate_slab_batched`` does ``a, b, c = ...``; both raised
    ``TypeError`` when the optional MLIP stack was partly unavailable.
    """
    from metalsurfer.optimization import _cache, _deps

    monkeypatch.setattr(_deps, "InFlightAutoBatcher", None, raising=False)
    result = _cache._get_inflight_autobatcher(object(), 100)

    assert isinstance(result, tuple)
    assert len(result) == 3
    autobatcher, cache_key, reused = result
    assert autobatcher is None
    assert cache_key is None
    assert reused is False
    # Both call-site shapes must work.
    assert result[0] is None
    _a, _b, _c = result


# ---------------------------------------------------------------------------
# #10 left-handed cells are valid and periodic
# ---------------------------------------------------------------------------


def test_cell_has_volume_accepts_left_handed_cells():
    """Regression: five call sites tested ``det > 0``.

    A left-handed cell (e.g. a loaded POSCAR with a flipped axis) has a negative
    determinant but is perfectly valid and periodic. Treating it as degenerate
    silently dropped PBC from distance checks and site enumeration.
    """
    right_handed = np.diag([4.0, 4.0, 12.0])
    left_handed = right_handed.copy()
    left_handed[2, 2] = -12.0

    assert float(np.linalg.det(left_handed)) < 0.0
    assert cell_has_volume(right_handed)
    assert cell_has_volume(left_handed)
    assert not cell_has_volume(np.zeros((3, 3)))
    assert not cell_has_volume(np.diag([4.0, 4.0, 0.0]))
    assert not cell_has_volume(np.zeros((2, 2)))


def test_left_handed_cell_keeps_periodic_distances():
    """A left-handed cell must still use the minimum-image convention."""
    from metalsurfer.placement.geometry import calculate_min_distance

    cell = np.diag([6.0, 6.0, -20.0])
    a = np.array([[0.5, 0.5, 0.0]])
    b = np.array([[5.5, 0.5, 0.0]])  # 1.0 A away across the x boundary

    d = calculate_min_distance(a, b, cell, use_pbc=True, pbc=[True, True, False])
    assert float(d) == pytest.approx(1.0, abs=1e-6)


def test_filters_mic_distances_use_left_handed_cells():
    from ase import Atoms

    from metalsurfer.filters import _mic_pairwise_distances

    cell = np.diag([6.0, 6.0, -20.0])
    atoms = Atoms("H2", positions=[[0.5, 0.5, 0.0], [5.5, 0.5, 0.0]], cell=cell)
    atoms.set_pbc([True, True, False])
    dist = _mic_pairwise_distances(atoms.get_positions(), atoms)
    assert float(dist[0, 1]) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# metadata sanity: the merge helper keeps JSON-serialisable rows intact
# ---------------------------------------------------------------------------


def test_merge_unions_columns_across_schema_versions(tmp_path):
    from metalsurfer.io_results import _merge_preserving_existing_molecules

    path = tmp_path / "detailed.csv"
    pd.DataFrame([{"molecule": "A", "E_ads": -1.0}]).to_csv(path, index=False)

    merged = _merge_preserving_existing_molecules(
        path,
        pd.DataFrame([{"molecule": "B", "E_ads": -2.0, "new_column": 1.0}]),
    )
    assert set(merged.columns) == {"molecule", "E_ads", "new_column"}
    assert set(merged["molecule"]) == {"A", "B"}
    # The pre-existing row must not have acquired a bogus value.
    assert pd.isna(merged.loc[merged["molecule"] == "A", "new_column"]).all()
    json.dumps(merged.to_dict(orient="records"), default=str)

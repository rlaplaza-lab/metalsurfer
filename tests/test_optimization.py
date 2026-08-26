"""Tests for optimization package: pure CPU helpers, TorchSimCalculator, setup_single_model."""

import gc
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import (
    TorchSimCalculator,
    _cache,
    _deps,
    _model,
    _optimize,
    _validation,
    optimize_isolated_molecules_batched,
    setup_single_model,
)

from .conftest import MLIP_CPU_MARKS, make_slab

# ---------------------------------------------------------------------------
# CPU unit tests (no MLIP/GPU required, no skip markers)
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal stand-in for a torch tensor that supports the ``.detach().cpu().numpy()`` chain."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._array

    def squeeze(self):
        return _FakeTensor(self._array.squeeze())

    def item(self):
        return float(self._array.squeeze())


def _make_atoms_with_cell() -> Atoms:
    """Atoms with PBC for TorchSim (needs cell)."""
    slab = make_slab(nx=2, ny=2, n_layers=2)
    slab.set_pbc([True, True, True])
    return slab


def _fake_ts_result(energy_val, n_atoms):
    return [
        {
            "potential_energy": _FakeTensor([[energy_val]]),
            "forces": _FakeTensor(np.zeros((n_atoms, 3))),
            "stress": _FakeTensor(np.eye(3) * 0.1),
        }
    ]


@pytest.fixture
def stub_autobatcher(monkeypatch: pytest.MonkeyPatch):
    _cache._AUTOBATCHER_CACHE.clear()
    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(
        _deps,
        "InFlightAutoBatcher",
        lambda *args, **kwargs: object(),
    )
    yield
    _cache._AUTOBATCHER_CACHE.clear()


# -- _validate_model_pbc -----------------------------------------------------


def test_validate_model_pbc_rejects_mixed():
    atoms = _make_atoms_with_cell()
    atoms.set_pbc([True, False, True])
    with pytest.raises(ValueError, match="mixed PBC"):
        _validation._validate_model_pbc(atoms, context="test")


def test_validate_model_pbc_rejects_bad_shape():
    atoms = _make_atoms_with_cell()
    with (
        patch.object(Atoms, "get_pbc", return_value=np.array([True, True])),
        pytest.raises(ValueError, match="invalid PBC shape"),
    ):
        _validation._validate_model_pbc(atoms, context="test")


@pytest.mark.parametrize("pbc", [[True, True, True], [False, False, False]])
def test_validate_model_pbc_passes(pbc):
    atoms = _make_atoms_with_cell()
    atoms.set_pbc(pbc)
    # should not raise
    _validation._validate_model_pbc(atoms, context="test")


# -- _resolve_ts_optimizer --------------------------------------------------


def test_resolve_ts_optimizer(monkeypatch: pytest.MonkeyPatch):
    from metalsurfer.exceptions import DependencyMissingError

    fire = object()
    ts_stub = MagicMock()
    ts_stub.Optimizer.fire = fire
    ts_stub.Optimizer.lbfgs = object()
    ts_stub.Optimizer.bfgs = object()
    monkeypatch.setattr(_deps, "ts", ts_stub)
    assert _validation._resolve_ts_optimizer("fire") is fire
    with pytest.raises(KeyError):
        _validation._resolve_ts_optimizer("does-not-exist")

    monkeypatch.setattr(_deps, "ts", None)
    with pytest.raises(DependencyMissingError):
        _validation._resolve_ts_optimizer("fire")


# -- _resolve_device --------------------------------------------------------


def test_resolve_device_passthrough():
    assert _validation._resolve_device(None) is None
    assert _validation._resolve_device("cpu") == "cpu"
    assert _validation._device_is_cuda("cuda") is True
    assert _validation._device_is_cuda("cuda:0") is True
    assert _validation._device_is_cuda("cpu") is False
    assert _validation._device_is_cuda(None) is False
    torch_dev = MagicMock()
    torch_dev.type = "cuda"
    assert _validation._device_is_cuda(torch_dev) is True
    torch_dev.type = "cpu"
    assert _validation._device_is_cuda(torch_dev) is False
    assert _validation._device_key(None) == "unknown"
    assert _validation._device_key("cuda") == "cuda"


def test_resolve_device_cuda_falls_back_when_no_torch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_deps, "torch", None)
    assert _validation._resolve_device("cuda") == "cpu"


def test_resolve_device_cuda_falls_back_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    monkeypatch.setattr(_deps, "torch", torch_stub)
    assert _validation._resolve_device("cuda") == "cpu"


def test_resolve_device_cuda_kept_when_available(monkeypatch: pytest.MonkeyPatch):
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = True
    monkeypatch.setattr(_deps, "torch", torch_stub)
    assert _validation._resolve_device("cuda") == "cuda"

    class _NoDevice:
        pass

    monkeypatch.setattr(_validation, "_resolve_device", lambda _d: None)
    assert (
        _validation._resolve_model_device(_NoDevice(), AdsorptionConfig(device="cuda"))
        == "cpu"
    )

    class _HasDevice:
        device = "cuda:1"

    assert (
        _validation._resolve_model_device(_HasDevice(), AdsorptionConfig(device="cpu"))
        == "cuda:1"
    )


# -- _positions_cell_hash ---------------------------------------------------


def test_positions_cell_hash_deterministic_and_changes():
    a = _make_atoms_with_cell()
    assert _validation._positions_cell_hash(a) == _validation._positions_cell_hash(a)
    b = a.copy()
    b.positions[0] += 0.01
    assert _validation._positions_cell_hash(a) != _validation._positions_cell_hash(b)


# -- _is_cuda_oom_error -----------------------------------------------------


@pytest.mark.parametrize(
    "msg",
    ["CUDA out of memory", "CUDA OOM", "cuda out of memory during forward"],
)
def test_is_cuda_oom_error_true(msg):
    assert _validation._is_cuda_oom_error(RuntimeError(msg)) is True


def test_is_cuda_oom_error_true_for_cuda_oom_type(monkeypatch):
    from types import SimpleNamespace

    from metalsurfer.optimization import _deps

    class FakeCudaOOM(RuntimeError):
        pass

    monkeypatch.setattr(
        _deps,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(OutOfMemoryError=FakeCudaOOM)),
    )
    assert _validation._is_cuda_oom_error(FakeCudaOOM("boom")) is True


@pytest.mark.parametrize("msg", ["boom", "shape mismatch", ""])
def test_is_cuda_oom_error_false(msg):
    assert _validation._is_cuda_oom_error(RuntimeError(msg)) is False


# -- _resolve_autobatcher_max_atoms_to_try ----------------------------------


def test_resolve_autobatcher_max_atoms_to_try_uses_config_override():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=12_345)
    cap, source = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=40,
        config=config,
    )
    assert cap == 12_345
    assert source == "config_override"


def test_resolve_autobatcher_max_atoms_to_try_uses_dynamic_policy():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    cap, source = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400,
        n_systems=10,
        config=config,
    )
    assert cap == 10_000
    assert source == "dynamic"


def test_resolve_autobatcher_max_atoms_to_try_floor_and_ceiling():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    floor_cap, _ = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=20, n_systems=2, config=config
    )
    ceil_cap, _ = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=2_000, n_systems=100, config=config
    )
    assert floor_cap == 5_000
    assert ceil_cap == 200_000
    assert all(
        c >= _validation._DYNAMIC_AUTOBATCHER_CAP_MIN
        and c <= _validation._DYNAMIC_AUTOBATCHER_CAP_MAX
        for c in (floor_cap, ceil_cap)
    )


def test_resolve_autobatcher_max_atoms_to_try_buckets():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    a, _ = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400, n_systems=10, config=config
    )
    b, _ = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=400, n_systems=11, config=config
    )
    assert a == b == 10_000


# -- _parallel_capacity_cache_key -------------------------------------------


def test_parallel_capacity_cache_key_deterministic():
    model = object()
    config = AdsorptionConfig()
    k1 = _validation._parallel_capacity_cache_key(model, 100, config)
    k2 = _validation._parallel_capacity_cache_key(model, 100, config)
    assert k1 == k2
    assert k1[0] == id(model)
    assert k1[1] == "unknown"  # no .device on model -> unified None key


# -- clear_autobatcher_cache ------------------------------------------------


def test_clear_autobatcher_cache_full_preserves_capacity(
    monkeypatch: pytest.MonkeyPatch,
):
    """Default clear evicts autobatchers (GPU memory) but keeps probed capacities."""
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    fake = object()
    _cache._AUTOBATCHER_CACHE[("k",)] = fake
    _cache._PARALLEL_CAPACITY_CACHE[("k",)] = 1
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache()
    assert _cache._AUTOBATCHER_CACHE == {}
    assert _cache._PARALLEL_CAPACITY_CACHE == {("k",): 1}
    _cache._PARALLEL_CAPACITY_CACHE.clear()


def test_clear_autobatcher_cache_clear_capacity_wipes_both(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    _cache._AUTOBATCHER_CACHE[("k",)] = object()
    _cache._PARALLEL_CAPACITY_CACHE[("k",)] = 1
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache(clear_capacity=True)
    assert _cache._AUTOBATCHER_CACHE == {}
    assert _cache._PARALLEL_CAPACITY_CACHE == {}


def test_clear_autobatcher_cache_threshold_eviction(monkeypatch: pytest.MonkeyPatch):
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    model = object()
    small = (id(model), "cpu", "n_atoms", 0.5, 1000, None, 100_000)
    large = (id(model), "cpu", "n_atoms", 0.5, 50_000, None, 100_000)
    _cache._AUTOBATCHER_CACHE[small] = object()
    _cache._AUTOBATCHER_CACHE[large] = object()
    _cache._PARALLEL_CAPACITY_CACHE[small] = 7
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache(max_n_atoms_threshold=10_000)
    assert small not in _cache._AUTOBATCHER_CACHE
    assert large in _cache._AUTOBATCHER_CACHE
    # threshold eviction never touches the capacity cache
    assert dict(_cache._PARALLEL_CAPACITY_CACHE) == {small: 7}
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._PARALLEL_CAPACITY_CACHE.clear()


def test_capacity_cache_helpers_round_trip():
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    key = ("model", "cpu", 100)
    assert _cache.capacity_cache_get(key) is None
    _cache.capacity_cache_set(key, 12)
    assert _cache.capacity_cache_get(key) == 12
    _cache._PARALLEL_CAPACITY_CACHE.clear()


def test_pop_autobatcher_returns_and_removes_entry():
    _cache._AUTOBATCHER_CACHE.clear()
    key = ("k",)
    sentinel = object()
    _cache._AUTOBATCHER_CACHE[key] = sentinel
    assert _cache.pop_autobatcher(key) is sentinel
    assert key not in _cache._AUTOBATCHER_CACHE
    assert _cache.pop_autobatcher(key) is None


def test_cache_helpers_are_thread_safe(
    monkeypatch: pytest.MonkeyPatch, stub_autobatcher
):
    """Concurrent get/clear must not raise and must leave a consistent cache."""
    import threading

    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = False
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache._PARALLEL_CAPACITY_CACHE.clear()

    models = [type("MockModel", (), {"device": "cpu"})() for _ in range(4)]
    errors: list[BaseException] = []
    barrier = threading.Barrier(4)

    def worker(idx: int) -> None:
        try:
            barrier.wait(timeout=5)
            for i in range(12):
                _cache._get_inflight_autobatcher(models[idx], 100)
                _cache.capacity_cache_set((idx, i), i)
                _cache.capacity_cache_get((idx, i))
                if i % 5 == 4:
                    barrier.wait(timeout=5)
                    if idx == 0:
                        _cache.clear_autobatcher_cache()
                    barrier.wait(timeout=5)
        except BaseException as exc:  # pragma: no cover - only on a real failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not any(t.is_alive() for t in threads)
    assert errors == []
    # capacity survived every default clear
    assert len(_cache._PARALLEL_CAPACITY_CACHE) == 4 * 12
    assert len(_cache._AUTOBATCHER_CACHE) == 4
    assert len(set(id(v) for v in _cache._AUTOBATCHER_CACHE.values())) == 4
    _cache._PARALLEL_CAPACITY_CACHE.clear()


def test_clear_autobatcher_cache_runs_cuda_path(monkeypatch: pytest.MonkeyPatch):
    _cache._AUTOBATCHER_CACHE.clear()
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.ipc_collect = MagicMock()
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache(drain_cuda=True)
    torch_stub.cuda.synchronize.assert_called()
    torch_stub.cuda.empty_cache.assert_called()


def test_clear_autobatcher_cache_default_skips_cuda_drain(
    monkeypatch: pytest.MonkeyPatch,
):
    """Per-batch eviction must not synchronize/empty_cache by default."""
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._AUTOBATCHER_CACHE[("k",)] = object()
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = True
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache()
    torch_stub.cuda.synchronize.assert_not_called()
    torch_stub.cuda.empty_cache.assert_not_called()
    assert _cache._AUTOBATCHER_CACHE == {}


def test_clear_autobatcher_cache_clear_capacity_drains_cuda(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._AUTOBATCHER_CACHE.clear()
    _cache._AUTOBATCHER_CACHE[("k",)] = object()
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = True
    torch_stub.cuda.ipc_collect = MagicMock()
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _cache.clear_autobatcher_cache(clear_capacity=True)
    torch_stub.cuda.synchronize.assert_called()
    torch_stub.cuda.empty_cache.assert_called()


def test_clip_frozen_indices_to_slab_drops_out_of_range():
    kept = _optimize._clip_frozen_indices_to_slab([0, 1, 5, 10], slab_size=5)
    assert kept == [0, 1]
    assert _optimize._clip_frozen_indices_to_slab([0, 1, 2], slab_size=5) == [0, 1, 2]
    assert _optimize._clip_frozen_indices_to_slab([], slab_size=5) == []


def test_get_inflight_autobatcher_returns_none_without_ts():
    batcher, cache_key = _cache._get_inflight_autobatcher(ts_model=None, max_n_atoms=0)
    assert batcher is None
    assert cache_key is None


# -- _maybe_clear_cuda_cache ------------------------------------------------


def test_maybe_clear_cuda_cache_clears_on_cuda(monkeypatch: pytest.MonkeyPatch):
    torch_stub = MagicMock()
    torch_stub.cuda.is_available.return_value = True
    monkeypatch.setattr(_deps, "torch", torch_stub)
    model = MagicMock()
    model.device = "cuda:0"
    _cache._maybe_clear_cuda_cache(model)
    torch_stub.cuda.empty_cache.assert_called_once()


def test_maybe_clear_cuda_cache_noop_without_torch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_deps, "torch", None)
    _cache._maybe_clear_cuda_cache(object())


# -- TorchSimCalculator -----------------------------------------------------


def test_torchsim_calculator_extracts_energy_forces_stress(
    monkeypatch: pytest.MonkeyPatch,
):
    atoms = _make_atoms_with_cell()[:6]
    atoms.set_cell([10, 10, 15])
    atoms.set_pbc([True, True, True])

    class _FakeTs:
        def static(self, system, model):
            return _fake_ts_result(-42.5, 6)

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    calc = TorchSimCalculator(object())
    calc.calculate(atoms, ["energy", "forces", "stress"])
    assert abs(calc.results["energy"] - (-42.5)) < 1e-6
    assert calc.results["forces"].shape == (6, 3)
    np.testing.assert_allclose(calc.results["stress"], [0.1, 0.1, 0.1, 0, 0, 0])
    assert calc._atoms_changed(atoms) is False
    # geometry edit invalidates cache
    edited = atoms.copy()
    edited.positions[0] += 0.5
    assert calc._atoms_changed(edited) is True


def test_torchsim_calculator_non_finite_energy_raises(monkeypatch: pytest.MonkeyPatch):
    class _FakeTs:
        def static(self, system, model):
            return [
                {
                    "potential_energy": _FakeTensor([[float("nan")]]),
                    "forces": _FakeTensor(np.zeros((3, 3))),
                }
            ]

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    calc = TorchSimCalculator(object())
    with pytest.raises(RuntimeError, match="non-finite energy"):
        calc.calculate(_make_atoms_with_cell()[:3], ["energy", "forces"])


# -- _forces_for_optimized_systems (D5) --------------------------------------


def test_forces_for_optimized_systems_recovers_via_static(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    n_systems = 3
    result = [_make_atoms_with_cell()[:5] for _ in range(n_systems)]
    energies = [_FakeTensor([[-100.0 - i]]) for i in range(n_systems)]
    # Optimised batch hid its forces -> _split_forces_by_system returns None.
    batch = SimpleNamespace(forces=None)

    per_system_forces = [
        np.full((5, 3), 0.1),
        np.full((5, 3), 0.2),
        np.full((5, 3), 0.3),
    ]
    captured = {}

    class _FakeStatic:
        def static(self, system=None, model=None):
            captured["system"] = system
            return [{"forces": _FakeTensor(f)} for f in per_system_forces]

    monkeypatch.setattr(_deps, "ts", _FakeStatic())

    out = _optimize._forces_for_optimized_systems(batch, energies, result, object())
    assert out is not None
    assert len(out) == n_systems
    for i in range(n_systems):
        assert out[i] is not None
        assert np.allclose(out[i], per_system_forces[i])
    # One fused static call over all finite-energy survivors.
    assert len(captured["system"]) == n_systems


def test_forces_for_optimized_systems_none_when_static_omits(
    monkeypatch: pytest.MonkeyPatch,
):
    from types import SimpleNamespace

    n_systems = 2
    result = [_make_atoms_with_cell()[:4] for _ in range(n_systems)]
    energies = [_FakeTensor([[-100.0]]) for _ in range(n_systems)]
    batch = SimpleNamespace(forces=None)

    class _FakeStatic:
        def static(self, system=None, model=None):
            # Both systems omit forces: must yield None, never zero-force fallback.
            return [{"forces": None}, {"forces": None}]

    monkeypatch.setattr(_deps, "ts", _FakeStatic())

    out = _optimize._forces_for_optimized_systems(batch, energies, result, object())
    assert out is not None
    assert len(out) == n_systems
    assert all(o is None for o in out)


# -- batch_static -----------------------------------------------------------


def test_batch_static_returns_energies_forces(monkeypatch: pytest.MonkeyPatch):
    class _FakeTs:
        def static(self, system, model):
            return [_fake_ts_result(-1.0, len(a))[0] for a in system]

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    atoms_list = [_make_atoms_with_cell()[:3], _make_atoms_with_cell()[:4]]
    out = _optimize.batch_static(atoms_list, ts_model=object())
    assert len(out) == 2
    assert all(np.isfinite(e) for e, _ in out)
    assert out[0][1].shape == (3, 3)
    assert out[1][1].shape == (4, 3)


def test_batch_static_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(_deps, "ts", MagicMock())
    assert _optimize.batch_static([], ts_model=object()) == []


def test_batch_static_raises_without_torchsim(monkeypatch: pytest.MonkeyPatch):
    from metalsurfer.exceptions import DependencyMissingError

    monkeypatch.setattr(_deps, "ts", None)
    with pytest.raises(DependencyMissingError, match="torch-sim-atomistic"):
        _optimize.batch_static([_make_atoms_with_cell()[:3]], ts_model=object())


# -- _voigt_6 ---------------------------------------------------------------


def test_voigt_6_mapping():
    stress = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    out = _model._voigt_6(stress)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0, 0.3, 0.2, 0.1])


# -- _model fairchem helpers + dependency-missing guards --------------------


def test_ensure_torch_checkpoint_safe_globals_noop_without_torch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(_deps, "torch", None)
    # should not raise
    _model._ensure_torch_checkpoint_safe_globals()


def test_ensure_torch_checkpoint_safe_globals_calls_add_safe_globals(
    monkeypatch: pytest.MonkeyPatch,
):
    add_sg = MagicMock()
    torch_stub = MagicMock()
    torch_stub.serialization.add_safe_globals = add_sg
    monkeypatch.setattr(_deps, "torch", torch_stub)
    _model._ensure_torch_checkpoint_safe_globals()
    add_sg.assert_called_once()


def test_fairchem_failure_messages():
    assert "Check HF token" in _model._fairchem_load_failure_message("boom", "uma")
    assert "weights_only" in _model._fairchem_pytorch26_unpickling_message()


def test_raise_fairchem_load_error_weights_only():
    with pytest.raises(RuntimeError, match="weights_only"):
        _model._raise_fairchem_load_error(
            RuntimeError("UnpicklingError weights only slice"), "uma"
        )


def test_raise_fairchem_load_error_generic():
    with pytest.raises(RuntimeError, match="boom"):
        _model._raise_fairchem_load_error(RuntimeError("boom"), "uma")


def test_setup_torchsim_model_raises_without_torchsim(monkeypatch: pytest.MonkeyPatch):
    from metalsurfer.exceptions import DependencyMissingError

    real_import = (
        __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
    )

    def fake_import(name, *args, **kwargs):
        if name == "torch_sim.models.fairchem":
            raise ImportError("no torch_sim")
        return real_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(DependencyMissingError, match="torch-sim-atomistic"),
    ):
        _model.setup_torchsim_model("uma-s-1p2", "cpu")


def test_setup_torchsim_model_forwards_task_name(monkeypatch: pytest.MonkeyPatch):
    """setup_torchsim_model passes task_name through to FairChemModel."""
    import sys
    import types

    captured: dict[str, object] = {}

    def _fake_fairchem_model(**kwargs):
        captured.update(kwargs)
        return object()

    fairchem_mod = types.ModuleType("torch_sim.models.fairchem")
    fairchem_mod.FairChemModel = _fake_fairchem_model

    device_sentinel = object()
    torch_stub = MagicMock()
    torch_stub.device.return_value = device_sentinel

    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(_deps, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch_sim.models.fairchem", fairchem_mod)

    model = _model.setup_torchsim_model("uma-s-1p1", "cpu", task_name="oc20")

    assert model is not None
    assert captured["model"] == "uma-s-1p1"
    assert captured["device"] is device_sentinel
    assert captured["task_name"] == "oc20"


def test_setup_torchsim_model_default_task_name_is_oc25(
    monkeypatch: pytest.MonkeyPatch,
):
    """setup_torchsim_model defaults to the oc25 task head."""
    import sys
    import types

    captured: dict[str, object] = {}

    def _fake_fairchem_model(**kwargs):
        captured.update(kwargs)
        return object()

    fairchem_mod = types.ModuleType("torch_sim.models.fairchem")
    fairchem_mod.FairChemModel = _fake_fairchem_model

    torch_stub = MagicMock()
    torch_stub.device.return_value = object()

    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(_deps, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch_sim.models.fairchem", fairchem_mod)

    _model.setup_torchsim_model()

    assert captured["task_name"] == "oc25"


def test_optimize_isolated_raises_without_torchsim(monkeypatch: pytest.MonkeyPatch):
    from metalsurfer.exceptions import DependencyMissingError

    monkeypatch.setattr(_deps, "ts", None)
    with pytest.raises(DependencyMissingError, match="torch-sim-atomistic"):
        optimize_isolated_molecules_batched(
            [_make_atoms_with_cell()[:3]],
            ts_model=MagicMock(),
            config=AdsorptionConfig(),
        )


def test_optimize_slab_raises_without_torchsim(monkeypatch: pytest.MonkeyPatch):
    from metalsurfer.exceptions import DependencyMissingError

    monkeypatch.setattr(_deps, "ts", None)
    monkeypatch.setattr(_deps, "InFlightAutoBatcher", None)
    monkeypatch.setattr(_deps, "ts_constraints", None)
    slab = _make_atoms_with_cell()
    with pytest.raises(DependencyMissingError, match="torch-sim-atomistic"):
        _optimize.optimize_adsorbate_slab_batched([slab], slab, ts_model=MagicMock())


def test_optimize_slab_raises_on_batch_size_mismatch(monkeypatch: pytest.MonkeyPatch):
    """A short batch must raise, not silently return all-None placements.

    torch-sim 0.5.2 guarantees count-preserving, original-order output, so this
    is only reachable with a broken/faked autobatcher -- but silently nulling
    every placement would hide real data loss.
    """

    class _FakeBatch:
        """Returns fewer systems than were submitted."""

        energy = None
        forces = None

        def to_atoms(self):
            return [_make_atoms_with_cell()]

    class _FakeTs:
        Optimizer = MagicMock()

        @staticmethod
        def generate_force_convergence_fn(force_tol, include_cell_forces):
            return object()

        @staticmethod
        def optimize(**kwargs):
            return _FakeBatch()

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(_deps, "InFlightAutoBatcher", lambda *a, **k: object())
    monkeypatch.setattr(_deps, "torch", None)
    monkeypatch.setattr(
        _optimize,
        "_make_state_with_frozen_constraint",
        lambda *args, **kwargs: object(),
    )
    _cache._AUTOBATCHER_CACHE.clear()

    slab = _make_atoms_with_cell()
    combined = [slab.copy(), slab.copy(), slab.copy()]
    with pytest.raises(RuntimeError, match=r"expected 3, got 1"):
        _optimize.optimize_adsorbate_slab_batched(
            combined,
            slab,
            ts_model=type("MockModel", (), {"device": "cpu"})(),
            config=AdsorptionConfig(device="cpu"),
        )
    _cache._AUTOBATCHER_CACHE.clear()


def test_optimize_slab_rebuilds_states_after_cuda_oom(monkeypatch: pytest.MonkeyPatch):
    """On CUDA OOM the retry must rebuild systems and release the first attempt.

    The retried ``ts.optimize`` receives freshly built states (disjoint tags),
    and the first-attempt states must be released (weakrefs dead) so the failed
    attempt's tensors cannot pin GPU memory.
    """
    import weakref

    calls: list[list[int]] = []
    refs: list[weakref.ref] = []
    counter = {"n": 0}

    class _State:
        def __init__(self, tag):
            self.tag = tag

    def _spy_make_state(*args, **kwargs):
        counter["n"] += 1
        state = _State(counter["n"])
        refs.append(weakref.ref(state))
        return state

    class _FakeBatch:
        def __init__(self, n):
            self.energy = [_FakeTensor([-1.0])] * n
            self.forces = None

        def to_atoms(self):
            return [make_slab(nx=2, ny=2, n_layers=2) for _ in range(len(self.energy))]

    class _FakeTs:
        Optimizer = MagicMock()

        @staticmethod
        def generate_force_convergence_fn(force_tol, include_cell_forces):
            return object()

        def optimize(self, **kwargs):
            tags = [s.tag for s in kwargs["system"]]
            calls.append(tags)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return _FakeBatch(len(tags))

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(_deps, "InFlightAutoBatcher", lambda *a, **k: object())
    monkeypatch.setattr(_deps, "torch", None)
    monkeypatch.setattr(
        _optimize, "_make_state_with_frozen_constraint", _spy_make_state
    )
    _cache._AUTOBATCHER_CACHE.clear()

    slab = _make_atoms_with_cell()
    combined = [slab.copy(), slab.copy(), slab.copy()]
    result = _optimize.optimize_adsorbate_slab_batched(
        combined,
        slab,
        ts_model=type("MockModel", (), {"device": "cpu"})(),
        config=AdsorptionConfig(device="cpu"),
    )
    _cache._AUTOBATCHER_CACHE.clear()

    assert counter["n"] == 6
    assert calls[0] == [1, 2, 3]
    assert calls[1] == [4, 5, 6]
    assert len(result) == 3
    assert all(r is not None for r in result)

    gc.collect()
    assert all(r() is None for r in refs[:3])


def test_optimize_slab_retries_after_batcher_capacity_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """TorchSim's batcher-capacity refusal must trigger the rebuild-and-retry path.

    ``InFlightAutoBatcher.load_states`` raises a plain ``ValueError`` ("...is
    greater than max_metric...") when incoming systems outgrow the probed
    bucket (e.g. a reused batcher after VRAM shrank). Before the fix this
    propagated and killed saturation runs mid-coverage.
    """
    calls: list[list[int]] = []
    counter = {"n": 0}

    class _State:
        def __init__(self, tag):
            self.tag = tag

    def _spy_make_state(*args, **kwargs):
        counter["n"] += 1
        return _State(counter["n"])

    class _FakeBatch:
        def __init__(self, n):
            self.energy = [_FakeTensor([-1.0])] * n
            self.forces = None

        def to_atoms(self):
            return [make_slab(nx=2, ny=2, n_layers=2) for _ in range(len(self.energy))]

    class _FakeTs:
        Optimizer = MagicMock()

        @staticmethod
        def generate_force_convergence_fn(force_tol, include_cell_forces):
            return object()

        def optimize(self, **kwargs):
            tags = [s.tag for s in kwargs["system"]]
            calls.append(tags)
            if len(calls) == 1:
                raise ValueError(
                    "Max metric of system with index 0 in states: 870 is greater "
                    "than max_metric 696.0, please set a larger max_metric or run "
                    "smaller systems metric."
                )
            return _FakeBatch(len(tags))

    monkeypatch.setattr(_deps, "ts", _FakeTs())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(_deps, "InFlightAutoBatcher", lambda *a, **k: object())
    monkeypatch.setattr(_deps, "torch", None)
    monkeypatch.setattr(
        _optimize, "_make_state_with_frozen_constraint", _spy_make_state
    )
    _cache._AUTOBATCHER_CACHE.clear()

    slab = _make_atoms_with_cell()
    combined = [slab.copy(), slab.copy()]
    result = _optimize.optimize_adsorbate_slab_batched(
        combined,
        slab,
        ts_model=type("MockModel", (), {"device": "cpu"})(),
        config=AdsorptionConfig(device="cpu"),
    )
    _cache._AUTOBATCHER_CACHE.clear()

    assert counter["n"] == 4
    assert calls[0] == [1, 2]
    assert calls[1] == [3, 4]
    assert len(result) == 2
    assert all(r is not None for r in result)


# ---------------------------------------------------------------------------
# MLIP-gated tests (only run when the full stack is installed)
# ---------------------------------------------------------------------------


class TestTorchSimCalculator:
    pytestmark = MLIP_CPU_MARKS
    """Unit tests with mocked model."""

    def test_calculator_interface_with_mock_model(self):
        """TorchSimCalculator returns energy/forces via ts.static() with mock."""
        import torch

        mock_model = type("MockModel", (), {})()
        mock_model.device = torch.device("cpu")
        mock_model.dtype = torch.float32

        energy_val = -42.5
        n_atoms = 6

        calc = TorchSimCalculator(mock_model)
        atoms = _make_atoms_with_cell()
        atoms = atoms[:n_atoms]
        atoms.set_cell([10.0, 10.0, 15.0])
        atoms.set_pbc([True, True, True])

        fake_result = [
            {
                "potential_energy": torch.tensor([[energy_val]], dtype=torch.float32),
                "forces": torch.randn(n_atoms, 3, dtype=torch.float32) * 0.1,
            }
        ]
        with patch("torch_sim.static", return_value=fake_result):
            calc.calculate(atoms, ["energy", "forces"])
            assert "energy" in calc.results
            assert abs(calc.results["energy"] - energy_val) < 1e-6
            assert "forces" in calc.results
            assert calc.results["forces"].shape == (n_atoms, 3)

            e = calc.get_potential_energy(atoms)
            assert abs(e - energy_val) < 1e-6
            f = calc.get_forces(atoms)
            assert f.shape == (n_atoms, 3)

    def test_get_stress_recomputes_when_missing_from_prior_calculate(self):
        """Same geometry: energy-only calculate then get_stress must not return zeros."""
        import torch

        mock_model = type("MockModel", (), {})()
        mock_model.device = torch.device("cpu")
        mock_model.dtype = torch.float32

        n_atoms = 6
        calc = TorchSimCalculator(mock_model)
        atoms = _make_atoms_with_cell()[:n_atoms]
        atoms.set_cell([10.0, 10.0, 15.0])
        atoms.set_pbc([True, True, True])

        energy_only = [
            {
                "potential_energy": torch.tensor([[-42.5]], dtype=torch.float32),
                "forces": torch.randn(n_atoms, 3, dtype=torch.float32) * 0.1,
            }
        ]
        with_stress = [
            {
                "potential_energy": torch.tensor([[-42.5]], dtype=torch.float32),
                "forces": torch.randn(n_atoms, 3, dtype=torch.float32) * 0.1,
                "stress": torch.eye(3, dtype=torch.float32) * 0.05,
            }
        ]
        with patch(
            "torch_sim.static", side_effect=[energy_only, with_stress]
        ) as mock_static:
            calc.calculate(atoms, ["energy", "forces"])
            assert "stress" not in calc.results
            stress = calc.get_stress(atoms)
            assert mock_static.call_count == 2
            assert stress.shape == (6,)
            assert not np.allclose(stress, 0.0)

    def test_get_forces_raises_when_missing(self):
        """Missing forces after calculate must raise, not return zeros."""
        import torch

        mock_model = type("MockModel", (), {})()
        mock_model.device = torch.device("cpu")
        mock_model.dtype = torch.float32

        n_atoms = 4
        calc = TorchSimCalculator(mock_model)
        atoms = _make_atoms_with_cell()[:n_atoms]
        atoms.set_cell([10.0, 10.0, 15.0])
        atoms.set_pbc([True, True, True])

        no_forces = [
            {
                "potential_energy": torch.tensor([[-10.0]], dtype=torch.float32),
            }
        ]
        with patch("torch_sim.static", return_value=no_forces):
            calc.calculate(atoms, ["energy", "forces"])
            with pytest.raises(RuntimeError, match="no forces"):
                calc.get_forces(atoms)


class TestTorchSimCalculatorDeps:
    """Calculator behavior without a real MLIP model."""

    def test_calculate_raises_when_torchsim_missing(self, monkeypatch):
        monkeypatch.setattr(_deps, "ts", None)
        calc = TorchSimCalculator(object())
        atoms = _make_atoms_with_cell()
        from metalsurfer.exceptions import DependencyMissingError

        with pytest.raises(DependencyMissingError, match="torch-sim"):
            calc.calculate(atoms, ["energy", "forces"])


class TestSetupSingleModel:
    pytestmark = MLIP_CPU_MARKS
    """Integration tests with real FairChemModel."""

    def test_setup_single_model_returns_calculator_and_model(self):
        """setup_single_model returns (calculator, ts_model) tuple."""
        calculator, ts_model = setup_single_model("uma-s-1p1", "cpu", task_name="oc20")
        assert calculator is not None
        assert ts_model is not None
        assert isinstance(calculator, TorchSimCalculator)

    def test_torchsim_calculator_single_point_with_real_model(self):
        """TorchSimCalculator from setup_single_model gives finite energy/forces."""
        calculator, _ = setup_single_model("uma-s-1p1", "cpu", task_name="oc20")
        atoms = _make_atoms_with_cell()
        atoms.calc = calculator

        energy = atoms.get_potential_energy()
        assert np.isfinite(energy)
        forces = atoms.get_forces()
        assert forces.shape == (len(atoms), 3)
        assert np.all(np.isfinite(forces))

        # The single point is not force-minimised, so we use generous (but
        # non-trivial) tolerances. Total force on a periodic system must vanish
        # (momentum conservation: internal forces sum to zero).
        assert np.linalg.norm(forces.sum(axis=0)) < 1e-2

        # Rotational covariance: forces transform as vectors under an SO(3)
        # rotation of the structure (positions and cell rotated by R). The real
        # UMA model does not expose stress for this system, so the stress-tensor
        # symmetry check from the plan is intentionally omitted here.
        from scipy.spatial.transform import Rotation as _Rot

        R = _Rot.from_euler("z", 30.0, degrees=True).as_matrix()
        rotated = atoms.copy()
        rotated.set_positions(rotated.get_positions() @ R.T)
        cell = np.asarray(rotated.get_cell(), dtype=float)
        rotated.set_cell(cell @ R.T)
        rotated.calc = calculator
        forces_rot = rotated.get_forces()
        assert np.allclose(forces_rot, forces @ R.T, atol=1e-2)

    def test_optimize_isolated_sequentially_skips_autobatcher(self):
        """With optimize_isolated_sequentially=True, _get_inflight_autobatcher is not called."""
        calculator, ts_model = setup_single_model("uma-s-1p1", "cpu", task_name="oc20")
        config = AdsorptionConfig(
            optimize_isolated_sequentially=True,
            device="cpu",
        )
        conformers = [_make_atoms_with_cell()[:6]]  # small system
        for c in conformers:
            c.set_cell([10.0, 10.0, 15.0])
            c.set_pbc([True, True, True])

        with patch(
            "metalsurfer.optimization._cache._get_inflight_autobatcher"
        ) as mock_get_ab:
            results = optimize_isolated_molecules_batched(
                conformers,
                ts_model,
                fmax=0.1,
                steps=5,
                config=config,
            )
            mock_get_ab.assert_not_called()
        assert len(results) == 1
        atoms, energy = results[0]
        assert np.isfinite(energy)


def test_get_inflight_autobatcher_saturation_reuses_small_growth(stub_autobatcher):
    """Saturation mode reuses the batcher for a small max_n_atoms increase.

    The entry is re-keyed under the requested size so that end-of-call
    eviction (``key[4] < max_n_atoms``) keeps — not evicts — the reused
    batcher for subsequent steps of the same size.
    """
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(
        device="cpu",
        saturation_autobatcher_reuse_growth_atoms=8,
        saturation_autobatcher_reuse_growth_fraction=0.0,
    )
    ab1, key1 = _cache._get_inflight_autobatcher(
        model,
        100,
        config=config,
        saturation_reuse=True,
    )
    ab2, key2 = _cache._get_inflight_autobatcher(
        model,
        105,
        config=config,
        saturation_reuse=True,
    )
    assert ab1 is not None
    assert ab2 is ab1
    # Re-keyed to the new size; old key is gone.
    assert key2 == (key1[0], key1[1], key1[2], key1[3], 105, key1[5], key1[6])
    assert _cache._AUTOBATCHER_CACHE.get(key1) is None
    assert _cache._AUTOBATCHER_CACHE.get(key2) is ab1
    # A repeat call at the reused size hits the exact key.
    ab3, key3 = _cache._get_inflight_autobatcher(
        model,
        105,
        config=config,
        saturation_reuse=True,
    )
    assert ab3 is ab1
    assert key3 == key2
    # End-of-call eviction at the current threshold must keep the entry.
    _cache.clear_autobatcher_cache(max_n_atoms_threshold=105)
    assert _cache._AUTOBATCHER_CACHE.get(key2) is ab1


def test_get_inflight_autobatcher_non_saturation_uses_exact_size_key(stub_autobatcher):
    """Non-saturation mode should not reuse different max_n_atoms keys."""
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(device="cpu")
    ab1, key1 = _cache._get_inflight_autobatcher(
        model,
        100,
        config=config,
        saturation_reuse=False,
    )
    ab2, key2 = _cache._get_inflight_autobatcher(
        model,
        105,
        config=config,
        saturation_reuse=False,
    )
    assert ab1 is not None
    assert ab2 is not None
    assert ab2 is not ab1
    assert key2 != key1


def test_get_inflight_autobatcher_uses_explicit_probe_cap(stub_autobatcher):
    model = type("MockModel", (), {"device": "cpu"})()
    config = AdsorptionConfig(device="cpu", autobatcher_max_atoms_to_try=123_456)
    _, key = _cache._get_inflight_autobatcher(
        model,
        100,
        config=config,
        max_atoms_to_try=7_000,
    )
    assert key is not None
    assert int(key[6]) == 7_000


def test_estimate_parallel_relaxation_capacity_fallback_without_torchsim(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(_deps, "ts", None)
    monkeypatch.setattr(_deps, "determine_max_batch_size", None)
    config = AdsorptionConfig()
    atoms = _make_atoms_with_cell()
    capacity = _optimize.estimate_parallel_relaxation_capacity(
        ts_model=object(),
        representative_atoms=atoms,
        config=config,
        frozen_indices=[],
    )
    assert capacity == 1


def test_estimate_parallel_relaxation_capacity_runtime_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(
        _optimize,
        "_make_state_with_frozen_constraint",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    monkeypatch.setattr(
        _deps,
        "calculate_memory_scalers",
        lambda *args, **kwargs: [100.0],
    )
    config = AdsorptionConfig(autobatcher_max_memory_scaler=1200.0)
    atoms = _make_atoms_with_cell()
    capacity = _optimize.estimate_parallel_relaxation_capacity(
        ts_model=object(),
        representative_atoms=atoms,
        config=config,
        frozen_indices=[],
    )
    assert capacity == 1
    cache_key = _validation._parallel_capacity_cache_key(object(), len(atoms), config)
    assert _cache.capacity_cache_get(cache_key) is None


def test_estimate_parallel_relaxation_capacity_value_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(
        _optimize,
        "_make_state_with_frozen_constraint",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(
        _deps,
        "calculate_memory_scalers",
        lambda *args, **kwargs: [100.0],
    )
    config = AdsorptionConfig(autobatcher_max_memory_scaler=1200.0)
    atoms = _make_atoms_with_cell()
    with pytest.raises(ValueError, match="bad config"):
        _optimize.estimate_parallel_relaxation_capacity(
            ts_model=object(),
            representative_atoms=atoms,
            config=config,
            frozen_indices=[],
        )


def test_estimate_parallel_relaxation_capacity_uses_memory_scaler(
    monkeypatch: pytest.MonkeyPatch,
):
    _cache._PARALLEL_CAPACITY_CACHE.clear()
    monkeypatch.setattr(_deps, "ts", object())
    monkeypatch.setattr(_deps, "ts_constraints", object())
    monkeypatch.setattr(
        _optimize,
        "_make_state_with_frozen_constraint",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        _deps,
        "calculate_memory_scalers",
        lambda state, memory_scales_with: [100.0],
    )
    config = AdsorptionConfig(
        autobatcher_max_memory_scaler=1200.0,
        autobatcher_max_memory_padding=0.5,
    )
    atoms = _make_atoms_with_cell()
    capacity = _optimize.estimate_parallel_relaxation_capacity(
        ts_model=object(),
        representative_atoms=atoms,
        config=config,
        frozen_indices=[],
    )
    assert capacity == 6


def test_resolve_autobatcher_max_atoms_to_try_is_conservative_vs_estimate():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    max_n_atoms = 333
    n_systems = 9
    cap, source = _validation._resolve_autobatcher_max_atoms_to_try(
        max_n_atoms=max_n_atoms,
        n_systems=n_systems,
        config=config,
    )
    estimated = (
        _validation._DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER * max_n_atoms * n_systems
    )
    assert cap >= estimated
    assert source == "dynamic"


class _FakeBatch:
    def __init__(self, forces, system_idx=None):
        self.forces = _FakeTensor(forces)
        if system_idx is None:
            self.system_idx = None
        else:
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


def test_split_forces_by_system_raises_without_system_idx_for_multiple_systems():
    from metalsurfer.optimization._optimize import _split_forces_by_system

    atom_counts = [3, 5]
    forces = np.arange(8 * 3, dtype=float).reshape(8, 3)
    batch = _FakeBatch(forces, system_idx=None)
    with pytest.raises(RuntimeError, match="system_idx"):
        _split_forces_by_system(batch, 2, atom_counts)


def test_split_forces_by_system_single_system_without_system_idx():
    from metalsurfer.optimization._optimize import _split_forces_by_system

    atom_counts = [4]
    forces = np.arange(4 * 3, dtype=float).reshape(4, 3)
    batch = _FakeBatch(forces, system_idx=None)
    per_system = _split_forces_by_system(batch, 1, atom_counts)
    assert per_system is not None
    assert [f.shape for f in per_system] == [(4, 3)]
    np.testing.assert_allclose(per_system[0], forces)


def test_split_forces_by_system_raises_on_atom_count_mismatch():
    from metalsurfer.optimization._optimize import _split_forces_by_system

    batch = _FakeBatch(np.zeros((8, 3)), np.array([0, 0, 0, 1, 1, 1, 1, 1]))
    with pytest.raises(RuntimeError, match="could not be split per system"):
        _split_forces_by_system(batch, 2, [4, 4])


def test_get_inflight_autobatcher_returns_pair_when_unavailable(monkeypatch):
    """Regression: a bare ``None`` broke the unpacking call site.

    ``optimize_isolated_molecules_batched`` does ``...[0]`` and
    ``optimize_adsorbate_slab_batched`` does ``a, b = ...``; both must keep
    working when the optional MLIP stack is partly unavailable.
    """
    from metalsurfer.optimization import _cache, _deps

    monkeypatch.setattr(_deps, "InFlightAutoBatcher", None, raising=False)
    result = _cache._get_inflight_autobatcher(object(), 100)

    assert isinstance(result, tuple)
    assert len(result) == 2
    autobatcher, cache_key = result
    assert autobatcher is None
    assert cache_key is None
    # Both call-site shapes must work.
    assert result[0] is None
    _a, _b = result


def test_optimize_and_evaluate_skips_preclear_when_saturation_reuse(monkeypatch):
    """Reuse path must not wipe the autobatcher cache before each BO batch."""
    from metalsurfer.workflow import shared as shared_mod

    from .conftest import make_placement_descriptor

    calls: list[dict] = []

    def _fake_clear(*_a, **kwargs):
        calls.append(dict(kwargs))

    monkeypatch.setattr(shared_mod, "clear_autobatcher_cache", _fake_clear)
    monkeypatch.setattr(
        shared_mod,
        "optimize_adsorbate_slab_batched",
        lambda *a, **k: [None],
    )

    slab = make_slab()
    combined = [slab.copy()]
    desc = make_placement_descriptor(placement_id=0)
    config = AdsorptionConfig(saturation_autobatcher_reuse=True)
    shared_mod._optimize_and_evaluate_placements(
        combined,
        [0],
        [desc],
        slab=slab,
        ts_model=object(),
        config=config,
        energies=(-1.0, -1.0),
        molecule_name="water",
        surface_symbols=["Pt"],
        saturation_reuse=True,
    )
    assert calls == []

    shared_mod._optimize_and_evaluate_placements(
        combined,
        [0],
        [desc],
        slab=slab,
        ts_model=object(),
        config=config,
        energies=(-1.0, -1.0),
        molecule_name="water",
        surface_symbols=["Pt"],
        saturation_reuse=False,
    )
    assert len(calls) == 1
    assert "max_n_atoms_threshold" in calls[0]
    assert calls[0].get("drain_cuda") in (None, False)

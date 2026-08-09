"""Autobatcher / capacity caches and their eviction helpers.

Cache-key lifecycle
-------------------
Both caches are keyed on ``id(ts_model)`` plus the tuning parameters that change
the memory estimate. ``id()`` is only unique among *live* objects, so a freed
model could in principle hand its address to a new one. In practice a
``FairChemModel`` is created once per run and kept alive for the whole workflow
(``setup_single_model`` returns it to the caller), and every cached value is
also keyed on the device string and the autobatcher tuning parameters, so the
residual collision risk is negligible. Callers that do swap models should call
:func:`clear_autobatcher_cache` with ``clear_capacity=True`` in between, since
capacity entries now outlive a single optimize call.

Concurrency
-----------
Both dicts are module-level mutable state, so every read-modify-write goes
through ``_CACHE_LOCK`` (a :class:`threading.RLock`). Callers outside this
module never touch the dicts directly: they use the small helpers below
(:func:`capacity_cache_get`, :func:`capacity_cache_set`,
:func:`pop_autobatcher`) so the lock never leaks. ``RLock`` rather than
``Lock`` because :func:`clear_autobatcher_cache` is reachable from ``finally``
blocks on paths that may already hold the lock. GPU and ``gc`` calls
(``torch.cuda.synchronize`` / ``empty_cache`` / ``ipc_collect`` /
``gc.collect``) are deliberately kept *outside* the locked region so a slow
CUDA sync never blocks another thread's cache lookup.

This makes the caches safe for threaded use. Multi-process isolation
(Ray/multiprocessing) is out of scope and already correct: separate processes
get separate module state.
"""

import contextlib
import gc
import logging
import threading
from typing import Any

from ..config import AdsorptionConfig
from . import _deps

logger = logging.getLogger(__name__)

_CACHE_LOCK = threading.RLock()
_AUTOBATCHER_CACHE: dict[tuple, Any] = {}
_PARALLEL_CAPACITY_CACHE: dict[tuple, int] = {}


def capacity_cache_get(cache_key: tuple) -> int | None:
    """Return the cached parallel-relaxation capacity for *cache_key*, if any."""
    with _CACHE_LOCK:
        return _PARALLEL_CAPACITY_CACHE.get(cache_key)


def capacity_cache_set(cache_key: tuple, n_systems: int) -> None:
    """Store the probed parallel-relaxation capacity for *cache_key*."""
    with _CACHE_LOCK:
        _PARALLEL_CAPACITY_CACHE[cache_key] = n_systems


def pop_autobatcher(cache_key: tuple) -> Any:
    """Remove and return the cached autobatcher for *cache_key* (``None`` if absent)."""
    with _CACHE_LOCK:
        return _AUTOBATCHER_CACHE.pop(cache_key, None)


def clear_autobatcher_cache(
    max_n_atoms_threshold: int | None = None,
    *,
    clear_capacity: bool = False,
) -> None:
    """Evict cached autobatchers to free GPU memory before larger runs.

    Call after isolated-molecule optimization (before slab+adsorbate) so memory
    from small-system batchers is released. If max_n_atoms_threshold is set,
    only evicts entries with max_n_atoms below it (keeps slab+adsorbate
    batchers when processing multiple molecules). If None, clears all.

    The parallel-capacity cache is *not* cleared by default. Autobatchers hold
    GPU tensors and are cheap to rebuild, so evicting them is the point of this
    function; the capacity cache in contrast holds a handful of ``int``s keyed
    on model identity + device + tuning parameters + ``max_n_atoms``, and each
    entry costs a full GPU memory probe to recompute. It must survive across
    molecules and BO batches, otherwise the probe is repeated per unit of work.
    Pass ``clear_capacity=True`` at a model/substrate boundary (e.g. when
    swapping ``ts_model``), where the estimate is no longer valid or no longer
    needed. It is ignored when *max_n_atoms_threshold* is set.
    """
    if max_n_atoms_threshold is None:
        with _CACHE_LOCK:
            # Drop strong refs explicitly: TorchSim InFlightAutoBatcher can retain GPU
            # tensors until batchers are GC'd; dict.clear() alone is easy to leave
            # garbage uncollected between pytest tests.
            _holders = list(_AUTOBATCHER_CACHE.values())
            _AUTOBATCHER_CACHE.clear()
            if clear_capacity:
                _PARALLEL_CAPACITY_CACHE.clear()
            del _holders
        logger.debug(
            "Cleared entire autobatcher cache (clear_capacity=%s)", clear_capacity
        )
    else:
        with _CACHE_LOCK:
            to_remove = [k for k in _AUTOBATCHER_CACHE if k[4] < max_n_atoms_threshold]
            for k in to_remove:
                del _AUTOBATCHER_CACHE[k]
        if to_remove:
            logger.debug(
                "Evicted %d autobatcher(s) with max_n_atoms < %d",
                len(to_remove),
                max_n_atoms_threshold,
            )
    torch = _deps.torch
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - needs GPU
        with contextlib.suppress(RuntimeError):
            torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
    if torch is not None and torch.cuda.is_available():  # pragma: no cover - needs GPU
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc):
            with contextlib.suppress(RuntimeError):
                ipc()


def _maybe_clear_cuda_cache(ts_model) -> None:
    """Clear CUDA cache before batched optimization to reduce OOM risk."""
    torch = _deps.torch
    if torch is None or not torch.cuda.is_available():
        return
    dev = getattr(ts_model, "device", None)
    on_cuda = (isinstance(dev, str) and dev.lower().startswith("cuda")) or (
        hasattr(dev, "type") and str(getattr(dev, "type", "")).lower() == "cuda"
    )
    if on_cuda:
        torch.cuda.empty_cache()


def _get_inflight_autobatcher(
    ts_model,
    max_n_atoms: int,
    *,
    memory_scales_with: str = "n_atoms",
    max_memory_padding: float = 0.5,
    max_memory_scaler: float | None = None,
    max_atoms_to_try: int = 100_000,
    config: AdsorptionConfig | None = None,
    saturation_reuse: bool = False,
):
    """Create or return cached InFlightAutoBatcher for batched relaxations."""
    if _deps.InFlightAutoBatcher is None or _deps.ts is None or ts_model is None:
        return None
    if config is not None:
        max_memory_padding = config.autobatcher_max_memory_padding
        max_memory_scaler = config.autobatcher_max_memory_scaler
    try:
        dev = getattr(ts_model, "device", None)
        key = (
            id(ts_model),
            str(dev),
            memory_scales_with,
            float(max_memory_padding),
            int(max_n_atoms),
            max_memory_scaler,
            int(max_atoms_to_try),
        )
        with _CACHE_LOCK:
            cached = _AUTOBATCHER_CACHE.get(key)
            if cached is not None:
                return cached, key, False
            if saturation_reuse:
                matching_candidates: list[tuple[tuple, Any]] = []
                for cache_key, cache_ab in _AUTOBATCHER_CACHE.items():
                    if (
                        cache_key[0] == key[0]
                        and cache_key[1] == key[1]
                        and cache_key[2] == key[2]
                        and cache_key[3] == key[3]
                        and cache_key[5] == key[5]
                        and cache_key[6] == key[6]
                    ):
                        matching_candidates.append((cache_key, cache_ab))
                matching_candidates.sort(key=lambda item: int(item[0][4]), reverse=True)
                growth_atoms = 32
                growth_fraction = 0.1
                if config is not None:
                    growth_atoms = config.saturation_autobatcher_reuse_growth_atoms
                    growth_fraction = (
                        config.saturation_autobatcher_reuse_growth_fraction
                    )
                for cache_key, cache_ab in matching_candidates:
                    cached_max_atoms = int(cache_key[4])
                    allowed_growth = max(
                        growth_atoms,
                        int(round(cached_max_atoms * growth_fraction)),
                    )
                    if (
                        cached_max_atoms
                        < max_n_atoms
                        <= (cached_max_atoms + allowed_growth)
                    ):
                        logger.info(
                            "Reusing saturation autobatcher from max_n_atoms=%d for %d "
                            "(allowed_growth=%d) to skip memory re-estimation",
                            cached_max_atoms,
                            max_n_atoms,
                            allowed_growth,
                        )
                        return cache_ab, cache_key, True
        kwargs: dict = {
            "memory_scales_with": memory_scales_with,
            "max_memory_padding": max_memory_padding,
            "max_atoms_to_try": max_atoms_to_try,
        }
        if max_memory_scaler is not None:
            kwargs["max_memory_scaler"] = max_memory_scaler
        # Constructed outside the lock: building an InFlightAutoBatcher can run a
        # GPU memory probe, which must not block other threads' cache lookups.
        ab: Any = _deps.InFlightAutoBatcher(ts_model, **kwargs)
        with _CACHE_LOCK:
            # A concurrent caller may have inserted the same key meanwhile; prefer
            # the already-published instance so all threads share one batcher.
            existing = _AUTOBATCHER_CACHE.get(key)
            if existing is not None:
                return existing, key, False
            _AUTOBATCHER_CACHE[key] = ab
        return ab, key, False
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to create InFlightAutoBatcher: %s", exc)
        return None, None, False

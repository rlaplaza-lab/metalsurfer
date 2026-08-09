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
:func:`clear_autobatcher_cache` in between.

Single-thread access is assumed; no lock is needed for current workflows.
"""

import contextlib
import gc
import logging
from typing import Any

from ..config import AdsorptionConfig
from . import _deps

logger = logging.getLogger(__name__)

_AUTOBATCHER_CACHE: dict[tuple, Any] = {}
_PARALLEL_CAPACITY_CACHE: dict[tuple, int] = {}


def clear_autobatcher_cache(
    max_n_atoms_threshold: int | None = None,
    ts_model=None,
) -> None:
    """Evict cached autobatchers to free GPU memory before larger runs.

    Call after isolated-molecule optimization (before slab+adsorbate) so memory
    from small-system batchers is released. If max_n_atoms_threshold is set,
    only evicts entries with max_n_atoms below it (keeps slab+adsorbate
    batchers when processing multiple molecules). If None, clears all.
    """
    if max_n_atoms_threshold is None:
        # Drop strong refs explicitly: TorchSim InFlightAutoBatcher can retain GPU
        # tensors until batchers are GC'd; dict.clear() alone is easy to leave
        # garbage uncollected between pytest tests.
        _holders = list(_AUTOBATCHER_CACHE.values())
        _AUTOBATCHER_CACHE.clear()
        _PARALLEL_CAPACITY_CACHE.clear()
        del _holders
        logger.debug("Cleared entire autobatcher cache")
    else:
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
                growth_fraction = config.saturation_autobatcher_reuse_growth_fraction
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
        ab: Any = _deps.InFlightAutoBatcher(ts_model, **kwargs)
        _AUTOBATCHER_CACHE[key] = ab
        return ab, key, False
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to create InFlightAutoBatcher: %s", exc)
        return None, None, False

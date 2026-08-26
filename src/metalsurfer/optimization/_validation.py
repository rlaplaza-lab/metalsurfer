"""Pure validation / resolution helpers for the optimization package.

Everything here is CPU-only logic (no MLIP stack required to execute), which
keeps the risky parts -- PBC checks, device fallback, geometry hashing, OOM
classification and autobatcher probe sizing -- unit-testable.
"""

import logging
import math
from typing import Any

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..exceptions import DependencyMissingError
from . import _deps

logger = logging.getLogger(__name__)

# Dynamic policy for the TorchSim ``max_atoms_to_try`` memory probe cap.
_DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER = 2.0
_DYNAMIC_AUTOBATCHER_CAP_MIN = 5_000
_DYNAMIC_AUTOBATCHER_CAP_MAX = 200_000
_DYNAMIC_AUTOBATCHER_CAP_BUCKET = 5_000


def _validate_model_pbc(atoms: Atoms, *, context: str) -> None:
    """Reject mixed-PBC systems before passing them to UMA/FairChem."""
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    if pbc.shape != (3,):
        raise ValueError(
            f"{context}: invalid PBC shape {pbc.shape}; expected 3 components."
        )
    if bool(pbc.all()) or bool((~pbc).all()):
        return
    raise ValueError(
        f"{context}: mixed PBC {pbc.tolist()} is not supported by the UMA model. "
        "Use either [True, True, True] with sufficient vacuum or [False, False, False]."
    )


def _positions_cell_hash(atoms: Atoms) -> int:
    """Fast content hash of positions + cell + atomic numbers for cache invalidation."""
    pos_bytes = atoms.get_positions().tobytes()
    cell_bytes = np.asarray(atoms.get_cell()).tobytes()
    num_bytes = atoms.get_atomic_numbers().tobytes()
    return hash((pos_bytes, cell_bytes, num_bytes))


def _resolve_ts_optimizer(name: str) -> Any:
    """Map config string to ``ts.Optimizer`` enum member."""
    ts = _deps.ts
    if ts is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "_resolve_ts_optimizer",
            "Install with: pip install torch-sim-atomistic",
        )
    _map = {
        "fire": ts.Optimizer.fire,
        "lbfgs": ts.Optimizer.lbfgs,
        "bfgs": ts.Optimizer.bfgs,
    }
    return _map[name]


def _device_is_cuda(device: Any) -> bool:
    """Return True when *device* names a CUDA target (str or torch.device)."""
    if isinstance(device, str):
        return device.lower().startswith("cuda")
    type_attr = getattr(device, "type", None)
    return isinstance(type_attr, str) and type_attr.lower() == "cuda"


def _device_key(device: Any) -> str:
    """Stable string key for cache entries; ``None`` maps to ``unknown``."""
    return str(device) if device is not None else "unknown"


def _resolve_device(device: str | None) -> str | None:
    """Resolve device string: use CPU when CUDA is requested but unavailable.

    Enables tests and CI (no GPU) to run without false failures when
    config defaults to device='cuda'.
    """
    if device is None or not _device_is_cuda(device):
        return device
    torch = _deps.torch
    if torch is None:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            logger.info("CUDA requested but not available; using CPU")
            return "cpu"
    except (RuntimeError, AttributeError):
        return "cpu"
    return device


def _resolve_model_device(ts_model: Any, config: AdsorptionConfig) -> str:
    """Resolve the device used for TorchSim state / autobatcher decisions."""
    model_device = getattr(ts_model, "device", None)
    if model_device is None:
        model_device = _resolve_device(config.device)
    if model_device is None:
        model_device = "cpu"
    return model_device


def _is_cuda_oom_error(exc: BaseException) -> bool:
    """Check whether *exc* looks like a CUDA out-of-memory failure."""
    torch = _deps.torch
    if torch is not None:
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cuda out of memory" in message
    )


def _is_batcher_capacity_error(exc: BaseException) -> bool:
    """Check whether TorchSim's batcher refused states beyond its probed bucket.

    ``InFlightAutoBatcher.load_states`` raises ``ValueError`` when an incoming
    system's memory-scaler metric exceeds the capacity probed at batcher
    creation. A cached/reused batcher can hit this when free VRAM shrank since
    the probe; callers should rebuild the autobatcher (fresh probe) and retry.
    """
    return "greater than max_metric" in str(exc)


def _resolve_autobatcher_max_atoms_to_try(
    *,
    max_n_atoms: int,
    n_systems: int,
    config: AdsorptionConfig,
) -> tuple[int, str]:
    """Resolve probe cap for TorchSim memory estimation.

    Returns ``(cap, source)`` where source is ``"config_override"`` or ``"dynamic"``.
    """
    override = config.autobatcher_max_atoms_to_try
    if override is not None:
        return int(override), "config_override"
    estimated = _DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER * max_n_atoms * n_systems
    bucketed = int(
        math.ceil(estimated / _DYNAMIC_AUTOBATCHER_CAP_BUCKET)
        * _DYNAMIC_AUTOBATCHER_CAP_BUCKET
    )
    cap = max(_DYNAMIC_AUTOBATCHER_CAP_MIN, min(_DYNAMIC_AUTOBATCHER_CAP_MAX, bucketed))
    return cap, "dynamic"


def _parallel_capacity_cache_key(
    ts_model,
    max_n_atoms: int,
    config: AdsorptionConfig,
    *,
    frozen_indices: list[int] | None = None,
) -> tuple:
    """Cache key for :func:`estimate_parallel_relaxation_capacity`.

    Keyed on ``id(ts_model)``; see the note in :mod:`._cache` about the
    lifetime assumption that makes this safe.
    """
    dev = getattr(ts_model, "device", None)
    frozen_key = tuple(sorted(int(i) for i in (frozen_indices or [])))
    return (
        id(ts_model),
        _device_key(dev),
        max_n_atoms,
        frozen_key,
        config.autobatcher_max_memory_padding,
        config.autobatcher_max_memory_scaler,
        config.autobatcher_max_atoms_to_try,
    )

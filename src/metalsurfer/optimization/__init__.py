"""TorchSim/ASE batched relaxations with slab freeze masks (`top_layer_tolerance`, etc.).

Package layout (all public names are re-exported here, so
``from metalsurfer.optimization import optimize_adsorbate_slab_batched`` keeps
working exactly as it did when this was a single module):

``_deps``
    Sole owner of the optional ``torch`` / ``torch_sim`` imports, the CUDA
    ``OutOfMemoryError`` registration and the ``_split_state`` device patch.
``_validation``
    Pure CPU logic: PBC validation, device resolution, geometry hashing, CUDA
    OOM classification and autobatcher probe-cap sizing.
``_cache``
    ``InFlightAutoBatcher`` / parallel-capacity caches plus their eviction and
    CUDA-cache helpers.
``_model``
    FairChem model setup and the ASE-compatible :class:`TorchSimCalculator`.
``_optimize``
    Batched relaxations, batched single points and GPU capacity probing.

Cache lifecycle
---------------
``_cache`` keys every entry on ``id(ts_model)`` together with the device string
and the autobatcher tuning parameters. A ``ts_model`` is created once per run
and held alive by the caller for the whole workflow, so the id stays stable;
callers that swap models mid-run should call :func:`clear_autobatcher_cache`
in between. See :mod:`metalsurfer.optimization._cache` for details.
"""

from ._cache import clear_autobatcher_cache
from ._model import TorchSimCalculator, setup_torchsim_model
from ._optimize import (
    batch_static,
    estimate_parallel_relaxation_capacity,
    optimize_adsorbate_slab_batched,
    optimize_isolated_molecules_batched,
    setup_single_model,
)
from ._validation import _resolve_device

__all__ = [
    "TorchSimCalculator",
    "_resolve_device",
    "batch_static",
    "clear_autobatcher_cache",
    "estimate_parallel_relaxation_capacity",
    "optimize_adsorbate_slab_batched",
    "optimize_isolated_molecules_batched",
    "setup_single_model",
    "setup_torchsim_model",
]

"""Optional heavy dependencies (``torch`` / ``torch_sim``) for the optimization package.

This module is the single owner of the MLIP-stack imports so the rest of the
package keeps importing (and stays unit-testable) in environments without
torch/torch-sim installed -- e.g. ``from metalsurfer import AdsorptionConfig``
must work in CI.

Consumers access the symbols through the module object (``_deps.ts``,
``_deps.torch``, ...) instead of importing the names directly, so a single
``monkeypatch.setattr(_deps, "ts", stub)`` is visible everywhere.

This module must not import any other ``metalsurfer.optimization`` submodule.
"""

from typing import Any

try:
    import torch as _torch_mod
except ImportError:  # pragma: no cover - depends on the optional MLIP stack
    torch: Any = None
else:  # pragma: no cover - depends on the optional MLIP stack
    torch = _torch_mod

# GPU memory probe errors (``torch.cuda.OutOfMemoryError`` subclasses ``RuntimeError``).
_CAPACITY_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    MemoryError,
    OSError,
)

ts: Any = None
ts_constraints: Any = None
InFlightAutoBatcher: Any = None
determine_max_batch_size: Any = None
calculate_memory_scalers: Any = None

try:  # pragma: no cover - requires the optional MLIP stack
    import torch_sim as _ts_mod
    import torch_sim.constraints as _ts_constraints_mod
    from torch_sim.autobatching import (
        InFlightAutoBatcher as _InFlightAutoBatcher,
    )
    from torch_sim.autobatching import (
        calculate_memory_scalers as _calculate_memory_scalers,
    )
    from torch_sim.autobatching import (
        determine_max_batch_size as _determine_max_batch_size,
    )

    ts = _ts_mod
    ts_constraints = _ts_constraints_mod
    InFlightAutoBatcher = _InFlightAutoBatcher
    determine_max_batch_size = _determine_max_batch_size
    calculate_memory_scalers = _calculate_memory_scalers
except ImportError:
    pass

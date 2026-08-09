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

# Errors that a GPU memory probe may raise. ``torch.cuda.OutOfMemoryError`` is
# appended when torch is available (it is not a subclass of MemoryError).
_CAPACITY_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    MemoryError,
    OSError,
)
if torch is not None:  # pragma: no cover - requires the optional MLIP stack
    _cuda = getattr(torch, "cuda", None)
    _oom = getattr(_cuda, "OutOfMemoryError", None) if _cuda is not None else None
    if isinstance(_oom, type) and issubclass(_oom, BaseException):
        _CAPACITY_PROBE_ERRORS = (*_CAPACITY_PROBE_ERRORS, _oom)

ts: Any = None
ts_constraints: Any = None
InFlightAutoBatcher: Any = None
determine_max_batch_size: Any = None
calculate_memory_scalers: Any = None


def _patched_split_state(state):  # pragma: no cover - requires the MLIP stack
    """Device-safe replacement for ``torch_sim.state._split_state``.

    Upstream ``_split_state`` uses ``torch.arange(...)`` without ``device=``, so
    the index bounds live on CPU while constraint tensors are on CUDA. This
    variant threads ``state.device`` through every tensor it creates.
    """
    from torch_sim.state import get_attrs_for_scope

    system_sizes = state.n_atoms_per_system.tolist()
    split_per_atom = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-atom"):
        if attr_name != "system_idx":
            split_per_atom[attr_name] = torch.split(attr_value, system_sizes, dim=0)
    split_per_system = {}
    for attr_name, attr_value in get_attrs_for_scope(state, "per-system"):
        if isinstance(attr_value, torch.Tensor):
            split_per_system[attr_name] = torch.split(attr_value, 1, dim=0)
        else:
            split_per_system[attr_name] = [attr_value] * state.n_systems
    global_attrs = dict(get_attrs_for_scope(state, "global"))
    states = []
    n_systems = len(system_sizes)
    zero_tensor = torch.tensor([0], device=state.device, dtype=torch.long)
    cumsum_atoms = torch.cat(
        (zero_tensor, torch.cumsum(state.n_atoms_per_system, dim=0))
    )
    for sys_idx in range(n_systems):
        per_system_dict = {
            attr_name: split_per_system[attr_name][sys_idx]
            for attr_name in split_per_system
        }
        system_attrs = {
            "system_idx": torch.zeros(
                system_sizes[sys_idx], device=state.device, dtype=torch.long
            ),
            **{
                attr_name: split_per_atom[attr_name][sys_idx]
                for attr_name in split_per_atom
            },
            **per_system_dict,
            **global_attrs,
        }
        atom_idx = torch.arange(
            cumsum_atoms[sys_idx].item(),
            cumsum_atoms[sys_idx + 1].item(),
            device=state.device,
        )
        new_constraints = [
            new_constraint
            for constraint in state.constraints
            if (new_constraint := constraint.select_sub_constraint(atom_idx, sys_idx))
        ]
        system_attrs["_constraints"] = new_constraints
        states.append(type(state)(**system_attrs))
    return states


try:  # pragma: no cover - requires the optional MLIP stack
    import torch_sim as _ts_mod
    import torch_sim.constraints as _ts_constraints_mod
    import torch_sim.state as _ts_state
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

    _ts_state._split_state = _patched_split_state
except ImportError:
    pass

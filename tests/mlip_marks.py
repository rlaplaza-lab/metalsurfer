"""MLIP and GPU pytest marks.

Torch, FairChem, and CUDA are imported only when the session can run ``mlip``
or ``gpu`` tests. ``pytest -m quick`` leaves that stack unloaded.
"""

from __future__ import annotations

import functools

import pytest

from . import _session_flags

_MLIP_REASON = "MLIP stack (torch/fairchem/torch-sim-atomistic) not installed"
_CUDA_REASON = "CUDA GPU required; skipped in CI (no GPU)"


def torch_is_installed() -> bool:
    """Whether torch imports. Skips the CUDA probe when the session cannot use it."""
    if not _session_flags.mlip_probe_enabled:
        return False
    from .optional_deps import has_torch

    return bool(has_torch)


def _skip_without_mlip() -> pytest.MarkDecorator:
    from .optional_deps import has_mlip_stack

    return pytest.mark.skipif(not has_mlip_stack, reason=_MLIP_REASON)


def _skip_without_cuda() -> pytest.MarkDecorator:
    from .optional_deps import cuda_available

    return pytest.mark.skipif(not cuda_available, reason=_CUDA_REASON)


def _cpu_marks() -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.mlip]
    if _session_flags.mlip_probe_enabled:
        marks.append(_skip_without_mlip())
    return marks


def _gpu_marks() -> list[pytest.MarkDecorator]:
    marks = [pytest.mark.mlip, pytest.mark.gpu, pytest.mark.no_fork]
    if _session_flags.mlip_probe_enabled:
        marks.append(_skip_without_mlip())
        marks.append(_skip_without_cuda())
    return marks


MLIP_CPU_MARKS = _cpu_marks()
GPU_MLIP_MARKS = _gpu_marks()


def gpu_mlip_test(fn):
    for mark in reversed(GPU_MLIP_MARKS):
        fn = mark(fn)
    return fn


@functools.lru_cache(maxsize=4)
def load_uma(model_name: str, device: str, task_name: str):
    """One UMA load per process. Imported here so collection stays free of MLIP."""
    from metalsurfer.optimization import setup_single_model

    return setup_single_model(model_name, device, task_name=task_name)

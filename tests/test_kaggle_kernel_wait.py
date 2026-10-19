"""The Kaggle poller must not treat a rejected push as a finished run."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1] / ".github" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location(
    "kaggle_run_kernel", _SCRIPTS / "kaggle_run_kernel.py"
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
advance_kernel_wait = _MODULE.advance_kernel_wait
push_was_rejected = _MODULE.push_was_rejected

_COMPLETE = 'metalsurfergpuci-full has status "KernelWorkerStatus.COMPLETE"'
_RUNNING = 'metalsurfergpuci-full has status "KernelWorkerStatus.RUNNING"'


def test_push_quota_error_is_rejected() -> None:
    output = "Kernel push error: Maximum batch GPU session count of 2 reached.\n"
    assert push_was_rejected(output)
    assert not push_was_rejected("Kernel version 42 pushed.\n")


def test_immediate_complete_is_not_success_until_grace_expires() -> None:
    decision, saw_active, stale_elapsed = advance_kernel_wait(
        _COMPLETE,
        saw_active=False,
        stale_elapsed=0,
        fetch_seconds=15,
        stale_complete_seconds=300,
    )
    assert decision is None
    assert saw_active is False
    assert stale_elapsed == 15


def test_complete_without_an_active_poll_becomes_stale() -> None:
    decision, saw_active, stale_elapsed = advance_kernel_wait(
        _COMPLETE,
        saw_active=False,
        stale_elapsed=285,
        fetch_seconds=15,
        stale_complete_seconds=300,
    )
    assert decision == "stale"
    assert saw_active is False
    assert stale_elapsed == 300


def test_complete_after_running_is_the_new_kernel() -> None:
    running, saw_active, stale_elapsed = advance_kernel_wait(
        _RUNNING,
        saw_active=False,
        stale_elapsed=30,
        fetch_seconds=15,
    )
    assert running is None
    assert saw_active is True
    assert stale_elapsed == 0

    decision, saw_active, stale_elapsed = advance_kernel_wait(
        _COMPLETE,
        saw_active=saw_active,
        stale_elapsed=stale_elapsed,
        fetch_seconds=15,
    )
    assert decision == "complete"
    assert saw_active is True
    assert stale_elapsed == 0


def test_kernel_error_and_cancel_stop_the_wait() -> None:
    error, _, _ = advance_kernel_wait(
        'slug has status "KernelWorkerStatus.ERROR"',
        saw_active=True,
        stale_elapsed=0,
        fetch_seconds=15,
    )
    assert error == "error"
    cancelled, _, _ = advance_kernel_wait(
        'slug has status "KernelWorkerStatus.CANCELLED"',
        saw_active=True,
        stale_elapsed=0,
        fetch_seconds=15,
    )
    assert cancelled == "cancel"

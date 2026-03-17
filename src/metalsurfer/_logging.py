"""Contextual logging: log_context() and ContextFilter inject molecule/surface_type/etc. into log records."""

import contextvars
import logging
from contextlib import contextmanager
from typing import Any

_LOG_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "adsorption_log_ctx", default=None
)


@contextmanager
def screening_log_context():
    """Add ContextFilter to metalsurfer logger for the duration of the block."""
    pkg_logger = logging.getLogger("metalsurfer")
    ctx_filter = ContextFilter()
    pkg_logger.addFilter(ctx_filter)
    try:
        yield
    finally:
        pkg_logger.removeFilter(ctx_filter)


@contextmanager
def log_context(**kwargs: Any):
    """Push key-value pairs into logging context for this scope."""
    prev = _LOG_CTX.get() or {}
    merged = {**prev, **kwargs}
    token = _LOG_CTX.set(merged)
    try:
        yield merged
    finally:
        _LOG_CTX.reset(token)


def get_log_context() -> dict[str, Any]:
    """Return current logging context (read-only)."""
    return dict(_LOG_CTX.get() or {})


_warned_once: set[str] = set()


def warn_once(logger: logging.Logger, key: str, message: str) -> None:
    """Emit a warning log message at most once per *key* across the process."""
    if key not in _warned_once:
        _warned_once.add(key)
        logger.warning(message)


class ContextFilter(logging.Filter):
    """Inject ctx_prefix into log records from current context."""

    _KEY_ORDER = ("molecule", "surface_type", "placement_id", "seed")

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CTX.get() or {}
        if ctx:
            parts = []
            for k in self._KEY_ORDER:
                if k in ctx:
                    parts.append(f"{k}={ctx[k]}")
            for k, v in ctx.items():
                if k not in self._KEY_ORDER:
                    parts.append(f"{k}={v}")
            record.ctx_prefix = "[" + " ".join(parts) + "] "
        else:
            record.ctx_prefix = ""
        return True

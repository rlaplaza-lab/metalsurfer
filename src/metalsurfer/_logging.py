"""Context vars and formatters for structured logging (e.g. ``ctx_prefix``)."""

import contextvars
import io
import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, TypeGuard

_LOG_CTX: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "adsorption_log_ctx", default=None
)
_FACTORY_INSTALLED = False
CTX_KEY_ORDER = ("molecule", "surface_type", "placement_id", "seed")


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


def _format_ctx_prefix(ctx: dict[str, Any]) -> str:
    if not ctx:
        return ""
    parts = []
    for k in CTX_KEY_ORDER:
        if k in ctx:
            parts.append(f"{k}={ctx[k]}")
    for k, v in ctx.items():
        if k not in CTX_KEY_ORDER:
            parts.append(f"{k}={v}")
    return "[" + " ".join(parts) + "] "


class ContextFilter(logging.Filter):
    """Inject ctx_prefix into log records from current context."""

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _LOG_CTX.get() or {}
        record.ctx_prefix = _format_ctx_prefix(ctx)
        return True


def _ctx_prefix_from_context() -> str:
    return _format_ctx_prefix(_LOG_CTX.get() or {})


def _install_log_record_factory() -> None:
    """Ensure every LogRecord always has ctx_prefix."""
    global _FACTORY_INSTALLED
    if _FACTORY_INSTALLED:
        return

    old_factory = logging.getLogRecordFactory()

    def record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        if not hasattr(record, "ctx_prefix"):
            record.ctx_prefix = _ctx_prefix_from_context()
        return record

    logging.setLogRecordFactory(record_factory)
    _FACTORY_INSTALLED = True


def ensure_log_record_defaults() -> None:
    """Ensure LogRecords include required contextual fields."""
    _install_log_record_factory()


class _LogStreamToLogger(io.TextIOBase):
    """File-like object that forwards writes to a logger.

    Designed for capturing non-logging output (e.g. progress bars) and turning
    it into coherent line-based log records.
    """

    def __init__(
        self,
        *,
        logger: logging.Logger,
        level: int,
        carriage_return_rate_limit_s: float = 1.0,
    ):
        super().__init__()
        self._logger = logger
        self._level = level
        self._rate_limit_s = float(carriage_return_rate_limit_s)

        self._pending: str = ""
        self._last_cr_text: str = ""
        self._last_emit_t: float = time.monotonic()
        self._last_emit_msg: str = ""

    def writable(self) -> bool:  # pragma: no cover
        return True

    def isatty(self) -> bool:  # pragma: no cover
        return False

    def _maybe_emit_cr_snapshot(self, snapshot: str) -> None:
        msg = snapshot.strip()
        if not msg:
            return

        now = time.monotonic()
        if now - self._last_emit_t >= self._rate_limit_s:
            if msg != self._last_emit_msg:
                self._logger.log(self._level, msg)
                self._last_emit_t = now
                self._last_emit_msg = msg
            self._last_cr_text = ""

    def _emit_final_line(self) -> None:
        msg = self._pending.strip()
        self._pending = ""
        if msg:
            self._logger.log(self._level, msg)
            self._last_emit_t = time.monotonic()
            self._last_emit_msg = msg
        self._last_cr_text = ""

    def flush(self) -> None:
        # Emit any trailing content at context exit.
        if self._pending.strip():
            self._emit_final_line()
            return
        if self._last_cr_text.strip():
            self._logger.log(self._level, self._last_cr_text.strip())
            self._last_cr_text = ""

    def write(self, s: str) -> int:
        if not s:
            return 0
        if not isinstance(s, str):
            s = str(s)

        for ch in s:
            if ch == "\n":
                if self._pending:
                    self._emit_final_line()
                else:
                    if self._last_cr_text.strip():
                        self._logger.log(self._level, self._last_cr_text.strip())
                        self._last_cr_text = ""
                continue

            if ch == "\r":
                snapshot = self._pending
                self._pending = ""
                if snapshot.strip():
                    self._last_cr_text = snapshot.strip()
                    self._maybe_emit_cr_snapshot(snapshot)
                else:
                    self._last_cr_text = ""
                continue

            self._pending += ch

        return len(s)


def _parse_level(level_name: str, default: int) -> int:
    level = getattr(logging, str(level_name).upper(), None)
    return level if isinstance(level, int) else default


def _ensure_context_filter(handler: logging.Handler) -> None:
    """Attach ContextFilter once so ctx_prefix is set at format time."""
    if not any(isinstance(f, ContextFilter) for f in handler.filters):
        handler.addFilter(ContextFilter())


def _is_console_stream_handler(
    handler: logging.Handler,
) -> TypeGuard[logging.StreamHandler]:
    """Return True only for handlers writing to the real stdout/stderr.

    ``logging.FileHandler`` subclasses ``StreamHandler``, so a naive isinstance
    check would retarget a caller's file handler at stdout (silently killing
    file logging and leaking the open file object). Restrict retargeting to
    handlers that are already pointed at a console stream.
    """
    if not isinstance(handler, logging.StreamHandler):
        return False
    if isinstance(handler, logging.FileHandler):
        return False
    stream = getattr(handler, "stream", None)
    if stream is None:
        return False
    return any(
        stream is console
        for console in (sys.stdout, sys.stderr, sys.__stdout__, sys.__stderr__)
    )


def configure_logging(
    *,
    default_level: str = "INFO",
    fmt: str = "%(asctime)s %(name)s %(levelname)s %(ctx_prefix)s%(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Configure project logging with sane HPC defaults.

    Environment overrides:
    - METALSURFER_LOG_LEVEL (default: INFO)
    - TORCHSIM_LOG_LEVEL (default: WARNING)

    Notes:
    - Metalsurfer logs are routed to stdout (not stderr) so HPC job
      launchers that split stdout/stderr capture INFO logs in `.out`.
    - When running under pytest, we avoid reconfiguring stream handlers by
      default to avoid interacting badly with pytest's logging capture.
    """
    _install_log_record_factory()

    root = logging.getLogger()
    level_name = os.getenv("METALSURFER_LOG_LEVEL", default_level)
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    running_pytest = "PYTEST_CURRENT_TEST" in os.environ or any(
        name.startswith("pytest") for name in sys.modules
    )
    force_stdout = os.getenv("METALSURFER_FORCE_STDOUT_LOGS", "") == "1"
    target_stream = sys.stdout if (force_stdout or not running_pytest) else sys.stderr

    root.setLevel(_parse_level(level_name, root.level))

    # Avoid touching stream handlers under pytest unless explicitly forced.
    if not (running_pytest and not force_stdout):
        if not root.handlers:
            logging.basicConfig(
                level=_parse_level(level_name, logging.INFO),
                format=fmt,
                datefmt=datefmt,
                stream=target_stream,
            )
        else:
            root.setLevel(_parse_level(level_name, root.level))

        stream_handler_found = False
        for handler in root.handlers:
            if _is_console_stream_handler(handler):
                handler.setStream(target_stream)
                handler.setFormatter(formatter)
                _ensure_context_filter(handler)
                stream_handler_found = True

        if not stream_handler_found:
            sh = logging.StreamHandler(target_stream)
            sh.setFormatter(formatter)
            _ensure_context_filter(sh)
            root.addHandler(sh)

    torchsim_level = _parse_level(
        os.getenv("TORCHSIM_LOG_LEVEL", "WARNING"),
        logging.WARNING,
    )
    for logger_name in ("torch_sim", "torchsim", "torch_sim.autobatching"):
        torch_logger = logging.getLogger(logger_name)
        torch_logger.setLevel(torchsim_level)
        # Avoid changing propagation if TorchSim already installed its own handlers.
        # When there are no handlers, propagation ensures logs reach our root setup.
        if not torch_logger.handlers:
            torch_logger.propagate = True
        if not (running_pytest and not force_stdout):
            for handler in torch_logger.handlers:
                if _is_console_stream_handler(handler):
                    handler.setStream(target_stream)
                    handler.setFormatter(formatter)


@contextmanager
def torchsim_output_capture(
    *,
    logger_name: str = "metalsurfer.torchsim",
    stdout_level: int = logging.INFO,
    stderr_level: int = logging.WARNING,
    carriage_return_rate_limit_s: float = 1.0,
):
    """Capture TorchSim's stdout/stderr and route through logging.

    Useful because TorchSim (and its progress bars) may print directly to
    stdout/stderr, bypassing Python's logging configuration.

    Notes:
    - stdout is mapped to INFO, stderr is mapped to WARNING.
    - stdout updates using carriage return (``\\r``) are rate-limited so we
      don't emit thousands of near-identical log lines.
    """

    pkg_logger = logging.getLogger(logger_name)
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    out_stream = _LogStreamToLogger(
        logger=pkg_logger,
        level=stdout_level,
        carriage_return_rate_limit_s=carriage_return_rate_limit_s,
    )
    err_stream = _LogStreamToLogger(
        logger=pkg_logger,
        level=stderr_level,
        carriage_return_rate_limit_s=carriage_return_rate_limit_s,
    )
    try:
        sys.stdout = out_stream
        sys.stderr = err_stream
        yield
    finally:
        try:
            out_stream.flush()
        finally:
            sys.stdout = old_stdout
        try:
            err_stream.flush()
        finally:
            sys.stderr = old_stderr


# Install record defaults at import time so early log emissions are safe
# even before configure_logging() is called by library entrypoints.
ensure_log_record_defaults()

"""Shared logging test helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager


class CaptureHandler(logging.Handler):
    def __init__(self, sink: list[logging.LogRecord]):
        super().__init__()
        self._sink = sink

    def emit(self, record: logging.LogRecord) -> None:
        self._sink.append(record)


@contextmanager
def configured_logger(
    logger: logging.Logger,
    *,
    level: int = logging.INFO,
    propagate: bool = False,
    handler: logging.Handler | None = None,
):
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    logger.handlers.clear()
    if handler is not None:
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = propagate
    try:
        yield
    finally:
        logger.handlers = old_handlers
        logger.setLevel(old_level)
        logger.propagate = old_propagate

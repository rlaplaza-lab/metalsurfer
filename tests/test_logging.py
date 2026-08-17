"""Tests for the CR-aware log stream in :mod:`metalsurfer._logging`."""

import logging

from metalsurfer._logging import _LogStreamToLogger, configure_logging

from ._logging_helpers import CaptureHandler, configured_logger


class TestLogStreamToLogger:
    def test_cr_rate_limited_not_duplicated(self):
        """Rapid CR-only updates within the rate window are coalesced."""
        logger = logging.getLogger("test.cr.ratelimit")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=100.0
            )
            stream.write("a\r")
            stream.write("b\r")
            stream.write("c\r")
            # Within the rate window, nothing is emitted yet.
            assert len(sink) == 0
            # Flush emits only the latest snapshot, exactly once.
            stream.flush()
            assert len(sink) == 1
            assert sink[0].getMessage() == "c"

    def test_cr_emitted_immediately_when_rate_limit_zero(self):
        """With a zero rate limit each CR snapshot is emitted once."""
        logger = logging.getLogger("test.cr.zerolimit")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=0.0
            )
            stream.write("a\r")
            stream.write("b\r")
            assert len(sink) == 2
            assert sink[0].getMessage() == "a"
            assert sink[1].getMessage() == "b"

    def test_trailing_newline_flushes_pending(self):
        """A trailing newline flushes any buffered (non-CR) content."""
        logger = logging.getLogger("test.cr.newline")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=100.0
            )
            stream.write("hello")
            assert len(sink) == 0
            stream.write("\n")
            assert len(sink) == 1
            assert sink[0].getMessage() == "hello"

    def test_flush_emits_trailing_cr_text(self):
        """flush emits the last CR snapshot when no pending line exists."""
        logger = logging.getLogger("test.cr.flush")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=100.0
            )
            stream.write("status\r")
            assert len(sink) == 0
            stream.flush()
            assert len(sink) == 1
            assert sink[0].getMessage() == "status"

    def test_newlines_split_records(self):
        """Each newline-terminated chunk becomes its own log record."""
        logger = logging.getLogger("test.cr.split")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=100.0
            )
            stream.write("line1\nline2\n")
            assert len(sink) == 2
            assert sink[0].getMessage() == "line1"
            assert sink[1].getMessage() == "line2"

    def test_incomplete_line_buffered_until_flush(self):
        """Content without a trailing newline is buffered until flush."""
        logger = logging.getLogger("test.cr.partial")
        sink: list[logging.LogRecord] = []
        handler = CaptureHandler(sink)
        with configured_logger(logger, handler=handler):
            stream = _LogStreamToLogger(
                logger=logger, level=logging.INFO, carriage_return_rate_limit_s=100.0
            )
            stream.write("partial")
            assert len(sink) == 0
            stream.flush()
            assert len(sink) == 1
            assert sink[0].getMessage() == "partial"


def test_configure_logging_preserves_file_handlers(tmp_path, monkeypatch):
    """Regression: ``FileHandler`` subclasses ``StreamHandler``.

    The isinstance sweep pointed the caller's file handler at stdout, silently
    killing file logging and dropping the open file object without closing it.
    """
    monkeypatch.setenv("METALSURFER_FORCE_STDOUT_LOGS", "1")
    log_path = tmp_path / "run.log"
    file_handler = logging.FileHandler(log_path)
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(file_handler)
    try:
        root.setLevel(logging.INFO)
        logging.getLogger("metalsurfer.test").info("before configure")
        configure_logging()
        logging.getLogger("metalsurfer.test").info("after configure")
        file_handler.flush()

        assert file_handler.stream.name == str(log_path)
        contents = log_path.read_text()
        assert "before configure" in contents
        assert "after configure" in contents
    finally:
        root.removeHandler(file_handler)
        file_handler.close()
        root.setLevel(previous_level)

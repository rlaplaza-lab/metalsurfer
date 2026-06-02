"""Timing utilities for performance measurement."""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def timer(name: str | None = None) -> Generator[list[float], None, None]:
    """Context manager for timing code blocks.

    Parameters
    ----------
    name
        Optional name for the timed block (used for logging).

    Yields
    ------
    list[float]
        A single-element list that will contain the elapsed time in seconds
        after the block completes. Use `elapsed[0]` to access the value.

    Examples
    --------
    >>> with timer("my_operation") as elapsed:
    ...     time.sleep(0.1)
    >>> print(f"Took {elapsed[0]:.2f}s")
    """
    elapsed: list[float] = []
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.append(time.perf_counter() - start)
        if name is not None:
            logger.debug("%s took %.3fs", name, elapsed[0])


@contextmanager
def timer_silent() -> Generator[list[float], None, None]:
    """Context manager for timing code blocks without logging.

    Yields a list that will contain the elapsed time after the block exits.

    Examples
    --------
    >>> with timer_silent() as t:
    ...     time.sleep(0.1)
    >>> print(f"Took {t[0]:.2f}s")
    """
    elapsed: list[float] = []
    start = time.perf_counter()
    try:
        yield elapsed
    finally:
        elapsed.append(time.perf_counter() - start)


def timed(
    name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., tuple[T, float]]]:
    """Decorator for timing function execution.

    Parameters
    ----------
    name
        Optional name for the timed function. If not provided, uses the function name.

    Returns
    -------
    Callable
        Decorated function that returns a tuple of (result, elapsed_time).

    Examples
    --------
    >>> @timed()
    ... def my_func():
    ...     time.sleep(0.1)
    ...     return 42
    >>> result, elapsed = my_func()
    >>> print(f"Result: {result}, took {elapsed:.2f}s")
    """

    def decorator(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
        func_name = name if name is not None else func.__name__

        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> tuple[T, float]:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug("%s took %.3fs", func_name, elapsed)
            return result, elapsed

        return wrapper

    return decorator


def timed_silent(
    name: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., tuple[T, float]]]:
    """Decorator for timing function execution without logging.

    Parameters
    ----------
    name
        Optional name for the timed function (ignored, for API consistency).

    Returns
    -------
    Callable
        Decorated function that returns a tuple of (result, elapsed_time).

    Examples
    --------
    >>> @timed_silent()
    ... def my_func():
    ...     time.sleep(0.1)
    ...     return 42
    >>> result, elapsed = my_func()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., tuple[T, float]]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> tuple[T, float]:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            return result, elapsed

        return wrapper

    return decorator

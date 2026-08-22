"""Placement-package fixtures."""

import pytest

from metalsurfer.placement.dissociative import (
    _DISSOCIATIVE_PAIR_CACHE,
    _DISSOCIATIVE_PAIR_CACHE_LOCK,
)
from metalsurfer.placement.site_context import (
    _SITE_CONTEXT_CACHE,
    _SITE_CONTEXT_CACHE_LOCK,
)


def _clear_placement_caches() -> None:
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()
    with _DISSOCIATIVE_PAIR_CACHE_LOCK:
        _DISSOCIATIVE_PAIR_CACHE.clear()


@pytest.fixture(autouse=True)
def _reset_placement_caches():
    """Isolate process-local site caches between tests."""
    _clear_placement_caches()
    yield
    _clear_placement_caches()

"""Internal utilities shared across metalsurfer sub-packages."""

from math import isfinite


def is_finite_number(value: object) -> bool:
    """Return True if *value* converts to a finite float."""
    if not isinstance(value, (int, float, str)):
        return False
    try:
        return bool(isfinite(float(value)))
    except (TypeError, ValueError):
        return False

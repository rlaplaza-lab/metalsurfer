"""Shared CSV cell coercion helpers for result rows."""

import json
from typing import Any


def is_missing(value: Any) -> bool:
    """Check whether a CSV cell value represents a missing entry.

    Parameters
    ----------
    value
        Cell value to inspect.
    """
    if value is None:
        return True
    text = str(value).strip()
    return text in {"", "nan", "none", "None"}


def with_default(value: Any, default: Any) -> Any:
    """Return *default* if *value* is missing, otherwise *value*.

    Parameters
    ----------
    value
        Cell value to inspect.
    default
        Fallback to return when *value* is missing.
    """
    return default if is_missing(value) else value


def float_or(value: Any, default: float) -> float:
    """Coerce a CSV cell to float, falling back to *default*.

    Parameters
    ----------
    value
        Cell value to coerce.
    default
        Fallback when *value* is missing.
    """
    return float(with_default(value, default))


def int_or_none(value: Any) -> int | None:
    """Coerce a CSV cell to int, returning None when missing.

    Parameters
    ----------
    value
        Cell value to coerce.
    """
    if is_missing(value):
        return None
    return int(value)


def parse_bool(value: Any, default: bool = False) -> bool:
    """Parse a CSV cell as a boolean.

    Parameters
    ----------
    value
        Cell value to parse.
    default
        Fallback when *value* is missing or unrecognised.
    """
    if is_missing(value):
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def parse_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    """Parse a CSV cell as a pair of floats.

    Parameters
    ----------
    value
        Cell value to parse.
    default
        Fallback when *value* is missing or malformed.
    """
    if is_missing(value):
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    text = str(value).strip().strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    return default


def parse_fragment_positions(
    value: Any,
) -> tuple[tuple[float, float, float], ...] | None:
    """Parse a CSV cell as a sequence of 3-D Cartesian positions.

    Parameters
    ----------
    value
        Cell value to parse (sequence or JSON string).
    """
    if is_missing(value):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"fragment_positions must be a sequence or JSON list, got {type(value)!r}"
        )
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in value)

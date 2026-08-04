"""Shared CSV cell coercion helpers for result rows."""

from __future__ import annotations

import json
from typing import Any


def is_missing(value: Any) -> bool:
    return value is None or str(value) == "nan"


def with_default(value: Any, default: Any) -> Any:
    return default if is_missing(value) else value


def float_or(value: Any, default: float) -> float:
    return float(with_default(value, default))


def int_or_none(value: Any) -> int | None:
    if is_missing(value):
        return None
    return int(value)


def parse_bool(value: Any, default: bool = False) -> bool:
    if is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def parse_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
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
    if is_missing(value):
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"fragment_positions must be a sequence or JSON list, got {type(value)!r}"
        )
    return tuple((float(p[0]), float(p[1]), float(p[2])) for p in value)

"""Shared cache-key serialization helpers for placement modules."""

import struct


def _pack_optional_float(value: float | None) -> bytes:
    """Serialize an optional float to a stable, comparable byte string.

    ``None`` and ``NaN`` both pack to a distinguishable sentinel so that cache
    keys treat "unset" consistently regardless of how the caller represents it.
    """
    if value is None:
        return b"\x00" + struct.pack("<d", float("nan"))
    return b"\x01" + struct.pack("<d", float(value))

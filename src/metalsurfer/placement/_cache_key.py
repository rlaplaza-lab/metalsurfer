"""Shared cache-key serialization helpers for placement modules."""

from __future__ import annotations

import struct
import threading
from collections.abc import Hashable

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig


def _pack_optional_float(value: float | None) -> bytes:
    """Serialize an optional float to a stable, comparable byte string.

    ``None`` and ``NaN`` both pack to a distinguishable sentinel so that cache
    keys treat "unset" consistently regardless of how the caller represents it.
    """
    if value is None:
        return b"\x00" + struct.pack("<d", float("nan"))
    return b"\x01" + struct.pack("<d", float(value))


def pack_voronoi_site_config_bytes(
    config: AdsorptionConfig,
    material_type: str,
    *,
    equivalence_tolerance_leading: bool = False,
    append_min_adsorbate_separation: bool = False,
) -> bytes:
    """Pack Voronoi/site-enumeration knobs shared across placement cache keys."""
    equiv = struct.pack("<d", float(config.site_equivalence_tolerance))
    head = (
        _pack_optional_float(config.voronoi_probe_radius)
        + _pack_optional_float(config.voronoi_max_site_distance)
        + _pack_optional_float(config.top_layer_tolerance)
        + struct.pack("<d", float(config.planar_z_variance_threshold))
    )
    tail = (
        struct.pack("<?", bool(config.voronoi_site_enrichment))
        + struct.pack("<?", bool(config.voronoi_auto_widen))
        + str(config.site_classification_method).encode()
        + b"\x00"
        + str(config.site_generator).encode()
        + b"\x00"
        + material_type.encode()
    )
    cfg_bytes = (
        equiv + head + tail if equivalence_tolerance_leading else head + equiv + tail
    )
    if append_min_adsorbate_separation:
        cfg_bytes += struct.pack("<d", float(config.min_adsorbate_separation))
    return cfg_bytes


def pack_atoms_geometry_bytes(atoms: Atoms, *, include_pbc: bool = False) -> bytes:
    """Pack positions, cell, atomic numbers; optionally PBC (byte-stable order)."""
    parts = [
        atoms.get_positions().tobytes(),
        np.asarray(atoms.get_cell()).tobytes(),
    ]
    if include_pbc:
        parts.append(np.asarray(atoms.get_pbc(), dtype=np.uint8).tobytes())
    parts.append(np.asarray(atoms.get_atomic_numbers(), dtype=np.int32).tobytes())
    return b"".join(parts)


class BoundedFifoCache[T]:
    """Thread-safe insertion-ordered cache; drops the oldest entry on overflow."""

    def __init__(self, max_entries: int) -> None:
        if max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {max_entries}")
        self._max_entries = int(max_entries)
        self._store: dict[Hashable, T] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> T | None:
        with self._lock:
            return self._store.get(key)

    def store(self, key: Hashable, value: T) -> T:
        with self._lock:
            existing = self._store.get(key)
            if existing is not None:
                return existing
            if len(self._store) >= self._max_entries:
                self._store.pop(next(iter(self._store)))
            self._store[key] = value
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

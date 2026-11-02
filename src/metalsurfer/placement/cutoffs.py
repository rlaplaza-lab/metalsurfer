"""Tabulated covalent and van der Waals cutoffs.

Covalent radii answer covalent contact (bonds, adsorbate fusion, the
chemisorption crash floor, contact counting). Van der Waals radii answer
optional steric overlap. A pair is inside a cutoff when
``d <= scale * (r_i + r_j)`` and clear when ``d`` is strictly greater.

Holes in a table are filled by linear interpolation in atomic number within
the same period. The other table is never used as a substitute.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from ase.data import atomic_numbers, chemical_symbols
from ase.data import covalent_radii as ase_covalent_radii
from ase.data import vdw_radii as ase_vdw_radii

from .._logging import warn_once

logger = logging.getLogger(__name__)

RadiusKind = Literal["covalent", "vdw"]

# Inclusive upper Z of periods 1–7.
_PERIOD_HI: tuple[int, ...] = (2, 10, 18, 36, 54, 86, 118)

__all__ = [
    "clear_of_cutoff",
    "mixed_covalent_thresholds",
    "pair_thresholds",
    "radius_for_symbol",
    "substrate_floor_thresholds",
    "radii_or_cached",
    "tabulated_radii",
    "within_cutoff",
]


def _labeled_table(kind: RadiusKind) -> tuple[str, np.ndarray]:
    if kind == "covalent":
        return "covalent", ase_covalent_radii
    if kind == "vdw":
        return "van der Waals", ase_vdw_radii
    raise ValueError(f"radius kind must be 'covalent' or 'vdw', got {kind!r}")


def _period_span(z: int) -> tuple[int, int]:
    lo = 1
    for hi in _PERIOD_HI:
        if z <= hi:
            return lo, hi
        lo = hi + 1
    raise ValueError(f"atomic number {z} is outside periods 1–7")


def _tabulated(table: np.ndarray, z: int) -> float | None:
    if z <= 0 or z >= len(table):
        return None
    radius = float(table[z])
    if np.isfinite(radius) and radius > 0.0:
        return radius
    return None


def _symbol(z: int) -> str:
    if 0 <= z < len(chemical_symbols):
        return str(chemical_symbols[z])
    return f"Z{z}"


@functools.cache
def radius_for_symbol(symbol: str, kind: RadiusKind) -> float:
    """Return a positive radius (Å), interpolating a same-period hole."""
    label, table = _labeled_table(kind)
    z = atomic_numbers.get(symbol)
    if z is None or int(z) <= 0:
        raise ValueError(f"no positive {label} radius for symbol(s): {symbol}")
    z_int = int(z)
    direct = _tabulated(table, z_int)
    if direct is not None:
        return direct

    lo_z, hi_z = _period_span(z_int)
    left: tuple[int, float] | None = None
    for zz in range(z_int - 1, lo_z - 1, -1):
        radius = _tabulated(table, zz)
        if radius is not None:
            left = (zz, radius)
            break
    right: tuple[int, float] | None = None
    for zz in range(z_int + 1, hi_z + 1):
        radius = _tabulated(table, zz)
        if radius is not None:
            right = (zz, radius)
            break

    if left is not None and right is not None:
        z_lo, r_lo = left
        z_hi, r_hi = right
        value = r_lo + (r_hi - r_lo) * (z_int - z_lo) / (z_hi - z_lo)
        warn_once(
            logger,
            f"radius-fill:{kind}:{symbol}",
            f"{label} radius for {symbol} is {value:.3f} Å, interpolated "
            f"between {_symbol(z_lo)} ({r_lo:.3f} Å) and {_symbol(z_hi)} "
            f"({r_hi:.3f} Å)",
        )
        return float(value)
    anchor = left if left is not None else right
    if anchor is not None:
        anchor_z, anchor_r = anchor
        warn_once(
            logger,
            f"radius-fill:{kind}:{symbol}",
            f"{label} radius for {symbol} is {anchor_r:.3f} Å, copied from "
            f"{_symbol(anchor_z)} ({anchor_r:.3f} Å) "
            "(one-sided fill; no second in-period anchor)",
        )
        return float(anchor_r)
    raise ValueError(
        f"no in-period {label} radius to interpolate for {symbol} (Z={z_int})"
    )


def tabulated_radii(symbols: Sequence[str], *, kind: RadiusKind) -> np.ndarray:
    """Per-symbol radii (Å). Unknown symbols raise; table holes are filled."""
    seq = list(symbols)
    out = np.empty(len(seq), dtype=float)
    missing: list[str] = []
    for i, sym in enumerate(seq):
        z = atomic_numbers.get(str(sym))
        if z is None or int(z) <= 0:
            missing.append(str(sym))
            continue
        out[i] = radius_for_symbol(str(sym), kind)
    if missing:
        label, _table = _labeled_table(kind)
        uniq = ", ".join(sorted(set(missing)))
        raise ValueError(f"no positive {label} radius for symbol(s): {uniq}")
    return out


def radii_or_cached(
    symbols: Sequence[str],
    cached: np.ndarray | None,
    *,
    kind: RadiusKind,
) -> np.ndarray:
    """Return *cached* when it matches *symbols* and every entry is positive."""
    n = len(symbols)
    if cached is not None:
        arr = np.asarray(cached, dtype=float).reshape(-1)
        if (
            arr.shape[0] == n
            and n > 0
            and bool(np.all(np.isfinite(arr)))
            and bool(np.all(arr > 0.0))
        ):
            return arr
    return tabulated_radii(symbols, kind=kind)


def _positive_scale(scale: float, name: str) -> float:
    arr = np.asarray(scale, dtype=float)
    if arr.size != 1:
        raise ValueError(f"{name} must be a positive finite scalar, got {scale!r}")
    value = float(arr.reshape(-1)[0])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {scale!r}")
    return value


def pair_thresholds(
    radii_a: np.ndarray,
    radii_b: np.ndarray,
    scale: float,
) -> np.ndarray:
    """``scale * (r_a[:, None] + r_b[None, :])``."""
    factor = _positive_scale(scale, "scale")
    ra = np.asarray(radii_a, dtype=float).reshape(-1)
    rb = np.asarray(radii_b, dtype=float).reshape(-1)
    return factor * (ra[:, None] + rb[None, :])


def within_cutoff(dists: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return true where ``d <= threshold`` (inside, including equality)."""
    return np.asarray(dists, dtype=float) <= np.asarray(thresholds, dtype=float)


def clear_of_cutoff(dists: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Return true where ``d > threshold``."""
    return np.asarray(dists, dtype=float) > np.asarray(thresholds, dtype=float)


def substrate_floor_thresholds(
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    scale: float,
    absolute_floor: float,
) -> np.ndarray:
    """``max(absolute_floor, scale * (r_i + r_j))`` for mol–substrate pairs."""
    floor = float(absolute_floor)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError(
            f"absolute_floor must be non-negative and finite, got {absolute_floor!r}"
        )
    return np.maximum(floor, pair_thresholds(moving_radii, fixed_radii, scale))


def mixed_covalent_thresholds(
    moving_radii: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    n_substrate: int,
    substrate_scale: float,
    adsorbate_scale: float,
    absolute_floor: float,
) -> np.ndarray:
    """Substrate columns use the crash floor; later columns use the fuse cutoff."""
    fixed = np.asarray(fixed_radii, dtype=float).reshape(-1)
    n_fixed = int(fixed.shape[0])
    n_sub = int(n_substrate)
    if n_sub < 0 or n_sub > n_fixed:
        raise ValueError(f"n_substrate ({n_sub}) must be in [0, {n_fixed}] (n_fixed)")
    thresholds = pair_thresholds(moving_radii, fixed, adsorbate_scale)
    if n_sub:
        thresholds[:, :n_sub] = substrate_floor_thresholds(
            moving_radii,
            fixed[:n_sub],
            scale=substrate_scale,
            absolute_floor=absolute_floor,
        )
    return thresholds

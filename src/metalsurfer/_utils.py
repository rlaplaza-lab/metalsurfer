"""Internal utilities shared across metalsurfer sub-packages."""

from collections.abc import Sequence
from math import isfinite

import numpy as np

# Cells with |det| below this are treated as degenerate (no usable volume), so
# periodic distance conventions are disabled rather than producing garbage.
CELL_DET_EPS: float = 1e-12


def require_unique_molecule_names(
    names: Sequence[str], *, context: str = "molecules"
) -> None:
    """Raise ``ValueError`` if *names* contains a duplicate after stripping.

    Parameters
    ----------
    names
        Molecule labels to check.
    context
        Prefix for the error message (for example ``molecules``).
    """
    seen: dict[str, int] = {}
    for index, name in enumerate(names):
        key = name.strip()
        if key in seen:
            raise ValueError(
                f"{context} contains duplicate name {name.strip()!r} "
                f"(first at index {seen[key]}, again at index {index})"
            )
        seen[key] = index


def is_finite_number(value: object) -> bool:
    """Return True if *value* converts to a finite float.

    Accepted types are ``int``, ``float``, ``str``, and NumPy integer /
    floating scalars. ``bool`` and ``np.bool_`` are rejected (``bool`` is an
    ``int`` subclass). ``Decimal`` / ``Fraction`` are out of scope for the
    CSV/JSON callers and return False.

    Parameters
    ----------
    value
        Value to test for finite float conversion.
    """
    # bool is an int subclass; reject before the numeric isinstance check.
    if isinstance(value, (bool, np.bool_)):
        return False
    if not isinstance(value, (int, float, str, np.floating, np.integer)):
        return False
    try:
        return bool(isfinite(float(value)))
    except (TypeError, ValueError, OverflowError):
        return False


def cell_has_volume(cell, *, eps: float = CELL_DET_EPS) -> bool:
    """Return True when *cell* spans a non-degenerate volume.

    Uses ``abs(det)`` deliberately. A left-handed cell (negative determinant,
    e.g. from a loaded POSCAR with a flipped axis) is perfectly valid and
    periodic; testing ``det > 0`` silently classifies it as degenerate, which
    drops PBC from distance checks and site enumeration. Keep every
    "is this cell usable?" test routed through here so the convention and the
    tolerance stay in one place.

    Parameters
    ----------
    cell
        3x3 cell matrix.
    eps
        Absolute determinant tolerance.
    """
    arr = np.asarray(cell, dtype=float)
    if arr.shape != (3, 3):
        return False
    return abs(float(np.linalg.det(arr))) > eps


def union_find_cluster(
    n: int,
    merge_pairs: list[tuple[int, int]],
) -> list[list[int]]:
    """Cluster ``n`` elements by union-find with path compression and union-by-rank.

    Parameters
    ----------
    n
        Number of elements, indexed ``0 .. n-1``.
    merge_pairs
        Pairs of indices to merge into the same cluster.
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in merge_pairs:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

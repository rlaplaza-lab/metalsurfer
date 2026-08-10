"""CPU-only unit tests for internal pure-Python helpers.

These raise coverage on small helper modules and lock their behaviour so
refactors cannot silently change parsing/validation semantics.
"""

import numpy as np
import pytest

from metalsurfer._csv_coerce import (
    float_or,
    int_or_none,
    is_missing,
    parse_bool,
    parse_float_pair,
    parse_fragment_positions,
    with_default,
)
from metalsurfer._utils import is_finite_number


def test_is_finite_number_accepts_finite_numeric():
    assert is_finite_number(3) is True
    assert is_finite_number(3.0) is True
    assert is_finite_number("3.0") is True
    assert is_finite_number("-1.25e3") is True


def test_is_finite_number_rejects_non_numeric_and_non_finite():
    assert is_finite_number(None) is False
    assert is_finite_number([1, 2]) is False
    assert is_finite_number({"a": 1}) is False
    assert is_finite_number("abc") is False
    assert is_finite_number(float("inf")) is False
    assert is_finite_number(float("nan")) is False
    assert is_finite_number(object()) is False


def test_is_missing_and_with_default():
    assert is_missing(None) is True
    assert is_missing("nan") is True
    assert is_missing(0) is False
    assert is_missing("") is True
    assert with_default(None, 7) == 7
    assert with_default("nan", 7) == 7
    assert with_default(3, 7) == 3


def test_float_or_falls_back_to_default_on_missing():
    assert float_or(None, 1.5) == 1.5
    assert float_or("nan", 1.5) == 1.5
    assert float_or("2.5", 1.5) == 2.5


def test_int_or_none_parses_or_returns_none():
    assert int_or_none(None) is None
    assert int_or_none("nan") is None
    assert int_or_none("42") == 42
    assert int_or_none(42) == 42
    assert int_or_none(3.9) == 3


def test_parse_bool_covers_all_branches():
    assert parse_bool(None) is False
    assert parse_bool(None, default=True) is True
    assert parse_bool(True) is True
    assert parse_bool(False) is False
    assert parse_bool(1) is True
    assert parse_bool(0) is False
    assert parse_bool(2) is True
    assert parse_bool("TRUE") is True
    assert parse_bool("yes") is True
    assert parse_bool("t") is True
    assert parse_bool("1") is True
    assert parse_bool("false") is False
    assert parse_bool("no") is False
    assert parse_bool("n") is False
    assert parse_bool("f") is False
    assert parse_bool("junk") is False
    assert parse_bool("junk", default=True) is True


def test_parse_float_pair_covers_all_branches():
    assert parse_float_pair(None, (0.0, 0.0)) == (0.0, 0.0)
    assert parse_float_pair((1.0, 2.0), (0.0, 0.0)) == (1.0, 2.0)
    assert parse_float_pair([1.0, 2.0], (0.0, 0.0)) == (1.0, 2.0)
    assert parse_float_pair("1.0, 2.0", (0.0, 0.0)) == (1.0, 2.0)
    assert parse_float_pair("[3.0, 4.0]", (0.0, 0.0)) == (3.0, 4.0)
    assert parse_float_pair("garbage", (9.0, 9.0)) == (9.0, 9.0)
    assert parse_float_pair("1.0", (9.0, 9.0)) == (9.0, 9.0)


def test_parse_fragment_positions_covers_all_branches():
    assert parse_fragment_positions(None) is None
    assert parse_fragment_positions("nan") is None
    assert parse_fragment_positions("[[1.0, 2.0, 3.0]]") == ((1.0, 2.0, 3.0),)
    assert parse_fragment_positions([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )
    with pytest.raises(
        TypeError, match="fragment_positions must be a sequence or JSON list"
    ):
        parse_fragment_positions(42)


def test_cell_has_volume_accepts_left_handed_cells():
    """Regression: five call sites tested ``det > 0``.

    A left-handed cell (e.g. a loaded POSCAR with a flipped axis) has a negative
    determinant but is perfectly valid and periodic. Treating it as degenerate
    silently dropped PBC from distance checks and site enumeration.
    """
    from metalsurfer._utils import cell_has_volume

    right_handed = np.diag([4.0, 4.0, 12.0])
    left_handed = right_handed.copy()
    left_handed[2, 2] = -12.0

    assert float(np.linalg.det(left_handed)) < 0.0
    assert cell_has_volume(right_handed)
    assert cell_has_volume(left_handed)
    assert not cell_has_volume(np.zeros((3, 3)))
    assert not cell_has_volume(np.diag([4.0, 4.0, 0.0]))
    assert not cell_has_volume(np.zeros((2, 2)))


def test_left_handed_cell_keeps_periodic_distances():
    """A left-handed cell must still use the minimum-image convention."""
    from metalsurfer.placement.geometry import calculate_min_distance

    cell = np.diag([6.0, 6.0, -20.0])
    a = np.array([[0.5, 0.5, 0.0]])
    b = np.array([[5.5, 0.5, 0.0]])  # 1.0 A away across the x boundary

    d = calculate_min_distance(a, b, cell, use_pbc=True, pbc=[True, True, False])
    assert float(d) == pytest.approx(1.0, abs=1e-6)

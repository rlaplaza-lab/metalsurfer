"""Shared wall-contact / same-height identity for policy and fill.

Wall ``z_fraction`` collapse and same-COM dedupe must agree between
:mod:`placement.policy` (grid sampling) and :mod:`workflow.placement_fill`
(retry bans). Keep both behind this module.
"""

from __future__ import annotations

from collections.abc import Callable

from ..models import PlacementSpec
from ._constants import _WALL_Z_FRACTIONS, _Z_FRACTIONS
from .pose.height import _height_interval_family_key

__all__ = [
    "_dedupe_same_height_specs",
    "_height_interval_family_key",
    "_kept_z_count",
    "_kept_z_fractions",
    "_same_height_key",
    "_wall_contact_z_fraction",
    "_wall_z_fraction_is_contact",
]


def _wall_contact_z_fraction() -> float:
    """Wall ``z_fraction`` that is the covalent-contact sample."""
    return float(_WALL_Z_FRACTIONS[0])


def _wall_z_fraction_is_contact(z_fraction: float) -> bool:
    """Whether a wall fraction shares the covalent-contact center of mass.

    Fractions at or below :func:`_wall_contact_z_fraction` land on that height.
    The contact sample itself is the representative; anything lower repeats it.
    """
    return float(z_fraction) <= _wall_contact_z_fraction()


def _kept_z_fractions(
    site_type: str | None, *, is_void: bool = False
) -> tuple[float, ...]:
    """Distinct center-of-mass samples for one site.

    Void and pore sites keep the full height axis. Wall sites keep the contact
    shell and the shells above it.
    """
    if is_void or site_type == "pore":
        return _Z_FRACTIONS
    return _WALL_Z_FRACTIONS


def _kept_z_count(site_type: str | None, *, is_void: bool = False) -> int:
    """Return how many ``z_fraction`` samples one site keeps."""
    return len(_kept_z_fractions(site_type, is_void=is_void))


def _same_height_key(
    spec: PlacementSpec,
    *,
    is_void: bool | None = None,
) -> tuple:
    """Identity of specs that place at the same COM.

    On wall sites every fraction on the contact shell shares that height.
    Fractions above the contact sample, and every void or dissociative
    fraction, stay distinct.
    """
    family = _height_interval_family_key(spec)
    zf = float(spec.z_fraction)
    void = bool(is_void) if is_void is not None else False
    if is_void is None and spec.site_type == "pore":
        void = True
    if spec.orientation_type == "dissociative" or void:
        return (*family, zf)
    if _wall_z_fraction_is_contact(zf):
        return (*family, "contact")
    return (*family, zf)


def _dedupe_same_height_specs(
    specs: list[PlacementSpec],
    *,
    site_is_void: Callable[[int], bool] | None = None,
) -> list[PlacementSpec]:
    """Keep one spec per collapsed height, preferring the contact sample."""
    best: dict[tuple, int] = {}
    order: list[tuple] = []
    for i, spec in enumerate(specs):
        is_void = (
            None if site_is_void is None else bool(site_is_void(int(spec.site_index)))
        )
        key = _same_height_key(spec, is_void=is_void)
        prev = best.get(key)
        if prev is None:
            best[key] = i
            order.append(key)
            continue
        prev_spec = specs[prev]
        contact = _wall_contact_z_fraction()
        if abs(float(spec.z_fraction) - contact) < abs(
            float(prev_spec.z_fraction) - contact
        ):
            best[key] = i
    return [specs[best[key]] for key in order]

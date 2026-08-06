"""Typed adsorption site records and dict helpers."""


from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Site",
    "site_from_dict",
    "with_symmetry",
]

# Coordinate rounding for hash/eq so float noise does not splinter identity.
_SITE_COORD_EQ_DECIMALS: int = 6


@dataclass(frozen=True, eq=False)
class Site:
    """One adsorption site (Voronoi / topology / hollow)."""

    xyz: np.ndarray
    normal: np.ndarray
    site_type: str
    slab_indices: tuple[int, ...]
    material_type: str
    site_source: str
    env_fingerprint: tuple
    nn_distance: float | None = None
    hollow_order: int | None = None
    symmetry_multiplicity: int | None = None
    symmetry_equivalent_sites: tuple | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "xyz", np.asarray(self.xyz, dtype=float).reshape(3).copy()
        )
        object.__setattr__(
            self, "normal", np.asarray(self.normal, dtype=float).reshape(3).copy()
        )
        object.__setattr__(
            self, "slab_indices", tuple(int(i) for i in self.slab_indices)
        )
        if self.symmetry_equivalent_sites is not None:
            object.__setattr__(
                self,
                "symmetry_equivalent_sites",
                tuple(self.symmetry_equivalent_sites),
            )

    def _identity_key(self) -> tuple:
        return (
            tuple(np.round(self.xyz, decimals=_SITE_COORD_EQ_DECIMALS).tolist()),
            tuple(np.round(self.normal, decimals=_SITE_COORD_EQ_DECIMALS).tolist()),
            self.site_type,
            self.material_type,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Site):
            return NotImplemented
        return self._identity_key() == other._identity_key()

    def __hash__(self) -> int:
        return hash(self._identity_key())

    @property
    def xy(self) -> np.ndarray:
        """Cartesian xy of the site vertex (defensive copy)."""
        return self.xyz[:2].copy()

    @property
    def z(self) -> float:
        """Cartesian z of the site vertex."""
        return float(self.xyz[2])


def with_symmetry(
    site: Site,
    *,
    symmetry_multiplicity: int,
    symmetry_equivalent_sites: tuple,
) -> Site:
    """Return a copy of *site* with symmetry orbit metadata."""
    return replace(
        site,
        symmetry_multiplicity=int(symmetry_multiplicity),
        symmetry_equivalent_sites=tuple(symmetry_equivalent_sites),
    )


def _coerce_slab_indices(raw_indices: object) -> tuple[int, ...]:
    if raw_indices is None:
        return ()
    if isinstance(raw_indices, (str, bytes)):
        raise TypeError(
            f"slab_indices must be an iterable of ints, got {type(raw_indices).__name__}"
        )
    if isinstance(raw_indices, np.ndarray):
        return tuple(int(i) for i in raw_indices.ravel())
    if isinstance(raw_indices, Iterable):
        return tuple(int(i) for i in raw_indices)
    raise TypeError(
        f"slab_indices must be an iterable of ints, got {type(raw_indices).__name__}"
    )


def site_from_dict(data: Mapping[str, object]) -> Site:
    """Build a :class:`Site` from a mapping (tests / CSV loaders).

    Requires ``xyz`` (length-3). Optional keys mirror :class:`Site` fields.
    """
    if "xyz" not in data:
        raise KeyError("site_from_dict requires 'xyz'")
    xyz = np.asarray(data["xyz"], dtype=float).reshape(3)
    normal = np.asarray(data.get("normal", (0.0, 0.0, 1.0)), dtype=float).reshape(3)
    slab_indices = _coerce_slab_indices(data.get("slab_indices", ()))
    env = data.get("env_fingerprint", ())
    if not isinstance(env, tuple):
        env = tuple(env) if isinstance(env, list) else ()
    nn = data.get("nn_distance")
    ho = data.get("hollow_order")
    mult = data.get("symmetry_multiplicity")
    equiv = data.get("symmetry_equivalent_sites")
    equiv_tuple: tuple | None = None
    if equiv is not None:
        equiv_tuple = tuple(equiv) if isinstance(equiv, (list, tuple)) else None
    nn_distance = float(nn) if isinstance(nn, (int, float, str)) else None
    return Site(
        xyz=xyz,
        normal=normal,
        site_type=str(data.get("site_type", "unknown")),
        slab_indices=slab_indices,
        material_type=str(data.get("material_type", "slab")),
        site_source=str(data.get("site_source", "voronoi")),
        env_fingerprint=env,
        nn_distance=nn_distance,
        hollow_order=int(ho) if isinstance(ho, (int, float, str)) else None,
        symmetry_multiplicity=int(mult)
        if isinstance(mult, (int, float, str))
        else None,
        symmetry_equivalent_sites=equiv_tuple,
    )

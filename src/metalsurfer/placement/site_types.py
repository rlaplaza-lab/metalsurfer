"""Typed adsorption site records."""

from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Site",
    "with_symmetry",
]


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
        """Coerce array and sequence fields after initialization."""
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
    """Return a copy of *site* with symmetry orbit metadata.

    Parameters
    ----------
    site
        :class:`Site` to copy.
    symmetry_multiplicity
        Multiplicity of the symmetry orbit.
    symmetry_equivalent_sites
        Tuple of equivalent site positions.
    """
    return replace(
        site,
        symmetry_multiplicity=int(symmetry_multiplicity),
        symmetry_equivalent_sites=tuple(symmetry_equivalent_sites),
    )

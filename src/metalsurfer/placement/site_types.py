"""Typed adsorption site records."""

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from ._constants import _VECTOR_NORM_EPS

__all__ = [
    "Site",
    "site_kind_from_type_and_supports",
    "with_symmetry",
]


def site_kind_from_type_and_supports(
    site_type: str, slab_indices: tuple[int, ...]
) -> Literal["wall", "void"]:
    """Return ``void`` for pores / empty supports, else ``wall``."""
    if site_type == "pore" or not slab_indices:
        return "void"
    return "wall"


@dataclass(frozen=True, eq=False)
class Site:
    """One adsorption site from any generator plugin.

    ``xyz`` is the catalog identity: support-plane anchor for wall-near sites
    (plugin lift/snap is probe-only) or free-volume centre for pores.
    ``kind`` is ``wall`` (nonempty supports) or ``void`` (pore / empty
    supports); placement height physics switches on this field.
    ``env_fingerprint`` is ``(support_symbols, distance_bins, side_label)``.
    ``tangent_basis`` is always set by shared classify for pose/orientation.
    ``clearance`` / ``nn_distance`` retain accessibility probe metadata.
    """

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
    clearance: float | None = None
    tangent_basis: np.ndarray | None = None
    kind: Literal["wall", "void"] | None = None

    def __post_init__(self) -> None:
        """Coerce array and sequence fields after initialization."""
        object.__setattr__(
            self, "xyz", np.asarray(self.xyz, dtype=float).reshape(3).copy()
        )
        normal = np.asarray(self.normal, dtype=float).reshape(3).copy()
        nrm = float(np.linalg.norm(normal))
        if nrm > _VECTOR_NORM_EPS:
            normal /= nrm
        object.__setattr__(self, "normal", normal)
        object.__setattr__(
            self, "slab_indices", tuple(int(i) for i in self.slab_indices)
        )
        if self.symmetry_equivalent_sites is not None:
            object.__setattr__(
                self,
                "symmetry_equivalent_sites",
                tuple(self.symmetry_equivalent_sites),
            )
        if self.tangent_basis is not None:
            tb = np.asarray(self.tangent_basis, dtype=float).reshape(2, 3).copy()
            object.__setattr__(self, "tangent_basis", tb)
        if self.kind is None:
            object.__setattr__(
                self,
                "kind",
                site_kind_from_type_and_supports(self.site_type, self.slab_indices),
            )

    @property
    def xy(self) -> np.ndarray:
        """Cartesian xy of the catalog anchor (support plane or pore centre)."""
        return self.xyz[:2].copy()

    @property
    def z(self) -> float:
        """Cartesian z of the catalog anchor (support plane or pore centre)."""
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

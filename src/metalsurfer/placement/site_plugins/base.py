"""Site generator plugin contract, registry helpers, and name resolution.

Registered plugins: ``topology`` (slab Delaunay hybrid / NP hull),
``voronoi`` (free-volume vertices), and internal ``adaptive_grid``
(atom-centred Cartesian grid with iterative refinement; all materials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from .._material import validate_material_type

if TYPE_CHECKING:
    from ase import Atoms
    from scipy.spatial import Delaunay, KDTree

# All factory-resolvable plugin ids (including internal A/B plugins).
SITE_GENERATORS: tuple[str, ...] = ("topology", "voronoi", "adaptive_grid")
# Plugins selectable via AdsorptionConfig / YAML (excludes internal A/B).
PUBLIC_SITE_GENERATORS: tuple[str, ...] = ("topology", "voronoi")

_PLUGIN_ALLOWED_MATERIALS: dict[str, frozenset[str]] = {
    "topology": frozenset({"slab", "nanoparticle"}),
    "voronoi": frozenset({"slab", "porous"}),
    "adaptive_grid": frozenset({"slab", "nanoparticle", "porous"}),
}

_AUTO_DEFAULTS: dict[str, str] = {
    "slab": "topology",
    "nanoparticle": "topology",
    "porous": "voronoi",
}


@dataclass
class SiteGenerationContext:
    """Shared geometry / window inputs prepared by the enumerator."""

    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    symbols: list[str]
    material_type: str
    probe_radius: float
    max_site_distance: float
    top_layer_tolerance: float
    enrich: bool
    planar_z_variance_threshold: float
    adsorbate: Atoms | None = None


@dataclass
class SiteCandidateBatch:
    """Raw candidate sites from a plugin before shared post-processing."""

    vertices: np.ndarray
    nn_dists: np.ndarray
    source_hints: list[str]
    atom_indices: list[tuple[int, ...]]
    inject_atop: bool = False
    has_topology_atop: bool = False
    apply_slab_height_mask: bool = False
    slab_top_atom_indices: np.ndarray | None = None
    accessibility_tree: KDTree | None = None
    topology_median_nn: float | None = None
    topology_primary_delaunay: Delaunay | None = None
    topology_expanded_xy: np.ndarray | None = None
    topology_expanded_origin: list[int] | None = None
    topology_expanded_tri: Delaunay | None = None
    reuse: Any = None
    early_empty: bool = False


def empty_candidate_batch(*, early_empty: bool = False) -> SiteCandidateBatch:
    """Return an empty candidate batch."""
    return SiteCandidateBatch(
        vertices=np.empty((0, 3), dtype=float),
        nn_dists=np.empty(0, dtype=float),
        source_hints=[],
        atom_indices=[],
        early_empty=early_empty,
    )


class SiteGenerator(Protocol):
    """Plugin that enumerates raw adsorption-site candidates."""

    name: str

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Produce a candidate batch for *ctx*."""
        ...


def resolved_site_generator_name(name: str, material_type: str) -> str:
    """Resolve ``auto`` / explicit name to a registered plugin id (never ``auto``)."""
    validate_material_type(material_type)
    if name == "auto":
        return _AUTO_DEFAULTS[material_type]
    if name not in SITE_GENERATORS:
        raise ValueError(
            f"Unknown site_generator {name!r}; "
            f"expected one of {('auto',) + SITE_GENERATORS}"
        )
    allowed = _PLUGIN_ALLOWED_MATERIALS[name]
    if material_type not in allowed:
        raise ValueError(
            f"site_generator={name!r} is incompatible with "
            f"material_type={material_type!r}; allowed materials: "
            f"{sorted(allowed)}"
        )
    return name

"""Site generator plugins for adsorption-site enumeration.

``auto`` resolves by material (slab/NP → topology, porous → voronoi).
``adaptive_grid`` is selectable on all materials but not chosen by ``auto``.
"""

from .adaptive_grid import AdaptiveGridGenerator
from .base import (
    PLUGIN_ALLOWED_MATERIALS,
    PUBLIC_SITE_GENERATORS,
    SITE_GENERATORS,
    SiteCandidateBatch,
    SiteGenerationContext,
    SiteGenerator,
    empty_candidate_batch,
    resolved_site_generator_name,
    slice_candidate_arrays,
)
from .topology_np import TopologyNPGenerator
from .topology_slab import TopologySlabGenerator
from .voronoi import VoronoiGenerator

__all__ = [
    "PLUGIN_ALLOWED_MATERIALS",
    "PUBLIC_SITE_GENERATORS",
    "SITE_GENERATORS",
    "AdaptiveGridGenerator",
    "SiteCandidateBatch",
    "SiteGenerationContext",
    "SiteGenerator",
    "TopologyNPGenerator",
    "TopologySlabGenerator",
    "VoronoiGenerator",
    "empty_candidate_batch",
    "resolve_site_generator",
    "resolved_site_generator_name",
    "slice_candidate_arrays",
]

# Explicit (resolved_name, material_type) → generator class.
_SITE_GENERATOR_FACTORY: dict[tuple[str, str], type] = {
    ("topology", "slab"): TopologySlabGenerator,
    ("topology", "nanoparticle"): TopologyNPGenerator,
    ("voronoi", "slab"): VoronoiGenerator,
    ("voronoi", "porous"): VoronoiGenerator,
    ("adaptive_grid", "slab"): AdaptiveGridGenerator,
    ("adaptive_grid", "nanoparticle"): AdaptiveGridGenerator,
    ("adaptive_grid", "porous"): AdaptiveGridGenerator,
}


def resolve_site_generator(name: str, material_type: str) -> SiteGenerator:
    """Return the plugin for *name* and *material_type*."""
    resolved = resolved_site_generator_name(name, material_type)
    return _SITE_GENERATOR_FACTORY[(resolved, material_type)]()

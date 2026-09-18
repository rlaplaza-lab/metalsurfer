"""Site generator plugins for adsorption-site enumeration.

``auto`` resolves by material (slab/NP → topology, porous → voronoi).
"""

from .base import (
    SITE_GENERATORS,
    SiteCandidateBatch,
    SiteGenerationContext,
    SiteGenerator,
    empty_candidate_batch,
    resolved_site_generator_name,
)
from .topology_np import TopologyNPGenerator
from .topology_slab import TopologySlabGenerator
from .voronoi import VoronoiGenerator

__all__ = [
    "SITE_GENERATORS",
    "SiteCandidateBatch",
    "SiteGenerationContext",
    "SiteGenerator",
    "TopologyNPGenerator",
    "TopologySlabGenerator",
    "VoronoiGenerator",
    "empty_candidate_batch",
    "resolve_site_generator",
    "resolved_site_generator_name",
]


def resolve_site_generator(name: str, material_type: str) -> SiteGenerator:
    """Return the plugin for *name* and *material_type*."""
    resolved = resolved_site_generator_name(name, material_type)
    if resolved == "topology":
        if material_type == "slab":
            return TopologySlabGenerator()
        return TopologyNPGenerator()
    return VoronoiGenerator()

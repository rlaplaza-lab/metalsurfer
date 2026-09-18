"""Site generator plugins for adsorption-site enumeration.

``auto`` resolves by material (slab/NP → topology, porous → voronoi).
``adaptive_grid`` is selectable on all materials but not chosen by ``auto``.
"""

from .adaptive_grid import AdaptiveGridGenerator
from .base import (
    PUBLIC_SITE_GENERATORS,
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
]


def resolve_site_generator(name: str, material_type: str) -> SiteGenerator:
    """Return the plugin for *name* and *material_type*."""
    resolved = resolved_site_generator_name(name, material_type)
    if resolved == "topology":
        if material_type == "slab":
            return TopologySlabGenerator()
        return TopologyNPGenerator()
    if resolved == "adaptive_grid":
        return AdaptiveGridGenerator()
    return VoronoiGenerator()

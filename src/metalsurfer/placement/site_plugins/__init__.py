"""Site generator plugins for adsorption-site enumeration.

``auto`` resolves by material (slab/NP → topology, porous → voronoi).
``adaptive_grid`` and ``rolling_probe`` are selectable on all materials but
not chosen by ``auto``.
"""

from .adaptive_grid import AdaptiveGridGenerator
from .base import (
    PLUGIN_ALLOWED_MATERIALS,
    SITE_GENERATORS,
    SiteCandidateBatch,
    SiteGenerationContext,
    SiteGenerator,
    empty_candidate_batch,
    resolved_site_generator_name,
    slice_candidate_arrays,
)
from .rolling_probe import RollingProbeGenerator
from .topology import TopologyGenerator
from .topology_np import TopologyNPGenerator
from .topology_slab import TopologySlabGenerator
from .voronoi import VoronoiGenerator

__all__ = [
    "PLUGIN_ALLOWED_MATERIALS",
    "SITE_GENERATORS",
    "AdaptiveGridGenerator",
    "RollingProbeGenerator",
    "SiteCandidateBatch",
    "SiteGenerationContext",
    "SiteGenerator",
    "TopologyGenerator",
    "TopologyNPGenerator",
    "TopologySlabGenerator",
    "VoronoiGenerator",
    "empty_candidate_batch",
    "resolve_site_generator",
    "resolved_site_generator_name",
    "slice_candidate_arrays",
]

# One entry per plugin id; material allowlists live in site_plugin_ids.
_SITE_GENERATOR_FACTORY: dict[str, type] = {
    "topology": TopologyGenerator,
    "voronoi": VoronoiGenerator,
    "adaptive_grid": AdaptiveGridGenerator,
    "rolling_probe": RollingProbeGenerator,
}


def resolve_site_generator(name: str, material_type: str) -> SiteGenerator:
    """Return the plugin for *name* and *material_type*."""
    resolved = resolved_site_generator_name(name, material_type)
    return _SITE_GENERATOR_FACTORY[resolved]()

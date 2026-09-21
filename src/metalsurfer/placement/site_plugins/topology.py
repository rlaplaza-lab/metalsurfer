"""Topology site plugin: dispatches slab vs nanoparticle implementations."""

from __future__ import annotations

from typing import Any

from .base import SiteCandidateBatch, SiteGenerationContext
from .topology_np import TopologyNPGenerator
from .topology_slab import TopologySlabGenerator


class TopologyGenerator:
    """Material-dispatching topology plugin (one factory entry for all materials)."""

    name = "topology"
    widens_distance_window = True
    uses_structure_pbc = False

    def __init__(self) -> None:
        self._slab = TopologySlabGenerator()
        self._np = TopologyNPGenerator()

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate topology candidates for *ctx*."""
        if ctx.material_type == "nanoparticle":
            return self._np.generate(ctx, reuse=reuse)
        return self._slab.generate(ctx, reuse=reuse)

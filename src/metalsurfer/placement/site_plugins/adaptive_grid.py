"""Adaptive Cartesian-grid site plugin (all material types)."""

from __future__ import annotations

from typing import Any

from .._constants import _ADAPTIVE_GRID_DEFAULT_SPACING
from ..site_adaptive_grid import _SOURCE_HINT, generate_adaptive_grid_sites
from .base import SiteCandidateBatch, SiteGenerationContext
from .wall_near import wall_near_batch_to_plugin


class AdaptiveGridGenerator:
    """Atom-centred adaptive grid with optional iterative shell refinement.

    One PBC/clearance path for every material (slab, nanoparticle, porous).
    Not selected by ``auto``. Near-atom shells (not pore centres); density set
    by ``adaptive_grid_spacing`` / ``merge_radius`` and optional
    ``adaptive_grid_refine_levels``. Final catalog: one site per support key
    snapped to a lateral pocket anchor (midpoint / circumcenter / centroid)
    at a target clearance, then NMS — balanced multi-atom pockets are scored
    so hollow-like sites win basins without densifying the catalog.

    Does **not** apply atop injection or slab height masking (those stay on
    topology / Voronoi slab). Emits the same :class:`SiteCandidateBatch` contract as topology
    / Voronoi; classify builds fingerprints and frames.
    """

    name = "adaptive_grid"
    widens_distance_window = False
    uses_structure_pbc = True

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate adaptive-grid candidates for *ctx*."""
        _ = reuse
        positions, cell, pbc = ctx.positions, ctx.cell, ctx.pbc
        spacing = (
            float(ctx.adaptive_grid_spacing)
            if ctx.adaptive_grid_spacing is not None
            else float(_ADAPTIVE_GRID_DEFAULT_SPACING)
        )
        levels = int(ctx.adaptive_grid_refine_levels)

        result = generate_adaptive_grid_sites(
            positions,
            cell,
            pbc,
            probe_radius=float(ctx.probe_radius),
            max_site_distance=float(ctx.max_site_distance),
            initial_spacing=spacing,
            max_levels=levels,
            n_jobs=int(ctx.n_jobs),
            symbols=list(ctx.symbols),
            side_policy=ctx.side_policy,
            nms_framework_scale=ctx.adaptive_grid_nms_framework_scale,
        )
        return wall_near_batch_to_plugin(
            result,
            positions=positions,
            cell=cell,
            pbc=pbc,
            source_hint=_SOURCE_HINT,
        )

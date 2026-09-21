"""Rolling-probe wall-near site plugin (all material types)."""

from __future__ import annotations

from typing import Any

from ..site_rolling_probe import _SOURCE_HINT, generate_rolling_probe_sites
from .base import SiteCandidateBatch, SiteGenerationContext, empty_candidate_batch
from .helpers import median_nn_or_fallback


class RollingProbeGenerator:
    """Connolly / SAS rolling-probe contacts (atop / bridge / hollow).

    Not selected by ``auto``. Wall-near on every accessible face (slab top by
    default, NP exterior, MOF **pore walls**); not pore centres. Geometric
    1/2/3 supports are retained so typing and catalog density stay comparable
    to ``adaptive_grid``. Emits the same :class:`SiteCandidateBatch` contract.
    """

    name = "rolling_probe"

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate rolling-probe candidates for *ctx*."""
        del reuse
        positions, cell, pbc = ctx.positions, ctx.cell, ctx.pbc

        result = generate_rolling_probe_sites(
            positions,
            cell,
            pbc,
            material_type=ctx.material_type,
            probe_radius=float(ctx.probe_radius),
            max_site_distance=float(ctx.max_site_distance),
            n_jobs=int(ctx.n_jobs),
            symbols=list(ctx.symbols),
            side_policy=ctx.side_policy,
        )

        if len(result.vertices) == 0:
            return empty_candidate_batch()

        return SiteCandidateBatch(
            vertices=result.vertices,
            nn_dists=result.nn_dists,
            source_hints=[_SOURCE_HINT] * len(result.vertices),
            atom_indices=list(result.atom_indices),
            accessibility_tree=result.accessibility_tree,
            topology_median_nn=float(
                median_nn_or_fallback(
                    result.nn_dists, reference_positions=positions, cell=cell, pbc=pbc
                )
            ),
            normals=result.normals,
            clearances=result.clearances,
        )

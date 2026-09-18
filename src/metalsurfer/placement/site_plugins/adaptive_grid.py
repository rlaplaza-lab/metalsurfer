"""Adaptive Cartesian-grid site plugin (all material types)."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from ..site_adaptive_grid import (
    _SOURCE_HINT,
    adaptive_grid_characteristic_length,
    generate_adaptive_grid_sites,
)
from ..site_coords import _height_along_slab_normal, top_layer_mask_by_normal
from .base import SiteCandidateBatch, SiteGenerationContext, empty_candidate_batch
from .helpers import median_nn_or_fallback

_EMPTY: tuple[int, ...] = ()


class AdaptiveGridGenerator:
    """Atom-centred adaptive grid with iterative near-atom shell refinement.

    Not selected by ``auto``. Select via ``AdsorptionConfig.site_generator`` or
    ``get_unified_sites(..., site_generator="adaptive_grid")``.
    """

    name = "adaptive_grid"

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate adaptive-grid candidates for *ctx*."""
        del reuse
        positions, cell, pbc = ctx.positions, ctx.cell, ctx.pbc
        material_type = ctx.material_type
        adsorbate = ctx.adsorbate if isinstance(ctx.adsorbate, Atoms) else None
        scale = ctx.grid_spacing_scale
        if scale is None and adsorbate is not None:
            scale = adaptive_grid_characteristic_length(
                float(ctx.probe_radius), adsorbate
            )

        vertices, nn_dists, _, accessibility_tree = generate_adaptive_grid_sites(
            positions,
            cell,
            pbc,
            material_type=material_type,
            probe_radius=float(ctx.probe_radius),
            max_site_distance=float(ctx.max_site_distance),
            grid_spacing_scale=scale,
            n_jobs=int(ctx.n_jobs),
        )

        if len(vertices) == 0:
            return empty_candidate_batch()

        # Drop interlayer / subsurface candidates. Use the top-atom plane (not
        # site-nn margins): adaptive nn distances are shell radii ~2 Å and would
        # otherwise keep sites between Cu layers.
        slab_top = None
        if material_type == "slab":
            tol = float(ctx.top_layer_tolerance)
            slab_top = np.nonzero(top_layer_mask_by_normal(positions, cell, tol))[0]
            h_surface = float(np.max(_height_along_slab_normal(positions, cell)))
            keep = _height_along_slab_normal(vertices, cell) >= h_surface - 0.25
            vertices = vertices[keep]
            nn_dists = nn_dists[keep]
            if len(vertices) == 0:
                return empty_candidate_batch()

        return SiteCandidateBatch(
            vertices=vertices,
            nn_dists=nn_dists,
            source_hints=[_SOURCE_HINT] * len(vertices),
            atom_indices=[_EMPTY] * len(vertices),
            inject_atop=False,
            apply_slab_height_mask=False,
            slab_top_atom_indices=slab_top,
            accessibility_tree=accessibility_tree,
            topology_median_nn=float(
                median_nn_or_fallback(
                    nn_dists, reference_positions=positions, cell=cell, pbc=pbc
                )
            ),
        )

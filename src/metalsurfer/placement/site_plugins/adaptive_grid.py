"""Adaptive Cartesian-grid site plugin (internal A/B, all material types)."""

from __future__ import annotations

from typing import Any

import numpy as np
from ase import Atoms

from ..site_adaptive_grid import _SOURCE_HINT, generate_adaptive_grid_sites
from ..site_coords import top_layer_mask_by_normal
from .base import SiteCandidateBatch, SiteGenerationContext, empty_candidate_batch
from .helpers import median_nn_or_fallback

_EMPTY: tuple[int, ...] = ()


class AdaptiveGridGenerator:
    """Atom-centred adaptive grid with iterative near-atom shell refinement.

    Not selected by ``auto`` and not on ``AdsorptionConfig``; use
    ``get_unified_sites(..., site_generator="adaptive_grid")``.
    """

    name = "adaptive_grid"

    def __init__(
        self,
        *,
        initial_spacing: float | None = None,
        max_levels: int | None = None,
    ) -> None:
        self._initial_spacing = initial_spacing
        self._max_levels = max_levels

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

        vertices, nn_dists, _, accessibility_tree = generate_adaptive_grid_sites(
            positions,
            cell,
            pbc,
            material_type=material_type,
            probe_radius=float(ctx.probe_radius),
            max_site_distance=float(ctx.max_site_distance),
            top_layer_tolerance=float(ctx.top_layer_tolerance),
            adsorbate=adsorbate,
            initial_spacing=self._initial_spacing,
            max_levels=self._max_levels,
        )

        if len(vertices) == 0 and material_type == "porous":
            return empty_candidate_batch(early_empty=True)

        slab_top = None
        apply_height = False
        inject_atop = material_type in ("slab", "nanoparticle")
        if material_type == "slab":
            slab_top = np.nonzero(
                top_layer_mask_by_normal(
                    positions, cell, float(ctx.top_layer_tolerance)
                )
            )[0]
            apply_height = True
            ref = positions[slab_top] if len(slab_top) else positions
        else:
            ref = positions

        return SiteCandidateBatch(
            vertices=vertices,
            nn_dists=nn_dists,
            source_hints=[_SOURCE_HINT] * len(vertices),
            atom_indices=[_EMPTY] * len(vertices),
            inject_atop=inject_atop,
            apply_slab_height_mask=apply_height,
            slab_top_atom_indices=slab_top,
            accessibility_tree=accessibility_tree,
            topology_median_nn=float(
                median_nn_or_fallback(
                    nn_dists, reference_positions=ref, cell=cell, pbc=pbc
                )
            ),
        )

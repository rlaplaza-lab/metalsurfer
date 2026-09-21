"""Voronoi free-volume plugin for porous frameworks and explicit slab A/B."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy.spatial import KDTree

from ..site_coords import (
    project_vertices_to_support_plane,
    top_layer_mask_by_normal,
)
from ..site_voronoi import _voronoi_sites
from .base import SiteCandidateBatch, SiteGenerationContext, empty_candidate_batch
from .helpers import (
    apply_slab_height_mask,
    candidate_enrichment_frames,
    inject_atop_sites,
    periodic_accessibility_tree,
)

logger = logging.getLogger(__name__)


class VoronoiGenerator:
    """Voronoi vertices with optional ridge enrichment.

    On slabs this skips topology (explicit ``site_generator="voronoi"`` A/B
    path). Planar cells may then rely on the shared atop-injection safety net.
    """

    name = "voronoi"
    widens_distance_window = True
    uses_structure_pbc = False

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate Voronoi free-volume candidates for *ctx*."""
        del reuse  # rebuild on auto-widen (extension margin depends on max)
        positions = ctx.positions
        cell = ctx.cell
        pbc = ctx.pbc
        material_type = ctx.material_type
        probe_radius = ctx.probe_radius
        max_site_distance = ctx.max_site_distance

        voronoi_positions = positions
        slab_top_atom_indices: np.ndarray | None = None
        accessibility_tree = None
        do_slab_post = False

        if material_type == "slab":
            slab_top_mask = top_layer_mask_by_normal(
                positions, cell, float(ctx.top_layer_tolerance)
            )
            slab_top_atom_indices = np.nonzero(slab_top_mask)[0]
            top_only = positions[slab_top_mask]
            if len(top_only) >= 4:
                voronoi_positions = top_only
            accessibility_tree = periodic_accessibility_tree(
                positions,
                cell,
                pbc,
                float(max_site_distance),
            )
            do_slab_post = True

        vertices, nn_dists, local_atoms = _voronoi_sites(
            voronoi_positions,
            cell,
            pbc,
            probe_radius=probe_radius,
            max_distance=max_site_distance,
            enrich=ctx.enrich,
            symbols=ctx.symbols,
            n_jobs=int(ctx.n_jobs),
        )
        source_hints = ["voronoi"] * len(vertices)
        if len(voronoi_positions) != len(positions):
            assert slab_top_atom_indices is not None
            remap = np.asarray(slab_top_atom_indices, dtype=int)
            atom_indices = [
                tuple(int(remap[j]) for j in atoms) for atoms in local_atoms
            ]
        else:
            atom_indices = list(local_atoms)

        if len(vertices) == 0 and material_type == "porous":
            logger.warning(
                "No accessible sites for %d-atom structure "
                "(probe_radius=%.2f, max_distance=%.2f, material_type=%r)",
                len(positions),
                float(probe_radius),
                float(max_site_distance),
                material_type,
            )
            return empty_candidate_batch(early_empty=True)

        if accessibility_tree is None:
            accessibility_tree = periodic_accessibility_tree(
                positions,
                cell,
                pbc,
                float(max_site_distance),
            )
        normals: np.ndarray | None
        clearances: np.ndarray | None
        normals, clearances = candidate_enrichment_frames(
            vertices,
            nn_dists,
            atom_indices,
            positions=positions,
            cell=cell,
            pbc=pbc,
            material_type=material_type,
            accessibility_tree=accessibility_tree,
        )
        vertices = project_vertices_to_support_plane(
            vertices, normals, atom_indices, positions
        )

        if do_slab_post:
            (
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                normals,
                clearances,
            ) = apply_slab_height_mask(
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                positions=positions,
                cell=cell,
                top_layer_tolerance=float(ctx.top_layer_tolerance),
                normals=normals,
                clearances=clearances,
            )
            local_tree = KDTree(positions)
            (
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                normals,
                clearances,
            ) = inject_atop_sites(
                vertices,
                nn_dists,
                source_hints,
                positions=positions,
                cell=cell,
                pbc=pbc,
                material_type="slab",
                local_tree=local_tree,
                accessibility_tree=accessibility_tree,
                median_nn=None,
                slab_top_atom_indices=slab_top_atom_indices,
                has_topology_atop=False,
                probe_radius=float(probe_radius),
                max_site_distance=float(max_site_distance),
                atom_indices=atom_indices,
                normals=normals,
                clearances=clearances,
            )

        return SiteCandidateBatch(
            vertices=vertices,
            nn_dists=nn_dists,
            source_hints=source_hints,
            atom_indices=atom_indices,
            has_topology_atop=False,
            slab_top_atom_indices=slab_top_atom_indices,
            accessibility_tree=accessibility_tree,
            normals=normals,
            clearances=clearances,
        )

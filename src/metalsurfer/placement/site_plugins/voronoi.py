"""Voronoi free-volume plugin for porous frameworks and explicit slab A/B."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._constants import _ATOP_INJECTION_HEIGHT_FACTOR
from ..site_coords import project_vertices_to_support_plane
from ..site_voronoi import _generate_slab_topology_sites, _voronoi_sites
from .base import SiteCandidateBatch, SiteGenerationContext, empty_candidate_batch
from .helpers import (
    candidate_enrichment_frames,
    inject_atop_sites,
    median_nn_or_fallback,
    periodic_accessibility_tree,
    slab_candidate_postprocess,
    slab_top_voronoi_inputs,
)

logger = logging.getLogger(__name__)


class VoronoiGenerator:
    """Voronoi vertices with optional ridge enrichment.

    On slabs this skips the hybrid topology+Voronoi path (explicit
    ``site_generator="voronoi"`` A/B). When the top layer is coplanar and 3D
    Voronoi yields no vertices, the top-layer atop/bridge/hollow mesh is used
    so the surface is still enumerated.
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
        _ = reuse  # no Qhull reuse
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
        topology_primary_delaunay = None
        topology_expanded_xy = None
        topology_expanded_origin = None
        topology_expanded_tri = None
        has_topology_atop = False
        topology_median_nn: float | None = None

        if material_type == "slab":
            _top_mask, slab_top_atom_indices, voronoi_positions = (
                slab_top_voronoi_inputs(positions, cell, float(ctx.top_layer_tolerance))
            )
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

        if (
            len(vertices) == 0
            and material_type == "slab"
            and slab_top_atom_indices is not None
            and accessibility_tree is not None
        ):
            # Coplanar top layer: 3D Voronoi is empty. Emit the Delaunay motif.
            logger.info(
                "Slab Voronoi produced no vertices; using top-layer "
                "atop/bridge/hollow topology mesh"
            )
            topology_median_nn = float(
                median_nn_or_fallback(
                    np.empty(0, dtype=float),
                    reference_positions=positions[slab_top_atom_indices],
                    cell=cell,
                    pbc=pbc,
                )
            )
            site_height = _ATOP_INJECTION_HEIGHT_FACTOR * topology_median_nn
            (
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                topology_primary_delaunay,
                topology_expanded_xy,
                topology_expanded_origin,
                topology_expanded_tri,
            ) = _generate_slab_topology_sites(
                positions,
                cell,
                pbc,
                slab_top_atom_indices,
                accessibility_tree,
                site_height,
                float(probe_radius),
                float(max_site_distance),
            )
            has_topology_atop = any(s == "topology_atop" for s in source_hints)

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
            vertices,
            normals,
            atom_indices,
            positions,
            cell=cell,
            pbc=pbc,
        )

        if do_slab_post:
            (
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                normals,
                clearances,
            ) = slab_candidate_postprocess(
                vertices,
                nn_dists,
                source_hints,
                atom_indices,
                positions=positions,
                cell=cell,
                pbc=pbc,
                top_layer_tolerance=float(ctx.top_layer_tolerance),
                accessibility_tree=accessibility_tree,
                probe_radius=float(probe_radius),
                max_site_distance=float(max_site_distance),
                slab_top_atom_indices=slab_top_atom_indices,
                has_topology_atop=has_topology_atop,
                median_nn=topology_median_nn,
                normals=normals,
                clearances=clearances,
            )
        elif material_type == "porous":
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
                material_type="porous",
                has_topology_atop=False,
                probe_radius=float(probe_radius),
                max_site_distance=float(max_site_distance),
                accessibility_tree=accessibility_tree,
                atom_indices=atom_indices,
                normals=normals,
                clearances=clearances,
            )

        return SiteCandidateBatch(
            vertices=vertices,
            nn_dists=nn_dists,
            source_hints=source_hints,
            atom_indices=atom_indices,
            has_topology_atop=has_topology_atop,
            slab_top_atom_indices=slab_top_atom_indices,
            accessibility_tree=accessibility_tree,
            topology_median_nn=topology_median_nn,
            topology_primary_delaunay=topology_primary_delaunay,
            topology_expanded_xy=topology_expanded_xy,
            topology_expanded_origin=topology_expanded_origin,
            topology_expanded_tri=topology_expanded_tri,
            normals=normals,
            clearances=clearances,
        )

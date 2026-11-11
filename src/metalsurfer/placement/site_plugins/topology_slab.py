"""Slab topology plugin: Delaunay atop/bridge/hollow with optional Voronoi enrich."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .._constants import _ATOP_INJECTION_HEIGHT_FACTOR
from ..site_coords import exposed_lower_band_indices
from ..site_voronoi import _generate_slab_topology_sites, _voronoi_sites
from .base import SiteCandidateBatch, SiteGenerationContext
from .helpers import (
    PlanarWidenScratch,
    candidate_enrichment_frames,
    median_nn_or_fallback,
    merge_dedup_site_arrays,
    periodic_accessibility_tree,
    slab_candidate_postprocess,
    slab_top_voronoi_inputs,
    top_layer_is_planar_from_arrays,
)

logger = logging.getLogger(__name__)


class TopologySlabGenerator:
    """Hybrid slab generator (topology + Voronoi when the top layer is rough)."""

    name = "topology"
    widens_distance_window = True
    uses_structure_pbc = False

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate slab topology (+ optional Voronoi) candidates for *ctx*."""
        positions = ctx.positions
        cell = ctx.cell
        pbc = ctx.pbc
        symbols = ctx.symbols
        probe_radius = ctx.probe_radius
        max_site_distance = ctx.max_site_distance
        top_layer_tolerance = ctx.top_layer_tolerance
        z_var_threshold = ctx.planar_z_variance_threshold
        enrich = ctx.enrich
        reuse_topology = reuse if isinstance(reuse, PlanarWidenScratch) else None

        slab_top_mask, slab_top_atom_indices, voronoi_positions = (
            slab_top_voronoi_inputs(positions, cell, float(top_layer_tolerance))
        )

        slab_skip_voronoi = top_layer_is_planar_from_arrays(
            positions,
            cell,
            float(top_layer_tolerance),
            z_var_threshold,
            top_mask=slab_top_mask,
        )
        if slab_skip_voronoi:
            logger.info(
                "Slab top layer is planar; skipping Voronoi vertex generation. "
                "Slab sites come from the topology generator "
                "(atop/bridge/hollow). probe_radius/max_site_distance still "
                "gate accessibility; site enrichment does not apply."
            )
            vertices = np.empty((0, 3), dtype=float)
            nn_dists = np.empty(0, dtype=float)
            source_hints: list[str] = []
            atom_indices: list[tuple[int, ...]] = []
        else:
            vertices, nn_dists, local_atoms = _voronoi_sites(
                voronoi_positions,
                cell,
                pbc,
                probe_radius=probe_radius,
                max_distance=max_site_distance,
                enrich=enrich,
                symbols=symbols,
                n_jobs=int(ctx.n_jobs),
            )
            source_hints = ["voronoi"] * len(vertices)
            if len(voronoi_positions) != len(positions):
                remap = np.asarray(slab_top_atom_indices, dtype=int)
                atom_indices = [
                    tuple(int(remap[j]) for j in atoms) for atoms in local_atoms
                ]
            else:
                atom_indices = list(local_atoms)

        accessibility_tree = periodic_accessibility_tree(
            positions,
            cell,
            pbc,
            float(max_site_distance),
        )
        median_nn = median_nn_or_fallback(
            nn_dists,
            reference_positions=positions[slab_top_atom_indices],
            cell=cell,
            pbc=pbc,
        )
        topology_median_nn = float(median_nn)
        site_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
        (
            topo_vertices,
            topo_dists,
            topo_sources,
            topo_atoms,
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
            primary_delaunay=(
                reuse_topology.primary_delaunay if reuse_topology else None
            ),
            exp2d=reuse_topology.exp_xy if reuse_topology else None,
            expanded_origin_local_index=(
                reuse_topology.exp_origin if reuse_topology else None
            ),
            exp_tri=reuse_topology.exp_tri if reuse_topology else None,
            reuse_delaunay=reuse_topology is not None,
        )
        scratch = PlanarWidenScratch(
            planar_skip_voronoi=bool(slab_skip_voronoi),
            primary_delaunay=topology_primary_delaunay,
            exp_xy=topology_expanded_xy,
            exp_origin=topology_expanded_origin,
            exp_tri=topology_expanded_tri,
        )
        has_topology_atop = any(s == "topology_atop" for s in topo_sources)
        if len(topo_vertices) > 0:
            vertices, nn_dists, source_hints, atom_indices, _, _ = (
                merge_dedup_site_arrays(
                    vertices,
                    nn_dists,
                    source_hints,
                    topo_vertices,
                    topo_dists,
                    topo_sources,
                    cell=cell,
                    pbc=pbc,
                    atom_indices=atom_indices,
                    new_atom_indices=topo_atoms,
                )
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
            material_type="slab",
            accessibility_tree=accessibility_tree,
        )
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
            top_layer_tolerance=float(top_layer_tolerance),
            accessibility_tree=accessibility_tree,
            probe_radius=float(probe_radius),
            max_site_distance=float(max_site_distance),
            slab_top_atom_indices=slab_top_atom_indices,
            has_topology_atop=has_topology_atop,
            median_nn=topology_median_nn,
            normals=normals,
            clearances=clearances,
        )

        # One metal–metal spacing, not the probe-to-atom distances stored in
        # topology_median_nn. A monatomic step is about one interlayer down.
        metal_nn = median_nn_or_fallback(
            np.empty(0, dtype=float),
            reference_positions=positions[slab_top_atom_indices],
            cell=cell,
            pbc=pbc,
        )
        extra_idx = exposed_lower_band_indices(
            positions,
            slab_top_atom_indices,
            list(symbols),
            cell,
            pbc,
            float(metal_nn),
        )
        if len(extra_idx) > 0:
            (
                extra_vertices,
                extra_dists,
                extra_sources,
                extra_atoms,
                _,
                _,
                _,
                _,
            ) = _generate_slab_topology_sites(
                positions,
                cell,
                pbc,
                extra_idx,
                accessibility_tree,
                site_height,
                float(probe_radius),
                float(max_site_distance),
            )
            if len(extra_vertices) > 0:
                extra_normals, extra_clearances = candidate_enrichment_frames(
                    extra_vertices,
                    extra_dists,
                    extra_atoms,
                    positions=positions,
                    cell=cell,
                    pbc=pbc,
                    material_type="slab",
                    accessibility_tree=accessibility_tree,
                )
                (
                    vertices,
                    nn_dists,
                    source_hints,
                    atom_indices,
                    normals,
                    clearances,
                ) = merge_dedup_site_arrays(
                    vertices,
                    nn_dists,
                    source_hints,
                    extra_vertices,
                    extra_dists,
                    extra_sources,
                    cell=cell,
                    pbc=pbc,
                    atom_indices=atom_indices,
                    new_atom_indices=extra_atoms,
                    normals=normals,
                    clearances=clearances,
                    new_normals=extra_normals,
                    new_clearances=extra_clearances,
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
            reuse=scratch,
            normals=normals,
            clearances=clearances,
        )

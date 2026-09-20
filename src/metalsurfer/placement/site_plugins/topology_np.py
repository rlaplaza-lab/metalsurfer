"""Nanoparticle topology plugin: convex-hull skin + NN-graph sites."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import KDTree

from .._constants import _ATOP_INJECTION_HEIGHT_FACTOR
from ..site_np import _generate_nanoparticle_topology_sites
from .base import SiteCandidateBatch, SiteGenerationContext
from .helpers import candidate_enrichment_frames, median_nn_or_fallback


class TopologyNPGenerator:
    """Hull + nearest-neighbour topology for finite metal clusters."""

    name = "topology"

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Enumerate hull topology candidates for *ctx*."""
        del reuse  # NP topology has no auto-widen Qhull reuse.
        positions = ctx.positions
        cell = ctx.cell
        pbc = ctx.pbc
        local_tree = KDTree(positions)
        metal_nn = median_nn_or_fallback(
            np.empty(0, dtype=float),
            reference_positions=positions,
            cell=cell,
            pbc=pbc,
        )
        topology_median_nn = float(metal_nn)
        site_height = _ATOP_INJECTION_HEIGHT_FACTOR * metal_nn
        (
            topo_vertices,
            topo_dists,
            topo_sources,
            topo_atoms,
        ) = _generate_nanoparticle_topology_sites(
            positions,
            local_tree,
            site_height,
            float(ctx.probe_radius),
            float(ctx.max_site_distance),
            metal_nn=metal_nn,
            cell=cell,
            pbc=pbc,
        )
        has_topology_atop = any(s == "topology_atop" for s in topo_sources)
        atom_indices = list(topo_atoms)
        normals, clearances = candidate_enrichment_frames(
            topo_vertices,
            topo_dists,
            atom_indices,
            positions=positions,
            cell=cell,
            pbc=pbc,
            material_type="nanoparticle",
            accessibility_tree=local_tree,
        )
        return SiteCandidateBatch(
            vertices=topo_vertices,
            nn_dists=topo_dists,
            source_hints=list(topo_sources),
            atom_indices=atom_indices,
            inject_atop=True,
            has_topology_atop=has_topology_atop,
            topology_median_nn=topology_median_nn,
            normals=normals,
            clearances=clearances,
        )

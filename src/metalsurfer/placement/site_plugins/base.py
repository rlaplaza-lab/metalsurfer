"""Site generator plugin contract, registry helpers, and name resolution.

Registered plugins: ``topology`` (slab Delaunay hybrid / NP hull),
``voronoi`` (free-volume vertices), and ``adaptive_grid``
(atom-centred Cartesian grid with iterative refinement; all materials).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np

from ...site_plugin_ids import (
    AUTO_SITE_GENERATOR_DEFAULTS,
    PLUGIN_ALLOWED_MATERIALS,
    PUBLIC_SITE_GENERATORS,
    SITE_GENERATORS,
)
from .._material import validate_material_type

if TYPE_CHECKING:
    from ase import Atoms
    from scipy.spatial import Delaunay, KDTree

# Re-export registry constants for plugin callers / tests.
__all__ = [
    "PLUGIN_ALLOWED_MATERIALS",
    "PUBLIC_SITE_GENERATORS",
    "SITE_GENERATORS",
    "SiteCandidateBatch",
    "SiteGenerationContext",
    "SiteGenerator",
    "empty_candidate_batch",
    "resolved_site_generator_name",
    "slice_candidate_arrays",
]


@dataclass
class SiteCandidateBatch:
    """Raw candidate sites from a plugin before shared post-processing.

    Core: ``vertices``, ``nn_dists``, ``source_hints``, ``atom_indices``.
    Optional enrichment: ``normals``, ``clearances``. Fingerprints and
    tangent frames are built in shared classify.
    """

    vertices: np.ndarray
    nn_dists: np.ndarray
    source_hints: list[str]
    atom_indices: list[tuple[int, ...]]
    inject_atop: bool = False
    has_topology_atop: bool = False
    apply_slab_height_mask: bool = False
    slab_top_atom_indices: np.ndarray | None = None
    accessibility_tree: KDTree | None = None
    topology_median_nn: float | None = None
    topology_primary_delaunay: Delaunay | None = None
    topology_expanded_xy: np.ndarray | None = None
    topology_expanded_origin: list[int] | None = None
    topology_expanded_tri: Delaunay | None = None
    reuse: Any = None
    early_empty: bool = False
    normals: np.ndarray | None = None
    clearances: np.ndarray | None = None


@dataclass
class SiteGenerationContext:
    """Shared geometry / window inputs prepared by the enumerator."""

    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    symbols: list[str]
    material_type: str
    probe_radius: float
    max_site_distance: float
    top_layer_tolerance: float
    enrich: bool
    planar_z_variance_threshold: float
    adsorbate: Atoms | None = None
    grid_spacing_scale: float | None = None
    adaptive_grid_spacing: float | None = None
    adaptive_grid_refine_levels: int = 0
    adaptive_grid_nms_framework_scale: float | None = None
    n_jobs: int = -2
    side_policy: Literal["all", "positive", "negative", "external"] = "positive"


def empty_candidate_batch(*, early_empty: bool = False) -> SiteCandidateBatch:
    """Return an empty candidate batch."""
    return SiteCandidateBatch(
        vertices=np.empty((0, 3), dtype=float),
        nn_dists=np.empty(0, dtype=float),
        source_hints=[],
        atom_indices=[],
        early_empty=early_empty,
    )


class SiteGenerator(Protocol):
    """Plugin that enumerates raw adsorption-site candidates."""

    name: str

    def generate(
        self,
        ctx: SiteGenerationContext,
        *,
        reuse: Any = None,
    ) -> SiteCandidateBatch:
        """Produce a candidate batch for *ctx*."""
        ...


def resolved_site_generator_name(name: str, material_type: str) -> str:
    """Resolve ``auto`` / explicit name to a registered plugin id (never ``auto``)."""
    validate_material_type(material_type)
    if name == "auto":
        return AUTO_SITE_GENERATOR_DEFAULTS[material_type]
    if name not in SITE_GENERATORS:
        raise ValueError(
            f"Unknown site_generator {name!r}; "
            f"expected one of {('auto',) + SITE_GENERATORS}"
        )
    allowed = PLUGIN_ALLOWED_MATERIALS[name]
    if material_type not in allowed:
        raise ValueError(
            f"site_generator={name!r} is incompatible with "
            f"material_type={material_type!r}; allowed materials: "
            f"{sorted(allowed)}"
        )
    return name


def slice_candidate_arrays(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    atom_indices: list[tuple[int, ...]],
    keep_mask: np.ndarray,
    *,
    normals: np.ndarray | None = None,
    clearances: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    list[tuple[int, ...]],
    np.ndarray | None,
    np.ndarray | None,
]:
    """Slice core batch arrays and optional enrichment by *keep_mask*."""
    mask = np.asarray(keep_mask, dtype=bool)
    kept = np.nonzero(mask)[0]
    out_normals = None if normals is None else np.asarray(normals, dtype=float)[mask]
    out_clearances = (
        None if clearances is None else np.asarray(clearances, dtype=float)[mask]
    )
    return (
        np.asarray(vertices, dtype=float)[mask],
        np.asarray(nn_dists, dtype=float)[mask],
        [source_hints[i] for i in kept],
        [atom_indices[i] for i in kept],
        out_normals,
        out_clearances,
    )

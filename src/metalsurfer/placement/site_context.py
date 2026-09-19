"""Cached site context for placement sampling."""

import hashlib
import logging
import struct
import threading
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..symmetry import SymmetryAnalysisError
from ._cache_key import _pack_optional_float
from ._material import material_aware_pbc, validate_material_type
from .site_enumeration import (
    _cluster_equivalent_sites,
    _get_unified_sites_with_plugin_vertices,
    get_symmetry_aware_sites,
)
from .site_plugins import resolved_site_generator_name
from .site_types import Site

logger = logging.getLogger(__name__)


# Sampling catalog used once adsorbates occupy the surface (full clustered lattice).
_CLUSTERED_UNDER_COVERAGE_SOURCE = "clustered_under_coverage"


@dataclass
class SiteContext:
    """Cached site catalog for a substrate geometry.

    ``sites`` is the catalog used for molecular placement sampling (clustered,
    then optionally symmetry-reduced on a clean substrate). Under coverage
    ``site_context_for_sampling`` expands ``sites`` to ``clustered_sites``.
    ``clustered_sites`` is always the fingerprint-aware geometric clustering
    result (full translational lattice). ``raw_unclustered`` is the
    pre-clustering ``get_unified_sites`` output. ``plugin_vertices`` is the
    raw plugin candidate coordinates before shared post-processing.
    """

    sites: list[Site]
    use_sites: bool
    source: str
    raw_unclustered: list[Site] | None = None
    clustered_sites: list[Site] | None = None
    plugin_vertices: np.ndarray | None = None


# Bounded FIFO cache for unique-sites (pre-symmetry) and resolved site contexts.
# Unique-sites + resolved context use 2 slots per geometry; 64 covers ~32 slabs
# (e.g. miller/facet sweeps) without thrashing.
_SITE_CONTEXT_CACHE_MAX_ENTRIES = 64
_SITE_CONTEXT_CACHE: dict[str, SiteContext] = {}
_SITE_CONTEXT_CACHE_LOCK = threading.Lock()


def _no_sites_context(
    *,
    raw_unclustered: list[Site] | None = None,
    clustered_sites: list[Site] | None = None,
    plugin_vertices: np.ndarray | None = None,
) -> SiteContext:
    return SiteContext(
        sites=[],
        use_sites=False,
        source="no_sites",
        raw_unclustered=raw_unclustered,
        clustered_sites=clustered_sites,
        plugin_vertices=plugin_vertices,
    )


def _unique_sites_cache_key(
    slab: Atoms,
    config: AdsorptionConfig,
) -> str:
    """Geometry + chemistry + Voronoi config key (pre-symmetry).

    PBC is keyed on :func:`material_aware_pbc` (what enumeration actually uses),
    not ``slab.get_pbc()``, so calculator-boundary PBC (e.g. ``[T,T,T]``) and
    material PBC (e.g. ``[T,T,F]`` for slabs) share one cache entry.

    For ``adaptive_grid``, spacing knobs (``adaptive_grid_spacing``,
    ``adaptive_grid_refine_levels``, ``adaptive_grid_nms_framework_scale``)
    are part of the key.
    """
    pos_bytes = slab.get_positions().tobytes()
    cell_bytes = np.asarray(slab.get_cell()).tobytes()
    pbc_bytes = np.asarray(
        material_aware_pbc(config.material_type), dtype=np.uint8
    ).tobytes()
    numbers_bytes = np.asarray(slab.get_atomic_numbers(), dtype=np.int32).tobytes()
    cfg_bytes = (
        _pack_optional_float(config.voronoi_probe_radius)
        + _pack_optional_float(config.voronoi_max_site_distance)
        + _pack_optional_float(config.top_layer_tolerance)
        + struct.pack("<d", float(config.planar_z_variance_threshold))
        + struct.pack("<d", float(config.site_equivalence_tolerance))
        + struct.pack("<?", bool(config.voronoi_site_enrichment))
        + struct.pack("<?", bool(config.voronoi_auto_widen))
        + str(config.site_classification_method).encode()
        + b"\x00"
        + str(config.site_generator).encode()
        + b"\x00"
        + str(config.side_policy).encode()
        + b"\x00"
        + config.material_type.encode()
    )
    scale_bytes = b""
    if str(config.site_generator) == "adaptive_grid":
        scale_bytes = (
            b"\x00ags\x00"
            + struct.pack("<d", float(config.adaptive_grid_spacing))
            + struct.pack("<i", int(config.adaptive_grid_refine_levels))
            + struct.pack("<d", float(config.adaptive_grid_nms_framework_scale))
        )
    return hashlib.sha256(
        pos_bytes + cell_bytes + pbc_bytes + numbers_bytes + cfg_bytes + scale_bytes
    ).hexdigest()


def _site_context_cache_key(
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    symmetry_broken: bool,
) -> str:
    base = _unique_sites_cache_key(slab, config)
    return hashlib.sha256(
        (
            base
            + f"|sym={int(bool(symmetry_broken))}"
            + f"|symtol={float(config.symmetry_tolerance)!r}"
        ).encode()
    ).hexdigest()


def _store_site_context_cache(cache_key: str, ctx: SiteContext) -> SiteContext:
    with _SITE_CONTEXT_CACHE_LOCK:
        if cache_key in _SITE_CONTEXT_CACHE:
            return _SITE_CONTEXT_CACHE[cache_key]
        if len(_SITE_CONTEXT_CACHE) >= _SITE_CONTEXT_CACHE_MAX_ENTRIES:
            _SITE_CONTEXT_CACHE.pop(next(iter(_SITE_CONTEXT_CACHE)))
        _SITE_CONTEXT_CACHE[cache_key] = ctx
    return ctx


def resolve_site_context_for_sampling(
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    symmetry_broken: bool,
) -> SiteContext:
    """Return clustered sites, then optional spglib orbit reduction.

    Symmetry reduction runs on the **clustered** catalog so geometric uniqueness
    and orbit reduction compose. Dissociative / adatom paths should use
    ``clustered_sites`` (full lattice), not ``sites`` after symmetry reduction.

    Skipped when *symmetry_broken*. For ``adaptive_grid``, catalog density
    follows ``config.adaptive_grid_spacing``.

    Parameters
    ----------
    slab_atoms
        :class:`~ase.Atoms` substrate.
    config
        :class:`~metalsurfer.config.AdsorptionConfig` with placement settings.
    symmetry_broken
        If True, skip symmetry reduction.
    """
    cache_key = _site_context_cache_key(
        slab_atoms,
        config,
        symmetry_broken=symmetry_broken,
    )

    with _SITE_CONTEXT_CACHE_LOCK:
        cached = _SITE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Reuses unique-sites entry in the same cache (key without |sym=).
    _core_ctx = _get_unique_sites_for_specs(slab_atoms, config)
    core_sites = _core_ctx.sites
    use_sites = _core_ctx.use_sites
    raw_unclustered = _core_ctx.raw_unclustered
    clustered_sites = _core_ctx.clustered_sites
    plugin_vertices = _core_ctx.plugin_vertices

    if not use_sites or not core_sites:
        result = _core_ctx
    elif symmetry_broken:
        logger.debug("Site context: symmetry broken, using clustered site set")
        result = SiteContext(
            sites=core_sites,
            use_sites=True,
            source=_core_ctx.source,
            raw_unclustered=raw_unclustered,
            clustered_sites=clustered_sites,
            plugin_vertices=plugin_vertices,
        )
    else:
        try:
            symmetry_aware_sites = get_symmetry_aware_sites(
                slab_atoms,
                top_layer_tolerance=config.top_layer_tolerance,
                symmetry_tolerance=config.symmetry_tolerance,
                material_type=config.material_type,
                probe_radius=config.voronoi_probe_radius,
                max_site_distance=config.voronoi_max_site_distance,
                enrich=config.voronoi_site_enrichment,
                site_classification_method=config.site_classification_method,
                raw_sites=core_sites,
                planar_z_variance_threshold=config.planar_z_variance_threshold,
                site_generator=config.site_generator,
            )
        except SymmetryAnalysisError as exc:
            logger.warning(
                "Symmetry site reduction failed; using clustered sites (%s)",
                exc,
            )
            symmetry_aware_sites = []

        if symmetry_aware_sites:
            logger.info(
                "Using symmetry-reduced sites (%d sites)", len(symmetry_aware_sites)
            )
            result = SiteContext(
                sites=symmetry_aware_sites,
                use_sites=True,
                source="symmetry_aware",
                raw_unclustered=raw_unclustered,
                clustered_sites=clustered_sites,
                plugin_vertices=plugin_vertices,
            )
        else:
            logger.debug("Using clustered sites (no symmetry-reduced set)")
            result = SiteContext(
                sites=core_sites,
                use_sites=True,
                source=_core_ctx.source,
                raw_unclustered=raw_unclustered,
                clustered_sites=clustered_sites,
                plugin_vertices=plugin_vertices,
            )

    return _store_site_context_cache(cache_key, result)


def skip_symmetry_for_sampling(
    *,
    symmetry_broken: bool,
    slab_for_sites: Atoms,
    full_slab: Atoms | None,
    config: AdsorptionConfig | None = None,
) -> bool:
    """Whether molecular sampling should skip orbit reduction.

    True when the substrate is already C1, when *full_slab* has an adsorbate
    suffix, or when *config.saturation_molecules_per_step* > 1 (n-tuplet
    co-adsorption needs distinct translational copies). Occupied vertices are
    dropped later by occupancy pruning.
    """
    if config is not None and int(config.saturation_molecules_per_step) > 1:
        return True
    if symmetry_broken:
        return True
    return full_slab is not None and len(full_slab) > len(slab_for_sites)


def site_context_for_occupied_surface(ctx: SiteContext) -> SiteContext:
    """Expand a symmetry-reduced sampling catalog to the clustered lattice."""
    clustered = ctx.clustered_sites
    if ctx.source != "symmetry_aware" or not ctx.use_sites or not clustered:
        return ctx
    logger.debug(
        "Sampling %d clustered sites (symmetry-reduced catalog had %d)",
        len(clustered),
        len(ctx.sites),
    )
    return SiteContext(
        sites=list(clustered),
        use_sites=True,
        source=_CLUSTERED_UNDER_COVERAGE_SOURCE,
        raw_unclustered=ctx.raw_unclustered,
        clustered_sites=clustered,
        plugin_vertices=ctx.plugin_vertices,
    )


def site_context_for_sampling(
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None = None,
    *,
    symmetry_broken: bool = False,
    full_slab: Atoms | None = None,
) -> SiteContext:
    """Return *site_context* or resolve the symmetry-aware sampling catalog.

    Used by generators / pose when callers omit an explicit context so the
    ``site_index`` catalog matches production screening. Expands a
    symmetry-reduced catalog to ``clustered_sites`` under coverage or n-tuplet.
    """
    skip = skip_symmetry_for_sampling(
        symmetry_broken=symmetry_broken,
        slab_for_sites=slab,
        full_slab=full_slab,
        config=config,
    )
    if site_context is not None:
        ctx = site_context
    else:
        ctx = resolve_site_context_for_sampling(
            slab,
            config,
            symmetry_broken=skip,
        )
    if skip:
        ctx = site_context_for_occupied_surface(ctx)
    return ctx


def _get_unique_sites_for_specs(
    slab: Atoms,
    config: AdsorptionConfig,
) -> SiteContext:
    """Get unique non-identical sites using unified site detection.

    Works for slabs, nanoparticles, and porous materials.
    Returns ``SiteContext(sites=[], use_sites=False, source="no_sites")`` when
    site detection yields nothing.

    Cached under the geometry key (no ``|sym=`` suffix) in the shared
    :data:`_SITE_CONTEXT_CACHE`. Both ``sites`` and ``clustered_sites`` are the
    geometric clustering result (no spglib).
    """
    cache_key = _unique_sites_cache_key(slab, config)
    with _SITE_CONTEXT_CACHE_LOCK:
        cached = _SITE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    validate_material_type(config.material_type)

    mat_type = config.material_type
    probe_radius = config.voronoi_probe_radius
    max_site_dist = config.voronoi_max_site_distance

    if len(slab) < 4:
        logger.warning(
            "Slab has fewer than 4 atoms (%d); cannot detect adsorption sites",
            len(slab),
        )
        return _store_site_context_cache(cache_key, _no_sites_context())

    raw_sites, plugin_vertices = _get_unified_sites_with_plugin_vertices(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_dist,
        top_layer_tolerance=config.top_layer_tolerance,
        material_type=mat_type,
        enrich=config.voronoi_site_enrichment,
        site_classification_method=config.site_classification_method,
        auto_widen=config.voronoi_auto_widen,
        planar_z_variance_threshold=config.planar_z_variance_threshold,
        site_generator=config.site_generator,
        adaptive_grid_spacing=float(config.adaptive_grid_spacing),
        adaptive_grid_refine_levels=int(config.adaptive_grid_refine_levels),
        adaptive_grid_nms_framework_scale=float(
            config.adaptive_grid_nms_framework_scale
        ),
        n_jobs=int(config.n_jobs),
        side_policy=config.side_policy,
    )
    if not raw_sites:
        logger.warning(
            "Unified site detection found no sites for %d-atom structure "
            "(probe_radius=%s, max_distance=%s, material_type=%r)",
            len(slab),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_dist:.2f}" if max_site_dist is not None else "auto",
            mat_type,
        )
        return _store_site_context_cache(
            cache_key,
            _no_sites_context(plugin_vertices=plugin_vertices),
        )

    cell = np.array(slab.get_cell())
    unique_sites = _cluster_equivalent_sites(
        raw_sites,
        cell,
        tolerance=config.site_equivalence_tolerance,
    )
    if not unique_sites:
        logger.warning(
            "Site clustering eliminated all %d raw sites for %d-atom structure "
            "(tolerance=%.3f, material_type=%r)",
            len(raw_sites),
            len(slab),
            config.site_equivalence_tolerance,
            mat_type,
        )
        return _store_site_context_cache(
            cache_key,
            _no_sites_context(
                raw_unclustered=raw_sites,
                plugin_vertices=plugin_vertices,
            ),
        )

    plugin_source = resolved_site_generator_name(
        config.site_generator, config.material_type
    )
    return _store_site_context_cache(
        cache_key,
        SiteContext(
            sites=unique_sites,
            use_sites=True,
            source=plugin_source,
            raw_unclustered=raw_sites,
            clustered_sites=unique_sites,
            plugin_vertices=plugin_vertices,
        ),
    )

"""Cached Voronoi site context for placement sampling."""

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
    get_symmetry_aware_sites,
    get_unified_sites,
)
from .site_types import Site

logger = logging.getLogger(__name__)


@dataclass
class SiteContext:
    """Cached result of Voronoi site detection for a given slab geometry."""

    sites: list[Site]
    use_sites: bool
    source: str
    # Pre-clustering output of :func:`get_unified_sites` (same as used for clustering).
    raw_unclustered: list[Site] | None = None


# Bounded FIFO cache for unique-sites (pre-symmetry) and resolved site contexts.
# Unique-sites + resolved context use 2 slots per geometry; 64 covers ~32 slabs
# (e.g. miller/facet sweeps) without thrashing.
_SITE_CONTEXT_CACHE_MAX_ENTRIES = 64
_SITE_CONTEXT_CACHE: dict[str, SiteContext] = {}
_SITE_CONTEXT_CACHE_LOCK = threading.Lock()


def _no_sites_context(
    *,
    raw_unclustered: list[Site] | None = None,
) -> SiteContext:
    return SiteContext(
        sites=[],
        use_sites=False,
        source="no_sites",
        raw_unclustered=raw_unclustered,
    )


def _unique_sites_cache_key(slab: Atoms, config: AdsorptionConfig) -> str:
    """Geometry + chemistry + Voronoi config key (pre-symmetry).

    PBC is keyed on :func:`material_aware_pbc` (what enumeration actually uses),
    not ``slab.get_pbc()``, so calculator-boundary PBC (e.g. ``[T,T,T]``) and
    material PBC (e.g. ``[T,T,F]`` for slabs) share one cache entry.
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
        + struct.pack("<d", float(config.site_equivalence_tolerance))
        + struct.pack("<?", bool(config.voronoi_site_enrichment))
        + struct.pack("<?", bool(config.voronoi_auto_widen))
        + str(config.site_classification_method).encode()
        + b"\x00"
        + config.material_type.encode()
    )
    return hashlib.sha256(
        pos_bytes + cell_bytes + pbc_bytes + numbers_bytes + cfg_bytes
    ).hexdigest()


def _site_context_cache_key(
    slab: Atoms, config: AdsorptionConfig, *, symmetry_broken: bool
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
    """Return clustered Voronoi sites, then optional spglib orbit reduction unless *symmetry_broken*.

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
        slab_atoms, config, symmetry_broken=symmetry_broken
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

    if not use_sites or not core_sites:
        result = _core_ctx
    elif symmetry_broken:
        logger.debug("Site context: symmetry broken, using clustered Voronoi set")
        result = SiteContext(
            sites=core_sites,
            use_sites=True,
            source="voronoi",
            raw_unclustered=raw_unclustered,
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
                raw_sites=raw_unclustered,
            )
        except SymmetryAnalysisError as exc:
            logger.info(
                "Symmetry site reduction failed; using clustered Voronoi sites (%s)",
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
            )
        else:
            logger.debug("Using clustered Voronoi sites (no symmetry-reduced set)")
            result = SiteContext(
                sites=core_sites,
                use_sites=True,
                source="voronoi",
                raw_unclustered=raw_unclustered,
            )

    return _store_site_context_cache(cache_key, result)


def _get_unique_sites_for_specs(
    slab: Atoms,
    config: AdsorptionConfig,
) -> SiteContext:
    """Get unique non-identical sites using unified Voronoi detection.

    Works for slabs, nanoparticles, and porous materials.
    Returns ``SiteContext(sites=[], use_sites=False, source="no_sites")`` when
    site detection yields nothing.

    Cached under the geometry key (no ``|sym=`` suffix) in the shared
    :data:`_SITE_CONTEXT_CACHE`.
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

    raw_sites = get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_dist,
        top_layer_tolerance=config.top_layer_tolerance,
        material_type=mat_type,
        enrich=config.voronoi_site_enrichment,
        site_classification_method=config.site_classification_method,
        auto_widen=config.voronoi_auto_widen,
    )
    if not raw_sites:
        logger.warning(
            "Unified Voronoi site detection found no sites for %d-atom structure "
            "(probe_radius=%s, max_distance=%s, material_type=%r)",
            len(slab),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_dist:.2f}" if max_site_dist is not None else "auto",
            mat_type,
        )
        return _store_site_context_cache(cache_key, _no_sites_context())

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
            _no_sites_context(raw_unclustered=raw_sites),
        )

    source = str(unique_sites[0].site_source)
    return _store_site_context_cache(
        cache_key,
        SiteContext(
            sites=unique_sites,
            use_sites=True,
            source=source,
            raw_unclustered=raw_sites,
        ),
    )

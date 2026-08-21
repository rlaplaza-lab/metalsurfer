"""Dissociative diatomic placement and hollow site pair enumeration."""

import hashlib
import struct
import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.spatial import KDTree

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementPose, PlacementSpec
from . import geometry as geom
from ._cache_key import _pack_optional_float
from ._constants import (
    _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE,
    _VECTOR_NORM_EPS,
)
from ._material import material_aware_pbc
from .occupancy import existing_adsorbate_positions, filter_sites_by_occupancy
from .orientation import _site_type_z_offset
from .pose import (
    _descriptor_from_placement,
    _resolve_surface_ref,
    _validate_posed_adsorbate,
)
from .site_context import SiteContext
from .site_coords import (
    _deduplicate_points,
    _periodic_image_offsets,
    _slab_normal,
    _slab_plane_projectors,
    top_layer_mask_by_normal,
)
from .site_enumeration import (
    _compute_site_z_base,
    get_hollow_sites_for_adatoms,
    get_unified_sites,
)
from .site_types import Site


@dataclass(frozen=True)
class _DissociativeSitePair:
    xyz1: np.ndarray
    normal1: np.ndarray
    xyz2: np.ndarray
    normal2: np.ndarray


_DISSOCIATIVE_PAIR_CACHE_MAX_ENTRIES = 16
_DISSOCIATIVE_PAIR_CACHE: dict[str, list[_DissociativeSitePair]] = {}
_DISSOCIATIVE_PAIR_CACHE_LOCK = threading.Lock()


def clear_dissociative_pair_caches() -> None:
    """Clear the process-local dissociative pair catalog cache."""
    with _DISSOCIATIVE_PAIR_CACHE_LOCK:
        _DISSOCIATIVE_PAIR_CACHE.clear()


def _is_dissociable_diatomic(adsorbate: Atoms) -> bool:
    """Check whether the molecule is a homonuclear diatomic (e.g. H2, O2, N2)."""
    syms = adsorbate.get_chemical_symbols()
    return len(syms) == 2 and syms[0] == syms[1]


def _site_outward_normal(
    site_xyz: np.ndarray,
    site: Site,
    *,
    material_type: str,
    reference_positions: np.ndarray,
    slab_normal: np.ndarray,
) -> np.ndarray:
    """Return unit outward normal for dissociative fragment placement."""
    normal = np.asarray(site.normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm > _VECTOR_NORM_EPS:
        return normal / norm
    if material_type == "nanoparticle":
        com = np.mean(reference_positions, axis=0)
        outward = np.asarray(site_xyz, dtype=float) - com
        norm = float(np.linalg.norm(outward))
        if norm > _VECTOR_NORM_EPS:
            return outward / norm
    return np.asarray(slab_normal, dtype=float)


def _dissociative_pair_cache_key(
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
) -> str:
    pos_bytes = slab_for_sites.get_positions().tobytes()
    cell_bytes = np.asarray(slab_for_sites.get_cell()).tobytes()
    numbers_bytes = np.asarray(
        slab_for_sites.get_atomic_numbers(), dtype=np.int32
    ).tobytes()
    cfg_bytes = (
        struct.pack("<d", float(config.hollow_site_dedup_tolerance))
        + _pack_optional_float(config.voronoi_probe_radius)
        + _pack_optional_float(config.voronoi_max_site_distance)
        + _pack_optional_float(config.top_layer_tolerance)
        + struct.pack("<?", bool(config.voronoi_site_enrichment))
        + str(config.site_classification_method).encode()
        + b"\x00"
        + config.material_type.encode()
    )
    return hashlib.sha256(
        pos_bytes + cell_bytes + numbers_bytes + cfg_bytes
    ).hexdigest()


def _get_dissociative_site_pairs(
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    existing_adsorbate_positions: np.ndarray | None = None,
    *,
    raw_sites: list[Site] | None = None,
    site_context: SiteContext | None = None,
) -> list[_DissociativeSitePair]:
    """Return outward-oriented site pairs for dissociative diatomic placement.

    Clean-slab catalogs are cached in a process-local store keyed by geometry and
    dissociative-relevant config (not on :class:`SiteContext`).
    """
    occupancy_active = (
        existing_adsorbate_positions is not None
        and len(existing_adsorbate_positions) > 0
    )
    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    cache_key: str | None = None
    # Skip cache when site_context/raw_sites alter the catalog for the same geometry.
    if not occupancy_active and raw_sites is None and site_context is None:
        cache_key = _dissociative_pair_cache_key(sites_slab, config)
        with _DISSOCIATIVE_PAIR_CACHE_LOCK:
            cached = _DISSOCIATIVE_PAIR_CACHE.get(cache_key)
        if cached is not None:
            return list(cached)

    pairs = _compute_dissociative_site_pairs(
        slab,
        config,
        slab_for_sites=slab_for_sites,
        existing_adsorbate_positions=existing_adsorbate_positions,
        raw_sites=raw_sites,
        site_context=site_context,
    )
    if cache_key is not None:
        with _DISSOCIATIVE_PAIR_CACHE_LOCK:
            if len(_DISSOCIATIVE_PAIR_CACHE) >= _DISSOCIATIVE_PAIR_CACHE_MAX_ENTRIES:
                _DISSOCIATIVE_PAIR_CACHE.pop(next(iter(_DISSOCIATIVE_PAIR_CACHE)))
            _DISSOCIATIVE_PAIR_CACHE[cache_key] = list(pairs)
    return pairs


def _compute_dissociative_site_pairs(
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    existing_adsorbate_positions: np.ndarray | None = None,
    *,
    raw_sites: list[Site] | None = None,
    site_context: SiteContext | None = None,
) -> list[_DissociativeSitePair]:
    """Return outward-oriented site pairs for dissociative diatomic placement."""
    if config.material_type not in ("slab", "nanoparticle"):
        return []

    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    # Pair uniqueness on slabs uses xy-only MIC (intentional for planar catalogs).
    pbc_xy = [bool(pbc[0]), bool(pbc[1]), False]
    slab_normal = _slab_normal(cell_arr)

    if raw_sites is not None:
        site_entries: list[Site] = list(raw_sites)
        if config.material_type == "slab":
            site_entries = [
                s for s in site_entries if s.site_type in ("hollow", "pore")
            ]
    elif site_context is not None and site_context.raw_unclustered is not None:
        site_entries = list(site_context.raw_unclustered)
        if config.material_type == "slab":
            site_entries = [
                s for s in site_entries if s.site_type in ("hollow", "pore")
            ]
    elif site_context is not None and site_context.sites:
        site_entries = list(site_context.sites)
        if config.material_type == "slab":
            site_entries = [
                s for s in site_entries if s.site_type in ("hollow", "pore")
            ]
    elif config.material_type == "slab":
        # Reuse hollow catalog (unified + hollow/pore filter + dedup).
        site_entries = get_hollow_sites_for_adatoms(
            sites_slab,
            top_layer_tolerance=config.top_layer_tolerance,
            dedup_tolerance=config.hollow_site_dedup_tolerance,
            material_type=config.material_type,
            probe_radius=config.voronoi_probe_radius,
            max_site_distance=config.voronoi_max_site_distance,
            enrich=config.voronoi_site_enrichment,
            site_classification_method=config.site_classification_method,
        )
    else:
        site_entries = get_unified_sites(
            sites_slab,
            top_layer_tolerance=config.top_layer_tolerance,
            material_type=config.material_type,
            probe_radius=config.voronoi_probe_radius,
            max_site_distance=config.voronoi_max_site_distance,
            enrich=config.voronoi_site_enrichment,
            site_classification_method=config.site_classification_method,
        )
    if len(site_entries) < 2:
        return []

    # Hollow helper already deduplicated; still dedup raw/context catalogs.
    used_hollow_helper = (
        raw_sites is None and site_context is None and config.material_type == "slab"
    )
    site_xyz = np.array(
        [np.asarray(s.xyz, dtype=float) for s in site_entries], dtype=float
    )
    if config.material_type == "slab" and not used_hollow_helper:
        keep = _deduplicate_points(
            site_xyz,
            config.hollow_site_dedup_tolerance,
            cell=cell_arr,
            pbc=np.asarray(pbc_xy, dtype=bool),
        )
        site_xyz = site_xyz[keep]
        site_entries = [site_entries[i] for i in np.nonzero(keep)[0]]
    if len(site_xyz) < 2:
        return []

    if (
        existing_adsorbate_positions is not None
        and len(existing_adsorbate_positions) > 0
    ):
        available = filter_sites_by_occupancy(
            site_entries,
            existing_adsorbate_positions,
            cell=cell_arr,
            pbc=pbc,
            min_separation=float(config.min_adsorbate_separation),
        )
        if len(available) < 2:
            return []
        site_xyz = np.asarray([s.xyz for s in available], dtype=float)
        site_entries = list(available)

    symbols = sites_slab.get_chemical_symbols()
    top_positions = sites_slab.get_positions()
    if config.material_type == "nanoparticle":
        radius_indices = list(range(len(top_positions)))
    else:
        top_mask = top_layer_mask_by_normal(
            top_positions,
            cell_arr,
            float(config.top_layer_tolerance),
        )
        radius_indices = list(np.nonzero(top_mask)[0])
    top_radii = [
        r
        for i in radius_indices
        if (r := geom._get_covalent_radius(symbols[int(i)])) is not None
    ]
    mean_top_radius = float(np.mean(top_radii)) if top_radii else 1.0

    site_3d = site_xyz
    if config.material_type == "slab":
        mean_nn_sep = _mean_nn_separation_mic(site_3d, cell_arr, pbc_xy)
    else:
        _nn_tree = KDTree(site_3d)
        nn_d, _ = _nn_tree.query(site_3d, k=2)
        # len(site_3d) >= 2 (early-return above), so query(..., k=2) is 2-D.
        mean_nn_sep = float(np.mean(np.asarray(nn_d, dtype=float)[:, 1]))

    atomic_constraint = _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE * (
        2.0 * mean_top_radius
    )
    surface_constraint = 0.8 * mean_nn_sep
    adaptive_min = min(atomic_constraint, surface_constraint)
    min_fragment_sep = max(_DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM, adaptive_min)
    max_adjacent_sep = float(
        np.clip(
            _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE * mean_nn_sep,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
        )
    )

    if config.material_type == "slab":
        pair_distances = _periodic_site_pair_candidates(
            site_3d, cell_arr, pbc_xy, max_adjacent_sep
        )
    else:
        tree = KDTree(site_3d)
        pair_distances = {
            (min(int(i), int(j)), max(int(i), int(j))): float(
                np.linalg.norm(site_3d[i] - site_3d[j])
            )
            for i, j in tree.query_pairs(r=max_adjacent_sep)
        }

    pairs: list[_DissociativeSitePair] = []
    # Sorting by the origin-index key keeps the catalog order deterministic:
    # ``spec.site_index`` indexes straight into the returned list.
    for (i, j), d in sorted(pair_distances.items()):
        if min_fragment_sep <= d <= max_adjacent_sep:
            normal_i = _site_outward_normal(
                site_3d[i],
                site_entries[i],
                material_type=config.material_type,
                reference_positions=top_positions,
                slab_normal=slab_normal,
            )
            normal_j = _site_outward_normal(
                site_3d[j],
                site_entries[j],
                material_type=config.material_type,
                reference_positions=top_positions,
                slab_normal=slab_normal,
            )
            xyz1 = site_3d[i].copy()
            if config.material_type == "slab":
                dvec, _ = find_mic(
                    (site_3d[j] - site_3d[i]).reshape(1, 3),
                    cell_arr,
                    pbc=pbc_xy,
                )
                xyz2 = site_3d[i] + dvec[0]
            else:
                xyz2 = site_3d[j].copy()
            pairs.append(
                _DissociativeSitePair(
                    xyz1=xyz1,
                    normal1=normal_i,
                    xyz2=xyz2,
                    normal2=normal_j,
                )
            )
    return pairs


def _mean_nn_separation_mic(
    site_3d: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
) -> float:
    """Mean nearest-neighbour separation under MIC for slab sites.

    Built from explicit periodic images plus a single KD-tree query instead of a
    per-site ``ase.geometry.find_mic`` call (which Minkowski-reduces the cell on
    every call for 2D-periodic slabs). Images are taken from the *original*
    basis, so heavily skewed, non-reduced cells could in principle disagree with
    ``find_mic``; this is the same assumption already made by
    ``_periodic_site_pair_candidates`` and ``site_coords._deduplicate_points``,
    and ASE-generated slab cells are (near-)reduced.
    """
    n = len(site_3d)
    if n < 2:
        return _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
    pbc_arr = np.asarray(pbc, dtype=bool)
    cell_arr = np.asarray(cell, dtype=float)
    # One full cell vector of margin guarantees at least the ±1 image shell on
    # every periodic axis, which is what the minimum image needs.
    margin = (
        float(np.max(np.linalg.norm(cell_arr[pbc_arr], axis=1)))
        if np.any(pbc_arr)
        else 0.0
    )
    offsets = _periodic_image_offsets(cell_arr, pbc_arr, margin)
    ext = np.vstack([np.asarray(site_3d, dtype=float) + off for off in offsets])
    tree = KDTree(ext)
    # Each site has exactly len(offsets) images of itself in *ext*, so asking for
    # one more neighbour always leaves at least one genuinely distinct site.
    k = min(len(ext), len(offsets) + 1)
    dists, idxs = tree.query(site_3d, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    idxs = np.atleast_2d(np.asarray(idxs))
    # Drop each site's own periodic images: in an elongated cell a self-image can
    # be closer than the nearest distinct site, so this filter is not optional.
    valid = (idxs % n) != np.arange(n)[:, None]
    nn = np.where(valid, dists, np.inf).min(axis=1)
    finite = nn[np.isfinite(nn)]
    if len(finite) == 0:
        return _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
    return float(np.mean(finite))


def _periodic_site_pair_candidates(
    site_3d: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
    max_sep: float,
) -> dict[tuple[int, int], float]:
    """Origin-index pairs within *max_sep*, mapped to their minimum-image distance.

    The distance comes straight from the extended image coordinates, so it is the
    true minimum-image distance for every pair that is kept (pairs further apart
    than *max_sep* are dropped and never get a distance). Images are enumerated
    from the *original* basis rather than a Minkowski-reduced one; see
    :func:`_mean_nn_separation_mic` for the same caveat.
    """
    pts = np.asarray(site_3d, dtype=float)
    n = len(pts)
    if n < 2:
        return {}
    pbc_arr = np.asarray(pbc, dtype=bool)
    offsets = _periodic_image_offsets(
        np.asarray(cell, dtype=float), pbc_arr, margin=float(max_sep)
    )
    # Stacking image-major keeps the origin index at ``idx % n``.
    ext = np.vstack([pts + off for off in offsets])
    tree = KDTree(ext)
    raw_pairs = tree.query_pairs(r=float(max_sep), output_type="ndarray")
    if len(raw_pairs) == 0:
        return {}
    origin_a = raw_pairs[:, 0] % n
    origin_b = raw_pairs[:, 1] % n
    keep = origin_a != origin_b
    if not np.any(keep):
        return {}
    origin_a = origin_a[keep]
    origin_b = origin_b[keep]
    lo = np.minimum(origin_a, origin_b)
    hi = np.maximum(origin_a, origin_b)
    dist = np.linalg.norm(ext[raw_pairs[keep, 0]] - ext[raw_pairs[keep, 1]], axis=1)
    # Per-key minimum without an n×n buffer: sort by (key, distance) and take the
    # first entry of each run of equal keys.
    flat_key = lo.astype(np.int64) * n + hi.astype(np.int64)
    order = np.lexsort((dist, flat_key))
    flat_sorted = flat_key[order]
    first = np.ones(len(flat_sorted), dtype=bool)
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    sel = order[first]
    return {(int(lo[s]), int(hi[s])): float(dist[s]) for s in sel}


def _place_dissociative_two_sites(
    adsorbate: Atoms,
    sites: Sequence[Site],
    *,
    config: AdsorptionConfig,
    spec: PlacementSpec,
    height_override: float,
    slab: Atoms,
    slab_for_sites: Atoms | None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Place a diatomic at two sites with a shared height offset.

    On slabs, ``height_override`` is the gap above the top-layer surface
    reference (same convention as molecular placement). Hollow Voronoi
    vertices often sit above the metal, so stacking the offset on
    ``site.xyz`` would overshoot the desorption gate. On nanoparticles,
    both fragments share one offset direction so pair spacing is preserved.
    """
    if len(sites) != 2 or len(adsorbate) != 2:
        return None
    site1, site2 = sites[0], sites[1]
    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    slab_normal = _slab_normal(cell_arr)
    ref_pos = sites_slab.get_positions()
    n1 = _site_outward_normal(
        np.asarray(site1.xyz, dtype=float),
        site1,
        material_type=config.material_type,
        reference_positions=ref_pos,
        slab_normal=slab_normal,
    )
    n2 = _site_outward_normal(
        np.asarray(site2.xyz, dtype=float),
        site2,
        material_type=config.material_type,
        reference_positions=ref_pos,
        slab_normal=slab_normal,
    )

    z_offset = float(height_override)
    base1 = np.asarray(site1.xyz, dtype=float)
    base2 = np.asarray(site2.xyz, dtype=float)

    if config.material_type == "slab":
        n_hat = np.asarray(slab_normal, dtype=float)
        n_hat = n_hat / (float(np.linalg.norm(n_hat)) + _VECTOR_NORM_EPS)
        surface_ref, _ = _resolve_surface_ref(
            site1,
            sites_slab,
            "slab",
            rough_slab_local_z=config.rough_slab_local_z,
            top_layer_tolerance=config.top_layer_tolerance,
            planar_z_variance_threshold=config.planar_z_variance_threshold,
        )
        target_h = float(surface_ref + z_offset)
        pos1 = base1 + (target_h - float(np.dot(base1, n_hat))) * n_hat
        pos2 = base2 + (target_h - float(np.dot(base2, n_hat))) * n_hat
        h_surface = float(surface_ref)
        site_reference_frame = (
            "local_site" if config.rough_slab_local_z else "global_top_layer"
        )
    else:
        # Nanoparticle / porous: offset both fragments along a shared normal so
        # divergent local site normals do not laterally expand the H–H spacing.
        n_sum = np.asarray(n1, dtype=float) + np.asarray(n2, dtype=float)
        n_norm = float(np.linalg.norm(n_sum))
        n_hat = n_sum / (n_norm + _VECTOR_NORM_EPS) if n_norm > _VECTOR_NORM_EPS else n1
        pos1 = base1 + z_offset * n_hat
        pos2 = base2 + z_offset * n_hat
        h_surface = 0.5 * (float(np.dot(base1, n_hat)) + float(np.dot(base2, n_hat)))
        site_reference_frame = "local_site"

    symbols = adsorbate.get_chemical_symbols()
    result = Atoms(symbols=symbols, positions=[pos1, pos2])
    result.set_cell(slab.get_cell())
    result.set_pbc(slab.get_pbc())

    fail_reason = _validate_posed_adsorbate(
        result, slab, config, slab_for_sites=slab_for_sites
    )
    if fail_reason is not None:
        return None

    centroid = (pos1 + pos2) / 2.0
    pinv_ab_T, _ = _slab_plane_projectors(cell_arr)
    xy_frac = np.mod(np.asarray(centroid, dtype=float) @ pinv_ab_T, 1.0)
    pose = PlacementPose(
        conformer_index=0,
        site_index=spec.site_index,
        site_type="hollow",
        placement_index=spec.placement_index,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=float(centroid[0]),
        y_abs=float(centroid[1]),
        z_fraction=spec.z_fraction,
        z_abs=float(centroid[2]),
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
    )
    descriptor = _descriptor_from_placement(
        pose,
        z_offset=z_offset,
        surface_ref=h_surface,
        shape="linear",
        slab_indices=None,
        site_source="dissociative_hollow_pair",
        site_reference_frame=site_reference_frame,
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        placement_mode_resolved="sites",
        fragment_positions=(
            (float(pos1[0]), float(pos1[1]), float(pos1[2])),
            (float(pos2[0]), float(pos2[1]), float(pos2[2])),
        ),
    )
    return result, descriptor


def _generate_dissociative_placement_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    site_context: SiteContext | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Place homonuclear-diatomic fragments at two surface sites."""
    if config.material_type not in ("slab", "nanoparticle"):
        return None, f"dissociative_not_supported_for_{config.material_type}"

    if not _is_dissociable_diatomic(adsorbate):
        return None, "not_dissociable_diatomic"

    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    existing_ads_pos = existing_adsorbate_positions(sites_slab, slab)

    pairs = _get_dissociative_site_pairs(
        slab,
        config,
        slab_for_sites=slab_for_sites,
        existing_adsorbate_positions=existing_ads_pos,
        site_context=site_context,
    )
    if not pairs:
        return None, "no_hollow_site_pairs"

    if spec.site_index < 0 or spec.site_index >= len(pairs):
        return None, "invalid_site_index"

    site_pair = pairs[spec.site_index]

    site_a = Site(
        xyz=np.asarray(site_pair.xyz1, dtype=float),
        normal=np.asarray(site_pair.normal1, dtype=float),
        site_type="hollow",
        slab_indices=(),
        material_type=config.material_type,
        site_source="dissociative_hollow_pair",
        env_fingerprint=((), "hollow"),
    )
    syms = adsorbate.get_chemical_symbols()
    z_lo, z_hi = _compute_site_z_base(config, sites_slab, site_a, syms)
    site_offset = _site_type_z_offset(sites_slab, site_a, "hollow")
    z_lo += site_offset
    z_hi += site_offset
    z_offset = z_lo + spec.z_fraction * (z_hi - z_lo)

    site_b = Site(
        xyz=np.asarray(site_pair.xyz2, dtype=float),
        normal=np.asarray(site_pair.normal2, dtype=float),
        site_type="hollow",
        slab_indices=(),
        material_type=config.material_type,
        site_source="dissociative_hollow_pair",
        env_fingerprint=((), "hollow"),
    )
    placed = _place_dissociative_two_sites(
        adsorbate,
        [site_a, site_b],
        config=config,
        spec=spec,
        height_override=float(z_offset),
        slab=slab,
        slab_for_sites=slab_for_sites,
    )
    if placed is None:
        return None, "dissociative_validation_failed"
    return placed, None

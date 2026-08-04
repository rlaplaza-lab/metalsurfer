"""Dissociative diatomic placement and hollow site pair enumeration."""

from __future__ import annotations

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
from ._constants import (
    _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE,
    _VECTOR_NORM_EPS,
)
from . import geometry as geom
from ._material import material_aware_pbc
from .occupancy import existing_adsorbate_positions, filter_sites_by_occupancy
from .orientation import _site_type_z_offset
from .pose import _descriptor_from_placement, _validate_posed_adsorbate
from .site_context import SiteContext
from .site_coords import (
    _deduplicate_points,
    _height_along_slab_normal,
    _periodic_image_offsets,
    _slab_normal,
    _slab_plane_projectors,
    top_layer_mask_by_normal,
)
from .site_enumeration import _compute_site_z_base, get_hollow_sites_for_adatoms, get_unified_sites
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
    """True if molecule is a homonuclear diatomic (e.g. H2, O2, N2)."""
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
    """Unit outward normal for dissociative fragment placement."""
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
        + struct.pack("<d", float(config.min_initial_distance))
        + _pack_optional_float_local(config.top_layer_tolerance)
        + config.material_type.encode()
    )
    return hashlib.sha256(pos_bytes + cell_bytes + numbers_bytes + cfg_bytes).hexdigest()


def _pack_optional_float_local(value: float | None) -> bytes:
    if value is None:
        return b"\x00" + struct.pack("<d", float("nan"))
    return b"\x01" + struct.pack("<d", float(value))


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
    if not occupancy_active and raw_sites is None:
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
        raw_sites is None
        and site_context is None
        and config.material_type == "slab"
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
            min_separation=float(config.min_initial_distance),
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
    if len(site_3d) >= 2:
        if config.material_type == "slab":
            mean_nn_sep = _mean_nn_separation_mic(site_3d, cell_arr, pbc_xy)
        else:
            _nn_tree = KDTree(site_3d)
            nn_d, _ = _nn_tree.query(site_3d, k=2)
            nn_d_arr = np.asarray(nn_d, dtype=float)
            if nn_d_arr.ndim == 2:
                mean_nn_sep = float(np.mean(nn_d_arr[:, 1]))
            else:
                mean_nn_sep = float(np.mean(nn_d_arr))

        atomic_constraint = _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE * (
            2.0 * mean_top_radius
        )
        surface_constraint = 0.8 * mean_nn_sep
        adaptive_min = min(atomic_constraint, surface_constraint)
        min_fragment_sep = max(
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM, adaptive_min
        )
    else:
        mean_nn_sep = _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
        min_fragment_sep = max(
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE * (2.0 * mean_top_radius),
        )
    max_adjacent_sep = float(
        np.clip(
            _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE * mean_nn_sep,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
        )
    )

    if config.material_type == "slab":
        candidate_pairs = _periodic_site_pair_candidates(
            site_3d, cell_arr, pbc_xy, max_adjacent_sep
        )
    else:
        tree = KDTree(site_3d)
        candidate_pairs = tree.query_pairs(r=max_adjacent_sep)

    pairs: list[_DissociativeSitePair] = []
    for i, j in sorted(candidate_pairs):
        if config.material_type == "slab":
            _, dists = find_mic(
                (site_3d[i] - site_3d[j]).reshape(1, 3),
                cell_arr,
                pbc=pbc_xy,
            )
            d = float(dists[0])
        else:
            d = float(np.linalg.norm(site_3d[i] - site_3d[j]))
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
            pairs.append(
                _DissociativeSitePair(
                    xyz1=site_3d[i].copy(),
                    normal1=normal_i,
                    xyz2=site_3d[j].copy(),
                    normal2=normal_j,
                )
            )
    return pairs


def _mean_nn_separation_mic(
    site_3d: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
) -> float:
    """Mean nearest-neighbour separation under MIC for slab sites."""
    n = len(site_3d)
    if n < 2:
        return _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
    nn = np.full(n, np.inf, dtype=float)
    for i in range(n):
        deltas = site_3d - site_3d[i]
        _, dists = find_mic(deltas, cell, pbc=pbc)
        dists = np.asarray(dists, dtype=float)
        dists[i] = np.inf
        nn[i] = float(np.min(dists))
    finite = nn[np.isfinite(nn)]
    if len(finite) == 0:
        return _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
    return float(np.mean(finite))


def _periodic_site_pair_candidates(
    site_3d: np.ndarray,
    cell: np.ndarray,
    pbc: list[bool] | np.ndarray,
    max_sep: float,
) -> set[tuple[int, int]]:
    """Origin-index pairs within *max_sep* including periodic images."""
    pbc_arr = np.asarray(pbc, dtype=bool)
    offsets = _periodic_image_offsets(cell, pbc_arr, margin=float(max_sep))
    extended: list[np.ndarray] = []
    origin_idx: list[int] = []
    for off in offsets:
        for i, pos in enumerate(site_3d):
            extended.append(pos + off)
            origin_idx.append(i)
    ext_arr = np.asarray(extended, dtype=float)
    tree = KDTree(ext_arr)
    raw_pairs = tree.query_pairs(r=float(max_sep))
    out: set[tuple[int, int]] = set()
    for ia, ib in raw_pairs:
        i = origin_idx[ia]
        j = origin_idx[ib]
        if i == j:
            continue
        out.add((min(i, j), max(i, j)))
    return out


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
    """Place a diatomic at two sites with shared height offset along each normal."""
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
    pos1 = np.asarray(site1.xyz, dtype=float) + z_offset * n1
    pos2 = np.asarray(site2.xyz, dtype=float) + z_offset * n2
    symbols = adsorbate.get_chemical_symbols()
    result = Atoms(symbols=symbols, positions=[pos1, pos2])
    result.set_cell(slab.get_cell())
    result.set_pbc(slab.get_pbc())

    fail_reason = _validate_posed_adsorbate(
        result, slab, config, slab_for_sites=slab_for_sites
    )
    if fail_reason is not None:
        return None

    if config.material_type == "nanoparticle":
        h_surface = 0.5 * (
            float(np.dot(np.asarray(site1.xyz, dtype=float), n1))
            + float(np.dot(np.asarray(site2.xyz, dtype=float), n2))
        )
        site_reference_frame = "local_site"
    else:
        heights = _height_along_slab_normal(sites_slab.get_positions(), cell_arr)
        h_surface = float(np.max(heights))
        site_reference_frame = "global_top_layer"

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


def place_at_sites(
    adsorbate: Atoms,
    sites: Sequence[Site],
    *,
    config: AdsorptionConfig,
    spec: PlacementSpec,
    height_override: float | None = None,
    slab: Atoms | None = None,
    slab_for_sites: Atoms | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Place a diatomic at exactly two sites (dissociative path)."""
    if not sites:
        return None
    if slab is None:
        raise ValueError("place_at_sites requires slab")
    if len(sites) != 2:
        raise ValueError(
            f"place_at_sites supports exactly 2 sites (dissociative), got {len(sites)}"
        )
    if height_override is None:
        raise ValueError("two-site place_at_sites requires height_override")
    return _place_dissociative_two_sites(
        adsorbate,
        sites,
        config=config,
        spec=spec,
        height_override=float(height_override),
        slab=slab,
        slab_for_sites=slab_for_sites,
    )


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

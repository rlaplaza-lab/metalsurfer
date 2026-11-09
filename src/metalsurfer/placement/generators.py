"""Build placements from specs: sites, orientations, and validation.

Public orchestration façade (enumerate, materialize, complexity/budget).
Private helpers live in ``dissociative``, ``orientation``, ``pose``,
and ``site_context`` — import those modules directly in tests.
"""

import logging
import math
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import groupby
from typing import Literal

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from . import geometry as geom
from . import policy
from ._constants import (
    _POLICY_BRANCH_SEED_STEP,
    _PORE_SITE_CAP_FLOOR,
    _PORE_SITE_CAP_MULTIPLIER,
    _PORE_SITE_CAP_NUM_PLACEMENTS_DEFAULT,
    _POROUS_SITE_INDEX_WEIGHT,
    _SITE_CAP_WALL_RESERVE_FRACTION,
    _SITE_NEIGHBOR_ABS_TOL_ANGSTROM,
)
from ._material import material_aware_pbc
from ._parallel import resolve_materialize_workers
from .clash import atom_radii_for_symbols
from .dissociative import (
    _generate_dissociative_placement_from_spec,
    _get_dissociative_site_pairs,
    _is_dissociable_diatomic,
)
from .occupancy import (
    _footprint_clearances_from_mic,
    _sites_clearance_and_anchor_mask,
    existing_adsorbate_cloud,
    incoming_inplane_radius,
)
from .orientation import (
    _adsorbate_binder_indices,
    _estimate_parallel_fraction,
    _is_flat_aromatic,
)
from .pose import (
    _finalize_placement,
    _pose_from_spec,
    _PoseBatchCache,
    build_pose_batch_cache,
)
from .site_context import (
    SiteContext,
    site_context_for_sampling,
)
from .site_types import Site

logger = logging.getLogger(__name__)


def generate_placements_from_specs(
    specs: Sequence[PlacementSpec],
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
    pose_cache: _PoseBatchCache | None = None,
) -> list[tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]]:
    """Materialize specs in input order, optionally via a thread pool.

    Each entry is ``(result, fail_reason)`` matching
    :func:`generate_placement_from_spec_with_reason`. Calculator attachment is
    left to the caller. Worker count comes from
    ``config.placement_materialize_workers``, inheriting the global
    ``config.n_jobs`` when unset (both joblib-style ``n_jobs``).

    Parameters
    ----------
    specs
        Sequence of placement specifications.
    conformers
        List of adsorbate conformers.
    slab
        Substrate slab.
    config
        Adsorption configuration.
    smiles
        Optional SMILES string for the adsorbate.
    site_context
        Optional precomputed site context.
    slab_for_sites
        Optional substrate for site detection.
    materialization_cache
        Optional cache of materialized placements keyed by placement index.
    pose_cache
        Prebuilt pose batch. Built here when omitted and any spec is uncached.
    """
    if not specs:
        return []

    placement_ref = slab_for_sites if slab_for_sites is not None else slab
    if pose_cache is None and (
        materialization_cache is None
        or any(
            materialization_cache.get(int(spec.placement_index)) is None
            for spec in specs
        )
    ):
        pose_cache = build_pose_batch_cache(placement_ref, conformers, config)

    def _one(
        spec: PlacementSpec,
    ) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
        cached = (
            materialization_cache.get(int(spec.placement_index))
            if materialization_cache is not None
            else None
        )
        if cached is not None:
            adsorbate, descriptor = cached
            return (adsorbate.copy(), descriptor), None
        return generate_placement_from_spec_with_reason(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
            pose_cache=pose_cache,
        )

    workers_setting = (
        config.placement_materialize_workers
        if config.placement_materialize_workers is not None
        else config.n_jobs
    )
    n_workers = resolve_materialize_workers(
        workers_setting,
        n_tasks=len(specs),
    )
    if n_workers == 1 or len(specs) == 1:
        return [_one(spec) for spec in specs]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        return list(pool.map(_one, specs))


@dataclass
class _SpecGridInfo:
    is_dissociative: bool
    unique_sites: list[Site]
    use_sites: bool
    site_indices: list[int]
    shape: str
    symbols: list[str]
    n_binders: int
    flat_aromatic: bool
    n_hollow_pairs: int
    clearances: np.ndarray | None = None
    conformer_clearances: np.ndarray | None = None
    site_source: str = "no_sites"
    n_clustered: int = 0
    # None on a bare surface, so the bins stay site_type.
    occupied_neighbor_counts: dict[int, int] | None = None


_SITE_TYPE_LOG_ORDER = ("hollow", "bridge", "atop", "pore")


def log_site_sample(
    info: _SpecGridInfo,
    n_placements: int,
    specs: Sequence[PlacementSpec] | None = None,
) -> None:
    """Log placements drawn against the open site catalog.

    INFO reports the draw and its share of open sites. DEBUG adds the catalog
    source, open versus full size, symmetry orbits when reduced, and per-type
    counts. Per-type sampled counts are included when *specs* is that draw.

    Parameters
    ----------
    info
        Placement grid for this draw.
    n_placements
        Screening sample size, not the oversample buffer.
    specs
        Specs in that sample. ``None`` skips per-type sampled counts.
    """
    if info.is_dissociative:
        n_open = info.n_hollow_pairs
        unit = "hollow pairs"
        catalog = Counter({"hollow": n_open}) if n_open else Counter()
        detail = f"Site catalog ({info.site_source}): {n_open} hollow pairs"
    else:
        indices = info.site_indices if info.use_sites else ()
        n_open = len(indices)
        unit = "sites"
        catalog = Counter(str(info.unique_sites[index].site_type) for index in indices)
        detail = (
            f"Site catalog ({info.site_source}): "
            f"{n_open} open, {len(info.unique_sites)} catalog"
        )
    percent = f" ({100.0 * n_placements / n_open:.0f}%)" if n_open else ""
    logger.info(
        "Site sample: %d placements, %d %s%s",
        n_placements,
        n_open,
        unit,
        percent,
    )

    if info.site_source == "symmetry_aware" and info.unique_sites:
        multiplicity = sum(
            site.symmetry_multiplicity or 1 for site in info.unique_sites
        )
        detail += (
            f", {len(info.unique_sites)} orbits "
            f"(multiplicity {multiplicity}, clustered {info.n_clustered})"
        )
    sampled = (
        None if specs is None else Counter(spec.site_type or "none" for spec in specs)
    )
    present = set(catalog)
    if sampled is not None:
        present |= set(sampled)
    names = [name for name in _SITE_TYPE_LOG_ORDER if name in present]
    names.extend(sorted(present - set(_SITE_TYPE_LOG_ORDER)))
    bits = [
        (
            f"{name} {catalog[name]}"
            if sampled is None
            else f"{name} {catalog[name]} (sampled {sampled[name]})"
        )
        for name in names
    ]
    if bits:
        detail = f"{detail}, {', '.join(bits)}"
    logger.debug("%s", detail)


def _site_band_key(site: Site) -> tuple[int, int]:
    kind_rank = 0 if site.kind == "void" else 1
    coord = {"hollow": 0, "bridge": 1, "atop": 2, "pore": 0}.get(str(site.site_type), 3)
    return (kind_rank, coord)


def _site_rank_key(
    site: Site,
    index: int,
    *,
    clearance: float,
) -> tuple[int, int, float, int]:
    """Sort key: void before wall, then coordination, then larger clearance."""
    clear = -float(clearance)
    return (*_site_band_key(site), clear, int(index))


def _span_ranked_clearances(
    sites: list[Site],
    ranked: list[int],
    clearances: np.ndarray,
) -> list[int]:
    """Within each kind/coordination band, open with a lateral contact.

    The first site is the closest non-overlapping clearance, then the farthest.
    Overlaps follow, mildest first. Equal clearances keep *ranked* order, so a
    bare slab is unchanged.
    """

    def _clearance(index: int) -> float:
        value = float(clearances[index])
        return value if np.isfinite(value) else float("inf")

    def _interleave(ordered: list[int]) -> list[int]:
        mixed: list[int] = []
        lo, hi = 0, len(ordered) - 1
        while lo <= hi:
            mixed.append(ordered[lo])
            lo += 1
            if lo <= hi:
                mixed.append(ordered[hi])
                hi -= 1
        return mixed

    def _span(band: list[int]) -> list[int]:
        if len(band) < 2 or _clearance(band[0]) == _clearance(band[-1]):
            return list(band)
        ordered = sorted(band, key=_clearance)
        clashes = [index for index in ordered if _clearance(index) < 0.0]
        contacts = [index for index in ordered if _clearance(index) >= 0.0]
        if not contacts:
            return _interleave(list(reversed(ordered)))
        return _interleave(contacts) + list(reversed(clashes))

    return [
        index
        for _key, band in groupby(
            ranked, key=lambda index: _site_band_key(sites[index])
        )
        for index in _span(list(band))
    ]


def _rank_site_indices(
    sites: list[Site],
    indices: list[int],
    *,
    clearances: np.ndarray | None = None,
) -> list[int]:
    """Order *indices* by kind, coordination, and clearance (not site_source)."""

    def _rank(i: int) -> tuple[int, int, float, int]:
        if clearances is not None and i < len(clearances):
            clear = float(clearances[i])
        else:
            nn_distance = sites[i].nn_distance
            clear = float(nn_distance) if nn_distance is not None else 0.0
        return _site_rank_key(sites[i], i, clearance=clear)

    return sorted(indices, key=_rank)


def _cap_ranked_site_indices(
    sites: list[Site],
    ranked: list[int],
    *,
    cap: int,
) -> list[int]:
    """Apply *cap* while reserving the last quarter for wall sites when present."""
    if len(ranked) <= cap:
        return list(ranked)
    wall = [i for i in ranked if sites[i].kind == "wall"]
    void = [i for i in ranked if sites[i].kind != "wall"]
    if not wall or not void:
        return ranked[:cap]
    wall_reserve = max(1, int(cap * _SITE_CAP_WALL_RESERVE_FRACTION))
    void_slots = cap - wall_reserve
    kept_void = void[:void_slots]
    kept_wall = wall[:wall_reserve]
    kept = set(kept_void) | set(kept_wall)
    return [i for i in ranked if i in kept]


def _spec_grid_info(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None,
    full_slab: Atoms | None = None,
) -> _SpecGridInfo:
    """Build enumerate/estimate inputs; footprint views leave SiteContext untouched."""
    is_dissociative = (
        config.enable_dissociative_placement
        and config.material_type in ("slab", "nanoparticle", "porous")
        and _is_dissociable_diatomic(conformers[0])
    )
    _ctx = site_context_for_sampling(slab, config, site_context, full_slab=full_slab)
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    clustered = _ctx.clustered_sites
    n_clustered = len(clustered) if clustered is not None else len(unique_sites)
    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    existing_ads_pos, existing_radii = existing_adsorbate_cloud(
        slab,
        full_slab,
    )
    occupancy_pos, occupancy_radii = existing_ads_pos, existing_radii
    if occupancy_pos is not None and occupancy_radii is not None:
        occupancy_pos, occupancy_radii = _adsorbate_atoms_on_substrate(
            occupancy_pos,
            occupancy_radii,
            slab,
            cell=cell_arr,
            pbc=pbc,
            connectivity_multiplier=float(config.connectivity_multiplier),
        )

    footprint_scale = float(config.occupancy_footprint_scale)
    conf_radii = [
        incoming_inplane_radius(conf, footprint_scale=footprint_scale)
        for conf in conformers
    ]
    clearances: np.ndarray | None = None
    conformer_clearances: np.ndarray | None = None
    occupied_ids: frozenset[int] | None = None

    if use_sites and unique_sites:
        if occupancy_pos is None or np.asarray(occupancy_pos).size == 0:
            site_indices = list(range(len(unique_sites)))
            clearances = np.full(len(unique_sites), np.inf, dtype=float)
        else:
            need_footprint = (
                config.occupancy_use_footprint
                and occupancy_radii is not None
                and any(r > 0.0 for r in conf_radii)
            )
            anchor_mask, min_dists, mic_vecs = _sites_clearance_and_anchor_mask(
                unique_sites,
                occupancy_pos,
                cell=cell_arr,
                pbc=pbc,
                min_separation=float(config.min_adsorbate_separation),
                need_mic_vecs=need_footprint,
            )
            anchor_ok = {i for i, keep in enumerate(anchor_mask) if keep}
            occupied_ids = frozenset(
                i for i, keep in enumerate(anchor_mask) if not keep
            )
            # Ligand atop/bridge sites sit on the ligand by construction.
            n_sub = len(slab)
            for i, site in enumerate(unique_sites):
                if any(int(idx) >= n_sub for idx in site.slab_indices):
                    anchor_ok.add(i)
            if need_footprint:
                assert occupancy_radii is not None
                # Best per-conformer disk ranks the site. Negative clearance
                # does not reject; clash handles residual packing.
                per_conf_clear: list[np.ndarray] = []
                for r_in in conf_radii:
                    if float(r_in) <= 0.0:
                        # Point footprint: fall back to anchor clearance.
                        per_conf_clear.append(min_dists.copy())
                    else:
                        per_conf_clear.append(
                            _footprint_clearances_from_mic(
                                unique_sites,
                                mic_vecs,
                                occupancy_radii,
                                float(r_in),
                            )
                        )
                site_indices = sorted(anchor_ok)
                conformer_clearances = np.stack(per_conf_clear, axis=0)
                clearances = np.max(conformer_clearances, axis=0)
            else:
                site_indices = sorted(anchor_ok)
                clearances = min_dists
        if not site_indices:
            logger.warning(
                "Occupancy pruning removed all %d sites under coverage; "
                "no site-based placement specs will be generated",
                len(unique_sites),
            )
            site_indices = []
            use_sites = False
        else:
            site_indices = _rank_site_indices(
                unique_sites,
                site_indices,
                clearances=clearances,
            )
            site_indices = _span_ranked_clearances(
                unique_sites,
                site_indices,
                clearances,
            )
            has_void = any(unique_sites[i].kind == "void" for i in site_indices)
            if has_void:
                pore_cap = max(
                    int(config.num_placements or _PORE_SITE_CAP_NUM_PLACEMENTS_DEFAULT)
                    * _PORE_SITE_CAP_MULTIPLIER,
                    _PORE_SITE_CAP_FLOOR,
                )
                site_indices = _cap_ranked_site_indices(
                    unique_sites,
                    site_indices,
                    cap=pore_cap,
                )
    else:
        site_indices = []
        use_sites = False

    conf0_pos = conformers[0].get_positions()
    ads_pos = conf0_pos - np.mean(conf0_pos, axis=0)
    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    symbols = conformers[0].get_chemical_symbols()
    binders = _adsorbate_binder_indices(symbols, smiles)
    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)

    n_hollow_pairs = 0
    if is_dissociative:
        working_slab = full_slab if full_slab is not None else slab
        pair_cloud = occupancy_pos
        if pair_cloud is not None and np.asarray(pair_cloud).size == 0:
            pair_cloud = None
        n_hollow_pairs = len(
            _get_dissociative_site_pairs(
                working_slab,
                config,
                slab_for_sites=slab,
                existing_adsorbate_positions=pair_cloud,
                site_context=_ctx,
            )
        )

    neighbor_counts: dict[int, int] | None = None
    if occupied_ids is not None and unique_sites:
        positions = {
            index: np.asarray(site.xyz, dtype=float)
            for index, site in enumerate(unique_sites)
        }
        site_types = {
            index: str(site.site_type) for index, site in enumerate(unique_sites)
        }
        neighbor_counts = geom.occupied_neighbor_counts(
            positions,
            occupied_ids,
            cell_arr,
            pbc,
            site_types=site_types,
        )
    return _SpecGridInfo(
        is_dissociative=is_dissociative,
        unique_sites=unique_sites,
        use_sites=use_sites,
        site_indices=site_indices,
        shape=shape,
        symbols=symbols,
        n_binders=len(binders),
        flat_aromatic=flat_aromatic,
        n_hollow_pairs=n_hollow_pairs,
        clearances=clearances,
        conformer_clearances=conformer_clearances,
        site_source=_ctx.source,
        n_clustered=n_clustered,
        occupied_neighbor_counts=neighbor_counts,
    )


def _site_type_for_grid(info: _SpecGridInfo, site_idx: int) -> str | None:
    """Site type for a grid index, or ``None`` when the catalog has no site."""
    if info.is_dissociative:
        return "hollow"
    sites = info.unique_sites
    if not info.use_sites or site_idx < 0 or site_idx >= len(sites):
        return None
    return str(sites[site_idx].site_type)


def _site_is_void_for_grid(info: _SpecGridInfo, site_idx: int) -> bool:
    """Whether the catalog site at *site_idx* is a void (full ``z_fraction`` axis)."""
    if info.is_dissociative:
        return False
    sites = info.unique_sites
    if not info.use_sites or site_idx < 0 or site_idx >= len(sites):
        return False
    return sites[site_idx].kind == "void"


def _substrate_ligand_site_split(
    info: _SpecGridInfo,
    n_substrate: int,
) -> tuple[list[int], list[int]] | None:
    """Split sampling indices into substrate and ligand supports, or None."""
    if not info.use_sites or not info.site_indices:
        return None
    substrate: list[int] = []
    ligand: list[int] = []
    for index in info.site_indices:
        site = info.unique_sites[int(index)]
        if any(int(atom) >= int(n_substrate) for atom in site.slab_indices):
            ligand.append(int(index))
        else:
            substrate.append(int(index))
    if not substrate or not ligand:
        return None
    return substrate, ligand


def _site_bin_for_grid(info: _SpecGridInfo, site_idx: int) -> str:
    """Occupancy count under coverage, otherwise site type."""
    counts = info.occupied_neighbor_counts
    if counts is not None:
        return str(int(counts.get(int(site_idx), 0)))
    site_type = _site_type_for_grid(info, site_idx)
    return str(site_type) if site_type is not None else "none"


def _clearance_for_grid(info: _SpecGridInfo) -> Callable[[int], float] | None:
    """Clearance lookup for the working set, or ``None`` when it was not ranked."""
    clearances = info.clearances
    if clearances is None:
        return None

    def _clearance(site_idx: int) -> float:
        if site_idx < 0 or site_idx >= len(clearances):
            return float("-inf")
        return float(clearances[int(site_idx)])

    return _clearance


def _finite_nonnegative_clearance(info: _SpecGridInfo, site_idx: int) -> float | None:
    """Substrate clearance, ignoring ligand infinities and footprint overlaps."""
    clearances = info.clearances
    if clearances is None or site_idx < 0 or site_idx >= len(clearances):
        return None
    value = float(clearances[int(site_idx)])
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        return None
    return value


def _clearest_site_per_type(info: _SpecGridInfo) -> list[int]:
    """Clearest count-0 site of each type, plus sites that tie its clearance.

    Count 0 is the terrace when any site has that count. Ligand sites with
    infinite clearance are not candidates, so the contact poses are not
    started on the adsorbate already down. A second site of the same type
    within the site-identity tolerance (fcc and hcp on an open terrace) is
    kept. One site of each type comes first; ties follow.
    """
    counts = info.occupied_neighbor_counts or {}
    has_terrace = any(int(counts.get(index, 0)) == 0 for index in info.site_indices)
    best: dict[str, float] = {}
    ranked: list[tuple[str, float, int]] = []
    for index in info.site_indices:
        if has_terrace and int(counts.get(int(index), 0)) != 0:
            continue
        clearance = _finite_nonnegative_clearance(info, int(index))
        if clearance is None:
            continue
        site_type = _site_type_for_grid(info, int(index)) or "none"
        current = best.get(site_type)
        if current is None or clearance > current:
            best[site_type] = clearance
        ranked.append((site_type, clearance, int(index)))
    primary: list[int] = []
    ties: list[int] = []
    seen: set[str] = set()
    for site_type, clearance, index in ranked:
        leader = best[site_type]
        if abs(clearance - leader) > float(_SITE_NEIGHBOR_ABS_TOL_ANGSTROM):
            continue
        if site_type in seen:
            ties.append(index)
        else:
            seen.add(site_type)
            primary.append(index)
    return [*primary, *ties]


def _upright_contact_spec(
    info: _SpecGridInfo,
    site_idx: int,
    placement_index: int,
) -> PlacementSpec:
    """One covalent-contact, upright pose on *site_idx*."""
    site_type = _site_type_for_grid(info, site_idx)
    z_values = policy._kept_z_fractions(
        site_type,
        is_void=_site_is_void_for_grid(info, site_idx),
    )
    orientation: Literal["EN-down", "vertical", "round"] = "round"
    if info.flat_aromatic:
        orientation = "EN-down"
    elif info.shape == "linear":
        orientation = "vertical"
    return PlacementSpec(
        conformer_index=0,
        orientation_type=orientation,
        face_flip=False,
        en_atom_index=None,
        site_index=int(site_idx),
        site_type=site_type,
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=float(z_values[0]),
        placement_index=int(placement_index),
    )


def _pose_identity(spec: PlacementSpec) -> tuple[object, ...]:
    """Geometry key shared by the contact prefix and the occupancy draw."""
    return (
        int(spec.site_index),
        int(spec.conformer_index),
        str(spec.orientation_type),
        float(spec.tilt_deg),
        float(spec.azimuth_deg),
        float(spec.azimuth_in_plane_deg),
        round(float(spec.z_fraction), 5),
        bool(spec.face_flip),
        None if spec.en_atom_index is None else int(spec.en_atom_index),
    )


def _preferred_site_bins(info: _SpecGridInfo, *, prefer_pores: bool) -> tuple[str, ...]:
    """Pore buckets lead on a bare catalog. Occupancy counts are not preferred."""
    if prefer_pores and info.occupied_neighbor_counts is None:
        return ("pore",)
    return ()


def _adsorbate_atoms_on_substrate(
    positions: np.ndarray,
    radii: np.ndarray,
    slab: Atoms,
    *,
    cell: np.ndarray,
    pbc: Sequence[bool],
    connectivity_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep atoms that are actually bound to the substrate.

    A tail standing off the surface does not occupy the sites beside the
    bound head, so the footprint disk follows atoms inside the connectivity
    cutoff. Nothing inside that cutoff means the placement is not adsorbed.
    """
    if len(positions) == 0 or len(slab) == 0:
        return positions, radii
    slab_pos = np.asarray(slab.get_positions(), dtype=float)
    slab_radii = atom_radii_for_symbols(list(slab.get_chemical_symbols()))
    _vecs, dists = geom._mol_slab_pairwise_mic(positions, slab_pos, cell, list(pbc))
    r_sum = radii[:, None] + slab_radii[None, :]
    nearest = np.min(dists / r_sum, axis=1)
    keep = nearest <= float(connectivity_multiplier)
    if not np.any(keep):
        return (
            np.zeros((0, positions.shape[-1]), dtype=float),
            np.zeros((0,), dtype=float),
        )
    return positions[keep], radii[keep]


def _enumerate_reserved_substrate_and_ligand(
    info: _SpecGridInfo,
    split: tuple[list[int], list[int]],
    *,
    n_desired: int,
    parallel_fraction: float,
    filter_spec: Callable[[PlacementSpec], bool] | None,
    seed: int,
    prefer_pores: bool,
    conformer_energies: list[float] | None,
    conformer_weighting: str,
    boltzmann_temperature: float,
    n_conformers: int,
) -> list[PlacementSpec]:
    """Keep about half of a short draw on the substrate when ligand supports exist."""
    substrate, ligand = split
    n_sub_draw = max(1, (int(n_desired) + 1) // 2)
    n_lig_draw = max(0, int(n_desired) - n_sub_draw)
    preferred = _preferred_site_bins(info, prefer_pores=prefer_pores)

    def _build(indices: list[int], n: int, draw_seed: int) -> list[PlacementSpec]:
        if n <= 0 or not indices:
            return []
        specs = policy.build_batch_placement_specs(
            n_conformers=n_conformers,
            site_indices=list(indices),
            site_type_for_index=lambda site_idx: _site_type_for_grid(info, site_idx),
            shape=info.shape,
            n_binders=info.n_binders,
            flat_aromatic=info.flat_aromatic,
            parallel_fraction=parallel_fraction,
            n_desired=n,
            filter_spec=filter_spec,
            dissociative=False,
            n_hollow_pairs=0,
            seed=draw_seed,
            preferred_site_types=preferred,
            site_index_weight=(_POROUS_SITE_INDEX_WEIGHT if prefer_pores else 0.0),
            conformer_energies=conformer_energies,
            conformer_weighting=conformer_weighting,
            boltzmann_temperature=boltzmann_temperature,
            site_is_void_for_index=lambda site_idx: _site_is_void_for_grid(
                info, site_idx
            ),
            site_bin_for_index=lambda site_idx: _site_bin_for_grid(info, site_idx),
            site_clearance_for_index=_clearance_for_grid(info),
        )
        return specs[:n]

    ligand_seed = seed + _POLICY_BRANCH_SEED_STEP
    specs_sub = _build(substrate, n_sub_draw, seed)
    specs_lig = _build(ligand, n_lig_draw, ligand_seed)
    if len(specs_sub) < n_sub_draw:
        n_lig = n_lig_draw + (n_sub_draw - len(specs_sub))
        specs_lig = _build(ligand, n_lig, ligand_seed)
    elif len(specs_lig) < n_lig_draw:
        n_sub = n_sub_draw + (n_lig_draw - len(specs_lig))
        specs_sub = _build(substrate, n_sub, seed)
    merged: list[PlacementSpec] = []
    for i in range(max(len(specs_sub), len(specs_lig))):
        if i < len(specs_sub):
            merged.append(specs_sub[i])
        if i < len(specs_lig) and len(merged) < n_desired:
            merged.append(specs_lig[i])
        if len(merged) >= n_desired:
            break
    return merged[:n_desired]


def _enumerate_covered_specs(
    info: _SpecGridInfo,
    *,
    n_desired: int,
    parallel_fraction: float,
    filter_spec: Callable[[PlacementSpec], bool] | None,
    seed: int,
    conformer_energies: list[float] | None,
    conformer_weighting: str,
    boltzmann_temperature: float,
    n_conformers: int,
) -> list[PlacementSpec]:
    """One contact pose per terrace type, then occupancy counts.

    The prefix is a single upright covalent-contact pose on the clearest
    count-0 site of each type, including a clearance tie. When an open site
    has a positive occupancy count and the request is longer than that
    prefix, the remaining slots come from the count round-robin. A geometry
    already emitted is skipped. A shortfall draws again with the next seed
    instead of repeating the same sample.
    """
    contact_sites = _clearest_site_per_type(info)
    counts = info.occupied_neighbor_counts or {}
    has_edge = any(int(counts.get(int(index), 0)) > 0 for index in info.site_indices)
    n_types = len(
        {_site_type_for_grid(info, index) or "none" for index in contact_sites}
    )
    n_prefix = len(contact_sites)
    if has_edge and int(n_desired) > n_types and n_prefix >= int(n_desired):
        n_prefix = int(n_desired) - 1
    else:
        n_prefix = min(int(n_desired), n_prefix)

    def _draw(n: int, draw_seed: int) -> list[PlacementSpec]:
        if n <= 0 or not info.site_indices:
            return []
        return policy.build_batch_placement_specs(
            n_conformers=n_conformers,
            site_indices=list(info.site_indices),
            site_type_for_index=lambda site_idx: _site_type_for_grid(info, site_idx),
            shape=info.shape,
            n_binders=info.n_binders,
            flat_aromatic=info.flat_aromatic,
            parallel_fraction=parallel_fraction,
            n_desired=n,
            filter_spec=filter_spec,
            dissociative=False,
            n_hollow_pairs=0,
            seed=draw_seed,
            preferred_site_types=(),
            site_index_weight=0.0,
            conformer_energies=conformer_energies,
            conformer_weighting=conformer_weighting,
            boltzmann_temperature=boltzmann_temperature,
            site_is_void_for_index=lambda site_idx: _site_is_void_for_grid(
                info, site_idx
            ),
            site_bin_for_index=lambda site_idx: _site_bin_for_grid(info, site_idx),
            site_clearance_for_index=_clearance_for_grid(info),
        )

    specs: list[PlacementSpec] = []
    seen: set[tuple[object, ...]] = set()
    for site_idx in contact_sites[:n_prefix]:
        spec = _upright_contact_spec(info, site_idx, len(specs))
        if filter_spec is not None and not filter_spec(spec):
            continue
        identity = _pose_identity(spec)
        if identity in seen:
            continue
        seen.add(identity)
        specs.append(spec)

    draw_seed = int(seed)
    while len(specs) < int(n_desired):
        batch = _draw(int(n_desired), draw_seed)
        added = 0
        for spec in batch:
            identity = _pose_identity(spec)
            if identity in seen:
                continue
            seen.add(identity)
            specs.append(spec)
            added += 1
            if len(specs) >= int(n_desired):
                break
        if added == 0:
            break
        draw_seed += _POLICY_BRANCH_SEED_STEP
    for index, spec in enumerate(specs[:n_desired]):
        spec.placement_index = index
    return specs[:n_desired]


def enumerate_placement_specs(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    site_context: SiteContext | None = None,
    seed: int | None = None,
    full_slab: Atoms | None = None,
    conformer_energies: list[float] | None = None,
    grid_info: _SpecGridInfo | None = None,
) -> list[PlacementSpec]:
    """Enumerate placement specs for diverse sampling.

    Normal, sequential, and n-tuple screens share one stratified height-shell
    walk over the open catalog. Under coverage the bucket is how many occupied
    sites of the same type lie within two neighbor-graph steps. Those counts
    are drawn equally, and ``site_type`` still rotates inside a count. A
    covered short draw keeps one upright contact pose on the clearest open
    site of each type, including a clearance tie, then fills the rest by
    occupancy count. A bare surface keeps ``site_type`` buckets, with pores
    preferred.

    *conformer_energies* (same order/length as *conformers*) enables the
    deterministic Boltzmann conformer prior when
    ``config.conformer_weighting == "boltzmann"``. Without them the draw stays
    conformer-agnostic (the default).

    Parameters
    ----------
    conformers
        List of adsorbate conformers.
    slab
        Substrate slab.
    config
        Adsorption configuration.
    smiles
        SMILES string or None.
    n_desired
        Number of specs to generate.
    filter_spec
        Optional callable to filter generated specs.
    site_context
        Optional precomputed site context.
    seed
        Optional random seed override.
    full_slab
        Optional full slab including pre-adsorbed atoms.
    conformer_energies
        Optional conformer energies for Boltzmann weighting.
    grid_info
        Optional precomputed :class:`_SpecGridInfo` (shared with estimate).
    """
    if not conformers:
        return []

    eff_seed = config.seed if seed is None else seed
    info = grid_info or _spec_grid_info(
        conformers, slab, config, smiles, site_context, full_slab=full_slab
    )
    if info.is_dissociative:
        if info.n_hollow_pairs < 1:
            return []
    elif not info.site_indices:
        # Occupancy pruned all sites (empty list) — do not fall back to random XY.
        return []

    parallel_fraction = config.flat_aromatic_parallel_fraction
    if config.adaptive_parallel_fraction and info.flat_aromatic:
        parallel_fraction = _estimate_parallel_fraction(info.symbols, smiles)

    prefer_pores = (
        any(info.unique_sites[i].kind == "void" for i in info.site_indices)
        if info.use_sites and info.site_indices
        else False
    )
    if info.occupied_neighbor_counts is not None and not info.is_dissociative:
        return _enumerate_covered_specs(
            info,
            n_desired=n_desired,
            parallel_fraction=parallel_fraction,
            filter_spec=filter_spec,
            seed=eff_seed,
            conformer_energies=conformer_energies,
            conformer_weighting=config.conformer_weighting,
            boltzmann_temperature=config.boltzmann_temperature,
            n_conformers=len(conformers),
        )
    split = _substrate_ligand_site_split(info, len(slab))
    if split is not None and not info.is_dissociative:
        return _enumerate_reserved_substrate_and_ligand(
            info,
            split,
            n_desired=n_desired,
            parallel_fraction=parallel_fraction,
            filter_spec=filter_spec,
            seed=eff_seed,
            prefer_pores=prefer_pores,
            conformer_energies=conformer_energies,
            conformer_weighting=config.conformer_weighting,
            boltzmann_temperature=config.boltzmann_temperature,
            n_conformers=len(conformers),
        )
    preferred = _preferred_site_bins(info, prefer_pores=prefer_pores)
    return policy.build_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=list(info.site_indices),
        site_type_for_index=lambda site_idx: _site_type_for_grid(info, site_idx),
        shape=info.shape,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        parallel_fraction=parallel_fraction,
        n_desired=n_desired,
        filter_spec=filter_spec,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
        seed=eff_seed,
        preferred_site_types=preferred,
        site_index_weight=(_POROUS_SITE_INDEX_WEIGHT if prefer_pores else 0.0),
        conformer_energies=conformer_energies,
        conformer_weighting=config.conformer_weighting,
        boltzmann_temperature=config.boltzmann_temperature,
        site_is_void_for_index=lambda site_idx: _site_is_void_for_grid(info, site_idx),
        site_bin_for_index=lambda site_idx: _site_bin_for_grid(info, site_idx),
        site_clearance_for_index=_clearance_for_grid(info),
    )


def estimate_placement_spec_capacity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
    full_slab: Atoms | None = None,
    grid_info: _SpecGridInfo | None = None,
) -> int:
    """Estimate total enumerated specs for current conformers/site grid.

    Parameters
    ----------
    conformers
        List of adsorbate conformers.
    slab
        Substrate slab.
    config
        Adsorption configuration.
    smiles
        SMILES string or None.
    site_context
        Optional precomputed site context.
    full_slab
        Optional full slab including pre-adsorbed atoms.
    grid_info
        Optional precomputed :class:`_SpecGridInfo` (shared with enumerate).
    """
    if not conformers:
        return 0
    info = grid_info or _spec_grid_info(
        conformers, slab, config, smiles, site_context, full_slab=full_slab
    )
    if info.is_dissociative:
        if info.n_hollow_pairs < 1:
            return 0
    elif not info.site_indices:
        return 0

    return policy.max_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info.site_indices,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
        site_type_for_index=lambda site_idx: _site_type_for_grid(info, site_idx),
        site_is_void_for_index=lambda site_idx: _site_is_void_for_grid(info, site_idx),
    )


def estimate_conformer_count(conformers: list[Atoms]) -> float:
    """Conformer count for budget distribution.

    Returns the number of unique conformers (with a floor of 1),
    directly reflecting the number of conformer-based placement
    enumerations that must be evaluated. Unlike
    :func:`estimate_placement_spec_capacity`, this does not query the
    policy grid and is safe to use for budget allocation without
    affecting capacity clamping.

    Parameters
    ----------
    conformers
        List of adsorbate conformers (already deduplicated).
    """
    return float(max(1, len(conformers)))


def distribute_placement_budget(
    complexities: dict[str, float],
    total_budget: int,
) -> dict[str, int]:
    """Split *total_budget* across molecules in proportion to complexity scores.

    Uses largest-remainder (Hamilton) allocation with a floor of 1 per molecule
    so the returned values always sum to exactly *total_budget*.

    When *total_budget* is smaller than the number of molecules, only the
    top-*total_budget* molecules by complexity receive 1 placement each; the
    rest are omitted (callers already skip molecules missing from the budget).

    Parameters
    ----------
    complexities
        Mapping from molecule name to complexity score.
    total_budget
        Total number of placements to distribute.
    """
    if not complexities:
        return {}
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive, got {total_budget}")

    names = list(complexities)
    n = len(names)
    if total_budget < n:
        # Prefer higher complexity; stable tie-break by name for determinism.
        ranked = sorted(
            names,
            key=lambda name: (-max(1.0, float(complexities[name])), name),
        )
        return {name: 1 for name in ranked[:total_budget]}

    scores = [max(1.0, float(complexities[name])) for name in names]
    total_score = sum(scores)

    # Reserve 1 per molecule, distribute the remainder proportionally.
    remaining = total_budget - n
    exact = [remaining * (s / total_score) for s in scores]
    floors = [int(x) for x in exact]
    allocated = remaining - sum(floors)
    frac_order = sorted(
        range(n),
        key=lambda i: (exact[i] - floors[i], scores[i], -i),
        reverse=True,
    )
    extras = [0] * n
    for k in range(allocated):
        extras[frac_order[k]] += 1

    return {names[i]: 1 + floors[i] + extras[i] for i in range(n)}


def generate_placement_from_spec(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement from spec. Returns (adsorbate, descriptor) or None.

    *slab* may already contain previously placed adsorbates (saturation); when
    it does, *slab_for_sites* must be the bare substrate used for site
    enumeration and ``surface_ref`` resolution.

    Parameters
    ----------
    spec
        Placement specification.
    conformers
        List of adsorbate conformers.
    slab
        Substrate slab (may include pre-adsorbed atoms).
    config
        Adsorption configuration.
    smiles
        Optional SMILES string.
    site_context
        Optional precomputed site context.
    slab_for_sites
        Optional bare substrate for site detection (required when *slab* has
        pre-adsorbates).
    """
    result, _ = generate_placement_from_spec_with_reason(
        spec,
        conformers,
        slab,
        config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
    )
    return result


def generate_placement_from_spec_with_reason(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
    pose_cache: _PoseBatchCache | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Generate placement from spec and provide a failure reason when unavailable.

    *slab* may already contain previously placed adsorbates (saturation); when
    it does, *slab_for_sites* must be the bare substrate used for site
    enumeration and ``surface_ref`` resolution.

    Parameters
    ----------
    spec
        Placement specification.
    conformers
        List of adsorbate conformers.
    slab
        Substrate slab (may include pre-adsorbed atoms).
    config
        Adsorption configuration.
    smiles
        Optional SMILES string.
    site_context
        Optional precomputed site context.
    slab_for_sites
        Optional bare substrate for site detection (required when *slab* has
        pre-adsorbates).
    pose_cache
        Optional per-batch slab/conformer cache.
    """
    if not conformers:
        return None, "no_conformers"
    if spec.conformer_index < 0 or spec.conformer_index >= len(conformers):
        logger.warning(
            "Spec conformer_index=%d out of range for %d conformers",
            spec.conformer_index,
            len(conformers),
        )
        return None, "invalid_conformer_index"

    if spec.orientation_type == "dissociative":
        # Same catalog enumerate indexes: clustered lattice via sampling
        # context, not the adatom-hollow shortcut used when context is omitted.
        reference = slab_for_sites if slab_for_sites is not None else slab
        resolved_ctx = site_context_for_sampling(
            reference,
            config,
            site_context,
            full_slab=slab if slab_for_sites is not None else None,
        )
        adsorbate = conformers[spec.conformer_index].copy()
        return _generate_dissociative_placement_from_spec(
            adsorbate,
            spec,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            site_context=resolved_ctx,
        )

    adsorbate = conformers[spec.conformer_index].copy()

    placement_ctx, pose_fail = _pose_from_spec(
        adsorbate,
        spec,
        slab,
        config,
        smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
        pose_cache=pose_cache,
    )
    if placement_ctx is None:
        return None, pose_fail

    result, fail_reason = _finalize_placement(
        placement_ctx,
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        allow_distance_recovery=True,
        pose_cache=pose_cache,
    )
    if result is not None:
        return result, None
    return None, fail_reason


__all__ = [
    "distribute_placement_budget",
    "enumerate_placement_specs",
    "estimate_conformer_count",
    "estimate_placement_spec_capacity",
    "generate_placement_from_spec",
    "generate_placement_from_spec_with_reason",
    "generate_placements_from_specs",
    "resolve_materialize_workers",
]

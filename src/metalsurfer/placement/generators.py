"""Build placements from specs: sites, orientations, and validation.

Public orchestration façade (enumerate, materialize, complexity/budget).
Private helpers live in ``dissociative``, ``orientation``, ``pose``,
and ``site_context`` — import those modules directly in tests.
"""

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from . import geometry as geom
from . import policy
from ._constants import (
    _PORE_SITE_CAP_FLOOR,
    _PORE_SITE_CAP_MULTIPLIER,
    _PORE_SITE_CAP_NUM_PLACEMENTS_DEFAULT,
    _POROUS_SITE_INDEX_WEIGHT,
)
from ._material import material_aware_pbc
from ._parallel import resolve_materialize_workers
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
    """
    if not specs:
        return []

    placement_ref = slab_for_sites if slab_for_sites is not None else slab
    pose_cache = None
    if materialization_cache is None or any(
        materialization_cache.get(int(spec.placement_index)) is None for spec in specs
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
    # Per-molecule view: which (conformer_index, site_index) pairs fit under
    # coverage. Does not mutate the shared SiteContext catalog.
    allows_conformer_site: Callable[[int, int], bool] | None = None


def _topology_first_site_indices(
    sites: list[Site],
    indices: list[int],
    *,
    clearances: np.ndarray | None = None,
) -> list[int]:
    """Order *indices*: topology / atop-injected sources first, then clearance.

    Homogeneous catalogs (e.g. pure ``adaptive_grid``) ignore the source key
    and sort only by footprint clearance. This is sampling policy, not a
    uniqueness pass.
    """

    def _rank(i: int) -> tuple[int, float, int]:
        site = sites[i]
        src = str(site.site_source)
        prefer = 0 if src.startswith("topology") or src == "atop_injected" else 1
        # Larger clearance first → negate for ascending sort.
        clear = (
            -float(clearances[i])
            if clearances is not None and i < len(clearances)
            else 0.0
        )
        return (prefer, clear, i)

    return sorted(indices, key=_rank)


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
        and config.material_type in ("slab", "nanoparticle")
        and _is_dissociable_diatomic(conformers[0])
    )
    _ctx = site_context_for_sampling(slab, config, site_context, full_slab=full_slab)
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    existing_ads_pos, existing_radii = existing_adsorbate_cloud(
        slab,
        full_slab,
        min_separation=float(config.min_adsorbate_separation),
    )

    footprint_scale = float(config.occupancy_footprint_scale)
    conf_radii = [
        incoming_inplane_radius(conf, footprint_scale=footprint_scale)
        for conf in conformers
    ]
    # (ci, si) pairs allowed by this molecule's footprint under coverage.
    allowed_pairs: set[tuple[int, int]] | None = None
    allows_conformer_site: Callable[[int, int], bool] | None = None

    if use_sites and unique_sites:
        if existing_ads_pos is None or np.asarray(existing_ads_pos).size == 0:
            site_indices = list(range(len(unique_sites)))
            clearances = np.full(len(unique_sites), np.inf, dtype=float)
        else:
            need_footprint = (
                config.occupancy_use_footprint
                and existing_radii is not None
                and any(r > 0.0 for r in conf_radii)
            )
            anchor_mask, min_dists, mic_vecs = _sites_clearance_and_anchor_mask(
                unique_sites,
                existing_ads_pos,
                cell=cell_arr,
                pbc=pbc,
                min_separation=float(config.min_adsorbate_separation),
                need_mic_vecs=need_footprint,
            )
            anchor_ok = {i for i, keep in enumerate(anchor_mask) if keep}
            if need_footprint:
                assert existing_radii is not None
                # Per-conformer footprint clearances on the shared MIC grid.
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
                                existing_radii,
                                float(r_in),
                            )
                        )
                # Site stays if any conformer disk clears (clearance >= 0).
                site_indices = []
                allowed_pairs = set()
                rank_clear = np.full(len(unique_sites), -np.inf, dtype=float)
                for si in sorted(anchor_ok):
                    fitted = False
                    best = -np.inf
                    for ci, clear_arr in enumerate(per_conf_clear):
                        c = float(clear_arr[si])
                        if c >= 0.0:
                            allowed_pairs.add((ci, si))
                            fitted = True
                            if c > best:
                                best = c
                    if fitted:
                        site_indices.append(si)
                        rank_clear[si] = best
                clearances = rank_clear
                _allowed_pairs = allowed_pairs

                def allows_conformer_site(ci: int, si: int) -> bool:
                    return (int(ci), int(si)) in _allowed_pairs

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
            allows_conformer_site = None
        else:
            site_indices = _topology_first_site_indices(
                unique_sites,
                site_indices,
                clearances=clearances,
            )
            if config.material_type == "porous":
                # Draw preference only — shared SiteContext.sites is unchanged.
                pore_indices = [
                    i for i in site_indices if str(unique_sites[i].site_type) == "pore"
                ]
                if pore_indices:
                    pore_indices.sort(
                        key=lambda i: -float(unique_sites[i].nn_distance or 0.0)
                    )
                    pore_cap = max(
                        int(
                            config.num_placements
                            or _PORE_SITE_CAP_NUM_PLACEMENTS_DEFAULT
                        )
                        * _PORE_SITE_CAP_MULTIPLIER,
                        _PORE_SITE_CAP_FLOOR,
                    )
                    site_indices = pore_indices[:pore_cap]
    else:
        site_indices = []
        use_sites = False

    conf0_pos = conformers[0].get_positions()
    ads_pos = conf0_pos - np.mean(conf0_pos, axis=0)
    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    symbols = conformers[0].get_chemical_symbols()
    binders = geom._binding_atom_candidates(symbols)
    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)

    n_hollow_pairs = 0
    if is_dissociative:
        working_slab = full_slab if full_slab is not None else slab
        n_hollow_pairs = len(
            _get_dissociative_site_pairs(
                working_slab,
                config,
                slab_for_sites=slab,
                existing_adsorbate_positions=existing_ads_pos,
                site_context=_ctx,
            )
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
        allows_conformer_site=allows_conformer_site,
    )


def _site_type_for_grid(info: _SpecGridInfo, site_idx: int) -> str | None:
    """Site type for a grid index, or ``None`` when the catalog has no site."""
    if info.is_dissociative:
        return "hollow"
    sites = info.unique_sites
    if not info.use_sites or site_idx < 0 or site_idx >= len(sites):
        return None
    return str(sites[site_idx].site_type)


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

    prefer_pores = config.material_type == "porous"
    return policy.build_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info.site_indices,
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
        preferred_site_types=("pore",) if prefer_pores else (),
        # Quality-sorted pore lists: keep open pores near the front of the draw.
        site_index_weight=(_POROUS_SITE_INDEX_WEIGHT if prefer_pores else 0.0),
        conformer_energies=conformer_energies,
        conformer_weighting=config.conformer_weighting,
        boltzmann_temperature=config.boltzmann_temperature,
        allows_conformer_site=info.allows_conformer_site,
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
    )


estimate_placement_capacity = estimate_placement_spec_capacity


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
        adsorbate = conformers[spec.conformer_index].copy()
        return _generate_dissociative_placement_from_spec(
            adsorbate,
            spec,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            site_context=site_context,
        )

    # Pass the catalog through unchanged. Pose resolves only when omitted.
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
    "estimate_placement_capacity",
    "estimate_placement_spec_capacity",
    "generate_placement_from_spec",
    "generate_placement_from_spec_with_reason",
    "generate_placements_from_specs",
    "resolve_materialize_workers",
]

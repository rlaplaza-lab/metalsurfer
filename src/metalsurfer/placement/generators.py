"""Build placements from specs: sites, orientations, and validation.

Public orchestration façade (enumerate, materialize, replay, complexity/budget).
Private helpers live in ``dissociative``, ``orientation``, ``pose``,
and ``site_context`` — import those modules directly in tests.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec
from . import geometry as geom
from . import policy
from ._material import material_aware_pbc
from .dissociative import (
    _generate_dissociative_placement_from_spec,
    _get_dissociative_site_pairs,
    _is_dissociable_diatomic,
)
from .occupancy import (
    available_site_indices,
    existing_adsorbate_positions,
)
from .orientation import (
    _estimate_parallel_fraction,
    _is_flat_aromatic,
)
from .pose import (
    _finalize_placement,
    _pose_from_descriptor,
    _pose_from_spec,
    generate_placement_from_pose,
)
from .site_context import (
    SiteContext,
    _get_unique_sites_for_specs,
)
from .site_types import Site

logger = logging.getLogger(__name__)


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


def _topology_first_site_indices(sites: list[Site], indices: list[int]) -> list[int]:
    """Order *indices* so topology-sourced sites come before Voronoi-only ones."""

    def _rank(i: int) -> tuple[int, int]:
        site = sites[i]
        src = str(site.site_source)
        prefer = 0 if src.startswith("topology") or src == "atop_injected" else 1
        return (prefer, i)

    return sorted(indices, key=_rank)


def _spec_grid_info(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None,
    full_slab: Atoms | None = None,
) -> _SpecGridInfo:
    """Compute the spec-enumeration inputs once for both enumerate and estimate."""
    is_dissociative = (
        (config.enable_dissociative_placement or config.skip_topology_check)
        and config.material_type in ("slab", "nanoparticle")
        and _is_dissociable_diatomic(conformers[0])
    )
    if (
        is_dissociative
        and config.skip_topology_check
        and not config.enable_dissociative_placement
    ):
        warnings.warn(
            "skip_topology_check=True enabling dissociative placement is deprecated; "
            "set enable_dissociative_placement=True (and keep skip_topology_check=True "
            "if connectivity filters should still be skipped).",
            DeprecationWarning,
            stacklevel=3,
        )
    _ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    existing_ads_pos = existing_adsorbate_positions(slab, full_slab)
    if use_sites and unique_sites:
        site_indices = available_site_indices(
            unique_sites,
            existing_ads_pos,
            cell=np.asarray(slab.get_cell(), dtype=float),
            pbc=material_aware_pbc(config.material_type),
            min_separation=float(config.min_initial_distance),
        )
        if not site_indices:
            logger.warning(
                "Occupancy pruning removed all %d sites under coverage; "
                "no site-based placement specs will be generated",
                len(unique_sites),
            )
            # Empty capacity: no random XY fallback under coverage.
            site_indices = []
            use_sites = False
        else:
            site_indices = _topology_first_site_indices(unique_sites, site_indices)
    else:
        # No sites / use_sites=False: empty capacity (random-XY fallback removed).
        site_indices = []
        use_sites = False

    ads_pos = conformers[0].get_positions() - np.mean(
        conformers[0].get_positions(), axis=0
    )
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
    )


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
) -> list[PlacementSpec]:
    """Enumerate placement specs for diverse sampling."""
    if not conformers:
        return []

    eff_seed = config.seed if seed is None else seed
    info = _spec_grid_info(
        conformers, slab, config, smiles, site_context, full_slab=full_slab
    )
    unique_sites = info.unique_sites
    use_sites = info.use_sites

    if info.is_dissociative:
        if info.n_hollow_pairs < 1:
            return []
    elif not info.site_indices:
        # Occupancy pruned all sites (empty list) — do not fall back to random XY.
        return []

    def site_type_for(site_idx: int) -> str | None:
        if info.is_dissociative:
            return "hollow"
        if not use_sites or site_idx < 0 or site_idx >= len(unique_sites):
            return None
        return str(unique_sites[site_idx].site_type)

    parallel_fraction = config.flat_aromatic_parallel_fraction
    if config.adaptive_parallel_fraction and info.flat_aromatic:
        parallel_fraction = _estimate_parallel_fraction(info.symbols, smiles)

    return policy.build_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info.site_indices,
        site_type_for_index=site_type_for,
        shape=info.shape,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        parallel_fraction=parallel_fraction,
        n_desired=n_desired,
        filter_spec=filter_spec,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
        seed=eff_seed,
    )


def estimate_placement_spec_capacity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
    full_slab: Atoms | None = None,
) -> int:
    """Estimate total enumerated specs for current conformers/site grid."""
    if not conformers:
        return 0
    info = _spec_grid_info(
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
        shape=info.shape,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
    )


def estimate_molecule_complexity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
    full_slab: Atoms | None = None,
) -> float:
    """Capacity-based placement-space score for budget allocation.

    Delegates to :func:`estimate_placement_spec_capacity` (policy-grid size).
    When *full_slab* is provided, capacity reflects occupancy-pruned sites on
    that structure (site detection still uses *slab* / *site_context*).
    Returns ``0.0`` when pruning leaves no available sites (callers should skip
    budgeting that molecule for the step).
    """
    capacity = estimate_placement_spec_capacity(
        conformers,
        slab,
        config,
        smiles,
        site_context=site_context,
        full_slab=full_slab,
    )
    if capacity <= 0:
        return 0.0
    return max(1.0, float(capacity))


def distribute_placement_budget(
    complexities: dict[str, float],
    total_budget: int,
) -> dict[str, int]:
    """Split *total_budget* across molecules in proportion to complexity scores.

    Uses largest-remainder (Hamilton) allocation with a floor of 1 per molecule
    so the returned values always sum to exactly *total_budget*.
    """
    if not complexities:
        return {}
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive, got {total_budget}")

    names = list(complexities)
    n = len(names)
    if total_budget < n:
        raise ValueError(
            f"total_budget ({total_budget}) must be >= number of molecules "
            f"({n}); cannot guarantee every molecule at least 1 placement"
        )

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
    """Generate adsorbate placement from spec. Returns (adsorbate, descriptor) or None."""
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
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Generate placement from spec and provide a failure reason when unavailable."""
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

    resolved_ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )

    adsorbate = conformers[spec.conformer_index].copy()

    placement_ctx, pose_fail = _pose_from_spec(
        adsorbate,
        spec,
        slab,
        config,
        smiles,
        site_context=resolved_ctx,
        slab_for_sites=slab_for_sites,
    )
    if placement_ctx is None:
        return None, pose_fail or "no_sites_found"

    result, fail_reason = _finalize_placement(
        placement_ctx,
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        allow_distance_recovery=True,
    )
    if result is not None:
        return result, None
    return None, fail_reason or "distance_check_failed"


def generate_placement_from_descriptor(
    descriptor: PlacementDescriptor,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
) -> Atoms | None:
    """Reproduce placement deterministically from descriptor."""
    _ = smiles
    if not conformers:
        return None
    if descriptor.conformer_index < 0 or descriptor.conformer_index >= len(conformers):
        logger.warning(
            "Descriptor conformer_index=%d out of range for %d conformers",
            descriptor.conformer_index,
            len(conformers),
        )
        return None

    if descriptor.orientation_type == "dissociative":
        if descriptor.fragment_positions is None:
            logger.warning(
                "Dissociative descriptor missing fragment_positions; cannot replay"
            )
            return None
        adsorbate = conformers[descriptor.conformer_index].copy()
        if len(descriptor.fragment_positions) != len(adsorbate):
            logger.warning(
                "Dissociative fragment_positions length %d != adsorbate atoms %d",
                len(descriptor.fragment_positions),
                len(adsorbate),
            )
            return None
        adsorbate.set_positions(np.asarray(descriptor.fragment_positions, dtype=float))
        adsorbate.set_cell(slab.get_cell())
        adsorbate.set_pbc(slab.get_pbc())
        return adsorbate

    if descriptor.x_abs is None or descriptor.y_abs is None or descriptor.z_abs is None:
        logger.warning(
            "Descriptor replay requires x_abs, y_abs, and z_abs; got x_abs=%s y_abs=%s z_abs=%s",
            descriptor.x_abs,
            descriptor.y_abs,
            descriptor.z_abs,
        )
        return None
    if None in (
        descriptor.quat_w,
        descriptor.quat_x,
        descriptor.quat_y,
        descriptor.quat_z,
    ):
        logger.warning("Descriptor replay requires quaternion components")
        return None
    pose = _pose_from_descriptor(descriptor)
    result = generate_placement_from_pose(
        pose, conformers, slab, config, site_context=site_context
    )
    if result is None:
        return None
    adsorbate, _ = result
    return adsorbate


__all__ = [
    "distribute_placement_budget",
    "enumerate_placement_specs",
    "estimate_molecule_complexity",
    "estimate_placement_spec_capacity",
    "generate_placement_from_descriptor",
    "generate_placement_from_spec",
    "generate_placement_from_spec_with_reason",
]

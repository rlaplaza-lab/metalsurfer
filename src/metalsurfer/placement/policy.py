"""Sampler policy for batch-based BO placement spec proposals."""

import itertools
import logging
import math
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from ..models import PlacementSpec
from ._constants import (
    _AZIMUTH,
    _AZIMUTH_IN_PLANE,
    _EARLY_CAP_WORKING_SET_MULTIPLIER,
    _GRID_BUILD_CAP,
    _PLACEMENT_GRID_COUNT_SEED,
    _POLICY_PRIOR_TILT_WEIGHT_PER_DEG,
    _POLICY_PRIOR_Z_FRACTION_TARGET,
    _POLICY_PRIOR_Z_FRACTION_WEIGHT,
    _TILT_FULL,
    _TILT_PARALLEL,
    _Z_FRACTIONS,
)

logger = logging.getLogger(__name__)

# Boltzmann constant in eV/K (conformer energies are in eV).
_K_B_EV_PER_K: float = 8.617e-5


def _unravel_product_index(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Decode a flat index into ``itertools.product`` coordinates (last axis fastest)."""
    coords: list[int] = []
    rem = int(flat)
    for size in reversed(shape):
        coords.append(rem % size)
        rem //= size
    return tuple(reversed(coords))


def _parallel_aip_values(tilt: float) -> tuple[float, ...]:
    """In-plane azimuth values at *tilt* for parallel placements.

    At tilt=0, azimuth and azimuth_in_plane both rotate about the surface
    normal and commute, so only one in-plane angle (0) is kept. Single source
    of truth for both :func:`max_batch_placement_specs` and the flat-aromatic
    builder branch.
    """
    return (0.0,) if float(tilt) == 0.0 else _AZIMUTH_IN_PLANE


def _flat_aromatic_branch_capacities(
    *,
    n_conformers: int,
    n_sites: int,
    n_binders: int,
) -> tuple[int, int]:
    """Uncapped ``(parallel, EN-down)`` flat-aromatic grid sizes.

    Derived from exactly the axis products
    :func:`build_batch_placement_specs` enumerates in its flat-aromatic
    branches (including the ``_parallel_aip_values`` collapse), so budget
    estimates cannot silently drift from the builder.
    """
    tilt_aip_pairs = sum(len(_parallel_aip_values(tl)) for tl in _TILT_PARALLEL)
    parallel = (
        n_conformers * 2 * tilt_aip_pairs * len(_AZIMUTH) * len(_Z_FRACTIONS) * n_sites
    )
    en_down = (
        n_conformers
        * max(n_binders, 1)
        * len(_TILT_FULL)
        * len(_AZIMUTH)
        * len(_Z_FRACTIONS)
        * n_sites
    )
    return parallel, en_down


def max_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    n_binders: int,
    flat_aromatic: bool,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
) -> int:
    """Closed-form count of policy-grid specs (per-branch clamp at ``_GRID_BUILD_CAP``).

    Mirrors the *uncapped* grids of :func:`build_batch_placement_specs`; the
    builder additionally applies working-set caps (``filter_spec``, early-cap
    multiplier), so the estimate is an upper bound per branch.

    Parameters
    ----------
    n_conformers
        Number of conformers.
    site_indices
        List of available site indices.
    n_binders
        Number of binding atoms.
    flat_aromatic
        Whether the molecule is flat and aromatic.
    dissociative
        Whether dissociative placement is enabled.
    n_hollow_pairs
        Number of hollow site pairs for dissociative placement.
    """
    n_sites = max(len(site_indices), 1)

    if dissociative:
        if n_hollow_pairs <= 0:
            return 0
        return min(n_hollow_pairs * len(_Z_FRACTIONS), _GRID_BUILD_CAP)

    if flat_aromatic:
        parallel, en_down = _flat_aromatic_branch_capacities(
            n_conformers=n_conformers, n_sites=n_sites, n_binders=n_binders
        )
        return min(parallel, _GRID_BUILD_CAP) + min(en_down, _GRID_BUILD_CAP)

    return min(
        n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * len(_Z_FRACTIONS) * n_sites,
        _GRID_BUILD_CAP,
    )


def _spec_prior_key(
    spec: PlacementSpec,
    tie: float,
    *,
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
) -> tuple[float, float]:
    """Sort key for soft physical priors: lower score is preferred.

    Prefers milder absolute tilt and ``z_fraction`` near *z_fraction_target*.
    *tie* breaks remaining ties deterministically (seeded shuffle rank).
    Site-index diversity for porous materials is handled by by-site round-robin
    in :func:`_stratified_sample`, not here.
    """
    tilt_pen = abs(float(spec.tilt_deg)) * _POLICY_PRIOR_TILT_WEIGHT_PER_DEG
    z_pen = (
        abs(float(spec.z_fraction) - float(z_fraction_target))
        * _POLICY_PRIOR_Z_FRACTION_WEIGHT
    )
    return (tilt_pen + z_pen, tie)


def _boltzmann_weights(
    energies: list[float],
    temperature: float,
) -> list[float] | None:
    """Deterministic Boltzmann weights ``exp(-(E_i - E_min) / (k_B * T))``.

    Returns ``None`` when weighting is not meaningful (non-positive/non-finite
    *temperature*, fewer than two finite energies, or all finite energies equal),
    which the caller treats as "fall back to the uniform draw". Non-finite
    entries get weight ``0.0``: they are not dropped, they simply sort behind
    every finite conformer in the proportional allocation.
    """
    if not math.isfinite(temperature) or temperature <= 0.0:
        return None

    finite = [(i, float(e)) for i, e in enumerate(energies) if math.isfinite(e)]
    if len(finite) < 2:
        return None

    finite_values = [e for _i, e in finite]
    e_min = min(finite_values)
    if max(finite_values) - e_min <= 0.0:
        # Degenerate (all equal, e.g. unscored conformers): uniform is exact.
        return None

    kt = _K_B_EV_PER_K * temperature
    weights = [0.0] * len(energies)
    for i, energy in finite:
        # Exponent is <= 0 by construction, so exp() cannot overflow.
        weights[i] = math.exp(-(energy - e_min) / kt)
    if sum(weights) <= 0.0:
        return None
    return weights


def resolve_conformer_weights(
    *,
    n_conformers: int,
    conformer_energies: list[float] | None,
    conformer_weighting: str = "uniform",
    boltzmann_temperature: float = 300.0,
) -> list[float] | None:
    """Resolve the per-conformer prior, or ``None`` for the uniform draw.

    Logs the reason whenever ``"boltzmann"`` was requested but the inputs cannot
    support it, so a silently uniform run is always explained.

    Parameters
    ----------
    n_conformers
        Number of conformers.
    conformer_energies
        Optional list of conformer energies (eV).
    conformer_weighting
        Weighting scheme (``"uniform"`` or ``"boltzmann"``).
    boltzmann_temperature
        Temperature for Boltzmann weighting (K).
    """
    if conformer_weighting != "boltzmann":
        return None
    if n_conformers < 2:
        return None
    if conformer_energies is None:
        logger.warning(
            "conformer_weighting='boltzmann' but no conformer energies were "
            "supplied; using the uniform conformer draw"
        )
        return None
    if len(conformer_energies) != n_conformers:
        logger.warning(
            "conformer_weighting='boltzmann': %d energies for %d conformers; "
            "using the uniform conformer draw",
            len(conformer_energies),
            n_conformers,
        )
        return None
    if not all(math.isfinite(float(e)) for e in conformer_energies):
        logger.warning(
            "conformer_weighting='boltzmann': non-finite conformer energies; "
            "affected conformers are de-prioritised (weight 0), not dropped"
        )

    weights = _boltzmann_weights(conformer_energies, boltzmann_temperature)
    if weights is None:
        logger.warning(
            "conformer_weighting='boltzmann': energies are degenerate or "
            "unusable (T=%.3g K); using the uniform conformer draw",
            boltzmann_temperature,
        )
    return weights


def _weighted_conformer_order(
    ordered_best_first: list[PlacementSpec],
    weights: list[float],
    limit: int,
) -> list[PlacementSpec]:
    """Interleave *ordered_best_first* across conformers proportionally to *weights*.

    Largest-remainder (Hamilton) allocation applied at every prefix: pick the
    conformer with the largest deficit ``share_i * k - count_i`` at draw ``k``,
    ties going to the lower ``conformer_index``. Every prefix of the result is
    therefore an exact proportional apportionment (per-conformer error < 1 slot),
    and relative order *within* a conformer is preserved, so the existing prior +
    seeded tie-break still decides which spec of that conformer comes next.
    The step is RNG-free: same inputs, same output.

    Only the first *limit* specs are built; the caller never draws more than that
    from one bucket.
    """
    limit = min(limit, len(ordered_best_first))
    if limit <= 0:
        return []

    groups: dict[int, list[PlacementSpec]] = defaultdict(list)
    for spec in ordered_best_first:
        groups[int(spec.conformer_index)].append(spec)
    if len(groups) < 2:
        return ordered_best_first[:limit]

    keys = sorted(groups)
    total = sum(max(weights[key], 0.0) for key in keys)
    if total <= 0.0:
        return ordered_best_first[:limit]
    shares = {key: max(weights[key], 0.0) / total for key in keys}
    counts = dict.fromkeys(keys, 0)
    cursors = dict.fromkeys(keys, 0)

    out: list[PlacementSpec] = []
    while len(out) < limit:
        draw = len(out) + 1
        best_key: int | None = None
        best_deficit = 0.0
        for key in keys:
            if cursors[key] >= len(groups[key]):
                continue
            deficit = shares[key] * draw - counts[key]
            if best_key is None or deficit > best_deficit:
                best_key = key
                best_deficit = deficit
        if best_key is None:
            break
        out.append(groups[best_key][cursors[best_key]])
        cursors[best_key] += 1
        counts[best_key] += 1
    return out


def _stratified_sample(
    specs: list[PlacementSpec],
    n_desired: int,
    seed: int,
    *,
    preferred_site_types: tuple[str, ...] = (),
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
    conformer_weights: list[float] | None = None,
) -> list[PlacementSpec]:
    """Sample up to *n_desired* specs stratified by ``site_type`` (seeded, deterministic).

    Within each site-type bucket, specs are ordered by a soft prior (milder tilt,
    ``z_fraction`` near *z_fraction_target*) with a seeded tie-break so draws
    remain deterministic. When *preferred_site_types* is set (e.g. ``("pore",)``
    for porous frameworks), those buckets are drawn first each round-robin pass.
    When *site_index_weight* > 0, draws also round-robin across ``site_index``
    values (quality-sorted) so open-pore lists keep multi-site coverage.
    When *conformer_weights* is set, each bucket is additionally interleaved
    across ``conformer_index`` in proportion to those weights
    (:func:`_weighted_conformer_order`); ``None`` keeps the conformer-agnostic
    ordering unchanged.
    """
    if len(specs) <= n_desired:
        return list(specs)

    buckets: dict[str, list[PlacementSpec]] = defaultdict(list)
    for spec in specs:
        key = str(spec.site_type) if spec.site_type is not None else "none"
        buckets[key].append(spec)

    weights = conformer_weights
    if weights is not None and not all(
        0 <= int(spec.conformer_index) < len(weights) for spec in specs
    ):
        logger.warning(
            "conformer weighting skipped: spec conformer_index out of range for "
            "%d weights; using the uniform conformer draw",
            len(weights),
        )
        weights = None

    # Preferred buckets first, then remaining keys sorted for determinism.
    preferred = [k for k in preferred_site_types if k in buckets]
    keys = preferred + [k for k in sorted(buckets) if k not in preferred]
    rng = random.Random(seed)
    for key in keys:
        bucket = buckets[key]
        ranks = list(range(len(bucket)))
        rng.shuffle(ranks)
        if site_index_weight > 0.0:
            # Round-robin across site indices so quality bias does not collapse
            # the draw onto a single open pore. The unused bucket shuffle above
            # keeps the RNG sequence aligned with the non-porous branch.
            by_site: dict[int, list[PlacementSpec]] = defaultdict(list)
            for spec in bucket:
                by_site[int(spec.site_index)].append(spec)
            ranked_by_site: dict[int, list[PlacementSpec]] = {}
            for si, site_specs in by_site.items():
                site_ranks = list(range(len(site_specs)))
                rng.shuffle(site_ranks)
                ordered = sorted(
                    zip(site_specs, site_ranks, strict=True),
                    key=lambda item: _spec_prior_key(
                        item[0],
                        float(item[1]),
                        z_fraction_target=z_fraction_target,
                    ),
                )
                # Best first for round-robin.
                ranked_by_site[si] = [spec for spec, _ in ordered]
            site_order = sorted(ranked_by_site.keys())
            best_first: list[PlacementSpec] = []
            while True:
                progressed = False
                for si in site_order:
                    site_bucket = ranked_by_site[si]
                    if site_bucket:
                        best_first.append(site_bucket.pop(0))
                        progressed = True
                if not progressed:
                    break
        else:
            ordered = sorted(
                zip(bucket, ranks, strict=True),
                key=lambda item: _spec_prior_key(
                    item[0],
                    float(item[1]),
                    z_fraction_target=z_fraction_target,
                ),
            )
            best_first = [spec for spec, _ in ordered]

        if weights is not None:
            best_first = _weighted_conformer_order(best_first, weights, n_desired)
        # Pop from end → reverse so preferred (earlier) specs come off first.
        buckets[key] = list(reversed(best_first))

    selected: list[PlacementSpec] = []
    # Round-robin across buckets until n_desired (preferred keys first each pass).
    while len(selected) < n_desired:
        progressed = False
        for key in keys:
            if len(selected) >= n_desired:
                break
            bucket = buckets[key]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    site_type_for_index: Callable[[int], str | None],
    shape: str,
    n_binders: int,
    flat_aromatic: bool,
    parallel_fraction: float,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
    seed: int = _PLACEMENT_GRID_COUNT_SEED,
    preferred_site_types: tuple[str, ...] = (),
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
    conformer_energies: list[float] | None = None,
    conformer_weighting: str = "uniform",
    boltzmann_temperature: float = 300.0,
) -> list[PlacementSpec]:
    """BO candidate ``PlacementSpec`` list: full Cartesian grid (capped), then stratified subsample to *n_desired* (*seed*).

    With ``conformer_weighting="boltzmann"`` and *conformer_energies* supplied,
    the subsample allocates spec slots per ``conformer_index`` in proportion to
    ``exp(-(E_i - E_min) / (k_B * boltzmann_temperature))``. The allocation is
    deterministic (no RNG) and degrades to the uniform draw whenever the
    energies cannot support weighting. The dissociative branch pins
    ``conformer_index=0``, so it is never weighted.

    Parameters
    ----------
    n_conformers
        Number of conformers.
    site_indices
        List of available site indices.
    site_type_for_index
        Callable mapping site index to site type string.
    shape
        Molecule shape classification.
    n_binders
        Number of binding atoms.
    flat_aromatic
        Whether the molecule is flat and aromatic.
    parallel_fraction
        Fraction of placements to orient parallel.
    n_desired
        Target number of specs in the returned list.
    filter_spec
        Optional callable to filter generated specs.
    dissociative
        Whether dissociative placement is enabled.
    n_hollow_pairs
        Number of hollow site pairs for dissociative placement.
    seed
        Random seed for deterministic subsampling.
    preferred_site_types
        Site types to prioritize in stratified sampling.
    z_fraction_target
        Preferred z-fraction for prior ordering.
    site_index_weight
        Weight for site-index prior penalty.
    conformer_energies
        Optional list of conformer energies (eV).
    conformer_weighting
        Conformer weighting scheme (``"uniform"`` or ``"boltzmann"``).
    boltzmann_temperature
        Temperature for Boltzmann weighting (K).
    """
    normalized_sites = site_indices if site_indices else [-1]
    conformer_weights = (
        None
        if dissociative
        else resolve_conformer_weights(
            n_conformers=n_conformers,
            conformer_energies=conformer_energies,
            conformer_weighting=conformer_weighting,
            boltzmann_temperature=boltzmann_temperature,
        )
    )
    base_fields = {
        "face_flip": False,
        "en_atom_index": None,
        "azimuth_in_plane_deg": 0.0,
    }

    def _fields(**overrides: Any) -> dict[str, Any]:
        return {**base_fields, **overrides}

    def _collect(
        items: Iterable[dict[str, Any]],
        cap: int,
    ) -> list[PlacementSpec]:
        """Collect specs from *items* until *cap*, applying ``filter_spec`` when set."""
        out: list[PlacementSpec] = []
        for fields in items:
            if len(out) >= cap:
                break
            spec = PlacementSpec(
                **fields,
                site_type=site_type_for_index(int(fields["site_index"])),
                placement_index=0,
            )
            if filter_spec is None or filter_spec(spec):
                out.append(spec)
        return out

    def _subsample(
        specs: list[PlacementSpec], n: int, sub_seed: int
    ) -> list[PlacementSpec]:
        return _stratified_sample(
            specs,
            n,
            sub_seed,
            preferred_site_types=preferred_site_types,
            z_fraction_target=z_fraction_target,
            site_index_weight=site_index_weight,
            conformer_weights=conformer_weights,
        )

    if dissociative:
        if n_hollow_pairs <= 0:
            specs = []
        else:
            # Shuffle pair indices before expanding z so early-cap is not
            # biased toward the first pairs in enumeration order.
            pair_indices = list(range(n_hollow_pairs))
            random.Random(seed).shuffle(pair_indices)
            items = (
                _fields(
                    conformer_index=0,
                    orientation_type="dissociative",
                    site_index=pair_idx,
                    tilt_deg=0.0,
                    azimuth_deg=0.0,
                    z_fraction=zfv,
                )
                for pair_idx, zfv in itertools.product(pair_indices, _Z_FRACTIONS)
            )
            working_cap = min(
                _GRID_BUILD_CAP,
                max(n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER, n_desired),
            )
            specs = _collect(items, cap=working_cap)
            if len(specs) > n_desired:
                specs = _subsample(specs, n_desired, seed)
    elif flat_aromatic:
        n_par = int(round(n_desired * parallel_fraction))
        n_par = max(0, min(n_par, n_desired))
        n_en = n_desired - n_par

        # These two branches cap the working set early (``*_cap`` below), so the
        # conformer axis must vary fastest under weighting: with it outermost an
        # early cap would truncate the working set to the first conformer(s) and
        # leave the proportional allocation nothing to allocate over.
        parallel_axes: Iterable[tuple[Any, ...]]
        if conformer_weights is None:
            parallel_axes = (
                (ci, ff, tl, azv, zfv, aip, si)
                for ci, ff, tl, azv, zfv, si in itertools.product(
                    range(n_conformers),
                    [False, True],
                    _TILT_PARALLEL,
                    _AZIMUTH,
                    _Z_FRACTIONS,
                    normalized_sites,
                )
                for aip in _parallel_aip_values(tl)
            )
        else:
            parallel_axes = (
                (ci, ff, tl, azv, zfv, aip, si)
                for ff, tl, azv, zfv, si, ci in itertools.product(
                    [False, True],
                    _TILT_PARALLEL,
                    _AZIMUTH,
                    _Z_FRACTIONS,
                    normalized_sites,
                    range(n_conformers),
                )
                for aip in _parallel_aip_values(tl)
            )
        parallel_items = (
            _fields(
                conformer_index=ci,
                orientation_type="parallel",
                face_flip=ff,
                site_index=si,
                tilt_deg=tl,
                azimuth_deg=azv,
                azimuth_in_plane_deg=aip,
                z_fraction=zfv,
            )
            for ci, ff, tl, azv, zfv, aip, si in parallel_axes
        )

        en_down_axes: Iterable[tuple[Any, ...]]
        if conformer_weights is None:
            en_down_axes = itertools.product(
                range(n_conformers),
                range(max(n_binders, 1)),
                _TILT_FULL,
                _AZIMUTH,
                _Z_FRACTIONS,
                normalized_sites,
            )
        else:
            en_down_axes = (
                (ci, ei, tl, azv, zfv, si)
                for ei, tl, azv, zfv, si, ci in itertools.product(
                    range(max(n_binders, 1)),
                    _TILT_FULL,
                    _AZIMUTH,
                    _Z_FRACTIONS,
                    normalized_sites,
                    range(n_conformers),
                )
            )
        en_down_items = (
            _fields(
                conformer_index=ci,
                orientation_type="EN-down",
                en_atom_index=ei if n_binders > 1 else None,
                site_index=si,
                tilt_deg=tl,
                azimuth_deg=azv,
                z_fraction=zfv,
            )
            for ci, ei, tl, azv, zfv, si in en_down_axes
        )

        # Size each working set for full n_desired so a filtered-out branch can
        # be topped up from the other side's surplus.
        shared_cap = min(
            _GRID_BUILD_CAP,
            max(n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER, n_desired),
        )
        par_pool = _collect(parallel_items, cap=shared_cap)
        en_pool = _collect(en_down_items, cap=shared_cap)

        n_par_take = min(n_par, len(par_pool))
        n_en_take = min(n_en, len(en_pool))
        deficit = n_desired - (n_par_take + n_en_take)
        if deficit > 0:
            par_extra = min(deficit, len(par_pool) - n_par_take)
            n_par_take += par_extra
            deficit -= par_extra
            n_en_take += min(deficit, len(en_pool) - n_en_take)

        par_specs = _subsample(par_pool, n_par_take, seed) if n_par_take > 0 else []
        en_specs = _subsample(en_pool, n_en_take, seed + 1) if n_en_take > 0 else []
        specs = par_specs + en_specs
    else:
        orient = "vertical" if shape == "linear" else "round"
        site_list = list(normalized_sites)
        n_sites = len(site_list)
        n_tilt = len(_TILT_FULL)
        n_az = len(_AZIMUTH)
        n_z = len(_Z_FRACTIONS)
        n_total = n_conformers * n_tilt * n_az * n_z * n_sites
        working_cap = min(
            _GRID_BUILD_CAP,
            max(n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER, n_desired),
            n_total if n_total > 0 else 0,
        )
        # Uniform sample avoids early-cap bias toward the product prefix.
        rng = random.Random(seed)
        if n_total <= 0:
            flat_indices: list[int] = []
        elif working_cap >= n_total:
            flat_indices = list(range(n_total))
        else:
            flat_indices = rng.sample(range(n_total), working_cap)

        # Decode product(ci, tl, az, z, si) with si fastest.
        shape_axes = (n_conformers, n_tilt, n_az, n_z, n_sites)

        def _else_items():
            for flat in flat_indices:
                ci, tl_i, az_i, z_i, si_i = _unravel_product_index(flat, shape_axes)
                yield _fields(
                    conformer_index=ci,
                    orientation_type=orient,
                    site_index=site_list[si_i],
                    tilt_deg=_TILT_FULL[tl_i],
                    azimuth_deg=_AZIMUTH[az_i],
                    z_fraction=_Z_FRACTIONS[z_i],
                )

        specs = _collect(_else_items(), cap=working_cap)
        if len(specs) > n_desired:
            specs = _subsample(specs, n_desired, seed)

    for i, spec in enumerate(specs):
        spec.placement_index = i

    return specs[:n_desired]

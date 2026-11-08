"""Sampler policy for batch-based BO placement spec proposals."""

import itertools
import logging
import math
import random
from collections import defaultdict, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from .._numeric_defaults import K_B_EV_PER_K
from ..models import PlacementSpec
from ._constants import (
    _AZIMUTH,
    _AZIMUTH_IN_PLANE,
    _EARLY_CAP_WORKING_SET_MULTIPLIER,
    _GRID_AXIS_KEY_DECIMALS,
    _GRID_BUILD_CAP,
    _PLACEMENT_GRID_COUNT_SEED,
    _POLICY_BRANCH_SEED_STEP,
    _POLICY_PRIOR_TILT_WEIGHT_PER_DEG,
    _POLICY_PRIOR_Z_FRACTION_TARGET,
    _POLICY_PRIOR_Z_FRACTION_WEIGHT,
    _SITE_NEIGHBOR_ABS_TOL_ANGSTROM,
    _TILT_FULL,
    _TILT_PARALLEL,
    _WALL_Z_FRACTIONS,
    _Z_FRACTIONS,
)

logger = logging.getLogger(__name__)


def _grid_axis_key(value: float) -> float:
    """Bucket a grid float so inexact values such as 0.1 share one key."""
    return round(float(value), _GRID_AXIS_KEY_DECIMALS)


def _site_rank_map(site_order: Sequence[int] | None) -> dict[int, int]:
    """First-seen position of each site in the spanned list."""
    unique = dict.fromkeys(int(site) for site in site_order or ())
    return {site: rank for rank, site in enumerate(unique)}


def _site_axis_key(site_index: int, site_rank: Mapping[int, int]) -> tuple[int, int]:
    """Span rank, then numeric id for sites missing from the span."""
    idx = int(site_index)
    rank = site_rank.get(idx)
    if rank is None:
        return (1, idx)
    return (0, rank)


def _unravel_product_index(flat: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Decode a flat index into ``itertools.product`` coordinates (last axis fastest)."""
    coords: list[int] = []
    rem = int(flat)
    for size in reversed(shape):
        coords.append(rem % size)
        rem //= size
    return tuple(reversed(coords))


def _working_set_flat_indices(n_total: int, cap: int, rng: random.Random) -> list[int]:
    """Product indices: full range when it fits, otherwise a seeded sample."""
    if n_total <= 0 or cap <= 0:
        return []
    if cap >= n_total:
        return list(range(n_total))
    return rng.sample(range(n_total), cap)


def _working_set_cap(n_desired: int, n_total: int | None = None) -> int:
    """Seeded working-set size, optionally clamped to the legal product."""
    cap = min(
        _GRID_BUILD_CAP,
        max(int(n_desired) * _EARLY_CAP_WORKING_SET_MULTIPLIER, int(n_desired)),
    )
    if n_total is None:
        return cap
    return min(cap, n_total if n_total > 0 else 0)


def _bin_sort_key(key: str) -> tuple[int, int | str]:
    """Numeric occupancy counts first, then other bin labels."""
    if key.lstrip("-").isdigit():
        return (0, int(key))
    return (1, key)


def _bin_pair_queues(
    n_pairs: int,
    pair_bins: Sequence[str],
    rng: random.Random,
) -> list[deque[int]]:
    """Seeded shuffle inside each bin, bins in deterministic order."""
    groups: dict[str, list[int]] = defaultdict(list)
    for pair in range(int(n_pairs)):
        key = str(pair_bins[pair]) if pair < len(pair_bins) else "none"
        groups[key].append(pair)
    queues: list[deque[int]] = []
    for key in sorted(groups, key=_bin_sort_key):
        items = groups[key]
        rng.shuffle(items)
        queues.append(deque(items))
    return queues


def _bins_are_occupancy_counts(pair_bins: Sequence[str], n_pairs: int) -> bool:
    """Report whether every *pair_bins* label is a numeric occupancy count.

    One count is enough. A covered terrace where every open site has the same
    neighbor count still has to rotate binding modes before extra tilts.
    """
    if int(n_pairs) <= 0:
        return False
    seen: set[str] = set()
    for pair in range(int(n_pairs)):
        seen.add(str(pair_bins[pair]) if pair < len(pair_bins) else "none")
    return bool(seen) and all(key.lstrip("-").isdigit() for key in seen)


def _spread_pair_orientation_flats(
    leading_size: int,
    n_pairs: int,
    pair_bins: Sequence[str],
    cap: int,
    rng: random.Random,
) -> list[int]:
    """One seeded orientation per pair, bins alternating, until *cap* fills."""
    pair_order = _drain_round_robin(_bin_pair_queues(n_pairs, pair_bins, rng))
    leads_left = {pair: list(range(int(leading_size))) for pair in pair_order}
    for queue in leads_left.values():
        rng.shuffle(queue)
    flats: list[int] = []
    while len(flats) < cap:
        progressed = False
        for pair in pair_order:
            queue = leads_left[pair]
            if not queue:
                continue
            lead = queue.pop()
            flats.append(int(lead) * int(n_pairs) + int(pair))
            progressed = True
            if len(flats) >= cap:
                return flats
        if not progressed:
            break
    return flats


def _finite_nonnegative(value: float) -> bool:
    """Return whether *value* is a real clearance and not an overlap."""
    return math.isfinite(value) and float(value) >= 0.0


def _bundle_sort_key(item: tuple[float, list[int]]) -> tuple[int, float, int]:
    """Finite open sites first, then non-finite clearance, then overlaps."""
    clearance, pairs = item
    pair = int(pairs[0])
    if _finite_nonnegative(clearance):
        return (0, -float(clearance), pair)
    if math.isfinite(clearance):
        return (2, -float(clearance), pair)
    return (1, 0.0, pair)


def _stored_pair_clearance(
    pair: int,
    pair_clearances: Sequence[float] | None,
) -> float:
    """Clearance of *pair*. An unranked catalog ties every site at zero."""
    if pair_clearances is None or pair >= len(pair_clearances):
        return 0.0
    return float(pair_clearances[pair])


def _occupancy_type_bundles(
    n_pairs: int,
    pair_bins: Sequence[str],
    pair_sites: Sequence[int] | None,
    pair_types: Sequence[str] | None,
    pair_clearances: Sequence[float] | None,
) -> tuple[dict[str, dict[str, list[tuple[float, list[int]]]]], list[str]] | None:
    """Group height pairs by occupancy count and site type.

    Returns None when the bins are not occupancy counts. Within a type,
    finite non-negative clearances come first, clearest first.
    """
    if not _bins_are_occupancy_counts(pair_bins, n_pairs):
        return None
    raw: dict[tuple[str, str], list[tuple[float, list[int]]]] = defaultdict(list)
    current_site: int | None = None
    current_key: tuple[str, str] | None = None
    current_pairs: list[int] = []

    def _store(key: tuple[str, str], pairs: list[int]) -> None:
        raw[key].append((_stored_pair_clearance(pairs[0], pair_clearances), pairs))

    for pair in range(int(n_pairs)):
        site = int(pair_sites[pair]) if pair_sites is not None else pair
        site_type = (
            str(pair_types[pair])
            if pair_types is not None and pair < len(pair_types)
            else "none"
        )
        key = (
            str(pair_bins[pair]) if pair < len(pair_bins) else "none",
            site_type,
        )
        if current_key is None or site != current_site or key != current_key:
            if current_pairs and current_key is not None:
                _store(current_key, current_pairs)
            current_site = site
            current_key = key
            current_pairs = [pair]
            continue
        current_pairs.append(pair)
    if current_pairs and current_key is not None:
        _store(current_key, current_pairs)

    by_count: dict[str, dict[str, list[tuple[float, list[int]]]]] = {}
    counts_order: list[str] = []
    for count, site_type in sorted(
        raw, key=lambda item: (_bin_sort_key(item[0]), item[1])
    ):
        typed = by_count.setdefault(count, {})
        typed[site_type] = sorted(raw[(count, site_type)], key=_bundle_sort_key)
        if count not in counts_order:
            counts_order.append(count)
    return by_count, counts_order


def _first_finite_bundle(
    entries: Sequence[tuple[float, list[int]]],
) -> tuple[float, list[int]] | None:
    """Clearest finite, non-negative bundle, or None when every site is blocked."""
    for clearance, pairs in entries:
        if _finite_nonnegative(clearance):
            return clearance, pairs
    return None


def _count_contact_representatives(
    by_count: Mapping[str, Mapping[str, Sequence[tuple[float, list[int]]]]],
    counts_order: Sequence[str],
) -> list[tuple[str, float, list[int]]]:
    """One finite contact bundle per occupancy count.

    The opening type rotates across counts. A non-finite clearance never
    takes the slot. A count with no open site is skipped.
    """
    winners: list[tuple[str, float, list[int]]] = []
    for index, count in enumerate(counts_order):
        types = sorted(by_count[count])
        if not types:
            continue
        rotated = types[index % len(types)]
        chosen = _first_finite_bundle(by_count[count][rotated])
        if chosen is None:
            for site_type in types:
                chosen = _first_finite_bundle(by_count[count][site_type])
                if chosen is not None:
                    break
        if chosen is None:
            continue
        clearance, pairs = chosen
        winners.append((str(count), float(clearance), list(pairs)))
    return winners


def _other_type_contact_columns(
    by_count: Mapping[str, Mapping[str, Sequence[tuple[float, list[int]]]]],
    winners: Sequence[tuple[str, float, list[int]]],
    reserved_pairs: set[int],
) -> list[deque[int]]:
    """Clearest finite contact of each remaining site type, one column per count."""
    columns: list[deque[int]] = []
    for count, _clearance, pairs in winners:
        winner_pair = int(pairs[0])
        column: deque[int] = deque()
        for site_type in sorted(by_count[count]):
            found = _first_finite_bundle(by_count[count][site_type])
            if found is None:
                continue
            pair = int(found[1][0])
            if pair == winner_pair or pair in reserved_pairs:
                continue
            column.append(pair)
        columns.append(column)
    return columns


def _working_set_pair_round_robin(
    leading_size: int,
    n_pairs: int,
    pair_bins: Sequence[str],
    cap: int,
    rng: random.Random,
    pair_sites: Sequence[int] | None = None,
    pair_types: Sequence[str] | None = None,
    pair_clearances: Sequence[float] | None = None,
) -> list[int]:
    """Capped flats. One contact per occupancy count, then the other site types.

    A bare site-type catalog keeps one seeded orientation per pair. Under
    coverage the first wave is one upright contact per count, rotating site
    type so a short draw is not all atops. Clearance ties of that winner come
    next, then the clearest finite contact of each other site type. Extra
    heights, tilts, and azimuths follow those contacts. Non-finite clearance
    does not take a contact slot while a real site is open.
    """
    n_last = int(n_pairs)
    n_lead = int(leading_size)
    n_total = n_lead * n_last
    if n_total <= 0 or cap <= 0:
        return []
    if cap >= n_total:
        return list(range(n_total))
    grouped = _occupancy_type_bundles(
        n_last,
        pair_bins,
        pair_sites,
        pair_types,
        pair_clearances,
    )
    if grouped is None:
        return _spread_pair_orientation_flats(n_lead, n_last, pair_bins, cap, rng)
    by_count, counts_order = grouped

    def _clearances_tie(left: float, right: float) -> bool:
        if math.isnan(left) or math.isnan(right):
            return False
        if (left >= 0.0) != (right >= 0.0):
            return False
        if math.isinf(left) or math.isinf(right):
            return bool(math.isinf(left) and math.isinf(right))
        return abs(left - right) <= float(_SITE_NEIGHBOR_ABS_TOL_ANGSTROM)

    flats: list[int] = []
    emitted: set[tuple[int, int]] = set()

    def _emit(pair: int, lead: int) -> bool:
        if len(flats) >= cap:
            return False
        key = (int(pair), int(lead))
        if key in emitted:
            return True
        flats.append(int(lead) * n_last + int(pair))
        emitted.add(key)
        return len(flats) < cap

    winners = _count_contact_representatives(by_count, counts_order)
    for _count, _clearance, pairs in winners:
        if not _emit(pairs[0], 0):
            return flats

    tie_queues: list[deque[list[int]]] = []
    for count, clearance, pairs in winners:
        winner_pair = pairs[0]
        typed_ties: list[deque[list[int]]] = []
        for site_type in sorted(by_count[count]):
            ties = deque(
                entry_pairs
                for entry_clear, entry_pairs in by_count[count][site_type]
                if entry_pairs[0] != winner_pair
                and _clearances_tie(entry_clear, clearance)
            )
            if ties:
                typed_ties.append(ties)
        tie_queues.append(deque(_drain_round_robin(typed_ties)))
    while any(tie_queues):
        progressed = False
        for tied in tie_queues:
            if not tied:
                continue
            bundle = tied.popleft()
            progressed = True
            if not _emit(bundle[0], 0):
                return flats
        if not progressed:
            break

    reserved = {int(pairs[0]) for _count, _clearance, pairs in winners}
    reserved.update(pair for pair, lead in emitted if lead == 0)
    for pair in _drain_round_robin(
        _other_type_contact_columns(by_count, winners, reserved)
    ):
        if not _emit(pair, 0):
            return flats

    for _count, _clearance, pairs in winners:
        for pair in pairs[1:]:
            if not _emit(pair, 0):
                return flats
    lead = 1
    while lead < n_lead and len(flats) < cap:
        for _count, _clearance, pairs in winners:
            if not _emit(pairs[0], lead):
                return flats
        lead += 1

    pending: list[deque[tuple[int, int]]] = []
    for count in counts_order:
        for site_type in sorted(by_count[count]):
            for _clearance, pairs in by_count[count][site_type]:
                left = deque(
                    (int(pair), int(orient))
                    for orient in range(n_lead)
                    for pair in pairs
                    if (int(pair), int(orient)) not in emitted
                )
                if left:
                    pending.append(left)
    while len(flats) < cap and pending:
        progressed = False
        for waiting in pending:
            if not waiting:
                continue
            pair, orient = waiting.popleft()
            progressed = True
            if not _emit(pair, orient):
                return flats
        if not progressed:
            break
    return flats


def _ensure_bin_pair_flats(
    flats: Sequence[int],
    *,
    pair_bins: Sequence[str],
    n_pairs: int,
    pair_axis_stride: int,
    cap: int,
    rng: random.Random,
    pair_sites: Sequence[int] | None = None,
    pair_types: Sequence[str] | None = None,
    pair_clearances: Sequence[float] | None = None,
) -> list[int]:
    """Prepend contact flats so every count and site type survives the cap.

    Bare catalogs keep one zero-lead flat per bin. Under coverage each
    occupancy count reserves its rotated contact, then one finite contact per
    other site type. Non-finite clearance is not reserved. Contacts stay
    ahead of the random fill and are truncated to *cap*.
    """
    if cap <= 0 or n_pairs <= 0:
        return []
    grouped = _occupancy_type_bundles(
        int(n_pairs),
        pair_bins,
        pair_sites,
        pair_types,
        pair_clearances,
    )
    if grouped is None:
        forced_pairs = [
            int(queue[0])
            for queue in _bin_pair_queues(int(n_pairs), pair_bins, rng)
            if queue
        ]
    else:
        by_count, counts_order = grouped
        winners = _count_contact_representatives(by_count, counts_order)
        forced_pairs = [int(pairs[0]) for _count, _clearance, pairs in winners]
        forced_pairs.extend(
            _drain_round_robin(
                _other_type_contact_columns(by_count, winners, set(forced_pairs))
            )
        )
    forced = [int(pair) * int(pair_axis_stride) for pair in forced_pairs]
    return list(dict.fromkeys([*forced, *[int(flat) for flat in flats]]))[:cap]


def _drain_round_robin[T](queues: Sequence[deque[T]]) -> list[T]:
    """Pop one item from each non-empty queue until every queue is empty."""
    drained: list[T] = []
    while True:
        progressed = False
        for queue in queues:
            if queue:
                drained.append(queue.popleft())
                progressed = True
        if not progressed:
            break
    return drained


def _parallel_aip_values(tilt: float) -> tuple[float, ...]:
    """In-plane azimuth values at *tilt* for parallel placements.

    At tilt=0, azimuth and azimuth_in_plane both rotate about the surface
    normal and commute, so only one in-plane angle (0) is kept. Single source
    of truth for both :func:`max_batch_placement_specs` and the flat-aromatic
    builder branch.
    """
    return (0.0,) if float(tilt) == 0.0 else _AZIMUTH_IN_PLANE


def _parallel_tilt_aip_counts() -> list[int]:
    """Return in-plane azimuth counts for each parallel tilt."""
    return [len(_parallel_aip_values(tl)) for tl in _TILT_PARALLEL]


def _wall_contact_z_fraction() -> float:
    """Wall ``z_fraction`` that is the covalent-contact sample."""
    return float(_WALL_Z_FRACTIONS[0])


def _wall_z_fraction_is_contact(z_fraction: float) -> bool:
    """Whether a wall fraction shares the covalent-contact center of mass.

    Fractions at or below :func:`_wall_contact_z_fraction` land on that height.
    The contact sample itself is the representative; anything lower repeats it.
    """
    return float(z_fraction) <= _wall_contact_z_fraction()


def _kept_z_fractions(
    site_type: str | None, *, is_void: bool = False
) -> tuple[float, ...]:
    """Distinct center-of-mass samples for one site.

    Void and pore sites keep the full height axis. Wall sites keep the contact
    shell and the shells above it.
    """
    if is_void or site_type == "pore":
        return _Z_FRACTIONS
    return _WALL_Z_FRACTIONS


def _kept_z_count(site_type: str | None, *, is_void: bool = False) -> int:
    """Return how many ``z_fraction`` samples one site keeps."""
    return len(_kept_z_fractions(site_type, is_void=is_void))


def _site_z_pairs(
    site_list: Sequence[int],
    z_for_site: Callable[[int], Sequence[float]],
) -> list[tuple[int, float]]:
    """Legal ``(site, z)`` pairs in span order, shells in grid order on a site."""
    return [(int(site), float(z)) for site in site_list for z in z_for_site(int(site))]


def _kept_z_total(
    site_indices: list[int],
    site_type_for_index: Callable[[int], str | None] | None,
    *,
    site_is_void_for_index: Callable[[int], bool] | None = None,
) -> int:
    """Sum of kept ``z_fraction`` samples over *site_indices*."""
    indices = site_indices if site_indices else [-1]
    if site_type_for_index is None and site_is_void_for_index is None:
        return len(indices) * len(_Z_FRACTIONS)
    return sum(
        _kept_z_count(
            None if site_type_for_index is None else site_type_for_index(int(i)),
            is_void=(
                False
                if site_is_void_for_index is None
                else bool(site_is_void_for_index(int(i)))
            ),
        )
        for i in indices
    )


def _flat_aromatic_parallel_total(*, n_conformers: int, n_site_z: int) -> int:
    """Return parallel flat-aromatic combo count (product×aip generator)."""
    n_aips = _parallel_tilt_aip_counts()
    return n_conformers * 2 * len(_AZIMUTH) * int(n_site_z) * sum(n_aips)


def _unravel_parallel_flat(
    flat: int,
    *,
    n_conformers: int,
    site_z_pairs: Sequence[tuple[int, float]],
    conformer_fastest: bool,
) -> tuple[int, bool, float, float, float, float, int]:
    """Decode parallel flat index to ``(ci, ff, tilt, az, zf, aip, site)``.

    Unweighted order matches
    ``product(ci, ff, tilt, az, pair)`` then ``aip`` (aip fastest).
    Weighted order matches
    ``product(ff, tilt, az, pair, ci)`` then ``aip``.
    *site_z_pairs* is the legal height axis, in span order.
    """
    n_aips = _parallel_tilt_aip_counts()
    n_az = len(_AZIMUTH)
    n_pairs = len(site_z_pairs)
    n_ff = 2
    per_tilt = [n_az * n_pairs * na for na in n_aips]
    block_tilt_sum = sum(per_tilt)
    rem = int(flat)
    if conformer_fastest:
        block_ff = block_tilt_sum * n_conformers
        ff_i = rem // block_ff
        rem %= block_ff
        tl_i = 0
        for i, chunk_base in enumerate(per_tilt):
            chunk = chunk_base * n_conformers
            if rem < chunk:
                tl_i = i
                break
            rem -= chunk
        na = n_aips[tl_i]
        block_az = n_pairs * n_conformers * na
        az_i = rem // block_az
        rem %= block_az
        block_pair = n_conformers * na
        pair_i = rem // block_pair
        rem %= block_pair
        ci = rem // na
        aip_i = rem % na
    else:
        block_ci = n_ff * block_tilt_sum
        ci = rem // block_ci
        rem %= block_ci
        ff_i = rem // block_tilt_sum
        rem %= block_tilt_sum
        tl_i = 0
        for i, chunk in enumerate(per_tilt):
            if rem < chunk:
                tl_i = i
                break
            rem -= chunk
        na = n_aips[tl_i]
        block_az = n_pairs * na
        az_i = rem // block_az
        rem %= block_az
        pair_i = rem // na
        aip_i = rem % na
    tl = float(_TILT_PARALLEL[tl_i])
    aips = _parallel_aip_values(tl)
    site, zf = site_z_pairs[pair_i]
    return (
        int(ci),
        bool(ff_i),
        tl,
        float(_AZIMUTH[az_i]),
        float(zf),
        float(aips[aip_i]),
        int(site),
    )


def _flat_aromatic_branch_capacities(
    *,
    n_conformers: int,
    n_binders: int,
    z_total: int,
) -> tuple[int, int]:
    """Uncapped ``(parallel, EN-down)`` flat-aromatic grid sizes.

    *z_total* is the sum of kept ``z_fraction`` samples over sites (wall sites
    omit fractions below the contact shell). The parallel in-plane collapse
    matches :func:`_parallel_aip_values`.
    """
    n_aips = _parallel_tilt_aip_counts()
    parallel = n_conformers * 2 * len(_AZIMUTH) * int(z_total) * sum(n_aips)
    en_down = (
        n_conformers
        * max(n_binders, 1)
        * len(_TILT_FULL)
        * len(_AZIMUTH)
        * int(z_total)
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
    site_type_for_index: Callable[[int], str | None] | None = None,
    site_is_void_for_index: Callable[[int], bool] | None = None,
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
    site_type_for_index
        When set, wall sites count only the contact shell and above. Omitted,
        the count is the full grid (an upper bound).
    site_is_void_for_index
        When set, void sites keep the full ``z_fraction`` axis (matches pose).
    """
    z_total = _kept_z_total(
        site_indices,
        site_type_for_index,
        site_is_void_for_index=site_is_void_for_index,
    )

    if dissociative:
        if n_hollow_pairs <= 0:
            return 0
        return min(n_hollow_pairs * len(_WALL_Z_FRACTIONS), _GRID_BUILD_CAP)

    if flat_aromatic:
        parallel, en_down = _flat_aromatic_branch_capacities(
            n_conformers=n_conformers, n_binders=n_binders, z_total=z_total
        )
        return min(parallel, _GRID_BUILD_CAP) + min(en_down, _GRID_BUILD_CAP)

    return min(
        n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * z_total,
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

    kt = K_B_EV_PER_K * temperature
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


# Site type, span rank, azimuth, face flip, binder, in-plane azimuth, conformer.
_AxisKey = str | tuple[int, int] | float | bool | tuple[bool, int] | int
# (spec, seeded rank, pose-axis key). Constant fields drop out of the robin.
_KeyedSpec = tuple[PlacementSpec, float, tuple[_AxisKey, ...]]


def _coprime_stride(n_keys: int) -> int:
    """Largest step at most ``n/2`` that still visits every key."""
    if n_keys <= 2:
        return 1
    step = n_keys // 2
    while math.gcd(step, n_keys) != 1:
        step -= 1
    return step


def _pose_axis_key(
    spec: PlacementSpec,
    *,
    cycle_conformer: bool,
    site_rank: Mapping[int, int],
) -> tuple[_AxisKey, ...]:
    """Pose axes in robin order. Site uses span rank; ``None`` binders sort last.

    Site type leads, then the spanned site, then azimuth. A count that holds
    both bridges and atops therefore alternates types before it repeats a
    site. Conformer is last, so a short draw does not spend its budget on
    extra conformers first. A bucket that is already one site type skips the
    constant type field and still opens on the next site.
    """
    en = spec.en_atom_index
    site_type = spec.site_type
    fields: tuple[_AxisKey, ...] = (
        str(site_type) if site_type is not None else "none",
        _site_axis_key(spec.site_index, site_rank),
        _grid_axis_key(spec.azimuth_deg),
        bool(spec.face_flip),
        (en is None, 0 if en is None else int(en)),
        _grid_axis_key(spec.azimuth_in_plane_deg),
    )
    if cycle_conformer:
        return (*fields, int(spec.conformer_index))
    return fields


def _phased_axis_order(
    rows: list[_KeyedSpec],
    phase: int,
    *,
    type_phase: int = 0,
) -> list[PlacementSpec]:
    """Round-robin *rows* by key fields, opening each field on index *phase*.

    *type_phase* rotates only the outermost key. Occupancy counts use it so
    successive counts open on different site types. Nested site and azimuth
    keys stay on *phase*.
    """
    if not rows[0][2]:
        ordered = sorted(rows, key=lambda row: row[1])
        return [spec for spec, _rank, _key in ordered]

    groups: dict[_AxisKey, list[_KeyedSpec]] = defaultdict(list)
    for spec, rank, key in rows:
        groups[key[0]].append((spec, rank, key[1:]))
    if len(groups) < 2:
        return _phased_axis_order(next(iter(groups.values())), phase)

    keys = sorted(groups)
    n_keys = len(keys)
    start = (phase + int(type_phase)) % n_keys
    step = _coprime_stride(n_keys)
    order = [keys[(start + offset * step) % n_keys] for offset in range(n_keys)]
    queues = [deque(_phased_axis_order(groups[key], phase)) for key in order]
    return _drain_round_robin(queues)


def _diverse_shell_order(
    items: list[tuple[PlacementSpec, float]],
    *,
    phase: int,
    cycle_conformer: bool,
    site_rank: Mapping[int, int],
    type_phase: int = 0,
) -> list[PlacementSpec]:
    """Milder tilts first; inside a tilt, phased pose-axis round-robin."""
    bands: dict[float, list[tuple[PlacementSpec, float]]] = defaultdict(list)
    for spec, rank in items:
        bands[_grid_axis_key(abs(float(spec.tilt_deg)))].append((spec, rank))
    ordered: list[PlacementSpec] = []
    for tilt_key in sorted(bands):
        rows = [
            (
                spec,
                rank,
                _pose_axis_key(
                    spec,
                    cycle_conformer=cycle_conformer,
                    site_rank=site_rank,
                ),
            )
            for spec, rank in bands[tilt_key]
        ]
        ordered.extend(_phased_axis_order(rows, phase, type_phase=type_phase))
    return ordered


def _round_robin_height_order(
    bucket: list[PlacementSpec],
    ranks: list[int],
    *,
    z_fraction_target: float,
    site_index_weight: float,
    site_rank: Mapping[int, int],
    diversify_axes: bool = False,
    cycle_conformer: bool = True,
    type_phase: int = 0,
) -> list[PlacementSpec]:
    """Order *bucket* so successive draws cycle ``z_fraction`` shells.

    Within one shell, milder tilts still come first. When *site_index_weight*
    is set, each shell also round-robins sites in span order before the next
    shell. When *diversify_axes* is set, each shell round-robins site, then
    azimuth, then conformer from a phased diagonal instead of the tilt-only
    prior.
    """
    by_z: dict[float, list[tuple[PlacementSpec, float]]] = defaultdict(list)
    for spec, rank in zip(bucket, ranks, strict=True):
        by_z[_grid_axis_key(spec.z_fraction)].append((spec, float(rank)))
    columns: list[deque[PlacementSpec]] = []
    for phase, z_key in enumerate(sorted(by_z)):
        items = by_z[z_key]
        if site_index_weight > 0.0:
            by_site: dict[int, list[tuple[PlacementSpec, float]]] = defaultdict(list)
            for spec_i, rank_f in items:
                by_site[int(spec_i.site_index)].append((spec_i, rank_f))
            site_queues: list[deque[PlacementSpec]] = []
            for site_idx in sorted(
                by_site, key=lambda idx: _site_axis_key(idx, site_rank)
            ):
                ordered = sorted(
                    by_site[site_idx],
                    key=lambda item: _spec_prior_key(
                        item[0],
                        item[1],
                        z_fraction_target=z_fraction_target,
                    ),
                )
                site_queues.append(deque(spec_i for spec_i, _ in ordered))
            columns.append(deque(_drain_round_robin(site_queues)))
        elif diversify_axes:
            columns.append(
                deque(
                    _diverse_shell_order(
                        items,
                        phase=phase,
                        cycle_conformer=cycle_conformer,
                        site_rank=site_rank,
                        type_phase=type_phase,
                    )
                )
            )
        else:
            ordered = sorted(
                items,
                key=lambda item: _spec_prior_key(
                    item[0],
                    item[1],
                    z_fraction_target=z_fraction_target,
                ),
            )
            columns.append(deque(spec for spec, _ in ordered))
    return _drain_round_robin(columns)


def _spec_site_bin(
    spec: PlacementSpec,
    site_bin_for_index: Callable[[int], str] | None,
) -> str:
    """Stratification bucket for *spec* (occupancy count or natural site type)."""
    if site_bin_for_index is not None:
        return str(site_bin_for_index(int(spec.site_index)))
    return str(spec.site_type) if spec.site_type is not None else "none"


def _stratified_sample(
    specs: list[PlacementSpec],
    n_desired: int,
    seed: int,
    *,
    preferred_site_types: tuple[str, ...] = (),
    z_fraction_target: float = _POLICY_PRIOR_Z_FRACTION_TARGET,
    site_index_weight: float = 0.0,
    conformer_weights: list[float] | None = None,
    site_order: Sequence[int] | None = None,
    site_bin_for_index: Callable[[int], str] | None = None,
) -> list[PlacementSpec]:
    """Sample up to *n_desired* specs stratified by site bin (seeded, deterministic).

    Buckets default to ``site_type``. When *site_bin_for_index* is set, that
    key is used instead (an occupancy count under coverage) while each spec
    still carries its natural ``site_type``. Within each bucket, draws cycle
    ``z_fraction`` shells first so a short batch includes the contact, the
    halfway height, and the van der Waals standoff. Milder tilts still come
    first inside one shell, with a seeded tie-break so draws remain
    deterministic. When *preferred_site_types* is set (pores), those buckets
    are drawn first each round-robin pass. Occupancy counts are not preferred.
    When *site_index_weight* > 0, each shell
    round-robins *site_order* (numeric id if omitted). Otherwise each shell
    round-robins site, azimuth, and conformer from a phased diagonal. When
    *conformer_weights* is set, each bucket is additionally interleaved across
    ``conformer_index`` in proportion to those weights
    (:func:`_weighted_conformer_order`); that pass owns conformer order, and
    ``None`` leaves conformer in the shell round-robin.
    """
    if len(specs) <= n_desired:
        return list(specs)

    buckets: dict[str, list[PlacementSpec]] = defaultdict(list)
    for spec in specs:
        buckets[_spec_site_bin(spec, site_bin_for_index)].append(spec)

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

    preferred = [k for k in preferred_site_types if k in buckets]
    keys = preferred + [k for k in sorted(buckets) if k not in preferred]
    site_rank = _site_rank_map(site_order)
    rng = random.Random(seed)
    for bucket_index, key in enumerate(keys):
        bucket = buckets[key]
        ranks = list(range(len(bucket)))
        rng.shuffle(ranks)
        # Numeric bins are occupancy counts. Rotate the opening site type so
        # a draw of one pose per count is not the same binding mode nine times.
        type_phase = bucket_index if str(key).lstrip("-").isdigit() else 0
        best_first = _round_robin_height_order(
            bucket,
            ranks,
            z_fraction_target=z_fraction_target,
            site_index_weight=site_index_weight,
            site_rank=site_rank,
            diversify_axes=site_index_weight <= 0.0,
            cycle_conformer=weights is None,
            type_phase=type_phase,
        )

        if weights is not None:
            best_first = _weighted_conformer_order(best_first, weights, n_desired)
        # Pop from end → reverse so preferred (earlier) specs come off first.
        buckets[key] = list(reversed(best_first))

    selected: list[PlacementSpec] = []
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


def _parallel_quota(n: int, parallel_fraction: float) -> int:
    if n <= 0:
        return 0
    return max(0, min(int(round(n * parallel_fraction)), n))


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
    site_is_void_for_index: Callable[[int], bool] | None = None,
    site_bin_for_index: Callable[[int], str] | None = None,
    site_clearance_for_index: Callable[[int], float] | None = None,
) -> list[PlacementSpec]:
    """BO candidate ``PlacementSpec`` list: full Cartesian grid (capped), then stratified subsample to *n_desired* (*seed*).

    Every shape uses the same height-shell stratified order. *site_bin_for_index*
    may remap sites into stratification buckets (an occupancy count under
    coverage) without changing the reported ``site_type``.

    With ``conformer_weighting="boltzmann"`` and *conformer_energies* supplied,
    the subsample allocates spec slots per ``conformer_index`` in proportion to
    ``exp(-(E_i - E_min) / (k_B * boltzmann_temperature))``. The allocation is
    deterministic (no RNG) and degrades to the uniform draw whenever the
    energies cannot support weighting. The dissociative branch pins
    ``conformer_index=0``, so it is never weighted.
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

    def _is_void_site(site_idx: int) -> bool:
        if site_is_void_for_index is not None:
            return bool(site_is_void_for_index(int(site_idx)))
        return site_type_for_index(int(site_idx)) == "pore"

    def _collect(
        items: Iterable[dict[str, Any]],
        cap: int,
    ) -> list[PlacementSpec]:
        """Collect specs from *items* until *cap*, applying filters when set."""
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
        specs: list[PlacementSpec],
        n: int,
        sub_seed: int,
        site_order: list[int],
    ) -> list[PlacementSpec]:
        # Under coverage a non-finite clearance is a ligand site. A bare
        # catalog is entirely non-finite, and then every site stays.
        if not dissociative and site_clearance_for_index is not None:
            finite = [
                spec
                for spec in specs
                if math.isfinite(float(site_clearance_for_index(int(spec.site_index))))
            ]
            if finite:
                specs = finite
        return _stratified_sample(
            specs,
            n,
            sub_seed,
            preferred_site_types=preferred_site_types,
            z_fraction_target=z_fraction_target,
            site_index_weight=site_index_weight,
            conformer_weights=conformer_weights,
            site_order=site_order,
            site_bin_for_index=site_bin_for_index,
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
                for pair_idx, zfv in itertools.product(pair_indices, _WALL_Z_FRACTIONS)
            )
            working_cap = _working_set_cap(n_desired)
            specs = _collect(items, cap=working_cap)
            if len(specs) > n_desired:
                specs = _subsample(specs, n_desired, seed, pair_indices)
    else:
        site_list = list(normalized_sites)

        def _z_for_site(site_idx: int) -> tuple[float, ...]:
            return _kept_z_fractions(
                site_type_for_index(int(site_idx)),
                is_void=_is_void_site(site_idx),
            )

        site_z_pairs = _site_z_pairs(site_list, _z_for_site)

        def _pair_bin(site: int) -> str:
            if site_bin_for_index is not None:
                return str(site_bin_for_index(int(site)))
            site_type = site_type_for_index(int(site))
            return str(site_type) if site_type is not None else "none"

        pair_sites = [site for site, _z in site_z_pairs]
        pair_types = [
            str(site_type_for_index(int(site)) or "none") for site in pair_sites
        ]
        pair_bins = [_pair_bin(site) for site in pair_sites]
        pair_clearances = (
            None
            if site_clearance_for_index is None
            else [float(site_clearance_for_index(int(site))) for site in pair_sites]
        )

        if flat_aromatic:
            n_par = _parallel_quota(n_desired, parallel_fraction)
            n_en = n_desired - n_par

            n_pairs = len(site_z_pairs)
            conformer_fastest = conformer_weights is not None
            n_par_total = _flat_aromatic_parallel_total(
                n_conformers=n_conformers, n_site_z=n_pairs
            )
            n_bind = max(n_binders, 1)
            n_en_total = (
                n_conformers * n_bind * len(_TILT_FULL) * len(_AZIMUTH) * n_pairs
            )
            shared_cap = _working_set_cap(n_desired)
            par_rng = random.Random(seed)
            par_flats = _working_set_flat_indices(n_par_total, shared_cap, par_rng)
            n_aips = _parallel_tilt_aip_counts()
            par_pair_stride = (
                int(n_aips[0])
                if not conformer_fastest
                else (int(n_conformers) * int(n_aips[0]))
            )
            par_flats = _ensure_bin_pair_flats(
                par_flats,
                pair_bins=pair_bins,
                n_pairs=n_pairs,
                pair_axis_stride=par_pair_stride,
                cap=shared_cap,
                rng=par_rng,
                pair_sites=pair_sites,
                pair_types=pair_types,
                pair_clearances=pair_clearances,
            )
            en_seed = seed + _POLICY_BRANCH_SEED_STEP
            en_leading = (
                n_bind * len(_TILT_FULL) * len(_AZIMUTH) * n_conformers
                if conformer_fastest
                else n_conformers * n_bind * len(_TILT_FULL) * len(_AZIMUTH)
            )
            en_rng = random.Random(en_seed)
            if conformer_fastest:
                en_flats = _working_set_flat_indices(n_en_total, shared_cap, en_rng)
                en_flats = _ensure_bin_pair_flats(
                    en_flats,
                    pair_bins=pair_bins,
                    n_pairs=n_pairs,
                    pair_axis_stride=n_conformers,
                    cap=shared_cap,
                    rng=en_rng,
                    pair_sites=pair_sites,
                    pair_types=pair_types,
                    pair_clearances=pair_clearances,
                )
            else:
                en_flats = _working_set_pair_round_robin(
                    en_leading,
                    n_pairs,
                    pair_bins,
                    shared_cap,
                    en_rng,
                    pair_sites=pair_sites,
                    pair_types=pair_types,
                    pair_clearances=pair_clearances,
                )

            def _par_items():
                for flat in par_flats:
                    ci, ff, tl, azv, zfv, aip, si = _unravel_parallel_flat(
                        flat,
                        n_conformers=n_conformers,
                        site_z_pairs=site_z_pairs,
                        conformer_fastest=conformer_fastest,
                    )
                    yield _fields(
                        conformer_index=ci,
                        orientation_type="parallel",
                        face_flip=ff,
                        site_index=si,
                        tilt_deg=tl,
                        azimuth_deg=azv,
                        azimuth_in_plane_deg=aip,
                        z_fraction=zfv,
                    )

            # Under weighting, conformer varies fastest so a full walk is not
            # conformer 0 first.
            en_shape = (
                (
                    n_bind,
                    len(_TILT_FULL),
                    len(_AZIMUTH),
                    n_pairs,
                    n_conformers,
                )
                if conformer_fastest
                else (
                    n_conformers,
                    n_bind,
                    len(_TILT_FULL),
                    len(_AZIMUTH),
                    n_pairs,
                )
            )

            def _en_items():
                for flat in en_flats:
                    coords = _unravel_product_index(flat, en_shape)
                    if conformer_fastest:
                        ei, tl_i, az_i, pair_i, ci = coords
                    else:
                        ci, ei, tl_i, az_i, pair_i = coords
                    site, z_fraction = site_z_pairs[pair_i]
                    yield _fields(
                        conformer_index=ci,
                        orientation_type="EN-down",
                        en_atom_index=ei if n_binders > 1 else None,
                        site_index=site,
                        tilt_deg=_TILT_FULL[tl_i],
                        azimuth_deg=_AZIMUTH[az_i],
                        z_fraction=z_fraction,
                    )

            # Size each working set for full n_desired so a filtered-out branch
            # can be topped up from the other side's surplus.
            par_pool = _collect(_par_items(), cap=shared_cap)
            en_pool = _collect(_en_items(), cap=shared_cap)
            n_par_take = min(n_par, len(par_pool))
            n_en_take = min(n_en, len(en_pool))
            deficit = n_desired - (n_par_take + n_en_take)
            if deficit > 0:
                par_extra = min(deficit, len(par_pool) - n_par_take)
                n_par_take += par_extra
                deficit -= par_extra
                n_en_take += min(deficit, len(en_pool) - n_en_take)

            par_specs = (
                _subsample(par_pool, n_par_take, seed, site_list)
                if n_par_take > 0
                else []
            )
            en_specs = (
                _subsample(en_pool, n_en_take, en_seed, site_list)
                if n_en_take > 0
                else []
            )
            specs = par_specs + en_specs
        else:
            orient = "vertical" if shape == "linear" else "round"
            n_pairs = len(site_z_pairs)
            n_tilt = len(_TILT_FULL)
            n_az = len(_AZIMUTH)
            n_total = n_conformers * n_tilt * n_az * n_pairs
            working_cap = _working_set_cap(n_desired, n_total)
            flat_indices = _working_set_pair_round_robin(
                n_conformers * n_tilt * n_az,
                n_pairs,
                pair_bins,
                working_cap,
                random.Random(seed),
                pair_sites=pair_sites,
                pair_types=pair_types,
                pair_clearances=pair_clearances,
            )

            shape_axes = (n_conformers, n_tilt, n_az, n_pairs)

            def _else_items():
                for flat in flat_indices:
                    ci, tl_i, az_i, pair_i = _unravel_product_index(flat, shape_axes)
                    site, z_fraction = site_z_pairs[pair_i]
                    yield _fields(
                        conformer_index=ci,
                        orientation_type=orient,
                        site_index=site,
                        tilt_deg=_TILT_FULL[tl_i],
                        azimuth_deg=_AZIMUTH[az_i],
                        z_fraction=z_fraction,
                    )

            pool = _collect(_else_items(), cap=working_cap)
            specs = _subsample(pool, n_desired, seed, site_list)

    for i, spec in enumerate(specs):
        spec.placement_index = i

    return specs[:n_desired]

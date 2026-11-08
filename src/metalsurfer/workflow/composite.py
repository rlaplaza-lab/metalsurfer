"""Joint n-adsorbate configs for n-tuplet saturation.

With ``AdsorptionConfig.saturation_molecules_per_step > 1`` each saturation
trial is one initial structure with exactly *n* adsorbates, relaxed together
via TorchSim. Placement-only CPU work builds mutually clear packs
(:func:`pack_exact_tuplet`, :func:`assemble_joint_config_groups`); relaxation
and validation run in batch (:func:`evaluate_composite_batch`).

INVARIANT (substrate-prefix contract): adsorbate atoms come strictly AFTER the
substrate prefix in every composite. Freeze constraints, desorption checks,
decomposition filters, symmetry analysis, and
:func:`workflow.shared._build_surface_reference_slab` all rely on it.

Energy representation: ``energy_adslab`` and ``energy_adsorbate`` are the
composite totals; ``energy_adsorption`` is **per molecule**
``E_ads_total / n`` where ``E_ads_total = E(composite) - E_slab - sum_i E_mol``.
Thus ``n * energy_adsorption = energy_adslab - energy_slab - energy_adsorbate``.
Per-unit identity lives in ``molecule``, ``placement_id``, ``placement_descriptor``,
and ``distance``.
"""

import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..filters import adsorbates_mutually_disconnected
from ..models import ScreeningResult
from ..optimization import optimize_adsorbate_slab_batched
from ..placement._material import calculator_pbc_for_atoms, material_aware_pbc
from ..placement.clash import (
    atom_radii_for_symbols,
    clash_bounds_for_adsorbate,
    compose_quaternion_with_azimuth,
    resolve_rigid_clash,
    tuplet_clash_rescue_floor,
)
from ..placement.cutoffs import mixed_covalent_thresholds
from ..placement.geometry import (
    _mol_slab_pairwise_distances,
    calculate_min_distance,
    compute_surface_site_frame,
    occupied_neighbor_counts,
)
from ..placement.occupancy import incoming_inplane_radius
from ..placement.site_coords import _slab_normal
from ..surface_prep import apply_material_pbc
from ..surface_prep.freeze import check_frozen_substrate_displacement
from .shared import _validate_geometry

logger = logging.getLogger(__name__)

__all__ = [
    "assemble_joint_config_groups",
    "assemble_quota_joint_configs",
    "build_composite_candidate",
    "evaluate_composite_batch",
    "pack_exact_tuplet",
]


def build_composite_candidate(
    slab_atoms: Atoms,
    adsorbates: Sequence[Atoms],
) -> Atoms:
    """Build one candidate: substrate prefix followed by *adsorbates* in order.

    Parameters
    ----------
    slab_atoms
        Current coverage slab (bare substrate prefix + any pre-adsorbed units).
    adsorbates
        Adsorbate-only fragments to append, in tuplet selection order.

    Returns
    -------
    Atoms
        Combined structure with PBC set via
        :func:`placement._material.calculator_pbc_for_atoms`. FixAtoms
        constraints refer to the untouched substrate prefix.
    """
    composite = slab_atoms.copy()
    for adsorbate in adsorbates:
        composite += adsorbate
    composite.set_pbc(calculator_pbc_for_atoms(composite))
    return composite


def _suffix_positions(result: ScreeningResult) -> np.ndarray:
    return np.asarray(result.atoms.get_positions()[result.slab_size :], dtype=float)


def _slab_site_frame(slab_atoms: Atoms) -> np.ndarray:
    """Local site frame from slab cell normal (planar default)."""
    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    normal = _slab_normal(cell)
    return compute_surface_site_frame(normal)


def _fixed_cloud_from_coverage_and_results(
    slab_atoms: Atoms,
    results: Sequence[ScreeningResult],
    suffixes: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage-slab positions and covalent radii, plus packed suffixes."""
    fixed_pos = np.asarray(slab_atoms.get_positions(), dtype=float)
    fixed_radii = atom_radii_for_symbols(list(slab_atoms.get_chemical_symbols()))
    for other, prev in zip(suffixes, results, strict=True):
        fixed_pos = np.vstack([fixed_pos, other])
        prev_syms = list(prev.atoms.get_chemical_symbols()[prev.slab_size :])
        fixed_radii = np.concatenate(
            [
                fixed_radii,
                atom_radii_for_symbols(prev_syms),
            ]
        )
    return fixed_pos, fixed_radii


def _min_dist_to_suffixes(
    suffix: np.ndarray,
    others: Sequence[np.ndarray],
    *,
    cell: np.ndarray,
    pbc: list[bool],
) -> float:
    if not others:
        return float("inf")
    mins = [
        float(np.min(_mol_slab_pairwise_distances(suffix, other, cell, pbc)))
        for other in others
        if np.asarray(other).size
    ]
    return min(mins) if mins else float("inf")


def _apply_suffix_to_result(
    result: ScreeningResult,
    new_suffix: np.ndarray,
    *,
    az_delta: float | None = None,
    slab_atoms: Atoms,
) -> ScreeningResult:
    """Return a copy with adsorbate suffix positions (and descriptor COM/quat) updated."""
    atoms = result.atoms.copy()
    pos = atoms.get_positions()
    pos[result.slab_size :] = np.asarray(new_suffix, dtype=float)
    atoms.set_positions(pos)
    com = np.mean(new_suffix, axis=0)
    desc = result.placement_descriptor
    quat_w = desc.quat_w
    quat_x = desc.quat_x
    quat_y = desc.quat_y
    quat_z = desc.quat_z
    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    n_hat = _slab_normal(cell)
    if (
        az_delta is not None
        and quat_w is not None
        and quat_x is not None
        and quat_y is not None
        and quat_z is not None
    ):
        quat_w, quat_x, quat_y, quat_z = compose_quaternion_with_azimuth(
            (quat_w, quat_x, quat_y, quat_z),
            az_delta,
            n_hat,
        )
    surface_ref = (
        float(desc.surface_ref_z_abs) if desc.surface_ref_z_abs is not None else 0.0
    )
    z_offset = float(np.dot(com, n_hat) - surface_ref)
    new_desc = replace(
        desc,
        x=float(com[0]),
        y=float(com[1]),
        x_abs=float(com[0]),
        y_abs=float(com[1]),
        z_abs=float(com[2]),
        z_offset=z_offset,
        quat_w=quat_w,
        quat_x=quat_x,
        quat_y=quat_y,
        quat_z=quat_z,
    )
    return replace(result, atoms=atoms, placement_descriptor=new_desc)


def _try_rescue_suffix(
    candidate: ScreeningResult,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    n_substrate: int,
    slab_atoms: Atoms,
    cell: np.ndarray,
    pbc: list[bool],
    config: AdsorptionConfig,
) -> ScreeningResult | None:
    """Clash-descend a candidate suffix against *fixed_pos*; None if unsalvageable."""
    suffix = _suffix_positions(candidate)
    ads = candidate.atoms[candidate.slab_size :].copy()
    origin = np.mean(suffix, axis=0)
    frame = _slab_site_frame(slab_atoms)
    footprint = incoming_inplane_radius(
        ads,
        footprint_scale=float(config.occupancy_footprint_scale),
    )
    bounds = clash_bounds_for_adsorbate(
        ads,
        config,
        footprint_radius=footprint,
    )
    moving_r = atom_radii_for_symbols(list(ads.get_chemical_symbols()))
    thresholds = mixed_covalent_thresholds(
        moving_r,
        fixed_radii,
        n_substrate=int(n_substrate),
        substrate_scale=float(config.min_contact_ratio),
        adsorbate_scale=float(config.connectivity_multiplier),
        absolute_floor=float(config.min_initial_distance),
    )
    new_pos, az_delta, ok = resolve_rigid_clash(
        ads,
        fixed_pos,
        fixed_radii,
        origin=origin,
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        thresholds=thresholds,
        bounds=bounds,
        moving_radii=moving_r,
    )
    if not ok:
        return None
    return _apply_suffix_to_result(
        candidate, new_pos, az_delta=az_delta, slab_atoms=slab_atoms
    )


def pack_exact_tuplet(
    winners: Sequence[ScreeningResult],
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    n_substrate: int | None = None,
) -> list[ScreeningResult] | None:
    """Pack *winners* into one exact-length tuplet or return ``None``.

    Unit 1 stays at its screened pose. Each later unit must either be mutually
    disconnected under ``connectivity_multiplier`` against the packed set, or
    (when ``placement_clash_descent`` is on) be rescued by rigid-body descent
    against the coverage slab + packed units and then re-validated with the same
    connectivity rule. Overlap with clash descent off, or a failed rescue /
    post-rescue bond, rejects the whole pack. Partial packs are never returned.

    *n_substrate* is the bare-substrate atom count used for per-atom clash
    scales (defaults to ``len(slab_atoms)`` when the coverage frame is bare).
    """
    if not winners:
        return None
    if len(winners) == 1:
        return [winners[0]]

    substrate_n = len(slab_atoms) if n_substrate is None else int(n_substrate)
    if substrate_n < 0 or substrate_n > len(slab_atoms):
        raise ValueError(
            f"n_substrate ({substrate_n}) must be in [0, {len(slab_atoms)}] "
            "(coverage slab length)"
        )

    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    adsorbate_scale = float(config.connectivity_multiplier)
    clash_on = bool(config.placement_clash_descent)

    packed: list[ScreeningResult] = [winners[0]]
    packed_suffixes: list[np.ndarray] = [_suffix_positions(winners[0])]

    for winner in winners[1:]:
        suffix = _suffix_positions(winner)
        clear = all(
            adsorbates_mutually_disconnected(
                winner.atoms[winner.slab_size :],
                prev.atoms[prev.slab_size :],
                adsorbate_scale,
                material_type=config.material_type,
                cell=cell,
            )
            for prev in packed
        )
        if clear:
            packed.append(winner)
            packed_suffixes.append(suffix)
            continue

        if not clash_on:
            return None

        cand_syms = list(winner.atoms.get_chemical_symbols()[winner.slab_size :])
        fixed_syms: list[str] = []
        for prev in packed:
            fixed_syms.extend(list(prev.atoms.get_chemical_symbols()[prev.slab_size :]))
        clearance_targets: list[np.ndarray] = list(packed_suffixes)
        if substrate_n < len(slab_atoms):
            fixed_syms.extend(list(slab_atoms.get_chemical_symbols()[substrate_n:]))
            pre_ads = np.asarray(slab_atoms.get_positions()[substrate_n:], dtype=float)
            if pre_ads.size:
                clearance_targets.append(pre_ads)
        rescue_floor = tuplet_clash_rescue_floor(cand_syms, fixed_syms)
        if (
            _min_dist_to_suffixes(suffix, clearance_targets, cell=cell, pbc=pbc)
            < rescue_floor
        ):
            return None

        fixed_pos, fixed_radii = _fixed_cloud_from_coverage_and_results(
            slab_atoms,
            packed,
            packed_suffixes,
        )
        rescued = _try_rescue_suffix(
            winner,
            fixed_pos,
            fixed_radii,
            n_substrate=substrate_n,
            slab_atoms=slab_atoms,
            cell=cell,
            pbc=pbc,
            config=config,
        )
        if rescued is None:
            return None
        # Descent clears its own thresholds; accept only when the shared
        # disconnect predicate agrees.
        still_clear = all(
            adsorbates_mutually_disconnected(
                rescued.atoms[rescued.slab_size :],
                prev.atoms[prev.slab_size :],
                adsorbate_scale,
                material_type=config.material_type,
                cell=cell,
            )
            for prev in packed
        )
        if still_clear and substrate_n < len(slab_atoms):
            still_clear = adsorbates_mutually_disconnected(
                rescued.atoms[rescued.slab_size :],
                slab_atoms[substrate_n:],
                adsorbate_scale,
                material_type=config.material_type,
                cell=cell,
            )
        if not still_clear:
            return None
        packed.append(rescued)
        packed_suffixes.append(_suffix_positions(rescued))

    return packed


def _pose_site(pose: ScreeningResult) -> int:
    return int(pose.placement_descriptor.site_index)


def _pose_com(pose: ScreeningResult) -> np.ndarray:
    fragment = pose.atoms[pose.slab_size :]
    return np.mean(np.asarray(fragment.get_positions(), dtype=float), axis=0)


def _site_positions(poses: Sequence[ScreeningResult]) -> dict[int, np.ndarray]:
    """One center per ``site_index``, the mean COM of poses on that site."""
    acc: dict[int, list[np.ndarray]] = {}
    for pose in poses:
        acc.setdefault(_pose_site(pose), []).append(_pose_com(pose))
    return {sid: np.mean(np.stack(rows), axis=0) for sid, rows in acc.items()}


def _count_bins_for_candidates(
    candidates: Sequence[int],
    occupied_sites: set[int],
    *,
    site_at: Callable[[int], int | None],
    positions: Mapping[int, np.ndarray],
    cell: np.ndarray,
    pbc: Sequence[bool],
    site_types: Mapping[int, str] | None = None,
) -> dict[int, list[int]]:
    """Incoming-order candidates grouped by occupied-neighbor count."""
    counts = occupied_neighbor_counts(
        positions,
        occupied_sites,
        cell,
        pbc,
        site_types=site_types,
    )
    bins: dict[int, list[int]] = defaultdict(list)
    for index in candidates:
        site = site_at(index)
        if site is None or int(site) in occupied_sites:
            continue
        bins[int(counts.get(int(site), 0))].append(index)
    return bins


def _accept_packed_pose(
    group: Sequence[ScreeningResult],
    index: int,
    *,
    pose_at: Callable[[int], ScreeningResult | None],
    site_at: Callable[[int], int | None],
    packed_sites: set[int],
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    n_substrate: int | None,
) -> tuple[list[ScreeningResult], int] | None:
    """Materialize *index* and keep it when the exact pack is clash-free."""
    site = site_at(index)
    if site is None or int(site) in packed_sites:
        return None
    pose = pose_at(index)
    if pose is None:
        return None
    if _pose_site(pose) in packed_sites:
        return None
    packed = pack_exact_tuplet(
        [*group, pose],
        slab_atoms,
        config,
        n_substrate=n_substrate,
    )
    if packed is None:
        return None
    return packed, index


def _typed_catalog(
    positions: Mapping[int, np.ndarray],
    site_types: Mapping[int, str] | None,
    group: Sequence[ScreeningResult],
    candidates: Sequence[int],
    *,
    pose_at: Callable[[int], ScreeningResult | None],
    site_at: Callable[[int], int | None],
) -> dict[int, str] | None:
    """Site types for *positions*, or None when any waypoint is unlabeled.

    A partial map would pull posed sites out of the catalog graph and leave
    the other anchors in a fake lattice. Catalog labels are kept, and pose
    labels fill only the holes. Pose materialization is skipped when the
    catalog already covers every waypoint.
    """
    labels: dict[int, str] = {}
    if site_types is not None:
        labels.update({int(site): str(label) for site, label in site_types.items()})
    if labels and set(positions).issubset(labels):
        return labels

    def _record(site: int | None, pose: ScreeningResult | None) -> None:
        if pose is None or site is None:
            return
        label = pose.placement_descriptor.site_type
        if label is not None:
            labels[int(site)] = str(label)

    for pose in group:
        _record(_pose_site(pose), pose)
    for index in candidates:
        _record(site_at(index), pose_at(index))
    if labels and set(positions).issubset(labels):
        return labels
    return None


def _select_packing_partner(
    group: Sequence[ScreeningResult],
    candidates: Sequence[int],
    *,
    pose_at: Callable[[int], ScreeningResult | None],
    site_at: Callable[[int], int | None],
    positions: Mapping[int, np.ndarray],
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    n_substrate: int | None,
    count_phase: int,
    site_types: Mapping[int, str] | None = None,
) -> tuple[list[ScreeningResult], int, int] | None:
    """Next clash-free member. Later members round-robin occupancy counts."""
    if not candidates:
        return None
    if not group:
        pose = pose_at(candidates[0])
        if pose is None:
            return None
        return [pose], candidates[0], count_phase

    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    packed_sites = {_pose_site(pose) for pose in group}
    typed = _typed_catalog(
        positions,
        site_types,
        group,
        candidates,
        pose_at=pose_at,
        site_at=site_at,
    )
    bins = _count_bins_for_candidates(
        candidates,
        packed_sites,
        site_at=site_at,
        positions=positions,
        cell=cell,
        pbc=pbc,
        site_types=typed,
    )
    if not bins:
        return None
    keys = sorted(bins)
    start = int(count_phase) % len(keys)
    for offset in range(len(keys)):
        key = keys[(start + offset) % len(keys)]
        for index in bins[key]:
            picked = _accept_packed_pose(
                group,
                index,
                pose_at=pose_at,
                site_at=site_at,
                packed_sites=packed_sites,
                slab_atoms=slab_atoms,
                config=config,
                n_substrate=n_substrate,
            )
            if picked is None:
                continue
            packed, chosen = picked
            return packed, chosen, int(count_phase) + offset + 1
    return None


def _grow_packed_group(
    group: Sequence[ScreeningResult],
    candidates: Sequence[int],
    *,
    pose_at: Callable[[int], ScreeningResult | None],
    site_at: Callable[[int], int | None],
    positions_at: Callable[
        [Sequence[ScreeningResult], Sequence[int]], Mapping[int, np.ndarray]
    ],
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    n_substrate: int | None,
    count_phase: int,
    n_target: int,
    site_types: Mapping[int, str] | None = None,
) -> tuple[list[ScreeningResult], list[int], int]:
    """Extend *group* up to *n_target*.

    *positions_at* receives the group so far and the indices already consumed.
    *site_types* labels every catalog waypoint. Occupied nodes stay the sites
    already in the group.
    """
    packed = list(group)
    chosen: list[int] = []
    blocked: set[int] = set()
    phase = int(count_phase)
    while len(packed) < n_target:
        live = [index for index in candidates if index not in blocked]
        picked = _select_packing_partner(
            packed,
            live,
            pose_at=pose_at,
            site_at=site_at,
            positions=positions_at(packed, chosen),
            slab_atoms=slab_atoms,
            config=config,
            n_substrate=n_substrate,
            count_phase=phase,
            site_types=site_types,
        )
        if picked is None:
            break
        packed, index, phase = picked
        blocked.add(index)
        chosen.append(index)
    return packed, chosen, phase


def _take_packing_member(
    group: Sequence[ScreeningResult],
    remaining: Sequence[ScreeningResult],
    *,
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    n_substrate: int | None,
    count_phase: int,
    positions: Mapping[int, np.ndarray] | None = None,
    site_types: Mapping[int, str] | None = None,
) -> tuple[list[ScreeningResult], int, int] | None:
    """Next packed member from an already-materialized pose list."""
    if not remaining:
        return None
    if positions is None:
        positions = _site_positions([*group, *remaining]) if group else {}

    def _positions(
        _current: Sequence[ScreeningResult],
        _chosen: Sequence[int],
    ) -> Mapping[int, np.ndarray]:
        return positions

    grown, chosen, phase = _grow_packed_group(
        group,
        range(len(remaining)),
        pose_at=remaining.__getitem__,
        site_at=lambda index: _pose_site(remaining[index]),
        positions_at=_positions,
        slab_atoms=slab_atoms,
        config=config,
        n_substrate=n_substrate,
        count_phase=count_phase,
        n_target=len(group) + 1,
        site_types=site_types,
    )
    if len(chosen) != 1:
        return None
    return grown, chosen[0], phase


def assemble_joint_config_groups(
    poses: Sequence[ScreeningResult],
    *,
    n_per_config: int,
    n_configs: int,
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    rng: np.random.Generator,
    n_substrate: int | None = None,
    site_positions: Mapping[int, np.ndarray] | None = None,
    site_types: Mapping[int, str] | None = None,
) -> list[list[ScreeningResult]]:
    """Build up to *n_configs* groups of exactly *n_per_config* poses.

    Later members round-robin occupancy-count bins of sites already in the
    group (catalog neighbor-graph distance at most 2). *site_positions* and
    *site_types* are the full catalog when the caller has one, so unsampled
    sites stay waypoints. A ratio at or below the connectivity cutoff is a
    hard clash. Poses are consumed across groups, and the count round-robin
    continues from one group to the next.
    """
    del rng
    if n_per_config <= 0 or n_configs <= 0 or not poses:
        return []
    positions = (
        dict(site_positions) if site_positions is not None else _site_positions(poses)
    )
    remaining = list(poses)

    def _positions(
        _current: Sequence[ScreeningResult],
        _chosen: Sequence[int],
        fixed: Mapping[int, np.ndarray] = positions,
    ) -> Mapping[int, np.ndarray]:
        return fixed

    groups: list[list[ScreeningResult]] = []
    count_phase = 0
    misses = 0
    while len(groups) < n_configs and remaining:
        if misses >= len(remaining):
            break
        group, chosen, count_phase = _grow_packed_group(
            [],
            range(len(remaining)),
            pose_at=remaining.__getitem__,
            site_at=lambda index: _pose_site(remaining[index]),
            positions_at=_positions,
            slab_atoms=slab_atoms,
            config=config,
            n_substrate=n_substrate,
            count_phase=count_phase,
            n_target=n_per_config,
            site_types=site_types,
        )
        if len(group) != n_per_config:
            remaining.pop(0)
            misses += 1
            continue
        misses = 0
        for index in sorted(chosen, reverse=True):
            remaining.pop(index)
        groups.append(group)
    return groups


def assemble_quota_joint_configs(
    pools: Mapping[str, Sequence[ScreeningResult]],
    *,
    quotas: Mapping[str, int],
    n_configs: int,
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    rng: np.random.Generator,
    n_substrate: int | None = None,
    site_positions: Mapping[int, np.ndarray] | None = None,
    site_types: Mapping[int, str] | None = None,
) -> list[list[ScreeningResult]]:
    """Build joint configs with exact per-species slot counts from *quotas*.

    Partner choice matches :func:`assemble_joint_config_groups`.
    """
    n_total = sum(int(q) for q in quotas.values())
    if n_total <= 0 or n_configs <= 0:
        return []
    if len(quotas) == 1 and int(next(iter(quotas.values()))) == 2:
        mol = next(iter(quotas))
        return assemble_joint_config_groups(
            pools.get(mol, []),
            n_per_config=2,
            n_configs=n_configs,
            slab_atoms=slab_atoms,
            config=config,
            rng=rng,
            n_substrate=n_substrate,
            site_positions=site_positions,
            site_types=site_types,
        )
    del rng
    remaining: dict[str, list[ScreeningResult]] = {
        mol: list(pools.get(mol, [])) for mol in quotas
    }

    recipe = [mol for mol, q in quotas.items() for _ in range(int(q))]
    groups: list[list[ScreeningResult]] = []
    count_phase = 0
    catalog = (
        dict(site_positions)
        if site_positions is not None
        else _site_positions([pose for poses in remaining.values() for pose in poses])
    )
    seed_mol = recipe[0]
    misses = 0
    while len(groups) < n_configs:
        if any(len(remaining[mol]) < int(quotas[mol]) for mol in quotas):
            break
        if misses >= len(remaining[seed_mol]):
            break
        group: list[ScreeningResult] = []
        taken: list[tuple[str, ScreeningResult]] = []
        failed = False
        for mol in recipe:
            picked = _take_packing_member(
                group,
                remaining[mol],
                slab_atoms=slab_atoms,
                config=config,
                n_substrate=n_substrate,
                count_phase=count_phase,
                positions=catalog,
                site_types=site_types,
            )
            if picked is None:
                failed = True
                break
            packed, index, count_phase = picked
            group = packed
            taken.append((mol, remaining[mol].pop(index)))
        if failed or len(group) != n_total:
            for mol, pose in reversed(taken):
                remaining[mol].insert(0, pose)
            if not remaining[seed_mol]:
                break
            remaining[seed_mol].pop(0)
            misses += 1
            continue
        misses = 0
        groups.append(group)
    return groups


def _unit_suffix_bounds(winners: Sequence[ScreeningResult]) -> list[int]:
    """Atom counts of each winner's adsorbate fragment, in selection order."""
    return [len(w.atoms) - w.slab_size for w in winners]


def _per_unit_surface_distances(
    opt_atoms: Atoms,
    *,
    n_substrate: int,
    unit_sizes: Sequence[int],
    config: AdsorptionConfig,
    surface_prefix_atoms: int | None = None,
) -> list[float]:
    """Per-unit min adsorbate-to-surface distance for a relaxed composite.

    When *surface_prefix_atoms* is set, only that bare-substrate prefix counts
    as the surface. When unset, the full coverage prefix ``n_substrate`` is used.
    """
    positions = opt_atoms.get_positions()
    substrate_positions = positions[:n_substrate]
    if surface_prefix_atoms is not None:
        if surface_prefix_atoms < 0 or surface_prefix_atoms > n_substrate:
            raise ValueError(
                f"surface_prefix_atoms ({surface_prefix_atoms}) must be in "
                f"[0, {n_substrate}] (n_substrate)"
            )
        substrate_positions = substrate_positions[:surface_prefix_atoms]
    cell = opt_atoms.get_cell()
    pbc = material_aware_pbc(config.material_type)
    distances: list[float] = []
    start = n_substrate
    for size in unit_sizes:
        unit_positions = positions[start : start + size]
        distances.append(
            calculate_min_distance(
                unit_positions,
                substrate_positions,
                cell,
                use_pbc=True,
                pbc=pbc,
            )
        )
        start += size
    return distances


def _rewrite_relaxed_composite(
    winners: Sequence[ScreeningResult],
    opt_atoms: Atoms,
    *,
    slab_atoms: Atoms,
    E_slab: float,
    config: AdsorptionConfig,
    unit_distances: Sequence[float],
) -> tuple[list[ScreeningResult], float, float]:
    """Map a validated relaxed composite onto per-unit rows.

    Returns ``(rewritten, e_ads_total, e_ads_per_mol)``.
    """
    n_substrate = len(slab_atoms)
    e_adslab = float(opt_atoms.get_potential_energy())
    e_mol_sum = float(sum(w.energy_adsorbate for w in winners))
    e_ads_total = e_adslab - E_slab - e_mol_sum
    e_ads_per_mol = e_ads_total / len(winners)
    rewritten: list[ScreeningResult] = []
    for k, winner in enumerate(winners):
        atoms_out = opt_atoms.copy()
        apply_material_pbc(atoms_out, config.material_type)
        rewritten.append(
            replace(
                winner,
                energy_adslab=e_adslab,
                energy_slab=E_slab,
                energy_adsorbate=e_mol_sum,
                energy_adsorption=e_ads_per_mol,
                atoms=atoms_out,
                slab_size=n_substrate,
                distance=float(unit_distances[k]),
            )
        )
    return rewritten, e_ads_total, e_ads_per_mol


def _validate_relaxed_composite(
    opt_atoms: Atoms,
    winners: Sequence[ScreeningResult],
    *,
    slab_atoms: Atoms,
    base_slab: Atoms,
    config: AdsorptionConfig,
    topology_check: Callable[[Atoms, list[str]], tuple[bool, str]] | None,
    log_prefix: str,
) -> tuple[list[float], str]:
    """Return per-unit surface distances or ``([], reason)`` on failure."""
    ok, reason = check_frozen_substrate_displacement(
        opt_atoms,
        base_slab,
        slab_size=len(base_slab),
    )
    if not ok:
        logger.debug("%scomposite frozen substrate drift: %s", log_prefix, reason)
        return [], f"frozen substrate drift: {reason}"

    ok, reason = _validate_geometry(opt_atoms, slab_atoms, config)
    if not ok:
        logger.debug("%scomposite geometry fail: %s", log_prefix, reason)
        return [], f"geometry fail: {reason}"

    unit_distances = _per_unit_surface_distances(
        opt_atoms,
        n_substrate=len(slab_atoms),
        unit_sizes=_unit_suffix_bounds(winners),
        config=config,
        surface_prefix_atoms=len(base_slab),
    )
    if not config.skip_desorption_check:
        for k, dist in enumerate(unit_distances):
            if dist > config.binding_distance_threshold:
                logger.debug(
                    "%scomposite unit %d (%s) desorbed: %.2f A",
                    log_prefix,
                    k,
                    winners[k].molecule,
                    dist,
                )
                return [], (f"unit {k} ({winners[k].molecule}) desorbed ({dist:.2f} A)")

    if topology_check is not None:
        ok, reason = topology_check(opt_atoms, [w.molecule for w in winners])
        if not ok:
            logger.debug(
                "%scomposite topology rearrangement guard: %s", log_prefix, reason
            )
            return [], f"topology rearrangement guard: {reason}"

    return unit_distances, ""


def _finalize_relaxed_composite(
    winners: Sequence[ScreeningResult],
    opt_atoms: Atoms | None,
    *,
    slab_atoms: Atoms,
    base_slab: Atoms,
    config: AdsorptionConfig,
    E_slab: float,
    topology_check: Callable[[Atoms, list[str]], tuple[bool, str]] | None,
    log_prefix: str,
) -> tuple[list[ScreeningResult], str]:
    """Validate + rewrite one relaxed composite; ``([], reason)`` on failure."""
    if opt_atoms is None:
        return [], "optimizer_returned_none"
    unit_distances, failure = _validate_relaxed_composite(
        opt_atoms,
        winners,
        slab_atoms=slab_atoms,
        base_slab=base_slab,
        config=config,
        topology_check=topology_check,
        log_prefix=log_prefix,
    )
    if failure:
        return [], failure
    rewritten, e_ads_total, e_ads_per_mol = _rewrite_relaxed_composite(
        winners,
        opt_atoms,
        slab_atoms=slab_atoms,
        E_slab=E_slab,
        config=config,
        unit_distances=unit_distances,
    )
    if e_ads_per_mol > config.max_adsorption_energy:
        return [], f"E_ads per molecule too high: {e_ads_per_mol:.4f} eV"
    logger.info(
        "%scomposite relaxed: %d units, E_ads/mol = %.4f eV (total %.4f eV)",
        log_prefix,
        len(rewritten),
        e_ads_per_mol,
        e_ads_total,
    )
    return rewritten, ""


def evaluate_composite_batch(
    config_groups: Sequence[Sequence[ScreeningResult]],
    *,
    slab_atoms: Atoms,
    base_slab: Atoms,
    ts_model: object,
    config: AdsorptionConfig,
    E_slab: float,
    topology_check: Callable[[Atoms, list[str]], tuple[bool, str]] | None = None,
    log_prefix: str = "",
) -> list[list[ScreeningResult]]:
    """Relax and validate many joint configs; return one result list per valid config."""
    if not config_groups:
        return []
    composites = [
        build_composite_candidate(slab_atoms, [w.atoms[w.slab_size :] for w in group])
        for group in config_groups
    ]
    optimized = optimize_adsorbate_slab_batched(
        composites,
        slab_atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab,
        saturation_reuse=True,
    )
    valid: list[list[ScreeningResult]] = []
    for group, opt_atoms in zip(config_groups, optimized, strict=True):
        rewritten, _failure = _finalize_relaxed_composite(
            group,
            opt_atoms,
            slab_atoms=slab_atoms,
            base_slab=base_slab,
            config=config,
            E_slab=E_slab,
            topology_check=topology_check,
            log_prefix=log_prefix,
        )
        if rewritten:
            valid.append(rewritten)
    return valid

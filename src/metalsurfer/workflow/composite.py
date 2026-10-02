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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..models import ScreeningResult
from ..optimization import optimize_adsorbate_slab_batched
from ..placement._material import calculator_pbc_for_atoms, material_aware_pbc
from ..placement.clash import (
    atom_radii_for_symbols,
    clash_bounds_for_adsorbate,
    compose_quaternion_with_azimuth,
    pair_floors_for_fixed_cloud,
    resolve_rigid_clash,
    tuplet_clash_rescue_floor,
)
from ..placement.geometry import (
    _mol_slab_pairwise_distances,
    calculate_min_distance,
    compute_surface_site_frame,
)
from ..placement.occupancy import _positions_mutually_clear, incoming_inplane_radius
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
    "evaluate_composite_commit",
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
    *,
    min_separation: float,
    n_substrate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coverage slab atoms plus already-accepted adsorbate suffixes.

    Returns ``(positions, radii, pair_floors)``. Bare-substrate atoms use floor
    0; pre-adsorbed coverage and packed suffixes use ``min_separation``.
    """
    fixed_pos = np.asarray(slab_atoms.get_positions(), dtype=float)
    fixed_radii = atom_radii_for_symbols(
        list(slab_atoms.get_chemical_symbols()),
        min_separation=float(min_separation),
    )
    floors = pair_floors_for_fixed_cloud(
        len(fixed_pos),
        n_substrate=int(n_substrate),
        adsorbate_separation=float(min_separation),
    )
    for other, prev in zip(suffixes, results, strict=True):
        fixed_pos = np.vstack([fixed_pos, other])
        prev_syms = list(prev.atoms.get_chemical_symbols()[prev.slab_size :])
        fixed_radii = np.concatenate(
            [
                fixed_radii,
                atom_radii_for_symbols(
                    prev_syms,
                    min_separation=float(min_separation),
                ),
            ]
        )
        floors = np.concatenate(
            [
                floors,
                np.full(len(other), float(min_separation), dtype=float),
            ]
        )
    return fixed_pos, fixed_radii, floors


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
    fixed_floors: np.ndarray,
    *,
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
    new_pos, az_delta, ok = resolve_rigid_clash(
        ads,
        fixed_pos,
        fixed_radii,
        origin=origin,
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        fixed_pair_floors=fixed_floors,
        bounds=bounds,
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
    clear under ``min_adsorbate_separation`` against the packed set, or (when
    ``placement_clash_descent`` is on) be rescued by rigid-body descent against
    the coverage slab + packed units. Overlap with clash descent off, or a
    failed rescue, rejects the whole pack. Partial packs are never returned.

    *n_substrate* is the bare-substrate atom count used for per-atom clash
    floors (defaults to ``len(slab_atoms)`` when the coverage frame is bare).
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
    min_sep = float(config.min_adsorbate_separation)
    clash_on = bool(config.placement_clash_descent)

    packed: list[ScreeningResult] = [winners[0]]
    packed_suffixes: list[np.ndarray] = [_suffix_positions(winners[0])]

    for winner in winners[1:]:
        suffix = _suffix_positions(winner)
        clear = all(
            _positions_mutually_clear(
                suffix,
                other,
                cell=cell,
                pbc=pbc,
                min_separation=min_sep,
            )
            for other in packed_suffixes
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
        rescue_floor = tuplet_clash_rescue_floor(
            cand_syms,
            fixed_syms,
            min_separation=min_sep,
        )
        if (
            _min_dist_to_suffixes(suffix, clearance_targets, cell=cell, pbc=pbc)
            < rescue_floor
        ):
            return None

        fixed_pos, fixed_radii, fixed_floors = _fixed_cloud_from_coverage_and_results(
            slab_atoms,
            packed,
            packed_suffixes,
            min_separation=min_sep,
            n_substrate=substrate_n,
        )
        rescued = _try_rescue_suffix(
            winner,
            fixed_pos,
            fixed_radii,
            fixed_floors,
            slab_atoms=slab_atoms,
            cell=cell,
            pbc=pbc,
            config=config,
        )
        if rescued is None:
            return None
        packed.append(rescued)
        packed_suffixes.append(_suffix_positions(rescued))

    return packed


def assemble_joint_config_groups(
    poses: Sequence[ScreeningResult],
    *,
    n_per_config: int,
    n_configs: int,
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    rng: np.random.Generator,
    n_substrate: int | None = None,
) -> list[list[ScreeningResult]]:
    """Build up to *n_configs* disjoint groups of exactly *n_per_config* poses.

    Greedy assembly from a shuffled pose pool; each group must pass
    :func:`pack_exact_tuplet`. Poses are consumed at most once.
    """
    if n_per_config <= 0 or n_configs <= 0 or not poses:
        return []
    remaining = list(poses)
    remaining = [remaining[int(i)] for i in rng.permutation(len(remaining))]
    groups: list[list[ScreeningResult]] = []
    while len(groups) < n_configs and remaining:
        group: list[ScreeningResult] = []
        scan = 0
        while scan < len(remaining) and len(group) < n_per_config:
            trial = group + [remaining[scan]]
            packed = pack_exact_tuplet(
                trial, slab_atoms, config, n_substrate=n_substrate
            )
            if packed is not None:
                group = packed
                remaining.pop(scan)
            else:
                scan += 1
        if len(group) != n_per_config:
            break
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
) -> list[list[ScreeningResult]]:
    """Build joint configs with exact per-species slot counts from *quotas*."""
    n_total = sum(int(q) for q in quotas.values())
    if n_total <= 0 or n_configs <= 0:
        return []
    remaining: dict[str, list[ScreeningResult]] = {
        mol: list(pools.get(mol, [])) for mol in quotas
    }
    for mol, poses in remaining.items():
        remaining[mol] = [poses[int(i)] for i in rng.permutation(len(poses))]

    recipe = [mol for mol, q in quotas.items() for _ in range(int(q))]
    groups: list[list[ScreeningResult]] = []
    while len(groups) < n_configs:
        if any(len(remaining[mol]) < int(quotas[mol]) for mol in quotas):
            break
        group: list[ScreeningResult] = []
        taken: list[tuple[str, ScreeningResult]] = []
        failed = False
        for mol in recipe:
            scan = 0
            found = False
            while scan < len(remaining[mol]):
                trial = group + [remaining[mol][scan]]
                packed = pack_exact_tuplet(
                    trial, slab_atoms, config, n_substrate=n_substrate
                )
                if packed is not None:
                    group = packed
                    taken.append((mol, remaining[mol].pop(scan)))
                    found = True
                    break
                scan += 1
            if not found:
                failed = True
                break
        if failed or len(group) != n_total:
            for mol, pose in reversed(taken):
                remaining[mol].insert(0, pose)
            break
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


def evaluate_composite_commit(
    *,
    winners: Sequence[ScreeningResult],
    slab_atoms: Atoms,
    base_slab: Atoms,
    ts_model: object,
    config: AdsorptionConfig,
    E_slab: float,
    topology_check: Callable[[Atoms, list[str]], tuple[bool, str]] | None = None,
    log_prefix: str = "",
) -> tuple[list[ScreeningResult], str]:
    """Relax and validate one composite; map results back onto per-unit rows.

    Validation mirrors ``_evaluate_optimized_candidate`` but for an n-tuplet:

    - frozen-substrate drift against the bare-substrate prefix;
    - geometry sanity via ``_validate_geometry``;
    - PER-UNIT desorption check;
    - optional connectivity-only topology guard;
    - ``max_adsorption_energy`` cap applied to per-molecule E_ads.
    """
    if not winners:
        return [], "no winners"
    composite = build_composite_candidate(
        slab_atoms, [w.atoms[w.slab_size :] for w in winners]
    )
    optimized = optimize_adsorbate_slab_batched(
        [composite],
        slab_atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab,
        saturation_reuse=True,
    )
    return _finalize_relaxed_composite(
        winners,
        optimized[0],
        slab_atoms=slab_atoms,
        base_slab=base_slab,
        config=config,
        E_slab=E_slab,
        topology_check=topology_check,
        log_prefix=log_prefix,
    )

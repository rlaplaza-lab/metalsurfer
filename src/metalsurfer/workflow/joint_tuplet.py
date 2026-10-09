"""Joint n-adsorbate screening for n-tuplet saturation steps.

Each trial is an exact-*n* clash-free config: CPU placement builds packs,
TorchSim relaxes all *n* adsorbates together, and the step commits the best
binding pack (no single-adsorbate screen / single-winner fallback). Stored
``energy_adsorption`` is per molecule (``E_ads_total / n``); ranking uses
``Ω_tuplet``. BO (homogeneous path) proposes joint configs from the
single-site surrogate and records shared ``Ω/n`` labels.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from ase import Atoms
from sklearn.preprocessing import StandardScaler

from ..config import AdsorptionConfig
from ..ml.bayesian import build_spec_features_geometry_aware
from ..ml.features import extract_features
from ..ml.schema import PlacementRecord
from ..models import (
    BOStepMemory,
    BOTransferInfo,
    PlacementDescriptor,
    ReferenceEnergies,
    ScreeningResult,
)
from ..placement.generators import (
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
)
from ..placement.site_context import SiteContext, site_context_for_sampling
from ..reporting import BOPlacementFailure, FailureSummary, PlacementFailure
from ..surface_prep import SlabContainer
from .bayesian import (
    _BOAcquisitionState,
    _occupancy_sigma_scales,
    _pack_bo_outputs,
    _TransferRoundState,
    bo_exploration_rng,
    run_bo_acquisition_loop,
)
from .composite import (
    assemble_joint_config_groups,
    assemble_quota_joint_configs,
    evaluate_composite_batch,
    pack_exact_tuplet,
)
from .placement_fill import fill_materialized_placements, materialize_specs
from .shared import (
    _build_surface_reference_slab,
    _prepare_molecule_screening,
    adsorption_ranking_energy,
    joint_config_ranking_energy,
)

__all__ = [
    "JointTupletScreenOutcome",
    "commit_best_joint_config",
    "enumerate_tuplet_compositions",
    "joint_config_ranking_energy",
    "joint_winning_molecule_label",
    "process_joint_tuplet_bayesian",
    "screen_joint_tuplet_homogeneous",
    "screen_joint_tuplet_multi",
]

TopologyCheck = Callable[[Atoms, list[str]], tuple[bool, str]]


@dataclass
class JointTupletScreenOutcome:
    """Return value for joint n-adsorbate saturation screening."""

    valid_configs: list[list[ScreeningResult]]
    flat_results: list[ScreeningResult]
    failure_summary: FailureSummary | None = None
    bo_memory: BOStepMemory | None = None
    transfer_info: BOTransferInfo | None = None
    slots_by_molecule: dict[str, int] | None = None


def joint_winning_molecule_label(group: Sequence[ScreeningResult]) -> str:
    """Species label for a committed joint pack (``A`` or ``A+B`` in pack order)."""
    if not group:
        raise ValueError("joint winning label requires a non-empty group")
    if len(group) == 1:
        return group[0].molecule
    return "+".join(row.molecule for row in group)


def enumerate_tuplet_compositions(
    molecules: Sequence[str],
    n: int,
) -> list[dict[str, int]]:
    """All non-negative integer count maps with ``sum(counts) == n``.

    Molecule order is preserved from *molecules*. Empty *molecules* yields
    ``[{}]`` only when *n* is 0.
    """
    names = list(molecules)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not names:
        return [{}] if n == 0 else []
    if len(names) == 1:
        return [{names[0]: int(n)}]

    compositions: list[dict[str, int]] = []

    def _recurse(remaining: int, index: int, current: dict[str, int]) -> None:
        if index == len(names) - 1:
            current[names[index]] = remaining
            compositions.append(dict(current))
            return
        for count in range(remaining + 1):
            current[names[index]] = count
            _recurse(remaining - count, index + 1, current)

    _recurse(int(n), 0, {})
    return compositions


def _composition_funding_order(
    compositions: Sequence[Mapping[str, int]],
    molecules: Sequence[str],
) -> list[int]:
    """Return indices with each species' pure pack first, then mixed packs."""
    names = list(molecules)
    pure_indices: list[int] = []
    for mol in names:
        for i, counts in enumerate(compositions):
            if int(counts.get(mol, 0)) > 0 and all(
                int(counts.get(other, 0)) == 0 for other in names if other != mol
            ):
                pure_indices.append(i)
                break
    pure_set = set(pure_indices)
    mixed_indices = [i for i in range(len(compositions)) if i not in pure_set]
    return pure_indices + mixed_indices


def _equal_split_budget(total: int, n_parts: int) -> list[int]:
    """Largest-remainder equal split; zero shares allowed when ``total < n_parts``."""
    if n_parts <= 0:
        raise ValueError(f"n_parts must be positive, got {n_parts}")
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if total < n_parts:
        return [1 if i < total else 0 for i in range(n_parts)]
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def _composition_budgets(
    compositions: Sequence[Mapping[str, int]],
    molecules: Sequence[str],
    n_configs: int,
) -> list[int]:
    """Per-composition joint-config shares; pures funded before mixtures."""
    order = _composition_funding_order(compositions, molecules)
    shares_in_order = _equal_split_budget(int(n_configs), len(compositions))
    budgets = [0] * len(compositions)
    for share_i, comp_i in enumerate(order):
        budgets[comp_i] = shares_in_order[share_i]
    return budgets


def commit_best_joint_config(
    valid_configs: Sequence[Sequence[ScreeningResult]],
    *,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[ScreeningResult | None, list[ScreeningResult], list[ScreeningResult]]:
    """Return ``(pool_best, committed, ranked_group)`` for the best joint config.

    ``committed`` is empty when no config binds (Ω ≥ 0). ``ranked_group`` is
    always the Ω-best pack (for logging unbound steps). ``pool_best`` is the
    best-ranked unit in that pack (for step bookkeeping).
    """
    if not valid_configs:
        return None, [], []

    def _rank(group: Sequence[ScreeningResult]) -> float:
        return joint_config_ranking_energy(
            group,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        )

    ordered = sorted(valid_configs, key=_rank)
    best_group = list(ordered[0])
    pool_best = min(
        best_group,
        key=lambda r: (
            adsorption_ranking_energy(
                r.energy_adsorption,
                activity_by_molecule[r.molecule],
                temperature,
                pressure,
            ),
            r.placement_id,
            r.molecule,
        ),
    )
    if _rank(best_group) >= 0:
        return pool_best, [], best_group
    return pool_best, best_group, best_group


def _unrelaxed_pose_stub(
    *,
    combined: Atoms,
    placement_id: int,
    descriptor: PlacementDescriptor,
    molecule_name: str,
    E_slab: float,
    E_mol: float,
    slab_size: int,
) -> ScreeningResult:
    """Placement-only row before joint TorchSim relaxation."""
    return ScreeningResult(
        molecule=molecule_name,
        placement_id=placement_id,
        energy_adslab=E_slab + E_mol,
        energy_slab=E_slab,
        energy_adsorbate=E_mol,
        energy_adsorption=0.0,
        atoms=combined,
        slab_size=slab_size,
        distance=0.0,
        placement_descriptor=descriptor,
    )


def _materialize_pose_pool(
    *,
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    base_slab: Atoms,
    calculator,
    conformers: list[Atoms],
    conformer_energies: list[float] | None,
    config: AdsorptionConfig,
    site_context: SiteContext,
    E_slab: float,
    E_mol: float,
    pool_size: int,
) -> list[ScreeningResult]:
    fill = fill_materialized_placements(
        conformers=conformers,
        slab_for_sites=_build_surface_reference_slab(slab.atoms, base_slab),
        config=replace(config, num_placements=pool_size),
        smiles=smiles,
        site_context=site_context,
        slab_atoms=slab.atoms,
        calculator=calculator,
        conformer_energies=conformer_energies,
    )
    slab_size = len(slab.atoms)
    return [
        _unrelaxed_pose_stub(
            combined=combined,
            placement_id=pid,
            descriptor=desc,
            molecule_name=molecule_name,
            E_slab=E_slab,
            E_mol=E_mol,
            slab_size=slab_size,
        )
        for combined, pid, desc in zip(
            fill.combined, fill.placement_ids, fill.descriptors, strict=True
        )
    ]


def _flatten_configs(
    configs: Sequence[Sequence[ScreeningResult]],
) -> list[ScreeningResult]:
    return [row for group in configs for row in group]


def _relax_groups(
    groups: Sequence[Sequence[ScreeningResult]],
    *,
    slab_atoms: Atoms,
    base_slab: Atoms,
    ts_model: object,
    config: AdsorptionConfig,
    E_slab: float,
    topology_check: TopologyCheck | None,
    log_prefix: str,
    slots_by_molecule: dict[str, int] | None = None,
) -> JointTupletScreenOutcome:
    valid = evaluate_composite_batch(
        groups,
        slab_atoms=slab_atoms,
        base_slab=base_slab,
        ts_model=ts_model,
        config=config,
        E_slab=E_slab,
        topology_check=topology_check,
        log_prefix=log_prefix,
    )
    return JointTupletScreenOutcome(
        valid_configs=valid,
        flat_results=_flatten_configs(valid),
        slots_by_molecule=slots_by_molecule,
    )


def screen_joint_tuplet_homogeneous(
    *,
    smiles: str,
    molecule_name: str,
    current_slab: SlabContainer,
    calculator,
    ref_step: ReferenceEnergies,
    ts_model: object,
    config: AdsorptionConfig,
    base_slab: Atoms,
    E_slab: float,
    symmetry_broken: bool,
    conformers: list[Atoms],
    conformer_energies: list[float] | None,
    site_context: SiteContext | None,
    topology_check: TopologyCheck | None = None,
    debug_sites_step: int | None = None,
    log_prefix: str = "",
) -> JointTupletScreenOutcome:
    """CPU placement + batched joint relaxation for one adsorbate species."""
    if config.num_placements is None:
        raise ValueError("num_placements must be set")
    if ref_step.get_molecule_energy(molecule_name) is None:
        raise ValueError(f"missing reference energy for {molecule_name!r}")

    slab_for_sites = _build_surface_reference_slab(current_slab.atoms, base_slab)
    ctx = site_context_for_sampling(
        slab_for_sites,
        config,
        site_context=site_context,
        symmetry_broken=symmetry_broken,
        full_slab=current_slab.atoms,
    )
    assembly_seed = (
        config.seed if debug_sites_step is None else config.seed + debug_sites_step
    )
    return screen_joint_tuplet_multi(
        active_molecules=[molecule_name],
        active_smiles={molecule_name: smiles},
        conformer_cache={molecule_name: (conformers, conformer_energies)},
        current_slab=current_slab,
        calculator=calculator,
        ref_step=ref_step,
        ts_model=ts_model,
        config=config,
        base_slab=base_slab,
        E_slab=E_slab,
        site_context=ctx,
        topology_check=topology_check,
        log_prefix=log_prefix,
        assembly_seed=assembly_seed,
        empty_pool_failure=True,
        assemble="exact",
    )


def screen_joint_tuplet_multi(
    *,
    active_molecules: Sequence[str],
    active_smiles: Mapping[str, str],
    conformer_cache: Mapping[str, tuple[list[Atoms], list[float] | None]],
    current_slab: SlabContainer,
    calculator,
    ref_step: ReferenceEnergies,
    ts_model: object,
    config: AdsorptionConfig,
    base_slab: Atoms,
    E_slab: float,
    site_context: SiteContext,
    topology_check: TopologyCheck | None = None,
    log_prefix: str = "",
    assembly_seed: int | None = None,
    empty_pool_failure: bool = False,
    assemble: Literal["quota", "exact"] = "quota",
) -> JointTupletScreenOutcome:
    """Joint screening over every species composition of size *n*.

    Each exact-*n* count map across *active_molecules* receives an equal share
    of ``num_placements``. Pose pools are sized from the slots each species
    fills across those shares (homogeneous oversample). Valid packs from every
    composition are ranked together by ``Ω_tuplet``.

    ``assemble="exact"`` uses :func:`assemble_joint_config_groups` (homogeneous
    path). Mixed campaigns keep the default ``"quota"`` assembler.
    """
    n = config.saturation_molecules_per_step
    n_configs = config.num_placements
    if n_configs is None:
        raise ValueError("num_placements must be set")

    compositions = enumerate_tuplet_compositions(active_molecules, n)
    if not compositions:
        return JointTupletScreenOutcome(valid_configs=[], flat_results=[])

    per_comp_budgets = _composition_budgets(
        compositions, active_molecules, int(n_configs)
    )
    oversample = float(config.placement_retry_oversample_max)
    slots_by_mol: dict[str, int] = {mol: 0 for mol in active_molecules}
    for quotas, n_share in zip(compositions, per_comp_budgets, strict=True):
        if n_share <= 0:
            continue
        for mol, count in quotas.items():
            slots_by_mol[mol] += int(count) * int(n_share)

    pools: dict[str, list[ScreeningResult]] = {}
    for mol in active_molecules:
        slots = slots_by_mol.get(mol, 0)
        if slots <= 0:
            pools[mol] = []
            continue
        E_mol = ref_step.get_molecule_energy(mol)
        if E_mol is None:
            pools[mol] = []
            continue
        confs, conf_energies = conformer_cache[mol]
        pool_size = max(1, int(math.ceil(slots * oversample)))
        pools[mol] = _materialize_pose_pool(
            smiles=active_smiles[mol],
            molecule_name=mol,
            slab=current_slab,
            base_slab=base_slab,
            calculator=calculator,
            conformers=confs,
            conformer_energies=conf_energies,
            config=config,
            site_context=site_context,
            E_slab=E_slab,
            E_mol=E_mol,
            pool_size=pool_size,
        )

    if empty_pool_failure and all(not pools.get(mol) for mol in active_molecules):
        return JointTupletScreenOutcome(
            valid_configs=[],
            flat_results=[],
            failure_summary=PlacementFailure(
                n_placements_attempted=n_configs,
                n_initial_placements=0,
            ),
            slots_by_molecule=dict(slots_by_mol),
        )

    seed = config.seed if assembly_seed is None else assembly_seed
    rng = np.random.default_rng(seed)

    if assemble == "exact":
        if len(active_molecules) != 1:
            raise ValueError(
                'assemble="exact" requires exactly one active molecule, '
                f"got {list(active_molecules)!r}"
            )
        groups = assemble_joint_config_groups(
            pools[active_molecules[0]],
            n_per_config=n,
            n_configs=n_configs,
            slab_atoms=current_slab.atoms,
            config=config,
            rng=rng,
            n_substrate=len(base_slab),
        )
    else:
        groups = []
        for quotas, n_share in zip(compositions, per_comp_budgets, strict=True):
            if n_share <= 0:
                continue
            active_quotas = {mol: int(c) for mol, c in quotas.items() if int(c) > 0}
            groups.extend(
                assemble_quota_joint_configs(
                    pools,
                    quotas=active_quotas,
                    n_configs=n_share,
                    slab_atoms=current_slab.atoms,
                    config=config,
                    rng=rng,
                    n_substrate=len(base_slab),
                )
            )
    return _relax_groups(
        groups,
        slab_atoms=current_slab.atoms,
        base_slab=base_slab,
        ts_model=ts_model,
        config=config,
        E_slab=E_slab,
        topology_check=topology_check,
        log_prefix=log_prefix,
        slots_by_molecule=dict(slots_by_mol),
    )


def _assemble_bo_joint_groups_from_specs(
    anchor_indices: Sequence[int],
    *,
    all_specs,
    valid_spec_indices: Sequence[int],
    materialization_cache: dict,
    conformers: list[Atoms],
    slab: SlabContainer,
    calculator,
    config: AdsorptionConfig,
    smiles: str,
    molecule_name: str,
    site_context: SiteContext,
    slab_for_sites: Atoms,
    E_slab: float,
    E_mol: float,
    n_per_config: int,
    rng: np.random.RandomState,
    n_substrate: int,
    blocked: set[int] | None = None,
) -> tuple[list[list[ScreeningResult]], list[set[int]]]:
    """Build one joint group per anchor (forward companion scan).

    Returns ``(groups, pool_index_sets)`` aligned by successful pack. Incomplete
    packs release their tentative companions so later anchors can use them.
    *blocked* pool indices are never chosen as companions.
    """
    slab_size = len(slab.atoms)
    groups: list[list[ScreeningResult]] = []
    pool_index_sets: list[set[int]] = []
    committed: set[int] = set(blocked or ())

    def _stub_from_pool(pool_pos: int) -> ScreeningResult | None:
        fill = materialize_specs(
            specs=[all_specs[valid_spec_indices[pool_pos]]],
            n_target=1,
            conformers=conformers,
            slab_atoms=slab.atoms,
            calculator=calculator,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
            materialization_cache=materialization_cache,
        )
        if not fill.combined:
            return None
        return _unrelaxed_pose_stub(
            combined=fill.combined[0],
            placement_id=fill.placement_ids[0],
            descriptor=fill.descriptors[0],
            molecule_name=molecule_name,
            E_slab=E_slab,
            E_mol=E_mol,
            slab_size=slab_size,
        )

    for anchor_pos in anchor_indices:
        if anchor_pos in committed:
            continue
        anchor = _stub_from_pool(anchor_pos)
        if anchor is None:
            continue
        group: list[ScreeningResult] = [anchor]
        tentative: set[int] = {anchor_pos}
        candidates = [
            p
            for p in range(len(valid_spec_indices))
            if p not in committed and p not in tentative
        ]
        rng.shuffle(candidates)
        scan = 0
        while len(group) < n_per_config and scan < len(candidates):
            pos = candidates[scan]
            stub = _stub_from_pool(pos)
            if stub is None:
                scan += 1
                continue
            packed = pack_exact_tuplet(
                group + [stub],
                slab.atoms,
                config,
                n_substrate=n_substrate,
            )
            if packed is None:
                scan += 1
                continue
            group = packed
            tentative.add(pos)
            scan += 1
        if len(group) == n_per_config:
            groups.append(group)
            pool_index_sets.append(set(tentative))
            committed.update(tentative)
    return groups, pool_index_sets


def process_joint_tuplet_bayesian(
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    calculator,
    reference_energies: ReferenceEnergies,
    ts_model=None,
    *,
    config: AdsorptionConfig,
    surface_type: str = "manual",
    base_slab_for_frozen: Atoms | None = None,
    slab_energy_override: float | None = None,
    symmetry_broken: bool = False,
    bo_step_memory_in: BOStepMemory | None = None,
    conformers: list[Atoms] | None = None,
    conformer_energies: list[float] | None = None,
    skip_workload_autotune: bool = False,
    site_context: SiteContext | None = None,
    debug_sites_step: int | None = None,
    topology_check: TopologyCheck | None = None,
    activity_by_molecule: Mapping[str, float] | None = None,
) -> JointTupletScreenOutcome:
    """BO-guided joint n-adsorbate screening (single adsorbate species)."""
    if activity_by_molecule is None:
        activity_by_molecule = {molecule_name: 1.0}

    n = config.saturation_molecules_per_step
    temperature = config.saturation_temperature
    pressure = config.saturation_pressure

    ctx, early_failure = _prepare_molecule_screening(
        smiles=smiles,
        molecule_name=molecule_name,
        slab=slab,
        calculator=calculator,
        reference_energies=reference_energies,
        ts_model=ts_model,
        config=config,
        base_slab_for_frozen=base_slab_for_frozen,
        slab_energy_override=slab_energy_override,
        symmetry_broken=symmetry_broken,
        bo_enabled=True,
        conformers=conformers,
        conformer_energies=conformer_energies,
        skip_workload_autotune=skip_workload_autotune,
        site_context=site_context,
        surface_type=surface_type,
        debug_sites_step=debug_sites_step,
    )
    if ctx is None:
        assert early_failure is not None
        return JointTupletScreenOutcome([], [], failure_summary=early_failure)

    slab = ctx.slab
    slab_for_sites = ctx.slab_for_sites
    effective_base_slab_for_frozen = ctx.effective_base_slab_for_frozen
    conformers = ctx.conformers
    conformer_energies = ctx.conformer_energies
    site_context = ctx.site_context
    config = ctx.config
    E_slab = ctx.E_slab
    E_mol = ctx.E_mol
    if site_context is None:
        raise RuntimeError("joint BO screening requires a resolved site_context")

    if config.bo.initial_random is None or config.bo.batch_size is None:
        raise ValueError("bo.initial_random and bo.batch_size required for joint BO")
    if config.num_placements is None:
        raise ValueError("num_placements required for joint BO")

    max_enumerated_specs = estimate_placement_spec_capacity(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
        full_slab=slab.atoms,
    )
    pool_size = (
        config.bo.candidate_pool_size
        if config.bo.candidate_pool_size is not None
        else max_enumerated_specs
    )
    if pool_size <= 0:
        return JointTupletScreenOutcome(
            [],
            [],
            failure_summary=BOPlacementFailure(n_candidate_specs=0, n_valid_pool=0),
        )

    all_specs = enumerate_placement_specs(
        conformers,
        slab_for_sites,
        config,
        smiles,
        pool_size,
        filter_spec=config.placement_filter,
        site_context=site_context,
        seed=config.seed,
        full_slab=slab.atoms,
        conformer_energies=conformer_energies,
    )
    if not all_specs:
        return JointTupletScreenOutcome(
            [],
            [],
            failure_summary=BOPlacementFailure(n_candidate_specs=0, n_valid_pool=0),
        )

    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] = {}
    candidate_features, valid_spec_indices = build_spec_features_geometry_aware(
        all_specs,
        conformers,
        slab.atoms,
        config,
        molecule=molecule_name,
        smiles=smiles,
        surface_id=surface_type,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
        materialization_cache=materialization_cache,
    )
    if candidate_features.empty:
        return JointTupletScreenOutcome(
            [],
            [],
            failure_summary=BOPlacementFailure(
                n_candidate_specs=len(all_specs),
                n_valid_pool=0,
            ),
        )

    scaled_candidate_features = StandardScaler().fit_transform(
        candidate_features.to_numpy(dtype=float)
    )
    valid_configs: list[list[ScreeningResult]] = []
    rng = bo_exploration_rng(config.seed, len(slab.atoms), molecule=molecule_name)
    transfer_state = _TransferRoundState()
    acq_state = _BOAcquisitionState()
    base_slab = effective_base_slab_for_frozen or slab.atoms
    surface_prefix = len(slab_for_sites)
    occupied = (
        slab.atoms[surface_prefix:] if len(slab.atoms) > surface_prefix else Atoms()
    )
    sigma_scale = _occupancy_sigma_scales(
        candidate_features=candidate_features,
        valid_spec_indices=valid_spec_indices,
        all_specs=all_specs,
        materialization_cache=materialization_cache,
        occupied=occupied,
        material_type=config.material_type,
        cell=np.asarray(slab.atoms.get_cell(), dtype=float),
        connectivity_multiplier=float(config.connectivity_multiplier),
    )

    def _record_group(group: Sequence[ScreeningResult]) -> None:
        label = joint_config_ranking_energy(
            group,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        ) / len(group)
        for row in group:
            features = extract_features(
                PlacementRecord.from_descriptor(
                    row.placement_descriptor,
                    molecule=molecule_name,
                    smiles=smiles,
                    surface_id=surface_type,
                    config=config,
                )
            )
            acq_state.observed_X_rows.append(features)
            acq_state.observed_y.append(label)
            if label < acq_state.best_energy:
                acq_state.best_energy = label
                acq_state.best_X_row = dict(features)
        acq_state.n_independent_trials += 1

    def _eval_anchor_batch(anchor_positions: list[int]) -> None:
        fresh = [p for p in anchor_positions if p not in acq_state.evaluated]
        if not fresh:
            return
        groups, pool_sets = _assemble_bo_joint_groups_from_specs(
            fresh,
            all_specs=all_specs,
            valid_spec_indices=valid_spec_indices,
            materialization_cache=materialization_cache,
            conformers=conformers,
            slab=slab,
            calculator=calculator,
            config=config,
            smiles=smiles,
            molecule_name=molecule_name,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
            E_slab=E_slab,
            E_mol=E_mol,
            n_per_config=n,
            rng=rng,
            n_substrate=len(base_slab),
            blocked=acq_state.evaluated,
        )
        acq_state.evaluated.update(fresh)
        for pool_set in pool_sets:
            acq_state.evaluated.update(pool_set)
        acq_state.total_evaluated += len(fresh)
        if not groups:
            return
        for group in evaluate_composite_batch(
            groups,
            slab_atoms=slab.atoms,
            base_slab=base_slab,
            ts_model=ts_model,
            config=config,
            E_slab=E_slab,
            topology_check=topology_check,
            log_prefix="Joint BO | ",
        ):
            valid_configs.append(group)
            _record_group(group)

    run_bo_acquisition_loop(
        config=config,
        candidate_features=candidate_features,
        scaled_candidate_features=scaled_candidate_features,
        n_pool=len(valid_spec_indices),
        molecule_name=molecule_name,
        slab_atoms=slab.atoms,
        bo_step_memory_in=bo_step_memory_in,
        transfer_state=transfer_state,
        state=acq_state,
        evaluate_batch=_eval_anchor_batch,
        sigma_scale=sigma_scale,
        rng=rng,
    )

    bo_memory, transfer_info = _pack_bo_outputs(acq_state, transfer_state)
    return JointTupletScreenOutcome(
        valid_configs=valid_configs,
        flat_results=_flatten_configs(valid_configs),
        bo_memory=bo_memory,
        transfer_info=transfer_info,
    )

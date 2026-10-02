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

import numpy as np
import pandas as pd
from ase import Atoms
from sklearn.preprocessing import StandardScaler

from ..config import AdsorptionConfig, resolved_bo_eval_budget
from ..filters import (
    min_interadsorbate_covalent_ratio,
    occupancy_sigma_scale,
)
from ..ml.bayesian import (
    build_spec_features_geometry_aware,
    score_and_select,
    select_initial_bo_indices,
    splice_exploration_picks,
)
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
from .bayesian import _build_round_surrogate, _TransferRoundState, bo_exploration_rng
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
    tuplet_ranking_energy,
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


def joint_config_ranking_energy(
    group: Sequence[ScreeningResult],
    *,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> float:
    """Ω (single unit) or Ω_tuplet (joint config) for ranking commits."""
    if not group:
        return 0.0
    if len(group) == 1:
        row = group[0]
        return adsorption_ranking_energy(
            row.energy_adsorption,
            activity_by_molecule[row.molecule],
            temperature,
            pressure,
        )
    return tuplet_ranking_energy(
        float(group[0].energy_adsorption) * len(group),
        [row.molecule for row in group],
        activity_by_molecule,
        temperature,
        pressure,
    )


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


def _equal_split_budget(total: int, n_parts: int) -> list[int]:
    """Largest-remainder equal split; requires ``total >= n_parts``."""
    if n_parts <= 0:
        raise ValueError(f"n_parts must be positive, got {n_parts}")
    if total < n_parts:
        raise ValueError(
            f"joint-config budget ({total}) is smaller than the number of "
            f"compositions ({n_parts}); raise num_placements or lower "
            "saturation_molecules_per_step"
        )
    base = total // n_parts
    rem = total % n_parts
    return [base + (1 if i < rem else 0) for i in range(n_parts)]


def commit_best_joint_config(
    valid_configs: Sequence[Sequence[ScreeningResult]],
    *,
    activity_by_molecule: Mapping[str, float],
    temperature: float,
    pressure: float,
) -> tuple[ScreeningResult | None, list[ScreeningResult]]:
    """Return ``(pool_best, committed)`` for the best binding joint config.

    ``committed`` is empty when no config binds (Ω ≥ 0). ``pool_best`` is the
    best-ranked unit among all valid configs (for step bookkeeping).
    """
    if not valid_configs:
        return None, []

    def _rank(group: Sequence[ScreeningResult]) -> float:
        return joint_config_ranking_energy(
            group,
            activity_by_molecule=activity_by_molecule,
            temperature=temperature,
            pressure=pressure,
        )

    ordered = sorted(valid_configs, key=_rank)
    best_group = ordered[0]
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
        return pool_best, []
    return pool_best, list(best_group)


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


def _pool_size_for_joint_screen(config: AdsorptionConfig) -> int:
    n_configs = config.num_placements
    if n_configs is None:
        raise ValueError("num_placements must be set for joint tuplet screening")
    n = config.saturation_molecules_per_step
    return max(
        1, int(math.ceil(n_configs * n * float(config.placement_retry_oversample_max)))
    )


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
    n = config.saturation_molecules_per_step
    n_configs = config.num_placements
    if n_configs is None:
        raise ValueError("num_placements must be set")

    E_mol = ref_step.get_molecule_energy(molecule_name)
    if E_mol is None:
        raise ValueError(f"missing reference energy for {molecule_name!r}")

    slab_for_sites = _build_surface_reference_slab(current_slab.atoms, base_slab)
    ctx = site_context_for_sampling(
        slab_for_sites,
        config,
        site_context=site_context,
        symmetry_broken=symmetry_broken,
        full_slab=current_slab.atoms,
    )
    poses = _materialize_pose_pool(
        smiles=smiles,
        molecule_name=molecule_name,
        slab=current_slab,
        base_slab=base_slab,
        calculator=calculator,
        conformers=conformers,
        conformer_energies=conformer_energies,
        config=config,
        site_context=ctx,
        E_slab=E_slab,
        E_mol=E_mol,
        pool_size=_pool_size_for_joint_screen(config),
    )
    if not poses:
        return JointTupletScreenOutcome(
            valid_configs=[],
            flat_results=[],
            failure_summary=PlacementFailure(
                n_placements_attempted=n_configs,
                n_initial_placements=0,
            ),
        )

    seed = config.seed if debug_sites_step is None else config.seed + debug_sites_step
    groups = assemble_joint_config_groups(
        poses,
        n_per_config=n,
        n_configs=n_configs,
        slab_atoms=current_slab.atoms,
        config=config,
        rng=np.random.default_rng(seed),
        n_substrate=len(base_slab),
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
    )


def screen_joint_tuplet_multi(
    *,
    active_molecules: Sequence[str],
    active_smiles: Mapping[str, str],
    conformer_cache: Mapping[str, tuple[list[Atoms], list[float]]],
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
) -> JointTupletScreenOutcome:
    """Joint screening over every species composition of size *n*.

    Each exact-*n* count map across *active_molecules* receives an equal share
    of ``num_placements``. Pose pools are sized from the slots each species
    fills across those shares (homogeneous oversample). Valid packs from every
    composition are ranked together by ``Ω_tuplet``.
    """
    n = config.saturation_molecules_per_step
    n_configs = config.num_placements
    if n_configs is None:
        raise ValueError("num_placements must be set")

    compositions = enumerate_tuplet_compositions(active_molecules, n)
    if not compositions:
        return JointTupletScreenOutcome(valid_configs=[], flat_results=[])

    per_comp_budgets = _equal_split_budget(int(n_configs), len(compositions))
    oversample = float(config.placement_retry_oversample_max)
    slots_by_mol: dict[str, int] = {mol: 0 for mol in active_molecules}
    for quotas, n_share in zip(compositions, per_comp_budgets, strict=True):
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

    groups: list[list[ScreeningResult]] = []
    rng = np.random.default_rng(config.seed)
    for quotas, n_share in zip(compositions, per_comp_budgets, strict=True):
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
) -> list[list[ScreeningResult]]:
    """Build one joint group per anchor index (greedy companions)."""
    slab_size = len(slab.atoms)
    groups: list[list[ScreeningResult]] = []
    used_pool: set[int] = set()

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
        if anchor_pos in used_pool:
            continue
        anchor = _stub_from_pool(anchor_pos)
        if anchor is None:
            continue
        group: list[ScreeningResult] = [anchor]
        used_pool.add(anchor_pos)
        candidates = [p for p in range(len(valid_spec_indices)) if p not in used_pool]
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
            used_pool.add(pos)
            candidates = [
                p for p in range(len(valid_spec_indices)) if p not in used_pool
            ]
            rng.shuffle(candidates)
            scan = 0
        if len(group) == n_per_config:
            groups.append(group)
    return groups


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
    evaluated_anchors: set[int] = set()
    valid_configs: list[list[ScreeningResult]] = []
    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    best_energy = float("inf")
    best_X_row: dict[str, float] | None = None
    rng = bo_exploration_rng(config.seed, len(slab.atoms), molecule=molecule_name)
    transfer_state = _TransferRoundState()
    bo_eval_budget = resolved_bo_eval_budget(config)
    total_evaluated = 0
    base_slab = effective_base_slab_for_frozen or slab.atoms

    def _record_group(group: Sequence[ScreeningResult]) -> None:
        nonlocal best_energy, best_X_row
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
            observed_X_rows.append(features)
            observed_y.append(label)
            if label < best_energy:
                best_energy = label
                best_X_row = dict(features)

    def _eval_anchor_batch(anchor_positions: list[int]) -> None:
        nonlocal total_evaluated
        fresh = [p for p in anchor_positions if p not in evaluated_anchors]
        if not fresh:
            return
        groups = _assemble_bo_joint_groups_from_specs(
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
        )
        evaluated_anchors.update(fresh)
        total_evaluated += len(fresh)
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

    n_initial = min(config.bo.initial_random, len(valid_spec_indices))
    initial_seed = int(
        bo_exploration_rng(
            config.seed, len(slab.atoms), molecule=molecule_name
        ).randint(0, 2**31 - 1)
    )
    _eval_anchor_batch(
        select_initial_bo_indices(
            candidate_features,
            n_initial,
            sampling=config.bo.initial_sampling,
            random_state=initial_seed,
        )
    )

    batches_run = 0
    while batches_run < config.bo.total_budget and total_evaluated < bo_eval_budget:
        unevaluated = [
            p for p in range(len(valid_spec_indices)) if p not in evaluated_anchors
        ]
        if not unevaluated:
            break
        if len(observed_X_rows) < 3:
            n_extra = min(config.bo.batch_size, len(unevaluated))
            next_anchors = rng.choice(unevaluated, size=n_extra, replace=False).tolist()
        else:
            surrogate, transfer_active = _build_round_surrogate(
                X_current=pd.DataFrame(observed_X_rows),
                y_current=np.array(observed_y),
                transfer_memory=bo_step_memory_in,
                state=transfer_state,
                config=config,
            )
            batch_size = min(config.bo.batch_size, len(unevaluated))
            acquisition = config.bo.acquisition
            f_best = best_energy if np.isfinite(best_energy) else None
            if acquisition in ("ei", "pi") and f_best is None:
                acquisition = "lcb"
            surface_prefix = len(slab_for_sites)
            occupied = (
                slab.atoms[surface_prefix:]
                if len(slab.atoms) > surface_prefix
                else Atoms()
            )
            cell_arr = np.asarray(slab.atoms.get_cell(), dtype=float)
            sigma_scale = np.ones(len(candidate_features), dtype=float)
            if len(occupied) > 0:
                for pool_i, spec_i in enumerate(valid_spec_indices):
                    spec = all_specs[spec_i]
                    cached = materialization_cache.get(int(spec.placement_index))
                    if cached is None:
                        continue
                    ads, _desc = cached
                    ratio = min_interadsorbate_covalent_ratio(
                        ads,
                        occupied,
                        material_type=config.material_type,
                        cell=cell_arr,
                    )
                    sigma_scale[pool_i] = occupancy_sigma_scale(
                        ratio, float(config.connectivity_multiplier)
                    )
            next_anchors = score_and_select(
                surrogate,
                candidate_features,
                batch_size=batch_size,
                kappa=config.bo.ucb_kappa,
                evaluated_indices=evaluated_anchors,
                acquisition=acquisition,
                f_best=f_best,
                scaled_features=scaled_candidate_features,
                n_jobs=config.n_jobs,
                sigma_scale=sigma_scale,
            )
            if transfer_active and config.bo.transfer.exploration_fraction > 0:
                next_anchors = splice_exploration_picks(
                    rng,
                    next_anchors,
                    pool_size=len(valid_spec_indices),
                    evaluated_indices=evaluated_anchors,
                    exploration_fraction=config.bo.transfer.exploration_fraction,
                )
        if not next_anchors:
            break
        _eval_anchor_batch(next_anchors)
        batches_run += 1

    transfer_info = BOTransferInfo(
        transfer_used=bool(transfer_state.used_rounds > 0),
        transfer_disabled_reason=transfer_state.disabled_reason,
        transfer_bad_rounds=int(transfer_state.bad_rounds),
        transfer_last_mae_delta=transfer_state.last_mae_delta,
        transfer_weight_share=float(transfer_state.weight_share),
    )
    return JointTupletScreenOutcome(
        valid_configs=valid_configs,
        flat_results=_flatten_configs(valid_configs),
        bo_memory=BOStepMemory(
            observed_X_rows=[dict(r) for r in observed_X_rows],
            observed_y=[float(v) for v in observed_y],
            best_energy=best_energy if np.isfinite(best_energy) else None,
            best_X_row=dict(best_X_row) if best_X_row is not None else None,
        ),
        transfer_info=transfer_info,
    )

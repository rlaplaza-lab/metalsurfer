"""Bayesian optimisation workflow orchestration."""

import logging
import zlib
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from ase import Atoms
from sklearn.preprocessing import StandardScaler

from ..config import (
    AdsorptionConfig,
    resolved_bo_eval_budget,
)
from ..ml.bayesian import (
    TransferCapableSurrogateType,
    _align_to_columns,
    build_spec_features_geometry_aware,
    build_transfer_surrogate,
    cumulative_refit_training_set,
    score_and_select,
    select_initial_bo_indices,
    splice_exploration_picks,
    train_surrogate,
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
from ..reporting import (
    BOPlacementFailure,
    BOValidationFailure,
    FailureSummary,
    FilterFailure,
)
from ..surface_prep import SlabContainer
from .core import _evaluate_placement_batch
from .shared import (
    MoleculeScreenOutcome,
    PlacementFailureEvent,
    _filter_and_label_duplicates,
    _infer_surface_symbols,
    _prepare_molecule_screening,
    _summarize_failure_events,
)

logger = logging.getLogger(__name__)


def _train_surrogate_for_bo(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    config: AdsorptionConfig,
    sample_weight: np.ndarray | None = None,
):
    """Fit BO surrogate. Per-sample weights for transfer-capable surrogates."""
    return train_surrogate(
        X,
        y,
        surrogate=config.bo.surrogate,
        n_estimators=config.bo.n_estimators,
        random_state=config.seed,
        sample_weight=sample_weight,
        n_jobs=config.n_jobs,
    )


def bo_exploration_rng(
    config_seed: int,
    n_slab_atoms: int,
    *,
    molecule: str | None = None,
) -> np.random.RandomState:
    """Coverage- and molecule-decorrelated RNG stream for BO exploration.

    The slab atom count grows across saturation steps, so successive
    saturation molecules draw from decorrelated streams. In multi-molecule
    steps every adsorbate shares the *same* slab, so *molecule* is mixed into
    the seed too; without it, competing molecules would replay identical
    random/exploration draws against their respective pools.
    """
    mol_key = 0 if molecule is None else zlib.crc32(molecule.encode("utf-8"))
    combined = int(config_seed) + 1_000_003 * int(n_slab_atoms) + 7_385_609 * mol_key
    return np.random.RandomState(combined % (2**31))


@dataclass
class _TransferRoundState:
    """Mutable cross-step transfer bookkeeping for one BO screening call."""

    used_rounds: int = 0
    bad_rounds: int = 0
    disabled: bool = False
    disabled_reason: str | None = None
    last_mae_delta: float | None = None
    weight_share: float = 0.0


def _build_round_surrogate(
    *,
    X_current: pd.DataFrame,
    y_current: np.ndarray,
    transfer_memory: BOStepMemory | None,
    state: _TransferRoundState,
    config: AdsorptionConfig,
    occupancy_placement_X: list[dict[str, float]] | None,
) -> tuple[Any, bool]:
    """Fit one acquisition round's surrogate with optional cross-step transfer.

    Supports both transfer modes (``weighted`` incremental trust gating and
    ``cumulative_refit`` proximity-weighted refits with the same MAE-based
    trust gate) and falls back to a plain current-observation fit whenever
    transfer is unavailable or disabled.

    Returns ``(surrogate, transfer_active)`` where *transfer_active* reports
    that transfer was attempted and remains enabled (it drives random
    exploration within the batch).
    """
    surrogate = None
    transfer = config.bo.transfer
    can_try_transfer = (
        transfer.enabled
        and transfer_memory is not None
        and len(transfer_memory.observed_X_rows) > 0
        and len(transfer_memory.observed_y) > 0
    )
    can_try_weighted = (
        can_try_transfer
        and transfer.mode == "weighted"
        and not state.disabled
        and len(X_current) >= transfer.min_step_observations
    )
    # Same entry gate as weighted mode: honour a previously disabled transfer
    # state and require enough current observations for the MAE trust
    # comparison to be meaningful.
    can_try_refit = (
        can_try_transfer
        and transfer.mode == "cumulative_refit"
        and not state.disabled
        and len(X_current) >= max(3, transfer.min_step_observations)
    )
    if can_try_refit and transfer_memory is not None:
        X_prior = pd.DataFrame(transfer_memory.observed_X_rows)
        y_prior = np.asarray(transfer_memory.observed_y, dtype=float)
        X_prior = _align_to_columns(X_prior, X_current)
        X_combined, y_combined, refit_weights = cumulative_refit_training_set(
            X_prior,
            y_prior,
            X_current,
            y_current,
            weight_cap=transfer.weight_cap,
            proximity_lengthscale=transfer.proximity_lengthscale,
            proximity_floor=transfer.proximity_floor,
        )
        refit_surrogate = _train_surrogate_for_bo(
            X_combined,
            y_combined,
            config=config,
            sample_weight=refit_weights,
        )
        n_prior = len(X_prior)
        prior_weight_sum = float(np.sum(refit_weights[:n_prior]))
        total_weight_sum = float(np.sum(refit_weights))
        state.weight_share = (
            prior_weight_sum / total_weight_sum if total_weight_sum > 0.0 else 0.0
        )
        state.used_rounds += 1

        # Trust gate mirroring weighted mode: an in-sample MAE comparison of
        # the transfer-informed fit against a current-observations-only
        # baseline feeds the same bad-round / patience state machine, so
        # degraded priors cannot pollute every subsequent step.
        baseline = _train_surrogate_for_bo(X_current, y_current, config=config)
        base_mae = float(
            np.mean(np.abs(np.asarray(baseline.predict(X_current)).ravel() - y_current))
        )
        refit_mae = float(
            np.mean(
                np.abs(
                    np.asarray(refit_surrogate.predict(X_current)).ravel() - y_current
                )
            )
        )
        mae_delta = refit_mae - base_mae
        state.last_mae_delta = mae_delta
        if mae_delta > transfer.mae_tolerance:
            state.bad_rounds += 1
        else:
            state.bad_rounds = 0
        if state.bad_rounds >= transfer.trust_patience:
            logger.warning(
                "BO cumulative_refit transfer disabled after %d degraded rounds "
                "(last MAE delta %.4f)",
                state.bad_rounds,
                mae_delta,
            )
            state.disabled = True
            state.disabled_reason = "trust_degraded_on_current_step_residuals"
            state.weight_share = 0.0
            surrogate = baseline
        else:
            surrogate = refit_surrogate
    elif can_try_weighted and transfer_memory is not None:
        transfer_result = build_transfer_surrogate(
            X_current,
            y_current,
            transfer_memory.observed_X_rows,
            transfer_memory.observed_y,
            surrogate=cast(TransferCapableSurrogateType, config.bo.surrogate),
            n_estimators=config.bo.n_estimators,
            random_state=config.seed,
            n_jobs=config.n_jobs,
            weight_cap=transfer.weight_cap,
            similarity_lengthscale=transfer.similarity_lengthscale,
            min_similarity=transfer.min_similarity,
            mae_tolerance=transfer.mae_tolerance,
            transfer_bad_rounds=state.bad_rounds,
            trust_patience=transfer.trust_patience,
            proximity_lengthscale=transfer.proximity_lengthscale,
            prior_step_ages=transfer_memory.step_ages,
            recency_lengthscale=transfer.recency_lengthscale,
            prior_placement_X=occupancy_placement_X,
            occupancy_lengthscale=transfer.occupancy_lengthscale,
            occupancy_floor=transfer.occupancy_floor,
        )
        state.weight_share = transfer_result.transfer_weight_share
        state.last_mae_delta = transfer_result.transfer_mae_delta
        state.bad_rounds = transfer_result.transfer_bad_rounds
        if transfer_result.transfer_used_this_round:
            state.used_rounds += 1
        if transfer_result.transfer_disabled:
            state.disabled = True
            state.disabled_reason = transfer_result.transfer_disabled_reason
        surrogate = transfer_result.surrogate

    transfer_active = (can_try_refit or can_try_weighted) and not state.disabled
    if surrogate is None:
        surrogate = _train_surrogate_for_bo(
            X_current,
            y_current,
            config=config,
        )
    return surrogate, transfer_active


def process_molecule_bayesian(
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    calculator,
    reference_energies: ReferenceEnergies,
    ts_model=None,
    *,
    config: AdsorptionConfig,
    surface_type: str = "manual",
    reference_smiles: str | None = None,
    base_slab_for_frozen: Atoms | None = None,
    slab_energy_override: float | None = None,
    symmetry_broken: bool = False,
    bo_step_memory_in: BOStepMemory | None = None,
    occupancy_placement_X: list[dict[str, float]] | None = None,
    conformers: list[Atoms] | None = None,
    conformer_energies: list[float] | None = None,
    skip_workload_autotune: bool = False,
    saturation_reuse: bool = True,
) -> MoleculeScreenOutcome:
    """Bayesian-optimisation-guided placement screening for one molecule.

    Parameters
    ----------
    smiles
        SMILES string of the molecule.
    molecule_name
        Human-readable molecule identifier.
    slab
        Substrate container.
    calculator
        ASE calculator instance.
    reference_energies
        Reference energies for slab and molecules.
    ts_model
        Transition-state model (optional).
    config
        Adsorption configuration (required).
    surface_type
        Surface type label.
    reference_smiles
        SMILES used for reference energy lookup.
    base_slab_for_frozen
        Base slab for freeze constraints.
    slab_energy_override
        Override slab reference energy.
    symmetry_broken
        Whether symmetry is broken.
    bo_step_memory_in
        Prior BO step memory for transfer learning.
    occupancy_placement_X
        Feature rows of placements already committed on the slab; used as
        occupancy anchors for transfer down-weighting.
    conformers
        Pre-generated conformers (optional).
    conformer_energies
        Energies aligned with conformers (optional).
    skip_workload_autotune
        Whether to skip workload autotuning.
    saturation_reuse
        Reuse the slab+adsorbate autobatcher across acquisition batches
        (default True; same molecule/slab for the whole call).
    """
    if reference_smiles is None:
        reference_smiles = smiles

    failure_summary: FailureSummary | None = None
    ml_records: list[PlacementRecord] = []
    transfer_info = BOTransferInfo()
    bo_memory: BOStepMemory | None = None

    def _outcome(results: list[ScreeningResult]) -> MoleculeScreenOutcome:
        return MoleculeScreenOutcome(
            results=results,
            failure_summary=failure_summary,
            ml_records=ml_records,
            bo_memory=bo_memory,
            transfer_info=transfer_info,
        )

    def _bail_outcome(summary: FailureSummary) -> MoleculeScreenOutcome:
        nonlocal failure_summary
        failure_summary = summary
        return _outcome([])

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
    )
    if ctx is None:
        assert early_failure is not None
        return _bail_outcome(early_failure)

    slab = ctx.slab
    slab_for_sites = ctx.slab_for_sites
    effective_base_slab_for_frozen = ctx.effective_base_slab_for_frozen
    conformers = ctx.conformers
    conformer_energies = ctx.conformer_energies
    site_context = ctx.site_context
    config = ctx.config
    E_slab = ctx.E_slab
    E_mol = ctx.E_mol

    if config.bo.initial_random is None:
        raise ValueError("config.bo.initial_random must be set for Bayesian screening")
    if config.bo.batch_size is None:
        raise ValueError("config.bo.batch_size must be set for Bayesian screening")
    num_placements = config.num_placements
    assert num_placements is not None
    bo_eval_budget = resolved_bo_eval_budget(config)

    max_enumerated_specs = estimate_placement_spec_capacity(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
        full_slab=slab.atoms,
    )
    if config.bo.candidate_pool_size is not None:
        pool_size = config.bo.candidate_pool_size
    else:
        pool_size = max_enumerated_specs
        if pool_size <= 0:
            pool_size = max(bo_eval_budget * 5, num_placements)
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
        logger.warning("No candidate specs generated for BO")
        return _bail_outcome(BOPlacementFailure(n_candidate_specs=0, n_valid_pool=0))

    logger.info(
        "BO: %d/%d candidate specs, batches=%d (initial=%d, batch=%d, eval_budget=%d, kappa=%.2f)",
        len(all_specs),
        max_enumerated_specs,
        config.bo.total_budget,
        config.bo.initial_random,
        config.bo.batch_size,
        bo_eval_budget,
        config.bo.ucb_kappa,
    )

    surface_symbols = _infer_surface_symbols(slab_for_sites)
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
        logger.warning("BO: no specs produced valid placements; aborting")
        return _bail_outcome(
            BOPlacementFailure(
                n_candidate_specs=len(all_specs),
                n_valid_pool=0,
            )
        )
    # Standardize the fixed candidate pool once for diversity across rounds.
    scaled_candidate_features = StandardScaler().fit_transform(
        candidate_features.to_numpy(dtype=float)
    )
    pid_to_pool_position: dict[int, int] = {
        all_specs[valid_spec_indices[p]].placement_index: p
        for p in range(len(valid_spec_indices))
    }
    evaluated_pool_positions: set[int] = set()
    all_results: list[ScreeningResult] = []
    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    bo_negative_records: list[PlacementRecord] = []
    total_evaluated = 0
    best_energy = float("inf")
    best_X_row: dict[str, float] | None = None
    bo_failure_events: list[PlacementFailureEvent] = []
    rng = bo_exploration_rng(config.seed, len(slab.atoms), molecule=molecule_name)
    transfer_state = _TransferRoundState()

    def _flush_bo_outputs() -> None:
        nonlocal bo_memory
        # Penalty observations live in bo_memory regardless of how this call
        # ends, so they must reach ml_records on every path — including
        # validation/filter bail-outs — to keep the dataset trail in sync with
        # what the surrogate actually trained on.
        ml_records.extend(bo_negative_records)
        bo_memory = BOStepMemory(
            observed_X_rows=[dict(r) for r in observed_X_rows],
            observed_y=[float(v) for v in observed_y],
            best_energy=best_energy if np.isfinite(best_energy) else None,
            best_X_row=dict(best_X_row) if best_X_row is not None else None,
        )
        transfer_info.transfer_used = bool(transfer_state.used_rounds > 0)
        transfer_info.transfer_disabled_reason = transfer_state.disabled_reason
        transfer_info.transfer_bad_rounds = int(transfer_state.bad_rounds)
        transfer_info.transfer_last_mae_delta = transfer_state.last_mae_delta
        transfer_info.transfer_weight_share = float(transfer_state.weight_share)

    def _failure_penalty(stage: str, reason: str) -> float:
        overrides = config.bo.failure_penalty_overrides
        if reason in overrides:
            return float(overrides[reason])
        if stage in overrides:
            return float(overrides[stage])
        return float(config.bo.failure_penalty_default)

    def _append_penalty_observation(record, stage: str, reason: str) -> None:
        penalty = _failure_penalty(stage, reason)
        record.converged = False
        record.failure_stage = stage
        record.failure_reason = reason
        record.is_penalty_label = True
        record.label_source = "bo_failure_penalty"
        record.energy_adsorption = penalty
        observed_X_rows.append(extract_features(record))
        observed_y.append(penalty)
        bo_negative_records.append(record)

    def _unevaluated() -> list[int]:
        return [
            p
            for p in range(len(valid_spec_indices))
            if p not in evaluated_pool_positions
        ]

    n_initial = min(config.bo.initial_random, len(valid_spec_indices))
    # Molecule-decorrelated initial seed: in multi-molecule steps every
    # adsorbate screens against a same-size pool with the same config.seed, so
    # "random" sampling would otherwise pick identical pool positions for all.
    initial_seed = int(
        bo_exploration_rng(
            config.seed, len(slab.atoms), molecule=molecule_name
        ).randint(0, 2**31 - 1)
    )
    initial_positions = select_initial_bo_indices(
        candidate_features,
        n_initial,
        sampling=config.bo.initial_sampling,
        random_state=initial_seed,
    )

    def _run_batch(pool_positions: list[int]) -> None:
        nonlocal total_evaluated, best_energy, best_X_row
        n_target = len(pool_positions)
        primary_set = set(pool_positions)
        batch_specs = [all_specs[valid_spec_indices[p]] for p in pool_positions]
        backfill_positions = [
            p
            for p in range(len(valid_spec_indices))
            if p not in evaluated_pool_positions and p not in primary_set
        ]
        backfill_specs = [all_specs[valid_spec_indices[p]] for p in backfill_positions]

        batch_results, batch_failures = _evaluate_placement_batch(
            batch_specs,
            conformers,
            slab,
            calculator,
            ts_model,
            config,
            smiles,
            E_slab,
            E_mol,
            molecule_name,
            surface_symbols,
            site_context=site_context,
            base_slab_for_frozen=effective_base_slab_for_frozen,
            slab_for_sites=slab_for_sites,
            materialization_cache=materialization_cache,
            backfill_specs=backfill_specs,
            n_target=n_target,
            saturation_reuse=saturation_reuse,
        )

        # Mark primary always. For backfill, only mark specs that produced an
        # outcome (result or failure) — oversampled successes trimmed before
        # relax must stay selectable for later BO batches.
        outcome_pids = {r.placement_id for r in batch_results}
        outcome_pids.update(event.placement_id for event in batch_failures)
        used_positions = list(pool_positions)
        for p in backfill_positions:
            pid = int(all_specs[valid_spec_indices[p]].placement_index)
            if pid in outcome_pids:
                used_positions.append(p)
        evaluated_pool_positions.update(used_positions)
        total_evaluated += len(used_positions)
        all_results.extend(batch_results)
        bo_failure_events.extend(batch_failures)

        for r in batch_results:
            record = PlacementRecord.from_descriptor(
                r.placement_descriptor,
                molecule=molecule_name,
                smiles=smiles,
                surface_id=surface_type,
                config=config,
            )
            features = extract_features(record)
            observed_X_rows.append(features)
            observed_y.append(r.energy_adsorption)
            if r.energy_adsorption < best_energy:
                best_energy = r.energy_adsorption
                best_X_row = dict(features)

        if config.bo.include_failure_negatives:
            for event in batch_failures:
                pool_pos = pid_to_pool_position.get(event.placement_id)
                if pool_pos is None:
                    continue
                spec = all_specs[valid_spec_indices[pool_pos]]
                record = (
                    PlacementRecord.from_descriptor(
                        event.descriptor,
                        molecule=molecule_name,
                        smiles=smiles,
                        surface_id=surface_type,
                        config=config,
                    )
                    if event.descriptor is not None
                    else PlacementRecord.from_spec(
                        spec,
                        molecule=molecule_name,
                        smiles=smiles,
                        surface_id=surface_type,
                        config=config,
                    )
                )
                _append_penalty_observation(record, event.stage, event.reason)

        if batch_results:
            batch_best = min(r.energy_adsorption for r in batch_results)
            logger.info(
                "BO batch: %d evaluated, %d valid, %d failed, batch_best=%.4f, overall_best=%.4f",
                len(used_positions),
                len(batch_results),
                len(batch_failures),
                batch_best,
                best_energy,
            )
        else:
            logger.info(
                "BO batch: %d evaluated, 0 valid results, %d failed",
                len(used_positions),
                len(batch_failures),
            )

    _run_batch(initial_positions)

    batches_run = 0
    while batches_run < config.bo.total_budget:
        if len(observed_X_rows) < 3:
            unevaluated = _unevaluated()
            if not unevaluated:
                break
            n_extra = min(config.bo.batch_size, len(unevaluated))
            next_positions = rng.choice(
                unevaluated, size=n_extra, replace=False
            ).tolist()
        else:
            unevaluated = _unevaluated()
            if not unevaluated:
                break
            X_current = pd.DataFrame(observed_X_rows)
            y_current = np.array(observed_y)

            surrogate, transfer_active = _build_round_surrogate(
                X_current=X_current,
                y_current=y_current,
                transfer_memory=bo_step_memory_in,
                state=transfer_state,
                config=config,
                occupancy_placement_X=occupancy_placement_X,
            )

            batch_size = min(config.bo.batch_size, len(unevaluated))
            acquisition = config.bo.acquisition
            f_best = best_energy if np.isfinite(best_energy) else None
            # EI/PI need a finite incumbent; before any valid E_ads, use LCB.
            if acquisition in ("ei", "pi") and f_best is None:
                logger.info(
                    "BO: no valid E_ads yet; using LCB instead of %s for this batch",
                    acquisition,
                )
                acquisition = "lcb"
            next_positions = score_and_select(
                surrogate,
                candidate_features,
                batch_size=batch_size,
                kappa=config.bo.ucb_kappa,
                evaluated_indices=evaluated_pool_positions,
                acquisition=acquisition,
                f_best=f_best,
                scaled_features=scaled_candidate_features,
                n_jobs=config.n_jobs,
            )
            # Random exploration only while transfer learning is active.
            # Pure BO screening keeps the full acquisition batch.
            explore_n = 0
            if transfer_active:
                explore_n = int(
                    np.ceil(batch_size * config.bo.transfer.exploration_fraction)
                )
            if explore_n > 0:
                next_positions = splice_exploration_picks(
                    rng,
                    next_positions,
                    pool_size=len(valid_spec_indices),
                    evaluated_indices=evaluated_pool_positions,
                    exploration_fraction=config.bo.transfer.exploration_fraction,
                )

        if not next_positions:
            logger.info("BO: no more candidates to evaluate")
            break

        _run_batch(next_positions)
        batches_run += 1

    logger.info(
        "BO complete: %d total evaluated (%d acquisition batches), %d valid results, best E_ads=%.4f eV",
        total_evaluated,
        batches_run,
        len(all_results),
        best_energy if np.isfinite(best_energy) else float("nan"),
    )
    _summarize_failure_events(bo_failure_events, label=f"{molecule_name} BO evaluation")

    if not all_results:
        _flush_bo_outputs()
        return _bail_outcome(
            BOValidationFailure(
                n_candidate_specs=len(all_specs),
                n_valid_pool=len(valid_spec_indices),
                n_evaluated=total_evaluated,
                n_valid_results=0,
            )
        )

    # BO also builds failure-penalty negatives for rejected non-duplicates.
    filtered, bo_duplicate_results, _t_filtering = _filter_and_label_duplicates(
        all_results,
        slab_atoms=slab.atoms,
        surface_symbols=surface_symbols,
        reference_smiles=reference_smiles,
        config=config,
        smiles=smiles,
        surface_type=surface_type,
        ml_records=bo_negative_records,
    )
    duplicate_result_ids = {id(result) for result in bo_duplicate_results}
    kept_result_ids = {id(result) for result in filtered}
    rejected_after_filter = [r for r in all_results if id(r) not in kept_result_ids]
    rejected_non_duplicate = [
        r for r in rejected_after_filter if id(r) not in duplicate_result_ids
    ]
    if rejected_non_duplicate:
        filter_failures = [
            PlacementFailureEvent(
                placement_id=r.placement_id,
                stage="filter",
                reason="filtered_out_postprocessing",
                descriptor=r.placement_descriptor,
            )
            for r in rejected_non_duplicate
        ]
        _summarize_failure_events(
            filter_failures, label=f"{molecule_name} BO post-filter"
        )
        if config.bo.include_failure_negatives:
            for event in filter_failures:
                if event.descriptor is None:
                    continue
                record = PlacementRecord.from_descriptor(
                    event.descriptor,
                    molecule=molecule_name,
                    smiles=smiles,
                    surface_id=surface_type,
                    config=config,
                )
                _append_penalty_observation(record, event.stage, event.reason)

    if bo_duplicate_results:
        logger.info(
            "BO post-filter deduplicated %d results (tracked for ML/BO)",
            len(bo_duplicate_results),
        )

    if not filtered:
        _flush_bo_outputs()
        return _bail_outcome(
            FilterFailure(
                n_before_filter=len(all_results),
                n_after_filter=0,
            )
        )

    logger.info(
        "BO filtered: %d -> %d results, E_ads range [%.4f, %.4f]",
        len(all_results),
        len(filtered),
        min(r.energy_adsorption for r in filtered),
        max(r.energy_adsorption for r in filtered),
    )
    _flush_bo_outputs()

    return _outcome(filtered)

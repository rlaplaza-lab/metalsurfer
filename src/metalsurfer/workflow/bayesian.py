"""Bayesian optimisation workflow orchestration."""

import logging
from typing import cast

import numpy as np
import pandas as pd
from ase import Atoms

from ..config import (
    BO_TRANSFER_CAPABLE_SURROGATES,
    AdsorptionConfig,
    resolved_bo_eval_budget,
)
from ..filters import filter_results
from ..ml.bayesian import (
    TransferCapableSurrogateType,
    build_spec_features_geometry_aware,
    build_transfer_surrogate,
    cumulative_refit_sample_weights,
    score_and_select,
    select_initial_bo_indices,
    train_surrogate,
)
from ..ml.features import extract_features
from ..ml.schema import PlacementRecord
from ..models import BOStepMemory, BOTransferInfo, ReferenceEnergies, ScreeningResult
from ..placement.generators import (
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
)
from ..surface_prep import SlabContainer
from .core import _evaluate_placement_batch
from .shared import (
    MoleculeScreenOutcome,
    PlacementFailureEvent,
    _infer_surface_symbols,
    _prepare_molecule_screening,
    _summarize_failure_events,
)

logger = logging.getLogger(__name__)

__all__ = ["BOTransferInfo", "process_molecule_bayesian"]


def _train_surrogate_for_bo(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    config: AdsorptionConfig,
    sample_weight: np.ndarray | None,
):
    """Fit BO surrogate. Per-sample weights for transfer-capable surrogates."""
    if (
        sample_weight is not None
        and config.bo.surrogate not in BO_TRANSFER_CAPABLE_SURROGATES
    ):
        raise ValueError(
            "sample_weight is only supported for transfer-capable BO surrogates "
            f"{BO_TRANSFER_CAPABLE_SURROGATES}, not {config.bo.surrogate!r}"
        )
    return train_surrogate(
        X,
        y,
        surrogate=config.bo.surrogate,
        n_estimators=100,
        random_state=config.seed,
        sample_weight=sample_weight,
    )


def process_molecule_bayesian(
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    calculator,
    reference_energies: ReferenceEnergies,
    ts_model=None,
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    reference_smiles: str | None = None,
    base_slab_for_frozen: Atoms | None = None,
    slab_energy_override: float | None = None,
    symmetry_broken: bool = False,
    bo_step_memory_in: BOStepMemory | None = None,
    bo_prior_step_memory: BOStepMemory | None = None,
    conformers: list[Atoms] | None = None,
    skip_workload_autotune: bool = False,
) -> MoleculeScreenOutcome:
    """Bayesian-optimisation-guided placement screening for one molecule."""
    if config is None:
        config = AdsorptionConfig()

    if reference_smiles is None:
        reference_smiles = smiles

    failure_summary: dict[str, object] = {}
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

    ctx = _prepare_molecule_screening(
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
        failure_summary=failure_summary,
        bo_enabled=True,
        conformers=conformers,
        skip_workload_autotune=skip_workload_autotune,
    )
    if ctx is None:
        return _outcome([])

    slab = ctx.slab
    slab_for_sites = ctx.slab_for_sites
    effective_base_slab_for_frozen = ctx.effective_base_slab_for_frozen
    conformers = ctx.conformers
    site_context = ctx.site_context
    config = ctx.config
    E_slab = ctx.E_slab
    E_mol = ctx.E_mol

    assert config.bo.initial_random is not None
    assert config.bo.batch_size is not None
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
    )
    if not all_specs:
        logger.warning("No candidate specs generated for BO")
        failure_summary["stage"] = "placement"
        failure_summary["n_candidate_specs"] = 0
        failure_summary["n_valid_pool"] = 0
        return _outcome([])

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
    materialization_cache: dict[int, tuple] = {}
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
        failure_summary["stage"] = "placement"
        failure_summary["n_candidate_specs"] = len(all_specs)
        failure_summary["n_valid_pool"] = 0
        return _outcome([])
    # Map feature-row indices back to the original all_specs list
    valid_pool_indices = valid_spec_indices

    evaluated_pool_positions: set[int] = set()
    all_results: list[ScreeningResult] = []
    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    bo_negative_records: list[PlacementRecord] = []
    total_evaluated = 0
    best_energy = float("inf")
    best_X_row: dict[str, float] | None = None
    bo_failure_events: list[PlacementFailureEvent] = []
    rng = np.random.RandomState(config.seed)
    transfer_disabled = False
    transfer_disabled_reason: str | None = None
    transfer_bad_rounds = 0
    transfer_used_rounds = 0
    transfer_last_mae_delta: float | None = None
    transfer_weight_share = 0.0

    def _flush_bo_outputs() -> None:
        nonlocal bo_memory
        bo_memory = BOStepMemory(
            observed_X_rows=[dict(r) for r in observed_X_rows],
            observed_y=[float(v) for v in observed_y],
            best_energy=best_energy if np.isfinite(best_energy) else None,
            best_X_row=dict(best_X_row) if best_X_row is not None else None,
        )
        transfer_info.transfer_enabled = bool(config.bo.transfer.enabled)
        transfer_info.transfer_used = bool(transfer_used_rounds > 0)
        transfer_info.transfer_disabled_reason = transfer_disabled_reason
        transfer_info.transfer_bad_rounds = int(transfer_bad_rounds)
        transfer_info.transfer_last_mae_delta = transfer_last_mae_delta
        transfer_info.transfer_weight_share = float(transfer_weight_share)

    def _failure_penalty(stage: str, reason: str) -> float:
        overrides = config.bo.failure_penalty_overrides
        if reason in overrides:
            return float(overrides[reason])
        if stage in overrides:
            return float(overrides[stage])
        return float(config.bo.failure_penalty_default)

    n_initial = min(config.bo.initial_random, len(valid_pool_indices))
    initial_positions = select_initial_bo_indices(
        candidate_features,
        n_initial,
        sampling=config.bo.initial_sampling,
        random_state=config.seed,
    )

    def _run_batch(pool_positions: list[int]) -> None:
        nonlocal total_evaluated, best_energy, best_X_row
        n_target = len(pool_positions)
        primary_set = set(pool_positions)
        batch_specs = [all_specs[valid_pool_indices[p]] for p in pool_positions]
        backfill_positions = [
            p
            for p in range(len(valid_pool_indices))
            if p not in evaluated_pool_positions and p not in primary_set
        ]
        backfill_specs = [
            all_specs[valid_pool_indices[p]] for p in backfill_positions
        ]

        batch_results, batch_failures, n_backfill_used = _evaluate_placement_batch(
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
        )

        used_positions = list(pool_positions) + backfill_positions[:n_backfill_used]
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
            pid_to_pool_position: dict[int, int] = {
                all_specs[valid_pool_indices[pos]].placement_index: pos
                for pos in used_positions
            }
            for event in batch_failures:
                pool_pos = pid_to_pool_position.get(event.placement_id)
                if pool_pos is None:
                    continue
                spec = all_specs[valid_pool_indices[pool_pos]]
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
                record.converged = False
                record.failure_stage = event.stage
                record.failure_reason = event.reason
                record.is_penalty_label = True
                record.label_source = "bo_failure_penalty"
                observed_X_rows.append(extract_features(record))
                observed_y.append(_failure_penalty(event.stage, event.reason))
                bo_negative_records.append(record)

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
        remaining_batches = config.bo.total_budget - batches_run
        if remaining_batches <= 0:
            break

        if len(observed_X_rows) < 3:
            unevaluated = [
                p
                for p in range(len(valid_pool_indices))
                if p not in evaluated_pool_positions
            ]
            if not unevaluated:
                break
            n_extra = min(config.bo.batch_size, len(unevaluated))
            next_positions = rng.choice(
                unevaluated, size=n_extra, replace=False
            ).tolist()
        else:
            X_current = pd.DataFrame(observed_X_rows)
            y_current = np.array(observed_y)
            X_train = X_current
            y_train = y_current
            sample_weight: np.ndarray | None = None
            surrogate = None

            transfer_memory = bo_step_memory_in
            can_try_transfer = (
                config.bo.transfer.enabled
                and transfer_memory is not None
                and len(transfer_memory.observed_X_rows) > 0
                and len(transfer_memory.observed_y) > 0
            )
            can_try_weighted = (
                can_try_transfer
                and config.bo.transfer.mode == "weighted"
                and not transfer_disabled
                and len(X_current) >= config.bo.transfer.min_step_observations
            )
            can_try_refit = (
                can_try_transfer
                and config.bo.transfer.mode == "cumulative_refit"
                and len(X_current) >= 3
            )
            if can_try_refit:
                assert transfer_memory is not None
                X_prior = pd.DataFrame(transfer_memory.observed_X_rows)
                y_prior = np.asarray(transfer_memory.observed_y, dtype=float)
                X_prior = X_prior.reindex(columns=X_current.columns, fill_value=0.0)
                X_combined = pd.concat([X_prior, X_current], ignore_index=True)
                y_combined = np.concatenate([y_prior, y_current])
                refit_weights = cumulative_refit_sample_weights(
                    len(X_current),
                    X_prior,
                    X_current,
                    weight_cap=config.bo.transfer.weight_cap,
                    proximity_lengthscale=config.bo.transfer.proximity_lengthscale,
                    proximity_floor=config.bo.transfer.proximity_floor,
                )
                surrogate = _train_surrogate_for_bo(
                    X_combined,
                    y_combined,
                    config=config,
                    sample_weight=refit_weights,
                )
                transfer_used_rounds += 1
            elif can_try_weighted:
                assert transfer_memory is not None
                prior_placement = None
                if (
                    bo_prior_step_memory is not None
                    and bo_prior_step_memory.best_X_row is not None
                ):
                    prior_placement = bo_prior_step_memory.best_X_row
                transfer_result = build_transfer_surrogate(
                    X_current,
                    y_current,
                    transfer_memory.observed_X_rows,
                    transfer_memory.observed_y,
                    surrogate=cast(TransferCapableSurrogateType, config.bo.surrogate),
                    n_estimators=100,
                    random_state=config.seed,
                    weight_cap=config.bo.transfer.weight_cap,
                    similarity_lengthscale=config.bo.transfer.similarity_lengthscale,
                    min_similarity=config.bo.transfer.min_similarity,
                    mae_tolerance=config.bo.transfer.mae_tolerance,
                    transfer_bad_rounds=transfer_bad_rounds,
                    trust_patience=config.bo.transfer.trust_patience,
                    proximity_lengthscale=config.bo.transfer.proximity_lengthscale,
                    proximity_floor=config.bo.transfer.proximity_floor,
                    prior_step_ages=transfer_memory.step_ages,
                    recency_lengthscale=config.bo.transfer.recency_lengthscale,
                    prior_placement_X=prior_placement,
                    occupancy_lengthscale=config.bo.transfer.occupancy_lengthscale,
                    occupancy_floor=config.bo.transfer.occupancy_floor,
                )
                transfer_weight_share = transfer_result.transfer_weight_share
                transfer_last_mae_delta = transfer_result.transfer_mae_delta
                transfer_bad_rounds = transfer_result.transfer_bad_rounds
                if transfer_result.transfer_used_this_round:
                    transfer_used_rounds += 1
                if transfer_result.transfer_disabled:
                    transfer_disabled = True
                    transfer_disabled_reason = transfer_result.transfer_disabled_reason
                surrogate = transfer_result.surrogate

            if surrogate is None or transfer_disabled:
                surrogate = _train_surrogate_for_bo(
                    X_train,
                    y_train,
                    config=config,
                    sample_weight=sample_weight,
                )

            unevaluated = [
                p
                for p in range(len(valid_pool_indices))
                if p not in evaluated_pool_positions
            ]
            if not unevaluated:
                break
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
            )
            # Random exploration only while transfer learning is active.
            # Pure BO screening keeps the full acquisition batch.
            explore_n = 0
            if can_try_transfer and not transfer_disabled:
                explore_n = int(
                    np.ceil(batch_size * config.bo.transfer.exploration_fraction)
                )
            if explore_n > 0:
                unevaluated = [
                    p
                    for p in range(len(valid_pool_indices))
                    if p not in evaluated_pool_positions
                ]
                available_for_random = [
                    p for p in unevaluated if p not in next_positions
                ]
                if available_for_random:
                    explore_n = min(
                        explore_n, len(available_for_random), len(next_positions)
                    )
                    random_positions = rng.choice(
                        available_for_random,
                        size=explore_n,
                        replace=False,
                    ).tolist()
                    next_positions[-explore_n:] = random_positions
                    next_positions = list(dict.fromkeys(next_positions))

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
        failure_summary["stage"] = "validation"
        failure_summary["n_candidate_specs"] = len(all_specs)
        failure_summary["n_valid_pool"] = len(valid_pool_indices)
        failure_summary["n_evaluated"] = total_evaluated
        failure_summary["n_valid_results"] = 0
        return _outcome([])

    # Filter + dedup labeling follows the same pattern as core's
    # ``_finalize_screen_results``, but BO also builds failure-penalty negatives
    # for rejected non-duplicates, so the shared helper is not used here.
    bo_duplicate_results: list[ScreeningResult] = []
    filtered = filter_results(
        all_results,
        slab=slab.atoms,
        surface_symbols=surface_symbols,
        reference_smiles=reference_smiles,
        config=config,
        duplicate_results_out=bo_duplicate_results,
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
                record.converged = False
                record.failure_stage = event.stage
                record.failure_reason = event.reason
                record.is_penalty_label = True
                record.label_source = "bo_failure_penalty"
                observed_X_rows.append(extract_features(record))
                observed_y.append(_failure_penalty(event.stage, event.reason))
                bo_negative_records.append(record)

    if bo_duplicate_results:
        logger.info(
            "BO post-filter deduplicated %d results (tracked for ML/BO)",
            len(bo_duplicate_results),
        )
        for duplicate in bo_duplicate_results:
            record = PlacementRecord.from_screening_result(
                duplicate,
                smiles=smiles,
                surface_id=surface_type,
                config=config,
            )
            record.label_source = "deduplicated_duplicate"
            bo_negative_records.append(record)

    if not filtered:
        _flush_bo_outputs()
        failure_summary["stage"] = "filter"
        failure_summary["n_candidate_specs"] = len(all_specs)
        failure_summary["n_valid_pool"] = len(valid_pool_indices)
        failure_summary["n_evaluated"] = total_evaluated
        failure_summary["n_before_filter"] = len(all_results)
        failure_summary["n_after_filter"] = 0
        return _outcome([])

    logger.info(
        "BO filtered: %d -> %d results, E_ads range [%.4f, %.4f]",
        len(all_results),
        len(filtered),
        min(r.energy_adsorption for r in filtered),
        max(r.energy_adsorption for r in filtered),
    )
    if bo_negative_records:
        ml_records.extend(bo_negative_records)
    _flush_bo_outputs()

    return _outcome(filtered)

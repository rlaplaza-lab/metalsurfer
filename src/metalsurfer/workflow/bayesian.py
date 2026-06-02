"""Bayesian optimisation workflow orchestration."""

import logging

import numpy as np
import pandas as pd
from ase import Atoms

from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..filters import filter_results
from ..ml.bayesian import (
    build_spec_features_geometry_aware,
    score_and_select,
    train_surrogate,
)
from ..ml.features import extract_features
from ..ml.schema import PlacementRecord
from ..models import BOStepMemory, ReferenceEnergies, ScreeningResult
from ..optimization import clear_autobatcher_cache
from ..placement.generators import (
    enumerate_placement_specs,
    estimate_placement_spec_capacity,
)
from ..surfaces import SlabContainer
from .core import _evaluate_placement_batch
from .shared import (
    PlacementFailureEvent,
    _compute_slab_energy,
    _infer_surface_symbols,
    _resolve_site_context_for_sampling,
    _summarize_failure_events,
    prepare_substrate_for_screening,
    write_substrate_step_metadata,
)

logger = logging.getLogger(__name__)


def _train_surrogate_for_bo(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    *,
    config: AdsorptionConfig,
    sample_weight: np.ndarray | None,
):
    """Fit BO surrogate. Per-sample weights are supported only for tree ensembles."""
    if sample_weight is not None and config.bo_surrogate in (
        "gradient_boost",
        "ridge",
    ):
        raise ValueError(
            "sample_weight is only supported for tree BO surrogates "
            f"(random_forest, extra_trees), not {config.bo_surrogate!r}"
        )
    return train_surrogate(
        X,
        y,
        surrogate=config.bo_surrogate,
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
    failure_summary_out: dict[str, object] | None = None,
    extra_ml_records_out: list[PlacementRecord] | None = None,
    symmetry_broken: bool = False,
    bo_step_memory_in: BOStepMemory | None = None,
    bo_step_memory_out: dict[str, BOStepMemory] | None = None,
    bo_transfer_info_out: dict[str, object] | None = None,
    allow_auto_resize: bool = True,
    step_metadata_out: dict[str, object] | None = None,
) -> list[ScreeningResult] | None:
    """Bayesian-optimisation-guided placement screening for one molecule."""
    if config is None:
        config = AdsorptionConfig(bo_enabled=True)

    if reference_smiles is None:
        reference_smiles = smiles

    E_slab = (
        slab_energy_override
        if slab_energy_override is not None
        else reference_energies.slab_energy
    )
    E_mol = reference_energies.get_molecule_energy(molecule_name)
    if E_mol is None:
        logger.error("Missing reference energy for %s", molecule_name)
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "reference"
            failure_summary_out["reason"] = (
                f"missing reference energy for {molecule_name}"
            )
        return None

    result = create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )
    if result is None:
        logger.error("Could not generate conformers for %s", molecule_name)
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "conformers"
            failure_summary_out["reason"] = (
                f"could not generate conformers for {molecule_name}"
            )
        return None
    conformers, _ = result

    substrate_ref = prepare_substrate_for_screening(
        slab,
        conformers,
        base_slab_for_frozen,
        config,
        allow_auto_resize=allow_auto_resize,
    )
    slab = substrate_ref.slab
    slab_for_sites = substrate_ref.slab_for_sites
    effective_base_slab_for_frozen = substrate_ref.effective_base_slab_for_frozen

    if substrate_ref.slab_was_resized:
        clear_autobatcher_cache()
        E_slab = _compute_slab_energy(
            slab.atoms, calculator, label="resized slab reference"
        )
        logger.info("Resized slab energy: %.4f eV", E_slab)

    write_substrate_step_metadata(
        step_metadata_out,
        slab_was_resized=substrate_ref.slab_was_resized,
        substrate_atoms_after_resize=substrate_ref.substrate_atoms_after_resize,
    )

    site_context = _resolve_site_context_for_sampling(
        slab_for_sites,
        config,
        symmetry_broken=symmetry_broken,
    )

    max_enumerated_specs = estimate_placement_spec_capacity(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
        full_slab=slab.atoms,
    )
    if config.bo_candidate_pool_size is not None:
        pool_size = config.bo_candidate_pool_size
    else:
        pool_size = max_enumerated_specs
        if pool_size <= 0:
            pool_size = max(config.bo_total_budget * 5, config.num_placements)
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
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "placement"
            failure_summary_out["n_candidate_specs"] = 0
            failure_summary_out["n_valid_pool"] = 0
        return None

    logger.info(
        "BO: %d/%d candidate specs, budget=%d (initial=%d, batch=%d, kappa=%.2f)",
        len(all_specs),
        max_enumerated_specs,
        config.bo_total_budget,
        config.bo_initial_random,
        config.bo_batch_size,
        config.bo_ucb_kappa,
    )

    surface_symbols = _infer_surface_symbols(slab_for_sites)
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
    )
    if candidate_features.empty:
        logger.warning("BO: no specs produced valid placements; aborting")
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "placement"
            failure_summary_out["n_candidate_specs"] = len(all_specs)
            failure_summary_out["n_valid_pool"] = 0
        return None
    # Map feature-row indices back to the original all_specs list
    valid_pool_indices = valid_spec_indices

    evaluated_pool_positions: set[int] = set()
    all_results: list[ScreeningResult] = []
    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    bo_negative_records: list[PlacementRecord] = []
    total_evaluated = 0
    best_energy = float("inf")
    bo_failure_events: list[PlacementFailureEvent] = []
    rng = np.random.RandomState(config.seed)
    transfer_disabled = False
    transfer_disabled_reason: str | None = None
    transfer_bad_rounds = 0
    transfer_used_rounds = 0
    transfer_last_mae_delta: float | None = None
    transfer_weight_share = 0.0

    def _flush_bo_outputs() -> None:
        if bo_step_memory_out is not None:
            bo_step_memory_out["memory"] = BOStepMemory(
                observed_X_rows=[dict(r) for r in observed_X_rows],
                observed_y=[float(v) for v in observed_y],
                best_energy=best_energy if np.isfinite(best_energy) else None,
            )
        if bo_transfer_info_out is not None:
            bo_transfer_info_out.update(
                {
                    "transfer_enabled": bool(config.bo_transfer_enabled),
                    "transfer_used": bool(transfer_used_rounds > 0),
                    "transfer_disabled_reason": transfer_disabled_reason,
                    "transfer_bad_rounds": int(transfer_bad_rounds),
                    "transfer_last_mae_delta": transfer_last_mae_delta,
                    "transfer_weight_share": float(transfer_weight_share),
                }
            )

    def _failure_penalty(stage: str, reason: str) -> float:
        overrides = config.bo_failure_penalty_overrides
        if reason in overrides:
            return float(overrides[reason])
        if stage in overrides:
            return float(overrides[stage])
        return float(config.bo_failure_penalty_default)

    n_initial = min(
        config.bo_initial_random, len(valid_pool_indices), config.bo_total_budget
    )
    initial_positions = rng.choice(
        len(valid_pool_indices), size=n_initial, replace=False
    ).tolist()

    def _run_batch(pool_positions: list[int]) -> None:
        nonlocal total_evaluated, best_energy
        batch_specs = [all_specs[valid_pool_indices[p]] for p in pool_positions]
        evaluated_pool_positions.update(pool_positions)

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
        )

        total_evaluated += len(pool_positions)
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
            observed_X_rows.append(extract_features(record))
            observed_y.append(r.energy_adsorption)
            if r.energy_adsorption < best_energy:
                best_energy = r.energy_adsorption

        if config.bo_include_failure_negatives:
            pid_to_pool_position: dict[int, int] = {
                all_specs[valid_pool_indices[pos]].placement_index: pos
                for pos in pool_positions
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
                len(pool_positions),
                len(batch_results),
                len(batch_failures),
                batch_best,
                best_energy,
            )
        else:
            logger.info(
                "BO batch: %d evaluated, 0 valid results, %d failed",
                len(pool_positions),
                len(batch_failures),
            )

    _run_batch(initial_positions)

    while total_evaluated < config.bo_total_budget:
        remaining_budget = config.bo_total_budget - total_evaluated
        if remaining_budget <= 0:
            break

        if len(observed_X_rows) < 3:
            unevaluated = [
                p
                for p in range(len(valid_pool_indices))
                if p not in evaluated_pool_positions
            ]
            if not unevaluated:
                break
            n_extra = min(config.bo_batch_size, remaining_budget, len(unevaluated))
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
                config.bo_transfer_enabled
                and transfer_memory is not None
                and not transfer_disabled
                and len(X_current) >= config.bo_transfer_min_step_observations
                and len(transfer_memory.observed_X_rows) > 0
                and len(transfer_memory.observed_y) > 0
            )
            if can_try_transfer:
                assert transfer_memory is not None
                X_prev = pd.DataFrame(transfer_memory.observed_X_rows).reindex(
                    columns=X_current.columns, fill_value=0.0
                )
                y_prev = np.array(transfer_memory.observed_y, dtype=float)
                center = X_current.mean(axis=0).to_numpy(dtype=float)
                prev_values = X_prev.to_numpy(dtype=float)
                dist = np.linalg.norm(prev_values - center, axis=1)
                similarity = np.exp(
                    -dist / float(config.bo_transfer_similarity_lengthscale)
                )
                mask = similarity >= config.bo_transfer_min_similarity
                X_prev = X_prev.loc[mask]
                y_prev = y_prev[mask]
                similarity = similarity[mask]
                if len(X_prev) > 0:
                    n_current = len(X_current)
                    max_transfer_weight = (
                        n_current
                        * config.bo_transfer_weight_cap
                        / max(1.0 - config.bo_transfer_weight_cap, 1e-8)
                    )
                    transfer_weights = similarity / max(float(np.sum(similarity)), 1e-8)
                    transfer_weights = transfer_weights * max_transfer_weight
                    transfer_weight_share = float(
                        np.sum(transfer_weights)
                        / (np.sum(transfer_weights) + float(n_current))
                    )

                    base_model = _train_surrogate_for_bo(
                        X_current,
                        y_current,
                        config=config,
                        sample_weight=None,
                    )
                    base_mae = float(
                        np.mean(np.abs(base_model.predict(X_current) - y_current))
                    )

                    X_train = pd.concat([X_current, X_prev], ignore_index=True)
                    y_train = np.concatenate([y_current, y_prev], axis=0)
                    sample_weight = np.concatenate(
                        [np.ones(n_current, dtype=float), transfer_weights], axis=0
                    )
                    transfer_model = _train_surrogate_for_bo(
                        X_train,
                        y_train,
                        config=config,
                        sample_weight=sample_weight,
                    )
                    transfer_mae = float(
                        np.mean(np.abs(transfer_model.predict(X_current) - y_current))
                    )
                    transfer_last_mae_delta = transfer_mae - base_mae
                    if transfer_last_mae_delta > config.bo_transfer_mae_tolerance:
                        transfer_bad_rounds += 1
                    else:
                        transfer_bad_rounds = 0
                        transfer_used_rounds += 1

                    if transfer_bad_rounds >= config.bo_transfer_trust_patience:
                        transfer_disabled = True
                        transfer_disabled_reason = (
                            "trust_degraded_on_current_step_residuals"
                        )
                        X_train = X_current
                        y_train = y_current
                        sample_weight = None
                    else:
                        surrogate = transfer_model

            if surrogate is None or transfer_disabled:
                surrogate = _train_surrogate_for_bo(
                    X_train,
                    y_train,
                    config=config,
                    sample_weight=sample_weight,
                )

            batch_size = min(config.bo_batch_size, remaining_budget)
            next_positions = score_and_select(
                surrogate,
                candidate_features,
                batch_size=batch_size,
                kappa=config.bo_ucb_kappa,
                evaluated_indices=evaluated_pool_positions,
                acquisition=config.bo_acquisition,
                f_best=best_energy if np.isfinite(best_energy) else None,
            )
            explore_n = int(
                np.ceil(batch_size * config.bo_transfer_exploration_fraction)
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

    logger.info(
        "BO complete: %d total evaluated, %d valid results, best E_ads=%.4f eV",
        total_evaluated,
        len(all_results),
        best_energy if np.isfinite(best_energy) else float("nan"),
    )
    _summarize_failure_events(bo_failure_events, label=f"{molecule_name} BO evaluation")

    if not all_results:
        _flush_bo_outputs()
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "validation"
            failure_summary_out["n_candidate_specs"] = len(all_specs)
            failure_summary_out["n_valid_pool"] = len(valid_pool_indices)
            failure_summary_out["n_evaluated"] = total_evaluated
            failure_summary_out["n_valid_results"] = 0
        return None

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
        if config.bo_include_failure_negatives:
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
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "filter"
            failure_summary_out["n_candidate_specs"] = len(all_specs)
            failure_summary_out["n_valid_pool"] = len(valid_pool_indices)
            failure_summary_out["n_evaluated"] = total_evaluated
            failure_summary_out["n_before_filter"] = len(all_results)
            failure_summary_out["n_after_filter"] = 0
        return None

    logger.info(
        "BO filtered: %d -> %d results, E_ads range [%.4f, %.4f]",
        len(all_results),
        len(filtered),
        min(r.energy_adsorption for r in filtered),
        max(r.energy_adsorption for r in filtered),
    )
    if extra_ml_records_out is not None and bo_negative_records:
        extra_ml_records_out.extend(bo_negative_records)
    _flush_bo_outputs()

    return filtered

"""Core single-molecule workflow orchestration."""

import logging
import os
import time

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..conformers import create_conformers_from_smiles
from ..filters import filter_results
from ..io_results import _write_clean_xyz
from ..ml.schema import PlacementRecord
from ..models import PlacementDescriptor, ReferenceEnergies, ScreeningResult
from ..optimization import (
    clear_autobatcher_cache,
    compute_frozen_indices,
    optimize_adsorbate_slab_batched,
)
from ..placement import generators as placement_generators
from ..placement.generators import enumerate_placement_specs
from ..surfaces import SlabContainer
from .shared import (
    PlacementFailureEvent,
    _compute_slab_energy,
    _evaluate_optimized_candidate,
    _infer_surface_symbols,
    _materialize_spec_placements,
    _resolve_site_context_for_sampling,
    _summarize_failure_events,
    build_representative_relaxation_atoms,
    prepare_substrate_for_screening,
    resolve_workload_config,
    write_substrate_step_metadata,
)

logger = logging.getLogger(__name__)


def _generate_placements_with_retry(
    conformers: list[Atoms],
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    site_context: placement_generators.SiteContext | None,
    slab_atoms: Atoms,
    calculator,
    base_slab_for_frozen: Atoms | None,
) -> tuple[
    list[Atoms], list[int], list[PlacementDescriptor], list[PlacementFailureEvent], int
]:
    """Generate placements with retry loop to meet requested count.

    Returns:
        Tuple of (all_combined, placement_ids, placement_descriptors, failures, n_attempts)
    """
    assert config.num_placements is not None
    num_placements = config.num_placements
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []

    max_attempts = (
        config.placement_retry_max_attempts if config.placement_retry_enabled else 1
    )
    seed_increment = config.placement_retry_diversity_seed_increment

    for attempt in range(max_attempts):
        # Check if we've already met the target
        if len(all_combined) >= num_placements:
            break

        # Calculate seed for this attempt
        attempt_seed = config.seed + (seed_increment * attempt)

        # Generate specs for remaining placements needed
        remaining = num_placements - len(all_combined)
        if remaining <= 0:
            break

        # Generate placement specs
        specs = enumerate_placement_specs(
            conformers,
            slab_for_sites,
            config,
            smiles,
            remaining,
            filter_spec=config.placement_filter,
            site_context=site_context,
            seed=attempt_seed,
            full_slab=slab_atoms,
        )

        # Materialize placements
        (
            new_combined,
            new_ids,
            new_descriptors,
            new_failures,
        ) = _materialize_spec_placements(
            specs=specs,
            conformers=conformers,
            slab_atoms=slab_atoms,
            calculator=calculator,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
        )

        # Track results
        all_combined.extend(new_combined)
        placement_ids.extend(new_ids)
        placement_descriptors.extend(new_descriptors)
        failures.extend(new_failures)

        # Log retry attempt if needed
        if attempt > 0 and new_combined:
            logger.debug(
                "Retry attempt %d/%d: generated %d new placements (total: %d/%d)",
                attempt + 1,
                max_attempts,
                len(new_combined),
                len(all_combined),
                num_placements,
            )

    return all_combined, placement_ids, placement_descriptors, failures, max_attempts


def process_molecule(
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
    saturation_reuse: bool = False,
    symmetry_broken: bool = False,
    allow_auto_resize: bool = True,
    step_metadata_out: dict[str, object] | None = None,
) -> list[ScreeningResult] | None:
    """Run the full placement-optimise-validate pipeline for one molecule."""
    if config is None:
        config = AdsorptionConfig()

    if reference_smiles is None:
        reference_smiles = smiles

    with log_context(
        molecule=molecule_name,
        surface_type=surface_type,
        seed=config.seed,
    ):
        return _process_molecule_body(
            smiles,
            molecule_name,
            slab,
            calculator,
            reference_energies,
            ts_model,
            config,
            surface_type,
            reference_smiles,
            base_slab_for_frozen=base_slab_for_frozen,
            slab_energy_override=slab_energy_override,
            failure_summary_out=failure_summary_out,
            extra_ml_records_out=extra_ml_records_out,
            saturation_reuse=saturation_reuse,
            symmetry_broken=symmetry_broken,
            allow_auto_resize=allow_auto_resize,
            step_metadata_out=step_metadata_out,
        )


def _process_molecule_body(
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    calculator,
    reference_energies: ReferenceEnergies,
    ts_model,
    config: AdsorptionConfig,
    surface_type: str,
    reference_smiles: str,
    base_slab_for_frozen: Atoms | None = None,
    slab_energy_override: float | None = None,
    failure_summary_out: dict[str, object] | None = None,
    extra_ml_records_out: list[PlacementRecord] | None = None,
    saturation_reuse: bool = False,
    symmetry_broken: bool = False,
    allow_auto_resize: bool = True,
    step_metadata_out: dict[str, object] | None = None,
) -> list[ScreeningResult] | None:
    t_mol_start = time.perf_counter()
    logger.info(
        "Processing %s on %s surface (SMILES: %s, seed: %d)",
        molecule_name,
        surface_type,
        smiles,
        config.seed,
    )

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

    t0 = time.perf_counter()
    conformer_pack = create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )
    t_conformers = time.perf_counter() - t0
    if conformer_pack is None:
        logger.error("Could not generate conformers for %s", molecule_name)
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "conformers"
            failure_summary_out["reason"] = (
                f"could not generate conformers for {molecule_name}"
            )
        return None
    conformers, conformer_energies = conformer_pack

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

    t0 = time.perf_counter()
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor] = []
    placement_failure_events: list[PlacementFailureEvent] = []

    site_context = _resolve_site_context_for_sampling(
        slab_for_sites,
        config,
        symmetry_broken=symmetry_broken,
    )

    freeze_ref = (
        effective_base_slab_for_frozen
        if effective_base_slab_for_frozen is not None
        else slab.atoms
    )
    frozen_indices = compute_frozen_indices(freeze_ref, config)
    representative_atoms = build_representative_relaxation_atoms(
        conformers,
        slab.atoms,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
    )
    config = resolve_workload_config(
        config,
        ts_model=ts_model,
        representative_atoms=representative_atoms,
        frozen_indices=frozen_indices,
        bo_enabled=config.bo_enabled,
    )
    assert config.num_placements is not None

    # Generate placements with optional retry loop
    (
        all_combined,
        placement_ids,
        placement_descriptors,
        placement_failure_events,
        n_placement_attempts,
    ) = _generate_placements_with_retry(
        conformers,
        slab_for_sites,
        config,
        smiles,
        site_context,
        slab.atoms,
        calculator,
        base_slab_for_frozen,
    )
    t_placement = time.perf_counter() - t0

    # Log retry info if applicable
    if config.placement_retry_enabled and n_placement_attempts > 1:
        logger.info(
            "Placement generation: %d attempts, %d/%d valid placements (%.2fs)",
            n_placement_attempts,
            len(all_combined),
            config.num_placements,
            t_placement,
        )
    else:
        logger.info(
            "Generated %d/%d valid initial placements (%.2fs)",
            len(all_combined),
            config.num_placements,
            t_placement,
        )
    _summarize_failure_events(
        placement_failure_events,
        label=f"{molecule_name} placement generation",
    )

    if config.debug_write_initial_placements and all_combined:
        xyz_dir = f"results_{surface_type}/xyz_structures/{molecule_name}_all"
        os.makedirs(xyz_dir, exist_ok=True)
        for combined, pid in zip(all_combined, placement_ids, strict=True):
            path = f"{xyz_dir}/initial_{pid:03d}.xyz"
            _write_clean_xyz(combined, path)
        logger.info(
            "Wrote %d initial placement XYZ files to %s", len(all_combined), xyz_dir
        )

    if not all_combined:
        logger.warning("No valid placements")
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "placement"
            failure_summary_out["n_placements_attempted"] = config.num_placements
            failure_summary_out["n_initial_placements"] = 0
            if config.placement_retry_enabled:
                failure_summary_out["n_retry_attempts"] = n_placement_attempts
        return None

    clear_autobatcher_cache()

    t0 = time.perf_counter()
    optimized = optimize_adsorbate_slab_batched(
        all_combined,
        slab.atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=effective_base_slab_for_frozen,
        saturation_reuse=saturation_reuse,
    )
    t_optimization = time.perf_counter() - t0

    # Early exit if optimization failed for all placements
    if not optimized:
        logger.warning("Optimization failed for all placements")
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "optimization"
            failure_summary_out["n_placements_attempted"] = len(placement_ids)
            failure_summary_out["n_initial_placements"] = len(all_combined)
        return None

    t0 = time.perf_counter()
    surface_symbols = _infer_surface_symbols(slab_for_sites)
    if base_slab_for_frozen is not None:
        logger.info(
            "Saturation surface reference: full_slab_atoms=%d, "
            "surface_ref_atoms=%d, freeze_ref_atoms=%d, frozen_policy=%s, "
            "surface_symbols=%s",
            len(slab.atoms),
            len(slab_for_sites),
            len(effective_base_slab_for_frozen or slab.atoms),
            "top_layer" if config.relax_top_layer else "full_substrate",
            surface_symbols,
        )
    results: list[ScreeningResult] = []
    validation_failures: dict[str, int] = {}
    n_optimization_failed = sum(1 for o in optimized if o is None)
    validation_failure_events: list[PlacementFailureEvent] = []
    for opt_atoms, pid, descriptor in zip(
        optimized, placement_ids, placement_descriptors, strict=True
    ):
        with log_context(placement_id=pid):
            result, failure_event = _evaluate_optimized_candidate(
                opt_atoms=opt_atoms,
                placement_id=pid,
                descriptor=descriptor,
                molecule_name=molecule_name,
                slab_atoms=slab.atoms,
                calculator=calculator,
                config=config,
                E_slab=E_slab,
                E_mol=E_mol,
                surface_symbols=surface_symbols,
            )
        if result is None:
            if failure_event is not None:
                validation_failures[failure_event.reason] = (
                    validation_failures.get(failure_event.reason, 0) + 1
                )
                validation_failure_events.append(failure_event)
            continue
        results.append(result)
        logger.info(
            "E_ads = %.4f eV, distance = %.2f A",
            result.energy_adsorption,
            result.distance,
        )
    t_validation = time.perf_counter() - t0

    _summarize_failure_events(
        validation_failure_events,
        label=f"{molecule_name} optimisation/validation",
    )

    if not results:
        logger.warning("No valid placements after validation")
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "validation"
            failure_summary_out["n_initial_placements"] = len(all_combined)
            failure_summary_out["n_optimized"] = (
                len(all_combined) - n_optimization_failed
            )
            failure_summary_out["n_optimization_failed"] = n_optimization_failed
            failure_summary_out["validation_failures"] = validation_failures
        return None

    t0 = time.perf_counter()
    n_before_filter = len(results)
    deduplicated_results: list[ScreeningResult] = []
    results = filter_results(
        results,
        slab=slab.atoms,
        surface_symbols=surface_symbols,
        reference_smiles=reference_smiles,
        config=config,
        duplicate_results_out=deduplicated_results,
    )
    t_filtering = time.perf_counter() - t0

    if extra_ml_records_out is not None and deduplicated_results:
        for dup in deduplicated_results:
            record = PlacementRecord.from_screening_result(
                dup,
                smiles=smiles,
                surface_id=surface_type,
                config=config,
            )
            record.label_source = "deduplicated_duplicate"
            extra_ml_records_out.append(record)

    if not results and failure_summary_out is not None:
        failure_summary_out["stage"] = "filter"
        failure_summary_out["n_before_filter"] = n_before_filter
        failure_summary_out["n_after_filter"] = 0

    t_mol_total = time.perf_counter() - t_mol_start

    logger.info(
        "%d unique configs, E_ads [%.4f, %.4f] eV | "
        "timing: conformers=%.2fs placement=%.2fs opt=%.2fs "
        "validation=%.2fs filter=%.2fs total=%.2fs",
        len(results),
        min(r.energy_adsorption for r in results) if results else float("nan"),
        max(r.energy_adsorption for r in results) if results else float("nan"),
        t_conformers,
        t_placement,
        t_optimization,
        t_validation,
        t_filtering,
        t_mol_total,
    )

    return results


def _evaluate_placement_batch(
    specs: list,
    conformers: list[Atoms],
    slab: SlabContainer,
    calculator,
    ts_model,
    config: AdsorptionConfig,
    smiles: str,
    E_slab: float,
    E_mol: float,
    molecule_name: str,
    surface_symbols: list[str] | None,
    site_context: placement_generators.SiteContext | None = None,
    base_slab_for_frozen: Atoms | None = None,
    slab_for_sites: Atoms | None = None,
) -> tuple[list[ScreeningResult], list[PlacementFailureEvent]]:
    """Run placement-generation + optimization + validation for a batch of specs."""
    (
        all_combined,
        placement_ids,
        placement_descriptors,
        failures,
    ) = _materialize_spec_placements(
        specs=specs,
        conformers=conformers,
        slab_atoms=slab.atoms,
        calculator=calculator,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
    )

    if not all_combined:
        return [], failures

    clear_autobatcher_cache()
    optimized = optimize_adsorbate_slab_batched(
        all_combined,
        slab.atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab_for_frozen,
    )

    # Early exit if optimization failed for all placements
    if not optimized:
        logger.warning("Optimization failed for all placements in saturation")
        return [], []

    results: list[ScreeningResult] = []
    for opt_atoms, pid, descriptor in zip(
        optimized, placement_ids, placement_descriptors, strict=True
    ):
        result, failure_event = _evaluate_optimized_candidate(
            opt_atoms=opt_atoms,
            placement_id=pid,
            descriptor=descriptor,
            molecule_name=molecule_name,
            slab_atoms=slab.atoms,
            calculator=calculator,
            config=config,
            E_slab=E_slab,
            E_mol=E_mol,
            surface_symbols=surface_symbols,
            log_prefix="BO batch ",
        )
        if result is None:
            if failure_event is not None:
                failures.append(failure_event)
            continue
        results.append(result)

    return results, failures

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
    optimize_adsorbate_slab_batched,
)
from ..placement import (
    enumerate_placement_specs,
)
from ..placement import generators as placement_generators
from ..surfaces import SlabContainer, auto_resize_slab_for_molecule
from .shared import (
    PlacementFailureEvent,
    _build_surface_reference_slab,
    _compute_slab_energy,
    _evaluate_optimized_candidate,
    _infer_surface_symbols,
    _materialize_spec_placements,
    _resolve_site_context_for_sampling,
    _summarize_failure_events,
)

logger = logging.getLogger(__name__)


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
    result = create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )
    t_conformers = time.perf_counter() - t0
    if result is None:
        logger.error("Could not generate conformers for %s", molecule_name)
        if failure_summary_out is not None:
            failure_summary_out["stage"] = "conformers"
            failure_summary_out["reason"] = (
                f"could not generate conformers for {molecule_name}"
            )
        return None
    conformers, conformer_energies = result

    slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab_for_frozen)

    if config.auto_resize_slab and allow_auto_resize:
        slab, was_resized = auto_resize_slab_for_molecule(
            slab, conformers, config.min_pbc_image_separation
        )
        if was_resized:
            clear_autobatcher_cache()
            E_slab = _compute_slab_energy(
                slab.atoms, calculator, label="resized slab reference"
            )
            logger.info("Resized slab energy: %.4f eV", E_slab)
            slab_for_sites = _build_surface_reference_slab(
                slab.atoms, base_slab_for_frozen
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

    specs = enumerate_placement_specs(
        conformers,
        slab_for_sites,
        config,
        smiles,
        config.num_placements,
        filter_spec=config.placement_filter,
        site_context=site_context,
    )
    (
        all_combined,
        placement_ids,
        generated_descriptors,
        generated_failures,
    ) = _materialize_spec_placements(
        specs=specs,
        conformers=conformers,
        slab_atoms=slab.atoms,
        calculator=calculator,
        config=config,
        smiles=smiles,
        site_context=site_context,
    )
    placement_descriptors.extend(generated_descriptors)
    placement_failure_events.extend(generated_failures)
    t_placement = time.perf_counter() - t0

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
        return None

    clear_autobatcher_cache()

    t0 = time.perf_counter()
    optimized = optimize_adsorbate_slab_batched(
        all_combined,
        slab.atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab_for_frozen,
        saturation_reuse=saturation_reuse,
    )
    t_optimization = time.perf_counter() - t0

    t0 = time.perf_counter()
    surface_symbols = _infer_surface_symbols(slab_for_sites)
    if base_slab_for_frozen is not None:
        logger.info(
            "Saturation surface reference: full_slab_atoms=%d, surface_ref_atoms=%d, surface_symbols=%s",
            len(slab.atoms),
            len(slab_for_sites),
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

    if validation_failures:
        logger.info(
            "Validation failures: %s",
            ", ".join(f"{reason}: {n}" for reason, n in validation_failures.items()),
        )
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

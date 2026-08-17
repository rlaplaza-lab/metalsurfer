"""Core single-molecule workflow orchestration."""

import logging
import os
import time

from ase import Atoms

from .._logging import log_context
from ..config import AdsorptionConfig
from ..io_results import _write_clean_xyz
from ..ml.schema import PlacementRecord
from ..models import (
    PlacementDescriptor,
    ReferenceEnergies,
    ScreeningResult,
)
from ..placement.site_context import SiteContext
from ..surface_prep import SlabContainer
from .placement_fill import (
    fill_materialized_placements,
    materialize_specs_filling_target,
)
from .shared import (
    MoleculeScreenOutcome,
    PlacementFailureEvent,
    _finalize_screen_results,
    _generation_failure_histogram,
    _infer_surface_symbols,
    _optimize_and_evaluate_placements,
    _prepare_molecule_screening,
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
    saturation_reuse: bool = False,
    symmetry_broken: bool = False,
    conformers: list[Atoms] | None = None,
    conformer_energies: list[float] | None = None,
    skip_workload_autotune: bool = False,
) -> MoleculeScreenOutcome:
    """Run the full placement-optimise-validate pipeline for one molecule.

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
        Adsorption configuration.
    surface_type
        Surface type label.
    reference_smiles
        SMILES used for reference energy lookup.
    base_slab_for_frozen
        Base slab for freeze constraints.
    slab_energy_override
        Override slab reference energy.
    saturation_reuse
        Whether to reuse saturation-optimized substrate.
    symmetry_broken
        Whether symmetry is broken.
    conformers
        Pre-generated conformers (optional).
    conformer_energies
        Energies aligned with conformers (optional).
    skip_workload_autotune
        Whether to skip workload autotuning.
    """
    if config is None:
        config = AdsorptionConfig()

    if reference_smiles is None:
        reference_smiles = smiles

    failure_summary: dict[str, object] = {}
    ml_records: list[PlacementRecord] = []

    with log_context(
        molecule=molecule_name,
        surface_type=surface_type,
        seed=config.seed,
    ):
        t_mol_start = time.perf_counter()
        logger.info(
            "Processing %s on %s surface (SMILES: %s, seed: %d)",
            molecule_name,
            surface_type,
            smiles,
            config.seed,
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
            bo_enabled=False,
            conformers=conformers,
            conformer_energies=conformer_energies,
            skip_workload_autotune=skip_workload_autotune,
        )
        if ctx is None:
            return MoleculeScreenOutcome(
                results=[],
                failure_summary=failure_summary,
                ml_records=ml_records,
            )

        slab = ctx.slab
        slab_for_sites = ctx.slab_for_sites
        effective_base_slab_for_frozen = ctx.effective_base_slab_for_frozen
        conformers = ctx.conformers
        site_context = ctx.site_context
        config = ctx.config
        E_mol = ctx.E_mol
        t_conformers = ctx.t_conformers
        E_slab = ctx.E_slab
        conformer_energies = ctx.conformer_energies

        t0 = time.perf_counter()
        fill = fill_materialized_placements(
            conformers=conformers,
            slab_for_sites=slab_for_sites,
            config=config,
            smiles=smiles,
            site_context=site_context,
            slab_atoms=slab.atoms,
            calculator=calculator,
            conformer_energies=conformer_energies,
        )
        all_combined = fill.combined
        placement_ids = fill.placement_ids
        placement_descriptors = fill.descriptors
        placement_failure_events = fill.failures
        n_placement_attempts = fill.n_attempts
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
                "Wrote %d initial placement XYZ files to %s",
                len(all_combined),
                xyz_dir,
            )

        if not all_combined:
            logger.warning("No valid placements")
            failure_summary["stage"] = "placement"
            failure_summary["n_placements_attempted"] = config.num_placements
            failure_summary["n_initial_placements"] = 0
            failure_summary["generation_failures"] = _generation_failure_histogram(
                placement_failure_events
            )
            if config.placement_retry_enabled:
                failure_summary["n_retry_attempts"] = n_placement_attempts
            return MoleculeScreenOutcome(
                results=[],
                failure_summary=failure_summary,
                ml_records=ml_records,
            )

        surface_symbols = _infer_surface_symbols(slab_for_sites)
        if base_slab_for_frozen is not None:
            logger.info(
                "Saturation surface reference: full_slab_atoms=%d, "
                "surface_ref_atoms=%d, freeze_ref_atoms=%d, surface_symbols=%s",
                len(slab.atoms),
                len(slab_for_sites),
                len(effective_base_slab_for_frozen or slab.atoms),
                surface_symbols,
            )

        t0 = time.perf_counter()
        results, validation_failure_events, n_optimization_failed = (
            _optimize_and_evaluate_placements(
                all_combined,
                placement_ids,
                placement_descriptors,
                slab=slab.atoms,
                calculator=calculator,
                ts_model=ts_model,
                config=config,
                energies=(E_slab, E_mol),
                molecule_name=molecule_name,
                surface_symbols=surface_symbols,
                base_slab_for_frozen=effective_base_slab_for_frozen,
                saturation_reuse=saturation_reuse,
            )
        )
        t_optimization = time.perf_counter() - t0
        t_validation = 0.0

        if not results and not validation_failure_events and n_optimization_failed == 0:
            failure_summary["stage"] = "optimization"
            failure_summary["n_placements_attempted"] = len(placement_ids)
            failure_summary["n_initial_placements"] = len(all_combined)
            return MoleculeScreenOutcome(
                results=[],
                failure_summary=failure_summary,
                ml_records=ml_records,
            )

        validation_failures: dict[str, int] = {}
        for failure_event in validation_failure_events:
            validation_failures[failure_event.reason] = (
                validation_failures.get(failure_event.reason, 0) + 1
            )
        for result in results:
            logger.info(
                "E_ads = %.4f eV, distance = %.2f A",
                result.energy_adsorption,
                result.distance,
            )

        _summarize_failure_events(
            validation_failure_events,
            label=f"{molecule_name} optimisation/validation",
        )

        if not results:
            logger.warning("No valid placements after validation")
            failure_summary["stage"] = "validation"
            failure_summary["n_initial_placements"] = len(all_combined)
            failure_summary["n_optimized"] = len(all_combined) - n_optimization_failed
            failure_summary["n_optimization_failed"] = n_optimization_failed
            failure_summary["validation_failures"] = validation_failures
            return MoleculeScreenOutcome(
                results=[],
                failure_summary=failure_summary,
                ml_records=ml_records,
            )

        results, t_filtering = _finalize_screen_results(
            results,
            slab_atoms=slab.atoms,
            surface_symbols=surface_symbols,
            reference_smiles=reference_smiles,
            config=config,
            smiles=smiles,
            surface_type=surface_type,
            failure_summary=failure_summary,
            ml_records=ml_records,
        )

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

        return MoleculeScreenOutcome(
            results=results,
            failure_summary=failure_summary,
            ml_records=ml_records,
        )


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
    site_context: SiteContext | None = None,
    base_slab_for_frozen: Atoms | None = None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
    backfill_specs: list | None = None,
    n_target: int | None = None,
) -> tuple[list[ScreeningResult], list[PlacementFailureEvent], int]:
    """Run placement-generation + optimization + validation for a batch of specs.

    When ``backfill_specs`` is provided, materialization failures in ``specs`` are
    replaced from the backfill list until ``n_target`` (default: ``len(specs)``)
    geometry-valid placements are obtained or the backfill is exhausted.

    Returns
    -------
    ``(results, failures, n_backfill_used)``
    """
    target = len(specs) if n_target is None else n_target
    fill = materialize_specs_filling_target(
        primary_specs=specs,
        backfill_specs=backfill_specs or [],
        n_target=target,
        conformers=conformers,
        slab_atoms=slab.atoms,
        calculator=calculator,
        config=config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
        materialization_cache=materialization_cache,
    )
    all_combined = fill.combined
    placement_ids = fill.placement_ids
    placement_descriptors = fill.descriptors
    failures = list(fill.failures)

    if not all_combined:
        return [], failures, fill.n_backfill_used

    results, validation_failure_events, _n_optimization_failed = (
        _optimize_and_evaluate_placements(
            all_combined,
            placement_ids,
            placement_descriptors,
            slab=slab.atoms,
            calculator=calculator,
            ts_model=ts_model,
            config=config,
            energies=(E_slab, E_mol),
            molecule_name=molecule_name,
            surface_symbols=surface_symbols,
            base_slab_for_frozen=base_slab_for_frozen,
            log_prefix="BO batch ",
        )
    )
    if not results and not validation_failure_events and _n_optimization_failed == 0:
        # Optimize returned empty; preserve prior batch behavior (drop gen failures).
        return [], [], fill.n_backfill_used

    failures.extend(validation_failure_events)
    return results, failures, fill.n_backfill_used

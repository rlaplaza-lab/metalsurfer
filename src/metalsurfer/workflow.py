"""Workflow: process_molecule (single), run_screening (batch). Compute-pure; I/O in io_results."""

import logging
import os
import time
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms

from ._logging import log_context, screening_log_context, warn_once
from .config import AdsorptionConfig
from .conformers import create_conformers_from_smiles
from .exceptions import OptimizationError
from .filters import filter_results
from .io_results import _write_clean_xyz
from .ml.dataset import DatasetLogger
from .models import (
    PlacementDescriptor,
    ReferenceEnergies,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)
from .optimization import (
    clear_autobatcher_cache,
    optimize_adsorbate_slab_batched,
    optimize_isolated_molecules_batched,
    setup_single_model,
)
from .placement import (
    calculate_min_distance,
    enumerate_placement_specs,
    generate_conformer_placement,
    generate_placement_from_spec,
)
from .surfaces import SlabContainer, auto_resize_slab_for_molecule

logger = logging.getLogger(__name__)


def calculate_reference_energies(
    slab: SlabContainer,
    calculator,
    molecules: list[str],
    smiles_list: list[str],
    ts_model=None,
    config: AdsorptionConfig | None = None,
) -> ReferenceEnergies:
    """Compute clean-slab and isolated-molecule energies."""
    if config is None:
        config = AdsorptionConfig()

    slab_copy = slab.atoms.copy()
    slab_copy.calc = calculator
    slab_energy = slab_copy.get_potential_energy()
    if not np.isfinite(slab_energy):
        raise OptimizationError(
            f"Clean slab energy is not finite: {slab_energy}. "
            "The calculator may have failed; check GPU stability and model output."
        )
    if abs(slab_energy) < 1e-6:
        raise OptimizationError(
            f"Clean slab energy is effectively zero ({slab_energy:.6e} eV). "
            "A real slab cannot have zero energy; the calculator likely returned "
            "a default. Check that the ML model produced valid output."
        )
    logger.info("Clean slab energy: %.4f eV", slab_energy)

    molecule_energies: dict[str, float] = {}
    for mol_name, smiles in zip(molecules, smiles_list, strict=False):
        logger.info("Calculating isolated %s energy...", mol_name)
        result = create_conformers_from_smiles(
            smiles, calculator=calculator, config=config, ts_model=ts_model
        )
        if result is None:
            if config.fail_on_conformer_failure:
                raise RuntimeError(
                    f"Could not create conformers for {mol_name} from SMILES: {smiles}"
                )
            logger.warning("Could not create %s from SMILES: %s", mol_name, smiles)
            continue
        conformers, _ = result
        opt_results = optimize_isolated_molecules_batched(
            conformers,
            ts_model,
            fmax=config.fmax,
            steps=config.reference_optimization_steps,
            config=config,
        )
        energies = [
            (e, i) for i, r in enumerate(opt_results) if r is not None for _, e in [r]
        ]
        if energies:
            energies.sort()
            molecule_energies[mol_name] = energies[0][0]
            logger.info(
                "%s isolated energy: %.4f eV (best conformer: %d)",
                mol_name,
                energies[0][0],
                energies[0][1],
            )
        else:
            if config.fail_on_conformer_failure:
                raise OptimizationError(
                    f"Failed to optimise any conformers for {mol_name}"
                )
            logger.error("Failed to optimise any conformers for %s", mol_name)

    clear_autobatcher_cache()
    return ReferenceEnergies(
        slab_energy=slab_energy,
        molecule_energies=molecule_energies,
    )


def _validate_geometry(
    atoms: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
) -> tuple[bool, str]:
    """Quick sanity checks on the optimised structure."""
    energy = atoms.get_potential_energy()
    if not np.isfinite(energy):
        return False, f"non-finite energy: {energy}"

    from scipy.spatial.distance import pdist

    dists = pdist(atoms.get_positions())
    if len(dists) > 0 and np.min(dists) < config.min_interatomic_distance:
        return False, f"atoms too close: {np.min(dists):.3f} A"

    forces = atoms.get_forces()
    slab_size = len(slab)
    ads_forces = forces[slab_size:]
    if len(ads_forces) > 0:
        max_f = float(np.max(np.linalg.norm(ads_forces, axis=1)))
        if max_f > config.max_force_convergence:
            return False, f"high adsorbate forces: {max_f:.3f} eV/A"

    return True, "geometry valid"


def _validate_adsorption(
    atoms: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
) -> tuple[bool, str]:
    slab_size = len(slab)
    adsorbate = atoms[slab_size:]
    if len(adsorbate) == 0:
        return False, "no adsorbate atoms"

    if config.skip_desorption_check:
        warn_once(
            logger,
            "skip_desorption",
            "skip_desorption_check=True: desorption distance validation skipped",
        )
        return True, "desorption check skipped"

    cell = atoms.get_cell()
    min_d = calculate_min_distance(
        adsorbate.get_positions(), slab.get_positions(), cell, use_pbc=True
    )
    if min_d > config.binding_distance_threshold:
        return False, f"desorbed ({min_d:.2f} A)"
    return True, f"adsorbed ({min_d:.2f} A)"


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
) -> list[ScreeningResult] | None:
    """Run the full placement-optimise-validate pipeline for one molecule.

    Returns a list of :class:`ScreeningResult` objects or ``None`` when no
    valid placements survive validation and filtering.

    When *base_slab_for_frozen* is set (e.g. for sequential saturation),
    only the original surface atoms are frozen during optimisation.
    When *slab_energy_override* is set, it overrides the reference slab energy.

    When *failure_summary_out* is a mutable dict and the pipeline fails,
    it is populated with debugging info (stage, counts, validation_failures).
    Use :func:`format_failure_summary` to produce a human-readable summary.
    """
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

    # generate conformers
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

    # auto-resize slab only for saturation/pre-adsorbed runs; for simple binding
    # energy, a larger cell is ineffective due to translational symmetry
    if config.auto_resize_slab and base_slab_for_frozen is not None:
        slab, was_resized = auto_resize_slab_for_molecule(
            slab, conformers, config.min_pbc_image_separation
        )
        if was_resized:
            # Free GPU memory from conformer creation before large slab calculation.
            # Without this, cached tensors/fragmentation can cause OOM when
            # transitioning from small (conformer) to large (resized slab) systems.
            clear_autobatcher_cache()
            resized_copy = slab.atoms.copy()
            resized_copy.calc = calculator
            E_slab = float(resized_copy.get_potential_energy())
            logger.info("Resized slab energy: %.4f eV", E_slab)

    # placements
    t0 = time.perf_counter()
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor | None] = []
    use_dissociative = (
        config.skip_topology_check
        and len(conformers) > 0
        and len(conformers[0]) == 2
        and conformers[0].get_chemical_symbols()[0]
        == conformers[0].get_chemical_symbols()[1]
    )
    if use_dissociative:
        # Saturation only: use metal atoms of current slab for site detection so
        # placements span the full (possibly resized) cell. Simple binding-energy
        # runs pass base_slab_for_frozen=None and use slab directly.
        if base_slab_for_frozen is not None:
            metal_symbols = set(base_slab_for_frozen.get_chemical_symbols())
            symbols = slab.atoms.get_chemical_symbols()
            mask = [s in metal_symbols for s in symbols]
            if all(mask):
                slab_for_sites = slab.atoms  # All metal, no copy
            elif any(mask):
                slab_for_sites = slab.atoms[mask].copy()
            else:
                slab_for_sites = None
        else:
            slab_for_sites = None
        for i in range(config.num_placements):
            adsorbate = generate_conformer_placement(
                conformers,
                conformer_energies,
                slab.atoms,
                i,
                config=config,
                smiles=smiles,
                slab_for_sites=slab_for_sites,
            )
            if adsorbate is None:
                continue
            # Descriptor for reproducibility: dissociative placements use centroid
            # and placeholder orientation/site fields (no single spec).
            pos = adsorbate.get_positions()
            centroid = pos.mean(axis=0)
            surface_z = float(np.max(slab.atoms.get_positions()[:, 2]))
            z_lo, z_hi = config.placement_z_range
            z_frac = (
                (centroid[2] - surface_z - z_lo) / (z_hi - z_lo) if z_hi > z_lo else 0.5
            )
            z_frac = max(0.0, min(1.0, z_frac))
            dissociative_descriptor = PlacementDescriptor(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=-1,
                site_type="hollow",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=z_frac,
                placement_index=i,
                x=float(centroid[0]),
                y=float(centroid[1]),
                z=float(centroid[2] - surface_z),
                shape="linear",
                slab_indices=None,
            )
            combined = slab.atoms + adsorbate
            combined.set_pbc([True, True, True])
            combined.calc = calculator
            all_combined.append(combined)
            placement_ids.append(i)
            placement_descriptors.append(dissociative_descriptor)
    else:
        specs = enumerate_placement_specs(
            conformers,
            slab.atoms,
            config,
            smiles,
            config.num_placements,
            filter_spec=config.placement_filter,
        )
        for spec in specs:
            result = generate_placement_from_spec(
                spec,
                conformers,
                slab.atoms,
                config,
                smiles=smiles,
            )
            if result is None:
                continue
            adsorbate, descriptor = result
            combined = slab.atoms + adsorbate
            combined.set_pbc([True, True, True])
            combined.calc = calculator
            all_combined.append(combined)
            placement_ids.append(descriptor.placement_index)
            placement_descriptors.append(descriptor)
    t_placement = time.perf_counter() - t0

    logger.info(
        "Generated %d/%d valid initial placements (%.2fs)",
        len(all_combined),
        config.num_placements,
        t_placement,
    )

    if config.debug_write_initial_placements and all_combined:
        xyz_dir = f"results_{surface_type}/xyz_structures/{molecule_name}_all"
        os.makedirs(xyz_dir, exist_ok=True)
        for combined, pid in zip(all_combined, placement_ids, strict=False):
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

    # Aggressive GPU cleanup before slab optimization: evict any cached autobatchers
    # from the isolated-molecule run and free their memory. Without this, the
    # previous torch_sim run can still occupy GPU memory when the slab batcher
    # starts its memory estimation.
    clear_autobatcher_cache()

    # batched optimisation
    t0 = time.perf_counter()
    optimized = optimize_adsorbate_slab_batched(
        all_combined,
        slab.atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab_for_frozen,
    )
    t_optimization = time.perf_counter() - t0

    # per-placement validation
    t0 = time.perf_counter()
    surface_symbols = _infer_surface_symbols(slab.atoms)
    results: list[ScreeningResult] = []
    validation_failures: dict[str, int] = {}
    n_optimization_failed = sum(1 for o in optimized if o is None)
    for opt_atoms, pid, descriptor in zip(
        optimized, placement_ids, placement_descriptors, strict=False
    ):
        if opt_atoms is None:
            continue
        if opt_atoms.calc is None:
            opt_atoms.calc = calculator

        with log_context(placement_id=pid):
            ok, reason = _validate_geometry(opt_atoms, slab.atoms, config)
            if not ok:
                logger.debug("geometry fail: %s", reason)
                validation_failures[reason] = validation_failures.get(reason, 0) + 1
                continue

            ok, reason = _validate_adsorption(opt_atoms, slab.atoms, config)
            if not ok:
                logger.debug("adsorption fail: %s", reason)
                validation_failures[reason] = validation_failures.get(reason, 0) + 1
                continue

            e_adslab = opt_atoms.get_potential_energy()
            e_ads = e_adslab - E_slab - E_mol
            if e_ads > config.max_adsorption_energy:
                reason = f"E_ads too high: {e_ads:.4f} eV"
                logger.debug("unrealistic E_ads: %.4f eV", e_ads)
                validation_failures[reason] = validation_failures.get(reason, 0) + 1
                continue

            slab_size = len(slab.atoms)
            mol_atoms = opt_atoms[slab_size:]
            slab_opt = opt_atoms[:slab_size]
            cell = opt_atoms.get_cell()
            dist = calculate_min_distance(
                mol_atoms.get_positions(),
                slab_opt.get_positions(),
                cell,
                use_pbc=True,
            )

            results.append(
                ScreeningResult(
                    molecule=molecule_name,
                    placement_id=pid,
                    energy_adslab=e_adslab,
                    energy_slab=E_slab,
                    energy_adsorbate=E_mol,
                    energy_adsorption=e_ads,
                    atoms=opt_atoms.copy(),
                    distance=dist,
                    placement_descriptor=descriptor,
                )
            )
            logger.info("E_ads = %.4f eV, distance = %.2f A", e_ads, dist)
    t_validation = time.perf_counter() - t0

    if validation_failures:
        logger.info(
            "Validation failures: %s",
            ", ".join(f"{reason}: {n}" for reason, n in validation_failures.items()),
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

    # centralised filtering (decomposition + desorption + duplicates)
    t0 = time.perf_counter()
    n_before_filter = len(results)
    results = filter_results(
        results,
        slab=slab.atoms,
        surface_symbols=surface_symbols,
        reference_smiles=reference_smiles,
        config=config,
    )
    t_filtering = time.perf_counter() - t0

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


def format_failure_summary(failure_summary: dict[str, object]) -> str:
    """Produce a human-readable multi-line summary from a failure_summary dict.

    Use when *failure_summary_out* passed to :func:`process_molecule` was
    populated (pipeline returned None or []).
    """
    lines = ["Failure summary:"]
    stage = failure_summary.get("stage", "unknown")
    lines.append(f"  Stage: {stage}")

    if stage == "reference" or stage == "conformers":
        reason = failure_summary.get("reason", "")
        if reason:
            lines.append(f"  Reason: {reason}")
    elif stage == "placement":
        n_attempted = failure_summary.get("n_placements_attempted", "?")
        n_initial = failure_summary.get("n_initial_placements", 0)
        lines.append(f"  Placements attempted: {n_attempted}")
        lines.append(f"  Initial placements: {n_initial}")
    elif stage == "validation":
        n_initial = failure_summary.get("n_initial_placements", "?")
        n_opt = failure_summary.get("n_optimized", "?")
        n_opt_fail = failure_summary.get("n_optimization_failed", 0)
        lines.append(f"  Initial placements: {n_initial}")
        lines.append(f"  Optimized: {n_opt} ({n_opt_fail} failed)")
        lines.append("  Passed validation: 0")
        vf = failure_summary.get("validation_failures", {})
        if vf:
            lines.append("  Validation failures:")
            for reason, count in sorted(vf.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason}: {count}")
    elif stage == "filter":
        n_before = failure_summary.get("n_before_filter", "?")
        n_after = failure_summary.get("n_after_filter", 0)
        lines.append(f"  Before filter: {n_before}")
        lines.append(f"  After filter: {n_after}")

    return "\n".join(lines)


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
    surface_type: str,
    base_slab_for_frozen: Atoms | None = None,
) -> list[ScreeningResult]:
    """Run placement-generation + optimization + validation for a batch of specs.

    Shared helper used by both the standard and BO workflows.
    Returns validated ScreeningResult objects (may be fewer than input specs).
    """
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor | None] = []

    for spec in specs:
        result = generate_placement_from_spec(
            spec, conformers, slab.atoms, config, smiles=smiles
        )
        if result is None:
            continue
        adsorbate, descriptor = result
        combined = slab.atoms + adsorbate
        combined.set_pbc([True, True, True])
        combined.calc = calculator
        all_combined.append(combined)
        placement_ids.append(descriptor.placement_index)
        placement_descriptors.append(descriptor)

    if not all_combined:
        return []

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
        optimized, placement_ids, placement_descriptors, strict=False
    ):
        if opt_atoms is None:
            continue
        if opt_atoms.calc is None:
            opt_atoms.calc = calculator

        ok, reason = _validate_geometry(opt_atoms, slab.atoms, config)
        if not ok:
            logger.debug("BO batch geometry fail: %s", reason)
            continue

        ok, reason = _validate_adsorption(opt_atoms, slab.atoms, config)
        if not ok:
            logger.debug("BO batch adsorption fail: %s", reason)
            continue

        e_adslab = opt_atoms.get_potential_energy()
        e_ads = e_adslab - E_slab - E_mol
        if e_ads > config.max_adsorption_energy:
            continue

        slab_size = len(slab.atoms)
        mol_atoms = opt_atoms[slab_size:]
        slab_opt = opt_atoms[:slab_size]
        cell = opt_atoms.get_cell()
        dist = calculate_min_distance(
            mol_atoms.get_positions(),
            slab_opt.get_positions(),
            cell,
            use_pbc=True,
        )

        results.append(
            ScreeningResult(
                molecule=smiles,
                placement_id=pid,
                energy_adslab=e_adslab,
                energy_slab=E_slab,
                energy_adsorbate=E_mol,
                energy_adsorption=e_ads,
                atoms=opt_atoms.copy(),
                distance=dist,
                placement_descriptor=descriptor,
            )
        )

    return results


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
) -> list[ScreeningResult] | None:
    """Bayesian-optimisation-guided placement screening for one molecule.

    1. Enumerate a large pool of candidate PlacementSpecs.
    2. Evaluate ``config.bo_initial_random`` placements at random.
    3. Train a Random Forest surrogate on observed (features, E_ads).
    4. Use acquisition (LCB, EI, or PI per ``config.bo_acquisition``) to pick the next ``config.bo_batch_size`` candidates.
    5. Repeat until ``config.bo_total_budget`` evaluations are exhausted.
    6. Apply the standard ``filter_results`` and return.
    """
    from .ml.bayesian import (
        build_candidate_features,
        score_and_select,
        train_surrogate,
    )
    from .ml.features import extract_features

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
        return None

    # Generate conformers
    result = create_conformers_from_smiles(
        smiles, calculator=calculator, config=config, ts_model=ts_model
    )
    if result is None:
        logger.error("Could not generate conformers for %s", molecule_name)
        return None
    conformers, conformer_energies = result

    pool_size = config.bo_candidate_pool_size or max(
        config.bo_total_budget * 5, config.num_placements
    )
    all_specs = enumerate_placement_specs(
        conformers,
        slab.atoms,
        config,
        smiles,
        pool_size,
        filter_spec=config.placement_filter,
    )
    if not all_specs:
        logger.warning("No candidate specs generated for BO")
        return None

    logger.info(
        "BO: %d candidate specs, budget=%d (initial=%d, batch=%d, kappa=%.2f)",
        len(all_specs),
        config.bo_total_budget,
        config.bo_initial_random,
        config.bo_batch_size,
        config.bo_ucb_kappa,
    )

    # Pre-generate placements + descriptors for feature extraction
    spec_descriptors: list[PlacementDescriptor | None] = []
    for spec in all_specs:
        gen_result = generate_placement_from_spec(
            spec, conformers, slab.atoms, config, smiles=smiles
        )
        if gen_result is not None:
            _, descriptor = gen_result
            spec_descriptors.append(descriptor)
        else:
            spec_descriptors.append(None)

    valid_pool_indices = [i for i, d in enumerate(spec_descriptors) if d is not None]
    if not valid_pool_indices:
        logger.warning("No valid placements in candidate pool")
        return None

    valid_descriptors = [spec_descriptors[i] for i in valid_pool_indices]
    candidate_features = build_candidate_features(
        valid_descriptors,
        molecule=molecule_name,
        smiles=smiles,
        surface_id=surface_type,
        config=config,
    )

    # Map pool_position -> valid_pool_index for bookkeeping
    pool_idx_to_spec = {
        pos: valid_pool_indices[pos] for pos in range(len(valid_pool_indices))
    }

    # BO loop state
    evaluated_pool_positions: set[int] = set()
    all_results: list[ScreeningResult] = []
    observed_X_rows: list[dict[str, float]] = []
    observed_y: list[float] = []
    total_evaluated = 0
    best_energy = float("inf")
    rng = np.random.RandomState(config.seed)

    # -- Initial random batch --
    n_initial = min(
        config.bo_initial_random, len(valid_pool_indices), config.bo_total_budget
    )
    initial_positions = rng.choice(
        len(valid_pool_indices), size=n_initial, replace=False
    ).tolist()

    def _run_batch(pool_positions: list[int]) -> None:
        nonlocal total_evaluated, best_energy
        batch_specs = [all_specs[pool_idx_to_spec[p]] for p in pool_positions]
        evaluated_pool_positions.update(pool_positions)

        batch_results = _evaluate_placement_batch(
            batch_specs,
            conformers,
            slab,
            calculator,
            ts_model,
            config,
            smiles,
            E_slab,
            E_mol,
            surface_type,
            base_slab_for_frozen=base_slab_for_frozen,
        )

        total_evaluated += len(pool_positions)
        all_results.extend(batch_results)

        # Collect training data from successful evaluations
        for r in batch_results:
            from .ml.bayesian import _record_from_descriptor

            record = _record_from_descriptor(
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

        if batch_results:
            batch_best = min(r.energy_adsorption for r in batch_results)
            logger.info(
                "BO batch: %d evaluated, %d valid, batch_best=%.4f, overall_best=%.4f",
                len(pool_positions),
                len(batch_results),
                batch_best,
                best_energy,
            )
        else:
            logger.info(
                "BO batch: %d evaluated, 0 valid results",
                len(pool_positions),
            )

    _run_batch(initial_positions)

    # -- Iterative BO batches --
    while total_evaluated < config.bo_total_budget:
        remaining_budget = config.bo_total_budget - total_evaluated
        if remaining_budget <= 0:
            break

        if len(observed_X_rows) < 3:
            # Not enough training data; fill with more random
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
            X_train = pd.DataFrame(observed_X_rows)
            y_train = np.array(observed_y)
            surrogate = train_surrogate(
                X_train,
                y_train,
                surrogate=config.bo_surrogate,
                n_estimators=100,
                random_state=config.seed,
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

    if not all_results:
        return None

    # Apply standard post-processing filters
    surface_symbols = _infer_surface_symbols(slab.atoms)
    # Fix molecule name in results (was set to smiles in _evaluate_placement_batch)
    for r in all_results:
        r.molecule = molecule_name
    filtered = filter_results(
        all_results,
        slab=slab.atoms,
        surface_symbols=surface_symbols,
        reference_smiles=reference_smiles,
        config=config,
    )

    if not filtered:
        return None

    logger.info(
        "BO filtered: %d -> %d results, E_ads range [%.4f, %.4f]",
        len(all_results),
        len(filtered),
        min(r.energy_adsorption for r in filtered),
        max(r.energy_adsorption for r in filtered),
    )

    return filtered


def _infer_surface_symbols(slab: Atoms) -> list[str]:
    """Return unique element symbols present in *slab*."""
    return sorted(set(slab.get_chemical_symbols()))


def _setup_screening_run(
    slab: SlabContainer,
    smiles_file: str,
    config: AdsorptionConfig,
    surface_type: str,
    skip_existing: bool,
    skip_saturation_file: bool = False,
) -> tuple[object, object, list[str], list[str], ReferenceEnergies, float] | None:
    """Setup model, load molecules, compute reference energies.

    Returns (calculator, ts_model, molecules, smiles_list, ref, t_ref_s) or None
    when no molecules to process.
    """
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    molecules, smiles_list, load_status = load_molecules(
        smiles_file,
        skip_existing=skip_existing,
        surface_type=surface_type,
        skip_saturation_file=skip_saturation_file,
    )
    if not molecules:
        if load_status == "all_skipped":
            logger.info("No molecules to process (all already in existing summary)")
        elif load_status == "empty_file":
            logger.info("No molecules to process (file empty or no valid rows)")
        else:
            logger.info("No molecules to process")
        return None
    t_ref_start = time.perf_counter()
    ref = calculate_reference_energies(
        slab, calculator, molecules, smiles_list, ts_model, config=config
    )
    t_ref_s = time.perf_counter() - t_ref_start
    return (calculator, ts_model, molecules, smiles_list, ref, t_ref_s)


def run_screening(
    slab: SlabContainer,
    smiles_file: str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[ScreeningRunResult]:
    """Full screening loop: load SMILES, compute references, process each molecule.

    Returns a list of :class:`ScreeningRunResult` objects (one per molecule).
    Persistence (CSV / XYZ / VASP) is **not** performed here; use the
    helpers in :mod:`metalsurfer.io_results` to save outputs.
    """
    if config is None:
        config = AdsorptionConfig()

    t_run_start = time.perf_counter()

    with (
        screening_log_context(),
        log_context(surface_type=surface_type, seed=config.seed),
    ):
        setup = _setup_screening_run(
            slab, smiles_file, config, surface_type, skip_existing
        )
        if setup is None:
            return []

        calculator, ts_model, molecules, smiles_list, ref, t_ref_s = setup
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)

        all_run_results: list[ScreeningRunResult] = []
        for smi, mol in zip(smiles_list, molecules, strict=False):
            mol_results = process_molecule(
                smi,
                mol,
                slab,
                calculator,
                ref,
                ts_model=ts_model,
                config=config,
                surface_type=surface_type,
                reference_smiles=smi,
            )
            if mol_results:
                summary = build_molecule_summary(mol, mol_results)
                run_result = ScreeningRunResult(
                    molecule=mol,
                    results=mol_results,
                    summary=summary,
                )
                all_run_results.append(run_result)
                ds_logger.add_results(mol_results, smiles=smi, surface_id=surface_type)

        ds_logger.flush()

    t_run_total = time.perf_counter() - t_run_start
    total_configs = sum(len(r.results) for r in all_run_results)
    logger.info(
        "Screening complete: %d molecules, %d configs, %.1fs total",
        len(molecules),
        total_configs,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecules),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_run_results


def run_screening_bayesian(
    slab: SlabContainer,
    smiles_file: str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[ScreeningRunResult]:
    """BO-guided screening loop: same interface as :func:`run_screening`.

    Uses :func:`process_molecule_bayesian` instead of :func:`process_molecule`
    to iteratively select placements via a Random Forest surrogate and UCB
    acquisition.  Falls back to :func:`run_screening` when ``config.bo_enabled``
    is ``False``.
    """
    if config is None:
        config = AdsorptionConfig(bo_enabled=True)

    if not config.bo_enabled:
        return run_screening(
            slab,
            smiles_file=smiles_file,
            config=config,
            surface_type=surface_type,
            skip_existing=skip_existing,
            run_metadata_out=run_metadata_out,
        )

    t_run_start = time.perf_counter()

    with (
        screening_log_context(),
        log_context(surface_type=surface_type, seed=config.seed),
    ):
        setup = _setup_screening_run(
            slab, smiles_file, config, surface_type, skip_existing
        )
        if setup is None:
            return []

        calculator, ts_model, molecules, smiles_list, ref, t_ref_s = setup
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)

        all_run_results: list[ScreeningRunResult] = []
        for smi, mol in zip(smiles_list, molecules, strict=False):
            mol_results = process_molecule_bayesian(
                smi,
                mol,
                slab,
                calculator,
                ref,
                ts_model=ts_model,
                config=config,
                surface_type=surface_type,
                reference_smiles=smi,
            )
            if mol_results:
                summary = build_molecule_summary(mol, mol_results)
                run_result = ScreeningRunResult(
                    molecule=mol,
                    results=mol_results,
                    summary=summary,
                )
                all_run_results.append(run_result)
                ds_logger.add_results(mol_results, smiles=smi, surface_id=surface_type)

        ds_logger.flush()

    t_run_total = time.perf_counter() - t_run_start
    total_configs = sum(len(r.results) for r in all_run_results)
    logger.info(
        "BO screening complete: %d molecules, %d configs, %.1fs total",
        len(molecules),
        total_configs,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecules),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_run_results


def run_saturation_screening(
    slab: SlabContainer,
    smiles_file: str = "smiles.csv",
    config: AdsorptionConfig | None = None,
    surface_type: str = "manual",
    skip_existing: bool = True,
    failure_summary_out: dict[str, object] | None = None,
    run_metadata_out: dict[str, Any] | None = None,
) -> list[SaturationRunResult]:
    """Sequential saturation: add molecules until best E_ads >= 0 (slab saturated).

    For each molecule in the SMILES file, runs an independent saturation loop:
    start from clean slab, add one molecule at a time using the lowest-energy
    adsorbate+slab as the next starting point, until no negative adsorption
    energy is found.

    Returns a list of :class:`SaturationRunResult` objects (one per molecule).
    """
    if config is None:
        config = AdsorptionConfig()

    t_run_start = time.perf_counter()

    with (
        screening_log_context(),
        log_context(surface_type=surface_type, seed=config.seed),
    ):
        setup = _setup_screening_run(
            slab,
            smiles_file,
            config,
            surface_type,
            skip_existing,
            skip_saturation_file=skip_existing,
        )
        if setup is None:
            return []

        calculator, ts_model, molecules, smiles_list, ref, t_ref_s = setup
        base_slab = slab.atoms.copy()
        base_slab.set_pbc([True, True, True])
        results_dir = f"results_{surface_type}"
        ds_logger = DatasetLogger(results_dir, config=config, surface_id=surface_type)

        all_saturation_results: list[SaturationRunResult] = []
        for smi, mol in zip(smiles_list, molecules, strict=False):
            E_mol = ref.get_molecule_energy(mol)
            if E_mol is None:
                if config.fail_on_missing_reference:
                    raise ValueError(
                        f"No reference energy for {mol}; cannot continue with "
                        "fail_on_missing_reference=True"
                    )
                logger.warning("Skipping %s: no reference energy", mol)
                continue

            current_slab = SlabContainer(base_slab.copy())
            steps: list[SaturationStepResult] = []
            step = 0

            while True:
                step += 1
                n_on_slab = 0 if step == 1 else step - 1
                logger.info(
                    "Saturation step %d for %s (n_molecules on slab: %d)",
                    step,
                    mol,
                    n_on_slab,
                )

                slab_copy = current_slab.atoms.copy()
                slab_copy.calc = calculator
                E_slab = float(slab_copy.get_potential_energy())

                ref_step = ReferenceEnergies(
                    slab_energy=E_slab,
                    molecule_energies=ref.molecule_energies,
                )

                mol_results = process_molecule(
                    smi,
                    mol,
                    current_slab,
                    calculator,
                    ref_step,
                    ts_model=ts_model,
                    config=config,
                    surface_type=surface_type,
                    reference_smiles=smi,
                    base_slab_for_frozen=base_slab,
                    slab_energy_override=E_slab,
                    failure_summary_out=failure_summary_out,
                )

                if not mol_results:
                    logger.warning(
                        "Step %d: no valid placements for %s; stopping saturation",
                        step,
                        mol,
                    )
                    break

                best = min(mol_results, key=lambda r: r.energy_adsorption)
                steps.append(
                    SaturationStepResult(
                        step=step,
                        molecule=mol,
                        n_molecules_on_slab=n_on_slab,
                        best_result=best,
                        all_results=mol_results,
                    )
                )
                ds_logger.add_results(mol_results, smiles=smi, surface_id=surface_type)

                logger.info(
                    "Step %d: best E_ads = %.4f eV (placement %d)",
                    step,
                    best.energy_adsorption,
                    best.placement_id,
                )

                if best.energy_adsorption >= 0:
                    logger.info(
                        "Slab saturated for %s at step %d (E_ads >= 0)",
                        mol,
                        step,
                    )
                    break

                current_slab = SlabContainer(best.atoms.copy())

            if steps:
                last_step = steps[-1]
                n_at_saturation = last_step.n_molecules_on_slab + (
                    1 if last_step.best_result.energy_adsorption < 0 else 0
                )
                final_atoms = (
                    last_step.best_result.atoms.copy()
                    if last_step.best_result.energy_adsorption < 0
                    else None
                )
                all_saturation_results.append(
                    SaturationRunResult(
                        molecule=mol,
                        steps=steps,
                        n_molecules_at_saturation=n_at_saturation,
                        final_slab_atoms=final_atoms,
                    )
                )

        ds_logger.flush()

    t_run_total = time.perf_counter() - t_run_start
    total_steps = sum(len(sr.steps) for sr in all_saturation_results)
    total_configs = sum(
        len(s.all_results) for sr in all_saturation_results for s in sr.steps
    )
    logger.info(
        "Saturation screening complete: %d molecules, %d total steps, %.1fs",
        len(molecules),
        total_steps,
        t_run_total,
    )
    if run_metadata_out is not None:
        run_metadata_out.update(
            n_molecules=len(molecules),
            total_configs=total_configs,
            t_ref_s=t_ref_s,
            t_total_s=t_run_total,
        )
    return all_saturation_results


def load_molecules(
    csv_file: str = "smiles.csv",
    skip_existing: bool = True,
    surface_type: str | None = None,
    skip_saturation_file: bool = False,
) -> tuple[list[str], list[str], str]:
    """Load molecules from a two-column (smiles, name) CSV.

    Returns (molecules, smiles, status) where status is:
    - "ok": molecules loaded (possibly filtered)
    - "all_skipped": file had molecules but all were in existing summary
    - "empty_file": file was empty or had no valid rows
    """
    results_dir = f"results_{surface_type}" if surface_type else "results_manual"
    df = pd.read_csv(csv_file, header=None, names=["smiles", "molecule"])
    df = df.dropna()
    all_molecules = df["molecule"].tolist()
    all_smiles = df["smiles"].tolist()

    if not all_molecules:
        return [], [], "empty_file"

    if skip_existing or skip_saturation_file:
        existing_molecules: set[str] = set()
        if skip_saturation_file:
            summary = f"{results_dir}/saturation_summary.csv"
        else:
            summary = f"{results_dir}/adsorption_energies_detailed.csv"
        if os.path.exists(summary):
            try:
                existing_df = pd.read_csv(summary)
                if "molecule" in existing_df.columns:
                    existing_molecules = set(existing_df["molecule"].values)
            except (
                pd.errors.EmptyDataError,
                pd.errors.ParserError,
            ) as e:
                logger.warning("Could not read existing summary %s: %s", summary, e)

        molecules = []
        smiles = []
        for m, s in zip(all_molecules, all_smiles, strict=False):
            if m not in existing_molecules:
                molecules.append(m)
                smiles.append(s)
        if existing_molecules:
            logger.info(
                "Skipped %d already-processed molecules",
                len(all_molecules) - len(molecules),
            )
        if not molecules:
            return [], [], "all_skipped"
        return molecules, smiles, "ok"

    return all_molecules, all_smiles, "ok"

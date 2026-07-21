"""Shared workflow helpers used across screening modes."""

import csv
import logging
import os
import time
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd
from ase import Atoms
from scipy.spatial.distance import pdist

from .._logging import warn_once
from ..config import AdsorptionConfig, resolved_bo_eval_budget
from ..conformers import create_conformers_from_smiles
from ..exceptions import OptimizationError
from ..io_results import results_dir_for
from ..models import PlacementDescriptor, ReferenceEnergies, ScreeningResult
from ..optimization import (
    check_frozen_substrate_displacement,
    estimate_parallel_relaxation_capacity,
    frozen_indices_from_constraints,
    log_substrate_freeze_policy,
    setup_single_model,
)
from ..placement import generators as placement_generators
from ..placement import sites as placement_sites
from ..placement._material import calculator_pbc_for_atoms, material_aware_pbc
from ..placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec_with_reason,
)
from ..placement.geometry import calculate_min_distance
from ..surfaces import SlabContainer, accept_substrate_for_api, validate_substrate

logger = logging.getLogger(__name__)
MIN_CALCULATOR_CELL_C_ANG = 18.0


def clear_site_context_cache() -> None:
    """Clear cached site contexts (mainly for tests and long-running sessions)."""
    placement_generators.clear_site_caches()


@dataclass
class PlacementFailureEvent:
    """Structured explanation for a failed placement candidate."""

    placement_id: int
    stage: str
    reason: str
    descriptor: PlacementDescriptor | None = None


def _summarize_failure_events(
    events: list[PlacementFailureEvent],
    *,
    label: str,
) -> dict[str, int]:
    """Log compact failure summaries; return stage:reason → count."""
    stage_reason_counts: dict[str, int] = {}
    if not events:
        return stage_reason_counts
    for event in events:
        key = f"{event.stage}:{event.reason}"
        stage_reason_counts[key] = stage_reason_counts.get(key, 0) + 1
        logger.debug(
            "%s failure pid=%d stage=%s reason=%s",
            label,
            event.placement_id,
            event.stage,
            event.reason,
        )
    summary = ", ".join(
        f"{k}={n}"
        for k, n in sorted(
            stage_reason_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    logger.warning("%s failures (%d): %s", label, len(events), summary)
    return stage_reason_counts


def _generation_failure_histogram(
    events: list[PlacementFailureEvent],
) -> dict[str, int]:
    """Count generation-stage failures by reason token."""
    counts: dict[str, int] = {}
    for event in events:
        if event.stage != "generation":
            continue
        counts[event.reason] = counts.get(event.reason, 0) + 1
    return counts


def _prepare_atoms_for_calculator(
    atoms: Atoms,
    *,
    label: str,
    min_cell_c: float = MIN_CALCULATOR_CELL_C_ANG,
) -> None:
    """Normalize PBC for UMA and enforce minimum c-vector size."""
    atoms.set_pbc(calculator_pbc_for_atoms(atoms))
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    if bool(pbc.all()):
        c_len = float(np.linalg.norm(np.asarray(atoms.get_cell())[2]))
        if c_len < min_cell_c:
            raise OptimizationError(
                f"{label}: periodic z-cell too small ({c_len:.3f} A). "
                f"Increase vacuum to at least {min_cell_c:.1f} A "
                "to avoid self-interaction between periodic images."
            )


def _compute_slab_energy(
    slab_atoms: Atoms,
    calculator,
    *,
    label: str,
) -> float:
    """Compute slab reference energy with consistent calculator prep."""
    slab_copy = slab_atoms.copy()
    _prepare_atoms_for_calculator(slab_copy, label=label)
    slab_copy.calc = calculator
    return float(slab_copy.get_potential_energy())


def _materialize_spec_placements(
    *,
    specs: list,
    conformers: list[Atoms],
    slab_atoms: Atoms,
    calculator,
    config: AdsorptionConfig,
    smiles: str,
    site_context: placement_generators.SiteContext | None,
    slab_for_sites: Atoms | None = None,
    materialization_cache: dict[int, tuple[Atoms, PlacementDescriptor]] | None = None,
) -> tuple[
    list[Atoms], list[int], list[PlacementDescriptor], list[PlacementFailureEvent]
]:
    """Build combined slab+adsorbate structures and track generation failures."""
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []

    for spec in specs:
        cached = (
            materialization_cache.get(int(spec.placement_index))
            if materialization_cache is not None
            else None
        )
        result: tuple[Atoms, PlacementDescriptor] | None
        fail_reason: str | None
        if cached is not None:
            adsorbate, descriptor = cached
            result = (adsorbate.copy(), descriptor)
            fail_reason = None
        else:
            result, fail_reason = generate_placement_from_spec_with_reason(
                spec,
                conformers,
                slab_atoms,
                config,
                smiles=smiles,
                site_context=site_context,
                slab_for_sites=slab_for_sites,
            )
        if result is None:
            failures.append(
                PlacementFailureEvent(
                    placement_id=spec.placement_index,
                    stage="generation",
                    reason=fail_reason or "unknown_generation_failure",
                    descriptor=None,
                )
            )
            continue
        adsorbate, descriptor = result
        combined = slab_atoms + adsorbate
        combined.set_pbc(calculator_pbc_for_atoms(combined))
        combined.calc = calculator
        all_combined.append(combined)
        placement_ids.append(descriptor.placement_index)
        placement_descriptors.append(descriptor)

    return all_combined, placement_ids, placement_descriptors, failures


def _validate_geometry(
    atoms: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
) -> tuple[bool, str]:
    """Quick sanity checks on the optimised structure."""
    energy = atoms.get_potential_energy()
    if not np.isfinite(energy):
        return False, f"non-finite energy: {energy}"

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
    surface_symbols: list[str] | None = None,
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

    slab_positions = slab.get_positions()
    if surface_symbols:
        slab_syms = np.array(slab.get_chemical_symbols())
        mask = np.isin(slab_syms, surface_symbols)
        if np.any(mask):
            slab_positions = slab_positions[mask]

    cell = np.asarray(atoms.get_cell())
    min_d = calculate_min_distance(
        adsorbate.get_positions(),
        slab_positions,
        cell,
        use_pbc=True,
        pbc=material_aware_pbc(config.material_type),
    )
    if min_d > config.binding_distance_threshold:
        return False, f"desorbed ({min_d:.2f} A)"
    return True, f"adsorbed ({min_d:.2f} A)"


def _surface_positions_for_distance(
    slab_atoms: Atoms,
    surface_symbols: list[str] | None,
) -> np.ndarray:
    slab_positions = slab_atoms.get_positions()
    if not surface_symbols:
        return slab_positions
    slab_syms = np.array(slab_atoms.get_chemical_symbols())
    mask = np.isin(slab_syms, surface_symbols)
    if np.any(mask):
        return slab_positions[mask]
    return slab_positions


def _evaluate_optimized_candidate(
    *,
    opt_atoms: Atoms | None,
    placement_id: int,
    descriptor: PlacementDescriptor,
    molecule_name: str,
    slab_atoms: Atoms,
    calculator,
    config: AdsorptionConfig,
    E_slab: float,
    E_mol: float,
    surface_symbols: list[str] | None,
    log_prefix: str = "",
) -> tuple[ScreeningResult | None, PlacementFailureEvent | None]:
    if opt_atoms is None:
        return None, PlacementFailureEvent(
            placement_id=placement_id,
            stage="optimization",
            reason="optimizer_returned_none",
            descriptor=descriptor,
        )
    if opt_atoms.calc is None:
        opt_atoms.calc = calculator

    ok, reason = check_frozen_substrate_displacement(
        opt_atoms,
        slab_atoms,
        slab_size=len(slab_atoms),
    )
    if not ok:
        logger.warning("%sfrozen substrate drift: %s", log_prefix, reason)
        return None, PlacementFailureEvent(
            placement_id=placement_id,
            stage="validation",
            reason=reason,
            descriptor=descriptor,
        )

    ok, reason = _validate_geometry(opt_atoms, slab_atoms, config)
    if not ok:
        logger.debug("%sgeometry fail: %s", log_prefix, reason)
        return None, PlacementFailureEvent(
            placement_id=placement_id,
            stage="validation",
            reason=reason,
            descriptor=descriptor,
        )

    ok, reason = _validate_adsorption(
        opt_atoms,
        slab_atoms,
        config,
        surface_symbols=surface_symbols,
    )
    if not ok:
        logger.debug("%sadsorption fail: %s", log_prefix, reason)
        return None, PlacementFailureEvent(
            placement_id=placement_id,
            stage="validation",
            reason=reason,
            descriptor=descriptor,
        )

    e_adslab = opt_atoms.get_potential_energy()
    e_ads = e_adslab - E_slab - E_mol
    if e_ads > config.max_adsorption_energy:
        reason = f"E_ads too high: {e_ads:.4f} eV"
        logger.debug("%sunrealistic E_ads: %.4f eV", log_prefix, e_ads)
        return None, PlacementFailureEvent(
            placement_id=placement_id,
            stage="energy_cap",
            reason=reason,
            descriptor=descriptor,
        )

    slab_size = len(slab_atoms)
    mol_atoms = opt_atoms[slab_size:]
    slab_opt = opt_atoms[:slab_size]
    dist = calculate_min_distance(
        mol_atoms.get_positions(),
        _surface_positions_for_distance(slab_opt, surface_symbols),
        np.asarray(opt_atoms.get_cell()),
        use_pbc=True,
        pbc=material_aware_pbc(config.material_type),
    )
    result = ScreeningResult(
        molecule=molecule_name,
        placement_id=placement_id,
        energy_adslab=e_adslab,
        energy_slab=E_slab,
        energy_adsorbate=E_mol,
        energy_adsorption=e_ads,
        atoms=opt_atoms.copy(),
        slab_size=slab_size,
        distance=dist,
        placement_descriptor=descriptor,
    )
    return result, None


@dataclass
class SubstrateRefState:
    """Resolved substrate references for placement and freeze policy."""

    slab: SlabContainer
    slab_for_sites: Atoms
    effective_base_slab_for_frozen: Atoms | None


def prepare_substrate_for_screening(
    slab: SlabContainer,
    conformers: list[Atoms],
    base_slab_for_frozen: Atoms | None,
    config: AdsorptionConfig,
) -> SubstrateRefState:
    """Resolve placement and freeze references without modifying the substrate."""
    validate_substrate(
        slab.atoms,
        material_type=config.material_type,
        config=config,
        conformers=conformers,
        require_bottom_anchor=False,
    )
    slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab_for_frozen)
    effective_base_slab_for_frozen = (
        base_slab_for_frozen.copy() if base_slab_for_frozen is not None else None
    )
    freeze_ref = (
        effective_base_slab_for_frozen
        if effective_base_slab_for_frozen is not None
        else slab.atoms
    )
    log_substrate_freeze_policy(freeze_ref)
    return SubstrateRefState(
        slab=slab,
        slab_for_sites=slab_for_sites,
        effective_base_slab_for_frozen=effective_base_slab_for_frozen,
    )


def build_representative_relaxation_atoms(
    conformers: list[Atoms],
    slab_atoms: Atoms,
    slab_for_sites: Atoms,
    config: AdsorptionConfig,
    smiles: str,
    *,
    site_context: placement_generators.SiteContext | None,
) -> Atoms:
    """Build one slab+adsorbate geometry for GPU parallel-capacity probing."""
    if not conformers:
        raise ValueError("conformers must not be empty")

    specs = enumerate_placement_specs(
        conformers,
        slab_for_sites,
        config,
        smiles,
        1,
        site_context=site_context,
        full_slab=slab_atoms,
    )
    if specs:
        result, _ = generate_placement_from_spec_with_reason(
            specs[0],
            conformers,
            slab_atoms,
            config,
            smiles=smiles,
            site_context=site_context,
            slab_for_sites=slab_for_sites,
        )
        if result is not None:
            adsorbate, _ = result
            combined = slab_atoms + adsorbate
            combined.set_pbc(calculator_pbc_for_atoms(combined))
            return combined

    largest = max(conformers, key=len)
    positions = largest.get_positions().copy()
    z_min = float(np.min(positions[:, 2]))
    slab_z_max = float(np.max(slab_atoms.get_positions()[:, 2]))
    z_lo, _ = placement_sites._compute_site_z_base(
        config,
        slab_atoms,
        None,
        largest.get_chemical_symbols(),
    )
    z_offset = slab_z_max + z_lo - z_min
    positions[:, 2] += z_offset
    ads = largest.copy()
    ads.set_positions(positions)
    combined = slab_atoms + ads
    combined.set_pbc(calculator_pbc_for_atoms(combined))
    return combined


def resolve_workload_config(
    config: AdsorptionConfig,
    *,
    ts_model,
    representative_atoms: Atoms,
    frozen_indices: list[int],
    bo_enabled: bool,
) -> AdsorptionConfig:
    """Fill auto placement/BO batch fields from probed GPU parallel capacity."""
    needs_autotune = config.num_placements is None or (
        bo_enabled
        and (config.bo_initial_random is None or config.bo_batch_size is None)
    )
    if not needs_autotune:
        return config

    capacity = estimate_parallel_relaxation_capacity(
        ts_model,
        representative_atoms,
        config,
        frozen_indices=frozen_indices,
    )

    updates: dict[str, int] = {}
    if config.num_placements is None:
        updates["num_placements"] = capacity
    if bo_enabled:
        if config.bo_initial_random is None:
            updates["bo_initial_random"] = capacity
        if config.bo_batch_size is None:
            updates["bo_batch_size"] = capacity

    resolved = config
    if "num_placements" in updates:
        resolved = replace(resolved, num_placements=updates["num_placements"])
    if "bo_initial_random" in updates:
        resolved = replace(resolved, bo_initial_random=updates["bo_initial_random"])
    if "bo_batch_size" in updates:
        resolved = replace(resolved, bo_batch_size=updates["bo_batch_size"])

    if bo_enabled:
        eval_budget = resolved_bo_eval_budget(resolved)
        logger.info(
            "Autotuned workload: parallel=%d, num_placements=%d, "
            "bo_initial=%d, bo_batch=%d, bo_batches=%d (eval_budget=%d)",
            capacity,
            resolved.num_placements,
            resolved.bo_initial_random,
            resolved.bo_batch_size,
            resolved.bo_total_budget,
            eval_budget,
        )
    else:
        logger.info(
            "Autotuned workload: parallel=%d, num_placements=%d",
            capacity,
            resolved.num_placements,
        )
    return resolved


def resolve_saturation_step_workload_config(
    config: AdsorptionConfig,
    *,
    ts_model,
    conformers: list[Atoms],
    slab_atoms: Atoms,
    slab_for_sites: Atoms,
    smiles: str,
    base_slab_for_frozen: Atoms | None,
    symmetry_broken: bool,
    bo_enabled: bool,
) -> AdsorptionConfig:
    """Resolve placement budget before multi-molecule budget splitting."""
    needs_autotune = config.num_placements is None or (
        bo_enabled
        and (config.bo_initial_random is None or config.bo_batch_size is None)
    )
    if not needs_autotune:
        return config

    site_context = _resolve_site_context_for_sampling(
        slab_for_sites,
        config,
        symmetry_broken=symmetry_broken,
    )
    freeze_ref = (
        base_slab_for_frozen if base_slab_for_frozen is not None else slab_atoms
    )
    frozen_indices = frozen_indices_from_constraints(freeze_ref)
    representative_atoms = build_representative_relaxation_atoms(
        conformers,
        slab_atoms,
        slab_for_sites,
        config,
        smiles,
        site_context=site_context,
    )
    return resolve_workload_config(
        config,
        ts_model=ts_model,
        representative_atoms=representative_atoms,
        frozen_indices=frozen_indices,
        bo_enabled=bo_enabled,
    )


def _build_surface_reference_slab(
    slab_atoms: Atoms,
    base_slab_for_frozen: Atoms | None,
) -> Atoms:
    """Build a substrate-only slab reference for placement/validation/filtering."""
    if base_slab_for_frozen is None:
        return slab_atoms

    surface_symbols = set(base_slab_for_frozen.get_chemical_symbols())
    symbols = slab_atoms.get_chemical_symbols()
    mask = [s in surface_symbols for s in symbols]
    if not any(mask):
        return slab_atoms
    if all(mask):
        return slab_atoms

    surface_slab = slab_atoms[mask].copy()
    surface_slab.set_cell(slab_atoms.get_cell())
    surface_slab.set_pbc(slab_atoms.get_pbc())
    return surface_slab


def _resolve_site_context_for_sampling(
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    symmetry_broken: bool,
) -> placement_generators.SiteContext:
    """Clustered Voronoi sites, then optional spglib orbit reduction unless *symmetry_broken*."""
    return placement_generators.resolve_site_context_for_sampling(
        slab_atoms,
        config,
        symmetry_broken=symmetry_broken,
    )


def _infer_surface_symbols(slab: Atoms) -> list[str]:
    """Return unique element symbols present in *slab*."""
    return sorted(set(slab.get_chemical_symbols()))


@dataclass
class ScreeningRunBootstrap:
    """Shared setup state for binding and saturation campaigns."""

    calculator: Any
    ts_model: Any
    molecule_pairs: list[tuple[str, str]]
    ref: ReferenceEnergies
    t_ref_s: float
    slab: SlabContainer


def _bootstrap_screening_run(
    slab: SlabContainer | Atoms,
    molecule_pairs: list[tuple[str, str]],
    config: AdsorptionConfig,
) -> ScreeningRunBootstrap:
    """Validate substrate, load MLIP, and compute reference energies."""
    from .reference import calculate_reference_energies

    slab_container = accept_substrate_for_api(slab, config=config)
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    molecule_names = [name for _, name in molecule_pairs]
    smiles_list = [smiles for smiles, _ in molecule_pairs]
    t_ref_start = time.perf_counter()
    ref = calculate_reference_energies(
        slab_container,
        calculator,
        molecules=molecule_names,
        smiles_list=smiles_list,
        ts_model=ts_model,
        config=config,
    )
    t_ref_s = time.perf_counter() - t_ref_start
    return ScreeningRunBootstrap(
        calculator=calculator,
        ts_model=ts_model,
        molecule_pairs=molecule_pairs,
        ref=ref,
        t_ref_s=t_ref_s,
        slab=slab_container,
    )


@dataclass
class MoleculeScreeningContext:
    """Prepared substrate and workload state for one molecule screening pass."""

    slab: SlabContainer
    slab_for_sites: Atoms
    effective_base_slab_for_frozen: Atoms | None
    conformers: list[Atoms]
    site_context: placement_generators.SiteContext | None
    config: AdsorptionConfig
    E_slab: float
    E_mol: float
    t_conformers: float


def _prepare_molecule_screening(
    *,
    smiles: str,
    molecule_name: str,
    slab: SlabContainer,
    calculator,
    reference_energies: ReferenceEnergies,
    ts_model,
    config: AdsorptionConfig,
    base_slab_for_frozen: Atoms | None = None,
    slab_energy_override: float | None = None,
    symmetry_broken: bool = False,
    failure_summary_out: dict[str, object] | None = None,
    bo_enabled: bool = False,
) -> MoleculeScreeningContext | None:
    """Shared preamble for standard and BO molecule screening."""
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
    conformers, _conformer_energies = conformer_pack

    substrate_ref = prepare_substrate_for_screening(
        slab,
        conformers,
        base_slab_for_frozen,
        config,
    )
    slab = substrate_ref.slab
    slab_for_sites = substrate_ref.slab_for_sites
    effective_base_slab_for_frozen = substrate_ref.effective_base_slab_for_frozen

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
    frozen_indices = frozen_indices_from_constraints(freeze_ref)
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
        bo_enabled=bo_enabled,
    )
    assert config.num_placements is not None

    return MoleculeScreeningContext(
        slab=slab,
        slab_for_sites=slab_for_sites,
        effective_base_slab_for_frozen=effective_base_slab_for_frozen,
        conformers=conformers,
        site_context=site_context,
        config=config,
        E_slab=E_slab,
        E_mol=E_mol,
        t_conformers=t_conformers,
    )


def _normalize_molecules_input(
    molecules: list[tuple[str, str]] | tuple[str, str] | str,
    *,
    skip_existing: bool,
    surface_type: str,
    skip_saturation_file: bool = False,
) -> tuple[list[tuple[str, str]], str, str]:
    """Normalize campaign *molecules* input to ``(smiles, name)`` pairs.

    Returns ``(pairs, load_status, molecules_source)`` where *molecules_source*
    is the CSV path or ``"<inline-molecules>"`` for run metadata.
    """
    if isinstance(molecules, str):
        molecule_names, smiles_list, load_status = load_molecules(
            molecules,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_saturation_file,
        )
        pairs = list(zip(smiles_list, molecule_names, strict=True))
        return pairs, load_status, molecules

    if not molecules:
        raise ValueError("molecules must be a non-empty list")

    pairs = _normalize_molecule_pairs(molecules)
    if skip_existing:
        molecule_names, smiles_list, load_status = load_molecules_from_pairs(
            pairs,
            skip_existing=skip_existing,
            surface_type=surface_type,
            skip_saturation_file=skip_saturation_file,
        )
        pairs = list(zip(smiles_list, molecule_names, strict=True))
    else:
        load_status = "ok"
    return pairs, load_status, "<inline-molecules>"


def _read_molecules_csv(csv_file: str) -> tuple[list[str], list[str]]:
    """Read a two-column (smiles, name) CSV, skipping a recognized header row."""
    smiles_aliases = {"smiles", "smile", "smi"}
    name_aliases = {"name", "molecule", "mol", "molecules"}

    with open(csv_file, newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        return [], []

    start = 0
    if len(rows[0]) >= 2:
        col0 = rows[0][0].strip().lower()
        col1 = rows[0][1].strip().lower()
        if col0 in smiles_aliases and col1 in name_aliases:
            start = 1

    all_smiles: list[str] = []
    all_molecules: list[str] = []
    for row in rows[start:]:
        if len(row) < 2:
            continue
        smiles = row[0].strip()
        name = row[1].strip()
        if not smiles or not name or smiles.lower() == "nan" or name.lower() == "nan":
            continue
        all_smiles.append(smiles)
        all_molecules.append(name)
    return all_smiles, all_molecules


def load_molecules(
    csv_file: str = "smiles.csv",
    skip_existing: bool = True,
    surface_type: str | None = None,
    skip_saturation_file: bool = False,
) -> tuple[list[str], list[str], str]:
    """Load molecules from a two-column (smiles, name) CSV."""
    results_dir = (
        str(results_dir_for(surface_type)) if surface_type else "results_manual"
    )
    all_smiles, all_molecules = _read_molecules_csv(csv_file)

    return _select_molecules_for_processing(
        all_molecules=all_molecules,
        all_smiles=all_smiles,
        skip_existing=skip_existing,
        skip_saturation_file=skip_saturation_file,
        results_dir=results_dir,
    )


def load_molecules_from_pairs(
    molecule_pairs: list[tuple[str, str]] | tuple[str, str],
    *,
    skip_existing: bool = True,
    surface_type: str | None = None,
    skip_saturation_file: bool = False,
) -> tuple[list[str], list[str], str]:
    """Load molecules from in-memory ``(smiles, name)`` tuples."""
    results_dir = (
        str(results_dir_for(surface_type)) if surface_type else "results_manual"
    )
    pairs = _normalize_molecule_pairs(molecule_pairs)
    all_smiles = [smiles for smiles, _ in pairs]
    all_molecules = [name for _, name in pairs]

    return _select_molecules_for_processing(
        all_molecules=all_molecules,
        all_smiles=all_smiles,
        skip_existing=skip_existing,
        skip_saturation_file=skip_saturation_file,
        results_dir=results_dir,
    )


def _normalize_molecule_pairs(
    molecule_pairs: list[tuple[str, str]] | tuple[str, str],
) -> list[tuple[str, str]]:
    if isinstance(molecule_pairs, tuple):
        if len(molecule_pairs) != 2:
            raise ValueError(
                "molecule tuple input must be a (smiles, molecule_name) pair"
            )
        smiles, molecule_name = molecule_pairs
        if not isinstance(smiles, str) or not isinstance(molecule_name, str):
            raise TypeError(
                "molecule tuple input must be a (smiles: str, molecule_name: str) pair"
            )
        return [(smiles, molecule_name)]

    normalized: list[tuple[str, str]] = []
    for pair in molecule_pairs:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(
                "molecule list input must contain only (smiles, molecule_name) tuples"
            )
        smiles, molecule_name = pair
        if not isinstance(smiles, str) or not isinstance(molecule_name, str):
            raise TypeError(
                "molecule list input must contain only (smiles: str, molecule_name: str) tuples"
            )
        normalized.append((smiles, molecule_name))
    return normalized


def _select_molecules_for_processing(
    *,
    all_molecules: list[str],
    all_smiles: list[str],
    skip_existing: bool,
    skip_saturation_file: bool,
    results_dir: str,
) -> tuple[list[str], list[str], str]:
    if len(all_molecules) != len(all_smiles):
        raise ValueError("molecule names and smiles must have matching lengths")

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
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                logger.warning("Could not read existing summary %s: %s", summary, e)

        molecules = []
        smiles = []
        for m, s in zip(all_molecules, all_smiles, strict=True):
            if m not in existing_molecules:
                molecules.append(m)
                smiles.append(s)
        if existing_molecules:
            skipped = len(all_molecules) - len(molecules)
            logger.warning(
                "Skipped %d already-processed molecule(s) listed in %s. "
                "Set skip_existing=False or remove that CSV to force a fresh run.",
                skipped,
                summary,
            )
        if not molecules:
            return [], [], "all_skipped"
        return molecules, smiles, "ok"

    return all_molecules, all_smiles, "ok"

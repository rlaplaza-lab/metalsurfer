"""Shared workflow helpers used across screening modes."""

import logging
import os
import threading
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from ase import Atoms
from scipy.spatial.distance import pdist

from .._logging import warn_once
from ..config import AdsorptionConfig, resolved_bo_eval_budget
from ..exceptions import OptimizationError
from ..models import PlacementDescriptor, ScreeningResult
from ..optimization import compute_frozen_indices, estimate_parallel_relaxation_capacity
from ..placement import generators as placement_generators
from ..placement import get_symmetry_aware_sites
from ..placement._material import material_aware_pbc
from ..placement.generators import (
    enumerate_placement_specs,
    generate_placement_from_spec_with_reason,
)
from ..placement.geometry import calculate_min_distance
from ..surfaces import SlabContainer, auto_resize_slab_for_molecule
from ..symmetry import SymmetryAnalysisError

logger = logging.getLogger(__name__)
MIN_CALCULATOR_CELL_C_ANG = 18.0

# Cache mapping substrate geometry hash -> SiteContext, reused across saturation steps.
_SITE_CONTEXT_CACHE_MAX_ENTRIES = 16
_SITE_CONTEXT_CACHE: dict[int, placement_generators.SiteContext] = {}
_SITE_CONTEXT_CACHE_LOCK = threading.Lock()


def clear_site_context_cache() -> None:
    """Clear cached site contexts (mainly for tests and long-running sessions)."""
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()


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
) -> None:
    """Log compact failure summaries and leave details to debug."""
    if not events:
        return
    stage_reason_counts: dict[str, int] = {}
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


def _prepare_atoms_for_calculator(
    atoms: Atoms,
    *,
    label: str,
    min_cell_c: float = MIN_CALCULATOR_CELL_C_ANG,
) -> None:
    """Normalize PBC for UMA and enforce minimum c-vector size."""
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    if pbc.shape != (3,):
        raise OptimizationError(
            f"{label}: invalid PBC shape {pbc.shape}; expected 3 components."
        )
    if not (bool(pbc.all()) or bool((~pbc).all())):
        logger.warning(
            "%s: mixed PBC %s normalized to [True, True, True] for calculator",
            label,
            pbc.tolist(),
        )
        atoms.set_pbc([True, True, True])
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
) -> tuple[
    list[Atoms], list[int], list[PlacementDescriptor], list[PlacementFailureEvent]
]:
    """Build combined slab+adsorbate structures and track generation failures."""
    all_combined: list[Atoms] = []
    placement_ids: list[int] = []
    placement_descriptors: list[PlacementDescriptor] = []
    failures: list[PlacementFailureEvent] = []

    for spec in specs:
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
        combined.set_pbc([True, True, True])
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


def _validate_initial_placement_geometry(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    surface_symbols: list[str] | None = None,
) -> tuple[bool, str]:
    """Pre-optimization check for surface contact; returns (ok, reason)."""
    if not config.strict_initial_placement and not config.require_multiple_contact:
        return True, "strict placement checks disabled"

    # Avoid circular import
    from ..placement.geometry import calculate_contact_quality

    slab_size = len(slab)
    if len(adsorbate) < 1:
        return False, "empty adsorbate"

    # First argument is the adsorbate (molecule) only; *slab* is the substrate.
    metrics = calculate_contact_quality(
        adsorbate,
        slab,
        contact_distance_threshold=config.contact_distance_threshold,
        exclude_slab_atoms=slab_size if surface_symbols is None else None,
        material_type=config.material_type,
    )

    contact_dist = metrics["contact_distance"]
    num_contacting = metrics["num_contacting_atoms"]

    # Check minimum contact distance
    if contact_dist > config.min_contact_distance:
        return (
            False,
            f"contact distance too large: {contact_dist:.3f} A (max {config.min_contact_distance:.3f})",
        )

    # Check minimum number of contacting atoms
    if num_contacting < config.min_contact_atoms:
        return (
            False,
            f"insufficient contacting atoms: {num_contacting} < {config.min_contact_atoms}",
        )

    # If multiple contact required, check for clustering
    if config.require_multiple_contact and num_contacting > 1:
        contact_atom_var = metrics["contact_atom_variance"]
        if contact_atom_var > 0.5:  # High variance indicates spread-out contacts
            return (
                False,
                f"poor contact clustering: variance {contact_atom_var:.3f} too high",
            )

    return (
        True,
        f"placement geometry valid (contacts={num_contacting}, distance={contact_dist:.3f}A)",
    )


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
    """Substrate geometry and freeze reference after optional auto-resize."""

    slab: SlabContainer
    slab_for_sites: Atoms
    effective_base_slab_for_frozen: Atoms | None
    slab_was_resized: bool
    substrate_atoms_after_resize: Atoms | None


def _resolve_base_slab_for_frozen(
    slab_atoms: Atoms,
    base_slab_for_frozen: Atoms | None,
    *,
    slab_was_resized: bool = False,
    substrate_atoms_after_resize: Atoms | None = None,
) -> Atoms | None:
    """Choose the substrate reference passed to ``optimize_adsorbate_slab_batched``.

    When the substrate was auto-resized in-plane, use the full current substrate
    so every repeated tile is included in top-layer detection or full freeze.
    Otherwise keep ``base_slab_for_frozen`` so saturation can freeze only the
    original substrate block while prior adsorbates relax.
    """
    if base_slab_for_frozen is None:
        return None
    if slab_was_resized:
        if substrate_atoms_after_resize is not None:
            return substrate_atoms_after_resize
        return slab_atoms.copy()
    return base_slab_for_frozen


def write_substrate_step_metadata(
    step_metadata_out: dict[str, object] | None,
    *,
    slab_was_resized: bool,
    substrate_atoms_after_resize: Atoms | None,
) -> None:
    """Record auto-resize outcome for saturation ``base_slab`` persistence."""
    if step_metadata_out is None:
        return
    step_metadata_out["slab_was_resized"] = slab_was_resized
    if substrate_atoms_after_resize is not None:
        step_metadata_out["substrate_atoms_after_resize"] = substrate_atoms_after_resize


def prepare_substrate_for_screening(
    slab: SlabContainer,
    conformers: list[Atoms],
    base_slab_for_frozen: Atoms | None,
    config: AdsorptionConfig,
    *,
    allow_auto_resize: bool,
) -> SubstrateRefState:
    """Optionally auto-resize the slab and resolve placement/freeze references."""
    slab_for_sites = _build_surface_reference_slab(slab.atoms, base_slab_for_frozen)

    slab_was_resized = False
    substrate_atoms_after_resize: Atoms | None = None
    if config.auto_resize_slab and allow_auto_resize:
        slab, was_resized = auto_resize_slab_for_molecule(
            slab,
            conformers,
            config.min_pbc_image_separation,
            material_type=config.material_type,
        )
        if was_resized:
            slab_was_resized = True
            substrate_atoms_after_resize = slab.atoms.copy()
            slab_for_sites = _build_surface_reference_slab(
                slab.atoms, base_slab_for_frozen
            )

    effective_base_slab_for_frozen = _resolve_base_slab_for_frozen(
        slab.atoms,
        base_slab_for_frozen,
        slab_was_resized=slab_was_resized,
        substrate_atoms_after_resize=substrate_atoms_after_resize,
    )

    return SubstrateRefState(
        slab=slab,
        slab_for_sites=slab_for_sites,
        effective_base_slab_for_frozen=effective_base_slab_for_frozen,
        slab_was_resized=slab_was_resized,
        substrate_atoms_after_resize=substrate_atoms_after_resize,
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
            combined.set_pbc([True, True, True])
            return combined

    largest = max(conformers, key=len)
    positions = largest.get_positions().copy()
    z_min = float(np.min(positions[:, 2]))
    slab_z_max = float(np.max(slab_atoms.get_positions()[:, 2]))
    z_offset = slab_z_max + config.placement_z_range[0] - z_min
    positions[:, 2] += z_offset
    ads = largest.copy()
    ads.set_positions(positions)
    combined = slab_atoms + ads
    combined.set_pbc([True, True, True])
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
    frozen_indices = compute_frozen_indices(freeze_ref, config)
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
    pos_bytes = slab_atoms.get_positions().tobytes()
    cell_bytes = np.asarray(slab_atoms.get_cell()).tobytes()
    pbc_bytes = str(list(slab_atoms.get_pbc())).encode()
    cache_key = hash(pos_bytes + cell_bytes + pbc_bytes + str(symmetry_broken).encode())

    with _SITE_CONTEXT_CACHE_LOCK:
        cached = _SITE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Core unified site detection; always performed
    _core_ctx = placement_generators._get_unique_sites_for_specs(slab_atoms, config)
    core_sites = _core_ctx.sites
    use_sites = _core_ctx.use_sites
    raw_unclustered = _core_ctx.raw_unclustered

    def _site_context(
        sites: list[dict[str, object]],
        source: str,
    ) -> placement_generators.SiteContext:
        return placement_generators.SiteContext(
            sites=sites,
            use_sites=True,
            source=source,
            raw_unclustered=raw_unclustered,
        )

    if not use_sites or not core_sites:
        result = _core_ctx
    elif symmetry_broken:
        logger.debug("Site context: symmetry broken, using clustered Voronoi set")
        result = _site_context(core_sites, "voronoi")
    else:
        # Reuse unclustered Voronoi sites from _get_unique_sites_for_specs (no second run).
        try:
            symmetry_aware_sites = get_symmetry_aware_sites(
                slab_atoms,
                top_layer_tolerance=config.top_layer_tolerance,
                symmetry_tolerance=config.symmetry_tolerance,
                material_type=config.material_type,
                probe_radius=config.voronoi_probe_radius,
                max_site_distance=config.voronoi_max_site_distance,
                enrich=config.voronoi_site_enrichment,
                site_classification_method=config.site_classification_method,
                raw_sites=raw_unclustered,
            )
        except SymmetryAnalysisError as exc:
            logger.info(
                "Symmetry site reduction failed; using clustered Voronoi sites (%s)",
                exc,
            )
            symmetry_aware_sites = []

        if symmetry_aware_sites:
            logger.info(
                "Using symmetry-reduced sites (%d sites)", len(symmetry_aware_sites)
            )
            result = _site_context(symmetry_aware_sites, "symmetry_aware")
        else:
            logger.debug("Using clustered Voronoi sites (no symmetry-reduced set)")
            result = _site_context(core_sites, "voronoi")

    with _SITE_CONTEXT_CACHE_LOCK:
        if len(_SITE_CONTEXT_CACHE) >= _SITE_CONTEXT_CACHE_MAX_ENTRIES:
            _SITE_CONTEXT_CACHE.pop(next(iter(_SITE_CONTEXT_CACHE)))
        _SITE_CONTEXT_CACHE[cache_key] = result
    return result


def _infer_surface_symbols(slab: Atoms) -> list[str]:
    """Return unique element symbols present in *slab*."""
    return sorted(set(slab.get_chemical_symbols()))


def load_molecules(
    csv_file: str = "smiles.csv",
    skip_existing: bool = True,
    surface_type: str | None = None,
    skip_saturation_file: bool = False,
) -> tuple[list[str], list[str], str]:
    """Load molecules from a two-column (smiles, name) CSV."""
    results_dir = f"results_{surface_type}" if surface_type else "results_manual"
    df = pd.read_csv(csv_file, header=None, names=["smiles", "molecule"])
    df = df.dropna()
    all_molecules = df["molecule"].tolist()
    all_smiles = df["smiles"].tolist()

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
    results_dir = f"results_{surface_type}" if surface_type else "results_manual"
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
            logger.info(
                "Skipped %d already-processed molecules",
                len(all_molecules) - len(molecules),
            )
        if not molecules:
            return [], [], "all_skipped"
        return molecules, smiles, "ok"

    return all_molecules, all_smiles, "ok"

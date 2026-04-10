"""Shared workflow helpers used across screening modes."""

import logging
import os
import threading
from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from ase import Atoms
from scipy.spatial.distance import pdist

from .._logging import warn_once
from ..config import AdsorptionConfig
from ..exceptions import OptimizationError
from ..models import PlacementDescriptor, ScreeningResult
from ..placement import (
    calculate_min_distance,
    generate_placement_from_spec_with_reason,
    get_symmetry_aware_sites,
    material_aware_pbc,
)
from ..placement import generators as placement_generators

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
        pbc=material_aware_pbc(slab),
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
        pbc=material_aware_pbc(slab_atoms),
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
    """Resolve placement site context using the universal scheme.

    Always uses core unified site detection. Optionally applies symmetry
    reduction on top if available and not broken.
    """
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

    if not use_sites or not core_sites:
        result = placement_generators.SiteContext(
            sites=[], use_sites=False, source="no_sites"
        )
    elif symmetry_broken:
        logger.info("Using core sites; symmetry breaking detected")
        result = placement_generators.SiteContext(
            sites=core_sites, use_sites=True, source="voronoi"
        )
    else:
        logger.debug("Attempting symmetry-aware site reduction")
        symmetry_aware_sites = get_symmetry_aware_sites(
            slab_atoms,
            top_layer_tolerance=config.top_layer_tolerance,
            symmetry_tolerance=config.symmetry_tolerance,
            material_type=config.material_type,
            enrich=config.voronoi_site_enrichment,
        )

        if symmetry_aware_sites:
            logger.info(
                "Using symmetry-reduced sites (%d sites)", len(symmetry_aware_sites)
            )
            result = placement_generators.SiteContext(
                sites=symmetry_aware_sites, use_sites=True, source="symmetry_aware"
            )
        else:
            logger.info("Using core unified sites (no symmetry-reduced set)")
            result = placement_generators.SiteContext(
                sites=core_sites, use_sites=True, source="voronoi"
            )

    with _SITE_CONTEXT_CACHE_LOCK:
        if len(_SITE_CONTEXT_CACHE) >= _SITE_CONTEXT_CACHE_MAX_ENTRIES:
            _SITE_CONTEXT_CACHE.pop(next(iter(_SITE_CONTEXT_CACHE)))
        _SITE_CONTEXT_CACHE[cache_key] = result
    return result


def _infer_surface_symbols(slab: Atoms) -> list[str]:
    """Return unique element symbols present in *slab*."""
    return sorted(set(slab.get_chemical_symbols()))


def format_failure_summary(failure_summary: dict[str, object]) -> str:
    """Produce a human-readable multi-line summary from a failure_summary dict."""
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
        if "n_candidate_specs" in failure_summary:
            lines.append(
                f"  Candidate specs: {failure_summary.get('n_candidate_specs', '?')}"
            )
        if "n_valid_pool" in failure_summary:
            lines.append(f"  Valid pool: {failure_summary.get('n_valid_pool', '?')}")
    elif stage == "validation":
        n_initial = failure_summary.get("n_initial_placements", "?")
        n_opt = failure_summary.get("n_optimized", "?")
        n_opt_fail = failure_summary.get("n_optimization_failed", 0)
        lines.append(f"  Initial placements: {n_initial}")
        lines.append(f"  Optimized: {n_opt} ({n_opt_fail} failed)")
        lines.append("  Passed validation: 0")
        if "n_evaluated" in failure_summary:
            lines.append(f"  BO evaluated: {failure_summary.get('n_evaluated', '?')}")
        if "n_valid_results" in failure_summary:
            lines.append(
                f"  BO valid results: {failure_summary.get('n_valid_results', '?')}"
            )
        vf = cast(dict[str, int], failure_summary.get("validation_failures", {}))
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

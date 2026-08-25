"""Composite-candidate construction and selection for n-tuplet saturation.

With ``AdsorptionConfig.saturation_molecules_per_step > 1`` a saturation step
may commit several placements simultaneously ("n-tuplet"). The per-molecule
screening pools are produced exactly as in legacy mode; this module then

1. greedily selects up to *n* mutually compatible winners
   (:func:`select_tuplet_winners`),
2. materializes ONE candidate containing the substrate prefix followed by all
   selected adsorbates in selection order
   (:func:`build_composite_candidate`),
3. relaxes and validates the composite, mapping the shared result back onto
   one :class:`~metalsurfer.models.ScreeningResult` per committed unit
   (:func:`evaluate_composite_commit`).

INVARIANT (substrate-prefix contract): adsorbate atoms come strictly AFTER the
substrate prefix in every composite. Freeze constraints, desorption checks,
decomposition filters, symmetry analysis, and
:func:`workflow.shared._build_surface_reference_slab` all rely on it.

Energy representation (agreed semantics): every committed unit's row carries
the FULL tuplet totals — ``energy_adslab`` is the relaxed composite energy,
``energy_adsorbate`` is the summed molecular reference of *all* tuplet members,
and ``energy_adsorption`` is the shared tuplet value
``E(composite) - E_slab(step) - sum_i E_mol(molecule_i)`` so the arithmetic
identity ``energy_adslab - energy_slab - energy_adsorbate == energy_adsorption``
holds on every row. Per-unit information survives in ``molecule``,
``placement_id``, ``placement_descriptor``, and the per-unit ``distance``.
"""

import logging
from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
from ase import Atoms

from ..config import AdsorptionConfig
from ..models import ScreeningResult
from ..optimization import optimize_adsorbate_slab_batched
from ..placement._material import calculator_pbc_for_atoms, material_aware_pbc
from ..placement.geometry import calculate_min_distance
from ..placement.occupancy import _positions_mutually_clear
from ..surface_prep.freeze import check_frozen_substrate_displacement
from .shared import _validate_geometry

logger = logging.getLogger(__name__)

__all__ = [
    "build_composite_candidate",
    "evaluate_composite_commit",
    "select_tuplet_winners",
]


def build_composite_candidate(
    slab_atoms: Atoms,
    adsorbates: Sequence[Atoms],
) -> Atoms:
    """Build one candidate: substrate prefix followed by *adsorbates* in order.

    Parameters
    ----------
    slab_atoms
        Current coverage slab (bare substrate prefix + any pre-adsorbed units).
    adsorbates
        Adsorbate-only fragments to append, in tuplet selection order.

    Returns
    -------
    Atoms
        Combined structure with PBC set via
        :func:`placement._material.calculator_pbc_for_atoms`. FixAtoms
        constraints refer to the untouched substrate prefix.
    """
    composite = slab_atoms.copy()
    for adsorbate in adsorbates:
        composite += adsorbate
    composite.set_pbc(calculator_pbc_for_atoms(composite))
    return composite


def select_tuplet_winners(
    candidates: Sequence[ScreeningResult],
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
    max_winners: int,
) -> list[ScreeningResult]:
    """Greedily pick up to *max_winners* mutually clear binders from *candidates*.

    Deterministic: candidates are ordered by ``(energy_adsorption,
    placement_id, molecule)``; each candidate is accepted iff it binds
    (negative E_ads) and is mutually clear from every already-accepted winner
    under MIC. Seed-regression safe because the sort key has full tie-breakers.

    Parameters
    ----------
    candidates
        Screening pool across all molecules of the step (single-adsorbate
        results computed against the current coverage slab).
    cell
        Unit cell matrix of the current slab.
    pbc
        Material-aware periodicity flags.
    min_separation
        Minimum adsorbate-atom to adsorbate-atom distance in Å
        (``AdsorptionConfig.min_adsorbate_separation``).
    max_winners
        Tuplet size cap (``saturation_molecules_per_step``).
    """
    ordered = sorted(
        candidates,
        key=lambda r: (r.energy_adsorption, r.placement_id, r.molecule),
    )
    accepted: list[ScreeningResult] = []
    accepted_suffixes: list[np.ndarray] = []
    for candidate in ordered:
        if len(accepted) >= max_winners:
            break
        # Sorted ascending by E_ads: the first non-binder ends the sweep.
        if candidate.energy_adsorption >= 0:
            break
        suffix = np.asarray(
            candidate.atoms.get_positions()[candidate.slab_size :], dtype=float
        )
        clear = all(
            _positions_mutually_clear(
                suffix,
                other,
                cell=cell,
                pbc=pbc,
                min_separation=min_separation,
            )
            for other in accepted_suffixes
        )
        if clear:
            accepted.append(candidate)
            accepted_suffixes.append(suffix)
    return accepted


def _unit_suffix_bounds(winners: Sequence[ScreeningResult]) -> list[int]:
    """Atom counts of each winner's adsorbate fragment, in selection order."""
    return [len(w.atoms) - w.slab_size for w in winners]


def _per_unit_surface_distances(
    opt_atoms: Atoms,
    *,
    n_substrate: int,
    unit_sizes: Sequence[int],
    config: AdsorptionConfig,
) -> list[float]:
    """Per-unit min adsorbate-to-substrate distance for a relaxed composite."""
    positions = opt_atoms.get_positions()
    substrate_positions = positions[:n_substrate]
    cell = opt_atoms.get_cell()
    pbc = material_aware_pbc(config.material_type)
    distances: list[float] = []
    start = n_substrate
    for size in unit_sizes:
        unit_positions = positions[start : start + size]
        distances.append(
            calculate_min_distance(
                unit_positions,
                substrate_positions,
                cell,
                use_pbc=True,
                pbc=pbc,
            )
        )
        start += size
    return distances


def evaluate_composite_commit(
    *,
    winners: Sequence[ScreeningResult],
    slab_atoms: Atoms,
    base_slab: Atoms,
    ts_model: object,
    config: AdsorptionConfig,
    E_slab: float,
    topology_check: Callable[[Atoms, list[str]], tuple[bool, str]] | None = None,
    log_prefix: str = "",
) -> tuple[list[ScreeningResult], str]:
    """Relax and validate one composite; map results back onto per-unit rows.

    Validation mirrors ``_evaluate_optimized_candidate`` but for an n-tuplet:

    - frozen-substrate drift against the bare-substrate prefix (unchanged
      contract: prefix size is ``len(base_slab)``);
    - geometry sanity via ``_validate_geometry`` (covers adsorbate-adsorbate
      collapses inside the tuplet);
    - PER-UNIT desorption check (a single global-min check would miss one
      desorbed member when its siblings stay bound);
    - optional connectivity-only topology guard over the whole adsorbate pool;
    - ``max_adsorption_energy`` cap applied to the tuplet-total E_ads.

    Parameters
    ----------
    winners
        Mutually clear binder selections from :func:`select_tuplet_winners`.
    slab_atoms
        Current coverage slab the winners were screened against.
    base_slab
        Bare freeze-constrained substrate (prefix reference).
    ts_model
        TorchSim model used by the batched optimizer.
    config
        Adsorption configuration.
    E_slab
        Energy of *slab_atoms* at this step.
    topology_check
        Optional ``(opt_atoms, pending_unit_names) -> (ok, reason)``
        connectivity guard; *pending_unit_names* lists each winner's molecule
        name in selection order so callers can assemble the full reference-unit
        list (units already on the slab plus these pending ones).
    log_prefix
        Log line prefix.

    Returns
    -------
    tuple[list[ScreeningResult], str]
        ``(rewritten_results, "")`` on success, otherwise ``([], reason)``.
        Every rewritten row shares the relaxed composite atoms and the full
        tuplet energies (see module docstring); ``distance`` stays per-unit.
    """
    if not winners:
        return [], "no winners"
    unit_sizes = _unit_suffix_bounds(winners)
    adsorbates = [w.atoms[w.slab_size :] for w in winners]
    composite = build_composite_candidate(slab_atoms, adsorbates)
    optimized = optimize_adsorbate_slab_batched(
        [composite],
        slab_atoms,
        ts_model,
        config=config,
        base_slab_for_frozen=base_slab,
        saturation_reuse=True,
    )
    opt_atoms = optimized[0]
    if opt_atoms is None:
        return [], "optimizer_returned_none"

    ok, reason = check_frozen_substrate_displacement(
        opt_atoms,
        base_slab,
        slab_size=len(base_slab),
    )
    if not ok:
        logger.debug("%scomposite frozen substrate drift: %s", log_prefix, reason)
        return [], f"frozen substrate drift: {reason}"

    ok, reason = _validate_geometry(opt_atoms, slab_atoms, config)
    if not ok:
        logger.debug("%scomposite geometry fail: %s", log_prefix, reason)
        return [], f"geometry fail: {reason}"

    n_substrate = len(slab_atoms)
    unit_distances = _per_unit_surface_distances(
        opt_atoms,
        n_substrate=n_substrate,
        unit_sizes=unit_sizes,
        config=config,
    )
    if not config.skip_desorption_check:
        for k, dist in enumerate(unit_distances):
            if dist > config.binding_distance_threshold:
                logger.debug(
                    "%scomposite unit %d (%s) desorbed: %.2f A",
                    log_prefix,
                    k,
                    winners[k].molecule,
                    dist,
                )
                return [], (f"unit {k} ({winners[k].molecule}) desorbed ({dist:.2f} A)")

    reference_unit_smiles = [w.molecule for w in winners]
    if topology_check is not None:
        ok, reason = topology_check(opt_atoms, reference_unit_smiles)
        if not ok:
            logger.debug(
                "%scomposite topology rearrangement guard: %s", log_prefix, reason
            )
            return [], f"topology rearrangement guard: {reason}"

    e_adslab = float(opt_atoms.get_potential_energy())
    e_mol_sum = float(sum(w.energy_adsorbate for w in winners))
    e_ads_tuplet = e_adslab - E_slab - e_mol_sum
    if e_ads_tuplet > config.max_adsorption_energy:
        return [], f"E_ads too high: {e_ads_tuplet:.4f} eV"

    rewritten = [
        replace(
            winner,
            energy_adslab=e_adslab,
            energy_slab=E_slab,
            energy_adsorbate=e_mol_sum,
            energy_adsorption=e_ads_tuplet,
            atoms=opt_atoms.copy(),
            slab_size=n_substrate,
            distance=float(unit_distances[k]),
        )
        for k, winner in enumerate(winners)
    ]
    logger.info(
        "%scomposite relaxed: %d units, E_ads(tuplet) = %.4f eV",
        log_prefix,
        len(rewritten),
        e_ads_tuplet,
    )
    return rewritten, ""

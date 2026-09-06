"""Composite-candidate construction and selection for n-tuplet saturation.

With ``AdsorptionConfig.saturation_molecules_per_step > 1`` a saturation step
may commit several placements simultaneously ("n-tuplet"). The per-molecule
screening pools are produced exactly as in legacy mode; this module then

1. greedily selects up to *n* mutually compatible winners
   (:func:`select_tuplet_winners`), optionally rescuing near-miss clashes via
   Packmol-style rigid-body descent,
2. sequentially packs units 2..n against frozen earlier winners
   (:func:`pack_tuplet_adsorbates`),
3. materializes ONE candidate containing the substrate prefix followed by all
   selected adsorbates in selection order
   (:func:`build_composite_candidate`),
4. relaxes and validates the composite, mapping the shared result back onto
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
from ..placement._constants import _TUPLET_CLASH_RESCUE_FLOOR
from ..placement._material import calculator_pbc_for_atoms, material_aware_pbc
from ..placement.clash import (
    atom_radii_for_symbols,
    compose_quaternion_with_azimuth,
    resolve_rigid_clash,
)
from ..placement.geometry import (
    _mol_slab_pairwise_distances,
    calculate_min_distance,
    compute_surface_site_frame,
)
from ..placement.occupancy import _positions_mutually_clear
from ..placement.site_coords import _slab_normal
from ..surface_prep.freeze import check_frozen_substrate_displacement
from .shared import _validate_geometry

logger = logging.getLogger(__name__)

__all__ = [
    "build_composite_candidate",
    "evaluate_composite_commit",
    "pack_tuplet_adsorbates",
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


def _suffix_positions(result: ScreeningResult) -> np.ndarray:
    return np.asarray(result.atoms.get_positions()[result.slab_size :], dtype=float)


def _slab_site_frame(slab_atoms: Atoms) -> np.ndarray:
    """Local site frame from slab cell normal (planar default)."""
    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    normal = _slab_normal(cell)
    return compute_surface_site_frame(normal)


def _fixed_cloud_from_coverage_and_results(
    slab_atoms: Atoms,
    results: Sequence[ScreeningResult],
    suffixes: Sequence[np.ndarray],
    *,
    min_separation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coverage slab atoms plus already-accepted adsorbate suffixes."""
    fixed_pos = np.asarray(slab_atoms.get_positions(), dtype=float)
    fixed_radii = atom_radii_for_symbols(
        list(slab_atoms.get_chemical_symbols()),
        min_separation=float(min_separation),
    )
    for other, prev in zip(suffixes, results, strict=True):
        fixed_pos = np.vstack([fixed_pos, other])
        prev_syms = list(prev.atoms.get_chemical_symbols()[prev.slab_size :])
        fixed_radii = np.concatenate(
            [
                fixed_radii,
                atom_radii_for_symbols(
                    prev_syms,
                    min_separation=float(min_separation),
                ),
            ]
        )
    return fixed_pos, fixed_radii


def _min_dist_to_suffixes(
    suffix: np.ndarray,
    others: Sequence[np.ndarray],
    *,
    cell: np.ndarray,
    pbc: list[bool],
) -> float:
    if not others:
        return float("inf")
    mins = [
        float(np.min(_mol_slab_pairwise_distances(suffix, other, cell, pbc)))
        for other in others
        if np.asarray(other).size
    ]
    return min(mins) if mins else float("inf")


def _apply_suffix_to_result(
    result: ScreeningResult,
    new_suffix: np.ndarray,
    *,
    az_delta: float | None = None,
    slab_atoms: Atoms,
) -> ScreeningResult:
    """Return a copy with adsorbate suffix positions (and descriptor COM/quat) updated."""
    atoms = result.atoms.copy()
    pos = atoms.get_positions()
    pos[result.slab_size :] = np.asarray(new_suffix, dtype=float)
    atoms.set_positions(pos)
    com = np.mean(new_suffix, axis=0)
    desc = result.placement_descriptor
    quat_w = desc.quat_w
    quat_x = desc.quat_x
    quat_y = desc.quat_y
    quat_z = desc.quat_z
    if (
        az_delta is not None
        and quat_w is not None
        and quat_x is not None
        and quat_y is not None
        and quat_z is not None
    ):
        normal = _slab_normal(np.asarray(slab_atoms.get_cell(), dtype=float))
        quat_w, quat_x, quat_y, quat_z = compose_quaternion_with_azimuth(
            (quat_w, quat_x, quat_y, quat_z),
            az_delta,
            normal,
        )
    new_desc = replace(
        desc,
        x=float(com[0]),
        y=float(com[1]),
        x_abs=float(com[0]),
        y_abs=float(com[1]),
        z_abs=float(com[2]),
        quat_w=quat_w,
        quat_x=quat_x,
        quat_y=quat_y,
        quat_z=quat_z,
    )
    return replace(result, atoms=atoms, placement_descriptor=new_desc)


def _try_rescue_suffix(
    candidate: ScreeningResult,
    fixed_pos: np.ndarray,
    fixed_radii: np.ndarray,
    *,
    slab_atoms: Atoms,
    cell: np.ndarray,
    pbc: list[bool],
    config: AdsorptionConfig,
) -> ScreeningResult | None:
    """Clash-descend a candidate suffix against *fixed_pos*; None if unsalvageable."""
    suffix = _suffix_positions(candidate)
    ads = candidate.atoms[candidate.slab_size :].copy()
    origin = np.mean(suffix, axis=0)
    frame = _slab_site_frame(slab_atoms)
    new_pos, az_delta, ok = resolve_rigid_clash(
        ads,
        fixed_pos,
        fixed_radii,
        origin=origin,
        site_frame=frame,
        cell=cell,
        pbc=pbc,
        config=config,
        include_substrate_min_sep=True,
    )
    if not ok:
        return None
    return _apply_suffix_to_result(
        candidate, new_pos, az_delta=az_delta, slab_atoms=slab_atoms
    )


def select_tuplet_winners(
    candidates: Sequence[ScreeningResult],
    *,
    cell: np.ndarray,
    pbc: list[bool],
    min_separation: float,
    max_winners: int,
    config: AdsorptionConfig | None = None,
    slab_atoms: Atoms | None = None,
) -> list[ScreeningResult]:
    """Greedily pick up to *max_winners* mutually clear binders from *candidates*.

    Deterministic: candidates are ordered by ``(energy_adsorption,
    placement_id, molecule)``; each candidate is accepted iff it binds
    (negative E_ads) and is mutually clear from every already-accepted winner
    under MIC. When ``config.placement_clash_descent`` is on and *slab_atoms*
    is provided, near-miss clashes (min distance ≥
    ``_TUPLET_CLASH_RESCUE_FLOOR``) are rescued by a bounded rigid-body slide
    instead of being skipped.

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
    config
        Optional config enabling clash-descent near-miss rescue.
    slab_atoms
        Current coverage slab (required for near-miss rescue fixed cloud).
    """
    ordered = sorted(
        candidates,
        key=lambda r: (r.energy_adsorption, r.placement_id, r.molecule),
    )
    accepted: list[ScreeningResult] = []
    accepted_suffixes: list[np.ndarray] = []
    clash_on = (
        config is not None
        and bool(config.placement_clash_descent)
        and slab_atoms is not None
    )
    cell_arr = np.asarray(cell, dtype=float)

    for candidate in ordered:
        if len(accepted) >= max_winners:
            break
        # Sorted ascending by E_ads: the first non-binder ends the sweep.
        if candidate.energy_adsorption >= 0:
            break
        suffix = _suffix_positions(candidate)
        clear = all(
            _positions_mutually_clear(
                suffix,
                other,
                cell=cell_arr,
                pbc=pbc,
                min_separation=min_separation,
            )
            for other in accepted_suffixes
        )
        if clear:
            accepted.append(candidate)
            accepted_suffixes.append(suffix)
            continue

        if not clash_on or not accepted_suffixes:
            continue

        min_d = _min_dist_to_suffixes(suffix, accepted_suffixes, cell=cell_arr, pbc=pbc)
        if min_d < float(_TUPLET_CLASH_RESCUE_FLOOR):
            continue

        assert config is not None and slab_atoms is not None
        fixed_pos, fixed_radii = _fixed_cloud_from_coverage_and_results(
            slab_atoms,
            accepted,
            accepted_suffixes,
            min_separation=float(min_separation),
        )

        rescued = _try_rescue_suffix(
            candidate,
            fixed_pos,
            fixed_radii,
            slab_atoms=slab_atoms,
            cell=cell_arr,
            pbc=pbc,
            config=config,
        )
        if rescued is None:
            continue
        accepted.append(rescued)
        accepted_suffixes.append(_suffix_positions(rescued))
    return accepted


def pack_tuplet_adsorbates(
    winners: Sequence[ScreeningResult],
    slab_atoms: Atoms,
    config: AdsorptionConfig,
) -> list[ScreeningResult]:
    """Sequentially clash-pack units 2..n against frozen unit 1 + coverage.

    Unit 1 stays at its screened/rescued pose. Units that fail descent are
    dropped (partial tuplet). When ``placement_clash_descent`` is False, returns
    *winners* unchanged.

    Parameters
    ----------
    winners
        Mutually clear (or rescued) binders from :func:`select_tuplet_winners`.
    slab_atoms
        Current coverage slab.
    config
        Adsorption configuration.
    """
    if not winners:
        return []
    if not config.placement_clash_descent or len(winners) == 1:
        return list(winners)

    cell = np.asarray(slab_atoms.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    packed: list[ScreeningResult] = [winners[0]]
    packed_suffixes: list[np.ndarray] = [_suffix_positions(winners[0])]

    for winner in winners[1:]:
        fixed_pos, fixed_radii = _fixed_cloud_from_coverage_and_results(
            slab_atoms,
            packed,
            packed_suffixes,
            min_separation=float(config.min_adsorbate_separation),
        )
        rescued = _try_rescue_suffix(
            winner,
            fixed_pos,
            fixed_radii,
            slab_atoms=slab_atoms,
            cell=cell,
            pbc=pbc,
            config=config,
        )
        if rescued is None:
            logger.info(
                "n-tuplet pack: dropping unit %s (placement_id=%s); clash descent failed",
                winner.molecule,
                winner.placement_id,
            )
            continue
        packed.append(rescued)
        packed_suffixes.append(_suffix_positions(rescued))
    return packed


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

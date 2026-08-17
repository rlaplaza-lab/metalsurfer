"""Batched TorchSim relaxations, single-point batching, and capacity probing.

The heavy call paths here (``ts.optimize`` / ``ts.static`` / ``ts.initialize_state``)
require the MLIP stack and a GPU to be meaningful, so they are marked
``# pragma: no cover``; the surrounding pure logic lives in :mod:`._validation`
and :mod:`._cache` and is unit-tested on CPU.
"""

import logging

import numpy as np
from ase import Atoms

from .._logging import torchsim_output_capture
from .._numeric_defaults import DEFAULT_FMAX
from ..config import AdsorptionConfig
from ..exceptions import DependencyMissingError
from ..surface_prep.freeze import frozen_indices_from_constraints
from . import _deps
from ._cache import (
    _get_inflight_autobatcher,
    _maybe_clear_cuda_cache,
    capacity_cache_get,
    capacity_cache_set,
    clear_autobatcher_cache,
    pop_autobatcher,
)
from ._model import TorchSimCalculator, setup_torchsim_model
from ._validation import (
    _DYNAMIC_AUTOBATCHER_CAP_BUCKET,
    _DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER,
    _is_cuda_oom_error,
    _parallel_capacity_cache_key,
    _positions_cell_hash,
    _resolve_autobatcher_max_atoms_to_try,
    _resolve_device,
    _resolve_ts_optimizer,
    _validate_model_pbc,
)

logger = logging.getLogger(__name__)


def setup_single_model(  # pragma: no cover - requires MLIP stack / GPU
    model_name: str = "uma-s-1p2",
    device: str = "cuda",
):
    """Create a single FairChemModel shared by calculator and TorchSim.

    Returns (calculator, ts_model) where calculator wraps ts_model.
    Prefer this over separate calculator and TorchSim model setup to reduce GPU memory.

    Parameters
    ----------
    model_name
        FairChem model name.
    device
        Device string (e.g. "cuda" or "cpu").
    """
    ts_model = setup_torchsim_model(model_name, device)
    calculator = TorchSimCalculator(ts_model)
    return calculator, ts_model


def _make_state_with_frozen_constraint(  # pragma: no cover - requires MLIP stack / GPU
    atoms: Atoms,
    frozen_indices: list[int],
    ts_model,
    device,
):
    """Build a TorchSim state for one system with FixAtoms constraint applied."""
    torch = _deps.torch
    state = _deps.ts.initialize_state(atoms, device=device, dtype=ts_model.dtype)
    target_dev = state.positions.device
    idx_tensor = torch.tensor(frozen_indices, dtype=torch.long, device=target_dev)
    state.constraints = [_deps.ts_constraints.FixAtoms(atom_idx=idx_tensor)]
    return state


def estimate_parallel_relaxation_capacity(
    ts_model,
    representative_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    frozen_indices: list[int],
) -> int:
    """Estimate how many slab+adsorbate relaxations can run in parallel on GPU.

    Mirrors TorchSim ``InFlightAutoBatcher`` memory probing. Returns at least 1.

    Parameters
    ----------
    ts_model
        TorchSim model instance.
    representative_atoms
        Representative ASE Atoms for the systems to relax.
    config
        Adsorption configuration.
    frozen_indices
        Atom indices that are frozen during relaxation.
    """
    max_n_atoms = len(representative_atoms)
    cache_key = _parallel_capacity_cache_key(
        ts_model, max_n_atoms, config, frozen_indices=frozen_indices
    )
    cached_capacity = capacity_cache_get(cache_key)
    if cached_capacity is not None:
        return cached_capacity

    fallback = 1
    uses_explicit_scaler = config.autobatcher_max_memory_scaler is not None
    if (
        _deps.ts is None
        or _deps.ts_constraints is None
        or _deps.calculate_memory_scalers is None
        or ts_model is None
        or (not uses_explicit_scaler and _deps.determine_max_batch_size is None)
    ):
        logger.warning(
            "TorchSim unavailable; using parallel relaxation capacity=%d",
            fallback,
        )
        capacity_cache_set(cache_key, fallback)
        return fallback

    try:
        _validate_model_pbc(
            representative_atoms,
            context="estimate_parallel_relaxation_capacity",
        )
        model_device = getattr(ts_model, "device", None)
        if model_device is None:
            model_device = _resolve_device(config.device)
        if model_device is None:
            model_device = "cpu"

        with torchsim_output_capture():
            state = _make_state_with_frozen_constraint(
                representative_atoms,
                frozen_indices,
                ts_model,
                model_device,
            )

        memory_scales_with = "n_atoms"
        first_metric = _deps.calculate_memory_scalers(state, memory_scales_with)[0]
        padding = config.autobatcher_max_memory_padding

        if first_metric <= 0 or not np.isfinite(first_metric):
            raise RuntimeError(
                f"Invalid memory scaler metric from probe: {first_metric!r}"
            )

        if config.autobatcher_max_memory_scaler is not None:
            n_systems = max(
                1, int(config.autobatcher_max_memory_scaler // first_metric)
            )
        else:  # pragma: no cover - requires MLIP stack / GPU
            resolved_max_atoms_to_try, _ = _resolve_autobatcher_max_atoms_to_try(
                max_n_atoms=max_n_atoms,
                n_systems=1,
                config=config,
            )
            probed = _deps.determine_max_batch_size(
                state,
                ts_model,
                max_atoms=resolved_max_atoms_to_try,
            )
            n_systems = max(1, int(probed * 0.8 * padding))

        capacity_cache_set(cache_key, n_systems)
        logger.info(
            "Probed parallel relaxation capacity=%d (max_n_atoms=%d, padding=%.2f)",
            n_systems,
            max_n_atoms,
            padding,
        )
        return n_systems
    except _deps._CAPACITY_PROBE_ERRORS as exc:
        logger.warning(
            "Parallel capacity probe failed (%s); using capacity=%d",
            exc,
            fallback,
        )
        return fallback


def batch_static(
    atoms_list: list[Atoms],
    ts_model,
) -> list[tuple[float, np.ndarray]]:
    """Batched single-point via ``ts.static(system=atoms_list, model=...)``.

    Returns a list of ``(energy, forces)`` tuples, one per input Atoms.
    Much faster than calling ``ts.static`` once per system because the model
    forward pass is fused across all systems.

    Parameters
    ----------
    atoms_list
        List of ASE Atoms objects.
    ts_model
        TorchSim model instance.
    """
    ts = _deps.ts
    if ts is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "batch_static",
            "Install with: pip install torch-sim-atomistic",
        )
    if not atoms_list:
        return []
    for i, atoms in enumerate(atoms_list):
        _validate_model_pbc(atoms, context=f"batch_static system[{i}]")
    with torchsim_output_capture():
        result_list = ts.static(system=atoms_list, model=ts_model)
    if len(result_list) != len(atoms_list):
        raise RuntimeError(
            "ML model returned mismatched batch size in ts.static: "
            f"expected {len(atoms_list)}, got {len(result_list)}."
        )
    out: list[tuple[float, np.ndarray]] = []
    for atoms, res in zip(atoms_list, result_list, strict=True):
        e = res.get("potential_energy")
        f = res.get("forces")
        if e is None:
            raise RuntimeError(
                "ML model returned no energy (out['potential_energy'] is None). "
                "Check GPU memory and model output."
            )
        energy = float(e.detach().cpu().numpy().squeeze())
        forces = (
            f.detach().cpu().numpy() if f is not None else np.zeros((len(atoms), 3))
        )
        out.append((energy, forces))
    return out


def _split_forces_by_system(
    batch, n_systems: int, atom_counts: list[int]
) -> list[np.ndarray] | None:
    """Split a torch-sim batch's per-atom force tensor into per-system arrays.

    ``forces`` is an *atom* attribute on torch-sim ``OptimState`` (unlike
    ``energy``/``stress``, which are *system* attributes), so it is a single
    ``(n_atoms_total, 3)`` tensor for the whole concatenated batch. Indexing it
    with a system index silently yields the force vector of one atom, which
    makes downstream per-adsorbate force-convergence checks operate on a
    ``(3,)`` array and never fire. Split on ``system_idx`` instead.

    Returns ``None`` when forces (or the system index) are unavailable, and
    raises ``RuntimeError`` if the split does not reproduce *atom_counts*.
    """
    forces = getattr(batch, "forces", None)
    if forces is None:
        return None
    forces_np = np.asarray(forces.detach().cpu().numpy(), dtype=float)
    system_idx = getattr(batch, "system_idx", None)
    if system_idx is None:
        return None
    idx_np = np.asarray(system_idx.detach().cpu().numpy()).reshape(-1)
    if forces_np.ndim != 2 or forces_np.shape[0] != idx_np.shape[0]:
        raise RuntimeError(
            "Batched forces do not align with the per-atom system index: "
            f"forces shape {forces_np.shape}, system_idx length {idx_np.shape[0]}."
        )
    per_system = [forces_np[idx_np == k] for k in range(n_systems)]
    for k, (got, expected) in enumerate(zip(per_system, atom_counts, strict=True)):
        if got.shape != (expected, 3):
            raise RuntimeError(
                "Batched forces could not be split per system: system "
                f"{k} has {got.shape[0]} force rows but {expected} atoms."
            )
    return per_system


# ---------------------------------------------------------------------------
# isolated molecule optimisation
# ---------------------------------------------------------------------------


def optimize_isolated_molecules_batched(  # pragma: no cover - requires MLIP stack / GPU
    conformers: list[Atoms],
    ts_model,
    fmax: float = DEFAULT_FMAX,
    steps: int = 100,
    config: AdsorptionConfig | None = None,
) -> list[tuple[Atoms, float]]:
    """Batch-optimise isolated molecule conformers (no constraints).

    Parameters
    ----------
    conformers
        List of ASE Atoms conformers.
    ts_model
        TorchSim model instance.
    fmax
        Force convergence threshold in eV/Å.
    steps
        Maximum optimization steps.
    config
        Adsorption configuration.
    """
    if not conformers:
        return []
    ts = _deps.ts
    if ts is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "optimize_isolated_molecules_batched",
            "Install with: pip install torch-sim-atomistic",
        )
    if ts_model is None:
        raise ValueError("ts_model must not be None")

    optimizer = _resolve_ts_optimizer(config.ts_optimizer if config else "fire")
    swaps = config.steps_between_swaps if config else 5
    logger.info("Batched optimisation of %d isolated conformers", len(conformers))
    with torchsim_output_capture():
        conv = ts.generate_force_convergence_fn(
            force_tol=fmax, include_cell_forces=False
        )
    _maybe_clear_cuda_cache(ts_model)
    try:
        ab = None
        if config is not None and not config.optimize_isolated_sequentially:
            max_n_atoms = max(len(a) for a in conformers)
            resolved_max_atoms_to_try, cap_source = (
                _resolve_autobatcher_max_atoms_to_try(
                    max_n_atoms=max_n_atoms,
                    n_systems=len(conformers),
                    config=config,
                )
            )
            logger.info(
                "Isolated autobatcher probe cap=%d (source=%s, max_n_atoms=%d, n_systems=%d, multiplier=%.2f, bucket=%d)",
                resolved_max_atoms_to_try,
                cap_source,
                max_n_atoms,
                len(conformers),
                _DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER,
                _DYNAMIC_AUTOBATCHER_CAP_BUCKET,
            )
            ab = _get_inflight_autobatcher(
                ts_model,
                max_n_atoms,
                config=config,
                max_atoms_to_try=resolved_max_atoms_to_try,
            )[0]
        with torchsim_output_capture():
            state = ts.optimize(
                system=conformers,
                model=ts_model,
                optimizer=optimizer,
                convergence_fn=conv,
                max_steps=steps,
                steps_between_swaps=swaps,
                autobatcher=ab if ab is not None else False,
            )
        atoms_list = state.to_atoms()
        energies = state.energy
        if len(atoms_list) != len(energies):
            raise RuntimeError(
                "TorchSim optimize returned mismatched lengths for atoms/energies: "
                f"{len(atoms_list)} vs {len(energies)}."
            )
        results = []
        for a, e in zip(atoms_list, energies, strict=True):
            results.append((a, e.item()))
        logger.info("Isolated optimisation complete: %d conformers", len(results))
        return results
    finally:
        clear_autobatcher_cache()


# ---------------------------------------------------------------------------
# slab+adsorbate batched optimisation
# ---------------------------------------------------------------------------


def optimize_adsorbate_slab_batched(  # pragma: no cover - requires MLIP stack / GPU
    combined_atoms_list: list[Atoms],
    slab: Atoms,
    ts_model,
    config: AdsorptionConfig | None = None,
    base_slab_for_frozen: Atoms | None = None,
    saturation_reuse: bool = False,
) -> list[Atoms | None]:
    """Batch-optimise slab+adsorbate systems, selectively freezing sub-surface.

    Uses TorchSim's ``InFlightAutoBatcher`` for GPU-efficient batching.

    Frozen indices are read from ASE ``FixAtoms`` on the freeze reference
    (:func:`frozen_indices_from_constraints`). When *base_slab_for_frozen* is
    provided (e.g. for sequential saturation or adatom workflows), it supplies
    the constraint-bearing substrate reference instead of *slab*.

    Ordering guarantee: with torch-sim-atomistic 0.5.2, ``ts.optimize`` ends
    with ``autobatcher.restore_original_order(...)``, which pairs completed
    states with their original input indices and raises on any count mismatch.
    The returned list is therefore in the original input order and has the same
    length as *combined_atoms_list*, which is what lets callers (e.g.
    :func:`metalsurfer.workflow.shared._optimize_and_evaluate_placements`) zip
    the results positionally against their placement descriptors. A violation
    of that invariant raises :class:`RuntimeError` rather than silently
    returning misaligned or all-``None`` results.

    Parameters
    ----------
    combined_atoms_list
        List of combined slab+adsorbate ASE Atoms.
    slab
        Reference slab Atoms.
    ts_model
        TorchSim model instance.
    config
        Adsorption configuration.
    base_slab_for_frozen
        Substrate reference used to read freeze constraints.
    saturation_reuse
        Whether to reuse autobatcher estimates across saturation steps.
    """
    if config is None:
        config = AdsorptionConfig()

    if not combined_atoms_list:
        return []

    ts = _deps.ts
    if ts is None or _deps.InFlightAutoBatcher is None or _deps.ts_constraints is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "optimize_adsorbate_slab_batched",
            "Install with: pip install torch-sim-atomistic",
        )
    if ts_model is None:
        raise ValueError("ts_model must not be None")
    for i, atoms in enumerate(combined_atoms_list):
        _validate_model_pbc(
            atoms, context=f"optimize_adsorbate_slab_batched system[{i}]"
        )

    # pragma: no cover below -- the remaining body builds TorchSim states and
    # drives ts.optimize on a real MLIP model (``mlip``/``gpu`` suites).
    slab_for_frozen = base_slab_for_frozen if base_slab_for_frozen is not None else slab
    slab_size = len(slab)
    ref_len = len(slab_for_frozen)
    if base_slab_for_frozen is not None and ref_len > slab_size:
        logger.warning(
            "Base_slab_for_frozen has %d atoms but slab reference has %d; "
            "frozen indices may not align with the substrate prefix",
            ref_len,
            slab_size,
        )
    frozen_indices = frozen_indices_from_constraints(slab_for_frozen)
    if frozen_indices and max(frozen_indices) >= ref_len:
        logger.warning(
            "Frozen index %d exceeds freeze reference length %d",
            max(frozen_indices),
            ref_len,
        )

    logger.info(
        "Batched optimisation of %d systems (slab=%d atoms, freeze_ref=%d, frozen=%d)",
        len(combined_atoms_list),
        slab_size,
        ref_len,
        len(frozen_indices),
    )

    max_steps = config.stage1_steps + config.stage2_steps
    optimizer = _resolve_ts_optimizer(config.ts_optimizer)
    swaps = config.steps_between_swaps

    model_device = getattr(ts_model, "device", None)
    if model_device is None:
        model_device = _resolve_device(config.device)
    if model_device is None:
        model_device = "cpu"
    with torchsim_output_capture():
        conv = ts.generate_force_convergence_fn(
            force_tol=config.fmax, include_cell_forces=False
        )
        sim_states = [
            _make_state_with_frozen_constraint(
                atoms, frozen_indices, ts_model, model_device
            )
            for atoms in combined_atoms_list
        ]

    try:
        max_n_atoms = max(len(a) for a in combined_atoms_list)
        resolved_max_atoms_to_try, cap_source = _resolve_autobatcher_max_atoms_to_try(
            max_n_atoms=max_n_atoms,
            n_systems=len(combined_atoms_list),
            config=config,
        )
        logger.info(
            "Autobatcher probe cap=%d (source=%s, max_n_atoms=%d, n_systems=%d, multiplier=%.2f, bucket=%d)",
            resolved_max_atoms_to_try,
            cap_source,
            max_n_atoms,
            len(combined_atoms_list),
            _DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER,
            _DYNAMIC_AUTOBATCHER_CAP_BUCKET,
        )
        _maybe_clear_cuda_cache(ts_model)
        use_saturation_reuse = saturation_reuse and config.saturation_autobatcher_reuse
        ab, cache_key, reused_prior_estimate = _get_inflight_autobatcher(
            ts_model,
            max_n_atoms,
            config=config,
            saturation_reuse=use_saturation_reuse,
            max_atoms_to_try=resolved_max_atoms_to_try,
        )
        if ab is None:
            raise RuntimeError("Could not create autobatcher")

        def _run_optimize(ab):
            with torchsim_output_capture():
                return ts.optimize(
                    system=sim_states,
                    model=ts_model,
                    optimizer=optimizer,
                    convergence_fn=conv,
                    max_steps=max_steps,
                    steps_between_swaps=swaps,
                    autobatcher=ab,
                )

        try:
            batch = _run_optimize(ab)
        except RuntimeError as exc:
            if (
                use_saturation_reuse
                and reused_prior_estimate
                and _is_cuda_oom_error(exc)
                and cache_key is not None
            ):
                logger.warning(
                    "OOM after reusing saturation autobatcher estimate for max_n_atoms=%d; "
                    "dropping reused cache entry and retrying with fresh estimate",
                    max_n_atoms,
                )
                pop_autobatcher(cache_key)
                _maybe_clear_cuda_cache(ts_model)
                ab, _, _ = _get_inflight_autobatcher(
                    ts_model,
                    max_n_atoms,
                    config=config,
                    saturation_reuse=False,
                    max_atoms_to_try=resolved_max_atoms_to_try,
                )
                if ab is None:
                    raise RuntimeError("Could not create autobatcher") from exc
                batch = _run_optimize(ab)
            else:
                raise
        result = batch.to_atoms()
        energies = batch.energy
        n_input = len(combined_atoms_list)
        n_returned = len(result)
        if n_returned != n_input:
            # torch-sim 0.5.2 guarantees count-preserving, original-order output
            # (``InFlightAutoBatcher.restore_original_order`` itself raises on a
            # count mismatch, and ``max_iterations`` force-converges stragglers),
            # so this branch is unreachable in practice. If it ever fires the
            # invariant is broken: results are mapped to inputs positionally and
            # the returned state carries no stable per-system id to recover a
            # permutation from, so continuing would misattribute energies to the
            # wrong descriptor. Fail loudly instead, matching the ``batch_static``
            # guard above.
            raise RuntimeError(
                "Autobatcher returned mismatched batch size in ts.optimize: "
                f"expected {n_input}, got {n_returned}. Results are mapped to "
                "inputs positionally and cannot be realigned."
            )
        forces_list = _split_forces_by_system(
            batch, n_returned, [len(a) for a in result]
        )
        for i, atoms in enumerate(result):
            calc = TorchSimCalculator(ts_model)
            if energies is not None and i < len(energies):
                calc.results["energy"] = float(
                    energies[i].detach().cpu().numpy().squeeze()
                )
            if forces_list is not None:
                calc.results["forces"] = forces_list[i]
            calc._last_positions_hash = _positions_cell_hash(atoms)
            atoms.calc = calc
        logger.info("Autobatcher optimisation succeeded: %d systems", n_returned)
        return result
    finally:
        if saturation_reuse and config.saturation_autobatcher_reuse:
            clear_autobatcher_cache(max_n_atoms_threshold=max_n_atoms)
        else:
            clear_autobatcher_cache()

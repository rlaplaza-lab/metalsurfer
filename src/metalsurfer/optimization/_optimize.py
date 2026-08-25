"""Batched TorchSim relaxations, single-point batching, and capacity probing.

The heavy call paths here (``ts.optimize`` / ``ts.static`` / ``ts.initialize_state``)
require the MLIP stack and a GPU to be meaningful, so they are marked
``# pragma: no cover``; the surrounding pure logic lives in :mod:`._validation`
and :mod:`._cache` and is unit-tested on CPU.
"""

import gc
import logging
from collections.abc import Callable, Sequence
from typing import Any

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
    _device_is_cuda,
    _is_cuda_oom_error,
    _parallel_capacity_cache_key,
    _positions_cell_hash,
    _resolve_autobatcher_max_atoms_to_try,
    _resolve_model_device,
    _resolve_ts_optimizer,
    _validate_model_pbc,
)

logger = logging.getLogger(__name__)


def setup_single_model(  # pragma: no cover - requires MLIP stack / GPU
    model_name: str = "uma-s-1p2",
    device: str = "cuda",
    task_name: str = "oc25",
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
    task_name
        UMA/FairChem task head used for energy/force evaluation.
        ``"oc25"`` targets (electro)catalysis and is only available on
        ``*-1p2`` checkpoints; use ``"oc20"`` with ``uma-s-1p1`` /
        ``uma-m-1p1`` models.
    """
    ts_model = setup_torchsim_model(model_name, device, task_name=task_name)
    calculator = TorchSimCalculator(ts_model)
    return calculator, ts_model


def _clip_frozen_indices_to_slab(
    frozen_indices: list[int],
    *,
    slab_size: int,
) -> list[int]:
    """Drop freeze indices that fall outside the combined-system slab prefix."""
    if not frozen_indices:
        return []
    kept = [i for i in frozen_indices if 0 <= i < slab_size]
    dropped = len(frozen_indices) - len(kept)
    if dropped:
        logger.warning(
            "Dropped %d frozen index(es) >= slab_size=%d (max was %d)",
            dropped,
            slab_size,
            max(frozen_indices),
        )
    return kept


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

    # Align atom_idx device before TorchSim select_sub_constraint (CPU/CUDA mix).
    base_cls = _deps.ts_constraints.FixAtoms

    class _DeviceAlignedFixAtoms(base_cls):  # type: ignore[misc,valid-type]
        def select_sub_constraint(self, atom_idx, sys_idx):
            if hasattr(atom_idx, "device") and atom_idx.device != self.atom_idx.device:
                atom_idx = atom_idx.to(device=self.atom_idx.device)
            return super().select_sub_constraint(atom_idx, sys_idx)

    state.constraints = [_DeviceAlignedFixAtoms(atom_idx=idx_tensor)]
    return state


def _run_optimize_with_oom_retry(
    run_optimize: Callable[[Any, Any], Any],
    *,
    build_systems: Callable[[], Any],
    initial_autobatcher: Any,
    ts_model,
    max_n_atoms: int,
    config: AdsorptionConfig,
    cache_key: tuple | None,
    resolved_max_atoms_to_try: int,
    context: str,
) -> Any:
    """Run *run_optimize* once; on CUDA OOM rebuild systems + autobatcher, retry once.

    The retried attempt gets freshly built systems: a failed CUDA attempt may
    leave mutated / NaN state behind, and holding the originals would also keep
    the failed attempt's device tensors alive across
    :func:`_maybe_clear_cuda_cache`.
    """
    systems = build_systems()
    oom_exc: RuntimeError | None = None
    try:
        return run_optimize(initial_autobatcher, systems)
    except RuntimeError as exc:
        if not _is_cuda_oom_error(exc):
            raise
        # Drop the traceback: its frames reference the states handed to
        # ts.optimize and would otherwise pin them for the whole retry.
        oom_exc = exc.with_traceback(None)
    del systems
    gc.collect()
    if cache_key is not None:
        pop_autobatcher(cache_key)
    _maybe_clear_cuda_cache(ts_model)
    logger.warning(
        "CUDA OOM during %s; dropped autobatcher cache entry, rebuilding "
        "systems and retrying once",
        context,
    )
    ab, _ = _get_inflight_autobatcher(
        ts_model,
        max_n_atoms,
        config=config,
        saturation_reuse=False,
        max_atoms_to_try=resolved_max_atoms_to_try,
    )
    if ab is None:
        raise RuntimeError("Could not create autobatcher after OOM") from oom_exc
    return run_optimize(ab, build_systems())


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
        model_device = _resolve_model_device(ts_model, config)

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
                1,
                int(config.autobatcher_max_memory_scaler * padding // first_metric),
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
            n_systems = max(1, int(probed * padding))

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
    *,
    zero_fallback: bool = True,
    validate_pbc: bool = True,
    require_energy: bool = True,
) -> list[tuple[float, np.ndarray | None]]:
    """Batched single-point via ``ts.static(system=atoms_list, model=...)``.

    Returns a list of ``(energy, forces)`` tuples, one per input Atoms.
    Much faster than calling ``ts.static`` once per system because the model
    forward pass is fused across all systems.

    When *zero_fallback* is False, systems whose forces are missing yield
    ``None`` forces instead of zeros. When *validate_pbc* is False, the
    per-system PBC check is skipped (callers that already validated the inputs
    can pass False for a hot path). When *require_energy* is False, a missing
    energy is tolerated (substituted with ``NaN``) — used by force-recovery
    paths that only need forces.

    Parameters
    ----------
    atoms_list
        List of ASE Atoms objects.
    ts_model
        TorchSim model instance.
    zero_fallback
        Replace missing forces with zeros when True, else ``None``.
    validate_pbc
        Validate per-system PBC before running the model.
    require_energy
        Raise if a system returns no energy when True, else substitute ``NaN``.
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
        if validate_pbc:
            _validate_model_pbc(atoms, context=f"batch_static system[{i}]")
    with torchsim_output_capture():
        result_list = ts.static(system=atoms_list, model=ts_model)
    if len(result_list) != len(atoms_list):
        raise RuntimeError(
            "ML model returned mismatched batch size in ts.static: "
            f"expected {len(atoms_list)}, got {len(result_list)}."
        )
    out: list[tuple[float, np.ndarray | None]] = []
    for atoms, res in zip(atoms_list, result_list, strict=True):
        e = res.get("potential_energy")
        f = res.get("forces")
        if e is None:
            if require_energy:
                raise RuntimeError(
                    "ML model returned no energy (out['potential_energy'] is None). "
                    "Check GPU memory and model output."
                )
            energy = float("nan")
        else:
            energy = float(e.detach().cpu().numpy().squeeze())
        forces = (
            f.detach().cpu().numpy()
            if f is not None
            else (np.zeros((len(atoms), 3)) if zero_fallback else None)
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
    ``(3,)`` array and never fire. Prefer ``system_idx`` when present; a single
    system without ``system_idx`` is an unambiguous contiguous split, but a
    multi-system batch without ``system_idx`` now raises because the contiguous
    per-atom-count split cannot be verified.

    Returns ``None`` when forces are unavailable or cannot be aligned, and
    raises ``RuntimeError`` if a ``system_idx`` split does not reproduce
    *atom_counts*.
    """
    forces = getattr(batch, "forces", None)
    if forces is None:
        return None
    forces_np = np.asarray(forces.detach().cpu().numpy(), dtype=float)
    if forces_np.ndim != 2 or forces_np.shape[1] != 3:
        return None
    expected_atoms = int(sum(atom_counts))
    if forces_np.shape[0] != expected_atoms:
        return None

    system_idx = getattr(batch, "system_idx", None)
    if system_idx is None:
        if n_systems > 1:
            raise RuntimeError(
                "Batched forces cannot be split per system: the returned batch has "
                f"{n_systems} systems but no per-atom system_idx, and a contiguous "
                "split by atom count cannot be verified."
            )
        per_system = [forces_np]
    else:
        idx_np = np.asarray(system_idx.detach().cpu().numpy()).reshape(-1)
        if forces_np.shape[0] != idx_np.shape[0]:
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


def _forces_for_optimized_systems(
    batch,
    energies,
    result: list[Atoms],
    ts_model,
) -> Sequence[np.ndarray | None]:
    """Return per-system force arrays, recovering them when the batched run hid them.

    Tries :func:`_split_forces_by_system` first (the fast path). If that returns
    ``None`` (forces unavailable or unaligned with atom counts), runs **one**
    ``ts.static`` over the finite-energy survivors and extracts per-system
    forces — the same fused path as :func:`batch_static`, but without its
    zero-force fallback. Systems whose forces are still missing yield ``None``
    (never zeros) so downstream force-convergence checks never see a spurious
    ``|F| == 0``. The optimised energies are kept as-is.
    """
    n_systems = len(result)
    atom_counts = [len(a) for a in result]
    forces_list = _split_forces_by_system(batch, n_systems, atom_counts)
    if forces_list is not None:
        return forces_list

    logger.warning(
        "Batched optimisation returned no splitable per-system forces; "
        "recovering via a single ts.static over %d survivors",
        n_systems,
    )
    survivor_idx: list[int] = []
    if energies is not None:
        for i in range(n_systems):
            if i < len(energies):
                try:
                    ev = float(energies[i].detach().cpu().numpy().squeeze())
                except Exception:
                    ev = float("nan")
                if np.isfinite(ev):
                    survivor_idx.append(i)
    else:
        survivor_idx = list(range(n_systems))
    if not survivor_idx:
        return [None] * n_systems

    survivor_atoms = [result[i] for i in survivor_idx]
    try:
        recovered = batch_static(
            survivor_atoms,
            ts_model,
            zero_fallback=False,
            validate_pbc=False,
            require_energy=False,
        )
    except Exception:
        logger.warning("ts.static force recovery failed", exc_info=True)
        return [None] * n_systems

    per_system = [
        None if forces is None or not np.any(forces) else forces
        for _energy, forces in recovered
    ]
    if len(per_system) != len(survivor_idx):
        return [None] * n_systems

    out: list[np.ndarray | None] = [None] * n_systems
    for k, i in enumerate(survivor_idx):
        out[i] = per_system[k]
    return out


# ---------------------------------------------------------------------------
# isolated molecule optimisation
# ---------------------------------------------------------------------------


def optimize_isolated_molecules_batched(  # pragma: no cover - requires MLIP stack / GPU
    conformers: list[Atoms],
    ts_model,
    fmax: float = DEFAULT_FMAX,
    steps: int = 100,
    *,
    config: AdsorptionConfig,
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

    optimizer = _resolve_ts_optimizer(config.ts_optimizer)
    swaps = config.steps_between_swaps
    logger.info("Batched optimisation of %d isolated conformers", len(conformers))
    with torchsim_output_capture():
        conv = ts.generate_force_convergence_fn(
            force_tol=fmax, include_cell_forces=False
        )
    _maybe_clear_cuda_cache(ts_model)
    try:
        ab = None
        cache_key: tuple | None = None
        max_n_atoms = 0
        resolved_max_atoms_to_try = 0
        if not config.optimize_isolated_sequentially and _device_is_cuda(
            getattr(ts_model, "device", None) or "cpu"
        ):
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
            ab, cache_key = _get_inflight_autobatcher(
                ts_model,
                max_n_atoms,
                config=config,
                max_atoms_to_try=resolved_max_atoms_to_try,
            )

        def _run_optimize(autobatcher, systems):
            with torchsim_output_capture():
                return ts.optimize(
                    system=systems,
                    model=ts_model,
                    optimizer=optimizer,
                    convergence_fn=conv,
                    max_steps=steps,
                    steps_between_swaps=swaps,
                    autobatcher=autobatcher if autobatcher is not None else False,
                )

        if ab is not None:
            state = _run_optimize_with_oom_retry(
                _run_optimize,
                build_systems=lambda: conformers,
                initial_autobatcher=ab,
                ts_model=ts_model,
                max_n_atoms=max_n_atoms,
                config=config,
                cache_key=cache_key,
                resolved_max_atoms_to_try=resolved_max_atoms_to_try,
                context="isolated-molecule",
            )
        else:
            state = _run_optimize(None, conformers)
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
    frozen_indices = _clip_frozen_indices_to_slab(
        frozen_indices_from_constraints(slab_for_frozen),
        slab_size=slab_size,
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

    model_device = _resolve_model_device(ts_model, config)
    with torchsim_output_capture():
        conv = ts.generate_force_convergence_fn(
            force_tol=config.fmax, include_cell_forces=False
        )

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
        use_saturation_reuse = saturation_reuse and config.saturation_autobatcher_reuse
        # TorchSim memory probing is CUDA-only; CPU uses a single sequential batch.
        use_autobatcher = _device_is_cuda(model_device)
        if use_autobatcher:
            ab, cache_key = _get_inflight_autobatcher(
                ts_model,
                max_n_atoms,
                config=config,
                saturation_reuse=use_saturation_reuse,
                max_atoms_to_try=resolved_max_atoms_to_try,
            )
            if ab is None:
                raise RuntimeError("Could not create autobatcher")
        else:
            ab, cache_key = None, None

        def _build_sim_states():
            with torchsim_output_capture():
                return [
                    _make_state_with_frozen_constraint(
                        atoms, frozen_indices, ts_model, model_device
                    )
                    for atoms in combined_atoms_list
                ]

        def _run_optimize(autobatcher, systems):
            with torchsim_output_capture():
                return ts.optimize(
                    system=systems,
                    model=ts_model,
                    optimizer=optimizer,
                    convergence_fn=conv,
                    max_steps=max_steps,
                    steps_between_swaps=swaps,
                    autobatcher=autobatcher if autobatcher is not None else False,
                )

        batch = _run_optimize_with_oom_retry(
            _run_optimize,
            build_systems=_build_sim_states,
            initial_autobatcher=ab,
            ts_model=ts_model,
            max_n_atoms=max_n_atoms,
            config=config,
            cache_key=cache_key,
            resolved_max_atoms_to_try=resolved_max_atoms_to_try,
            context="slab+adsorbate",
        )
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
        forces_list = _forces_for_optimized_systems(batch, energies, result, ts_model)
        out: list[Atoms | None] = []
        for i, atoms in enumerate(result):
            energy_val: float | None = None
            if energies is not None and i < len(energies):
                energy_val = float(energies[i].detach().cpu().numpy().squeeze())
            if energy_val is None or not np.isfinite(energy_val):
                logger.warning(
                    "Batched optimisation system %d returned non-finite energy %s; "
                    "dropping candidate",
                    i,
                    energy_val,
                )
                out.append(None)
                continue
            forces_i = forces_list[i] if forces_list is not None else None
            if forces_i is not None and not np.all(np.isfinite(forces_i)):
                logger.warning(
                    "Batched optimisation system %d returned non-finite forces; "
                    "dropping candidate",
                    i,
                )
                out.append(None)
                continue
            calc = TorchSimCalculator(ts_model)
            calc.results["energy"] = energy_val
            if forces_i is not None:
                calc.results["forces"] = forces_i
            calc._last_positions_hash = _positions_cell_hash(atoms)
            atoms.calc = calc
            out.append(atoms)
        n_ok = sum(1 for a in out if a is not None)
        logger.info(
            "Autobatcher optimisation succeeded: %d/%d systems",
            n_ok,
            n_returned,
        )
        return out
    finally:
        clear_autobatcher_cache(max_n_atoms_threshold=max_n_atoms)

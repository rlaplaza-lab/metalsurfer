"""TorchSim/ASE batched relaxations with slab freeze masks (`top_layer_tolerance`, etc.)."""

import contextlib
import gc
import logging
import math
from typing import Any, NoReturn, cast

import numpy as np
from ase import Atoms
from ase.calculators.calculator import all_changes

from ._logging import torchsim_output_capture
from .config import AdsorptionConfig
from .exceptions import DependencyMissingError
from .surface_prep.freeze import frozen_indices_from_constraints

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# optional heavy imports (safe fallbacks for environments without the MLIP
# stack -- keeps ``from metalsurfer import AdsorptionConfig`` working in CI)
# ---------------------------------------------------------------------------
try:
    import torch as _torch_mod
except ImportError:
    torch: Any = None
else:
    torch = _torch_mod

_CAPACITY_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    MemoryError,
    OSError,
)
if torch is not None:
    _cuda = getattr(torch, "cuda", None)
    _oom = getattr(_cuda, "OutOfMemoryError", None) if _cuda is not None else None
    if isinstance(_oom, type) and issubclass(_oom, BaseException):
        _CAPACITY_PROBE_ERRORS = (*_CAPACITY_PROBE_ERRORS, _oom)

ts: Any = None
ts_constraints: Any = None
InFlightAutoBatcher: Any = None
determine_max_batch_size: Any = None
calculate_memory_scalers: Any = None

try:
    import torch_sim as _ts_mod
    import torch_sim.constraints as _ts_constraints_mod
    import torch_sim.state as _ts_state
    from torch_sim.autobatching import (
        InFlightAutoBatcher as _InFlightAutoBatcher,
    )
    from torch_sim.autobatching import (
        calculate_memory_scalers as _calculate_memory_scalers,
    )
    from torch_sim.autobatching import (
        determine_max_batch_size as _determine_max_batch_size,
    )

    ts = _ts_mod
    ts_constraints = _ts_constraints_mod
    InFlightAutoBatcher = _InFlightAutoBatcher
    determine_max_batch_size = _determine_max_batch_size
    calculate_memory_scalers = _calculate_memory_scalers

    # Upstream _split_state uses torch.arange(...) without device=, so bounds
    # live on CPU while constraint tensors are on CUDA. Patch to use state.device.
    def _patched_split_state(state):
        from torch_sim.state import get_attrs_for_scope

        system_sizes = state.n_atoms_per_system.tolist()
        split_per_atom = {}
        for attr_name, attr_value in get_attrs_for_scope(state, "per-atom"):
            if attr_name != "system_idx":
                split_per_atom[attr_name] = torch.split(attr_value, system_sizes, dim=0)
        split_per_system = {}
        for attr_name, attr_value in get_attrs_for_scope(state, "per-system"):
            if isinstance(attr_value, torch.Tensor):
                split_per_system[attr_name] = torch.split(attr_value, 1, dim=0)
            else:
                split_per_system[attr_name] = [attr_value] * state.n_systems
        global_attrs = dict(get_attrs_for_scope(state, "global"))
        states = []
        n_systems = len(system_sizes)
        zero_tensor = torch.tensor([0], device=state.device, dtype=torch.long)
        cumsum_atoms = torch.cat(
            (zero_tensor, torch.cumsum(state.n_atoms_per_system, dim=0))
        )
        for sys_idx in range(n_systems):
            per_system_dict = {
                attr_name: split_per_system[attr_name][sys_idx]
                for attr_name in split_per_system
            }
            system_attrs = {
                "system_idx": torch.zeros(
                    system_sizes[sys_idx], device=state.device, dtype=torch.long
                ),
                **{
                    attr_name: split_per_atom[attr_name][sys_idx]
                    for attr_name in split_per_atom
                },
                **per_system_dict,
                **global_attrs,
            }
            atom_idx = torch.arange(
                cumsum_atoms[sys_idx].item(),
                cumsum_atoms[sys_idx + 1].item(),
                device=state.device,
            )
            new_constraints = [
                new_constraint
                for constraint in state.constraints
                if (
                    new_constraint := constraint.select_sub_constraint(
                        atom_idx, sys_idx
                    )
                )
            ]
            system_attrs["_constraints"] = new_constraints
            states.append(type(state)(**system_attrs))
        return states

    _ts_state._split_state = _patched_split_state
except ImportError:
    pass

# NOTE: single-thread access assumed; no lock needed for current workflows.
_AUTOBATCHER_CACHE: dict[tuple, Any] = {}
_PARALLEL_CAPACITY_CACHE: dict[tuple, int] = {}
_DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER = 2.0
_DYNAMIC_AUTOBATCHER_CAP_MIN = 5_000
_DYNAMIC_AUTOBATCHER_CAP_MAX = 200_000
_DYNAMIC_AUTOBATCHER_CAP_BUCKET = 5_000


def _validate_model_pbc(atoms: Atoms, *, context: str) -> None:
    """Reject mixed-PBC systems before passing them to UMA/FairChem."""
    pbc = np.array(atoms.get_pbc(), dtype=bool)
    if pbc.shape != (3,):
        raise ValueError(
            f"{context}: invalid PBC shape {pbc.shape}; expected 3 components."
        )
    if bool(pbc.all()) or bool((~pbc).all()):
        return
    raise ValueError(
        f"{context}: mixed PBC {pbc.tolist()} is not supported by the UMA model. "
        "Use either [True, True, True] with sufficient vacuum or [False, False, False]."
    )


def _positions_cell_hash(atoms: Atoms) -> int:
    """Fast content hash of positions + cell + atomic numbers for cache invalidation."""
    pos_bytes = atoms.get_positions().tobytes()
    cell_bytes = np.asarray(atoms.get_cell()).tobytes()
    num_bytes = atoms.get_atomic_numbers().tobytes()
    return hash((pos_bytes, cell_bytes, num_bytes))


def _resolve_ts_optimizer(name: str):
    """Map config string to ``ts.Optimizer`` enum member."""
    if ts is None:
        return None
    _map = {
        "fire": ts.Optimizer.fire,
        "lbfgs": ts.Optimizer.lbfgs,
        "bfgs": ts.Optimizer.bfgs,
    }
    return _map.get(name, ts.Optimizer.fire)


def clear_autobatcher_cache(
    max_n_atoms_threshold: int | None = None,
    ts_model=None,
) -> None:
    """Evict cached autobatchers to free GPU memory before larger runs.

    Call after isolated-molecule optimization (before slab+adsorbate) so memory
    from small-system batchers is released. If max_n_atoms_threshold is set,
    only evicts entries with max_n_atoms below it (keeps slab+adsorbate
    batchers when processing multiple molecules). If None, clears all.
    """
    if max_n_atoms_threshold is None:
        # Drop strong refs explicitly: TorchSim InFlightAutoBatcher can retain GPU
        # tensors until batchers are GC'd; dict.clear() alone is easy to leave
        # garbage uncollected between pytest tests.
        _holders = list(_AUTOBATCHER_CACHE.values())
        _AUTOBATCHER_CACHE.clear()
        _PARALLEL_CAPACITY_CACHE.clear()
        del _holders
        logger.debug("Cleared entire autobatcher cache")
    else:
        to_remove = [k for k in _AUTOBATCHER_CACHE if k[4] < max_n_atoms_threshold]
        for k in to_remove:
            del _AUTOBATCHER_CACHE[k]
        if to_remove:
            logger.debug(
                "Evicted %d autobatcher(s) with max_n_atoms < %d",
                len(to_remove),
                max_n_atoms_threshold,
            )
    if torch is not None and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.synchronize()
    for _ in range(3):
        gc.collect()
    if torch is not None and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()
        ipc = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc):
            with contextlib.suppress(RuntimeError):
                ipc()


def _maybe_clear_cuda_cache(ts_model) -> None:
    """Clear CUDA cache before batched optimization to reduce OOM risk."""
    if torch is None or not torch.cuda.is_available():
        return
    dev = getattr(ts_model, "device", None)
    on_cuda = (isinstance(dev, str) and dev.lower().startswith("cuda")) or (
        hasattr(dev, "type") and str(getattr(dev, "type", "")).lower() == "cuda"
    )
    if on_cuda:
        torch.cuda.empty_cache()


def _get_inflight_autobatcher(
    ts_model,
    max_n_atoms: int,
    *,
    memory_scales_with: str = "n_atoms",
    max_memory_padding: float = 0.5,
    max_memory_scaler: float | None = None,
    max_atoms_to_try: int = 100_000,
    config: AdsorptionConfig | None = None,
    saturation_reuse: bool = False,
):
    """Create or return cached InFlightAutoBatcher for batched relaxations."""
    if InFlightAutoBatcher is None or ts is None or ts_model is None:
        return None
    if config is not None:
        max_memory_padding = config.autobatcher_max_memory_padding
        max_memory_scaler = config.autobatcher_max_memory_scaler
    try:
        dev = getattr(ts_model, "device", None)
        key = (
            id(ts_model),
            str(dev),
            memory_scales_with,
            float(max_memory_padding),
            int(max_n_atoms),
            max_memory_scaler,
            int(max_atoms_to_try),
        )
        cached = _AUTOBATCHER_CACHE.get(key)
        if cached is not None:
            return cached, key, False
        if saturation_reuse:
            matching_candidates: list[tuple[tuple, Any]] = []
            for cache_key, cache_ab in _AUTOBATCHER_CACHE.items():
                if (
                    cache_key[0] == key[0]
                    and cache_key[1] == key[1]
                    and cache_key[2] == key[2]
                    and cache_key[3] == key[3]
                    and cache_key[5] == key[5]
                    and cache_key[6] == key[6]
                ):
                    matching_candidates.append((cache_key, cache_ab))
            matching_candidates.sort(key=lambda item: int(item[0][4]), reverse=True)
            growth_atoms = 32
            growth_fraction = 0.1
            if config is not None:
                growth_atoms = config.saturation_autobatcher_reuse_growth_atoms
                growth_fraction = config.saturation_autobatcher_reuse_growth_fraction
            for cache_key, cache_ab in matching_candidates:
                cached_max_atoms = int(cache_key[4])
                allowed_growth = max(
                    growth_atoms,
                    int(round(cached_max_atoms * growth_fraction)),
                )
                if (
                    cached_max_atoms
                    < max_n_atoms
                    <= (cached_max_atoms + allowed_growth)
                ):
                    logger.info(
                        "Reusing saturation autobatcher from max_n_atoms=%d for %d "
                        "(allowed_growth=%d) to skip memory re-estimation",
                        cached_max_atoms,
                        max_n_atoms,
                        allowed_growth,
                    )
                    return cache_ab, cache_key, True
        kwargs: dict = {
            "memory_scales_with": memory_scales_with,
            "max_memory_padding": max_memory_padding,
            "max_atoms_to_try": max_atoms_to_try,
        }
        if max_memory_scaler is not None:
            kwargs["max_memory_scaler"] = max_memory_scaler
        ab: Any = InFlightAutoBatcher(ts_model, **kwargs)
        _AUTOBATCHER_CACHE[key] = ab
        return ab, key, False
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to create InFlightAutoBatcher: %s", exc)
        return None, None, False


def _is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return (
        "out of memory" in message
        or "cuda oom" in message
        or "cuda out of memory" in message
    )


def _resolve_autobatcher_max_atoms_to_try(
    *,
    max_n_atoms: int,
    n_systems: int,
    config: AdsorptionConfig,
) -> tuple[int, str]:
    """Resolve probe cap for TorchSim memory estimation.

    Returns ``(cap, source)`` where source is ``"config_override"`` or ``"dynamic"``.
    """
    override = config.autobatcher_max_atoms_to_try
    if override is not None:
        return int(override), "config_override"
    estimated = _DYNAMIC_AUTOBATCHER_CAP_MULTIPLIER * max_n_atoms * n_systems
    bucketed = int(
        math.ceil(estimated / _DYNAMIC_AUTOBATCHER_CAP_BUCKET)
        * _DYNAMIC_AUTOBATCHER_CAP_BUCKET
    )
    cap = max(_DYNAMIC_AUTOBATCHER_CAP_MIN, min(_DYNAMIC_AUTOBATCHER_CAP_MAX, bucketed))
    return cap, "dynamic"


def _parallel_capacity_cache_key(
    ts_model,
    max_n_atoms: int,
    config: AdsorptionConfig,
) -> tuple:
    dev = getattr(ts_model, "device", None)
    dev_key = str(dev) if dev is not None else "unknown"
    return (
        id(ts_model),
        dev_key,
        max_n_atoms,
        config.autobatcher_max_memory_padding,
        config.autobatcher_max_memory_scaler,
        config.autobatcher_max_atoms_to_try,
    )


def estimate_parallel_relaxation_capacity(
    ts_model,
    representative_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    frozen_indices: list[int],
) -> int:
    """Estimate how many slab+adsorbate relaxations can run in parallel on GPU.

    Mirrors TorchSim ``InFlightAutoBatcher`` memory probing. Returns at least 1.
    """
    max_n_atoms = len(representative_atoms)
    cache_key = _parallel_capacity_cache_key(ts_model, max_n_atoms, config)
    if cache_key in _PARALLEL_CAPACITY_CACHE:
        return _PARALLEL_CAPACITY_CACHE[cache_key]

    fallback = 1
    uses_explicit_scaler = config.autobatcher_max_memory_scaler is not None
    if (
        ts is None
        or ts_constraints is None
        or calculate_memory_scalers is None
        or ts_model is None
        or (not uses_explicit_scaler and determine_max_batch_size is None)
    ):
        logger.warning(
            "TorchSim unavailable; using parallel relaxation capacity=%d",
            fallback,
        )
        _PARALLEL_CAPACITY_CACHE[cache_key] = fallback
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
        first_metric = calculate_memory_scalers(state, memory_scales_with)[0]
        padding = config.autobatcher_max_memory_padding

        if config.autobatcher_max_memory_scaler is not None:
            effective_scaler = config.autobatcher_max_memory_scaler * padding
            n_systems = max(1, int(effective_scaler // first_metric))
        else:
            resolved_max_atoms_to_try, _ = _resolve_autobatcher_max_atoms_to_try(
                max_n_atoms=max_n_atoms,
                n_systems=1,
                config=config,
            )
            probed = determine_max_batch_size(
                state,
                ts_model,
                max_atoms=resolved_max_atoms_to_try,
            )
            n_systems = max(1, int(probed * 0.8 * padding))

        _PARALLEL_CAPACITY_CACHE[cache_key] = n_systems
        logger.info(
            "Probed parallel relaxation capacity=%d (max_n_atoms=%d, padding=%.2f)",
            n_systems,
            max_n_atoms,
            padding,
        )
        return n_systems
    except _CAPACITY_PROBE_ERRORS as exc:
        logger.warning(
            "Parallel capacity probe failed (%s); using capacity=%d",
            exc,
            fallback,
        )
        _PARALLEL_CAPACITY_CACHE[cache_key] = fallback
        return fallback


# ---------------------------------------------------------------------------
# model setup helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> str | None:
    """Resolve device string: use CPU when CUDA is requested but unavailable.

    Enables tests and CI (no GPU) to run without false failures when
    config defaults to device='cuda'.
    """
    if device is None or not (
        isinstance(device, str) and device.lower().startswith("cuda")
    ):
        return device
    if torch is None:
        return "cpu"
    try:
        if not torch.cuda.is_available():
            logger.info("CUDA requested but not available; using CPU")
            return "cpu"
    except (RuntimeError, AttributeError):
        return "cpu"
    return device


def _ensure_torch_checkpoint_safe_globals() -> None:
    """Allow PyTorch 2.6+ to unpickle FairChem checkpoints that reference ``slice``."""
    if torch is None:
        return
    try:
        add_sg = torch.serialization.add_safe_globals
    except AttributeError:
        return
    with contextlib.suppress(TypeError, ValueError):
        add_sg([slice])


def _fairchem_pytorch26_unpickling_message() -> str:
    return (
        "FairChem model loading failed due to PyTorch 2.6+ weights_only changes "
        "(UnpicklingError involving slice). metalsurfer registers slice via "
        "add_safe_globals; if this persists, see "
        "https://pytorch.org/docs/stable/generated/torch.load.html and "
        "https://github.com/facebookresearch/fairchem"
    )


def _fairchem_load_failure_message(error_msg: str, model_name: str) -> str:
    return (
        f"FairChem model loading failed: {error_msg}. "
        f"Check HF token, network, and model name {model_name!r}. "
        "See https://github.com/facebookresearch/fairchem"
    )


def _raise_fairchem_load_error(exc: Exception, model_name: str) -> NoReturn:
    error_msg = str(exc)
    if (
        "UnpicklingError" in error_msg
        and ("weights_only" in error_msg or "weights only" in error_msg)
        and "slice" in error_msg
    ):
        raise RuntimeError(_fairchem_pytorch26_unpickling_message()) from exc
    raise RuntimeError(_fairchem_load_failure_message(error_msg, model_name)) from exc

def setup_torchsim_model(model_name: str = "uma-s-1p2", device: str = "cuda"):
    """Create a TorchSim FairChemModel wrapper.

    Uses torch-sim-atomistic FairChemModel API: model, device, task_name.
    """
    try:
        from torch_sim.models.fairchem import FairChemModel
    except ImportError as exc:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "setup_torchsim_model",
            "Install with: pip install torch-sim-atomistic",
        ) from exc

    resolved_device = _resolve_device(device)
    if resolved_device is None:
        raise ValueError("device must be set for TorchSim model initialization")
    device = resolved_device
    _ensure_torch_checkpoint_safe_globals()
    logger.info("Initializing TorchSim FairChemModel (%s) on %s...", model_name, device)
    dev = torch.device(device) if torch is not None and device else None
    try:
        with torchsim_output_capture():
            model = cast(Any, FairChemModel)(
                model=model_name, device=dev, task_name="oc20"
            )
    except Exception as exc:
        _raise_fairchem_load_error(exc, model_name)
    logger.info("TorchSim model created successfully")
    return model


class TorchSimCalculator:
    """ASE calculator that wraps a TorchSim ModelInterface for single-point energy/forces.

    Uses ``ts.static()`` under the hood for efficient single-point evaluation.
    Outputs are in ASE units (eV, eV/Å).

    Cache invalidation uses a content hash of positions, cell, and atomic
    numbers so that in-place mutations of the same ``Atoms`` object (common
    during ASE optimization loops) are detected correctly.
    """

    def __init__(self, ts_model):
        """Wrap a TorchSim model (e.g. FairChemModel) for ASE compatibility."""
        self._model = ts_model
        self.results: dict = {}
        self._last_positions_hash: int | None = None

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        """Run single-point calculation via ``ts.static()``.

        ``system_changes`` is accepted for ASE calculator compatibility but
        ignored; each call recomputes from the current ``Atoms`` geometry.
        """
        _ = system_changes
        if ts is None or atoms is None:
            return
        _validate_model_pbc(atoms, context="TorchSimCalculator.calculate")
        properties = properties or ["energy", "forces"]
        with torchsim_output_capture():
            result_list = ts.static(system=atoms, model=self._model)
        out = result_list[0]
        energy = out.get("potential_energy")
        forces = out.get("forces")
        if energy is None:
            raise RuntimeError(
                "ML model returned no energy (out['potential_energy'] is None). "
                "This may indicate GPU memory issues, model output format changes, "
                "or first-run initialization failure on HPC."
            )
        e_val = float(energy.detach().cpu().numpy().squeeze())
        if not np.isfinite(e_val):
            raise RuntimeError(
                f"ML model returned non-finite energy: {e_val}. "
                "Check GPU stability and model output."
            )
        self.results["energy"] = e_val
        if forces is not None:
            self.results["forces"] = forces.detach().cpu().numpy()
        if "stress" in properties and "stress" in out and out["stress"] is not None:
            s = out["stress"].detach().cpu().numpy()
            self.results["stress"] = _voigt_6(s.squeeze())
        self._last_positions_hash = _positions_cell_hash(atoms)

    def _atoms_changed(self, atoms) -> bool:
        """True when positions/cell/species changed since the last calculation."""
        if atoms is None or self._last_positions_hash is None:
            return True
        return _positions_cell_hash(atoms) != self._last_positions_hash

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return energy in eV.

        ``force_consistent`` is accepted for ASE compatibility but ignored.
        """
        _ = force_consistent
        if atoms is not None and self._atoms_changed(atoms):
            self.calculate(atoms, ["energy", "forces"])
        energy = self.results.get("energy")
        if energy is None or not np.isfinite(energy):
            raise RuntimeError(
                f"Calculator has no valid energy (got {energy}). "
                "The model may have failed to produce energy for this system."
            )
        return energy

    def get_forces(self, atoms=None):
        """Return forces in eV/Å, shape (n_atoms, 3)."""
        if atoms is not None and self._atoms_changed(atoms):
            self.calculate(atoms, ["energy", "forces"])
        n = len(atoms) if atoms is not None else 0
        return self.results.get("forces", np.zeros((n, 3)))

    def get_stress(self, atoms=None):
        """Return stress in Voigt order (xx, yy, zz, yz, xz, xy)."""
        if atoms is not None and self._atoms_changed(atoms):
            self.calculate(atoms, ["energy", "forces", "stress"])
        return self.results.get("stress", np.zeros(6))


def _voigt_6(stress_3x3) -> np.ndarray:
    """Convert 3x3 stress to Voigt 6-component form."""
    s = np.asarray(stress_3x3).reshape(3, 3)
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])


def setup_single_model(model_name: str = "uma-s-1p2", device: str = "cuda"):
    """Create a single FairChemModel shared by calculator and TorchSim.

    Returns (calculator, ts_model) where calculator wraps ts_model.
    Prefer this over separate calculator and TorchSim model setup to reduce GPU memory.
    """
    ts_model = setup_torchsim_model(model_name, device)
    calculator = TorchSimCalculator(ts_model)
    return calculator, ts_model


def batch_static(
    atoms_list: list[Atoms],
    ts_model,
) -> list[tuple[float, np.ndarray]]:
    """Batched single-point via ``ts.static(system=atoms_list, model=...)``.

    Returns a list of ``(energy, forces)`` tuples, one per input Atoms.
    Much faster than calling ``ts.static`` once per system because the model
    forward pass is fused across all systems.
    """
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


# ---------------------------------------------------------------------------
# isolated molecule optimisation
# ---------------------------------------------------------------------------


def optimize_isolated_molecules_batched(
    conformers: list[Atoms],
    ts_model,
    fmax: float = 0.05,
    steps: int = 100,
    config: AdsorptionConfig | None = None,
) -> list[tuple[Atoms, float]]:
    """Batch-optimise isolated molecule conformers (no constraints)."""
    if not conformers:
        return []
    if ts is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "optimize_isolated_molecules_batched",
            "Install with: pip install torch-sim-atomistic",
        )
    if ts_model is None:
        raise ValueError("ts_model must not be None")

    use_autobatcher = config is not None and not config.optimize_isolated_sequentially
    optimizer = _resolve_ts_optimizer(config.ts_optimizer if config else "fire")
    swaps = config.steps_between_swaps if config else 5
    logger.info("Batched optimisation of %d isolated conformers...", len(conformers))
    with torchsim_output_capture():
        conv = ts.generate_force_convergence_fn(
            force_tol=fmax, include_cell_forces=False
        )
    _maybe_clear_cuda_cache(ts_model)
    try:
        ab = None
        if use_autobatcher and config is not None:
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


def _make_state_with_frozen_constraint(
    atoms: Atoms,
    frozen_indices: list[int],
    ts_model,
    device,
):
    """Build a TorchSim state for one system with FixAtoms constraint applied."""
    state = ts.initialize_state(atoms, device=device, dtype=ts_model.dtype)
    target_dev = state.positions.device
    idx_tensor = torch.tensor(frozen_indices, dtype=torch.long, device=target_dev)
    state.constraints = [ts_constraints.FixAtoms(atom_idx=idx_tensor)]
    return state


def optimize_adsorbate_slab_batched(
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
    """
    if config is None:
        config = AdsorptionConfig()

    if not combined_atoms_list:
        return []

    if ts is None or InFlightAutoBatcher is None or ts_constraints is None:
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
            "base_slab_for_frozen has %d atoms but slab reference has %d; "
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

    if not saturation_reuse:
        clear_autobatcher_cache(max_n_atoms_threshold=slab_size)

    logger.info(
        "Batched optimisation of %d systems (slab=%d atoms, freeze_ref=%d, frozen=%d)...",
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

        try:
            with torchsim_output_capture():
                batch = ts.optimize(
                    system=sim_states,
                    model=ts_model,
                    optimizer=optimizer,
                    convergence_fn=conv,
                    max_steps=max_steps,
                    steps_between_swaps=swaps,
                    autobatcher=ab,
                )
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
                _AUTOBATCHER_CACHE.pop(cache_key, None)
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
                with torchsim_output_capture():
                    batch = ts.optimize(
                        system=sim_states,
                        model=ts_model,
                        optimizer=optimizer,
                        convergence_fn=conv,
                        max_steps=max_steps,
                        steps_between_swaps=swaps,
                        autobatcher=ab,
                    )
            else:
                raise
        result = batch.to_atoms()
        energies = batch.energy
        forces_list = batch.forces
        for i, atoms in enumerate(result):
            calc = TorchSimCalculator(ts_model)
            if energies is not None and i < len(energies):
                calc.results["energy"] = float(
                    energies[i].detach().cpu().numpy().squeeze()
                )
            if forces_list is not None and i < len(forces_list):
                calc.results["forces"] = forces_list[i].detach().cpu().numpy()
            calc._last_positions_hash = _positions_cell_hash(atoms)
            atoms.calc = calc
        logger.info("Autobatcher optimisation succeeded: %d systems", len(result))
        return result
    finally:
        if saturation_reuse and config.saturation_autobatcher_reuse:
            clear_autobatcher_cache(max_n_atoms_threshold=max_n_atoms)
        else:
            clear_autobatcher_cache()

"""Batched geometry optimisation with selective atom freezing.

The default policy is:

* adsorbate atoms: always relax
* slab top-layer atoms: relax (can be frozen via config)
* slab sub-surface atoms: frozen

Top-layer detection uses a z-coordinate tolerance
(``config.top_layer_tolerance``).
"""

import contextlib
import gc
import logging

import numpy as np
from ase import Atoms

from .config import AdsorptionConfig
from .exceptions import DependencyMissingError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# optional heavy imports (safe fallbacks for environments without the MLIP
# stack -- keeps ``from metalsurfer import AdsorptionConfig`` working in CI)
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore
except ImportError:
    torch = None

try:
    import torch_sim as ts  # type: ignore
    import torch_sim.constraints as ts_constraints  # type: ignore
    import torch_sim.state as _ts_state  # type: ignore
    from torch_sim.autobatching import InFlightAutoBatcher  # type: ignore

    # Patch torch_sim.state._split_state: torch.arange() without device= defaults to
    # CPU even when given CUDA tensor bounds, causing device mismatch in
    # constraint.select_sub_constraint(atom_idx, ...). Fix by passing device=state.device.
    # This patch can likely be removed once the next torch-sim release is out.
    _orig_split_state = _ts_state._split_state

    def _patched_split_state(state):
        from torch_sim.state import get_attrs_for_scope  # noqa: PLC0415

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
            # Fix: pass device=state.device so atom_idx matches constraint tensors
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
            states.append(type(state)(**system_attrs))  # type: ignore[arg-type]
        return states

    _ts_state._split_state = _patched_split_state
except ImportError:
    ts = None
    ts_constraints = None
    InFlightAutoBatcher = None

_AUTOBATCHER_CACHE: dict[tuple, "InFlightAutoBatcher"] = {}


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
    global _AUTOBATCHER_CACHE
    if max_n_atoms_threshold is None:
        _AUTOBATCHER_CACHE.clear()
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
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        with contextlib.suppress(RuntimeError):
            torch.cuda.empty_cache()


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
):
    """Create or return cached InFlightAutoBatcher for batched relaxations."""
    if InFlightAutoBatcher is None or ts is None or ts_model is None:
        return None
    if config is not None:
        max_memory_padding = config.autobatcher_max_memory_padding
        max_memory_scaler = config.autobatcher_max_memory_scaler
        max_atoms_to_try = config.autobatcher_max_atoms_to_try
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
            return cached
        kwargs: dict = {
            "memory_scales_with": memory_scales_with,
            "max_memory_padding": max_memory_padding,
            "max_atoms_to_try": max_atoms_to_try,
        }
        if max_memory_scaler is not None:
            kwargs["max_memory_scaler"] = max_memory_scaler
        ab = InFlightAutoBatcher(ts_model, **kwargs)
        _AUTOBATCHER_CACHE[key] = ab
        return ab
    except (RuntimeError, TypeError, ValueError) as exc:
        logger.debug("Failed to create InFlightAutoBatcher: %s", exc)
        return None


# ---------------------------------------------------------------------------
# top-layer identification
# ---------------------------------------------------------------------------


def identify_top_layer_indices(
    slab: Atoms,
    tolerance: float = 0.5,
) -> list[int]:
    """Return atom indices belonging to the topmost surface layer.

    Atoms whose z-coordinate is within *tolerance* angstrom of the maximum
    z-position are considered "top layer".
    """
    positions = slab.get_positions()
    z_max = float(np.max(positions[:, 2]))
    return [i for i, p in enumerate(positions) if p[2] >= z_max - tolerance]


def compute_frozen_indices(
    slab: Atoms,
    config: AdsorptionConfig | None = None,
) -> list[int]:
    """Determine which slab atom indices should be frozen during optimisation.

    Default policy: freeze everything *except* the top layer.
    If ``config.relax_top_layer`` is ``False``, the entire slab is frozen.
    If ``config.freeze_symbols`` is set, only atoms whose symbol is in that
    list are frozen (regardless of layer).
    """
    if config is None:
        config = AdsorptionConfig()

    n_slab = len(slab)

    if config.freeze_symbols is not None:
        syms = slab.get_chemical_symbols()
        return [i for i, s in enumerate(syms) if s in config.freeze_symbols]

    if not config.relax_top_layer:
        return list(range(n_slab))

    top_indices = set(
        identify_top_layer_indices(slab, tolerance=config.top_layer_tolerance)
    )
    return [i for i in range(n_slab) if i not in top_indices]


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


def setup_calculator(model_name: str = "uma-s-1p1", device: str = "cuda"):
    """Create a FAIRChem ASE calculator for the given model."""
    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
    except ImportError as exc:
        raise DependencyMissingError(
            "fairchem-core",
            "setup_calculator",
            "Install with: pip install fairchem-core",
        ) from exc

    device = _resolve_device(device)
    logger.info("Initializing FAIRChem calculator (%s) on %s...", model_name, device)
    predictor = pretrained_mlip.get_predict_unit(model_name, device=device)
    calc = FAIRChemCalculator(predictor, task_name="oc20")
    logger.info("Calculator initialized successfully")
    return calc


def setup_torchsim_model(model_name: str = "uma-s-1p1", device: str = "cuda"):
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

    device = _resolve_device(device)
    logger.info("Initializing TorchSim FairChemModel (%s) on %s...", model_name, device)
    dev = torch.device(device) if torch is not None and device else None
    model = FairChemModel(model=model_name, device=dev, task_name="oc20")
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
        system_changes=None,
    ):
        """Run single-point calculation via ``ts.static()``."""
        if ts is None or atoms is None:
            return
        properties = properties or ["energy", "forces"]
        result_list = ts.static(system=atoms, model=self._model)
        out = result_list[0]
        energy = out.get("energy")
        forces = out.get("forces")
        if energy is not None:
            self.results["energy"] = float(energy.detach().cpu().numpy().squeeze())
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
        """Return energy in eV."""
        if atoms is not None and self._atoms_changed(atoms):
            self.calculate(atoms, ["energy", "forces"])
        return self.results.get("energy", 0.0)

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


def setup_single_model(model_name: str = "uma-s-1p1", device: str = "cuda"):
    """Create a single FairChemModel shared by calculator and TorchSim.

    Returns (calculator, ts_model) where calculator wraps ts_model.
    Use this instead of setup_calculator + setup_torchsim_model to reduce GPU memory.
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
    result_list = ts.static(system=atoms_list, model=ts_model)
    out: list[tuple[float, np.ndarray]] = []
    for res in result_list:
        e = res.get("energy")
        f = res.get("forces")
        energy = float(e.detach().cpu().numpy().squeeze()) if e is not None else 0.0
        forces = f.detach().cpu().numpy() if f is not None else np.zeros((0, 3))
        out.append((energy, forces))
    return out


def precompute_results(
    atoms_list: list[Atoms],
    ts_model,
    calculator: "TorchSimCalculator",
) -> None:
    """Run batched ``ts.static()`` and pre-populate each Atoms' calculator cache.

    After this call, ``atoms.get_potential_energy()`` and ``atoms.get_forces()``
    return instantly from the cache without triggering another model forward pass.
    """
    if not atoms_list:
        return
    results = batch_static(atoms_list, ts_model)
    for atoms, (energy, forces) in zip(atoms_list, results, strict=False):
        calc = TorchSimCalculator(calculator._model)
        calc.results["energy"] = energy
        calc.results["forces"] = forces
        calc._last_positions_hash = _positions_cell_hash(atoms)
        atoms.calc = calc


# ---------------------------------------------------------------------------
# isolated molecule optimisation
# ---------------------------------------------------------------------------


def optimize_isolated_molecules_batched(
    conformers: list[Atoms],
    ts_model,
    fmax: float = 0.05,
    steps: int = 100,
    config: AdsorptionConfig | None = None,
) -> list[tuple[Atoms, float] | None]:
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
    conv = ts.generate_force_convergence_fn(force_tol=fmax, include_cell_forces=False)
    _maybe_clear_cuda_cache(ts_model)
    try:
        ab = (
            _get_inflight_autobatcher(
                ts_model, max(len(a) for a in conformers), config=config
            )
            if use_autobatcher
            else None
        )
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
        results = []
        for a, e in zip(atoms_list, energies, strict=False):
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
) -> list[Atoms | None]:
    """Batch-optimise slab+adsorbate systems, selectively freezing sub-surface.

    Uses TorchSim's ``InFlightAutoBatcher`` for GPU-efficient batching.

    The frozen indices are computed via :func:`compute_frozen_indices` using
    the supplied *slab* reference and *config*. When *base_slab_for_frozen*
    is provided (e.g. for sequential saturation), it is used for frozen indices
    instead of *slab*, so only the original surface atoms are frozen.
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

    slab_for_frozen = base_slab_for_frozen if base_slab_for_frozen is not None else slab
    frozen_indices = compute_frozen_indices(slab_for_frozen, config)
    slab_size = len(slab)

    clear_autobatcher_cache(max_n_atoms_threshold=slab_size)

    logger.info(
        "Batched optimisation of %d systems (slab=%d atoms, frozen=%d)...",
        len(combined_atoms_list),
        slab_size,
        len(frozen_indices),
    )

    conv = ts.generate_force_convergence_fn(
        force_tol=config.fmax, include_cell_forces=False
    )
    max_steps = config.stage1_steps + config.stage2_steps
    optimizer = _resolve_ts_optimizer(config.ts_optimizer)
    swaps = config.steps_between_swaps

    model_device = getattr(ts_model, "device", None) or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    sim_states = [
        _make_state_with_frozen_constraint(
            atoms, frozen_indices, ts_model, model_device
        )
        for atoms in combined_atoms_list
    ]

    try:
        max_n_atoms = max(len(a) for a in combined_atoms_list)
        _maybe_clear_cuda_cache(ts_model)
        ab = _get_inflight_autobatcher(ts_model, max_n_atoms, config=config)
        if ab is None:
            raise RuntimeError("Could not create autobatcher")

        batch = ts.optimize(
            system=sim_states,
            model=ts_model,
            optimizer=optimizer,
            convergence_fn=conv,
            max_steps=max_steps,
            steps_between_swaps=swaps,
            autobatcher=ab,
        )
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
        clear_autobatcher_cache()

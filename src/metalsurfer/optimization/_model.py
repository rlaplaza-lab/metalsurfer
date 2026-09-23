"""TorchSim model setup and the ASE-compatible ``TorchSimCalculator`` wrapper."""

import contextlib
import logging
from typing import Any, NoReturn, cast

import numpy as np
import scipy.special as sp_special
from ase import Atoms

from .._logging import torchsim_output_capture
from ..exceptions import DependencyMissingError
from . import _deps
from ._validation import (
    _device_key,
    _positions_cell_hash,
    _resolve_device,
    _validate_model_pbc,
)

logger = logging.getLogger(__name__)


def _ensure_scipy_sph_harm() -> None:
    """Restore ``scipy.special.sph_harm`` for FairChem on SciPy 1.17+.

    FairChem still does ``from scipy.special import sph_harm`` during
    ``setup_imports``. SciPy 1.17 removed that name in favor of ``sph_harm_y``.
    """
    if getattr(sp_special, "sph_harm", None) is not None:
        return
    sph_harm_y = getattr(sp_special, "sph_harm_y", None)
    if sph_harm_y is None:
        return

    def _legacy_sph_harm(
        m: Any, n: Any, theta: Any, phi: Any, *args: Any, **kwargs: Any
    ) -> Any:
        # Legacy sph_harm(m, n, theta_azim, phi_polar) vs
        # sph_harm_y(n, m, theta_polar, phi_azim).
        return sph_harm_y(n, m, phi, theta)

    cast(Any, sp_special).sph_harm = _legacy_sph_harm


def _ensure_torch_checkpoint_safe_globals() -> None:
    """Allow PyTorch 2.6+ to unpickle FairChem checkpoints that reference ``slice``."""
    torch = _deps.torch
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


def _premove_fairchem_predictor_to_device(model: Any, dev: Any) -> None:
    """Pre-move the FairChem predictor weights to ``dev`` before first predict.

    fairchem-core >= 2.20 changed ``MLIPPredictUnit._lazy_init`` to run the
    MOLE merge (``prepare_for_inference``) *before* ``move_to_device``. The
    merge performs embedding lookups with batch indices; when torch-sim
    hands over CUDA-resident batches (torch-sim-atomistic >= 0.6 behavior)
    while the model weights are still on CPU, the first predict call dies
    with "Expected all tensors to be on the same device". Moving the model
    to the target device right after construction closes that window on
    every fairchem version: on 2.11 the pre-move is redundant but harmless
    (``move_to_device`` runs again in ``_lazy_init``), and on >= 2.20 it is
    the difference between the merge working and crashing.
    """
    try:
        predictor = getattr(model, "predictor", None)
        target = getattr(predictor, "device", None)
        inner = getattr(predictor, "model", None)
        if predictor is None or target is None or inner is None:
            return
        # ``dev`` may be torch.device("cuda") while the predictor resolves to
        # "cuda:0"; treat any cuda target as the intended device.
        if str(target) != str(dev) and not str(target).startswith("cuda"):
            return
        inner.to(target)
        # Task normalizers / element references live outside ``model``.
        tasks = getattr(predictor, "tasks", None) or {}
        for task in tasks.values():
            for sub in (
                getattr(task, "normalizer", None),
                getattr(task, "element_references", None),
            ):
                if sub is not None:
                    sub.to(target)
    except (AttributeError, TypeError, RuntimeError) as exc:
        logger.debug("FairChem predictor pre-move skipped: %s", exc)


def setup_torchsim_model(  # pragma: no cover - requires MLIP stack / GPU
    model_name: str = "uma-s-1p2",
    device: str = "cuda",
    task_name: str = "oc25",
):
    """Create a TorchSim FairChemModel wrapper.

    Uses torch-sim-atomistic FairChemModel API: model, device, task_name.

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
    if _deps.ts is None:
        raise DependencyMissingError(
            "torch-sim-atomistic",
            "setup_torchsim_model",
            "Install with: pip install torch-sim-atomistic",
        )
    try:
        from torch_sim.models.fairchem import FairChemModel
    except ImportError as exc:
        raise DependencyMissingError(
            "fairchem",
            "setup_torchsim_model",
            "Install FairChem (e.g. pip install fairchem-core) and ensure "
            "torch-sim-atomistic is built with FairChem support",
        ) from exc

    resolved_device = _resolve_device(device)
    if resolved_device is None:
        raise ValueError("device must be set for TorchSim model initialization")
    device = resolved_device
    _ensure_torch_checkpoint_safe_globals()
    _ensure_scipy_sph_harm()
    logger.info("Initializing TorchSim FairChemModel (%s) on %s", model_name, device)
    torch = _deps.torch
    dev = torch.device(device)
    try:
        with torchsim_output_capture():
            model = cast(Any, FairChemModel)(
                model=model_name, device=dev, task_name=task_name
            )
            _premove_fairchem_predictor_to_device(model, dev)
    except DependencyMissingError:
        raise
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

    def __init__(
        self,
        ts_model: Any,
        *,
        model_name: str | None = None,
        device: str | None = None,
        task_name: str | None = None,
    ) -> None:
        """Wrap a TorchSim model (e.g. FairChemModel) for ASE compatibility.

        Parameters
        ----------
        ts_model
            TorchSim model instance.
        model_name
            FairChem checkpoint name used to build *ts_model*, if known.
        device
            Resolved device string used at construction, if known.
        task_name
            FairChem task head used at construction, if known.
        """
        self._model = ts_model
        self._model_name = model_name
        self._device = device
        self._task_name = task_name
        self.results: dict[str, Any] = {}
        self._last_positions_hash: int | None = None

    def matches_setup(self, model_name: str, device: str, task_name: str) -> bool:
        """Return whether this calculator was built for the given setup args."""
        if self._model is None or self._model_name is None or self._task_name is None:
            return False
        if self._model_name != model_name or self._task_name != task_name:
            return False
        resolved = _resolve_device(device)
        if resolved is None or self._device is None:
            return False
        return _device_key(self._device) == _device_key(resolved)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        *,
        positions_hash: int | None = None,
    ) -> None:
        """Run single-point calculation via ``ts.static()``.

        Parameters
        ----------
        atoms
            ASE Atoms object.
        properties
            List of requested properties (e.g. ["energy", "forces"]).
        positions_hash
            Optional precomputed geometry hash; skips a second hash when the
            caller already computed one for cache invalidation.
        """
        if atoms is None:
            return
        ts = _deps.ts
        if ts is None:
            raise DependencyMissingError(
                "torch-sim-atomistic",
                "TorchSimCalculator.calculate",
                "Install with: pip install torch-sim-atomistic",
            )
        self.results = {}
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
        self._last_positions_hash = (
            positions_hash
            if positions_hash is not None
            else _positions_cell_hash(atoms)
        )

    def _geometry_hash_if_changed(self, atoms) -> tuple[bool, int | None]:
        """Return ``(changed, hash)`` for *atoms* relative to the last calculate."""
        if atoms is None:
            return True, None
        current = _positions_cell_hash(atoms)
        if self._last_positions_hash is None or current != self._last_positions_hash:
            return True, current
        return False, current

    def get_potential_energy(self, atoms=None, force_consistent=False):
        """Return energy in eV.

        ``force_consistent`` is accepted for ASE compatibility but ignored.

        Parameters
        ----------
        atoms
            ASE Atoms object.
        force_consistent
            Accepted for ASE compatibility but ignored.
        """
        _ = force_consistent
        if atoms is not None:
            changed, positions_hash = self._geometry_hash_if_changed(atoms)
            if changed or "energy" not in self.results:
                self.calculate(
                    atoms, ["energy", "forces"], positions_hash=positions_hash
                )
        energy = self.results.get("energy")
        if energy is None or not np.isfinite(energy):
            raise RuntimeError(
                f"Calculator has no valid energy (got {energy}). "
                "The model may have failed to produce energy for this system."
            )
        return energy

    def get_forces(self, atoms=None):
        """Return forces in eV/Å, shape (n_atoms, 3).

        Parameters
        ----------
        atoms
            ASE Atoms object.
        """
        if atoms is not None:
            changed, positions_hash = self._geometry_hash_if_changed(atoms)
            if changed or "forces" not in self.results:
                self.calculate(
                    atoms, ["energy", "forces"], positions_hash=positions_hash
                )
        forces = self.results.get("forces")
        if forces is None:
            n = len(atoms) if atoms is not None else 0
            raise RuntimeError(
                f"Calculator has no forces (expected shape ({n}, 3)). "
                "The model may have failed to produce forces for this system."
            )
        return forces

    def get_stress(self, atoms=None):
        """Return stress in Voigt order (xx, yy, zz, yz, xz, xy).

        Parameters
        ----------
        atoms
            ASE Atoms object.
        """
        if atoms is not None:
            changed, positions_hash = self._geometry_hash_if_changed(atoms)
            if changed or "stress" not in self.results:
                self.calculate(
                    atoms,
                    ["energy", "forces", "stress"],
                    positions_hash=positions_hash,
                )
        stress = self.results.get("stress")
        if stress is None:
            raise RuntimeError(
                "Calculator has no stress. "
                "The model may have failed to produce stress for this system."
            )
        return stress


def _voigt_6(stress_3x3) -> np.ndarray:
    """Convert 3x3 stress to Voigt 6-component form."""
    s = np.asarray(stress_3x3).reshape(3, 3)
    return np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

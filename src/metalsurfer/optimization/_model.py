"""TorchSim model setup and the ASE-compatible ``TorchSimCalculator`` wrapper."""

import contextlib
import logging
from typing import Any, NoReturn, cast

import numpy as np
from ase import Atoms

from .._logging import torchsim_output_capture
from ..exceptions import DependencyMissingError
from . import _deps
from ._validation import _positions_cell_hash, _resolve_device, _validate_model_pbc

logger = logging.getLogger(__name__)


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
    logger.info("Initializing TorchSim FairChemModel (%s) on %s", model_name, device)
    torch = _deps.torch
    dev = torch.device(device)
    try:
        with torchsim_output_capture():
            model = cast(Any, FairChemModel)(
                model=model_name, device=dev, task_name=task_name
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

    def __init__(self, ts_model: Any) -> None:
        """Wrap a TorchSim model (e.g. FairChemModel) for ASE compatibility.

        Parameters
        ----------
        ts_model
            TorchSim model instance.
        """
        self._model = ts_model
        self.results: dict[str, Any] = {}
        self._last_positions_hash: int | None = None

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
    ) -> None:
        """Run single-point calculation via ``ts.static()``.

        Parameters
        ----------
        atoms
            ASE Atoms object.
        properties
            List of requested properties (e.g. ["energy", "forces"]).
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
        self._last_positions_hash = _positions_cell_hash(atoms)

    def _atoms_changed(self, atoms) -> bool:
        """Check whether positions/cell/species changed since the last calculation."""
        if atoms is None or self._last_positions_hash is None:
            return True
        return _positions_cell_hash(atoms) != self._last_positions_hash

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
        if atoms is not None and (
            self._atoms_changed(atoms) or "energy" not in self.results
        ):
            self.calculate(atoms, ["energy", "forces"])
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
        if atoms is not None and (
            self._atoms_changed(atoms) or "forces" not in self.results
        ):
            self.calculate(atoms, ["energy", "forces"])
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
        if atoms is not None and (
            self._atoms_changed(atoms) or "stress" not in self.results
        ):
            self.calculate(atoms, ["energy", "forces", "stress"])
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

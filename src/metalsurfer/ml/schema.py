"""Canonical descriptor schema for reproducible binding energy records.

Every field needed to reproduce a binding energy computation from scratch
is captured in :class:`PlacementRecord`.  The :class:`ComputationContext`
stores method/calculator settings so that identical descriptors can be
distinguished when computed with different levels of theory.

Units convention:
    - Coordinates: Angstrom
    - Angles: degrees
    - Energies: eV
    - Forces: eV/Angstrom
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..config import AdsorptionConfig
from ..models import ScreeningResult

logger = logging.getLogger(__name__)


@dataclass
class ComputationContext:
    """Calculator and method settings that affect binding energy values.

    Stored alongside placement geometry so that results computed with
    different models or convergence criteria are distinguishable.
    """

    model_name: str = "uma-s-1p1"
    fmax: float = 0.05
    stage1_steps: int = 50
    stage2_steps: int = 150
    device: str = "cuda"
    seed: int = 42
    placement_z_range: tuple[float, float] = (2.0, 3.0)
    placement_z_scale_by_covalent_radius: bool = True
    min_initial_distance: float = 1.5
    min_contact_ratio: float = 0.8
    top_layer_tolerance: float = 0.5
    relax_top_layer: bool = True

    @classmethod
    def from_config(cls, config: AdsorptionConfig) -> "ComputationContext":
        return cls(
            model_name=config.model_name,
            fmax=config.fmax,
            stage1_steps=config.stage1_steps,
            stage2_steps=config.stage2_steps,
            device=config.device,
            seed=config.seed,
            placement_z_range=config.placement_z_range,
            placement_z_scale_by_covalent_radius=config.placement_z_scale_by_covalent_radius,
            min_initial_distance=config.min_initial_distance,
            min_contact_ratio=config.min_contact_ratio,
            top_layer_tolerance=config.top_layer_tolerance,
            relax_top_layer=config.relax_top_layer,
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["placement_z_range"] = list(d["placement_z_range"])
        return d

    def settings_hash(self) -> str:
        """Deterministic hash of computation settings (first 12 hex chars)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def config_to_context_row(config: AdsorptionConfig) -> dict[str, Any]:
    """Build a flat dict of computation context fields for CSV / reproducibility.

    Use this when writing detailed or saturation CSVs so each row carries
    the settings used to generate the pose (model_name, placement_mode,
    fmax, stage1_steps, stage2_steps, seed, etc.) and is compatible with
    PlacementRecord.from_flat_dict() for ML.
    """
    ctx = ComputationContext.from_config(config)
    row: dict[str, Any] = ctx.to_dict()
    row["placement_mode"] = config.placement_mode
    row["reference_optimization_steps"] = config.reference_optimization_steps
    row["context_hash"] = ctx.settings_hash()
    return row


@dataclass
class PlacementRecord:
    """Complete record tying placement descriptors to a binding energy result.

    This is the fundamental row in the ML dataset.  Every field needed
    to (a) reproduce the computation and (b) train a regression model
    is present.
    """

    # --- Identity ---
    molecule: str
    smiles: str
    surface_id: str
    placement_id: int

    # --- Placement geometry (from PlacementDescriptor) ---
    conformer_index: int
    orientation_type: Literal["parallel", "EN-down", "vertical", "round"]
    face_flip: bool
    en_atom_index: int | None
    site_index: int
    site_type: str | None
    tilt_deg: float
    azimuth_deg: float
    azimuth_in_plane_deg: float
    z_fraction: float
    x: float
    y: float
    z: float
    shape: str
    slab_indices: tuple[int, ...] | None = None

    # --- Energies (eV) ---
    energy_adsorption: float = 0.0
    energy_adslab: float = 0.0
    energy_slab: float = 0.0
    energy_adsorbate: float = 0.0

    # --- Post-optimization metadata ---
    distance: float = 0.0
    converged: bool = True

    # --- Computation context ---
    context: ComputationContext = field(default_factory=ComputationContext)

    @classmethod
    def from_screening_result(
        cls,
        result: ScreeningResult,
        smiles: str,
        surface_id: str,
        config: AdsorptionConfig | None = None,
    ) -> "PlacementRecord":
        """Build a record from a validated ScreeningResult."""
        d = result.placement_descriptor
        ctx = (
            ComputationContext.from_config(config)
            if config is not None
            else ComputationContext()
        )

        return cls(
            molecule=result.molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=result.placement_id,
            conformer_index=d.conformer_index,
            orientation_type=d.orientation_type,
            face_flip=d.face_flip,
            en_atom_index=d.en_atom_index,
            site_index=d.site_index,
            site_type=d.site_type,
            tilt_deg=d.tilt_deg,
            azimuth_deg=d.azimuth_deg,
            azimuth_in_plane_deg=d.azimuth_in_plane_deg,
            z_fraction=d.z_fraction,
            x=d.x,
            y=d.y,
            z=d.z,
            shape=d.shape,
            slab_indices=d.slab_indices,
            energy_adsorption=result.energy_adsorption,
            energy_adslab=result.energy_adslab,
            energy_slab=result.energy_slab,
            energy_adsorbate=result.energy_adsorbate,
            distance=result.distance,
            converged=True,
            context=ctx,
        )

    def record_hash(self) -> str:
        """Deterministic hash uniquely identifying this placement + method combination.

        Combines placement geometry, identity, and computation settings.
        """
        key_dict = {
            "molecule": str(self.molecule),
            "smiles": str(self.smiles),
            "surface_id": str(self.surface_id),
            "placement_id": int(self.placement_id),
            "conformer_index": int(self.conformer_index),
            "orientation_type": str(self.orientation_type),
            "face_flip": bool(self.face_flip),
            "en_atom_index": int(self.en_atom_index)
            if self.en_atom_index is not None
            else None,
            "site_index": int(self.site_index),
            "site_type": str(self.site_type) if self.site_type is not None else None,
            "tilt_deg": round(float(self.tilt_deg), 6),
            "azimuth_deg": round(float(self.azimuth_deg), 6),
            "azimuth_in_plane_deg": round(float(self.azimuth_in_plane_deg), 6),
            "z_fraction": round(float(self.z_fraction), 6),
            "x": round(float(self.x), 6),
            "y": round(float(self.y), 6),
            "z": round(float(self.z), 6),
            "context_hash": self.context.settings_hash(),
        }
        blob = json.dumps(key_dict, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten into a single-level dict suitable for DataFrame rows."""
        row: dict[str, Any] = {
            "record_hash": self.record_hash(),
            "molecule": self.molecule,
            "smiles": self.smiles,
            "surface_id": self.surface_id,
            "placement_id": self.placement_id,
            "conformer_index": self.conformer_index,
            "orientation_type": self.orientation_type,
            "face_flip": self.face_flip,
            "en_atom_index": self.en_atom_index,
            "site_index": self.site_index,
            "site_type": self.site_type,
            "tilt_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "azimuth_in_plane_deg": self.azimuth_in_plane_deg,
            "z_fraction": self.z_fraction,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "shape": self.shape,
            "slab_indices": (
                ",".join(str(i) for i in self.slab_indices)
                if self.slab_indices is not None
                else None
            ),
            "energy_adsorption": self.energy_adsorption,
            "energy_adslab": self.energy_adslab,
            "energy_slab": self.energy_slab,
            "energy_adsorbate": self.energy_adsorbate,
            "distance": self.distance,
            "converged": self.converged,
            "model_name": self.context.model_name,
            "fmax": self.context.fmax,
            "stage1_steps": self.context.stage1_steps,
            "stage2_steps": self.context.stage2_steps,
            "seed": self.context.seed,
            "context_hash": self.context.settings_hash(),
        }
        return row

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> "PlacementRecord":
        """Reconstruct a PlacementRecord from a flattened dict (e.g. CSV row)."""
        slab_indices_raw = row.get("slab_indices")
        slab_indices = None
        if slab_indices_raw and str(slab_indices_raw) != "nan":
            slab_indices = tuple(int(x) for x in str(slab_indices_raw).split(","))

        ctx = ComputationContext(
            model_name=str(row.get("model_name", "uma-s-1p1")),
            fmax=float(row.get("fmax", 0.05)),
            stage1_steps=int(row.get("stage1_steps", 50)),
            stage2_steps=int(row.get("stage2_steps", 150)),
            seed=int(row.get("seed", 42)),
        )

        en_atom_index_raw = row.get("en_atom_index")
        en_atom_index = None
        if en_atom_index_raw is not None and str(en_atom_index_raw) != "nan":
            en_atom_index = int(en_atom_index_raw)

        return cls(
            molecule=str(row["molecule"]),
            smiles=str(row.get("smiles", "")),
            surface_id=str(row.get("surface_id", "")),
            placement_id=int(row["placement_id"]),
            conformer_index=int(row["conformer_index"]),
            orientation_type=row["orientation_type"],
            face_flip=bool(row.get("face_flip", False)),
            en_atom_index=en_atom_index,
            site_index=int(row["site_index"]),
            site_type=row.get("site_type"),
            tilt_deg=float(row["tilt_deg"]),
            azimuth_deg=float(row["azimuth_deg"]),
            azimuth_in_plane_deg=float(row["azimuth_in_plane_deg"]),
            z_fraction=float(row.get("z_fraction", 0.5)),
            x=float(row["x"]),
            y=float(row["y"]),
            z=float(row["z"]),
            shape=str(row.get("shape", "round")),
            slab_indices=slab_indices,
            energy_adsorption=float(row.get("energy_adsorption", 0.0)),
            energy_adslab=float(row.get("energy_adslab", 0.0)),
            energy_slab=float(row.get("energy_slab", 0.0)),
            energy_adsorbate=float(row.get("energy_adsorbate", 0.0)),
            distance=float(row.get("distance", 0.0)),
            converged=bool(row.get("converged", True)),
            context=ctx,
        )

"""`PlacementRecord` and `ComputationContext` for binding-energy rows.

Units: Å, degrees, eV, eV/Å (forces).
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, cast

import numpy as np

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor, PlacementSpec, ScreeningResult

logger = logging.getLogger(__name__)
SCHEMA_VERSION = "2.2"


def _is_missing(value: Any) -> bool:
    return value is None or str(value) == "nan"


def _with_default(value: Any, default: Any) -> Any:
    return default if _is_missing(value) else value


def _float_or(value: Any, default: float) -> float:
    return float(_with_default(value, default))


def _int_or_none(value: Any) -> int | None:
    if _is_missing(value):
        return None
    return int(value)


def _parse_bool(value: Any, default: bool = False) -> bool:
    if _is_missing(value):
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return default


def _parse_float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if _is_missing(value):
        return default
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return float(value[0]), float(value[1])
    text = str(value).strip().strip("[]()")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        return float(parts[0]), float(parts[1])
    return default


def _context_from_config(config: AdsorptionConfig | None) -> "ComputationContext":
    return (
        ComputationContext.from_config(config)
        if config is not None
        else ComputationContext()
    )


def _descriptor_geometry_values(
    descriptor: PlacementDescriptor,
) -> dict[str, float | int | bool | str | tuple[int, ...] | None]:
    surface_ref_z_abs = (
        float(descriptor.surface_ref_z_abs)
        if descriptor.surface_ref_z_abs is not None
        else 0.0
    )
    z_offset = float(descriptor.z_offset)
    z_abs = (
        float(descriptor.z_abs)
        if descriptor.z_abs is not None
        else surface_ref_z_abs + z_offset
    )
    return {
        "conformer_index": descriptor.conformer_index,
        "orientation_type": descriptor.orientation_type,
        "face_flip": descriptor.face_flip,
        "en_atom_index": descriptor.en_atom_index,
        "site_index": descriptor.site_index,
        "site_type": descriptor.site_type,
        "tilt_deg": descriptor.tilt_deg,
        "azimuth_deg": descriptor.azimuth_deg,
        "azimuth_in_plane_deg": descriptor.azimuth_in_plane_deg,
        "z_fraction": descriptor.z_fraction,
        "x_abs": float(descriptor.x_abs)
        if descriptor.x_abs is not None
        else float(descriptor.x),
        "y_abs": float(descriptor.y_abs)
        if descriptor.y_abs is not None
        else float(descriptor.y),
        "z_offset": z_offset,
        "surface_ref_z_abs": surface_ref_z_abs,
        "z_abs": z_abs,
        "x": float(descriptor.x),
        "y": float(descriptor.y),
        "shape": descriptor.shape,
        "slab_indices": descriptor.slab_indices,
        "placement_mode_resolved": descriptor.placement_mode_resolved,
        "site_source": descriptor.site_source,
        "site_reference_frame": descriptor.site_reference_frame,
        "site_xy_frac_a": float(descriptor.site_xy_frac_a or 0.0),
        "site_xy_frac_b": float(descriptor.site_xy_frac_b or 0.0),
        "quat_w": float(descriptor.quat_w) if descriptor.quat_w is not None else 1.0,
        "quat_x": float(descriptor.quat_x) if descriptor.quat_x is not None else 0.0,
        "quat_y": float(descriptor.quat_y) if descriptor.quat_y is not None else 0.0,
        "quat_z": float(descriptor.quat_z) if descriptor.quat_z is not None else 0.0,
    }


@dataclass
class ComputationContext:
    """Calculator and method settings that affect binding energy values.

    Stored alongside placement geometry so that results computed with
    different models or convergence criteria are distinguishable.
    """

    model_name: str = "uma-s-1p2"
    fmax: float = 0.05
    stage1_steps: int = 50
    stage2_steps: int = 150
    device: str = "cuda"
    seed: int = 42
    placement_z_range: tuple[float, float] = (0.7, 1.25)
    placement_z_scale_by_covalent_radius: bool = True
    min_initial_distance: float = 1.5
    min_contact_ratio: float = 0.8
    top_layer_tolerance: float = 0.5
    symmetry_tolerance: float = 0.1
    site_equivalence_tolerance: float = 0.05
    hollow_site_dedup_tolerance: float = 0.1
    planar_z_variance_threshold: float = 0.01

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
            symmetry_tolerance=config.symmetry_tolerance,
            site_equivalence_tolerance=config.site_equivalence_tolerance,
            hollow_site_dedup_tolerance=config.hollow_site_dedup_tolerance,
            planar_z_variance_threshold=config.planar_z_variance_threshold,
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
    the settings used to generate the pose (model_name,
    fmax, stage1_steps, stage2_steps, seed, etc.) and is compatible with
    PlacementRecord.from_flat_dict() for ML.
    """
    ctx = ComputationContext.from_config(config)
    row: dict[str, Any] = {f"ctx_{k}": v for k, v in ctx.to_dict().items()}
    row.update(ctx.to_dict())
    row["reference_optimization_steps"] = config.reference_optimization_steps
    row["context_hash"] = ctx.settings_hash()
    row["schema_version"] = SCHEMA_VERSION
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
    orientation_type: Literal[
        "parallel", "EN-down", "vertical", "round", "dissociative"
    ]
    face_flip: bool
    en_atom_index: int | None
    site_index: int
    site_type: str | None
    tilt_deg: float
    azimuth_deg: float
    azimuth_in_plane_deg: float
    z_fraction: float
    x_abs: float = 0.0
    y_abs: float = 0.0
    z_offset: float = 0.0
    surface_ref_z_abs: float = 0.0
    z_abs: float = 0.0
    x: float = 0.0
    y: float = 0.0
    shape: str = "round"
    slab_indices: tuple[int, ...] | None = None
    placement_mode_resolved: str = "no_sites"
    site_source: str = "no_sites"
    site_reference_frame: str = "global_top_layer"
    site_xy_frac_a: float = 0.0
    site_xy_frac_b: float = 0.0
    quat_w: float = 1.0
    quat_x: float = 0.0
    quat_y: float = 0.0
    quat_z: float = 0.0

    # --- Energies (eV) ---
    energy_adsorption: float = 0.0
    energy_adslab: float = 0.0
    energy_slab: float = 0.0
    energy_adsorbate: float = 0.0

    # --- Post-optimization metadata ---
    distance: float = 0.0
    converged: bool = True
    failure_stage: str | None = None
    failure_reason: str | None = None
    is_penalty_label: bool = False
    label_source: str = "observed"

    # --- Computation context ---
    context: ComputationContext = field(default_factory=ComputationContext)

    def __post_init__(self) -> None:
        """Canonicalize quaternion representation for deterministic records."""
        from ..placement.geometry import normalize_quaternion

        q = normalize_quaternion(
            np.array(
                [
                    float(self.quat_w),
                    float(self.quat_x),
                    float(self.quat_y),
                    float(self.quat_z),
                ],
                dtype=float,
            )
        )
        self.quat_w, self.quat_x, self.quat_y, self.quat_z = (
            float(q[0]),
            float(q[1]),
            float(q[2]),
            float(q[3]),
        )

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
        geometry = cast(Any, _descriptor_geometry_values(d))
        return cls(
            molecule=result.molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=result.placement_id,
            **geometry,
            energy_adsorption=result.energy_adsorption,
            energy_adslab=result.energy_adslab,
            energy_slab=result.energy_slab,
            energy_adsorbate=result.energy_adsorbate,
            distance=result.distance,
            converged=True,
            failure_stage=None,
            failure_reason=None,
            is_penalty_label=False,
            label_source="observed",
            context=_context_from_config(config),
        )

    @classmethod
    def from_descriptor(
        cls,
        descriptor: PlacementDescriptor,
        *,
        molecule: str = "",
        smiles: str = "",
        surface_id: str = "",
        config: AdsorptionConfig | None = None,
        placement_id: int | None = None,
    ) -> "PlacementRecord":
        return cls(
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=descriptor.placement_index
            if placement_id is None
            else placement_id,
            **cast(Any, _descriptor_geometry_values(descriptor)),
            context=_context_from_config(config),
        )

    @classmethod
    def from_spec(
        cls,
        spec: PlacementSpec,
        *,
        molecule: str = "",
        smiles: str = "",
        surface_id: str = "",
        config: AdsorptionConfig | None = None,
    ) -> "PlacementRecord":
        return cls(
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=spec.placement_index,
            conformer_index=spec.conformer_index,
            orientation_type=spec.orientation_type,
            face_flip=spec.face_flip,
            en_atom_index=spec.en_atom_index,
            site_index=spec.site_index,
            site_type=spec.site_type,
            tilt_deg=spec.tilt_deg,
            azimuth_deg=spec.azimuth_deg,
            azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
            z_fraction=spec.z_fraction,
            shape="round",
            placement_mode_resolved="sites" if spec.site_index >= 0 else "no_sites",
            site_source="spec_placeholder",
            site_reference_frame="global_top_layer",
            context=_context_from_config(config),
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
            "x_abs": round(float(self.x_abs), 6),
            "y_abs": round(float(self.y_abs), 6),
            "z_offset": round(float(self.z_offset), 6),
            "surface_ref_z_abs": round(float(self.surface_ref_z_abs), 6),
            "z_abs": round(float(self.z_abs), 6),
            "quat_w": round(float(self.quat_w), 6),
            "quat_x": round(float(self.quat_x), 6),
            "quat_y": round(float(self.quat_y), 6),
            "quat_z": round(float(self.quat_z), 6),
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
            "x_abs": self.x_abs,
            "y_abs": self.y_abs,
            "z_offset": self.z_offset,
            "surface_ref_z_abs": self.surface_ref_z_abs,
            "z_abs": self.z_abs,
            "x": self.x,
            "y": self.y,
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
            "failure_stage": self.failure_stage,
            "failure_reason": self.failure_reason,
            "is_penalty_label": self.is_penalty_label,
            "label_source": self.label_source,
            "placement_mode_resolved": self.placement_mode_resolved,
            "site_source": self.site_source,
            "site_reference_frame": self.site_reference_frame,
            "site_xy_frac_a": self.site_xy_frac_a,
            "site_xy_frac_b": self.site_xy_frac_b,
            "quat_w": self.quat_w,
            "quat_x": self.quat_x,
            "quat_y": self.quat_y,
            "quat_z": self.quat_z,
            "schema_version": SCHEMA_VERSION,
            "model_name": self.context.model_name,
            "fmax": self.context.fmax,
            "stage1_steps": self.context.stage1_steps,
            "stage2_steps": self.context.stage2_steps,
            "seed": self.context.seed,
            "context_hash": self.context.settings_hash(),
        }
        for key, value in self.context.to_dict().items():
            row[f"ctx_{key}"] = value
        return row

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> "PlacementRecord":
        """Reconstruct a PlacementRecord from a flattened dict (e.g. CSV row)."""

        def _ctx_value(name: str, default: Any) -> Any:
            return _with_default(
                row.get(f"ctx_{name}"), _with_default(row.get(name), default)
            )

        slab_indices_raw = row.get("slab_indices")
        slab_indices = None
        if slab_indices_raw and not _is_missing(slab_indices_raw):
            slab_indices = tuple(int(x) for x in str(slab_indices_raw).split(","))

        ctx = ComputationContext(
            model_name=str(_ctx_value("model_name", "uma-s-1p2")),
            fmax=float(_ctx_value("fmax", 0.05)),
            stage1_steps=int(_ctx_value("stage1_steps", 50)),
            stage2_steps=int(_ctx_value("stage2_steps", 150)),
            device=str(_ctx_value("device", "cuda")),
            seed=int(_ctx_value("seed", 42)),
            placement_z_range=_parse_float_pair(
                _ctx_value("placement_z_range", [0.7, 1.25]),
                default=(0.7, 1.25),
            ),
            placement_z_scale_by_covalent_radius=_parse_bool(
                _ctx_value("placement_z_scale_by_covalent_radius", True),
                default=True,
            ),
            min_initial_distance=float(_ctx_value("min_initial_distance", 1.5)),
            min_contact_ratio=float(_ctx_value("min_contact_ratio", 0.8)),
            top_layer_tolerance=float(_ctx_value("top_layer_tolerance", 0.5)),
            symmetry_tolerance=float(_ctx_value("symmetry_tolerance", 0.1)),
            site_equivalence_tolerance=float(
                _ctx_value("site_equivalence_tolerance", 0.05)
            ),
            hollow_site_dedup_tolerance=float(
                _ctx_value("hollow_site_dedup_tolerance", 0.1)
            ),
            planar_z_variance_threshold=float(
                _ctx_value("planar_z_variance_threshold", 0.01)
            ),
        )

        en_atom_index = _int_or_none(row.get("en_atom_index"))
        tilt_deg = float(row["tilt_deg"])
        azimuth_deg = float(row["azimuth_deg"])
        azimuth_in_plane_deg = float(row["azimuth_in_plane_deg"])
        q_w = _float_or(row.get("quat_w"), 1.0)
        q_x = _float_or(row.get("quat_x"), 0.0)
        q_y = _float_or(row.get("quat_y"), 0.0)
        q_z = _float_or(row.get("quat_z"), 0.0)

        return cls(
            molecule=str(row["molecule"]),
            smiles=str(row.get("smiles", "")),
            surface_id=str(row.get("surface_id", "")),
            placement_id=int(row["placement_id"]),
            conformer_index=int(row["conformer_index"]),
            orientation_type=row["orientation_type"],
            face_flip=_parse_bool(row.get("face_flip", False), default=False),
            en_atom_index=en_atom_index,
            site_index=int(row["site_index"]),
            site_type=row.get("site_type"),
            tilt_deg=tilt_deg,
            azimuth_deg=azimuth_deg,
            azimuth_in_plane_deg=azimuth_in_plane_deg,
            z_fraction=float(row.get("z_fraction", 0.5)),
            x_abs=_float_or(row.get("x_abs"), 0.0),
            y_abs=_float_or(row.get("y_abs"), 0.0),
            z_offset=_float_or(row.get("z_offset"), _float_or(row.get("z"), 0.0)),
            surface_ref_z_abs=_float_or(row.get("surface_ref_z_abs"), 0.0),
            z_abs=float(
                _with_default(
                    row.get("z_abs"),
                    _float_or(row.get("surface_ref_z_abs"), 0.0)
                    + _float_or(row.get("z_offset"), 0.0),
                )
            ),
            x=_float_or(row.get("x"), 0.0),
            y=_float_or(row.get("y"), 0.0),
            shape=str(row.get("shape", "round")),
            slab_indices=slab_indices,
            placement_mode_resolved=str(row.get("placement_mode_resolved", "no_sites")),
            site_source=str(row.get("site_source", "no_sites")),
            site_reference_frame=str(
                row.get("site_reference_frame", "global_top_layer")
            ),
            site_xy_frac_a=float(row.get("site_xy_frac_a", 0.0)),
            site_xy_frac_b=float(row.get("site_xy_frac_b", 0.0)),
            quat_w=q_w,
            quat_x=q_x,
            quat_y=q_y,
            quat_z=q_z,
            energy_adsorption=float(row.get("energy_adsorption", 0.0)),
            energy_adslab=float(row.get("energy_adslab", 0.0)),
            energy_slab=float(row.get("energy_slab", 0.0)),
            energy_adsorbate=float(row.get("energy_adsorbate", 0.0)),
            distance=float(row.get("distance", 0.0)),
            converged=_parse_bool(row.get("converged", True), default=True),
            failure_stage=(
                str(row.get("failure_stage"))
                if not _is_missing(row.get("failure_stage"))
                else None
            ),
            failure_reason=(
                str(row.get("failure_reason"))
                if not _is_missing(row.get("failure_reason"))
                else None
            ),
            is_penalty_label=_parse_bool(
                row.get("is_penalty_label", False), default=False
            ),
            label_source=str(row.get("label_source", "observed")),
            context=ctx,
        )

"""`PlacementRecord` and `ComputationContext` for binding-energy rows.

Units: Å, degrees, eV, eV/Å (forces).
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .._csv_coerce import (
    is_missing as _is_missing,
)
from .._csv_coerce import (
    parse_bool as _parse_bool,
)
from .._csv_coerce import (
    parse_float_pair as _parse_float_pair,
)
from .._csv_coerce import (
    with_default as _with_default,
)
from .._numeric_defaults import (
    DEFAULT_FMAX,
    DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    DEFAULT_SEED,
    DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    DEFAULT_SYMMETRY_TOLERANCE,
    MIN_CONTACT_RATIO_DEFAULT,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..models import (
    PlacementDescriptor,
    PlacementSpec,
    ScreeningResult,
)
from ..placement.geometry import normalize_quaternion

SCHEMA_VERSION = "3.0"


def _context_from_config(config: AdsorptionConfig | None) -> "ComputationContext":
    return (
        ComputationContext.from_config(config)
        if config is not None
        else ComputationContext()
    )


def _normalized_descriptor(
    descriptor: PlacementDescriptor,
    *,
    placement_index: int | None = None,
) -> PlacementDescriptor:
    """Fill optional absolute-pose fields with the same defaults as CSV lean rows."""
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
    return PlacementDescriptor(
        conformer_index=descriptor.conformer_index,
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
        z_fraction=descriptor.z_fraction,
        placement_index=(
            descriptor.placement_index if placement_index is None else placement_index
        ),
        x=float(descriptor.x),
        y=float(descriptor.y),
        z_offset=z_offset,
        x_abs=(
            float(descriptor.x_abs)
            if descriptor.x_abs is not None
            else float(descriptor.x)
        ),
        y_abs=(
            float(descriptor.y_abs)
            if descriptor.y_abs is not None
            else float(descriptor.y)
        ),
        surface_ref_z_abs=surface_ref_z_abs,
        z_abs=z_abs,
        shape=descriptor.shape,
        slab_indices=descriptor.slab_indices,
        placement_mode_resolved=descriptor.placement_mode_resolved,
        site_source=descriptor.site_source,
        site_reference_frame=descriptor.site_reference_frame,
        site_xy_frac_a=float(descriptor.site_xy_frac_a or 0.0),
        site_xy_frac_b=float(descriptor.site_xy_frac_b or 0.0),
        quat_w=float(descriptor.quat_w) if descriptor.quat_w is not None else 1.0,
        quat_x=float(descriptor.quat_x) if descriptor.quat_x is not None else 0.0,
        quat_y=float(descriptor.quat_y) if descriptor.quat_y is not None else 0.0,
        quat_z=float(descriptor.quat_z) if descriptor.quat_z is not None else 0.0,
        fragment_positions=descriptor.fragment_positions,
    )


def _descriptor_from_spec(spec: PlacementSpec) -> PlacementDescriptor:
    return PlacementDescriptor(
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
        placement_index=spec.placement_index,
        x=0.0,
        y=0.0,
        z_offset=0.0,
        x_abs=0.0,
        y_abs=0.0,
        surface_ref_z_abs=0.0,
        z_abs=0.0,
        shape="round",
        placement_mode_resolved="sites" if spec.site_index >= 0 else "no_sites",
        site_source="spec_placeholder",
        site_reference_frame="global_top_layer",
        site_xy_frac_a=0.0,
        site_xy_frac_b=0.0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
    )


@dataclass
class ComputationContext:
    """Calculator and method settings that affect binding energy values.

    Stored alongside placement geometry so that results computed with
    different models or convergence criteria are distinguishable.
    """

    model_name: str = "uma-s-1p2"
    fmax: float = DEFAULT_FMAX
    stage1_steps: int = 50
    stage2_steps: int = 150
    device: str = "cuda"
    seed: int = DEFAULT_SEED
    placement_z_range: tuple[float, float] = (0.7, 1.25)
    placement_z_scale_by_covalent_radius: bool = True
    min_initial_distance: float = MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    min_contact_ratio: float = MIN_CONTACT_RATIO_DEFAULT
    top_layer_tolerance: float = 0.5
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE
    site_equivalence_tolerance: float = DEFAULT_SITE_EQUIVALENCE_TOLERANCE
    hollow_site_dedup_tolerance: float = DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
    planar_z_variance_threshold: float = DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD

    @classmethod
    def from_config(cls, config: AdsorptionConfig) -> "ComputationContext":
        """Build a computation context from an AdsorptionConfig.

        Parameters
        ----------
        config
            Adsorption configuration to translate.

        Returns
        -------
        ComputationContext
        """
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
        """Serialize to a plain dictionary.

        Returns
        -------
        dict[str, Any]
        """
        d = asdict(self)
        d["placement_z_range"] = list(d["placement_z_range"])
        return d

    def settings_hash(self) -> str:
        """Deterministic hash of computation settings (first 12 hex chars)."""
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def to_config(self) -> AdsorptionConfig:
        """Build an AdsorptionConfig matching this computation context."""
        cfg = AdsorptionConfig()
        cfg.model_name = self.model_name
        cfg.fmax = self.fmax
        cfg.stage1_steps = self.stage1_steps
        cfg.stage2_steps = self.stage2_steps
        cfg.device = self.device
        cfg.seed = self.seed
        cfg.placement_z_range = self.placement_z_range
        cfg.placement_z_scale_by_covalent_radius = (
            self.placement_z_scale_by_covalent_radius
        )
        cfg.min_initial_distance = self.min_initial_distance
        cfg.min_contact_ratio = self.min_contact_ratio
        cfg.top_layer_tolerance = self.top_layer_tolerance
        cfg.symmetry_tolerance = self.symmetry_tolerance
        cfg.site_equivalence_tolerance = self.site_equivalence_tolerance
        cfg.hollow_site_dedup_tolerance = self.hollow_site_dedup_tolerance
        cfg.planar_z_variance_threshold = self.planar_z_variance_threshold
        return cfg


def config_to_context_row(
    config: AdsorptionConfig,
    *,
    include_provenance: bool | None = None,
) -> dict[str, Any]:
    """Build computation-context fields for CSV / reproducibility.

    Lean default (``include_provenance=False`` / config default): ``context_hash``
    and ``schema_version`` only. Rich mode adds full ``ctx_*`` settings and
    ``reference_optimization_steps``.

    Parameters
    ----------
    config
        Adsorption configuration to translate.
    include_provenance
        If True, include full provenance fields.
    """
    if include_provenance is None:
        include_provenance = bool(config.export_placement_provenance)
    ctx = ComputationContext.from_config(config)
    row: dict[str, Any] = {
        "context_hash": ctx.settings_hash(),
        "schema_version": SCHEMA_VERSION,
    }
    if include_provenance:
        row.update({f"ctx_{k}": v for k, v in ctx.to_dict().items()})
        row["reference_optimization_steps"] = config.reference_optimization_steps
    return row


@dataclass
class PlacementRecord:
    """ML dataset row: initial placement + post-relax energies/labels.

    Geometry lives on ``descriptor`` (:class:`PlacementDescriptor`). Absolute pose
    fields and ``conformer_index`` are the training features and describe the
    **initial** placement. Site/orientation fields are initial enumeration
    provenance only; adsorbates may move during relaxation. ``distance`` and
    ``energy_*`` are post-relax outcomes. CSV exports default to lean
    feature+label columns; set ``export_placement_provenance=True`` for
    ``initial_*`` provenance.
    """

    # --- Identity ---
    molecule: str
    smiles: str
    surface_id: str
    placement_id: int

    # --- Initial placement geometry ---
    descriptor: PlacementDescriptor

    # --- Energies (eV, post-relax) ---
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
        self.descriptor = _normalized_descriptor(
            self.descriptor, placement_index=self.placement_id
        )
        q = normalize_quaternion(
            np.array(
                [
                    float(self.descriptor.quat_w or 1.0),
                    float(self.descriptor.quat_x or 0.0),
                    float(self.descriptor.quat_y or 0.0),
                    float(self.descriptor.quat_z or 0.0),
                ],
                dtype=float,
            )
        )
        self.descriptor.quat_w = float(q[0])
        self.descriptor.quat_x = float(q[1])
        self.descriptor.quat_y = float(q[2])
        self.descriptor.quat_z = float(q[3])

    def to_placement_descriptor(self) -> PlacementDescriptor:
        """Return the stored PlacementDescriptor for placement replay.

        Absolute pose fields must be finite. Dissociative replay also needs
        ``fragment_positions`` on the in-memory record (CSV lean exports omit it;
        enable ``export_placement_provenance`` to persist ``initial_fragment_positions``).
        """
        d = self.descriptor
        required_fields = {
            "x_abs": d.x_abs,
            "y_abs": d.y_abs,
            "z_abs": d.z_abs,
            "quat_w": d.quat_w,
            "quat_x": d.quat_x,
            "quat_y": d.quat_y,
            "quat_z": d.quat_z,
        }
        missing = [
            name
            for name, value in required_fields.items()
            if not _is_finite_number(value)
        ]
        if missing:
            missing_csv = ", ".join(missing)
            raise ValueError(
                "PlacementRecord is missing finite deterministic geometry fields: "
                f"{missing_csv}"
            )
        return d

    def to_config(self) -> AdsorptionConfig:
        """Build an AdsorptionConfig matching the computation context."""
        return self.context.to_config()

    @classmethod
    def from_screening_result(
        cls,
        result: ScreeningResult,
        smiles: str,
        surface_id: str,
        config: AdsorptionConfig | None = None,
    ) -> "PlacementRecord":
        """Build a record from a validated ScreeningResult.

        Parameters
        ----------
        result
            Screening result to convert.
        smiles
            SMILES string of the molecule.
        surface_id
            Surface identifier.
        config
            Optional adsorption config for context.
        """
        return cls(
            molecule=result.molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=result.placement_id,
            descriptor=_normalized_descriptor(
                result.placement_descriptor, placement_index=result.placement_id
            ),
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
        """Build a PlacementRecord from a descriptor.

        Parameters
        ----------
        descriptor
            Placement descriptor to convert.
        molecule
            Molecule name.
        smiles
            SMILES string.
        surface_id
            Surface identifier.
        config
            Optional adsorption config for context.
        placement_id
            Optional placement index override.

        Returns
        -------
        PlacementRecord
        """
        pid = descriptor.placement_index if placement_id is None else placement_id
        return cls(
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=pid,
            descriptor=_normalized_descriptor(descriptor, placement_index=pid),
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
        """Build a PlacementRecord from a placement spec.

        Parameters
        ----------
        spec
            Placement spec to convert.
        molecule
            Molecule name.
        smiles
            SMILES string.
        surface_id
            Surface identifier.
        config
            Optional adsorption config for context.

        Returns
        -------
        PlacementRecord
        """
        return cls(
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=spec.placement_index,
            descriptor=_descriptor_from_spec(spec),
            context=_context_from_config(config),
        )

    def record_hash(self) -> str:
        """Deterministic hash uniquely identifying this placement + method combination.

        Combines placement geometry, identity, and computation settings.
        """
        d = self.descriptor
        key_dict = {
            "molecule": str(self.molecule),
            "smiles": str(self.smiles),
            "surface_id": str(self.surface_id),
            "placement_id": int(self.placement_id),
            "conformer_index": int(d.conformer_index),
            "orientation_type": str(d.orientation_type),
            "face_flip": bool(d.face_flip),
            "en_atom_index": int(d.en_atom_index)
            if d.en_atom_index is not None
            else None,
            "site_index": int(d.site_index),
            "site_type": str(d.site_type) if d.site_type is not None else None,
            "tilt_deg": round(float(d.tilt_deg), 6),
            "azimuth_deg": round(float(d.azimuth_deg), 6),
            "azimuth_in_plane_deg": round(float(d.azimuth_in_plane_deg), 6),
            "z_fraction": round(float(d.z_fraction), 6),
            "x_abs": round(float(d.x_abs or 0.0), 6),
            "y_abs": round(float(d.y_abs or 0.0), 6),
            "z_offset": round(float(d.z_offset), 6),
            "surface_ref_z_abs": round(float(d.surface_ref_z_abs or 0.0), 6),
            "z_abs": round(float(d.z_abs or 0.0), 6),
            "quat_w": round(float(d.quat_w or 1.0), 6),
            "quat_x": round(float(d.quat_x or 0.0), 6),
            "quat_y": round(float(d.quat_y or 0.0), 6),
            "quat_z": round(float(d.quat_z or 0.0), 6),
            "fragment_positions": (
                tuple(
                    (
                        round(float(p[0]), 6),
                        round(float(p[1]), 6),
                        round(float(p[2]), 6),
                    )
                    for p in d.fragment_positions
                )
                if d.fragment_positions is not None
                else None
            ),
            "context_hash": self.context.settings_hash(),
        }
        blob = json.dumps(key_dict, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def to_flat_dict(self, *, include_provenance: bool = False) -> dict[str, Any]:
        """Flatten into a single-level dict suitable for DataFrame rows.

        Lean default: identity, ML pose features, energies/labels, and
        ``context_hash``. Rich mode adds ``initial_*`` provenance and ``ctx_*``.
        Geometry columns come from :meth:`PlacementDescriptor.to_row`.

        Parameters
        ----------
        include_provenance
            If True, include full provenance and context fields.
        """
        row: dict[str, Any] = {
            "record_hash": self.record_hash(),
            "molecule": self.molecule,
            "smiles": self.smiles,
            "surface_id": self.surface_id,
            "placement_id": self.placement_id,
            **self.descriptor.to_row(include_provenance=include_provenance),
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
            "schema_version": SCHEMA_VERSION,
            "context_hash": self.context.settings_hash(),
        }
        if include_provenance:
            for key, value in self.context.to_dict().items():
                row[f"ctx_{key}"] = value
        return row

    @classmethod
    def from_flat_dict(cls, row: dict[str, Any]) -> "PlacementRecord":
        """Reconstruct a PlacementRecord from a flattened dict (e.g. CSV row).

        Accepts schema 3.0 ``initial_*`` provenance columns and ``ctx_*`` context
        columns. Lean rows without provenance use safe defaults.
        Geometry is inflated via :meth:`PlacementDescriptor.from_row`.

        Parameters
        ----------
        row
            Flat dictionary representing a placement record.
        """

        def _ctx_value(name: str, default: Any) -> Any:
            return _with_default(row.get(f"ctx_{name}"), default)

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
            min_initial_distance=float(
                _ctx_value(
                    "min_initial_distance", MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
                )
            ),
            min_contact_ratio=float(
                _ctx_value("min_contact_ratio", MIN_CONTACT_RATIO_DEFAULT)
            ),
            top_layer_tolerance=float(_ctx_value("top_layer_tolerance", 0.5)),
            symmetry_tolerance=float(
                _ctx_value("symmetry_tolerance", DEFAULT_SYMMETRY_TOLERANCE)
            ),
            site_equivalence_tolerance=float(
                _ctx_value(
                    "site_equivalence_tolerance", DEFAULT_SITE_EQUIVALENCE_TOLERANCE
                )
            ),
            hollow_site_dedup_tolerance=float(
                _ctx_value(
                    "hollow_site_dedup_tolerance", DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
                )
            ),
            planar_z_variance_threshold=float(
                _ctx_value(
                    "planar_z_variance_threshold", DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
                )
            ),
        )

        placement_id = int(row["placement_id"])
        return cls(
            molecule=str(row["molecule"]),
            smiles=str(row.get("smiles", "")),
            surface_id=str(row.get("surface_id", "")),
            placement_id=placement_id,
            descriptor=PlacementDescriptor.from_row(row, placement_index=placement_id),
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

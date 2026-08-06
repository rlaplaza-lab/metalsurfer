"""YAML campaign schema parsing for the Python campaign API."""


from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from .config import AdsorptionConfig, fold_bo_config

CampaignKind = Literal["adsorption", "adsorption_bo", "saturation", "saturation_bo"]

_SUBSTRATE_KEYS = frozenset(
    {
        "bulk_id",
        "slab_file",
        "slab",
        "miller_indices",
        "supercell",
        "alloy_host",
        "alloy_guest",
        "alloy_fraction",
        "enforce_top_layer_fraction",
        "adatom_symbol",
        "adatom_coverage",
        "align",
        "slab_relaxation_mode",
        "slab_relaxation_optimizer",
        "slab_relaxation_fmax",
        "slab_relaxation_steps",
        "adatom_relaxation_mode",
        "adatom_relaxation_optimizer",
        "adatom_relaxation_fmax",
        "adatom_relaxation_steps",
        "relax_top_layer",
        "freeze_symbols",
        "top_layer_tolerance",
    }
)


@dataclass(frozen=True)
class CampaignDocument:
    """Parsed campaign YAML ready for substrate prep and run_* dispatch."""

    campaign: CampaignKind
    surface_type: str
    substrate: dict[str, Any]
    molecules: list[tuple[str, str]]
    config: AdsorptionConfig

    @property
    def results_dir(self) -> str:
        return f"results_{self.surface_type}"


def _require_mapping(data: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"{context} must be a mapping, got {type(data).__name__}")
    return data


def _parse_molecules(raw: Any) -> list[tuple[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("molecules must be a non-empty list")
    pairs: list[tuple[str, str]] = []
    for index, item in enumerate(raw):
        entry = _require_mapping(item, context=f"molecules[{index}]")
        smiles = entry.get("smiles")
        name = entry.get("name")
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"molecules[{index}].smiles must be a non-empty string")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"molecules[{index}].name must be a non-empty string")
        pairs.append((smiles.strip(), name.strip()))
    return pairs


def _normalize_substrate(raw: Any) -> dict[str, Any]:
    substrate = _require_mapping(raw, context="substrate")
    unknown = set(substrate) - _SUBSTRATE_KEYS
    if unknown:
        quoted = ", ".join(sorted(unknown))
        raise ValueError(f"substrate contains unknown keys: {quoted}")
    for tuple_key in ("miller_indices", "supercell"):
        if tuple_key in substrate:
            value = substrate[tuple_key]
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"substrate.{tuple_key} must be a 3-element list")
            substrate[tuple_key] = tuple(int(v) for v in value)
    return substrate


def _parse_campaign_kind(raw: Any) -> CampaignKind:
    allowed: tuple[CampaignKind, ...] = (
        "adsorption",
        "adsorption_bo",
        "saturation",
        "saturation_bo",
    )
    if raw not in allowed:
        quoted = ", ".join(repr(item) for item in allowed)
        raise ValueError(f"campaign must be one of {quoted}, got {raw!r}")
    return raw


def parse_campaign_dict(data: dict[str, Any]) -> CampaignDocument:
    """Validate a campaign mapping and return a :class:`CampaignDocument`."""
    campaign = _parse_campaign_kind(data.get("campaign"))
    surface_type = data.get("surface_type")
    if not isinstance(surface_type, str) or not surface_type.strip():
        raise ValueError("surface_type must be a non-empty string")

    substrate = _normalize_substrate(data.get("substrate", {}))
    sources = [
        substrate.get("bulk_id") is not None,
        substrate.get("slab_file") is not None,
        substrate.get("slab") is not None,
    ]
    if sum(sources) != 1:
        raise ValueError(
            "substrate must specify exactly one of bulk_id, slab_file, or slab"
        )

    molecules = _parse_molecules(data.get("molecules"))
    config_raw = data.get("config", {})
    if config_raw is None:
        config_raw = {}
    if not isinstance(config_raw, dict):
        raise ValueError("config must be a mapping when provided")
    config_payload = dict(config_raw)
    # Nested ``bo:`` / ``bo.transfer:`` only (flat ``bo_*`` keys are rejected).
    config_payload["bo"] = fold_bo_config(config_payload)
    config = AdsorptionConfig(**config_payload)

    return CampaignDocument(
        campaign=campaign,
        surface_type=surface_type.strip(),
        substrate=substrate,
        molecules=molecules,
        config=config,
    )


def load_campaign_yaml(path: str | Path) -> CampaignDocument:
    """Load and validate a campaign YAML file."""
    yaml_path = Path(path)
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"Campaign file is empty: {yaml_path}")
    return parse_campaign_dict(_require_mapping(raw, context="campaign root"))

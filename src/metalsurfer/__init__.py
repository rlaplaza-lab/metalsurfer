"""Package root: lazy exports for core campaign and placement APIs."""

__version__ = "0.4.0"

import importlib

from ._logging import ensure_log_record_defaults
from .config import AdsorptionConfig, BOConfig, BOTransferConfig
from .exceptions import (
    DependencyMissingError,
    GeometryValidationError,
    OptimizationError,
)
from .models import (
    BindingCampaignResult,
    MoleculeCampaignSummary,
    MoleculeSummary,
    MultiMolSaturationRunResult,
    MultiMolSaturationStepResult,
    ReferenceEnergies,
    SaturationCampaignResult,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
    ScreeningRunResult,
    TimingInfo,
)

__all__ = [
    "__version__",
    "AdsorptionConfig",
    "BOConfig",
    "BOTransferConfig",
    "resolved_bo_eval_budget",
    "bo_eval_schedule",
    "run_adsorption",
    "run_adsorption_bo",
    "run_saturation",
    "run_saturation_bo",
    "run_campaign",
    "load_campaign_yaml",
    "CampaignDocument",
    "BindingCampaignResult",
    "MoleculeCampaignSummary",
    "MoleculeSummary",
    "ReferenceEnergies",
    "SaturationCampaignResult",
    "SaturationRunResult",
    "SaturationStepResult",
    "MultiMolSaturationRunResult",
    "MultiMolSaturationStepResult",
    "ScreeningResult",
    "ScreeningRunResult",
    "TimingInfo",
    "DependencyMissingError",
    "GeometryValidationError",
    "OptimizationError",
    "configure_logging",
    "results_dir_for",
    "prepare_substrate",
    "finalize_substrate",
    "enumerate_placement_specs",
    "generate_placement_from_spec",
]

_LAZY_MODULES = {
    "_logging": {"configure_logging"},
    "config": {"resolved_bo_eval_budget", "bo_eval_schedule"},
    "surface_prep": {
        "prepare_substrate",
        "finalize_substrate",
    },
    "placement": {
        "generate_placement_from_spec",
        "enumerate_placement_specs",
    },
    "campaigns": {
        "run_adsorption",
        "run_adsorption_bo",
        "run_saturation",
        "run_saturation_bo",
        "run_campaign",
    },
    "campaign_schema": {
        "load_campaign_yaml",
        "CampaignDocument",
    },
    "io_results": {
        "results_dir_for",
    },
}

ensure_log_record_defaults()


def __getattr__(name: str):
    # Intentional lazy import: heavy/optional submodules loaded on first access.
    for mod, names in _LAZY_MODULES.items():
        if name in names:
            return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

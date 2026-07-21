"""Adsorption on arbitrary materials."""

__version__ = "0.3.1"

import importlib

from ._logging import ensure_log_record_defaults
from .config import AdsorptionConfig
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
    "resolved_bo_eval_budget",
    "bo_eval_schedule",
    "run_adsorption",
    "run_adsorption_bo",
    "run_saturation",
    "run_saturation_bo",
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
]

_LAZY_MODULES = {
    "_logging": {"configure_logging"},
    "config": {"resolved_bo_eval_budget", "bo_eval_schedule"},
    "surface_prep": {
        "prepare_substrate",
        "finalize_substrate",
        "relax_substrate",
        "resize_substrate_for_molecule",
        "apply_material_pbc",
        "SlabContainer",
        "create_slab_from_bulk",
        "create_slab_from_atoms",
        "substitute_alloy",
        "deposit_adatoms",
        "auto_resize_substrate_for_molecule",
        "compute_minimum_supercell",
        "ensure_slab_z_alignment",
        "apply_surface_constraints",
        "validate_substrate",
        "accept_substrate_for_api",
        "coerce_slab_container",
    },
    "conformers": {"create_conformers_from_smiles", "select_conformer_boltzmann"},
    "placement": {
        "generate_placement_from_spec",
        "generate_placement_from_descriptor",
        "enumerate_placement_specs",
        "calculate_min_distance",
        "get_symmetry_aware_sites",
        "get_symmetry_info",
    },
    "optimization": {
        "setup_calculator",
        "setup_torchsim_model",
        "setup_single_model",
        "TorchSimCalculator",
        "optimize_isolated_molecules_batched",
        "optimize_adsorbate_slab_batched",
        "batch_static",
        "identify_top_layer_indices",
        "identify_relaxable_surface_indices",
        "compute_frozen_indices",
        "frozen_indices_from_constraints",
    },
    "filters": {"filter_results", "check_decomposition", "check_desorption"},
    "workflow": {
        "process_molecule",
        "process_molecule_bayesian",
        "run_saturation_screening",
        "calculate_reference_energies",
        "load_molecules",
    },
    "campaigns": {
        "run_adsorption",
        "run_adsorption_bo",
        "run_saturation",
        "run_saturation_bo",
    },
    "io_results": {
        "setup_directories",
        "save_molecule_results",
        "save_single_molecule_results",
        "screening_run_result",
        "save_summary_results",
        "save_saturation_results",
        "save_multi_mol_saturation_results",
        "write_run_metadata",
        "write_run_settings",
        "results_dir",
        "results_dir_for",
    },
    "ml": {
        "BindingEnergyPredictor",
        "ComputationContext",
        "DatasetLogger",
        "PlacementRecord",
        "evaluate_model",
        "extract_features",
        "extract_features_from_dataset",
        "grouped_cross_validate",
        "load_dataset",
        "train_model",
    },
    "symmetry": {"SymmetryAnalyzer", "SymmetryAnalysisError"},
}

ensure_log_record_defaults()


def __getattr__(name: str):
    # Intentional lazy import: heavy/optional submodules loaded on first access.
    for mod, names in _LAZY_MODULES.items():
        if name in names:
            return getattr(importlib.import_module(f".{mod}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

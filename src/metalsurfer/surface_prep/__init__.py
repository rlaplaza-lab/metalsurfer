"""Substrate and material preparation for metalsurfer campaigns.

All helpers run **before** ``run_adsorption``, ``run_saturation``, or related
campaign APIs. Entry points validate substrates via
:func:`~metalsurfer.surface_prep.accept_substrate_for_api` but never align,
resize, rewrite constraints, or re-equilibrate ionic positions.

:func:`prepare_substrate` **equilibrates substrate ionic positions by default**
(``slab_relaxation_mode="ionic_only"``) and attaches ASE ``FixAtoms`` for
adsorption via ``relax_top_layer`` / ``freeze_symbols`` prep kwargs (default:
entire substrate frozen). ``relax_top_layer=True`` is a material-aware shortcut
(simple height band on slabs; see
:func:`~metalsurfer.surface_prep.identify_relaxable_surface_indices`). Campaign APIs read
those ``FixAtoms`` only.

Typical flow: :func:`prepare_substrate` → optional
:func:`resize_substrate_for_molecule` after conformer generation → campaign API.

Step-by-step alternative: ``create_slab_from_*`` → ``substitute_alloy`` /
``deposit_adatoms`` → optional :func:`relax_substrate` → :func:`finalize_substrate`.

Prep-time ASE relaxation (``slab_relaxation_*``) is separate from TorchSim placement
relaxation (``fmax``, ``stage1_steps``, ``stage2_steps``).
"""


import importlib
from typing import Any

__all__ = [
    # Orchestration
    "prepare_substrate",
    "finalize_substrate",
    "relax_substrate",
    "resize_substrate_for_molecule",
    # Layout, PBC, constraints, validation
    "apply_material_pbc",
    "apply_surface_constraints",
    "ensure_slab_z_alignment",
    "validate_substrate",
    "validate_substrate_conformer_sizing",
    "accept_substrate_for_api",
    "coerce_slab_container",
    # Construction and modification
    "SlabContainer",
    "create_slab_from_bulk",
    "create_slab_from_atoms",
    "substitute_alloy",
    "deposit_adatoms",
    # Sizing
    "auto_resize_substrate_for_molecule",
    "compute_minimum_supercell",
    # Freeze policy
    "identify_relaxable_surface_indices",
    "identify_top_layer_indices",
    "top_layer_indices_by_height",
    "compute_frozen_indices",
    "frozen_indices_from_constraints",
    "max_frozen_substrate_displacement",
    "check_frozen_substrate_displacement",
    "format_atom_index_ranges",
    "log_substrate_freeze_policy",
]

_LAZY_MODULES: dict[str, set[str]] = {
    "freeze": {
        "identify_relaxable_surface_indices",
        "identify_top_layer_indices",
        "top_layer_indices_by_height",
        "compute_frozen_indices",
        "frozen_indices_from_constraints",
        "max_frozen_substrate_displacement",
        "check_frozen_substrate_displacement",
        "format_atom_index_ranges",
        "log_substrate_freeze_policy",
    },
    "_surfaces": {
        "SlabContainer",
        "accept_substrate_for_api",
        "apply_surface_constraints",
        "auto_resize_substrate_for_molecule",
        "coerce_slab_container",
        "compute_minimum_supercell",
        "create_slab_from_atoms",
        "create_slab_from_bulk",
        "deposit_adatoms",
        "ensure_slab_z_alignment",
        "substitute_alloy",
        "validate_substrate",
        "validate_substrate_conformer_sizing",
    },
    "prep": {
        "apply_material_pbc",
        "finalize_substrate",
        "prepare_substrate",
        "relax_substrate",
        "resize_substrate_for_molecule",
    },
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MODULES["freeze"]:
        return getattr(importlib.import_module(".freeze", __name__), name)
    if name in _LAZY_MODULES["_surfaces"]:
        return getattr(importlib.import_module("._surfaces", __name__), name)
    if name in _LAZY_MODULES["prep"]:
        return getattr(importlib.import_module(".prep", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))

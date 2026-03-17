"""Configuration for adsorption screening workflows."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from .models import PlacementSpec


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_range_tuple(name: str, value: tuple[float, float]) -> None:
    if len(value) != 2:
        raise ValueError(
            f"{name} must be a 2-tuple (low, high), got length {len(value)}"
        )
    if value[0] >= value[1]:
        raise ValueError(
            f"{name} lower bound ({value[0]}) must be less than "
            f"upper bound ({value[1]})"
        )


@dataclass
class AdsorptionConfig:
    """Configuration for an adsorption screening run. Primary knobs: model_name, num_conformers, num_placements.

    For dissociative adsorption (e.g. H2 → 2H) or other processes involving bond breaking,
    set skip_topology_check=True to bypass connectivity/formula/bond-pattern checks.
    Set skip_desorption_check=True to bypass adsorbate-to-surface distance validation.
    """

    model_name: str = "uma-s-1p1"
    num_conformers: int = 10
    num_placements: int = 100
    device: str = "cuda"
    fmax: float = 0.05
    stage1_steps: int = 50
    stage2_steps: int = 150
    reference_optimization_steps: int = 100
    placement_x_range: tuple[float, float] = (-4.0, 4.0)
    placement_y_range: tuple[float, float] = (-4.0, 4.0)
    placement_z_range: tuple[float, float] = (2.0, 3.0)
    placement_z_scale_by_covalent_radius: bool = True
    conformer_sampling: Literal["boltzmann", "cycle", "mixed"] = "cycle"
    placement_mode: Literal["random", "sites", "auto", "envelope"] = "auto"
    placement_filter: Callable[[PlacementSpec], bool] | None = field(
        default=None, repr=False
    )
    flat_aromatic_parallel_fraction: float = (
        0.5  # Fraction of flat-aromatic placements in horizontal (π-stacking) vs EN-down.
        # Default 0.5 ensures both orientations are explored.
    )
    min_initial_distance: float = 1.5
    min_contact_ratio: float = (
        0.8  # Lower bound: (r_mol+r_surf)*ratio avoids covalent binding
    )
    max_initial_distance: float | None = (
        None  # Upper bound for initial placement (optional)
    )
    top_layer_tolerance: float = 0.5
    planar_z_variance_threshold: float = (
        0.01  # Max z variance (Å²) for planar classification
    )
    relax_top_layer: bool = True
    freeze_symbols: list[str] | None = None
    min_interatomic_distance: float = 0.5
    max_force_convergence: float = 0.05
    binding_distance_threshold: float = (
        4.0  # Post-opt: reject if adsorbate-surface > this (desorbed)
    )
    max_adsorption_energy: float = 5.0
    vacuum_box_size: float = 20.0
    energy_dedup_threshold: float = 0.05
    rmsd_dedup_threshold: float = 0.1
    connectivity_multipliers: list[float] = field(default_factory=lambda: [1.2, 1.3])
    seed: int = 42
    boltzmann_temperature: float = 300.0
    auto_resize_slab: bool = True
    min_pbc_image_separation: float = 8.0
    vasp_encut: int = 400
    vasp_ediff: float = 1e-6
    vasp_ediffg: float = -0.02
    vasp_nsw: int = 100
    vasp_kpoints: tuple[int, int, int] = (4, 4, 1)
    saturation: bool = False
    skip_topology_check: bool = False
    skip_desorption_check: bool = False

    # Strict workflow: fail instead of skipping molecules
    fail_on_missing_reference: bool = False
    fail_on_conformer_failure: bool = False

    # Debug: write XYZ files of initial placements (before optimization) to the same
    # xyz_structures/{molecule}_all/ dir as final conformer XYZs (as initial_*.xyz)
    debug_write_initial_placements: bool = False

    # TorchSim optimisation tuning
    optimize_isolated_sequentially: bool = False
    ts_optimizer: Literal["fire", "lbfgs", "bfgs"] = "fire"
    steps_between_swaps: int = 5

    # TorchSim autobatcher tuning (helps avoid CUDA OOM)
    autobatcher_max_memory_padding: float = 0.5
    autobatcher_max_memory_scaler: float | None = None
    autobatcher_max_atoms_to_try: int = 100_000

    # Bayesian optimisation (surrogate-guided placement selection)
    bo_enabled: bool = False
    bo_initial_random: int = 10
    bo_batch_size: int = 10
    bo_total_budget: int = 100
    # Lower kappa biases toward exploitation; empirically improves early-best
    # discovery on heterogeneous graphene test runs.
    bo_ucb_kappa: float = 1.0
    bo_acquisition: Literal["lcb", "ei", "pi"] = "lcb"
    bo_surrogate: Literal[
        "random_forest",
        "extra_trees",
        "gradient_boost",
        "ridge",
    ] = "random_forest"
    bo_candidate_pool_size: int | None = None

    def __post_init__(self) -> None:
        _check_positive_int("num_conformers", self.num_conformers)
        _check_positive_int("num_placements", self.num_placements)
        _check_positive_int("stage1_steps", self.stage1_steps)
        _check_positive_int("stage2_steps", self.stage2_steps)
        _check_positive_int(
            "reference_optimization_steps", self.reference_optimization_steps
        )
        _check_positive_int("vasp_nsw", self.vasp_nsw)
        _check_positive_int("vasp_encut", self.vasp_encut)
        _check_positive("fmax", self.fmax)
        _check_positive("min_initial_distance", self.min_initial_distance)
        if not 0.5 <= self.min_contact_ratio <= 1.2:
            raise ValueError(
                f"min_contact_ratio must be in [0.5, 1.2], got {self.min_contact_ratio}"
            )
        if self.max_initial_distance is not None and self.max_initial_distance <= 0:
            raise ValueError(
                f"max_initial_distance must be positive when set, got {self.max_initial_distance}"
            )
        if self.placement_mode not in ("random", "sites", "auto", "envelope"):
            raise ValueError(
                f"placement_mode must be 'random', 'sites', 'auto', or 'envelope', "
                f"got {self.placement_mode!r}"
            )
        if not 0.0 <= self.flat_aromatic_parallel_fraction <= 1.0:
            raise ValueError(
                f"flat_aromatic_parallel_fraction must be in [0.0, 1.0], "
                f"got {self.flat_aromatic_parallel_fraction}"
            )
        if self.conformer_sampling not in ("boltzmann", "cycle", "mixed"):
            raise ValueError(
                f"conformer_sampling must be 'boltzmann', 'cycle', or 'mixed', "
                f"got {self.conformer_sampling!r}"
            )
        _check_positive("top_layer_tolerance", self.top_layer_tolerance)
        _check_positive("planar_z_variance_threshold", self.planar_z_variance_threshold)
        _check_positive("min_interatomic_distance", self.min_interatomic_distance)
        _check_positive("max_force_convergence", self.max_force_convergence)
        _check_positive("binding_distance_threshold", self.binding_distance_threshold)
        _check_positive("max_adsorption_energy", self.max_adsorption_energy)
        _check_positive("vacuum_box_size", self.vacuum_box_size)
        _check_positive("boltzmann_temperature", self.boltzmann_temperature)
        if self.auto_resize_slab:
            _check_positive("min_pbc_image_separation", self.min_pbc_image_separation)
        _check_non_negative("energy_dedup_threshold", self.energy_dedup_threshold)
        _check_non_negative("rmsd_dedup_threshold", self.rmsd_dedup_threshold)
        _check_range_tuple("placement_x_range", self.placement_x_range)
        _check_range_tuple("placement_y_range", self.placement_y_range)
        _check_range_tuple("placement_z_range", self.placement_z_range)
        if not self.connectivity_multipliers:
            raise ValueError("connectivity_multipliers must be a non-empty list")
        for i, m in enumerate(self.connectivity_multipliers):
            if m <= 0:
                raise ValueError(
                    f"connectivity_multipliers[{i}] must be positive, got {m}"
                )
        if len(self.vasp_kpoints) != 3:
            raise ValueError(
                f"vasp_kpoints must be a 3-tuple, got length {len(self.vasp_kpoints)}"
            )
        for i, k in enumerate(self.vasp_kpoints):
            if not isinstance(k, int) or k <= 0:
                raise ValueError(
                    f"vasp_kpoints[{i}] must be a positive integer, got {k!r}"
                )

        if not self.model_name:
            raise ValueError("model_name must be a non-empty string")

        # Bayesian optimisation knobs
        if self.bo_enabled:
            _check_positive_int("bo_initial_random", self.bo_initial_random)
            _check_positive_int("bo_batch_size", self.bo_batch_size)
            _check_positive_int("bo_total_budget", self.bo_total_budget)
            if self.bo_initial_random > self.bo_total_budget:
                raise ValueError(
                    f"bo_initial_random ({self.bo_initial_random}) must not exceed "
                    f"bo_total_budget ({self.bo_total_budget})"
                )
            if self.bo_ucb_kappa < 0:
                raise ValueError(
                    f"bo_ucb_kappa must be non-negative, got {self.bo_ucb_kappa}"
                )
            if self.bo_acquisition not in ("lcb", "ei", "pi"):
                raise ValueError(
                    f"bo_acquisition must be 'lcb', 'ei', or 'pi', got {self.bo_acquisition!r}"
                )
            if self.bo_surrogate not in (
                "random_forest",
                "extra_trees",
                "gradient_boost",
                "ridge",
            ):
                raise ValueError(
                    "bo_surrogate must be one of "
                    "'random_forest', 'extra_trees', 'gradient_boost', 'ridge', "
                    f"got {self.bo_surrogate!r}"
                )
            if self.bo_candidate_pool_size is not None:
                _check_positive_int(
                    "bo_candidate_pool_size", self.bo_candidate_pool_size
                )
        if self.ts_optimizer not in ("fire", "lbfgs", "bfgs"):
            raise ValueError(
                f"ts_optimizer must be 'fire', 'lbfgs', or 'bfgs', "
                f"got {self.ts_optimizer!r}"
            )
        _check_positive_int("steps_between_swaps", self.steps_between_swaps)
        if not 0.1 <= self.autobatcher_max_memory_padding <= 1.0:
            raise ValueError(
                f"autobatcher_max_memory_padding must be in [0.1, 1.0], got {self.autobatcher_max_memory_padding}"
            )
        if (
            self.autobatcher_max_memory_scaler is not None
            and self.autobatcher_max_memory_scaler <= 0
        ):
            raise ValueError(
                f"autobatcher_max_memory_scaler must be positive when set, got {self.autobatcher_max_memory_scaler}"
            )
        _check_positive_int(
            "autobatcher_max_atoms_to_try", self.autobatcher_max_atoms_to_try
        )

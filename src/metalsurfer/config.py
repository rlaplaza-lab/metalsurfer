"""Configuration for adsorption screening workflows."""

from collections.abc import Callable
from dataclasses import dataclass, field
from math import isfinite
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


def _check_choice(name: str, value: str, *, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        quoted = ", ".join(repr(item) for item in allowed[:-1])
        message = f"{quoted}, or {allowed[-1]!r}" if quoted else repr(allowed[-1])
        raise ValueError(f"{name} must be {message}, got {value!r}")


CONFORMER_SAMPLING_OPTIONS: tuple[str, ...] = ("boltzmann", "cycle", "mixed")
MATERIAL_TYPE_OPTIONS: tuple[str, ...] = ("slab", "nanoparticle", "porous")
BO_ACQUISITION_OPTIONS: tuple[str, ...] = ("lcb", "ei", "pi")
BO_SURROGATE_OPTIONS: tuple[str, ...] = (
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
)
TS_OPTIMIZER_OPTIONS: tuple[str, ...] = ("fire", "lbfgs", "bfgs")


@dataclass
class AdsorptionConfig:
    """Configuration for an adsorption screening run. Primary knobs: model_name, num_conformers, num_placements.

    For dissociative adsorption (e.g. H2 → 2H) or other processes involving bond breaking,
    set skip_topology_check=True to disable molecular decomposition checks.
    Set skip_desorption_check=True to disable post-optimization distance validation.
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
    # Explicit material type: "slab" (2D periodic), "nanoparticle" (0D), or "porous" (3D periodic)
    material_type: Literal["slab", "nanoparticle", "porous"] = "slab"
    # Voronoi site generation parameters
    voronoi_probe_radius: float = 1.2  # min distance from framework atom to site (Å)
    voronoi_max_site_distance: float = 4.0  # max distance for accessible sites (Å)
    voronoi_site_enrichment: bool = True  # geodesic ridge subdivision for denser sites
    conformer_sampling: Literal["boltzmann", "cycle", "mixed"] = "cycle"
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
    symmetry_tolerance: float = 0.1
    site_equivalence_tolerance: float = 0.05
    hollow_site_dedup_tolerance: float = 0.2
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
    multi_molecule_saturation: bool = False
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
    # When set, forces TorchSim memory-estimation probe cap. When None, Metalsurfer
    # computes a conservative per-call cap from current workload.
    autobatcher_max_atoms_to_try: int | None = None
    # Saturation-only: allow reusing prior autobatcher estimate for small size growth
    saturation_autobatcher_reuse: bool = True
    saturation_autobatcher_reuse_growth_atoms: int = 32
    saturation_autobatcher_reuse_growth_fraction: float = 0.1

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
    bo_include_failure_negatives: bool = True
    bo_failure_penalty_default: float = 10.0
    bo_failure_penalty_overrides: dict[str, float] = field(
        default_factory=lambda: {
            "generation": 18.0,
            "optimization": 20.0,
            "validation": 14.0,
            "energy_cap": 12.0,
            "filter": 11.0,
        }
    )
    bo_transfer_enabled: bool = False
    bo_transfer_min_step_observations: int = 5
    bo_transfer_weight_cap: float = 0.35
    bo_transfer_similarity_lengthscale: float = 1.0
    bo_transfer_min_similarity: float = 0.05
    bo_transfer_trust_patience: int = 2
    bo_transfer_mae_tolerance: float = 0.0
    bo_transfer_exploration_fraction: float = 0.2

    def __post_init__(self) -> None:
        positive_int_fields = (
            ("num_conformers", self.num_conformers),
            ("num_placements", self.num_placements),
            ("stage1_steps", self.stage1_steps),
            ("stage2_steps", self.stage2_steps),
            ("reference_optimization_steps", self.reference_optimization_steps),
            ("vasp_nsw", self.vasp_nsw),
            ("vasp_encut", self.vasp_encut),
            ("steps_between_swaps", self.steps_between_swaps),
            (
                "saturation_autobatcher_reuse_growth_atoms",
                self.saturation_autobatcher_reuse_growth_atoms,
            ),
        )
        for field_name, field_value in positive_int_fields:
            _check_positive_int(field_name, field_value)

        positive_fields = (
            ("fmax", self.fmax),
            ("min_initial_distance", self.min_initial_distance),
            ("top_layer_tolerance", self.top_layer_tolerance),
            ("symmetry_tolerance", self.symmetry_tolerance),
            ("site_equivalence_tolerance", self.site_equivalence_tolerance),
            ("hollow_site_dedup_tolerance", self.hollow_site_dedup_tolerance),
            ("planar_z_variance_threshold", self.planar_z_variance_threshold),
            ("min_interatomic_distance", self.min_interatomic_distance),
            ("max_force_convergence", self.max_force_convergence),
            ("binding_distance_threshold", self.binding_distance_threshold),
            ("max_adsorption_energy", self.max_adsorption_energy),
            ("vacuum_box_size", self.vacuum_box_size),
            ("boltzmann_temperature", self.boltzmann_temperature),
        )
        if self.auto_resize_slab:
            positive_fields += (
                ("min_pbc_image_separation", self.min_pbc_image_separation),
            )
        for field_name, field_value in positive_fields:
            _check_positive(field_name, field_value)

        non_negative_fields = (
            ("energy_dedup_threshold", self.energy_dedup_threshold),
            ("rmsd_dedup_threshold", self.rmsd_dedup_threshold),
        )
        for field_name, field_value in non_negative_fields:
            _check_non_negative(field_name, field_value)

        range_fields = (
            ("placement_x_range", self.placement_x_range),
            ("placement_y_range", self.placement_y_range),
            ("placement_z_range", self.placement_z_range),
        )
        for field_name, field_value in range_fields:
            _check_range_tuple(field_name, field_value)

        if not 0.5 <= self.min_contact_ratio <= 1.2:
            raise ValueError(
                f"min_contact_ratio must be in [0.5, 1.2], got {self.min_contact_ratio}"
            )
        if self.max_initial_distance is not None and self.max_initial_distance <= 0:
            raise ValueError(
                f"max_initial_distance must be positive when set, got {self.max_initial_distance}"
            )
        if not 0.0 <= self.flat_aromatic_parallel_fraction <= 1.0:
            raise ValueError(
                f"flat_aromatic_parallel_fraction must be in [0.0, 1.0], "
                f"got {self.flat_aromatic_parallel_fraction}"
            )
        _check_choice(
            "conformer_sampling",
            self.conformer_sampling,
            allowed=CONFORMER_SAMPLING_OPTIONS,
        )
        _check_choice(
            "material_type",
            self.material_type,
            allowed=MATERIAL_TYPE_OPTIONS,
        )
        if self.voronoi_probe_radius <= 0:
            raise ValueError(
                f"voronoi_probe_radius must be positive, got {self.voronoi_probe_radius}"
            )
        if self.voronoi_max_site_distance <= self.voronoi_probe_radius:
            raise ValueError(
                f"voronoi_max_site_distance ({self.voronoi_max_site_distance}) must be "
                f"greater than voronoi_probe_radius ({self.voronoi_probe_radius})"
            )
        if not isinstance(self.voronoi_site_enrichment, bool):
            raise ValueError(
                "voronoi_site_enrichment must be a bool, "
                f"got {type(self.voronoi_site_enrichment).__name__}"
            )
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
            _check_choice(
                "bo_acquisition",
                self.bo_acquisition,
                allowed=BO_ACQUISITION_OPTIONS,
            )
            _check_choice(
                "bo_surrogate",
                self.bo_surrogate,
                allowed=BO_SURROGATE_OPTIONS,
            )
            if self.bo_candidate_pool_size is not None:
                _check_positive_int(
                    "bo_candidate_pool_size", self.bo_candidate_pool_size
                )
            if (
                not isfinite(self.bo_failure_penalty_default)
                or self.bo_failure_penalty_default < 0
            ):
                raise ValueError(
                    "bo_failure_penalty_default must be a finite non-negative value, "
                    f"got {self.bo_failure_penalty_default!r}"
                )
            if not isinstance(self.bo_failure_penalty_overrides, dict):
                raise ValueError(
                    "bo_failure_penalty_overrides must be a dict[str, float], "
                    f"got {type(self.bo_failure_penalty_overrides).__name__}"
                )
            for key, value in self.bo_failure_penalty_overrides.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(
                        "bo_failure_penalty_overrides keys must be non-empty strings, "
                        f"got {key!r}"
                    )
                if not isfinite(value) or value < 0:
                    raise ValueError(
                        "bo_failure_penalty_overrides values must be finite non-negative, "
                        f"got {value!r} for key {key!r}"
                    )
            _check_positive_int(
                "bo_transfer_min_step_observations",
                self.bo_transfer_min_step_observations,
            )
            _check_positive_int(
                "bo_transfer_trust_patience", self.bo_transfer_trust_patience
            )
            if (
                not isfinite(self.bo_transfer_weight_cap)
                or not 0.0 <= self.bo_transfer_weight_cap < 1.0
            ):
                raise ValueError(
                    "bo_transfer_weight_cap must be finite in [0.0, 1.0), "
                    f"got {self.bo_transfer_weight_cap!r}"
                )
            _check_positive(
                "bo_transfer_similarity_lengthscale",
                self.bo_transfer_similarity_lengthscale,
            )
            if (
                not isfinite(self.bo_transfer_min_similarity)
                or not 0.0 <= self.bo_transfer_min_similarity <= 1.0
            ):
                raise ValueError(
                    "bo_transfer_min_similarity must be finite in [0.0, 1.0], "
                    f"got {self.bo_transfer_min_similarity!r}"
                )
            if (
                not isfinite(self.bo_transfer_mae_tolerance)
                or self.bo_transfer_mae_tolerance < 0.0
            ):
                raise ValueError(
                    "bo_transfer_mae_tolerance must be finite and non-negative, "
                    f"got {self.bo_transfer_mae_tolerance!r}"
                )
            if (
                not isfinite(self.bo_transfer_exploration_fraction)
                or not 0.0 <= self.bo_transfer_exploration_fraction <= 1.0
            ):
                raise ValueError(
                    "bo_transfer_exploration_fraction must be finite in [0.0, 1.0], "
                    f"got {self.bo_transfer_exploration_fraction!r}"
                )
        _check_choice("ts_optimizer", self.ts_optimizer, allowed=TS_OPTIMIZER_OPTIONS)
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
        if self.autobatcher_max_atoms_to_try is not None:
            _check_positive_int(
                "autobatcher_max_atoms_to_try", self.autobatcher_max_atoms_to_try
            )
        if not 0.0 <= self.saturation_autobatcher_reuse_growth_fraction <= 1.0:
            raise ValueError(
                "saturation_autobatcher_reuse_growth_fraction must be in [0.0, 1.0], "
                f"got {self.saturation_autobatcher_reuse_growth_fraction}"
            )

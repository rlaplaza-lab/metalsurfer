"""Configuration for adsorption screening workflows."""

from collections.abc import Callable
from dataclasses import InitVar, dataclass, field
from math import isfinite
from typing import Literal
from warnings import warn

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
SITE_CLASSIFICATION_OPTIONS: tuple[str, ...] = ("auto", "distance_ratio", "delaunay")
BO_ACQUISITION_OPTIONS: tuple[str, ...] = ("lcb", "ei", "pi")
# Legacy coarse generation reason → split tokens (one-release BO override alias).
BO_LEGACY_GENERATION_REASON_ALIASES: dict[str, tuple[str, ...]] = {
    "initial_distance_or_site_constraints": (
        "too_close",
        "too_far",
        "vdw_overlap",
        "adsorbate_overlap",
        "missing_z_abs",
    ),
}
BO_INITIAL_SAMPLING_OPTIONS: tuple[str, ...] = (
    "random",
    "spread",
    "spread_xyz",
    "stratified",
)
BO_SURROGATE_OPTIONS: tuple[str, ...] = (
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "ensemble",
)
TS_OPTIMIZER_OPTIONS: tuple[str, ...] = ("fire", "lbfgs", "bfgs")
SLAB_RELAXATION_MODE_OPTIONS: tuple[str, ...] = (
    "none",
    "ionic_only",
    "cell_only",
    "full",
)
SLAB_RELAXATION_OPTIMIZER_OPTIONS: tuple[str, ...] = ("lbfgs", "bfgs", "fire")


def resolved_bo_eval_budget(config: "AdsorptionConfig") -> int:
    """Total BO placement evaluations after auto-resolution of batch sizes."""
    if config.bo_initial_random is None or config.bo_batch_size is None:
        raise ValueError(
            "resolved_bo_eval_budget requires bo_initial_random and bo_batch_size "
            "to be resolved (not None)"
        )
    return config.bo_initial_random + config.bo_total_budget * config.bo_batch_size


def bo_eval_schedule(config: "AdsorptionConfig") -> list[int]:
    """Cumulative placement-evaluation counts for BO replay curves.

    Matches the live workflow: one initial-random batch plus
    ``bo_total_budget`` acquisition batches of ``bo_batch_size`` placements.
    """
    if config.bo_initial_random is None or config.bo_batch_size is None:
        raise ValueError(
            "bo_eval_schedule requires bo_initial_random and bo_batch_size "
            "to be resolved (not None)"
        )
    initial = int(config.bo_initial_random)
    batch = int(config.bo_batch_size)
    budget = resolved_bo_eval_budget(config)
    schedule = [initial]
    current = initial
    while current < budget:
        current = min(current + batch, budget)
        schedule.append(current)
    return schedule


@dataclass
class AdsorptionConfig:
    """Configuration for adsorption screening, Bayesian search, and saturation.

    Primary knobs: ``model_name``, ``num_conformers``, ``num_placements``,
    and ``material_type``. For dissociative adsorption (e.g. H₂ → 2H), set
    ``skip_topology_check=True``. Reference energies remain isolated-molecule
    energies; positive E_ads can result when the relaxed adsorbate dissociates.

    Full field documentation:
    https://metalsurfer.readthedocs.io/en/latest/api/config.html
    """

    model_name: str = "uma-s-1p1"
    num_conformers: int = 10
    num_placements: int | None = None
    device: str = "cuda"
    fmax: float = 0.05
    stage1_steps: int = 50
    stage2_steps: int = 150
    reference_optimization_steps: int = 100
    placement_x_range: tuple[float, float] = (-4.0, 4.0)
    placement_y_range: tuple[float, float] = (-4.0, 4.0)
    placement_z_range: tuple[float, float] = (0.7, 1.25)
    placement_z_scale_by_covalent_radius: bool = True
    material_type: Literal["slab", "nanoparticle", "porous"] = "slab"
    voronoi_probe_radius: float | None = None
    voronoi_max_site_distance: float | None = None
    voronoi_site_enrichment: bool = True
    site_classification_method: Literal["auto", "distance_ratio", "delaunay"] = "auto"
    conformer_sampling: Literal["boltzmann", "cycle", "mixed"] = "cycle"
    placement_filter: Callable[[PlacementSpec], bool] | None = field(
        default=None, repr=False
    )
    flat_aromatic_parallel_fraction: float = 0.5
    adaptive_parallel_fraction: bool = False
    min_initial_distance: float = 1.5
    min_contact_ratio: float = 0.8
    max_initial_distance: float | None = None
    top_layer_tolerance: float = 0.5
    symmetry_tolerance: float = 0.1
    site_equivalence_tolerance: float = 0.05
    hollow_site_dedup_tolerance: float = 0.1
    planar_z_variance_threshold: float = 0.01
    rough_slab_local_z: bool = True
    min_interatomic_distance: float = 0.5
    max_force_convergence: float = 0.05
    binding_distance_threshold: float = 4.0
    strict_initial_placement: bool = False
    reject_vdw_overlaps: bool = False
    vdw_overlap_scale: float = 1.0
    max_closest_approach: float = 0.8
    # Deprecated ctor-only alias for max_closest_approach.
    min_contact_distance: InitVar[float | None] = None
    min_contact_atoms: int = 1
    contact_distance_threshold: float = 2.5
    require_multiple_contact: bool = False
    max_adsorption_energy: float = 5.0
    energy_dedup_threshold: float = 0.05
    rmsd_dedup_threshold: float = 0.1
    connectivity_multipliers: list[float] = field(default_factory=lambda: [1.2, 1.3])
    seed: int = 42
    boltzmann_temperature: float = 300.0
    min_pbc_image_separation: float = 8.0
    vacuum_box_size: float = 20.0
    vasp_encut: int = 400
    vasp_ediff: float = 1e-6
    vasp_ediffg: float = -0.02
    vasp_nsw: int = 100
    vasp_kpoints: tuple[int, int, int] = (4, 4, 1)
    write_vasp_inputs: bool = False
    multi_molecule_saturation: bool = False
    saturation_save_all_placements: bool = True
    save_benchmark_dataset: bool = False
    saturation_discard_topology_rearrangements: bool = True
    saturation_max_steps: int | None = None
    skip_topology_check: bool = False
    skip_desorption_check: bool = False
    fail_on_missing_reference: bool = False
    fail_on_conformer_failure: bool = False
    debug_write_initial_placements: bool = False
    placement_retry_enabled: bool = True
    placement_retry_max_attempts: int = 3
    placement_retry_diversity_seed_increment: int = 1000
    optimize_isolated_sequentially: bool = False
    ts_optimizer: Literal["fire", "lbfgs", "bfgs"] = "fire"
    steps_between_swaps: int = 5
    slab_relaxation_mode: Literal["none", "ionic_only", "cell_only", "full"] = (
        "ionic_only"
    )
    slab_relaxation_optimizer: Literal["lbfgs", "bfgs", "fire"] = "lbfgs"
    slab_relaxation_fmax: float | None = None
    slab_relaxation_steps: int = 200
    autobatcher_max_memory_padding: float = 0.5
    autobatcher_max_memory_scaler: float | None = None
    autobatcher_max_atoms_to_try: int | None = None
    saturation_autobatcher_reuse: bool = True
    saturation_autobatcher_reuse_growth_atoms: int = 32
    saturation_autobatcher_reuse_growth_fraction: float = 0.1
    bo_enabled: bool = False
    bo_initial_random: int | None = None
    bo_initial_sampling: Literal["random", "spread", "spread_xyz", "stratified"] = (
        "spread_xyz"
    )
    bo_batch_size: int | None = None
    bo_total_budget: int = 18
    bo_ucb_kappa: float = 1.96
    bo_acquisition: Literal["lcb", "ei", "pi"] = "ei"
    bo_surrogate: Literal[
        "random_forest",
        "extra_trees",
        "gradient_boost",
        "ridge",
        "ensemble",
    ] = "ridge"
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
    bo_transfer_enabled: bool = True
    bo_transfer_mode: Literal["weighted", "cumulative_refit"] = "weighted"
    bo_transfer_min_step_observations: int = 5
    bo_transfer_weight_cap: float = 0.35
    bo_transfer_similarity_lengthscale: float = 4.0
    bo_transfer_min_similarity: float = 0.05
    bo_transfer_trust_patience: int = 2
    bo_transfer_mae_tolerance: float = 0.0
    bo_transfer_exploration_fraction: float = 0.2
    bo_transfer_proximity_lengthscale: float = 1.0
    bo_transfer_proximity_floor: float = 0.0
    bo_transfer_prior_step_window: int | None = 2
    bo_transfer_recency_lengthscale: float = 4.0
    bo_transfer_occupancy_lengthscale: float = 1.0
    bo_transfer_occupancy_floor: float = 0.0

    def __post_init__(self, min_contact_distance: float | None = None) -> None:
        if min_contact_distance is not None:
            warn(
                "AdsorptionConfig.min_contact_distance is deprecated; "
                "use max_closest_approach (max allowed closest-approach distance).",
                DeprecationWarning,
                stacklevel=2,
            )
            object.__setattr__(
                self, "max_closest_approach", float(min_contact_distance)
            )
        _check_positive("max_closest_approach", self.max_closest_approach)
        _check_positive_int("min_contact_atoms", self.min_contact_atoms)
        _check_positive("contact_distance_threshold", self.contact_distance_threshold)

        positive_int_fields: list[tuple[str, int]] = [
            ("num_conformers", self.num_conformers),
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
        ]
        if self.num_placements is not None:
            positive_int_fields.append(("num_placements", self.num_placements))
        for int_name, int_value in positive_int_fields:
            _check_positive_int(int_name, int_value)

        # Placement retry validation
        if self.placement_retry_enabled:
            _check_positive_int(
                "placement_retry_max_attempts", self.placement_retry_max_attempts
            )
            _check_positive_int(
                "placement_retry_diversity_seed_increment",
                self.placement_retry_diversity_seed_increment,
            )

        positive_fields: list[tuple[str, float]] = [
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
            ("min_pbc_image_separation", self.min_pbc_image_separation),
        ]
        for pos_name, pos_value in positive_fields:
            _check_positive(pos_name, pos_value)

        non_negative_fields: list[tuple[str, float]] = [
            ("energy_dedup_threshold", self.energy_dedup_threshold),
            ("rmsd_dedup_threshold", self.rmsd_dedup_threshold),
        ]
        for nn_name, nn_value in non_negative_fields:
            _check_non_negative(nn_name, nn_value)

        range_fields: list[tuple[str, tuple[float, float]]] = [
            ("placement_x_range", self.placement_x_range),
            ("placement_y_range", self.placement_y_range),
            ("placement_z_range", self.placement_z_range),
        ]
        for range_name, range_value in range_fields:
            _check_range_tuple(range_name, range_value)

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
        if self.voronoi_probe_radius is not None and self.voronoi_probe_radius <= 0:
            raise ValueError(
                f"voronoi_probe_radius must be positive, got {self.voronoi_probe_radius}"
            )
        if (
            self.voronoi_probe_radius is not None
            and self.voronoi_max_site_distance is not None
            and self.voronoi_max_site_distance <= self.voronoi_probe_radius
        ):
            raise ValueError(
                f"voronoi_max_site_distance ({self.voronoi_max_site_distance}) must be "
                f"greater than voronoi_probe_radius ({self.voronoi_probe_radius})"
            )
        if not isinstance(self.voronoi_site_enrichment, bool):
            raise ValueError(
                "voronoi_site_enrichment must be a bool, "
                f"got {type(self.voronoi_site_enrichment).__name__}"
            )
        _check_choice(
            "site_classification_method",
            self.site_classification_method,
            allowed=SITE_CLASSIFICATION_OPTIONS,
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

        _check_choice("device", self.device, allowed=("cuda", "cpu"))

        # Bayesian optimisation knobs
        if self.bo_enabled:
            if self.bo_initial_random is not None:
                _check_positive_int("bo_initial_random", self.bo_initial_random)
            if self.bo_batch_size is not None:
                _check_positive_int("bo_batch_size", self.bo_batch_size)
            _check_positive_int("bo_total_budget", self.bo_total_budget)
            if self.bo_ucb_kappa < 0:
                raise ValueError(
                    f"bo_ucb_kappa must be non-negative, got {self.bo_ucb_kappa}"
                )
            _check_choice(
                "bo_initial_sampling",
                self.bo_initial_sampling,
                allowed=BO_INITIAL_SAMPLING_OPTIONS,
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
            _check_choice(
                "bo_transfer_mode",
                self.bo_transfer_mode,
                allowed=("weighted", "cumulative_refit"),
            )
            if self.bo_transfer_enabled and self.bo_surrogate == "gradient_boost":
                raise ValueError(
                    "bo_transfer_enabled requires a surrogate that supports "
                    "per-sample weights (random_forest, extra_trees, ensemble, or "
                    f"ridge); sample_weight is not supported for "
                    f"bo_surrogate={self.bo_surrogate!r}"
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
            for penalty_key, penalty_value in self.bo_failure_penalty_overrides.items():
                if not isinstance(penalty_key, str) or not penalty_key:
                    raise ValueError(
                        "bo_failure_penalty_overrides keys must be non-empty strings, "
                        f"got {penalty_key!r}"
                    )
                if not isfinite(penalty_value) or penalty_value < 0:
                    raise ValueError(
                        "bo_failure_penalty_overrides values must be finite non-negative, "
                        f"got {penalty_value!r} for key {penalty_key!r}"
                    )
            _check_positive_int(
                "bo_transfer_min_step_observations",
                self.bo_transfer_min_step_observations,
            )
            _check_positive_int(
                "bo_transfer_trust_patience", self.bo_transfer_trust_patience
            )
            if self.bo_transfer_prior_step_window is not None:
                _check_positive_int(
                    "bo_transfer_prior_step_window",
                    self.bo_transfer_prior_step_window,
                )
            _check_positive(
                "bo_transfer_recency_lengthscale",
                self.bo_transfer_recency_lengthscale,
            )
            _check_positive(
                "bo_transfer_occupancy_lengthscale",
                self.bo_transfer_occupancy_lengthscale,
            )
            if (
                not isfinite(self.bo_transfer_occupancy_floor)
                or not 0.0 <= self.bo_transfer_occupancy_floor <= 1.0
            ):
                raise ValueError(
                    "bo_transfer_occupancy_floor must be finite in [0.0, 1.0], "
                    f"got {self.bo_transfer_occupancy_floor!r}"
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
            _check_positive(
                "bo_transfer_proximity_lengthscale",
                self.bo_transfer_proximity_lengthscale,
            )
            if (
                not isfinite(self.bo_transfer_proximity_floor)
                or not 0.0 <= self.bo_transfer_proximity_floor <= 1.0
            ):
                raise ValueError(
                    "bo_transfer_proximity_floor must be finite in [0.0, 1.0], "
                    f"got {self.bo_transfer_proximity_floor!r}"
                )
        _check_choice("ts_optimizer", self.ts_optimizer, allowed=TS_OPTIMIZER_OPTIONS)
        _check_choice(
            "slab_relaxation_mode",
            self.slab_relaxation_mode,
            allowed=SLAB_RELAXATION_MODE_OPTIONS,
        )
        _check_choice(
            "slab_relaxation_optimizer",
            self.slab_relaxation_optimizer,
            allowed=SLAB_RELAXATION_OPTIMIZER_OPTIONS,
        )
        _check_positive_int("slab_relaxation_steps", self.slab_relaxation_steps)
        if self.slab_relaxation_fmax is not None:
            _check_positive("slab_relaxation_fmax", self.slab_relaxation_fmax)
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
        if self.saturation_max_steps is not None:
            _check_positive_int("saturation_max_steps", self.saturation_max_steps)


# InitVar leaves a class attribute equal to its default; remove it so
# dataclasses.replace does not pick up a stale class default via getattr.
# ``__getattr__`` returns None so replace passes the InitVar through as unset.
def _adsorption_config_getattr(self: "AdsorptionConfig", name: str) -> float | None:
    if name == "min_contact_distance":
        return None
    raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")


AdsorptionConfig.__getattr__ = _adsorption_config_getattr  # type: ignore[attr-defined]
if hasattr(AdsorptionConfig, "min_contact_distance"):
    delattr(AdsorptionConfig, "min_contact_distance")

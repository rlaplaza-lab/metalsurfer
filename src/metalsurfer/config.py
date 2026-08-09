"""Configuration for adsorption screening workflows."""

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from math import isfinite
from typing import Any, Literal

from ._numeric_defaults import (
    CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM,
    CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM,
    DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    DEFAULT_SYMMETRY_TOLERANCE,
    MIN_CONTACT_RATIO_DEFAULT,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from .models import PlacementSpec


def _default_bo_failure_penalty_overrides() -> dict[str, float]:
    return {
        "generation": 18.0,
        "optimization": 20.0,
        "validation": 14.0,
        "energy_cap": 12.0,
        "filter": 11.0,
    }


@dataclass
class BOTransferConfig:
    """Cross-step Bayesian transfer hyperparameters for saturation BO."""

    enabled: bool = True
    mode: Literal["weighted", "cumulative_refit"] = "weighted"
    min_step_observations: int = 5
    weight_cap: float = 0.35
    similarity_lengthscale: float = 4.0
    min_similarity: float = 0.05
    trust_patience: int = 2
    mae_tolerance: float = 0.05
    exploration_fraction: float = 0.2
    proximity_lengthscale: float = 1.0
    proximity_floor: float = 0.0
    prior_step_window: int | None = 2
    recency_lengthscale: float = 4.0
    occupancy_lengthscale: float = 1.0
    occupancy_floor: float = 0.0


@dataclass
class BOConfig:
    """Bayesian placement-selection hyperparameters."""

    initial_random: int | None = None
    initial_sampling: Literal["random", "spread", "spread_xyz", "stratified"] = (
        "spread_xyz"
    )
    batch_size: int | None = None
    total_budget: int = 18
    ucb_kappa: float = 1.96
    acquisition: Literal["lcb", "ei", "pi"] = "ei"
    surrogate: Literal[
        "random_forest",
        "extra_trees",
        "gradient_boost",
        "ridge",
        "gaussian_process",
        "ensemble",
    ] = "gradient_boost"
    candidate_pool_size: int | None = None
    include_failure_negatives: bool = True
    failure_penalty_default: float = 10.0
    failure_penalty_overrides: dict[str, float] = field(
        default_factory=_default_bo_failure_penalty_overrides
    )
    transfer: BOTransferConfig = field(default_factory=BOTransferConfig)


_BO_TRANSFER_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(BOTransferConfig)
)
_BO_TOP_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(BOConfig) if f.name != "transfer"
)


def _bo_transfer_from_mapping(data: Mapping[str, Any] | None) -> BOTransferConfig:
    if data is None:
        return BOTransferConfig()
    if isinstance(data, BOTransferConfig):
        return data
    unknown = set(data) - _BO_TRANSFER_FIELDS
    if unknown:
        quoted = ", ".join(sorted(unknown))
        raise ValueError(f"bo.transfer contains unknown keys: {quoted}")
    return BOTransferConfig(**dict(data))


def _bo_config_from_mapping(data: Mapping[str, Any] | None) -> BOConfig:
    if data is None:
        return BOConfig()
    if isinstance(data, BOConfig):
        return data
    payload = dict(data)
    transfer = _bo_transfer_from_mapping(payload.pop("transfer", None))
    unknown = set(payload) - _BO_TOP_FIELDS
    if unknown:
        quoted = ", ".join(sorted(unknown))
        raise ValueError(f"bo contains unknown keys: {quoted}")
    return BOConfig(**payload, transfer=transfer)


def fold_bo_config(config_data: dict[str, Any]) -> BOConfig:
    """Extract nested ``bo:`` from a campaign config mapping into a :class:`BOConfig`.

    Mutates *config_data* by removing the ``bo`` key. Flat ``bo_*`` keys are
    rejected — use nested ``bo:`` / ``bo.transfer:`` instead.
    """
    flat = [key for key in config_data if key.startswith("bo_") and key != "bo"]
    if flat:
        quoted = ", ".join(sorted(flat))
        raise ValueError(
            "Flat BO keys are not supported; nest under 'bo:' / 'bo.transfer:' "
            f"(got: {quoted})"
        )
    return _bo_config_from_mapping(config_data.pop("bo", None))


def _check_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _check_non_negative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


def _check_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_range_tuple(
    name: str, value: tuple[float, float], *, allow_equal: bool = False
) -> None:
    if len(value) != 2:
        raise ValueError(
            f"{name} must be a 2-tuple (low, high), got length {len(value)}"
        )
    if allow_equal:
        if value[0] > value[1]:
            raise ValueError(
                f"{name} lower bound ({value[0]}) must be less than or equal to "
                f"upper bound ({value[1]})"
            )
    elif value[0] >= value[1]:
        raise ValueError(
            f"{name} lower bound ({value[0]}) must be less than "
            f"upper bound ({value[1]})"
        )


def _check_choice(name: str, value: str, *, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        quoted = ", ".join(repr(item) for item in allowed[:-1])
        message = f"{quoted}, or {allowed[-1]!r}" if quoted else repr(allowed[-1])
        raise ValueError(f"{name} must be {message}, got {value!r}")


def _check_unit_interval(
    name: str, value: float, *, exclusive_upper: bool = False
) -> None:
    upper_ok = value < 1.0 if exclusive_upper else value <= 1.0
    if not isfinite(value) or not value >= 0.0 or not upper_ok:
        bound = "[0.0, 1.0)" if exclusive_upper else "[0.0, 1.0]"
        raise ValueError(f"{name} must be finite in {bound}, got {value!r}")


def _check_finite_nonneg(name: str, value: float) -> None:
    if not isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")


CONFORMER_SAMPLING_OPTIONS: tuple[str, ...] = ("boltzmann", "cycle", "mixed")
MATERIAL_TYPE_OPTIONS: tuple[str, ...] = ("slab", "nanoparticle", "porous")
SITE_CLASSIFICATION_OPTIONS: tuple[str, ...] = ("auto", "distance_ratio", "delaunay")
BO_ACQUISITION_OPTIONS: tuple[str, ...] = ("lcb", "ei", "pi")
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
    "gaussian_process",
    "ensemble",
)
BO_TRANSFER_CAPABLE_SURROGATES: tuple[str, ...] = (
    "random_forest",
    "extra_trees",
    "gradient_boost",
    "ridge",
    "ensemble",
)
TS_OPTIMIZER_OPTIONS: tuple[str, ...] = ("fire", "lbfgs", "bfgs")
SLAB_RELAXATION_MODE = Literal["none", "ionic_only", "cell_only", "full"]
SLAB_RELAXATION_OPTIMIZER = Literal["lbfgs", "bfgs", "fire"]
SLAB_RELAXATION_MODE_OPTIONS: tuple[SLAB_RELAXATION_MODE, ...] = (
    "none",
    "ionic_only",
    "cell_only",
    "full",
)
SLAB_RELAXATION_OPTIMIZER_OPTIONS: tuple[SLAB_RELAXATION_OPTIMIZER, ...] = (
    "lbfgs",
    "bfgs",
    "fire",
)


def _validate_placement(root: "AdsorptionConfig") -> None:
    _check_positive("max_closest_approach", root.max_closest_approach)
    _check_positive_int("min_contact_atoms", root.min_contact_atoms)
    _check_positive("contact_distance_threshold", root.contact_distance_threshold)

    if root.num_placements is not None:
        _check_positive_int("num_placements", root.num_placements)

    if root.placement_retry_enabled:
        _check_positive_int(
            "placement_retry_max_attempts", root.placement_retry_max_attempts
        )
        _check_positive_int(
            "placement_retry_diversity_seed_increment",
            root.placement_retry_diversity_seed_increment,
        )
        _check_positive_int(
            "placement_retry_early_stop_patience",
            root.placement_retry_early_stop_patience,
        )
    if root.placement_retry_early_stop_patience < 1:
        raise ValueError(
            "placement_retry_early_stop_patience must be >= 1, "
            f"got {root.placement_retry_early_stop_patience}"
        )
    if root.placement_retry_oversample_max < 1.0:
        raise ValueError(
            "placement_retry_oversample_max must be >= 1.0, "
            f"got {root.placement_retry_oversample_max}"
        )
    if root.placement_materialize_workers == 0:
        raise ValueError(
            "placement_materialize_workers must be != 0 "
            "(positive count, or joblib-style negative: -1=all CPUs, -2=all but one)"
        )

    for pos_name, pos_value in (
        ("min_initial_distance", root.min_initial_distance),
        ("top_layer_tolerance", root.top_layer_tolerance),
        ("symmetry_tolerance", root.symmetry_tolerance),
        ("site_equivalence_tolerance", root.site_equivalence_tolerance),
        ("hollow_site_dedup_tolerance", root.hollow_site_dedup_tolerance),
        ("planar_z_variance_threshold", root.planar_z_variance_threshold),
    ):
        _check_positive(pos_name, pos_value)

    _check_range_tuple("placement_z_range", root.placement_z_range)
    for xy_name, xy_value in (
        ("placement_x_range", root.placement_x_range),
        ("placement_y_range", root.placement_y_range),
    ):
        _check_range_tuple(xy_name, xy_value, allow_equal=True)

    if not 0.5 <= root.min_contact_ratio <= 1.2:
        raise ValueError(
            f"min_contact_ratio must be in [0.5, 1.2], got {root.min_contact_ratio}"
        )
    if root.max_initial_distance is not None and root.max_initial_distance <= 0:
        raise ValueError(
            "max_initial_distance must be positive when set, "
            f"got {root.max_initial_distance}"
        )
    if not 0.0 <= root.flat_aromatic_parallel_fraction <= 1.0:
        raise ValueError(
            "flat_aromatic_parallel_fraction must be in [0.0, 1.0], "
            f"got {root.flat_aromatic_parallel_fraction}"
        )
    if root.voronoi_probe_radius is not None and root.voronoi_probe_radius <= 0:
        raise ValueError(
            f"voronoi_probe_radius must be positive, got {root.voronoi_probe_radius}"
        )
    if (
        root.voronoi_probe_radius is not None
        and root.voronoi_max_site_distance is not None
        and root.voronoi_max_site_distance <= root.voronoi_probe_radius
    ):
        raise ValueError(
            f"voronoi_max_site_distance ({root.voronoi_max_site_distance}) must be "
            f"greater than voronoi_probe_radius ({root.voronoi_probe_radius})"
        )
    if not isinstance(root.voronoi_site_enrichment, bool):
        raise ValueError(
            "voronoi_site_enrichment must be a bool, "
            f"got {type(root.voronoi_site_enrichment).__name__}"
        )
    _check_choice(
        "site_classification_method",
        root.site_classification_method,
        allowed=SITE_CLASSIFICATION_OPTIONS,
    )


def _validate_relaxation(root: "AdsorptionConfig") -> None:
    for int_name, int_value in (
        ("stage1_steps", root.stage1_steps),
        ("stage2_steps", root.stage2_steps),
        ("reference_optimization_steps", root.reference_optimization_steps),
        ("steps_between_swaps", root.steps_between_swaps),
        (
            "saturation_autobatcher_reuse_growth_atoms",
            root.saturation_autobatcher_reuse_growth_atoms,
        ),
    ):
        _check_positive_int(int_name, int_value)

    _check_positive("fmax", root.fmax)
    _check_choice("ts_optimizer", root.ts_optimizer, allowed=TS_OPTIMIZER_OPTIONS)
    _check_choice(
        "slab_relaxation_mode",
        root.slab_relaxation_mode,
        allowed=SLAB_RELAXATION_MODE_OPTIONS,
    )
    _check_choice(
        "slab_relaxation_optimizer",
        root.slab_relaxation_optimizer,
        allowed=SLAB_RELAXATION_OPTIMIZER_OPTIONS,
    )
    _check_positive_int("slab_relaxation_steps", root.slab_relaxation_steps)
    if root.slab_relaxation_fmax is not None:
        _check_positive("slab_relaxation_fmax", root.slab_relaxation_fmax)
    if not 0.1 <= root.autobatcher_max_memory_padding <= 1.0:
        raise ValueError(
            "autobatcher_max_memory_padding must be in [0.1, 1.0], "
            f"got {root.autobatcher_max_memory_padding}"
        )
    if (
        root.autobatcher_max_memory_scaler is not None
        and root.autobatcher_max_memory_scaler <= 0
    ):
        raise ValueError(
            "autobatcher_max_memory_scaler must be positive when set, "
            f"got {root.autobatcher_max_memory_scaler}"
        )
    if root.autobatcher_max_atoms_to_try is not None:
        _check_positive_int(
            "autobatcher_max_atoms_to_try", root.autobatcher_max_atoms_to_try
        )
    if not 0.0 <= root.saturation_autobatcher_reuse_growth_fraction <= 1.0:
        raise ValueError(
            "saturation_autobatcher_reuse_growth_fraction must be in [0.0, 1.0], "
            f"got {root.saturation_autobatcher_reuse_growth_fraction}"
        )


def _validate_bo_transfer(transfer: BOTransferConfig) -> None:
    _check_choice(
        "bo.transfer.mode",
        transfer.mode,
        allowed=("weighted", "cumulative_refit"),
    )
    _check_positive_int(
        "bo.transfer.min_step_observations",
        transfer.min_step_observations,
    )
    _check_positive_int("bo.transfer.trust_patience", transfer.trust_patience)
    if transfer.prior_step_window is not None:
        _check_positive_int(
            "bo.transfer.prior_step_window",
            transfer.prior_step_window,
        )
    _check_positive(
        "bo.transfer.recency_lengthscale",
        transfer.recency_lengthscale,
    )
    _check_positive(
        "bo.transfer.occupancy_lengthscale",
        transfer.occupancy_lengthscale,
    )
    _check_unit_interval("bo.transfer.occupancy_floor", transfer.occupancy_floor)
    _check_unit_interval(
        "bo.transfer.weight_cap", transfer.weight_cap, exclusive_upper=True
    )
    _check_positive(
        "bo.transfer.similarity_lengthscale",
        transfer.similarity_lengthscale,
    )
    _check_unit_interval("bo.transfer.min_similarity", transfer.min_similarity)
    _check_finite_nonneg("bo.transfer.mae_tolerance", transfer.mae_tolerance)
    _check_unit_interval(
        "bo.transfer.exploration_fraction", transfer.exploration_fraction
    )
    _check_positive(
        "bo.transfer.proximity_lengthscale",
        transfer.proximity_lengthscale,
    )
    _check_unit_interval("bo.transfer.proximity_floor", transfer.proximity_floor)


def _validate_bo(root: "AdsorptionConfig") -> None:
    bo = root.bo
    if not isinstance(bo, BOConfig):
        raise ValueError(f"bo must be a BOConfig, got {type(bo).__name__}")
    if bo.initial_random is not None:
        _check_positive_int("bo.initial_random", bo.initial_random)
    if bo.batch_size is not None:
        _check_positive_int("bo.batch_size", bo.batch_size)
    _check_positive_int("bo.total_budget", bo.total_budget)
    if bo.ucb_kappa < 0:
        raise ValueError(f"bo.ucb_kappa must be non-negative, got {bo.ucb_kappa}")
    _check_choice(
        "bo.initial_sampling",
        bo.initial_sampling,
        allowed=BO_INITIAL_SAMPLING_OPTIONS,
    )
    _check_choice(
        "bo.acquisition",
        bo.acquisition,
        allowed=BO_ACQUISITION_OPTIONS,
    )
    _check_choice(
        "bo.surrogate",
        bo.surrogate,
        allowed=BO_SURROGATE_OPTIONS,
    )
    if bo.transfer.enabled and bo.surrogate not in BO_TRANSFER_CAPABLE_SURROGATES:
        raise ValueError(
            "bo.transfer.enabled requires a surrogate that supports "
            "per-sample weights "
            f"({', '.join(BO_TRANSFER_CAPABLE_SURROGATES)}); "
            "sample_weight is not supported for "
            f"bo.surrogate={bo.surrogate!r}"
        )
    if bo.candidate_pool_size is not None:
        _check_positive_int("bo.candidate_pool_size", bo.candidate_pool_size)
    if not isfinite(bo.failure_penalty_default) or bo.failure_penalty_default < 0:
        raise ValueError(
            "bo.failure_penalty_default must be a finite non-negative value, "
            f"got {bo.failure_penalty_default!r}"
        )
    if not isinstance(bo.failure_penalty_overrides, dict):
        raise ValueError(
            "bo.failure_penalty_overrides must be a dict[str, float], "
            f"got {type(bo.failure_penalty_overrides).__name__}"
        )
    for penalty_key, penalty_value in bo.failure_penalty_overrides.items():
        if not isinstance(penalty_key, str) or not penalty_key:
            raise ValueError(
                "bo.failure_penalty_overrides keys must be non-empty strings, "
                f"got {penalty_key!r}"
            )
        if not isfinite(penalty_value) or penalty_value < 0:
            raise ValueError(
                "bo.failure_penalty_overrides values must be finite non-negative, "
                f"got {penalty_value!r} for key {penalty_key!r}"
            )
    if not isinstance(bo.transfer, BOTransferConfig):
        raise ValueError(
            f"bo.transfer must be a BOTransferConfig, got {type(bo.transfer).__name__}"
        )
    _validate_bo_transfer(bo.transfer)


def _validate_io(root: "AdsorptionConfig") -> None:
    _check_positive_int("vasp_nsw", root.vasp_nsw)
    _check_positive_int("vasp_encut", root.vasp_encut)
    if len(root.vasp_kpoints) != 3:
        raise ValueError(
            f"vasp_kpoints must be a 3-tuple, got length {len(root.vasp_kpoints)}"
        )
    for i, k in enumerate(root.vasp_kpoints):
        if not isinstance(k, int) or k <= 0:
            raise ValueError(f"vasp_kpoints[{i}] must be a positive integer, got {k!r}")


def resolved_bo_eval_budget(config: "AdsorptionConfig") -> int:
    """Total BO placement evaluations after auto-resolution of batch sizes."""
    if config.bo.initial_random is None or config.bo.batch_size is None:
        raise ValueError(
            "resolved_bo_eval_budget requires bo.initial_random and bo.batch_size "
            "to be resolved (not None)"
        )
    return config.bo.initial_random + config.bo.total_budget * config.bo.batch_size


def bo_eval_schedule(config: "AdsorptionConfig") -> list[int]:
    """Cumulative placement-evaluation counts for BO replay curves.

    Matches the live workflow: one initial-random batch plus
    ``bo.total_budget`` acquisition batches of ``bo.batch_size`` placements.
    """
    if config.bo.initial_random is None or config.bo.batch_size is None:
        raise ValueError(
            "bo_eval_schedule requires bo.initial_random and bo.batch_size "
            "to be resolved (not None)"
        )
    initial = int(config.bo.initial_random)
    batch = int(config.bo.batch_size)
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
    ``enable_dissociative_placement=True`` (and usually ``skip_topology_check=True``
    so connectivity filters allow fragmented adsorbates). Use ``run_*_bo``
    (or YAML ``campaign: adsorption_bo`` / ``saturation_bo``) for Bayesian
    placement selection; nested ``bo`` / ``bo.transfer`` hold BO hyperparameters
    only. Reference energies remain isolated-molecule energies; positive E_ads
    can result when the relaxed adsorbate dissociates.

    Full field documentation:
    https://metalsurfer.readthedocs.io/en/latest/api/config.html
    """

    model_name: str = "uma-s-1p2"
    num_conformers: int = 10
    num_placements: int | None = None
    device: str = "cuda"
    fmax: float = 0.05
    stage1_steps: int = 50
    stage2_steps: int = 150
    reference_optimization_steps: int = 100
    placement_x_range: tuple[float, float] = (-0.5, 0.5)
    placement_y_range: tuple[float, float] = (-0.5, 0.5)
    placement_z_range: tuple[float, float] = (0.7, 1.25)
    placement_z_scale_by_covalent_radius: bool = True
    placement_distance_recovery: bool = True
    material_type: Literal["slab", "nanoparticle", "porous"] = "slab"
    # ``voronoi_probe_radius`` / ``voronoi_max_site_distance`` / ``voronoi_auto_widen``
    # apply to *every* material type: on slabs they gate the topology generator's
    # accessibility window and drive the one-shot widen retry.
    voronoi_probe_radius: float | None = None
    voronoi_max_site_distance: float | None = None
    # Ridge enrichment of Voronoi vertices. Porous / nanoparticle only: a planar
    # slab top layer has no 3D Voronoi diagram, so slab sites come entirely from
    # the topology generator and this flag is a no-op there.
    voronoi_site_enrichment: bool = True
    voronoi_auto_widen: bool = True
    site_classification_method: Literal["auto", "distance_ratio", "delaunay"] = "auto"
    # Deprecated no-op: conformer choice is now carried explicitly by
    # ``PlacementSpec.conformer_index`` and enumerated by the placement policy,
    # so nothing reads this. Setting it emits a DeprecationWarning.
    conformer_sampling: Literal["boltzmann", "cycle", "mixed"] = "cycle"
    placement_filter: Callable[[PlacementSpec], bool] | None = field(
        default=None, repr=False
    )
    flat_aromatic_parallel_fraction: float = 0.5
    adaptive_parallel_fraction: bool = True
    min_initial_distance: float = MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    min_contact_ratio: float = MIN_CONTACT_RATIO_DEFAULT
    max_initial_distance: float | None = None
    top_layer_tolerance: float = 0.5
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE
    site_equivalence_tolerance: float = DEFAULT_SITE_EQUIVALENCE_TOLERANCE
    hollow_site_dedup_tolerance: float = DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
    planar_z_variance_threshold: float = DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
    rough_slab_local_z: bool = True
    min_interatomic_distance: float = 0.5
    max_force_convergence: float = 0.05
    binding_distance_threshold: float = 4.0
    strict_initial_placement: bool = False
    reject_vdw_overlaps: bool = False
    vdw_overlap_scale: float = 1.0
    max_closest_approach: float = CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM
    min_contact_atoms: int = 1
    contact_distance_threshold: float = CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM
    require_multiple_contact: bool = False
    max_adsorption_energy: float = 5.0
    energy_dedup_threshold: float = 0.05
    rmsd_dedup_threshold: float = 0.1
    connectivity_multipliers: list[float] = field(default_factory=lambda: [1.2, 1.3])
    seed: int = 42
    # Deprecated no-op: see ``conformer_sampling``.
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
    # When False (default), CSV exports keep ML feature geometry + labels only.
    # When True, also write initial_* placement provenance and full ctx_* settings.
    export_placement_provenance: bool = False
    saturation_discard_topology_rearrangements: bool = True
    saturation_max_steps: int | None = None
    skip_topology_check: bool = False
    enable_dissociative_placement: bool = False
    skip_desorption_check: bool = False
    fail_on_missing_reference: bool = False
    fail_on_conformer_failure: bool = False
    debug_write_initial_placements: bool = False
    placement_retry_enabled: bool = True
    placement_retry_max_attempts: int = 8
    placement_retry_diversity_seed_increment: int = 1000
    # Max specs requested per deficit round as a multiple of remaining slots.
    placement_retry_oversample_max: float = 6.0
    # When True, clamp the fill target to the enumerable spec capacity so the
    # retry loop cannot spin until max_attempts on an unreachable target.
    placement_fill_clamp_to_capacity: bool = True
    # Consecutive zero-yield retry attempts before giving up early (a plateau
    # signal). placement_retry_max_attempts remains the absolute hard cap.
    placement_retry_early_stop_patience: int = 2
    # Joblib-style n_jobs for placement materialization threads (-2 = all but one CPU).
    placement_materialize_workers: int = -2
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
    bo: BOConfig = field(default_factory=BOConfig)

    def _warn_deprecated_conformer_selection(self) -> None:
        """Warn when a caller sets one of the two no-op conformer knobs.

        ``conformer_sampling`` / ``boltzmann_temperature`` predate spec-based
        placement. Conformer choice is now carried explicitly by
        ``PlacementSpec.conformer_index`` and enumerated by the placement
        policy, so no code path reads either field. They are still validated and
        accepted so existing configs and YAML campaigns keep loading, but a
        non-default value would silently do nothing, which is worse than saying
        so.
        """
        stale = [
            name
            for name, value, default in (
                ("conformer_sampling", self.conformer_sampling, "cycle"),
                ("boltzmann_temperature", self.boltzmann_temperature, 300.0),
            )
            if value != default
        ]
        if stale:
            warnings.warn(
                f"{', '.join(stale)} no longer affect(s) conformer selection. "
                "Conformers are chosen per placement via "
                "PlacementSpec.conformer_index; use num_conformers and the "
                "placement policy instead. This field is a deprecated no-op and "
                "will be removed in a future release.",
                DeprecationWarning,
                stacklevel=3,
            )

    def __post_init__(self) -> None:
        if isinstance(self.bo, Mapping) and not isinstance(self.bo, BOConfig):
            object.__setattr__(self, "bo", _bo_config_from_mapping(self.bo))
        _validate_placement(self)
        _validate_relaxation(self)
        _validate_io(self)
        _validate_bo(self)

        positive_int_fields: list[tuple[str, int]] = [
            ("num_conformers", self.num_conformers),
        ]
        for int_name, int_value in positive_int_fields:
            _check_positive_int(int_name, int_value)

        positive_fields: list[tuple[str, float]] = [
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

        _check_choice(
            "conformer_sampling",
            self.conformer_sampling,
            allowed=CONFORMER_SAMPLING_OPTIONS,
        )
        self._warn_deprecated_conformer_selection()
        _check_choice(
            "material_type",
            self.material_type,
            allowed=MATERIAL_TYPE_OPTIONS,
        )
        if not self.connectivity_multipliers:
            raise ValueError("connectivity_multipliers must be a non-empty list")
        for i, m in enumerate(self.connectivity_multipliers):
            if m <= 0:
                raise ValueError(
                    f"connectivity_multipliers[{i}] must be positive, got {m}"
                )

        if not self.model_name:
            raise ValueError("model_name must be a non-empty string")

        _check_choice("device", self.device, allowed=("cuda", "cpu"))

        if self.saturation_max_steps is not None:
            _check_positive_int("saturation_max_steps", self.saturation_max_steps)


def resolve_adsorption_config(
    config: AdsorptionConfig | None,
) -> AdsorptionConfig:
    """Return ``config`` or a default :class:`AdsorptionConfig` when ``None``."""
    return config if config is not None else AdsorptionConfig()

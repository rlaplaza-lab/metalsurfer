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
SITE_CLASSIFICATION_OPTIONS: tuple[str, ...] = ("distance_ratio", "delaunay")
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

    Attributes:
        model_name: Name of the MLIP model to use for energy calculations
        num_conformers: Number of conformers to generate for each molecule
        num_placements: Number of placement attempts per conformer
        device: Device to use for MLIP calculations ('cuda' or 'cpu')
        fmax: Maximum force threshold for optimization convergence (eV/Å)
        stage1_steps: Number of optimization steps in stage 1 (coarse optimization)
        stage2_steps: Number of optimization steps in stage 2 (fine optimization)
        reference_optimization_steps: Number of optimization steps for reference calculations
        placement_x_range: Range for x-coordinate placement (Å)
        placement_y_range: Range for y-coordinate placement (Å)
        placement_z_range: Range for z-coordinate placement (Å)
        placement_z_scale_by_covalent_radius: Scale z-placement by adsorbate covalent radius
        material_type: Type of material ('slab', 'nanoparticle', or 'porous')
        voronoi_probe_radius: Minimum distance from framework atoms to Voronoi sites (Å)
        voronoi_max_site_distance: Maximum distance for accessible Voronoi sites (Å)
        voronoi_site_enrichment: Enable geodesic ridge subdivision for denser sites
        site_classification_method: Method for classifying adsorption sites ('distance_ratio' or 'delaunay')
        conformer_sampling: Method for conformer sampling ('boltzmann', 'cycle', or 'mixed')
        placement_filter: Optional function to filter placement specifications
        flat_aromatic_parallel_fraction: Fraction of flat-aromatic placements in horizontal orientation
        adaptive_parallel_fraction: Use molecule-aware estimate for parallel fraction
        min_initial_distance: Minimum initial distance between adsorbate and surface (Å)
        min_contact_ratio: Lower bound for contact distance to avoid covalent binding
        max_initial_distance: Upper bound for initial placement distance (Å, optional)
        top_layer_tolerance: Tolerance for identifying top layer atoms (Å)
        symmetry_tolerance: Tolerance for symmetry detection (Å)
        site_equivalence_tolerance: Tolerance for site equivalence detection (Å)
        hollow_site_dedup_tolerance: Tolerance for hollow site deduplication (Å)
        planar_z_variance_threshold: Maximum z variance for planar classification (Å²)
        rough_slab_local_z: Use local z-reference for rough slabs instead of global max
        relax_top_layer: Allow relaxation of top layer atoms
        freeze_symbols: List of element symbols to freeze during optimization
        min_interatomic_distance: Minimum allowed interatomic distance (Å)
        max_force_convergence: Maximum force for convergence (eV/Å)
        binding_distance_threshold: Maximum distance for binding classification (Å)
        strict_initial_placement: Enable stricter placement validation checks
        reject_vdw_overlaps: Reject placements with van der Waals overlaps
        vdw_overlap_scale: Scale factor for VDW radius sum in overlap detection
        min_contact_distance: Minimum contact distance for binding atoms (Å)
        min_contact_atoms: Minimum number of atoms within contact distance
        contact_distance_threshold: Distance threshold for counting contacting atoms (Å)
        min_adsorbate_separation: Minimum distance between adsorbates in saturation (Å)
        require_multiple_contact: Require multiple atoms in contact region
        max_initial_placement_quality: Quality threshold for initial placements ('any' or 'good')
        max_adsorption_energy: Maximum allowed adsorption energy (eV)
        energy_dedup_threshold: Energy threshold for deduplication (eV)
        rmsd_dedup_threshold: RMSD threshold for deduplication (Å)
        connectivity_multipliers: Multipliers for connectivity analysis
        seed: Random seed for reproducibility
        boltzmann_temperature: Temperature for Boltzmann sampling (K)
        auto_resize_slab: Automatically resize slab to accommodate adsorbates
        min_pbc_image_separation: Minimum separation between periodic images (Å)
        vacuum_box_size: Size of vacuum box for isolated conformer calculations (Å)
        vasp_encut: VASP ENcut parameter (eV)
        vasp_ediff: VASP EDIFF parameter (eV)
        vasp_ediffg: VASP EDIFFG parameter (eV/Å)
        vasp_nsw: VASP NSW parameter (number of steps)
        vasp_kpoints: VASP k-points grid (tuple of 3 integers)
        saturation: Enable saturation coverage calculations
        multi_molecule_saturation: Enable multi-molecule saturation calculations
        skip_topology_check: Skip molecular topology validation checks
        skip_desorption_check: Skip post-optimization desorption validation
        fail_on_missing_reference: Fail if reference calculation is missing
        fail_on_conformer_failure: Fail if conformer generation fails
        debug_write_initial_placements: Write XYZ files of initial placements for debugging
        placement_retry_enabled: Enable placement retry mechanism
        placement_retry_max_attempts: Maximum number of placement retry attempts
        placement_retry_diversity_seed_increment: Seed increment for retry diversity
        optimize_isolated_sequentially: Optimize isolated molecules sequentially
        ts_optimizer: TorchSim optimizer ('fire', 'lbfgs', or 'bfgs')
        steps_between_swaps: Number of steps between optimizer swaps
        autobatcher_max_memory_padding: Memory padding for autobatcher (fraction)
        autobatcher_max_memory_scaler: Memory scaler for autobatcher (optional)
        autobatcher_max_atoms_to_try: Maximum atoms to try in autobatcher (optional)
        saturation_autobatcher_reuse: Reuse autobatcher estimates in saturation
        saturation_autobatcher_reuse_growth_atoms: Growth threshold for autobatcher reuse (atoms)
        saturation_autobatcher_reuse_growth_fraction: Growth threshold for autobatcher reuse (fraction)
        bo_enabled: Enable Bayesian optimization for placement selection
        bo_initial_random: Number of initial random samples for BO
        bo_batch_size: Batch size for BO iterations
        bo_total_budget: Total budget for BO iterations
        bo_ucb_kappa: Kappa parameter for UCB acquisition function
        bo_acquisition: BO acquisition function ('lcb', 'ei', or 'pi')
        bo_surrogate: BO surrogate model ('random_forest', 'extra_trees', 'gradient_boost', or 'ridge')
        bo_candidate_pool_size: Size of candidate pool for BO (optional)
        bo_include_failure_negatives: Include failure penalties in BO
        bo_failure_penalty_default: Default failure penalty for BO
        bo_failure_penalty_overrides: Override failure penalties for specific failure types
        bo_transfer_enabled: Enable transfer learning in BO
        bo_transfer_min_step_observations: Minimum observations for transfer learning
        bo_transfer_weight_cap: Maximum weight for transfer learning
        bo_transfer_similarity_lengthscale: Length scale for transfer similarity
        bo_transfer_min_similarity: Minimum similarity for transfer learning
        bo_transfer_trust_patience: Patience for transfer trust evaluation
        bo_transfer_mae_tolerance: MAE tolerance for transfer learning
        bo_transfer_exploration_fraction: Exploration fraction for transfer learning
    """

    model_name: str = (
        "uma-s-1p1"  # Name of the MLIP model to use for energy calculations
    )
    num_conformers: int = 10  # Number of conformers to generate for each molecule
    num_placements: int = 100  # Number of placement attempts per conformer
    device: str = "cuda"  # Device to use for MLIP calculations ('cuda' or 'cpu')
    fmax: float = 0.05  # Maximum force threshold for optimization convergence (eV/Å)
    stage1_steps: int = (
        50  # Number of optimization steps in stage 1 (coarse optimization)
    )
    stage2_steps: int = (
        150  # Number of optimization steps in stage 2 (fine optimization)
    )
    reference_optimization_steps: int = (
        100  # Number of optimization steps for reference calculations
    )
    placement_x_range: tuple[float, float] = (
        -4.0,
        4.0,
    )  # Range for x-coordinate placement (Å)
    placement_y_range: tuple[float, float] = (
        -4.0,
        4.0,
    )  # Range for y-coordinate placement (Å)
    placement_z_range: tuple[float, float] = (
        2.0,
        3.0,
    )  # Range for z-coordinate placement (Å)
    placement_z_scale_by_covalent_radius: bool = (
        True  # Scale z-placement by adsorbate covalent radius
    )
    # Explicit material type: "slab" (2D periodic), "nanoparticle" (0D), or "porous" (3D periodic)
    material_type: Literal["slab", "nanoparticle", "porous"] = "slab"
    # Voronoi site generation parameters
    voronoi_probe_radius: float = 1.2  # min distance from framework atom to site (Å)
    voronoi_max_site_distance: float = 4.0  # max distance for accessible sites (Å)
    voronoi_site_enrichment: bool = True  # geodesic ridge subdivision for denser sites
    # Site classification: "distance_ratio" (default) or "delaunay" (slab only).
    site_classification_method: Literal["distance_ratio", "delaunay"] = "distance_ratio"
    conformer_sampling: Literal["boltzmann", "cycle", "mixed"] = (
        "cycle"  # Method for conformer sampling
    )
    placement_filter: Callable[[PlacementSpec], bool] | None = field(
        default=None, repr=False
    )  # Optional function to filter placement specifications
    flat_aromatic_parallel_fraction: float = (
        0.5  # Fraction of flat-aromatic placements in horizontal (π-stacking) vs EN-down.
        # Default 0.5 ensures both orientations are explored.
    )
    # When True, override flat_aromatic_parallel_fraction with a molecule-aware
    # estimate (high for pure aromatics, low for strong EN-down binders).
    adaptive_parallel_fraction: bool = False
    min_initial_distance: float = (
        1.5  # Minimum initial distance between adsorbate and surface (Å)
    )
    min_contact_ratio: float = (
        0.8  # Lower bound: (r_mol+r_surf)*ratio avoids covalent binding
    )
    max_initial_distance: float | None = (
        None  # Upper bound for initial placement distance (Å, optional)
    )
    top_layer_tolerance: float = 0.5  # Tolerance for identifying top layer atoms (Å)
    symmetry_tolerance: float = 0.1  # Tolerance for symmetry detection (Å)
    site_equivalence_tolerance: float = (
        0.05  # Tolerance for site equivalence detection (Å)
    )
    hollow_site_dedup_tolerance: float = (
        0.1  # Tolerance for hollow site deduplication (Å)
    )
    planar_z_variance_threshold: float = (
        0.01  # Max z variance (Å²) for planar classification
    )
    # When True, use per-site local z as surface reference on rough (non-planar)
    # slabs instead of the global np.max(z).
    rough_slab_local_z: bool = True
    relax_top_layer: bool = True  # Allow relaxation of top layer atoms
    freeze_symbols: list[str] | None = (
        None  # List of element symbols to freeze during optimization
    )
    min_interatomic_distance: float = 0.5  # Minimum allowed interatomic distance (Å)
    max_force_convergence: float = 0.05  # Maximum force for convergence (eV/Å)
    binding_distance_threshold: float = (
        4.0  # Post-opt: reject if adsorbate-surface > this (desorbed)
    )
    # Enhanced placement validation (Phase 1-3 improvements)
    strict_initial_placement: bool = False  # Enable all stricter placement checks
    reject_vdw_overlaps: bool = (
        False  # Reject placements with VDW overlaps (stricter than covalent)
    )
    vdw_overlap_scale: float = (
        1.0  # Scale factor for VDW radius sum (>1 more strict, <1 more lenient)
    )
    min_contact_distance: float = 0.8  # Minimum contact distance (Å) for binding atom
    min_contact_atoms: int = (
        1  # Require N molecule atoms within contact_distance_threshold
    )
    contact_distance_threshold: float = (
        2.5  # Distance threshold (Å) for counting "contacting" atoms
    )
    min_adsorbate_separation: float = (
        2.0  # In saturation: minimum distance between adsorbates
    )
    require_multiple_contact: bool = (
        False  # Require multiple atoms in contact region (contact quality)
    )
    max_initial_placement_quality: str = (
        "any"  # "any" (default) or "good" (multiple contacts)
    )
    max_adsorption_energy: float = 5.0  # Maximum allowed adsorption energy (eV)
    energy_dedup_threshold: float = 0.05  # Energy threshold for deduplication (eV)
    rmsd_dedup_threshold: float = 0.1  # RMSD threshold for deduplication (Å)
    connectivity_multipliers: list[float] = field(
        default_factory=lambda: [1.2, 1.3]
    )  # Multipliers for connectivity analysis
    seed: int = 42  # Random seed for reproducibility
    boltzmann_temperature: float = 300.0  # Temperature for Boltzmann sampling (K)
    auto_resize_slab: bool = True  # Automatically resize slab to accommodate adsorbates
    min_pbc_image_separation: float = (
        8.0  # Minimum separation between periodic images (Å)
    )
    # Isolated conformer / gas-phase box edge length (Å) for conformers module
    vacuum_box_size: float = 20.0
    vasp_encut: int = 400  # VASP ENcut parameter (eV)
    vasp_ediff: float = 1e-6  # VASP EDIFF parameter (eV)
    vasp_ediffg: float = -0.02  # VASP EDIFFG parameter (eV/Å)
    vasp_nsw: int = 100  # VASP NSW parameter (number of steps)
    vasp_kpoints: tuple[int, int, int] = (
        4,
        4,
        1,
    )  # VASP k-points grid (tuple of 3 integers)
    saturation: bool = False  # Enable saturation coverage calculations
    multi_molecule_saturation: bool = (
        False  # Enable multi-molecule saturation calculations
    )
    skip_topology_check: bool = False  # Skip molecular topology validation checks
    skip_desorption_check: bool = False  # Skip post-optimization desorption validation

    # Strict workflow: fail instead of skipping molecules
    fail_on_missing_reference: bool = False  # Fail if reference calculation is missing
    fail_on_conformer_failure: bool = False  # Fail if conformer generation fails

    # Debug: write XYZ files of initial placements (before optimization) to the same
    # xyz_structures/{molecule}_all/ dir as final conformer XYZs (as initial_*.xyz)
    debug_write_initial_placements: bool = False

    # Placement retry configuration: attempt to generate requested number of valid
    # placements by retrying failed specs with different random seeds.
    placement_retry_enabled: bool = True  # Enable placement retry mechanism
    placement_retry_max_attempts: int = 3  # Maximum number of placement retry attempts
    placement_retry_diversity_seed_increment: int = (
        1000  # Seed increment for retry diversity
    )

    # TorchSim optimisation tuning
    optimize_isolated_sequentially: bool = (
        False  # Optimize isolated molecules sequentially
    )
    ts_optimizer: Literal["fire", "lbfgs", "bfgs"] = "fire"  # TorchSim optimizer
    steps_between_swaps: int = 5  # Number of steps between optimizer swaps

    # TorchSim autobatcher tuning (helps avoid CUDA OOM)
    autobatcher_max_memory_padding: float = (
        0.5  # Memory padding for autobatcher (fraction)
    )
    autobatcher_max_memory_scaler: float | None = (
        None  # Memory scaler for autobatcher (optional)
    )
    # When set, forces TorchSim memory-estimation probe cap. When None, Metalsurfer
    # computes a conservative per-call cap from current workload.
    autobatcher_max_atoms_to_try: int | None = (
        None  # Maximum atoms to try in autobatcher (optional)
    )
    # Saturation-only: allow reusing prior autobatcher estimate for small size growth
    saturation_autobatcher_reuse: bool = (
        True  # Reuse autobatcher estimates in saturation
    )
    saturation_autobatcher_reuse_growth_atoms: int = (
        32  # Growth threshold for autobatcher reuse (atoms)
    )
    saturation_autobatcher_reuse_growth_fraction: float = (
        0.1  # Growth threshold for autobatcher reuse (fraction)
    )

    # Bayesian optimisation (surrogate-guided placement selection)
    bo_enabled: bool = False  # Enable Bayesian optimization for placement selection
    bo_initial_random: int = 10  # Number of initial random samples for BO
    bo_batch_size: int = 10  # Batch size for BO iterations
    bo_total_budget: int = 100  # Total budget for BO iterations
    # Lower kappa biases toward exploitation; empirically improves early-best
    # discovery on heterogeneous graphene test runs.
    bo_ucb_kappa: float = 1.0  # Kappa parameter for UCB acquisition function
    bo_acquisition: Literal["lcb", "ei", "pi"] = "lcb"  # BO acquisition function
    bo_surrogate: Literal[
        "random_forest",
        "extra_trees",
        "gradient_boost",
        "ridge",
    ] = "random_forest"  # BO surrogate model
    bo_candidate_pool_size: int | None = (
        None  # Size of candidate pool for BO (optional)
    )
    bo_include_failure_negatives: bool = True  # Include failure penalties in BO
    bo_failure_penalty_default: float = 10.0  # Default failure penalty for BO
    bo_failure_penalty_overrides: dict[str, float] = field(
        default_factory=lambda: {
            "generation": 18.0,
            "optimization": 20.0,
            "validation": 14.0,
            "energy_cap": 12.0,
            "filter": 11.0,
        }
    )  # Override failure penalties for specific failure types
    bo_transfer_enabled: bool = False  # Enable transfer learning in BO
    bo_transfer_min_step_observations: int = (
        5  # Minimum observations for transfer learning
    )
    bo_transfer_weight_cap: float = 0.35  # Maximum weight for transfer learning
    bo_transfer_similarity_lengthscale: float = (
        1.0  # Length scale for transfer similarity
    )
    bo_transfer_min_similarity: float = 0.05  # Minimum similarity for transfer learning
    bo_transfer_trust_patience: int = 2  # Patience for transfer trust evaluation
    bo_transfer_mae_tolerance: float = 0.0  # MAE tolerance for transfer learning
    bo_transfer_exploration_fraction: float = (
        0.2  # Exploration fraction for transfer learning
    )

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

        # Placement retry validation
        if self.placement_retry_enabled:
            _check_positive_int(
                "placement_retry_max_attempts", self.placement_retry_max_attempts
            )
            _check_positive_int(
                "placement_retry_diversity_seed_increment",
                self.placement_retry_diversity_seed_increment,
            )

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
            if self.bo_transfer_enabled and self.bo_surrogate in (
                "gradient_boost",
                "ridge",
            ):
                raise ValueError(
                    "bo_transfer_enabled requires a tree ensemble surrogate "
                    "(random_forest or extra_trees); per-sample weights are not supported "
                    f"for bo_surrogate={self.bo_surrogate!r}"
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

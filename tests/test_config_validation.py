"""Tests for AdsorptionConfig validation rules."""

import pytest

from metalsurfer import _numeric_defaults as numeric_defaults
from metalsurfer.config import (
    AdsorptionConfig,
    BOConfig,
    BOTransferConfig,
    bo_eval_schedule,
    fold_bo_config,
    resolved_bo_eval_budget,
)
from metalsurfer.ml.schema import ComputationContext
from metalsurfer.placement import _constants as placement_constants
from metalsurfer.placement import geometry as placement_geometry
from metalsurfer.symmetry import SymmetryAnalyzer

from .conftest import make_slab

# ---------------------------------------------------------------------------
# valid construction
# ---------------------------------------------------------------------------


def test_valid_device_values():
    AdsorptionConfig(device="cuda")
    AdsorptionConfig(device="cpu")
    AdsorptionConfig(device="cuda:0")
    AdsorptionConfig(device="cuda:3")


@pytest.mark.parametrize("device", ["gpu", "", "CUDA", "cuda:abc", "cuda:-1"])
def test_invalid_device_raises(device):
    with pytest.raises(ValueError, match="device"):
        AdsorptionConfig(device=device)


def test_positive_int_rejects_bool():
    with pytest.raises(ValueError, match="positive integer"):
        AdsorptionConfig(num_conformers=True)


def test_initial_distance_order_validated():
    with pytest.raises(ValueError, match="min_initial_distance must be <="):
        AdsorptionConfig(min_initial_distance=5.0, max_initial_distance=2.0)


def test_default_config():
    config = AdsorptionConfig()
    assert config.model_name == "uma-s-1p2"
    assert config.num_conformers == 10
    assert config.fmax == 0.05
    assert config.write_vasp_inputs is False
    assert config.adaptive_parallel_fraction is True
    assert config.placement_distance_recovery is True
    assert config.voronoi_auto_widen is True
    assert config.placement_x_range == (-0.5, 0.5)
    assert config.placement_y_range == (-0.5, 0.5)
    assert config.placement_retry_oversample_max == 6.0
    assert config.placement_retry_max_attempts == 8
    assert config.placement_fill_clamp_to_capacity is True
    assert config.placement_retry_early_stop_patience == 2
    assert config.placement_materialize_workers == -2
    assert (
        config.min_initial_distance
        == numeric_defaults.MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    )
    assert config.min_contact_ratio == numeric_defaults.MIN_CONTACT_RATIO_DEFAULT
    assert (
        config.max_closest_approach
        == numeric_defaults.CONTACT_MAX_CLOSEST_APPROACH_ANGSTROM
    )
    assert (
        config.contact_distance_threshold
        == numeric_defaults.CONTACT_DISTANCE_THRESHOLD_DEFAULT_ANGSTROM
    )
    assert config.symmetry_tolerance == numeric_defaults.DEFAULT_SYMMETRY_TOLERANCE
    assert (
        config.site_equivalence_tolerance
        == numeric_defaults.DEFAULT_SITE_EQUIVALENCE_TOLERANCE
    )
    assert (
        config.hollow_site_dedup_tolerance
        == numeric_defaults.DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE
    )
    assert (
        config.planar_z_variance_threshold
        == numeric_defaults.DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
    )


def test_placement_retry_oversample_max_rejects_below_one():
    with pytest.raises(ValueError, match="placement_retry_oversample_max"):
        AdsorptionConfig(placement_retry_oversample_max=0.5)


def test_placement_retry_early_stop_patience_defaults_and_rejects_below_one():
    config = AdsorptionConfig()
    assert config.placement_retry_early_stop_patience == 2
    assert config.placement_fill_clamp_to_capacity is True
    with pytest.raises(ValueError, match="placement_retry_early_stop_patience"):
        AdsorptionConfig(placement_retry_early_stop_patience=0)
    with pytest.raises(ValueError, match="placement_retry_early_stop_patience"):
        AdsorptionConfig(placement_retry_early_stop_patience=-1)


def test_placement_materialize_workers_rejects_zero():
    with pytest.raises(ValueError, match="placement_materialize_workers"):
        AdsorptionConfig(placement_materialize_workers=0)


def test_numeric_defaults_single_source_of_truth():
    """Config, ML context, symmetry, and placement share the same defaults."""
    cfg = AdsorptionConfig()
    ctx = ComputationContext()
    assert ctx.min_initial_distance == cfg.min_initial_distance
    assert ctx.min_contact_ratio == cfg.min_contact_ratio
    assert ctx.symmetry_tolerance == cfg.symmetry_tolerance
    assert ctx.site_equivalence_tolerance == cfg.site_equivalence_tolerance
    assert ctx.hollow_site_dedup_tolerance == cfg.hollow_site_dedup_tolerance
    assert ctx.planar_z_variance_threshold == cfg.planar_z_variance_threshold

    analyzer = SymmetryAnalyzer(make_slab())
    assert analyzer.symmetry_tolerance == numeric_defaults.DEFAULT_SYMMETRY_TOLERANCE

    assert (
        placement_constants._MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
        == numeric_defaults.MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    )
    assert (
        placement_constants._MIN_CONTACT_RATIO_DEFAULT
        == numeric_defaults.MIN_CONTACT_RATIO_DEFAULT
    )
    assert (
        placement_geometry.check_initial_placement_distance.__defaults__[0]
        == numeric_defaults.MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM
    )
    assert (
        placement_geometry.check_initial_placement_distance.__defaults__[1]
        == numeric_defaults.MIN_CONTACT_RATIO_DEFAULT
    )


def test_valid_custom_config():
    config = AdsorptionConfig(
        num_conformers=5,
        num_placements=50,
        fmax=0.01,
        placement_z_range=(1.5, 4.0),
    )
    assert config.num_conformers == 5
    assert config.placement_z_range == (1.5, 4.0)


@pytest.mark.parametrize(
    "field, default, override",
    [
        ("optimize_isolated_sequentially", False, True),
        ("adaptive_parallel_fraction", True, False),
        ("placement_z_scale_by_covalent_radius", True, False),
        ("export_placement_provenance", False, True),
        ("voronoi_site_enrichment", True, False),
    ],
)
def test_bool_config_default_and_override(field, default, override):
    assert getattr(AdsorptionConfig(), field) is default
    assert getattr(AdsorptionConfig(**{field: override}), field) is override


def test_site_classification_method_auto_default_and_invalid():
    assert AdsorptionConfig().site_classification_method == "auto"
    c = AdsorptionConfig(site_classification_method="auto")
    assert c.site_classification_method == "auto"
    with pytest.raises(ValueError, match="site_classification_method"):
        AdsorptionConfig(site_classification_method="invalid")


def test_ts_optimizer_defaults():
    config = AdsorptionConfig()
    assert config.ts_optimizer == "fire"
    assert config.steps_between_swaps == 5


def test_ts_optimizer_custom():
    config = AdsorptionConfig(ts_optimizer="lbfgs", steps_between_swaps=10)
    assert config.ts_optimizer == "lbfgs"
    assert config.steps_between_swaps == 10


def test_slab_relaxation_defaults():
    config = AdsorptionConfig()
    assert config.slab_relaxation_mode == "ionic_only"
    assert config.slab_relaxation_optimizer == "lbfgs"
    assert config.slab_relaxation_fmax is None
    assert config.slab_relaxation_steps == 200


def test_finalize_substrate_default_freezes_entire_substrate():
    from metalsurfer.surface_prep import finalize_substrate

    slab = make_slab(nx=2, ny=2, n_layers=2)
    config = AdsorptionConfig(material_type="slab")
    constrained = finalize_substrate(slab, config, align=False)
    assert len(constrained.atoms.constraints) == 1
    frozen = constrained.atoms.constraints[0].get_indices()
    assert len(frozen) == len(slab)


@pytest.mark.parametrize("mode", ["none", "ionic_only", "cell_only", "full"])
def test_slab_relaxation_mode_values(mode):
    config = AdsorptionConfig(slab_relaxation_mode=mode)
    assert config.slab_relaxation_mode == mode


def test_slab_relaxation_custom_values():
    config = AdsorptionConfig(
        slab_relaxation_mode="full",
        slab_relaxation_optimizer="bfgs",
        slab_relaxation_fmax=0.02,
        slab_relaxation_steps=123,
    )
    assert config.slab_relaxation_mode == "full"
    assert config.slab_relaxation_optimizer == "bfgs"
    assert config.slab_relaxation_fmax == 0.02
    assert config.slab_relaxation_steps == 123


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"ts_optimizer": "adam"}, "ts_optimizer"),
        ({"steps_between_swaps": -1}, "steps_between_swaps"),
        ({"slab_relaxation_mode": "invalid"}, "slab_relaxation_mode"),
        (
            {"slab_relaxation_optimizer": "invalid"},
            "slab_relaxation_optimizer",
        ),
        ({"slab_relaxation_fmax": 0.0}, "slab_relaxation_fmax"),
        ({"slab_relaxation_steps": 0}, "slab_relaxation_steps"),
    ],
)
def test_optimizer_config_invalid_rejected(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        AdsorptionConfig(**kwargs)


def test_autobatcher_config_defaults():
    config = AdsorptionConfig()
    assert config.autobatcher_max_memory_padding == 0.5
    assert config.autobatcher_max_memory_scaler is None
    assert config.autobatcher_max_atoms_to_try is None
    assert config.saturation_autobatcher_reuse is True
    assert config.saturation_autobatcher_reuse_growth_atoms == 32
    assert config.saturation_autobatcher_reuse_growth_fraction == 0.1


def test_autobatcher_config_custom():
    config = AdsorptionConfig(
        autobatcher_max_memory_padding=0.7,
        autobatcher_max_memory_scaler=500.0,
        autobatcher_max_atoms_to_try=50_000,
        saturation_autobatcher_reuse=False,
        saturation_autobatcher_reuse_growth_atoms=16,
        saturation_autobatcher_reuse_growth_fraction=0.25,
    )
    assert config.autobatcher_max_memory_padding == 0.7
    assert config.autobatcher_max_memory_scaler == 500.0
    assert config.autobatcher_max_atoms_to_try == 50_000
    assert config.saturation_autobatcher_reuse is False
    assert config.saturation_autobatcher_reuse_growth_atoms == 16
    assert config.saturation_autobatcher_reuse_growth_fraction == 0.25


def test_autobatcher_config_custom_none_probe_cap():
    config = AdsorptionConfig(autobatcher_max_atoms_to_try=None)
    assert config.autobatcher_max_atoms_to_try is None


def test_autobatcher_max_memory_padding_out_of_range_rejected():
    with pytest.raises(ValueError, match="autobatcher_max_memory_padding.*0.1"):
        AdsorptionConfig(autobatcher_max_memory_padding=0.05)
    with pytest.raises(ValueError, match="autobatcher_max_memory_padding"):
        AdsorptionConfig(autobatcher_max_memory_padding=1.5)


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"autobatcher_max_memory_scaler": -1.0}, "autobatcher_max_memory_scaler"),
        ({"autobatcher_max_atoms_to_try": 0}, "autobatcher_max_atoms_to_try.*positive"),
        (
            {"saturation_autobatcher_reuse_growth_atoms": 0},
            "saturation_autobatcher_reuse_growth_atoms.*positive",
        ),
        (
            {"saturation_autobatcher_reuse_growth_fraction": -0.01},
            "saturation_autobatcher_reuse_growth_fraction",
        ),
        (
            {"saturation_autobatcher_reuse_growth_fraction": 1.1},
            "saturation_autobatcher_reuse_growth_fraction",
        ),
    ],
)
def test_autobatcher_invalid_rejected(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        AdsorptionConfig(**kwargs)


# ---------------------------------------------------------------------------
# positive integer checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "num_conformers",
        "num_placements",
        "stage1_steps",
        "stage2_steps",
        "reference_optimization_steps",
        "vasp_nsw",
        "vasp_encut",
    ],
)
def test_zero_positive_int_rejected(field):
    with pytest.raises(ValueError, match=f"{field}.*positive integer"):
        AdsorptionConfig(**{field: 0})


@pytest.mark.parametrize(
    "field",
    [
        "num_conformers",
        "num_placements",
        "stage1_steps",
        "stage2_steps",
    ],
)
def test_negative_int_rejected(field):
    with pytest.raises(ValueError, match=f"{field}.*positive integer"):
        AdsorptionConfig(**{field: -1})


# ---------------------------------------------------------------------------
# positive float checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "fmax",
        "min_initial_distance",
        "top_layer_tolerance",
        "planar_z_variance_threshold",
        "min_interatomic_distance",
        "max_force_convergence",
        "binding_distance_threshold",
        "max_adsorption_energy",
        "vacuum_box_size",
        "boltzmann_temperature",
    ],
)
def test_zero_positive_float_rejected(field):
    with pytest.raises(ValueError, match=f"{field}.*positive"):
        AdsorptionConfig(**{field: 0.0})


@pytest.mark.parametrize(
    "field",
    [
        "fmax",
        "min_initial_distance",
        "top_layer_tolerance",
        "planar_z_variance_threshold",
    ],
)
def test_negative_float_rejected(field):
    with pytest.raises(ValueError, match=f"{field}.*positive"):
        AdsorptionConfig(**{field: -0.1})


# ---------------------------------------------------------------------------
# non-negative float checks
# ---------------------------------------------------------------------------


def test_negative_energy_dedup_rejected():
    with pytest.raises(ValueError, match="energy_dedup_threshold.*non-negative"):
        AdsorptionConfig(energy_dedup_threshold=-0.01)


def test_zero_energy_dedup_accepted():
    config = AdsorptionConfig(energy_dedup_threshold=0.0)
    assert config.energy_dedup_threshold == 0.0


# ---------------------------------------------------------------------------
# range tuple checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"placement_x_range": (1.0,)}, "placement_x_range.*2-tuple.*length"),
        ({"placement_x_range": (1.0, 2.0, 3.0)}, "placement_x_range.*2-tuple.*length"),
        ({"placement_z_range": (5.0, 2.0)}, "placement_z_range.*lower bound"),
        ({"placement_z_range": (2.0, 2.0)}, "placement_z_range.*lower bound"),
        ({"placement_x_range": (4.0, -4.0)}, "placement_x_range.*lower bound"),
    ],
)
def test_placement_range_invalid_rejected(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        AdsorptionConfig(**kwargs)


def test_equal_xy_range_allowed_disables_lateral_recovery():
    """Equal XY bounds are valid and mean no in-plane distance recovery."""
    cfg = AdsorptionConfig(
        placement_x_range=(0.0, 0.0),
        placement_y_range=(0.0, 0.0),
    )
    assert cfg.placement_x_range == (0.0, 0.0)
    assert cfg.placement_y_range == (0.0, 0.0)


# ---------------------------------------------------------------------------
# connectivity multipliers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "multipliers",
    [[], [1.2, -0.5], [0.0]],
)
def test_invalid_connectivity_multipliers_rejected(multipliers):
    match = "non-empty" if multipliers == [] else "positive"
    with pytest.raises(ValueError, match=f"connectivity_multipliers.*{match}"):
        AdsorptionConfig(connectivity_multipliers=multipliers)


# ---------------------------------------------------------------------------
# VASP kpoints
# ---------------------------------------------------------------------------


def test_invalid_kpoints_length():
    with pytest.raises(ValueError, match="vasp_kpoints.*3-tuple"):
        AdsorptionConfig(vasp_kpoints=(4, 4))


@pytest.mark.parametrize("kpoints", [(4, 0, 1), (0, 4, 1), (4, 4, 0)])
def test_zero_kpoint_rejected(kpoints):
    with pytest.raises(ValueError, match="vasp_kpoints.*positive integer"):
        AdsorptionConfig(vasp_kpoints=kpoints)


# ---------------------------------------------------------------------------
# model_name
# ---------------------------------------------------------------------------


def test_empty_model_name_rejected():
    with pytest.raises(ValueError, match="model_name.*non-empty"):
        AdsorptionConfig(model_name="")


# ---------------------------------------------------------------------------
# min_contact_ratio, max_initial_distance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["cycle", "boltzmann", "mixed"])
def test_conformer_sampling_valid_values(mode):
    """conformer_sampling accepts cycle, boltzmann, mixed."""
    config = AdsorptionConfig(conformer_sampling=mode)
    assert config.conformer_sampling == mode


def test_conformer_sampling_invalid_rejected():
    with pytest.raises(ValueError, match="conformer_sampling"):
        AdsorptionConfig(conformer_sampling="invalid")


def test_flat_aromatic_parallel_fraction_valid():
    """flat_aromatic_parallel_fraction accepts values in [0, 1]."""
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=0.5)
    assert config.flat_aromatic_parallel_fraction == 0.5
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=0.0)
    assert config.flat_aromatic_parallel_fraction == 0.0
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=1.0)
    assert config.flat_aromatic_parallel_fraction == 1.0


@pytest.mark.parametrize("fraction", [-0.1, 1.5])
def test_flat_aromatic_parallel_fraction_out_of_range_rejected(fraction):
    with pytest.raises(ValueError, match="flat_aromatic_parallel_fraction"):
        AdsorptionConfig(flat_aromatic_parallel_fraction=fraction)


def test_voronoi_site_enrichment_requires_bool():
    with pytest.raises(ValueError, match="voronoi_site_enrichment"):
        AdsorptionConfig(voronoi_site_enrichment="yes")


def test_min_contact_ratio_valid():
    config = AdsorptionConfig(min_contact_ratio=0.9)
    assert config.min_contact_ratio == 0.9


@pytest.mark.parametrize("ratio", [0.3, 1.5])
def test_min_contact_ratio_out_of_range_rejected(ratio):
    with pytest.raises(ValueError, match="min_contact_ratio"):
        AdsorptionConfig(min_contact_ratio=ratio)


def test_max_initial_distance_optional():
    config = AdsorptionConfig()
    assert config.max_initial_distance is None


def test_max_initial_distance_positive_when_set():
    config = AdsorptionConfig(max_initial_distance=3.5)
    assert config.max_initial_distance == 3.5


def test_max_initial_distance_zero_rejected():
    with pytest.raises(ValueError, match="max_initial_distance"):
        AdsorptionConfig(max_initial_distance=0.0)


# ---------------------------------------------------------------------------
# min_pbc_image_separation defaults
# ---------------------------------------------------------------------------


def test_default_min_pbc_image_separation():
    config = AdsorptionConfig()
    assert config.min_pbc_image_separation == 8.0


@pytest.mark.parametrize("separation", [0.0, -1.0])
def test_non_positive_min_pbc_image_separation_rejected(separation):
    with pytest.raises(ValueError, match="min_pbc_image_separation.*positive"):
        AdsorptionConfig(min_pbc_image_separation=separation)


# ---------------------------------------------------------------------------
# actionable error messages
# ---------------------------------------------------------------------------


def test_error_message_includes_field_and_value():
    try:
        AdsorptionConfig(fmax=-0.5)
    except ValueError as e:
        msg = str(e)
        assert "fmax" in msg
        assert "-0.5" in msg


# ---------------------------------------------------------------------------
# Bayesian optimisation (BO) config
# ---------------------------------------------------------------------------


def test_bo_defaults():
    c = AdsorptionConfig()
    assert c.num_placements is None
    assert c.bo.initial_random is None
    assert c.bo.initial_sampling == "spread_xyz"
    assert c.bo.batch_size is None
    assert c.bo.total_budget == 18
    assert c.bo.ucb_kappa == 1.96
    assert c.bo.acquisition == "ei"
    assert c.bo.surrogate == "gradient_boost"
    assert c.bo.candidate_pool_size is None
    assert c.bo.include_failure_negatives is True
    assert c.bo.failure_penalty_default == 10.0
    assert c.bo.failure_penalty_overrides["generation"] > 0.0
    assert c.bo.transfer.enabled is True
    assert c.bo.transfer.mode == "weighted"
    assert c.bo.transfer.min_step_observations == 5
    assert c.bo.transfer.weight_cap == 0.35
    assert c.bo.transfer.similarity_lengthscale == 4.0
    assert c.bo.transfer.min_similarity == 0.05
    assert c.bo.transfer.trust_patience == 2
    assert c.bo.transfer.mae_tolerance == 0.05
    assert c.bo.transfer.exploration_fraction == 0.2
    assert c.bo.transfer.proximity_lengthscale == 1.0
    assert c.bo.transfer.proximity_floor == 0.0
    assert c.bo.transfer.prior_step_window == 2
    assert c.bo.transfer.recency_lengthscale == 4.0
    assert c.bo.transfer.occupancy_lengthscale == 1.0
    assert c.bo.transfer.occupancy_floor == 0.0


def test_resolved_bo_eval_budget():
    config = AdsorptionConfig(
        bo=BOConfig(initial_random=10, batch_size=5, total_budget=18),
    )
    assert resolved_bo_eval_budget(config) == 10 + 18 * 5


def test_bo_eval_schedule():
    config = AdsorptionConfig(
        bo=BOConfig(initial_random=10, batch_size=5, total_budget=18),
    )
    assert bo_eval_schedule(config) == [
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        45,
        50,
        55,
        60,
        65,
        70,
        75,
        80,
        85,
        90,
        95,
        100,
    ]


@pytest.mark.parametrize(
    ("kwargs", "error_match"),
    [
        ({"bo": {"initial_sampling": "latin_hypercube"}}, "bo.initial_sampling"),
        ({"bo": {"ucb_kappa": -1.0}}, "bo.ucb_kappa"),
        ({"bo": {"acquisition": "invalid"}}, "bo.acquisition"),
        ({"bo": {"surrogate": "invalid"}}, "bo.surrogate"),
        ({"bo": {"candidate_pool_size": 0}}, "bo.candidate_pool_size"),
        ({"bo": {"failure_penalty_default": -1.0}}, "bo.failure_penalty_default"),
        (
            {"bo": {"failure_penalty_overrides": {"validation": -0.1}}},
            "bo.failure_penalty_overrides values",
        ),
        ({"bo": {"transfer": {"weight_cap": 1.0}}}, "bo.transfer.weight_cap"),
        (
            {"bo": {"transfer": {"proximity_lengthscale": 0.0}}},
            "bo.transfer.proximity_lengthscale",
        ),
        (
            {"bo": {"transfer": {"proximity_floor": 1.5}}},
            "bo.transfer.proximity_floor",
        ),
    ],
)
def test_bo_invalid_config_rejected(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        AdsorptionConfig(**kwargs)


def test_bo_candidate_pool_size_and_failure_penalty_accepted():
    c = AdsorptionConfig(
        bo=BOConfig(
            candidate_pool_size=500,
            failure_penalty_default=22.5,
            failure_penalty_overrides={"validation": 17.0},
        ),
    )
    assert c.bo.candidate_pool_size == 500
    assert c.bo.failure_penalty_default == 22.5
    assert c.bo.failure_penalty_overrides["validation"] == 17.0


def test_bo_transfer_config_validation():
    c = AdsorptionConfig(
        bo=BOConfig(
            transfer=BOTransferConfig(
                enabled=True,
                weight_cap=0.25,
                min_step_observations=3,
                similarity_lengthscale=0.5,
                min_similarity=0.1,
                trust_patience=3,
                mae_tolerance=0.02,
                exploration_fraction=0.15,
                proximity_lengthscale=0.5,
                proximity_floor=0.1,
            ),
        ),
    )
    assert c.bo.transfer.enabled is True
    assert c.bo.transfer.weight_cap == 0.25
    assert c.bo.transfer.proximity_lengthscale == 0.5
    assert c.bo.transfer.proximity_floor == 0.1


def test_bo_transfer_requires_weighted_surrogate():
    with pytest.raises(ValueError, match="bo.transfer.enabled requires"):
        AdsorptionConfig(
            bo=BOConfig(
                surrogate="gaussian_process", transfer=BOTransferConfig(enabled=True)
            ),
        )
    for sur in ("ridge", "gradient_boost", "random_forest", "extra_trees", "ensemble"):
        c = AdsorptionConfig(
            bo=BOConfig(surrogate=sur, transfer=BOTransferConfig(enabled=True)),  # type: ignore[arg-type]
        )
        assert c.bo.surrogate == sur
    gp = AdsorptionConfig(
        bo=BOConfig(
            surrogate="gaussian_process",
            transfer=BOTransferConfig(enabled=False),
        ),
    )
    assert gp.bo.surrogate == "gaussian_process"


def test_saturation_max_steps_must_be_positive_when_set():
    with pytest.raises(ValueError, match="saturation_max_steps"):
        AdsorptionConfig(saturation_max_steps=0)
    c = AdsorptionConfig(saturation_max_steps=1)
    assert c.saturation_max_steps == 1


def test_flat_bo_constructor_kwargs_rejected():
    with pytest.raises(TypeError):
        AdsorptionConfig(bo_initial_random=2)  # type: ignore[call-arg]


def test_fold_bo_config_rejects_flat_keys():
    with pytest.raises(ValueError, match="Flat BO keys"):
        fold_bo_config({"bo_initial_random": 2, "num_conformers": 1})

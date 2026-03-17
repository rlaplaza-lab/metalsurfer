"""Tests for AdsorptionConfig validation rules."""

import pytest

from metalsurfer.config import AdsorptionConfig

# ---------------------------------------------------------------------------
# valid construction
# ---------------------------------------------------------------------------


def test_default_config():
    config = AdsorptionConfig()
    assert config.num_conformers == 10
    assert config.fmax == 0.05


def test_valid_custom_config():
    config = AdsorptionConfig(
        num_conformers=5,
        num_placements=50,
        fmax=0.01,
        placement_z_range=(1.5, 4.0),
    )
    assert config.num_conformers == 5
    assert config.placement_z_range == (1.5, 4.0)


def test_optimize_isolated_sequentially_default():
    config = AdsorptionConfig()
    assert config.optimize_isolated_sequentially is False


def test_optimize_isolated_sequentially_custom():
    config = AdsorptionConfig(optimize_isolated_sequentially=True)
    assert config.optimize_isolated_sequentially is True


def test_ts_optimizer_defaults():
    config = AdsorptionConfig()
    assert config.ts_optimizer == "fire"
    assert config.steps_between_swaps == 5


def test_ts_optimizer_custom():
    config = AdsorptionConfig(ts_optimizer="lbfgs", steps_between_swaps=10)
    assert config.ts_optimizer == "lbfgs"
    assert config.steps_between_swaps == 10


def test_ts_optimizer_invalid_rejected():
    with pytest.raises(ValueError, match="ts_optimizer"):
        AdsorptionConfig(ts_optimizer="adam")


def test_steps_between_swaps_invalid_rejected():
    with pytest.raises(ValueError, match="steps_between_swaps"):
        AdsorptionConfig(steps_between_swaps=-1)


def test_autobatcher_config_defaults():
    config = AdsorptionConfig()
    assert config.autobatcher_max_memory_padding == 0.5
    assert config.autobatcher_max_memory_scaler is None
    assert config.autobatcher_max_atoms_to_try == 100_000


def test_autobatcher_config_custom():
    config = AdsorptionConfig(
        autobatcher_max_memory_padding=0.7,
        autobatcher_max_memory_scaler=500.0,
        autobatcher_max_atoms_to_try=50_000,
    )
    assert config.autobatcher_max_memory_padding == 0.7
    assert config.autobatcher_max_memory_scaler == 500.0
    assert config.autobatcher_max_atoms_to_try == 50_000


def test_autobatcher_max_memory_padding_out_of_range_rejected():
    with pytest.raises(ValueError, match="autobatcher_max_memory_padding.*0.1"):
        AdsorptionConfig(autobatcher_max_memory_padding=0.05)
    with pytest.raises(ValueError, match="autobatcher_max_memory_padding"):
        AdsorptionConfig(autobatcher_max_memory_padding=1.5)


def test_autobatcher_max_memory_scaler_negative_rejected():
    with pytest.raises(ValueError, match="autobatcher_max_memory_scaler"):
        AdsorptionConfig(autobatcher_max_memory_scaler=-1.0)


def test_autobatcher_max_atoms_to_try_zero_rejected():
    with pytest.raises(ValueError, match="autobatcher_max_atoms_to_try.*positive"):
        AdsorptionConfig(autobatcher_max_atoms_to_try=0)


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


def test_placement_range_wrong_length_rejected():
    """placement_*_range must be a 2-tuple."""
    with pytest.raises(ValueError, match="placement_x_range.*2-tuple.*length"):
        AdsorptionConfig(placement_x_range=(1.0,))
    with pytest.raises(ValueError, match="placement_x_range.*2-tuple.*length"):
        AdsorptionConfig(placement_x_range=(1.0, 2.0, 3.0))


def test_inverted_z_range_rejected():
    with pytest.raises(ValueError, match="placement_z_range.*lower bound"):
        AdsorptionConfig(placement_z_range=(5.0, 2.0))


def test_equal_z_range_rejected():
    with pytest.raises(ValueError, match="placement_z_range.*lower bound"):
        AdsorptionConfig(placement_z_range=(2.0, 2.0))


def test_inverted_x_range_rejected():
    with pytest.raises(ValueError, match="placement_x_range.*lower bound"):
        AdsorptionConfig(placement_x_range=(4.0, -4.0))


# ---------------------------------------------------------------------------
# connectivity multipliers
# ---------------------------------------------------------------------------


def test_empty_multipliers_rejected():
    with pytest.raises(ValueError, match="connectivity_multipliers.*non-empty"):
        AdsorptionConfig(connectivity_multipliers=[])


def test_negative_multiplier_rejected():
    with pytest.raises(ValueError, match="connectivity_multipliers.*positive"):
        AdsorptionConfig(connectivity_multipliers=[1.2, -0.5])


def test_zero_multiplier_rejected():
    with pytest.raises(ValueError, match="connectivity_multipliers.*positive"):
        AdsorptionConfig(connectivity_multipliers=[0.0])


# ---------------------------------------------------------------------------
# VASP kpoints
# ---------------------------------------------------------------------------


def test_invalid_kpoints_length():
    with pytest.raises(ValueError, match="vasp_kpoints.*3-tuple"):
        AdsorptionConfig(vasp_kpoints=(4, 4))


def test_zero_kpoint_rejected():
    with pytest.raises(ValueError, match="vasp_kpoints.*positive integer"):
        AdsorptionConfig(vasp_kpoints=(4, 0, 1))


# ---------------------------------------------------------------------------
# model_name
# ---------------------------------------------------------------------------


def test_empty_model_name_rejected():
    with pytest.raises(ValueError, match="model_name.*non-empty"):
        AdsorptionConfig(model_name="")


# ---------------------------------------------------------------------------
# placement_mode, min_contact_ratio, max_initial_distance
# ---------------------------------------------------------------------------


def test_placement_z_scale_by_covalent_radius_default():
    """placement_z_scale_by_covalent_radius defaults to True."""
    config = AdsorptionConfig()
    assert config.placement_z_scale_by_covalent_radius is True


def test_placement_z_scale_by_covalent_radius_disabled():
    """placement_z_scale_by_covalent_radius=False disables site-aware z."""
    config = AdsorptionConfig(placement_z_scale_by_covalent_radius=False)
    assert config.placement_z_scale_by_covalent_radius is False


@pytest.mark.parametrize("mode", ["cycle", "boltzmann", "mixed"])
def test_conformer_sampling_valid_values(mode):
    """conformer_sampling accepts cycle, boltzmann, mixed."""
    config = AdsorptionConfig(conformer_sampling=mode)
    assert config.conformer_sampling == mode


def test_conformer_sampling_invalid_rejected():
    with pytest.raises(ValueError, match="conformer_sampling"):
        AdsorptionConfig(conformer_sampling="invalid")


@pytest.mark.parametrize("mode", ["random", "sites", "auto", "envelope"])
def test_placement_mode_valid_values(mode):
    """placement_mode accepts random, sites, auto, envelope."""
    config = AdsorptionConfig(placement_mode=mode)
    assert config.placement_mode == mode


def test_placement_mode_invalid_rejected():
    with pytest.raises(ValueError, match="placement_mode"):
        AdsorptionConfig(placement_mode="invalid")


def test_flat_aromatic_parallel_fraction_valid():
    """flat_aromatic_parallel_fraction accepts values in [0, 1]."""
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=0.5)
    assert config.flat_aromatic_parallel_fraction == 0.5
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=0.0)
    assert config.flat_aromatic_parallel_fraction == 0.0
    config = AdsorptionConfig(flat_aromatic_parallel_fraction=1.0)
    assert config.flat_aromatic_parallel_fraction == 1.0


def test_flat_aromatic_parallel_fraction_out_of_range_rejected():
    with pytest.raises(ValueError, match="flat_aromatic_parallel_fraction"):
        AdsorptionConfig(flat_aromatic_parallel_fraction=-0.1)
    with pytest.raises(ValueError, match="flat_aromatic_parallel_fraction"):
        AdsorptionConfig(flat_aromatic_parallel_fraction=1.5)


def test_min_contact_ratio_valid():
    config = AdsorptionConfig(min_contact_ratio=0.9)
    assert config.min_contact_ratio == 0.9


def test_min_contact_ratio_out_of_range_rejected():
    with pytest.raises(ValueError, match="min_contact_ratio"):
        AdsorptionConfig(min_contact_ratio=0.3)
    with pytest.raises(ValueError, match="min_contact_ratio"):
        AdsorptionConfig(min_contact_ratio=1.5)


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
# auto-resize slab defaults
# ---------------------------------------------------------------------------


def test_default_auto_resize_slab_enabled():
    config = AdsorptionConfig()
    assert config.auto_resize_slab is True


def test_default_min_pbc_image_separation():
    config = AdsorptionConfig()
    assert config.min_pbc_image_separation == 8.0


def test_zero_min_pbc_image_separation_rejected():
    with pytest.raises(ValueError, match="min_pbc_image_separation.*positive"):
        AdsorptionConfig(min_pbc_image_separation=0.0)


def test_negative_min_pbc_image_separation_rejected():
    with pytest.raises(ValueError, match="min_pbc_image_separation.*positive"):
        AdsorptionConfig(min_pbc_image_separation=-1.0)


def test_auto_resize_disabled_skips_separation_check():
    config = AdsorptionConfig(auto_resize_slab=False, min_pbc_image_separation=-1.0)
    assert config.auto_resize_slab is False


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

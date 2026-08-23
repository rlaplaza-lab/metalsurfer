"""Pose finalization, replay, and validation."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.features import FEATURE_NAMES, extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.models import PlacementSpec
from metalsurfer.placement import (
    enumerate_placement_specs,
    generate_placement_from_spec,
    generate_placement_from_spec_with_reason,
)
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.geometry import (
    calculate_contact_quality,
    check_initial_contact_quality,
)
from metalsurfer.placement.pose import (
    _validate_posed_adsorbate,
)
from metalsurfer.placement.site_context import (
    _get_unique_sites_for_specs,
)

from ..conftest import (
    adsorption_config_factory,
    make_ethanol,
    make_slab,
    make_water,
    place_adsorbate_above_slab,
    water_conformers,
)
from ._helpers import (
    _assert_replay_matches,
    _first_successful_placement,
    _tilted_make_slab,
)


def test_validate_initial_placement_geometry_with_strict_config():
    """check_initial_contact_quality should accept good contact under strict config."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )

    # max_closest_approach: max allowed closest-approach distance
    config = AdsorptionConfig(
        strict_initial_placement=True,
        min_contact_atoms=1,
        max_closest_approach=3.0,
        contact_distance_threshold=2.5,
    )
    assert config.max_closest_approach == 3.0

    ok, reason = check_initial_contact_quality(
        water,
        slab,
        strict_initial_placement=config.strict_initial_placement,
        require_multiple_contact=config.require_multiple_contact,
        max_closest_approach=float(config.max_closest_approach),
        min_contact_atoms=int(config.min_contact_atoms),
        contact_distance_threshold=config.contact_distance_threshold,
        material_type=config.material_type,
    )
    assert ok, f"Should pass strict validation with good contact: {reason}"
    assert reason == "placement_geometry_valid"


def test_validate_initial_placement_geometry_rejects_poor_contact():
    """check_initial_contact_quality should reject poor contact placements."""
    slab = make_slab()
    water = make_water().copy()

    # Place water far away
    pos = water.get_positions()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 5.0
    water.set_positions(pos)
    water.set_cell(slab.get_cell())
    water.set_pbc(slab.get_pbc())

    config = AdsorptionConfig(
        strict_initial_placement=True,
        max_closest_approach=3.0,
        contact_distance_threshold=2.5,
        min_contact_atoms=3,  # Require 3 contacting atoms
    )

    ok, reason = check_initial_contact_quality(
        water,
        slab,
        strict_initial_placement=config.strict_initial_placement,
        require_multiple_contact=config.require_multiple_contact,
        max_closest_approach=float(config.max_closest_approach),
        min_contact_atoms=int(config.min_contact_atoms),
        contact_distance_threshold=config.contact_distance_threshold,
        material_type=config.material_type,
    )
    assert not ok, "Should reject placement with poor contact"
    assert reason in {
        "contact_distance_too_large",
        "insufficient_contact_atoms",
    }


def test_require_multiple_contact_rejects_single_contact():
    """A monoatomic adsorbate can have at most one contacting atom → reject."""
    slab = make_slab()
    mono = Atoms("He", positions=[[2.0, 2.0, 0.0]])
    pos = mono.get_positions().copy()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.0
    mono.set_positions(pos)
    mono.set_cell(slab.get_cell())
    mono.set_pbc(slab.get_pbc())

    ok, reason = check_initial_contact_quality(
        mono,
        slab,
        strict_initial_placement=False,
        require_multiple_contact=True,
        max_closest_approach=3.5,
        min_contact_atoms=1,
        contact_distance_threshold=2.5,
        material_type="slab",
    )
    assert not ok
    assert reason == "insufficient_contact_atoms"


def test_require_multiple_contact_accepts_multi_atom_contact():
    """Water placed for multi-atom contact should pass require_multiple_contact."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.0, x_shift=2.0, y_shift=2.0
    )

    metrics = calculate_contact_quality(
        water, slab, contact_distance_threshold=2.5, material_type="slab"
    )
    assert int(metrics["num_contacting_atoms"]) >= 2, metrics

    ok, reason = check_initial_contact_quality(
        water,
        slab,
        strict_initial_placement=False,
        require_multiple_contact=True,
        max_closest_approach=3.5,
        min_contact_atoms=1,
        contact_distance_threshold=2.5,
        material_type="slab",
    )
    assert ok, reason
    assert reason == "placement_geometry_valid"


def test_saturation_finalize_rejects_adsorbate_overlap():
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )
    covered = slab + water

    # New adsorbate coincident with pre-adsorbed water → adsorbate_overlap.
    clash = water.copy()
    config = AdsorptionConfig()
    reason = _validate_posed_adsorbate(clash, covered, config, slab_for_sites=slab)
    assert reason == "adsorbate_overlap"

    # Far from prior adsorbate but above substrate → not adsorbate_overlap.
    far = water.copy()
    far_pos = far.get_positions().copy()
    far_pos[:, 0] += 6.0
    far_pos[:, 1] += 6.0
    far.set_positions(far_pos)
    reason_far = _validate_posed_adsorbate(far, covered, config, slab_for_sites=slab)
    assert reason_far != "adsorbate_overlap"


def test_validate_posed_adsorbate_uses_material_pbc(monkeypatch):
    """Separation check uses material PBC, not calculator-promoted 3D PBC."""
    captured = {}

    def _fake_separation(ads, pre, *, cell, pbc=None, **kwargs):
        captured["pbc"] = list(pbc)
        return True, 99.0

    monkeypatch.setattr(
        "metalsurfer.placement.pose.geom.check_adsorbate_separation",
        _fake_separation,
    )

    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )
    covered = slab + water
    config = AdsorptionConfig()
    _validate_posed_adsorbate(water, covered, config, slab_for_sites=slab)
    assert captured["pbc"] == material_aware_pbc("slab")


def test_strict_initial_placement_e2e_reason():
    slab = make_slab()
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=12,
        seed=1,
        strict_initial_placement=True,
        min_initial_distance=0.3,
        contact_distance_threshold=0.4,
        max_closest_approach=0.5,
        min_contact_atoms=1,
    )
    conformers = [make_water()]
    specs = enumerate_placement_specs(conformers, slab, config, "O", 12, seed=1)
    reasons = set()
    for spec in specs:
        _result, reason = generate_placement_from_spec_with_reason(
            spec, conformers, slab, config, smiles="O"
        )
        if reason is not None:
            reasons.add(reason)
    assert reasons
    assert any(
        r
        in {
            "too_close",
            "too_far",
            "vdw_overlap",
            "distance_check_failed",
            "contact_distance_too_large",
            "insufficient_contact_atoms",
            "no_sites_found",
        }
        for r in reasons
    )
    assert "initial_distance_or_site_constraints" not in reasons


def test_rotated_slab_pose_round_trip():
    slab = make_slab()
    cell = np.array(slab.get_cell(), dtype=float)
    rot = np.array(
        [[0.866, -0.5, 0.0], [0.5, 0.866, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    cell[:3] = rot @ cell[:3]
    slab.set_cell(cell)
    config = adsorption_config_factory(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    adsorbate, descriptor = result
    _assert_replay_matches(
        "pose", adsorbate, descriptor, spec, water_conformers(), slab, config
    )


def test_tilted_slab_pose_round_trip():
    """Slab tilted so the surface normal is not Cartesian +z."""
    slab = _tilted_make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    adsorbate, descriptor = result
    _assert_replay_matches(
        "pose", adsorbate, descriptor, spec, water_conformers(), slab, config
    )


def test_generate_placement_from_spec_invalid_conformer_index():
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab")
    water = make_water()
    spec = PlacementSpec(
        conformer_index=3,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="atop",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    result, reason = generate_placement_from_spec_with_reason(
        spec, [water], slab, config
    )
    assert result is None
    assert reason == "invalid_conformer_index"


@pytest.mark.parametrize(
    "mol_factory, smiles, n_desired, extra_cfg",
    [
        (make_ethanol, "CCO", 20, {"placement_z_range": (2.0, 3.0)}),
        (make_water, "O", 24, {}),
    ],
)
def test_placement_specs_deterministic_across_runs(
    mol_factory, smiles, n_desired, extra_cfg
):
    """Same seed → identical specs; different seed → different enumeration."""
    slab = make_slab()
    mol = mol_factory()
    cfg_kwargs = {
        "material_type": "slab",
        "num_placements": n_desired,
        "seed": 7,
        **extra_cfg,
    }
    specs_a = enumerate_placement_specs(
        [mol], slab, AdsorptionConfig(**cfg_kwargs), smiles, n_desired=n_desired
    )
    specs_b = enumerate_placement_specs(
        [mol], slab, AdsorptionConfig(**cfg_kwargs), smiles, n_desired=n_desired
    )
    specs_c = enumerate_placement_specs(
        [mol],
        slab,
        AdsorptionConfig(**{**cfg_kwargs, "seed": 99}),
        smiles,
        n_desired=n_desired,
    )
    assert specs_a == specs_b
    assert specs_a != specs_c


def test_molecular_ml_features_are_injective():
    """Distinct molecular placements must yield distinct BO feature vectors."""
    slab = make_slab()
    water = make_water()
    config = AdsorptionConfig(material_type="slab", num_placements=48, seed=0)
    specs = enumerate_placement_specs([water], slab, config, "O", n_desired=48)
    feature_rows: list[tuple[float, ...]] = []
    for spec in specs:
        generated = generate_placement_from_spec(spec, [water], slab, config)
        if generated is None:
            continue
        _, descriptor = generated
        assert descriptor.fragment_positions is None
        record = PlacementRecord.from_descriptor(
            descriptor, molecule="water", smiles="O"
        )
        feats = extract_features(record)
        assert list(feats.keys()) == FEATURE_NAMES
        assert "fragment_positions" not in feats
        feature_rows.append(tuple(round(feats[name], 10) for name in FEATURE_NAMES))
    assert len(feature_rows) >= 16
    assert len(set(feature_rows)) == len(feature_rows)


def test_invalid_site_index_reason_distinct_from_no_sites():
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", num_placements=1)
    ctx = _get_unique_sites_for_specs(slab, config)
    assert ctx.use_sites and len(ctx.sites) > 0
    bad = PlacementSpec(
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=len(ctx.sites) + 10,
        site_type="atop",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    _result, reason = generate_placement_from_spec_with_reason(
        bad, water_conformers(), slab, config, smiles="O", site_context=ctx
    )
    assert _result is None
    assert reason == "invalid_site_index"


def test_validate_posed_adsorbate_single_mic_with_strict(monkeypatch):
    import metalsurfer.placement.geometry as geom_mod

    calls = {"n": 0}
    orig = geom_mod._mol_slab_pairwise_distances

    def _counted(mol_pos, slab_pos, cell, pbc):
        calls["n"] += 1
        return orig(mol_pos, slab_pos, cell, pbc)

    monkeypatch.setattr(geom_mod, "_mol_slab_pairwise_distances", _counted)

    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )
    config = AdsorptionConfig(
        strict_initial_placement=True,
        min_initial_distance=0.3,
        contact_distance_threshold=0.4,
        max_closest_approach=0.5,
    )
    _validate_posed_adsorbate(water, slab, config)
    # Distance gate and contact-quality gate must share ONE MIC matrix.
    assert calls["n"] == 1


def test_validate_posed_adsorbate_scratch_equivalence():
    from metalsurfer.placement.pose import _build_slab_distance_scratch

    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )
    config = AdsorptionConfig()
    reason_no_scratch = _validate_posed_adsorbate(water, slab, config)
    scratch = _build_slab_distance_scratch(slab, None, "slab")
    reason_scratch = _validate_posed_adsorbate(
        water, slab, config, slab_scratch=scratch
    )
    assert reason_no_scratch == reason_scratch
    assert reason_no_scratch is None


def test_check_initial_placement_distance_reuses_slab_scratch():
    """4.4: passing a slab scratch (with precomputed covalent radii) must agree
    with the no-scratch path and avoid re-slicing the ASE object."""
    from metalsurfer.placement import geometry as geom
    from metalsurfer.placement.pose import _build_slab_distance_scratch

    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.2, x_shift=2.0, y_shift=2.0
    )
    config = AdsorptionConfig()

    scratch = _build_slab_distance_scratch(slab, None, "slab")
    assert scratch.slab_cov_r is not None

    base = geom.check_initial_placement_distance(
        water, slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        material_type="slab",
    )
    with_scratch = geom.check_initial_placement_distance(
        water, slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        material_type="slab",
        slab_scratch=scratch,
    )
    assert base == with_scratch


def test_resolve_surface_ref_no_site_nanoparticle_uses_radial_com(caplog):
    """1.4: nanoparticle replay with no site uses the COM radial distance.

    Cartesian z-max is arbitrary for a radially symmetric cluster and must not
    be used as the z-offset reference.
    """
    import logging

    from metalsurfer.placement.pose import _resolve_surface_ref

    # Anisotropic cluster offset from the origin so Cartesian z-max differs from
    # the radial distance from the centre of mass.
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 2.0],
        ],
        dtype=float,
    )
    slab = Atoms("Cu4", positions=positions)

    with caplog.at_level(logging.WARNING, logger="metalsurfer.placement.pose"):
        ref, is_local = _resolve_surface_ref(None, slab, "nanoparticle")

    com = positions.mean(axis=0)
    expected = float(np.max(np.linalg.norm(positions - com, axis=1)))
    assert ref == pytest.approx(expected)
    assert ref != pytest.approx(float(np.max(positions[:, 2])))
    assert is_local is False
    assert "without a site" in caplog.text

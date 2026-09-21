"""Pose finalization, replay, and validation."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.features import (
    FEATURE_NAMES,
    extract_features,
    placement_pose_from_features,
)
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.models import PlacementPose, PlacementSpec
from metalsurfer.placement import (
    enumerate_placement_specs,
    generate_placement_from_spec,
    generate_placement_from_spec_with_reason,
)
from metalsurfer.placement._constants import (
    _FRAME_REF_ALIGNMENT_DOT_THRESHOLD,
    _LATERAL_OFFSET_REF_SWITCH_DOT,
)
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.geometry import (
    calculate_contact_quality,
    check_initial_contact_quality,
)
from metalsurfer.placement.pose import (
    _apply_lateral_offset,
    _com_height_from_z_fraction,
    _contact_shell_indices,
    _feasible_height_interval,
    _pairwise_contact_com_height,
    _PlacementContext,
    _validate_posed_adsorbate,
    _z_fraction_from_com_height,
    build_pose_batch_cache,
    generate_placement_from_pose,
)
from metalsurfer.placement.site_context import (
    _get_unique_sites_for_specs,
)
from metalsurfer.placement.site_coords import _derive_top_layer_tolerance
from metalsurfer.placement.site_enumeration import _get_site_surface_radii
from metalsurfer.placement.site_types import Site
from metalsurfer.surface_prep import apply_surface_constraints

from ..conftest import (
    adsorption_config_factory,
    make_ethanol,
    make_porous_framework,
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


def test_molecular_ml_features_encode_absolute_pose():
    """BO features are the initial-pose replay ingredients (COM + quat + conformer).

    Distinct placements must yield distinct feature vectors, including those that
    differ only by an in-plane lattice translation.
    """
    slab = make_slab()
    water = make_water()
    config = AdsorptionConfig(material_type="slab", num_placements=48, seed=0)
    specs = enumerate_placement_specs([water], slab, config, "O", n_desired=48)
    feature_rows: list[tuple[float, ...]] = []
    x_abs_values: list[float] = []
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
        assert "x" in feats and "y" in feats
        assert "fragment_positions" not in feats
        feature_rows.append(tuple(round(feats[name], 10) for name in FEATURE_NAMES))
        x_abs_values.append(float(descriptor.x_abs))
    assert len(FEATURE_NAMES) == 8
    assert len(feature_rows) >= 16
    assert len(set(feature_rows)) >= 16
    # Pure in-plane translations must NOT collide once x/y are features.
    translation_collisions = sum(
        1
        for i in range(len(feature_rows))
        for j in range(i + 1, len(feature_rows))
        if feature_rows[i] == feature_rows[j] and x_abs_values[i] != x_abs_values[j]
    )
    assert translation_collisions == 0


def test_feature_row_replays_molecular_placement():
    """extract_features → placement_pose_from_features → generate_placement_from_pose."""
    slab = make_slab()
    water = make_water()
    config = AdsorptionConfig(material_type="slab", num_placements=8, seed=0)
    _spec, generated = _first_successful_placement([water], slab, config, smiles="O")
    assert generated is not None
    adsorbate, descriptor = generated
    record = PlacementRecord.from_descriptor(descriptor, molecule="water", smiles="O")
    feats = extract_features(record)
    assert list(feats.keys()) == FEATURE_NAMES
    assert len(FEATURE_NAMES) == 8
    pose = placement_pose_from_features(
        feats, placement_index=descriptor.placement_index
    )
    replay = generate_placement_from_pose(pose, [water], slab, config)
    assert replay is not None
    replayed, _ = replay
    np.testing.assert_allclose(
        adsorbate.get_positions(), replayed.get_positions(), atol=1e-10
    )


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
        water,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        material_type="slab",
    )
    with_scratch = geom.check_initial_placement_distance(
        water,
        slab,
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

    with caplog.at_level(logging.DEBUG, logger="metalsurfer.placement.pose"):
        ref, is_local = _resolve_surface_ref(None, slab, "nanoparticle")

    com = positions.mean(axis=0)
    expected = float(np.max(np.linalg.norm(positions - com, axis=1)))
    assert ref == pytest.approx(expected)
    # Local-radius mode must differ clearly from the plain max-z fallback.
    assert abs(ref - float(np.max(positions[:, 2]))) > 0.5
    assert is_local is False
    assert "without a site" in caplog.text


def test_apply_lateral_offset_near_x_normal_stays_finite():
    """Lateral recovery uses a looser (0.9) ref-switch than site-frame (0.95).

    A normal nearly aligned with +x must still produce a finite in-plane offset
    rather than a near-zero cross product / NaN basis.
    """
    assert _LATERAL_OFFSET_REF_SWITCH_DOT < _FRAME_REF_ALIGNMENT_DOT_THRESHOLD

    # Dot with [1,0,0] ≈ 0.92: between the two thresholds.
    n_hat = np.array([0.92, 0.0, np.sqrt(1.0 - 0.92**2)], dtype=float)
    assert abs(float(np.dot(n_hat, [1.0, 0.0, 0.0]))) > _LATERAL_OFFSET_REF_SWITCH_DOT
    assert (
        abs(float(np.dot(n_hat, [1.0, 0.0, 0.0]))) < _FRAME_REF_ALIGNMENT_DOT_THRESHOLD
    )

    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type=None,
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=0.0,
        y_abs=0.0,
        z_fraction=0.5,
        z_abs=0.0,
    )
    ctx = _PlacementContext(
        pose=pose,
        site=None,
        mat_type="nanoparticle",
        surface_ref=0.0,
        is_local_ref=False,
        source="test",
        canonical_pos=np.zeros((1, 3)),
        use_sites=False,
        rotated_pos=np.zeros((1, 3)),
        normal=n_hat,
    )
    slab = Atoms("Cu", positions=[[0.0, 0.0, 0.0]])
    shifted = _apply_lateral_offset(
        np.array([0.0, 0.0, 0.0], dtype=float),
        dx=0.5,
        dy=0.25,
        ctx=ctx,
        slab=slab,
    )
    assert np.isfinite(shifted).all()
    assert float(np.linalg.norm(shifted)) > 0.1


def test_pose_batch_cache_surface_radii_use_derived_top_depth():
    """Cached top-layer radii must match ``_get_site_surface_radii``, not planarity tol."""
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab")
    derived = float(_derive_top_layer_tolerance(list(slab.get_chemical_symbols())))
    assert derived != float(config.top_layer_tolerance)
    cache = build_pose_batch_cache(slab, [], config)
    assert cache.r_surface_top_layer == pytest.approx(
        _get_site_surface_radii(slab, None)
    )


def test_water_en_down_contact_atom_is_oxygen():
    """EN-down water places O at/above the pairwise contact gate (zf=0.5)."""
    from metalsurfer.placement.geometry import min_pair_clearance_angstrom
    from metalsurfer.placement.pose import (
        _contact_atom_index,
        _height_above_supports,
        _pose_from_spec,
    )
    from metalsurfer.placement.site_enumeration import _get_site_surface_radii

    slab = make_slab()
    water = make_water()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx_sites = _get_unique_sites_for_specs(slab, config)
    assert ctx_sites.use_sites and ctx_sites.sites
    # Prefer an atop site when available.
    site_index = next(
        (i for i, s in enumerate(ctx_sites.sites) if s.site_type == "atop"),
        0,
    )
    site = ctx_sites.sites[site_index]
    symbols = list(water.get_chemical_symbols())
    o_idx = symbols.index("O")
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="EN-down",
        face_flip=False,
        en_atom_index=o_idx,
        site_index=site_index,
        site_type=str(site.site_type),
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    ctx, fail = _pose_from_spec(water, spec, slab, config, "O", site_context=ctx_sites)
    assert fail is None and ctx is not None
    n_hat = np.asarray(ctx.normal, dtype=float)
    contact_idx = _contact_atom_index(
        ctx.rotated_pos,
        n_hat,
        symbols,
        orientation_type=spec.orientation_type,
        en_atom_index=spec.en_atom_index,
    )
    assert contact_idx == o_idx
    r_surface = _get_site_surface_radii(slab, site)
    surface_symbol = (
        slab.get_chemical_symbols()[int(site.slab_indices[0])]
        if site.slab_indices
        else None
    )
    gate = min_pair_clearance_angstrom(
        "O",
        surface_symbol,
        min_distance=float(config.min_initial_distance),
        min_contact_ratio=float(config.min_contact_ratio),
        reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
        vdw_overlap_scale=float(config.vdw_overlap_scale),
        r_surface_fallback=float(r_surface),
    )
    contact_ref = _height_above_supports(
        site,
        slab.get_positions(),
        n_hat,
        reduce="max",
        fallback=float(ctx.surface_ref),
    )
    atom_h = (
        ctx.rotated_pos + np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs])
    ) @ n_hat
    # Oxygen clears the pairwise gate; H must not dig below O.
    assert float(atom_h[o_idx]) >= contact_ref + gate - 0.05
    h_idxs = [i for i, s in enumerate(symbols) if s == "H"]
    for hi in h_idxs:
        assert float(atom_h[hi]) >= float(atom_h[o_idx]) - 0.5


def test_pairwise_contact_raises_com_when_non_binder_is_closest():
    """Tilted molecule: pairwise 1D raises COM so a low H clears the gate."""
    from metalsurfer.placement.geometry import check_initial_placement_distance
    from metalsurfer.placement.pose import _pose_from_spec

    slab = make_slab()
    # Linear-ish OH with H sticking below O along -z before orientation.
    mol = Atoms("OH", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, -0.96]])
    mol.center()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx_sites = _get_unique_sites_for_specs(slab, config)
    site_index = next(
        (i for i, s in enumerate(ctx_sites.sites) if s.site_type == "atop"),
        0,
    )
    site = ctx_sites.sites[site_index]
    o_idx = 0
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="EN-down",
        face_flip=False,
        en_atom_index=o_idx,
        site_index=site_index,
        site_type=str(site.site_type),
        tilt_deg=45.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    ctx, fail = _pose_from_spec(mol, spec, slab, config, "O", site_context=ctx_sites)
    assert fail is None and ctx is not None
    ads = mol.copy()
    ads.set_positions(
        ctx.rotated_pos
        + np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs], dtype=float)
    )
    ok, _dist, reason = check_initial_placement_distance(
        ads,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        material_type="slab",
    )
    assert ok, reason


def test_void_clashing_interval_stays_inside_window(monkeypatch):
    """A fully clashing void interval collapses inside the diversity window."""
    monkeypatch.setattr(
        "metalsurfer.placement.pose.geom.min_pair_clearance_angstrom",
        lambda *_args, **_kwargs: 1.0e6,
    )
    site = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="pore",
        kind="void",
        slab_indices=(),
        material_type="porous",
        site_source="test",
        env_fingerprint=(("C",), (0,), 0),
    )
    interval = _feasible_height_interval(
        np.zeros((1, 3), dtype=float),
        ["H"],
        site=site,
        place_normal=np.array([0.0, 0.0, 1.0]),
        slab_positions=np.array([[0.0, 0.0, 5.0], [1.5, 0.0, 5.0]], dtype=float),
        slab_symbols=["C", "C"],
        cell=np.eye(3) * 20.0,
        pbc=[True, True, True],
        config=AdsorptionConfig(material_type="porous", seed=0),
        r_surface=0.7,
        z_base_lo=-5.0,
        z_base_hi=2.0,
    )
    assert interval is not None
    base_h = 5.0
    half = 0.5 * (2.0 - (-5.0))
    assert interval.com_lo >= base_h - half - 1e-6
    assert interval.com_hi <= base_h + half + 1e-6
    assert interval.com_lo == pytest.approx(interval.com_nominal)
    assert interval.com_hi == pytest.approx(interval.com_nominal)


def test_void_clear_centre_stays_at_centre(monkeypatch):
    """A void centre that already clears keeps nominal at the void vertex."""
    monkeypatch.setattr(
        "metalsurfer.placement.pose.geom.min_pair_clearance_angstrom",
        lambda *_args, **_kwargs: 0.5,
    )
    site = Site(
        xyz=np.array([0.0, 0.0, 5.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="pore",
        kind="void",
        slab_indices=(),
        material_type="porous",
        site_source="test",
        env_fingerprint=(("C",), (0,), 0),
    )
    # Framework atoms far from the void centre along the normal.
    interval = _feasible_height_interval(
        np.zeros((1, 3), dtype=float),
        ["H"],
        site=site,
        place_normal=np.array([0.0, 0.0, 1.0]),
        slab_positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float),
        slab_symbols=["C", "C"],
        cell=np.eye(3) * 20.0,
        pbc=[True, True, True],
        config=AdsorptionConfig(material_type="porous", seed=0),
        r_surface=0.7,
        z_base_lo=-2.0,
        z_base_hi=2.0,
    )
    assert interval is not None
    assert interval.com_nominal == pytest.approx(5.0, abs=1e-9)
    assert interval.com_lo <= interval.com_nominal
    assert interval.com_hi >= interval.com_nominal


def test_z_fraction_com_height_round_trip():
    """Inverse then forward recovers COM height within 1e-9."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        lo = float(rng.uniform(-2.0, 2.0))
        hi = lo + float(rng.uniform(0.0, 3.0))
        nominal = float(rng.uniform(lo, hi)) if hi > lo else lo
        zf = float(rng.uniform(0.0, 1.0))
        h = _com_height_from_z_fraction(zf, lo, nominal, hi)
        zf2 = _z_fraction_from_com_height(h, lo, nominal, hi)
        h2 = _com_height_from_z_fraction(zf2, lo, nominal, hi)
        assert h2 == pytest.approx(h, abs=1e-9)


def test_z_fraction_from_com_height_point_interval():
    assert _z_fraction_from_com_height(1.0, 1.0, 1.0, 1.0) == 0.5


def test_analytic_height_recovery_z_fraction_replays_nudged_com(monkeypatch):
    """Recovered z_fraction maps back to the clamped COM height."""
    from metalsurfer.placement import pose as pose_mod
    from metalsurfer.placement.pose import _analytic_height_recovery

    n_hat = np.array([0.0, 0.0, 1.0], dtype=float)
    origin = np.array([1.0, 2.0, 5.0], dtype=float)
    com_lo, com_nominal, com_hi = 4.5, 5.0, 6.0
    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type="atop",
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=float(origin[0]),
        y_abs=float(origin[1]),
        z_abs=float(origin[2]),
        z_fraction=0.5,
    )
    ctx = _PlacementContext(
        pose=pose,
        site=Site(
            xyz=np.array([1.0, 2.0, 4.0]),
            normal=n_hat,
            site_type="atop",
            slab_indices=(0,),
            material_type="slab",
            site_source="test",
            env_fingerprint=(("Ru",), (0,), 1),
        ),
        mat_type="slab",
        surface_ref=4.0,
        is_local_ref=True,
        source="test",
        canonical_pos=np.zeros((1, 3)),
        use_sites=True,
        rotated_pos=np.zeros((1, 3)),
        normal=n_hat,
        z_base_lo=0.5,
        z_base_hi=2.0,
        com_lo=com_lo,
        com_nominal=com_nominal,
        com_hi=com_hi,
    )
    slab = Atoms("H", positions=[[1.0, 2.0, 4.0]], cell=[10, 10, 10], pbc=True)
    ads = Atoms("H", positions=[origin.copy()])

    class _Cfg:
        min_initial_distance = 1.0
        min_contact_ratio = 1.0
        reject_vdw_overlaps = False
        vdw_overlap_scale = 1.0
        max_initial_distance = None
        strict_initial_placement = False
        max_closest_approach = 2.0

    monkeypatch.setattr(pose_mod, "_contact_penetration", lambda *_a, **_k: (0.5, 0.3))
    out = _analytic_height_recovery(
        ctx,
        ads,
        slab,
        _Cfg(),  # type: ignore[arg-type]
        "too_close",
        slab_for_sites=slab,
        slab_scratch=None,
    )
    assert out is not None
    center, new_zf = out
    expected_h = min(com_hi, max(com_lo, float(origin[2]) + 0.3))
    assert float(center[2]) == pytest.approx(expected_h, abs=1e-9)
    replay_h = _com_height_from_z_fraction(new_zf, com_lo, com_nominal, com_hi)
    assert replay_h == pytest.approx(expected_h, abs=1e-9)


def test_z_fraction_offsets_com_around_pairwise_contact():
    """z_fraction ≤ 0.5 clips to contact; values above explore the upper window."""
    from metalsurfer.placement.pose import _pose_from_spec

    slab = make_slab()
    water = make_water()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx_sites = _get_unique_sites_for_specs(slab, config)
    site_index = next(
        (i for i, s in enumerate(ctx_sites.sites) if s.site_type == "atop"),
        0,
    )
    site = ctx_sites.sites[site_index]
    o_idx = list(water.get_chemical_symbols()).index("O")
    heights = []
    for zf in (0.1, 0.5, 0.9):
        spec = PlacementSpec(
            conformer_index=0,
            orientation_type="EN-down",
            face_flip=False,
            en_atom_index=o_idx,
            site_index=site_index,
            site_type=str(site.site_type),
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=zf,
            placement_index=0,
        )
        ctx, fail = _pose_from_spec(
            water, spec, slab, config, "O", site_context=ctx_sites
        )
        assert fail is None and ctx is not None
        n_hat = np.asarray(ctx.normal, dtype=float)
        com = np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs], dtype=float)
        heights.append(float(np.dot(com, n_hat)))
    # Wall-near lower bound is the contact solve: low z_fraction clips to it.
    assert heights[0] == pytest.approx(heights[1], abs=1e-9)
    assert heights[2] > heights[1]


def test_contact_height_uses_support_atoms_not_lifted_site_vertex():
    """Catalog xyz is the support-plane anchor; contact ignores bogus lift."""
    from metalsurfer.placement.geometry import min_pair_clearance_angstrom
    from metalsurfer.placement.pose import (
        _framework_plane_height,
        _height_above_supports,
        _pose_from_spec,
        _resolve_surface_ref,
    )
    from metalsurfer.placement.site_enumeration import _get_site_surface_radii
    from metalsurfer.placement.site_types import Site

    # Two-height top layer (bridging O above metal).
    a = 2.7
    positions = []
    symbols = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * a, iy * a, 0.0])
            symbols.append("Ti")
            positions.append([ix * a + 0.5 * a, iy * a + 0.5 * a, 1.2])
            symbols.append("O")
    slab = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[3 * a, 3 * a, 20.0],
        pbc=[True, True, False],
    )
    slab = apply_surface_constraints(slab)
    config = AdsorptionConfig(
        material_type="slab",
        seed=0,
        rough_slab_local_z=True,
        # Include both Ti and O in the top-layer window → non-planar local ref.
        top_layer_tolerance=1.5,
        planar_z_variance_threshold=0.01,
    )
    ctx_sites = _get_unique_sites_for_specs(slab, config)
    assert ctx_sites.use_sites and ctx_sites.sites
    site_index = next(
        (i for i, s in enumerate(ctx_sites.sites) if s.slab_indices),
        0,
    )
    site = ctx_sites.sites[site_index]
    assert site.slab_indices
    water = make_water()
    o_idx = list(water.get_chemical_symbols()).index("O")
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="EN-down",
        face_flip=False,
        en_atom_index=o_idx,
        site_index=site_index,
        site_type=str(site.site_type),
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    ctx, fail = _pose_from_spec(water, spec, slab, config, "O", site_context=ctx_sites)
    assert fail is None and ctx is not None
    n_hat = np.asarray(ctx.normal, dtype=float)
    pos = np.asarray(slab.get_positions(), dtype=float)
    support_h = _height_above_supports(
        site,
        pos,
        n_hat,
        reduce="max",
        fallback=float(ctx.surface_ref),
    )
    # Catalog identity is the support-plane anchor (plugin lift is probe-only).
    site_h = float(np.dot(np.asarray(site.xyz, dtype=float), n_hat))
    assert site_h == pytest.approx(support_h, abs=0.15)
    # Local surface_ref and contact share the framework plane (never a lift).
    assert ctx.is_local_ref
    assert float(ctx.surface_ref) == pytest.approx(support_h, abs=1e-9)
    r_surface = _get_site_surface_radii(slab, site)
    surface_symbol = slab.get_chemical_symbols()[int(site.slab_indices[0])]
    gate = min_pair_clearance_angstrom(
        "O",
        surface_symbol,
        min_distance=float(config.min_initial_distance),
        min_contact_ratio=float(config.min_contact_ratio),
        reject_vdw_overlaps=bool(config.reject_vdw_overlaps),
        vdw_overlap_scale=float(config.vdw_overlap_scale),
        r_surface_fallback=float(r_surface),
    )
    atom_h = (
        ctx.rotated_pos + np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs])
    ) @ n_hat
    # Contact clears the pairwise gate above the support plane.
    assert float(atom_h[o_idx]) >= support_h + gate - 0.15

    # Plugin-agnostic: a wrongly lifted vertex still uses the support plane.
    for source in (
        "topology_hollow",
        "voronoi",
        "adaptive_grid",
        "rolling_probe",
        "injected_atop",
    ):
        lifted = Site(
            xyz=np.asarray(site.xyz, dtype=float) + 2.0 * n_hat,
            normal=np.asarray(site.normal, dtype=float),
            site_type=site.site_type,
            slab_indices=tuple(site.slab_indices),
            material_type="slab",
            site_source=source,
            env_fingerprint=site.env_fingerprint,
        )
        fw = _framework_plane_height(lifted, pos, n_hat, reduce="max")
        ref, local = _resolve_surface_ref(
            lifted,
            slab,
            "slab",
            rough_slab_local_z=True,
            top_layer_tolerance=1.5,
            planar_z_variance_threshold=0.01,
        )
        assert local
        assert fw == pytest.approx(support_h, abs=1e-9)
        assert ref == pytest.approx(support_h, abs=1e-9)
        assert _height_above_supports(
            lifted,
            pos,
            n_hat,
            reduce="max",
            fallback=ref,
        ) == pytest.approx(support_h, abs=1e-9)


def test_pose_does_not_expand_symmetry_reduced_catalog():
    """Passed SiteContext is indexed as-is even when full_slab has adsorbates."""
    from metalsurfer.placement.site_context import (
        resolve_site_context_for_sampling,
        site_context_for_occupied_surface,
    )

    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(material_type="slab", seed=0)
    reduced = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
    assert reduced.clustered_sites is not None
    assert len(reduced.clustered_sites) > len(reduced.sites)
    ads = Atoms(
        "H",
        positions=[reduced.clustered_sites[0].xyz + np.array([0.0, 0.0, 0.2])],
    )
    full = slab.copy() + ads
    wide_index = len(reduced.sites)  # valid only on the clustered catalog
    assert wide_index < len(reduced.clustered_sites)
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=wide_index,
        site_type="atop",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    _bad, reason = generate_placement_from_spec_with_reason(
        spec,
        [make_water()],
        full,
        config,
        smiles="O",
        site_context=reduced,
        slab_for_sites=slab,
    )
    assert _bad is None
    assert reason == "invalid_site_index"
    sampling = site_context_for_occupied_surface(reduced)
    ok = generate_placement_from_spec(
        spec,
        [make_water()],
        full,
        config,
        smiles="O",
        site_context=sampling,
        slab_for_sites=slab,
    )
    assert ok is not None


def _shell_vs_full_nominal(
    *,
    slab: Atoms,
    site,
    rotated_pos: np.ndarray,
    symbols: list[str],
    config: AdsorptionConfig,
) -> tuple[float, float]:
    positions = slab.get_positions()
    slab_syms = list(slab.get_chemical_symbols())
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    r_surface = _get_site_surface_radii(slab, site)
    shell_idx = _contact_shell_indices(
        site.xyz,
        rotated_pos=rotated_pos,
        symbols=symbols,
        slab_positions=positions,
        slab_symbols=slab_syms,
        config=config,
        r_surface=r_surface,
        cell=cell,
        pbc=pbc,
        place_normal=site.normal,
    )
    shell_pos = positions[shell_idx]
    shell_syms = [slab_syms[int(i)] for i in shell_idx]
    kwargs = dict(
        rotated_pos=rotated_pos,
        symbols=symbols,
        site_xyz=site.xyz,
        place_normal=site.normal,
        cell=cell,
        pbc=pbc,
        config=config,
        r_surface=r_surface,
    )
    shell_h = _pairwise_contact_com_height(
        **kwargs, slab_positions=shell_pos, slab_symbols=shell_syms
    )
    full_h = _pairwise_contact_com_height(
        **kwargs, slab_positions=positions, slab_symbols=slab_syms
    )
    return float(shell_h), float(full_h)


def test_contact_shell_matches_full_slab_nominal():
    """Local shell contact nominal agrees with all-pairs within 1e-6 Å."""
    from ase.build import molecule

    slab = make_slab(nx=3, ny=3)
    config = AdsorptionConfig(material_type="slab")
    ctx = _get_unique_sites_for_specs(slab, config)
    site = next(s for s in ctx.sites if s.kind == "wall" and s.slab_indices)
    h2 = molecule("H2")
    h2_pos = h2.get_positions() - h2.get_center_of_mass()
    co2 = molecule("CO2")
    co2_pos = co2.get_positions() - co2.get_center_of_mass()

    for rotated, symbols in (
        (h2_pos, list(h2.get_chemical_symbols())),
        (co2_pos, list(co2.get_chemical_symbols())),
    ):
        shell_h, full_h = _shell_vs_full_nominal(
            slab=slab,
            site=site,
            rotated_pos=rotated,
            symbols=symbols,
            config=config,
        )
        assert shell_h == pytest.approx(full_h, abs=1e-6)

    porous = make_porous_framework()
    pconfig = AdsorptionConfig(material_type="porous")
    pctx = _get_unique_sites_for_specs(porous, pconfig)
    psite = next(s for s in pctx.sites if s.kind == "wall" and s.slab_indices)
    shell_h, full_h = _shell_vs_full_nominal(
        slab=porous,
        site=psite,
        rotated_pos=h2_pos,
        symbols=list(h2.get_chemical_symbols()),
        config=pconfig,
    )
    assert shell_h == pytest.approx(full_h, abs=1e-6)


def test_contact_shell_ignores_far_atoms_and_uses_near_ones():
    """An atom inside the ball is selected; one far outside is not."""
    from ase.build import molecule

    config = AdsorptionConfig(material_type="slab")
    site_xyz = np.array([0.0, 0.0, 0.0])
    h = molecule("H")
    rotated = h.get_positions() - h.get_center_of_mass()
    symbols = ["H"]
    positions = np.array(
        [
            [0.0, 0.0, -2.0],
            [50.0, 50.0, -2.0],
        ],
        dtype=float,
    )
    slab_syms = ["Pt", "Pt"]
    idx = _contact_shell_indices(
        site_xyz,
        rotated_pos=rotated,
        symbols=symbols,
        slab_positions=positions,
        slab_symbols=slab_syms,
        config=config,
        r_surface=1.3,
        cell=np.eye(3) * 80.0,
        pbc=[False, False, False],
        place_normal=np.array([0.0, 0.0, 1.0]),
    )
    assert 0 in set(int(i) for i in idx)
    assert 1 not in set(int(i) for i in idx)

    # Closer substrate atom raises the pairwise contact height.
    near = Atoms(
        "Pt2",
        positions=[[0.0, 0.0, -2.5], [0.0, 0.0, -2.0]],
        cell=[40, 40, 40],
        pbc=False,
    )
    closer = Atoms(
        "Pt3",
        positions=[[0.0, 0.0, -2.5], [0.0, 0.0, -2.0], [0.0, 0.0, -1.2]],
        cell=[40, 40, 40],
        pbc=False,
    )
    cell = np.eye(3) * 40.0
    pbc = [False, False, False]
    site = Site(
        xyz=site_xyz,
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="atop",
        kind="wall",
        slab_indices=(0,),
        material_type="slab",
        site_source="test",
        env_fingerprint=((), (), 0),
    )

    def _h(slab_atoms: Atoms) -> float:
        return float(
            _pairwise_contact_com_height(
                rotated,
                symbols,
                site_xyz=site.xyz,
                place_normal=site.normal,
                slab_positions=slab_atoms.get_positions(),
                slab_symbols=list(slab_atoms.get_chemical_symbols()),
                cell=cell,
                pbc=pbc,
                config=config,
                r_surface=1.3,
            )
        )

    assert abs(_h(closer) - _h(near)) > 1e-4

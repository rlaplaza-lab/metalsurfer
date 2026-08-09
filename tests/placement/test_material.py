"""Material-aware PBC and per-material site generation/typing."""

import math

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import PlacementPose, PlacementSpec
from metalsurfer.placement import (
    check_initial_placement_distance,
    enumerate_placement_specs,
    generate_placement_from_spec,
    get_unified_sites,
    material_aware_pbc,
)
from metalsurfer.placement._material import (
    calculator_pbc_for_atoms,
    material_type_for_placement,
)
from metalsurfer.placement.geometry import (
    detect_vdw_overlaps,
)
from metalsurfer.placement.pose import (
    _finalize_placement,
    _PlacementContext,
    _resolve_surface_ref,
)
from metalsurfer.placement.site_context import (
    SiteContext,
    _get_unique_sites_for_specs,
)
from metalsurfer.placement.site_enumeration import (
    _cluster_equivalent_sites,
    _compute_site_z_base,
    _get_site_surface_radii,
)
from metalsurfer.placement.site_types import site_from_dict

from ..conftest import (
    adsorption_config_factory,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    make_water,
    water_conformers,
)
from ._helpers import (
    _LOCAL_SITE_MATERIAL_PARAMS,
    _assert_replay_matches,
    _generate_placements,
)


def test_material_type_for_placement():
    assert material_type_for_placement(None, when_no_site="slab") == "slab"
    porous = site_from_dict(
        {"xyz": [0.0, 0.0, 1.0], "site_type": "atop", "material_type": "porous"}
    )
    assert material_type_for_placement(porous, when_no_site="slab") == "porous"
    nanoparticle = site_from_dict(
        {"xyz": [0.0, 0.0, 1.0], "material_type": "nanoparticle"}
    )
    assert (
        material_type_for_placement(nanoparticle, when_no_site="slab") == "nanoparticle"
    )

def test_material_aware_pbc():
    assert material_aware_pbc("slab") == [True, True, False]
    assert material_aware_pbc("nanoparticle") == [False, False, False]
    assert material_aware_pbc("porous") == [True, True, True]

def test_material_aware_pbc_unknown_raises():
    with pytest.raises(ValueError, match="material_type"):
        material_aware_pbc("bulk")

def test_calculator_pbc_for_atoms():
    slab = Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 20], pbc=[True, True, False])
    assert calculator_pbc_for_atoms(slab) == [True, True, True]

    porous = Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 20], pbc=True)
    assert calculator_pbc_for_atoms(porous) == [True, True, True]

    cluster = Atoms("Pt", positions=[[0, 0, 0]], cell=[20, 20, 20], pbc=False)
    assert calculator_pbc_for_atoms(cluster) == [False, False, False]

def test_prepare_atoms_for_calculator_maps_slab_pbc():
    from metalsurfer.workflow.shared import _prepare_atoms_for_calculator

    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[5, 5, 20], pbc=[True, True, False])
    _prepare_atoms_for_calculator(atoms, label="test slab")
    assert list(atoms.get_pbc()) == [True, True, True]

def test_slab_placements_are_above_surface_reference():
    slab = make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=50, placement_z_range=(1.5, 2.0)
    )
    results = _generate_placements(
        water_conformers(), slab, config, smiles="O", n_desired=50
    )
    assert len(results) >= 50
    for _, adsorbate, descriptor in results:
        assert descriptor.surface_ref_z_abs is not None
        assert descriptor.z_abs is not None
        assert descriptor.z_abs >= descriptor.surface_ref_z_abs
        assert descriptor.orientation_type == "round"
        ok, dist, reason = check_initial_placement_distance(
            adsorbate, slab, material_type="slab"
        )
        assert ok, reason
        assert 1.2 <= dist <= 4.0

@pytest.mark.parametrize(
    "material_type,factory,num_placements,z_range,n_desired",
    _LOCAL_SITE_MATERIAL_PARAMS,
)
def test_local_site_material_enumeration_generation_and_reproducibility(
    material_type, factory, num_placements, z_range, n_desired
):

    structure = factory()
    config = adsorption_config_factory(
        material_type=material_type,
        num_placements=num_placements,
        placement_z_range=z_range,
        reject_vdw_overlaps=True,
    )
    conformers = water_conformers()
    results = _generate_placements(
        conformers, structure, config, smiles="O", n_desired=n_desired
    )
    min_ok = max(
        8 if material_type == "porous" else 10, int(math.ceil(0.7 * n_desired))
    )
    assert len(results) >= min_ok, (
        f"{material_type}: expected >= {min_ok}/{n_desired} successes, got {len(results)}"
    )
    visited_sites = {spec.site_index for spec, _, _ in results}
    assert len(visited_sites) >= 2, (
        f"{material_type}: expected multi-site coverage, got {sorted(visited_sites)}"
    )
    d_hi = 4.5 if material_type == "porous" else 3.5
    for _spec, adsorbate_i, desc in results:
        ok, dist, reason = check_initial_placement_distance(
            adsorbate_i,
            structure,
            reject_vdw_overlaps=True,
            material_type=material_type,
        )
        assert ok, f"{material_type} placement failed contact gate: {reason}"
        assert 1.0 <= dist <= d_hi, (
            f"{material_type} adsorbate–surface distance out of band: {dist:.3f}"
        )
        overlaps, _ = detect_vdw_overlaps(
            adsorbate_i, structure, material_type=material_type
        )
        assert len(overlaps) == 0, f"{material_type} placement has VDW clashes"
        assert desc.surface_ref_z_abs is not None and np.isfinite(
            desc.surface_ref_z_abs
        )
        assert desc.z_abs is not None and np.isfinite(desc.z_abs)
        assert desc.orientation_type == "round"

    spec, adsorbate, descriptor = results[0]
    _assert_replay_matches(
        "spec", adsorbate, descriptor, spec, conformers, structure, config
    )
    _assert_replay_matches(
        "descriptor", adsorbate, descriptor, spec, conformers, structure, config
    )
    _assert_replay_matches(
        "pose", adsorbate, descriptor, spec, conformers, structure, config
    )

@pytest.mark.parametrize(
    "material_type,factory,num_placements,z_range,n_desired",
    _LOCAL_SITE_MATERIAL_PARAMS,
)
def test_local_site_material_placement_center_matches_site_geometry(
    material_type, factory, num_placements, z_range, n_desired
):
    structure = factory()
    config = adsorption_config_factory(
        material_type=material_type,
        num_placements=num_placements,
        placement_z_range=z_range,
        # Recovery can laterally nudge abs coords away from pure site+offset.
        placement_distance_recovery=False,
    )
    conformers = water_conformers()
    site_ctx = _get_unique_sites_for_specs(structure, config)
    unique_sites, use_sites = site_ctx.sites, site_ctx.use_sites
    assert use_sites and len(unique_sites) > 0

    specs = enumerate_placement_specs(
        conformers, structure, config, "O", n_desired=n_desired
    )
    n_matched = 0
    for spec in specs:
        result = generate_placement_from_spec(
            spec, conformers, structure, config, smiles="O"
        )
        if result is None or spec.site_index < 0:
            continue
        _, descriptor = result
        site = unique_sites[spec.site_index]
        n_hat = np.asarray(site.normal, dtype=float)
        n_hat = n_hat / float(np.linalg.norm(n_hat))
        expected = (
            np.asarray(site.xyz, dtype=float) + float(descriptor.z_offset) * n_hat
        )
        got = np.array(
            [descriptor.x_abs, descriptor.y_abs, descriptor.z_abs], dtype=float
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)

        surface_ref, is_local = _resolve_surface_ref(site, structure, material_type)
        assert is_local
        assert surface_ref == pytest.approx(float(np.dot(site.xyz, n_hat)), abs=1e-9)
        assert descriptor.surface_ref_z_abs == pytest.approx(surface_ref, abs=1e-6)
        assert float(np.dot(got, n_hat)) == pytest.approx(
            float(descriptor.surface_ref_z_abs) + float(descriptor.z_offset),
            abs=1e-6,
        )
        n_matched += 1

    assert n_matched >= min(5, max(1, n_desired // 2)), (
        f"Expected multiple successful {material_type} site-based placements, "
        f"got {n_matched}"
    )

def test_porous_clustered_sites_prefer_pores_first():
    """Open pore sites are ordered ahead of wall-adjacent sites after clustering."""
    porous = make_porous_framework()
    raw = get_unified_sites(porous, material_type="porous")
    sites = _cluster_equivalent_sites(
        raw, np.asarray(porous.get_cell(), dtype=float), tolerance=0.05
    )
    pore_indices = [i for i, s in enumerate(sites) if s.site_type == "pore"]
    assert pore_indices, "SiO₂ porous fixture must expose pore-classified sites"
    non_pore_indices = [i for i, s in enumerate(sites) if s.site_type != "pore"]
    if non_pore_indices:
        assert max(pore_indices) < min(non_pore_indices)
    # Within pores, larger free-volume (nn_distance) comes first.
    pore_nns = [
        float(s.nn_distance) if s.nn_distance is not None else -1.0
        for s in sites
        if s.site_type == "pore"
    ]
    assert pore_nns == sorted(pore_nns, reverse=True)

@pytest.mark.parametrize(
    "material_type,fail_reason,expect_raise",
    [
        ("nanoparticle", "too_close", True),
        ("nanoparticle", "too_far", False),
        ("porous", "too_close", False),
        ("porous", "too_far", True),
    ],
)
def test_local_site_distance_recovery_height_direction(
    material_type, fail_reason, expect_raise, monkeypatch
):
    """NP raises on too_close / lowers on too_far; porous inverts that."""
    from metalsurfer.placement.pose import _recover_distance_failure

    structure = (
        make_nanoparticle()
        if material_type == "nanoparticle"
        else make_porous_framework()
    )
    water = make_water()
    pos = water.get_positions().copy()
    pos -= pos.mean(axis=0)
    site = get_unified_sites(structure, material_type=material_type)[0]
    n_hat = np.asarray(site.normal, dtype=float)
    n_hat = n_hat / float(np.linalg.norm(n_hat))
    surface_ref, _ = _resolve_surface_ref(site, structure, material_type)
    zf0 = 0.4
    center = np.asarray(site.xyz, dtype=float) + 2.0 * n_hat
    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type=site.site_type,
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=float(center[0]),
        y_abs=float(center[1]),
        z_fraction=zf0,
        z_abs=float(center[2]),
        orientation_type="round",
    )
    ctx = _PlacementContext(
        pose=pose,
        site=site,
        mat_type=material_type,
        surface_ref=float(surface_ref),
        is_local_ref=True,
        source="test",
        canonical_pos=pos,
        use_sites=True,
        rotated_pos=pos,
        z_base_lo=0.5,
        z_base_hi=3.5,
        normal=n_hat,
    )

    monkeypatch.setattr(
        "metalsurfer.placement.pose._validate_posed_adsorbate",
        lambda *args, **kwargs: None,
    )
    config = AdsorptionConfig(
        material_type=material_type,
        placement_distance_recovery=True,
        placement_x_range=(0.0, 0.0),
        placement_y_range=(0.0, 0.0),
    )
    # First height candidate is accepted; direction encodes material policy.
    new_ctx, reason = _recover_distance_failure(
        ctx, water.copy(), structure, config, fail_reason
    )
    assert reason is None
    zf_final = float(new_ctx.pose.z_fraction)
    assert abs(zf_final - zf0) >= 0.05
    if expect_raise:
        assert zf_final > zf0
    else:
        assert zf_final < zf0
    new_center = np.array(
        [new_ctx.pose.x_abs, new_ctx.pose.y_abs, new_ctx.pose.z_abs], dtype=float
    )
    delta_along_n = float(np.dot(new_center - center, n_hat))
    if expect_raise:
        assert delta_along_n > 0.0
    else:
        assert delta_along_n < 0.0

def test_resolve_surface_ref_rough_slab():
    """On a rough slab, local z should be used when rough_slab_local_z=True."""
    # Build a stepped slab: two terraces at different z
    positions = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * 2.7, iy * 2.7, 0.0])
            positions.append([ix * 2.7, iy * 2.7, 2.7])  # terrace 1
    # Add a step: higher terrace
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 5.4])
    slab = Atoms(
        symbols=["Ru"] * len(positions),
        positions=positions,
        cell=[8.1, 8.1, 20.0],
        pbc=[True, True, True],
    )

    site_low = site_from_dict({"z": 2.7, "xyz": np.array([0.0, 0.0, 2.7])})

    # With rough_slab_local_z=True and non-planar slab: use site z
    ref_low, is_local = _resolve_surface_ref(
        site_low,
        slab,
        "slab",
        rough_slab_local_z=True,
    )
    # The slab is non-planar so this should return the site's own z
    # (or global max if planar check says it's still planar)
    assert isinstance(ref_low, float)

    # Without rough_slab_local_z: always global max
    ref_global, is_local_g = _resolve_surface_ref(
        site_low,
        slab,
        "slab",
        rough_slab_local_z=False,
    )
    assert ref_global == float(np.max(slab.get_positions()[:, 2]))
    assert not is_local_g

def test_compute_site_z_base_multiplicative_from_covalent_radii():
    """z_lo/hi scale with placement_z_range × (r_mol + r_surface)."""
    from ase.data import atomic_numbers, covalent_radii

    slab = make_slab()
    top_index = int(np.argmax(slab.get_positions()[:, 2]))
    site = site_from_dict(
        {
            "xyz": slab.get_positions()[top_index],
            "site_type": "atop",
            "slab_indices": (top_index,),
        }
    )
    config = AdsorptionConfig(placement_z_range=(1.0, 1.5))
    z_lo, z_hi = _compute_site_z_base(config, slab, site, ["H"])
    r_h = float(covalent_radii[atomic_numbers["H"]])
    r_surface = _get_site_surface_radii(slab, site)
    assert r_surface is not None
    r_sum = r_h + r_surface
    assert z_lo == pytest.approx(1.0 * r_sum)
    assert z_hi == pytest.approx(1.5 * r_sum)

def test_compute_site_z_base_literal_when_scaling_disabled():
    slab = make_slab()
    config = AdsorptionConfig(
        placement_z_range=(2.1, 3.4),
        placement_z_scale_by_covalent_radius=False,
    )
    z_lo, z_hi = _compute_site_z_base(config, slab, None, ["H"])
    assert (z_lo, z_hi) == (2.1, 3.4)

def test_compute_site_z_base_same_formula_for_pore_site():
    from ase.data import atomic_numbers, covalent_radii

    slab = make_porous_framework()
    sites = get_unified_sites(slab, material_type="porous")
    pore_sites = [s for s in sites if s.site_type == "pore"]
    assert pore_sites, "SiO₂ porous fixture must expose pore sites"
    site = pore_sites[0]
    config = AdsorptionConfig(placement_z_range=(1.0, 1.5))
    z_lo, z_hi = _compute_site_z_base(config, slab, site, ["O"])
    r_o = float(covalent_radii[atomic_numbers["O"]])
    r_surface = _get_site_surface_radii(slab, site)
    assert r_surface is not None
    r_sum = r_o + r_surface
    assert z_lo == pytest.approx(1.0 * r_sum)
    assert z_hi == pytest.approx(1.5 * r_sum)

def test_saturation_placement_height_uses_reference_slab():
    """Saturation placement should not anchor new molecules above old adsorbates."""
    slab = make_slab()
    slab_top = float(np.max(slab.get_positions()[:, 2]))
    top_index = int(np.argmax(slab.get_positions()[:, 2]))
    site_xy = slab.get_positions()[top_index, :2]
    site_context = SiteContext(
        sites=[
            site_from_dict(
                {
                    "xy": site_xy,
                    "xyz": np.array([site_xy[0], site_xy[1], slab_top]),
                    "z": slab_top,
                    "site_type": "atop",
                    "material_type": "slab",
                    "slab_indices": (top_index,),
                }
            )
        ],
        use_sites=True,
        source="test",
    )
    existing_adsorbate_top = slab_top + 10.0
    existing_adsorbate = Atoms(
        "C",
        positions=[
            [slab.cell[0, 0] / 2.0, slab.cell[1, 1] / 2.0, existing_adsorbate_top]
        ],
    )
    full_slab = slab + existing_adsorbate
    full_slab.set_cell(slab.get_cell())
    full_slab.set_pbc(slab.get_pbc())

    config = AdsorptionConfig(
        device="cpu",
        rough_slab_local_z=False,
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=False,
    )
    adsorbate = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    spec = PlacementSpec(
        conformer_index=0,
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

    reference_result = generate_placement_from_spec(
        spec,
        [adsorbate],
        full_slab,
        config,
        site_context=site_context,
        slab_for_sites=slab,
    )
    assert reference_result is not None
    _, reference_descriptor = reference_result
    assert reference_descriptor.surface_ref_z_abs == pytest.approx(slab_top)
    assert reference_descriptor.z_abs == pytest.approx(
        slab_top + reference_descriptor.z_offset
    )

    full_slab_result = generate_placement_from_spec(
        spec,
        [adsorbate],
        full_slab,
        config,
        site_context=site_context,
    )
    assert full_slab_result is not None
    _, full_slab_descriptor = full_slab_result
    assert full_slab_descriptor.surface_ref_z_abs == pytest.approx(
        existing_adsorbate_top
    )

def test_distance_recovery_rescues_too_close_placement():
    """Height recovery should accept a placement that starts too close."""
    slab = make_slab()
    water = make_water()
    pos = water.get_positions().copy()
    pos -= pos.mean(axis=0)
    surface_z = float(np.max(slab.get_positions()[:, 2]))
    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type="atop",
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=5.0,
        y_abs=5.0,
        z_fraction=0.0,
        z_abs=surface_z + 0.35,
        orientation_type="round",
    )
    ctx = _PlacementContext(
        pose=pose,
        site=None,
        mat_type="slab",
        surface_ref=surface_z,
        is_local_ref=False,
        source="test",
        canonical_pos=pos,
        use_sites=False,
        rotated_pos=pos,
        z_base_lo=0.5,
        z_base_hi=3.5,
        normal=np.array([0.0, 0.0, 1.0]),
    )

    fail, reason = _finalize_placement(
        ctx,
        water.copy(),
        slab,
        AdsorptionConfig(placement_distance_recovery=False),
        allow_distance_recovery=True,
    )
    assert fail is None
    assert reason == "too_close"

    ok, ok_reason = _finalize_placement(
        ctx,
        water.copy(),
        slab,
        AdsorptionConfig(placement_distance_recovery=True),
        allow_distance_recovery=True,
    )
    assert ok is not None, ok_reason
    adsorbate_ok, descriptor = ok
    assert descriptor.z_fraction > 0.0
    assert descriptor.z_abs is not None
    assert float(descriptor.z_abs) > surface_z + 0.35
    gate_ok, min_d, gate_reason = check_initial_placement_distance(
        adsorbate_ok, slab, material_type="slab"
    )
    assert gate_ok, (min_d, gate_reason)
    assert 1.2 <= float(min_d) <= 4.0

def test_distance_recovery_height_only_when_xy_disabled():
    """Zero XY ranges still allow height recovery."""
    slab = make_slab()
    water = make_water()
    pos = water.get_positions().copy()
    pos -= pos.mean(axis=0)
    surface_z = float(np.max(slab.get_positions()[:, 2]))
    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type="atop",
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=5.0,
        y_abs=5.0,
        z_fraction=0.0,
        z_abs=surface_z + 0.35,
        orientation_type="round",
    )
    ctx = _PlacementContext(
        pose=pose,
        site=None,
        mat_type="slab",
        surface_ref=surface_z,
        is_local_ref=False,
        source="test",
        canonical_pos=pos,
        use_sites=False,
        rotated_pos=pos,
        z_base_lo=0.5,
        z_base_hi=3.5,
        normal=np.array([0.0, 0.0, 1.0]),
    )
    config = AdsorptionConfig(
        placement_distance_recovery=True,
        placement_x_range=(0.0, 0.0),
        placement_y_range=(0.0, 0.0),
    )
    result, reason = _finalize_placement(
        ctx, water.copy(), slab, config, allow_distance_recovery=True
    )
    assert result is not None, reason
    adsorbate_ok, descriptor = result
    assert descriptor.x_abs == pytest.approx(5.0, abs=1e-6)
    assert descriptor.y_abs == pytest.approx(5.0, abs=1e-6)
    assert float(descriptor.z_abs) > surface_z + 0.35
    gate_ok, min_d, gate_reason = check_initial_placement_distance(
        adsorbate_ok, slab, material_type="slab"
    )
    assert gate_ok, (min_d, gate_reason)
    assert 1.2 <= float(min_d) <= 4.0


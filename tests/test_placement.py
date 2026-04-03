"""Placement tests for universal slab/nanoparticle/porous pathways."""

import random

import numpy as np
import pytest
from ase import Atoms
from scipy.spatial import KDTree

import metalsurfer.placement.sites as site_module
from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import check_desorption
from metalsurfer.models import PlacementPose
from metalsurfer.placement import (
    _classify_molecule_shape,
    _cluster_equivalent_sites,
    _is_flat_aromatic_with_en,
    _random_rotation_matrix,
    calculate_min_distance,
    check_initial_placement_distance,
    detect_material_type,
    enumerate_placement_specs,
    generate_placement_from_descriptor,
    generate_placement_from_pose,
    generate_placement_from_spec,
    get_symmetry_aware_sites,
    get_unified_sites,
    material_aware_pbc,
)
from metalsurfer.placement.generators import _get_unique_sites_for_specs
from metalsurfer.placement.policy import (
    Z_FRACTIONS,
    build_batch_placement_specs,
    max_batch_placement_specs,
)
from metalsurfer.placement.sites import (
    _classify_voronoi_site,
    _compute_local_normal,
    _voronoi_sites,
)
from metalsurfer.workflow import shared as workflow_shared

from .conftest import (
    make_ethanol,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    make_water,
    place_molecule_on_slab,
)


def _first_successful_placement(conformers, slab, config, smiles, n_desired=20):
    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        smiles,
        n_desired=n_desired,
    )
    for spec in specs:
        result = generate_placement_from_spec(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
        )
        if result is not None:
            return spec, result
    return None, None


def _generate_placements(conformers, slab, config, smiles=None, n_desired=30):
    specs = enumerate_placement_specs(conformers, slab, config, smiles, n_desired)
    results = []
    for spec in specs:
        result = generate_placement_from_spec(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
        )
        if result is not None:
            results.append((spec, result[0], result[1]))
    return results


# ---------------------------------------------------------------------------
# Shape and rotation invariants
# ---------------------------------------------------------------------------


def test_classify_molecule_shape_linear_flat_round():
    shape_h2, _, _ = _classify_molecule_shape(
        Atoms("H2", positions=[[0, 0, 0], [0.74, 0, 0]]).get_positions()
    )
    assert shape_h2 == "linear"

    shape_flat, _, _ = _classify_molecule_shape(
        np.array(
            [
                [1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0.0]
                for i in range(6)
            ]
        )
    )
    assert shape_flat == "flat"

    shape_ch4, _, _ = _classify_molecule_shape(
        Atoms(
            "CH4",
            positions=[
                [0, 0, 0],
                [1.09, 1.09, 1.09],
                [-1.09, -1.09, 1.09],
                [-1.09, 1.09, -1.09],
                [1.09, -1.09, -1.09],
            ],
        ).get_positions()
    )
    assert shape_ch4 == "round"


def test_random_rotation_is_proper_orthogonal_matrix():
    rng = random.Random(42)
    for _ in range(25):
        rot = _random_rotation_matrix(rng)
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-12)


def test_flat_aromatic_detection_requires_ring_and_en_atoms():
    assert _is_flat_aromatic_with_en("c1(C=O)cc(OC)c(O)cc1") is True
    assert _is_flat_aromatic_with_en("c1ccccc1") is False
    assert _is_flat_aromatic_with_en("CCO") is False


# ---------------------------------------------------------------------------
# Distance and material typing invariants
# ---------------------------------------------------------------------------


def test_calculate_min_distance_mic_wraps_periodic_boundary():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.5, 9.5, 5.0]])
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True)
    assert d < 2.0


def test_initial_placement_distance_accepts_and_rejects_expected_heights():
    slab = make_slab()
    water = make_water()
    surface_z = float(np.max(slab.get_positions()[:, 2]))

    near = water.copy()
    p_near = near.get_positions().copy()
    p_near[:, 2] += surface_z + 0.3
    near.set_positions(p_near)
    near.set_cell(slab.get_cell())
    near.set_pbc(slab.get_pbc())
    ok_near, _ = check_initial_placement_distance(near, slab)
    assert not ok_near

    valid = water.copy()
    p_valid = valid.get_positions().copy()
    p_valid[:, 2] += surface_z + 2.4
    p_valid[:, 0] += 2.0
    p_valid[:, 1] += 2.0
    valid.set_positions(p_valid)
    valid.set_cell(slab.get_cell())
    valid.set_pbc(slab.get_pbc())
    ok_valid, _ = check_initial_placement_distance(valid, slab)
    assert ok_valid


def test_detect_material_type_and_material_aware_pbc_cover_all_modes():
    slab = make_slab()
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    assert detect_material_type(slab) == "slab"
    assert detect_material_type(nanoparticle) == "nanoparticle"
    assert detect_material_type(porous) == "porous"

    assert material_aware_pbc(slab) == [True, True, False]
    assert material_aware_pbc(nanoparticle) == [False, False, False]
    assert material_aware_pbc(porous) == [True, True, True]


# ---------------------------------------------------------------------------
# Unified site generation and clustering
# ---------------------------------------------------------------------------


def test_get_unified_sites_slab_nanoparticle_porous_have_expected_metadata():
    slab_sites = get_unified_sites(make_slab(), material_type="slab")
    np_sites = get_unified_sites(make_nanoparticle(), material_type="nanoparticle")
    porous_sites = get_unified_sites(make_porous_framework(), material_type="porous")

    assert len(slab_sites) > 0
    assert len(np_sites) > 0
    assert len(porous_sites) > 0

    for sites, mat in (
        (slab_sites, "slab"),
        (np_sites, "nanoparticle"),
        (porous_sites, "porous"),
    ):
        for site in sites:
            assert site["material_type"] == mat
            assert site["site_source"] == "voronoi"
            assert "nn_distance" in site and site["nn_distance"] is not None
            assert np.asarray(site["xyz"]).shape == (3,)
            assert np.linalg.norm(np.asarray(site["normal"])) > 0.5


def test_cluster_equivalent_sites_reduces_or_keeps_sites_per_material():
    slab = make_slab()
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    slab_raw = get_unified_sites(slab, material_type="slab")
    np_raw = get_unified_sites(nanoparticle, material_type="nanoparticle")
    porous_raw = get_unified_sites(porous, material_type="porous")

    slab_unique = _cluster_equivalent_sites(
        slab_raw, np.asarray(slab.get_cell()), tolerance=0.05
    )
    np_unique = _cluster_equivalent_sites(
        np_raw, np.asarray(nanoparticle.get_cell()), tolerance=0.05
    )
    porous_unique = _cluster_equivalent_sites(
        porous_raw, np.asarray(porous.get_cell()), tolerance=0.05
    )

    assert 0 < len(slab_unique) <= len(slab_raw)
    assert 0 < len(np_unique) <= len(np_raw)
    assert 0 < len(porous_unique) <= len(porous_raw)


def test_cluster_equivalent_sites_preserves_step_sites_with_same_xy():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    sites = [
        {
            "xy": np.array([1.0, 1.0]),
            "z": 5.0,
            "xyz": np.array([1.0, 1.0, 5.0]),
            "site_type": "atop",
            "material_type": "slab",
        },
        {
            "xy": np.array([1.0, 1.0]),
            "z": 6.0,
            "xyz": np.array([1.0, 1.0, 6.0]),
            "site_type": "atop",
            "material_type": "slab",
        },
    ]
    unique = _cluster_equivalent_sites(sites, cell, tolerance=0.05)
    assert len(unique) == 2


def test_classify_voronoi_site_types_for_simple_geometries():
    positions_atop = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    site_type, _ = _classify_voronoi_site(np.array([0.8, 0.0, 0.0]), positions_atop)
    assert site_type == "atop"

    positions_bridge = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [6.0, 2.0, 0.0],
        ]
    )
    site_type, idx = _classify_voronoi_site(
        np.array([1.0, 0.0, 0.0]), positions_bridge, k=4
    )
    assert site_type == "bridge"
    assert len(idx) == 2

    positions_hollow = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [1.0, np.sqrt(3.0), 0.0],
            [6.0, 6.0, 0.0],
        ]
    )
    site_type, idx = _classify_voronoi_site(
        np.array([1.0, np.sqrt(3.0) / 3.0, 0.0]), positions_hollow, k=4
    )
    assert site_type == "hollow"
    assert len(idx) == 3


def test_compute_local_normal_points_outward_from_surface_centroid():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
        ]
    )
    vertex = np.array([1.0, 1.0, 2.0])
    normal = _compute_local_normal(vertex, positions)
    assert np.isclose(np.linalg.norm(normal), 1.0, atol=1e-12)
    assert normal[2] > 0.9


def test_voronoi_nn_distances_match_periodic_image_query_for_porous():
    porous = make_porous_framework()
    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell())
    pbc = np.asarray(porous.get_pbc(), dtype=bool)

    vertices, nn_dists = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
    )
    assert len(vertices) > 0

    shifts = [([-1, 0, 1] if pbc[d] else [0]) for d in range(3)]
    extended = []
    for i in shifts[0]:
        for j in shifts[1]:
            for k in shifts[2]:
                offset = i * cell[0] + j * cell[1] + k * cell[2]
                extended.append(positions + offset)
    extended_positions = np.vstack(extended)

    expected, _ = KDTree(extended_positions).query(vertices, k=1)
    np.testing.assert_allclose(nn_dists, np.ravel(expected), atol=1e-8)


def test_voronoi_enrichment_increases_or_preserves_site_count_on_porous():
    porous = make_porous_framework()
    positions = porous.get_positions()
    cell = np.asarray(porous.get_cell())
    pbc = np.asarray(porous.get_pbc(), dtype=bool)

    vertices_base, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=False,
    )
    vertices_enriched, _ = _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=1.0,
        max_distance=4.5,
        enrich=True,
    )

    assert len(vertices_enriched) >= len(vertices_base)


def test_voronoi_enrichment_uses_ridge_vertices(monkeypatch):
    class _FakeVoronoi:
        def __init__(self):
            self.vertices = np.array(
                [
                    [2.0, 2.0, 2.0],
                    [6.0, 2.0, 2.0],
                    [4.0, 2.0, 2.0],
                ],
                dtype=float,
            )
            # Input-point connectivity (not valid for Voronoi vertex graph).
            self.ridge_points = np.array([[0, 1]], dtype=int)
            # Voronoi vertex connectivity used for enrichment.
            self.ridge_vertices = [[0, 2], [2, 1]]

    fake_vor = _FakeVoronoi()
    monkeypatch.setattr(site_module, "Voronoi", lambda _pts: fake_vor)

    captured = {}

    def _capture_ridges(
        vertices,
        nn_dists,
        ridge_vertices,
        raw_to_kept,
        extended_positions,
        framework_tree,
        probe_radius,
        max_distance,
    ):
        captured["ridge_vertices"] = ridge_vertices
        return vertices, nn_dists

    monkeypatch.setattr(site_module, "_enrich_along_ridges", _capture_ridges)

    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    cell = np.eye(3) * 10.0
    pbc = np.array([False, False, False], dtype=bool)

    _voronoi_sites(
        positions,
        cell,
        pbc,
        probe_radius=0.0,
        max_distance=100.0,
        enrich=True,
    )

    assert captured["ridge_vertices"] == fake_vor.ridge_vertices


def test_site_context_cache_clear_resets_cached_entries():
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(material_type="slab")

    workflow_shared.clear_site_context_cache()
    assert len(workflow_shared._SITE_CONTEXT_CACHE) == 0

    workflow_shared._resolve_site_context_for_sampling(
        slab,
        config,
        symmetry_broken=True,
    )
    assert len(workflow_shared._SITE_CONTEXT_CACHE) == 1

    workflow_shared.clear_site_context_cache()
    assert len(workflow_shared._SITE_CONTEXT_CACHE) == 0


# ---------------------------------------------------------------------------
# Slab pathway: enumeration, placement, reproducibility
# ---------------------------------------------------------------------------


def test_slab_enumeration_and_generation_have_high_success_and_site_coverage():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=50,
        placement_z_range=(2.0, 3.0),
    )
    results = _generate_placements(
        [make_water()], slab, config, smiles="O", n_desired=50
    )

    assert len(results) >= 45
    visited_sites = {spec.site_index for spec, _, _ in results}
    assert len(visited_sites) >= 2


def test_slab_reproducible_same_spec_same_positions_and_descriptor_replay():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", num_placements=10, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        [make_water()], slab, config, "O", n_desired=10
    )
    assert spec is not None and result is not None

    ads1, desc1 = result
    again = generate_placement_from_spec(spec, [make_water()], slab, config, smiles="O")
    assert again is not None
    ads2, _ = again
    np.testing.assert_allclose(ads1.get_positions(), ads2.get_positions(), atol=1e-10)

    replayed = generate_placement_from_descriptor(
        desc1, [make_water()], slab, config, smiles="O"
    )
    assert replayed is not None
    np.testing.assert_allclose(
        ads1.get_positions(), replayed.get_positions(), atol=1e-10
    )


def test_slab_pose_roundtrip_reproduces_geometry():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", num_placements=8, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        [make_water()], slab, config, "O", n_desired=8
    )
    assert spec is not None and result is not None

    adsorbate, descriptor = result
    pose = PlacementPose(
        conformer_index=descriptor.conformer_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        placement_index=descriptor.placement_index,
        quat_w=float(descriptor.quat_w),
        quat_x=float(descriptor.quat_x),
        quat_y=float(descriptor.quat_y),
        quat_z=float(descriptor.quat_z),
        x_abs=float(descriptor.x_abs),
        y_abs=float(descriptor.y_abs),
        z_fraction=float(descriptor.z_fraction),
        z_abs=float(descriptor.z_abs),
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
    )
    rebuilt = generate_placement_from_pose(pose, [make_water()], slab, config)
    assert rebuilt is not None
    rebuilt_ads, _ = rebuilt
    np.testing.assert_allclose(
        adsorbate.get_positions(), rebuilt_ads.get_positions(), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Nanoparticle pathway: universal center along local normal
# ---------------------------------------------------------------------------


def test_nanoparticle_enumeration_generation_and_reproducibility():
    nanoparticle = make_nanoparticle()
    config = AdsorptionConfig(
        material_type="nanoparticle",
        num_placements=20,
        placement_z_range=(1.5, 2.5),
    )
    results = _generate_placements(
        [make_water()], nanoparticle, config, smiles="O", n_desired=20
    )
    assert len(results) >= 1

    spec, adsorbate, descriptor = results[0]
    same = generate_placement_from_spec(
        spec, [make_water()], nanoparticle, config, smiles="O"
    )
    assert same is not None
    np.testing.assert_allclose(
        adsorbate.get_positions(), same[0].get_positions(), atol=1e-10
    )

    replayed = generate_placement_from_descriptor(
        descriptor, [make_water()], nanoparticle, config, smiles="O"
    )
    assert replayed is not None
    np.testing.assert_allclose(
        adsorbate.get_positions(), replayed.get_positions(), atol=1e-10
    )


def test_nanoparticle_placement_center_is_site_plus_normal_scaled_offset():
    nanoparticle = make_nanoparticle()
    config = AdsorptionConfig(
        material_type="nanoparticle",
        num_placements=20,
        placement_z_range=(1.5, 2.5),
    )
    _site_ctx = _get_unique_sites_for_specs(nanoparticle, config)
    unique_sites, use_sites = _site_ctx.sites, _site_ctx.use_sites
    assert use_sites and len(unique_sites) > 0

    specs = enumerate_placement_specs(
        [make_water()], nanoparticle, config, "O", n_desired=20
    )
    matched = False
    for spec in specs:
        result = generate_placement_from_spec(
            spec, [make_water()], nanoparticle, config, smiles="O"
        )
        if result is None or spec.site_index < 0:
            continue
        _, descriptor = result
        site = unique_sites[spec.site_index]
        expected = np.asarray(site["xyz"], dtype=float) + float(
            descriptor.z_offset
        ) * np.asarray(site["normal"], dtype=float)
        got = np.array(
            [descriptor.x_abs, descriptor.y_abs, descriptor.z_abs], dtype=float
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)
        matched = True
        break

    assert matched, "Expected at least one successful nanoparticle site-based placement"


# ---------------------------------------------------------------------------
# Porous pathway: universal center along local normal
# ---------------------------------------------------------------------------


def test_porous_enumeration_generation_and_reproducibility():
    porous = make_porous_framework()
    config = AdsorptionConfig(
        material_type="porous",
        num_placements=12,
        placement_z_range=(1.5, 3.0),
    )
    results = _generate_placements(
        [make_water()], porous, config, smiles="O", n_desired=12
    )
    assert len(results) >= 1

    spec, adsorbate, descriptor = results[0]
    same = generate_placement_from_spec(
        spec, [make_water()], porous, config, smiles="O"
    )
    assert same is not None
    np.testing.assert_allclose(
        adsorbate.get_positions(), same[0].get_positions(), atol=1e-10
    )

    replayed = generate_placement_from_descriptor(
        descriptor, [make_water()], porous, config, smiles="O"
    )
    assert replayed is not None
    np.testing.assert_allclose(
        adsorbate.get_positions(), replayed.get_positions(), atol=1e-10
    )


def test_porous_placement_center_is_site_plus_normal_scaled_offset():
    porous = make_porous_framework()
    config = AdsorptionConfig(
        material_type="porous",
        num_placements=12,
        placement_z_range=(1.5, 3.0),
    )
    _site_ctx = _get_unique_sites_for_specs(porous, config)
    unique_sites, use_sites = _site_ctx.sites, _site_ctx.use_sites
    assert use_sites and len(unique_sites) > 0

    specs = enumerate_placement_specs([make_water()], porous, config, "O", n_desired=12)
    matched = False
    for spec in specs:
        result = generate_placement_from_spec(
            spec, [make_water()], porous, config, smiles="O"
        )
        if result is None or spec.site_index < 0:
            continue
        _, descriptor = result
        site = unique_sites[spec.site_index]
        expected = np.asarray(site["xyz"], dtype=float) + float(
            descriptor.z_offset
        ) * np.asarray(site["normal"], dtype=float)
        got = np.array(
            [descriptor.x_abs, descriptor.y_abs, descriptor.z_abs], dtype=float
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)
        matched = True
        break

    assert matched, "Expected at least one successful porous site-based placement"


# ---------------------------------------------------------------------------
# Orientation strategies and strict descriptor requirements
# ---------------------------------------------------------------------------


def test_flat_aromatic_specs_include_parallel_and_en_down_when_applicable():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=24,
        placement_z_range=(2.0, 3.0),
        flat_aromatic_parallel_fraction=0.5,
    )
    from metalsurfer.conformers import create_conformers_from_smiles

    result = create_conformers_from_smiles(
        "c1(C=O)cc(OC)c(O)cc1",
        config=AdsorptionConfig(num_conformers=3),
    )
    if result is None:
        pytest.skip("RDKit required")
    conformers, _ = result

    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        "c1(C=O)cc(OC)c(O)cc1",
        n_desired=24,
    )
    kinds = {spec.orientation_type for spec in specs}
    assert "parallel" in kinds
    assert "EN-down" in kinds


def test_descriptor_replay_requires_explicit_absolute_geometry():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", num_placements=8, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        [make_water()], slab, config, "O", n_desired=8
    )
    assert spec is not None and result is not None

    _, descriptor = result
    descriptor.z_abs = None
    replayed = generate_placement_from_descriptor(
        descriptor, [make_water()], slab, config, smiles="O"
    )
    assert replayed is None


# ---------------------------------------------------------------------------
# Cross-material desorption checks
# ---------------------------------------------------------------------------


def test_check_desorption_slab_adsorbed_and_desorbed_cases():
    slab = make_slab()

    adsorbed = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    ok_ads, _ = check_desorption(adsorbed, slab, binding_threshold=4.0)
    assert ok_ads

    desorbed = slab + make_water()
    pos = desorbed.get_positions().copy()
    pos[len(slab) :, 2] = float(np.max(slab.get_positions()[:, 2])) + 12.0
    desorbed.set_positions(pos)
    desorbed.set_cell(slab.get_cell())
    desorbed.set_pbc(slab.get_pbc())
    ok_des, reason = check_desorption(desorbed, slab, binding_threshold=4.0)
    assert not ok_des
    assert "too far" in reason


def test_check_desorption_nanoparticle_and_porous():
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    water_far = make_water()
    water_far.set_positions(water_far.get_positions() + np.array([20.0, 20.0, 20.0]))
    np_combined = nanoparticle + water_far
    np_combined.set_cell(nanoparticle.get_cell())
    np_combined.set_pbc(nanoparticle.get_pbc())
    ok_np, _ = check_desorption(np_combined, nanoparticle, binding_threshold=4.0)
    assert not ok_np

    water_near = make_water()
    sites = get_unified_sites(porous, material_type="porous")
    site = sites[0]
    center = np.asarray(site["xyz"], dtype=float) + 1.0 * np.asarray(
        site["normal"], dtype=float
    )
    wpos = water_near.get_positions().copy()
    wpos -= np.mean(wpos, axis=0)
    wpos += center
    water_near.set_positions(wpos)
    porous_combined = porous + water_near
    porous_combined.set_cell(porous.get_cell())
    porous_combined.set_pbc(porous.get_pbc())
    ok_porous, _ = check_desorption(porous_combined, porous, binding_threshold=4.0)
    assert ok_porous


# ---------------------------------------------------------------------------
# Symmetry-aware site reduction
# ---------------------------------------------------------------------------


def test_symmetry_aware_sites_are_consistent_with_core_sites_on_slab():
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(
        material_type="slab", symmetry_tolerance=0.1, site_equivalence_tolerance=0.05
    )

    _site_ctx = _get_unique_sites_for_specs(slab, config)
    core_sites, use_sites = _site_ctx.sites, _site_ctx.use_sites
    assert use_sites and len(core_sites) > 0

    reduced = get_symmetry_aware_sites(
        slab,
        top_layer_tolerance=config.top_layer_tolerance,
        symmetry_tolerance=config.symmetry_tolerance,
        material_type="slab",
    )
    assert 0 < len(reduced) <= len(core_sites)


# ---------------------------------------------------------------------------
# Enumeration stability
# ---------------------------------------------------------------------------


def test_enumerate_specs_is_deterministic_for_same_inputs():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    specs1 = enumerate_placement_specs(
        [make_ethanol()], slab, config, "CCO", n_desired=20
    )
    specs2 = enumerate_placement_specs(
        [make_ethanol()], slab, config, "CCO", n_desired=20
    )

    assert len(specs1) == len(specs2)
    for s1, s2 in zip(specs1, specs2, strict=True):
        assert s1.placement_index == s2.placement_index
        assert s1.conformer_index == s2.conformer_index
        assert s1.site_index == s2.site_index
        assert s1.orientation_type == s2.orientation_type
        assert s1.tilt_deg == s2.tilt_deg
        assert s1.azimuth_deg == s2.azimuth_deg
        assert s1.z_fraction == s2.z_fraction


# ---------------------------------------------------------------------------
# Policy: build_batch_placement_specs / max_batch_placement_specs
# ---------------------------------------------------------------------------


def _site_type_atop(idx: int) -> str:
    return "atop"


def test_max_batch_specs_matches_build_batch_uncapped_round():
    """max_batch_placement_specs == len(build_batch_placement_specs) when uncapped."""
    n_conformers = 2
    site_indices = [0, 1, 2]

    expected = max_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=site_indices,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
    )
    actual = build_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=site_indices,
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=10**7,
    )
    assert len(actual) == expected


def test_max_batch_specs_matches_build_batch_uncapped_flat_aromatic():
    """Flat-aromatic parallel + EN-down counts are consistent between the two functions."""
    expected = max_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1],
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
    )
    actual = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=10**7,
    )
    assert len(actual) == expected


def test_build_batch_specs_respects_n_desired():
    n_desired = 7
    specs = build_batch_placement_specs(
        n_conformers=3,
        site_indices=[0, 1, 2],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=n_desired,
    )
    assert len(specs) <= n_desired


def test_build_batch_specs_dissociative_branch():
    n_hollow_pairs = 3
    specs = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0],
        site_type_for_index=lambda idx: "hollow",
        shape="linear",
        n_binders=0,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=100,
        dissociative=True,
        n_hollow_pairs=n_hollow_pairs,
    )
    assert len(specs) > 0
    assert all(s.orientation_type == "dissociative" for s in specs)
    assert all(0 <= s.site_index < n_hollow_pairs for s in specs)


def test_max_batch_specs_dissociative_equals_pairs_times_z_fractions():
    n_hollow_pairs = 4
    count = max_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        shape="linear",
        n_binders=0,
        flat_aromatic=False,
        dissociative=True,
        n_hollow_pairs=n_hollow_pairs,
    )
    assert count == n_hollow_pairs * len(Z_FRACTIONS)


def test_build_batch_specs_flat_aromatic_generates_both_orientation_types():
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=10**7,
    )
    orientation_types = {s.orientation_type for s in specs}
    assert "parallel" in orientation_types
    assert "EN-down" in orientation_types


def test_build_batch_specs_filter_spec_reduces_count():
    all_specs = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0, 1],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=10**7,
    )
    filtered_specs = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0, 1],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=10**7,
        filter_spec=lambda s: s.tilt_deg == 0.0,
    )
    assert len(filtered_specs) < len(all_specs)
    assert all(s.tilt_deg == 0.0 for s in filtered_specs)


def test_build_batch_specs_placement_indices_are_unique():
    specs = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0, 1, 2],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=50,
    )
    ids = [s.placement_index for s in specs]
    assert len(set(ids)) == len(ids), "placement_index values must be unique"

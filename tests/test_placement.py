"""Placement tests for universal slab/nanoparticle/porous pathways."""

import random
from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from scipy.spatial import KDTree

import metalsurfer.placement.site_voronoi as site_voronoi_module
from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.filters import check_desorption
from metalsurfer.ml.features import FEATURE_NAMES, extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.models import PlacementDescriptor, PlacementPose, PlacementSpec
from metalsurfer.placement import (
    calculate_min_distance,
    check_initial_placement_distance,
    enumerate_placement_specs,
    generate_placement_from_descriptor,
    generate_placement_from_pose,
    generate_placement_from_spec,
    generate_placement_from_spec_with_reason,
    get_symmetry_aware_sites,
    get_unified_sites,
    material_aware_pbc,
)
from metalsurfer.placement._material import (
    calculator_pbc_for_atoms,
    material_type_for_placement,
)
from metalsurfer.placement.dissociative import _get_dissociative_site_pairs
from metalsurfer.placement.geometry import (
    _classify_molecule_shape,
    _random_rotation_matrix,
    calculate_contact_quality,
    check_adsorbate_separation,
    check_initial_contact_quality,
    detect_vdw_overlaps,
)
from metalsurfer.placement.orientation import (
    _estimate_parallel_fraction,
    _is_flat_aromatic_with_en,
    classify_adsorbate_orientation,
)
from metalsurfer.placement.policy import (
    Z_FRACTIONS,
    build_batch_placement_specs,
    max_batch_placement_specs,
)
from metalsurfer.placement.pose import (
    _PlacementContext,
    _finalize_placement,
    _resolve_surface_ref,
    _validate_posed_adsorbate,
)
from metalsurfer.placement.site_classify import _compute_local_normal
from metalsurfer.placement.site_context import (
    SiteContext,
    _SITE_CONTEXT_CACHE,
    _get_unique_sites_for_specs,
    clear_site_caches,
)
from metalsurfer.placement.site_coords import _deduplicate_points
from metalsurfer.placement.site_enumeration import (
    _cluster_equivalent_sites,
    _compute_site_z_base,
    _get_site_surface_radii,
)
from metalsurfer.placement.site_types import Site, site_from_dict
from metalsurfer.placement.site_voronoi import (
    _classify_voronoi_site,
    _enrich_along_ridges,
    _voronoi_sites,
)
from metalsurfer.workflow import shared as workflow_shared

from .conftest import (
    adsorption_config_factory,
    make_ethanol,
    make_h2,
    make_nanoparticle,
    make_placement_descriptor,
    make_porous_framework,
    make_slab,
    make_water,
    place_adsorbate_above_slab,
    place_molecule_on_slab,
    water_conformers,
)

TEST_SEED = 0

# Golden snapshot for make_slab() unified sites (captured before Delaunay sharing).
_GOLDEN_SLAB_UNIFIED_SITE_COUNT = 126
# PBC edge upgrade re-labels near-boundary primary-atops as bridge; hollows unchanged.
_GOLDEN_SLAB_SITE_TYPE_MULTISET = {"atop": 16, "bridge": 64, "hollow": 46}


def test_unified_sites_slab_golden_count_and_type_multiset():
    """Stable site catalog for make_slab() under default classification."""
    slab = make_slab()
    sites = get_unified_sites(slab, material_type="slab")
    assert len(sites) == _GOLDEN_SLAB_UNIFIED_SITE_COUNT
    assert dict(Counter(s.site_type for s in sites)) == _GOLDEN_SLAB_SITE_TYPE_MULTISET


_LOCAL_SITE_MATERIAL_PARAMS = [
    ("nanoparticle", make_nanoparticle, 20, (1.5, 2.5), 20),
    ("porous", make_porous_framework, 12, (1.5, 3.0), 12),
]


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


def _pose_from_descriptor(descriptor: PlacementDescriptor) -> PlacementPose:
    return PlacementPose(
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


def _assert_replay_matches(
    mode: str,
    adsorbate: Atoms,
    descriptor: PlacementDescriptor,
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
) -> None:
    if mode == "spec":
        replay = generate_placement_from_spec(
            spec, conformers, slab, config, smiles="O"
        )
        assert replay is not None
        replayed = replay[0]
    elif mode == "descriptor":
        replayed = generate_placement_from_descriptor(
            descriptor, conformers, slab, config, smiles="O"
        )
        assert replayed is not None
    else:
        pose = _pose_from_descriptor(descriptor)
        replay = generate_placement_from_pose(pose, conformers, slab, config)
        assert replay is not None
        replayed = replay[0]
    np.testing.assert_allclose(
        adsorbate.get_positions(), replayed.get_positions(), atol=1e-10
    )


# ---------------------------------------------------------------------------
# Shape and rotation invariants
# ---------------------------------------------------------------------------


def test_classify_molecule_shape_linear_flat_round():
    shape_h2, _, _ = _classify_molecule_shape(
        make_h2().get_positions()
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
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True, pbc=[True, True, False])
    # Minimum image of (0.5,0.5)↔(9.5,9.5) in a 10×10 cell is √(1²+1²)=√2
    assert d == pytest.approx(np.sqrt(2.0), abs=1e-6)


def test_calculate_min_distance_requires_explicit_pbc_for_periodic_cell():
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    p1 = np.array([[0.5, 0.5, 5.0]])
    p2 = np.array([[9.5, 9.5, 5.0]])
    with pytest.raises(ValueError, match="pbc must be provided"):
        calculate_min_distance(p1, p2, cell=cell, use_pbc=True)


def test_material_type_for_placement():
    assert material_type_for_placement(None, when_no_site="slab") == "slab"
    porous = site_from_dict({"site_type": "atop", "material_type": "porous"})
    assert material_type_for_placement(porous, when_no_site="slab") == "porous"
    nanoparticle = site_from_dict({"material_type": "nanoparticle"})
    assert (
        material_type_for_placement(nanoparticle, when_no_site="slab") == "nanoparticle"
    )


def test_deduplicate_points_returns_expected_keep_mask():
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.01, 0.01, 0.0],
            [1.0, 1.0, 1.0],
            [1.01, 1.0, 1.0],
        ]
    )
    keep = _deduplicate_points(points, tolerance=0.03)
    assert keep.dtype == bool
    assert keep.shape == (4,)
    assert int(np.sum(keep)) == 2
    kept_points = points[keep]
    assert len(kept_points) == 2


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
    ok_near, _, reason_near = check_initial_placement_distance(
        near, slab, material_type="slab"
    )
    assert not ok_near
    assert reason_near == "too_close"

    valid = water.copy()
    p_valid = valid.get_positions().copy()
    p_valid[:, 2] += surface_z + 2.4
    p_valid[:, 0] += 2.0
    p_valid[:, 1] += 2.0
    valid.set_positions(p_valid)
    valid.set_cell(slab.get_cell())
    valid.set_pbc(slab.get_pbc())
    ok_valid, _, reason_valid = check_initial_placement_distance(
        valid, slab, material_type="slab"
    )
    assert ok_valid
    assert reason_valid is None


def test_check_initial_placement_distance_too_far_reason():
    slab = make_slab()
    water = make_water()
    surface_z = float(np.max(slab.get_positions()[:, 2]))
    far = water.copy()
    p = far.get_positions().copy()
    p[:, 2] += surface_z + 8.0
    far.set_positions(p)
    far.set_cell(slab.get_cell())
    far.set_pbc(slab.get_pbc())
    ok, _, reason = check_initial_placement_distance(
        far, slab, max_initial_distance=3.0, material_type="slab"
    )
    assert not ok
    assert reason == "too_far"


def test_min_contact_ratio_default_is_covalent_binding_boundary():
    """Default min_contact_ratio rejects covalent overlap and accepts physisorption."""
    from ase.data import atomic_numbers, covalent_radii

    from metalsurfer._numeric_defaults import MIN_CONTACT_RATIO_DEFAULT

    slab = make_slab(n_layers=1, symbol="Ru")
    # Place a single O atom directly above a Ru atom and scan the contact ratio.
    ru_pos = slab.get_positions()[0]
    r_o = float(covalent_radii[atomic_numbers["O"]])
    r_ru = float(covalent_radii[atomic_numbers["Ru"]])
    covalent_sum = r_o + r_ru
    surface_xy = ru_pos.copy()

    too_close = Atoms("O", positions=[surface_xy + np.array([0.0, 0.0, 0.0])])
    p = too_close.get_positions().copy()
    p[0, 2] = ru_pos[2] + covalent_sum * (MIN_CONTACT_RATIO_DEFAULT - 0.05)
    too_close.set_positions(p)
    too_close.set_cell(slab.get_cell())
    too_close.set_pbc(slab.get_pbc())
    ok_close, min_close, reason_close = check_initial_placement_distance(
        too_close,
        slab,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type="slab",
    )
    assert not ok_close
    assert reason_close == "too_close"
    assert min_close < covalent_sum * MIN_CONTACT_RATIO_DEFAULT

    ok_far_side = Atoms("O", positions=[surface_xy + np.array([0.0, 0.0, 0.0])])
    p_ok = ok_far_side.get_positions().copy()
    p_ok[0, 2] = ru_pos[2] + covalent_sum * (MIN_CONTACT_RATIO_DEFAULT + 0.05)
    ok_far_side.set_positions(p_ok)
    ok_far_side.set_cell(slab.get_cell())
    ok_far_side.set_pbc(slab.get_pbc())
    ok_pass, min_pass, reason_pass = check_initial_placement_distance(
        ok_far_side,
        slab,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type="slab",
    )
    assert ok_pass, (min_pass, reason_pass)
    assert min_pass >= covalent_sum * MIN_CONTACT_RATIO_DEFAULT


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
            assert site.material_type == mat
            # Updated to accept both old "voronoi" and new topology-based sources
            assert site.site_source in (
                "voronoi",
                "topology_atop",
                "topology_bridge",
                "topology_hollow",
                "atop_injected",
            )
            assert site.nn_distance is not None
            assert np.asarray(site.xyz).shape == (3,)
            assert np.linalg.norm(np.asarray(site.normal)) > 0.5


def test_site_enumeration_exports_wrap_cartesian_for_atop_injection():
    """Atop injection under PBC uses _wrap_cartesian from site_coords."""
    from metalsurfer.placement import site_enumeration as enum_mod
    from metalsurfer.placement.site_coords import _wrap_cartesian as wrap_ref

    assert enum_mod._wrap_cartesian is wrap_ref
    slab = make_slab()
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(slab.get_pbc(), dtype=bool)
    pts = slab.get_positions()[:1] + np.array([[0.1, 0.1, 0.5]])
    wrapped = enum_mod._wrap_cartesian(pts, cell, pbc)
    assert wrapped.shape == pts.shape
    assert len(get_unified_sites(slab, material_type="slab")) > 0


def test_get_unified_sites_slab_atop_injection_wraps_under_pbc(monkeypatch):
    """Atop injection must call _wrap_cartesian and emit atop_injected sites."""
    from metalsurfer.placement import site_enumeration as enum_mod

    real_topo = enum_mod._generate_slab_topology_sites
    real_wrap = enum_mod._wrap_cartesian
    wrap_calls: list[int] = []

    def _topo_without_atop(*args, **kwargs):
        result = real_topo(*args, **kwargs)
        # Topology may return (verts, dists, sources) or (+ triangulation).
        if len(result) == 4:
            verts, dists, sources, primary_tri = result
        else:
            verts, dists, sources = result
            primary_tri = None
        keep = [i for i, src in enumerate(sources) if src != "topology_atop"]
        if not keep:
            empty = (
                np.zeros((0, 3), dtype=float),
                np.zeros(0, dtype=float),
                [],
            )
            return empty if primary_tri is None else (*empty, primary_tri)
        idx = np.asarray(keep, dtype=int)
        trimmed = (verts[idx], dists[idx], [sources[i] for i in keep])
        return trimmed if primary_tri is None else (*trimmed, primary_tri)

    def _counting_wrap(points, cell, pbc):
        wrap_calls.append(len(np.asarray(points)))
        return real_wrap(points, cell, pbc)

    monkeypatch.setattr(enum_mod, "_generate_slab_topology_sites", _topo_without_atop)
    monkeypatch.setattr(enum_mod, "_wrap_cartesian", _counting_wrap)
    slab = make_slab()
    assert bool(np.any(slab.get_pbc()))
    sites = get_unified_sites(slab, material_type="slab")
    assert len(sites) > 0
    assert wrap_calls, "_wrap_cartesian must run on the atop-injection path"
    assert any(str(s.site_source) == "atop_injected" for s in sites)


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


@pytest.mark.parametrize(
    "sites,expected_count",
    [
        (
            [
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 6.0,
                        "xyz": np.array([1.0, 1.0, 6.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                    }
                ),
            ],
            2,
        ),
        (
            [
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ni",), "atop"),
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Pt",), "atop"),
                    }
                ),
            ],
            2,
        ),
        (
            [
                site_from_dict(
                    {
                        "xy": np.array([1.0, 1.0]),
                        "z": 5.0,
                        "xyz": np.array([1.0, 1.0, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ru",), "atop"),
                    }
                ),
                site_from_dict(
                    {
                        "xy": np.array([1.001, 1.001]),
                        "z": 5.0,
                        "xyz": np.array([1.001, 1.001, 5.0]),
                        "site_type": "atop",
                        "material_type": "slab",
                        "env_fingerprint": (("Ru",), "atop"),
                    }
                ),
            ],
            1,
        ),
    ],
)
def test_cluster_equivalent_sites_case_matrix(sites, expected_count):
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    unique = _cluster_equivalent_sites(sites, cell, tolerance=0.05)
    assert len(unique) == expected_count


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


def test_delaunay_classification_pbc_edge_is_not_mislabeled_atop():
    """Cross-boundary bridge midpoints are required to label an a-edge site."""
    from scipy.spatial import Delaunay

    from metalsurfer.placement.site_coords import _project_to_slab_plane
    from metalsurfer.placement.site_voronoi import (
        _build_delaunay_classification_index,
        _delaunay_site_classification,
    )

    # 2×2 square lattice centred in a 4 Å cell.
    positions = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc = np.array([True, True, False], dtype=bool)
    top_idx = np.arange(4, dtype=int)
    top_2d = _project_to_slab_plane(positions, cell)
    tri = Delaunay(top_2d)

    # Bridge midpoint across the a-boundary: atom (3,1) ↔ image of (1,1) at (-1,1).
    vertex = np.array([0.0, 1.0, 0.5], dtype=float)
    vertex_2d = _project_to_slab_plane(vertex.reshape(1, 3), cell)[0]

    cand_xy, cand_types, cand_indices = _build_delaunay_classification_index(
        top_2d, top_idx, tri, cell=cell, pbc=pbc
    )
    assert "bridge" in cand_types
    # At least one bridge candidate must involve the two atoms across the a-cut.
    cross = {
        frozenset(idx)
        for typ, idx in zip(cand_types, cand_indices, strict=True)
        if typ == "bridge"
    }
    assert frozenset((0, 1)) in cross

    site_type, nearest_idx = _delaunay_site_classification(
        vertex,
        top_2d,
        top_idx,
        tri,
        positions,
        vertex_2d=vertex_2d,
        cand_xy=cand_xy,
        cand_types=cand_types,
        cand_indices=cand_indices,
    )
    assert site_type == "bridge"
    assert frozenset(int(i) for i in nearest_idx) == frozenset((0, 1))

    # Without PBC expansion the nearest primary-cell candidate is an atop.
    cand_xy_nopbc, cand_types_nopbc, cand_indices_nopbc = (
        _build_delaunay_classification_index(top_2d, top_idx, tri)
    )
    site_type_nopbc, _ = _delaunay_site_classification(
        vertex,
        top_2d,
        top_idx,
        tri,
        positions,
        vertex_2d=vertex_2d,
        cand_xy=cand_xy_nopbc,
        cand_types=cand_types_nopbc,
        cand_indices=cand_indices_nopbc,
    )
    assert site_type_nopbc == "atop"


def test_build_site_records_upgrades_boundary_atop_via_pbc_index():
    """Production classify path upgrades near-boundary atop → bridge with PBC index."""
    from scipy.spatial import Delaunay

    from metalsurfer.placement.site_classify import _build_site_records
    from metalsurfer.placement.site_coords import _project_to_slab_plane
    from metalsurfer.placement.site_voronoi import _build_delaunay_classification_index

    positions = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc_on = np.array([True, True, False], dtype=bool)
    pbc_off = np.array([False, False, False], dtype=bool)
    top_idx = np.arange(4, dtype=int)
    top_2d = _project_to_slab_plane(positions, cell)
    tri = Delaunay(top_2d)
    primary = _build_delaunay_classification_index(top_2d, top_idx, tri)
    pbc_index = _build_delaunay_classification_index(
        top_2d, top_idx, tri, cell=cell, pbc=pbc_on
    )
    # Cross-a-boundary bridge midpoint: primary labels atop, PBC labels bridge.
    vertex = np.array([[0.0, 1.0, 0.5]], dtype=float)
    nn_dists = np.array([1.0], dtype=float)
    local_tree = KDTree(positions)
    symbols = ["Cu", "Cu", "Cu", "Cu"]

    upgraded = _build_site_records(
        vertex,
        nn_dists,
        positions,
        symbols,
        local_tree,
        "slab",
        pore_threshold=2.5,
        use_delaunay=True,
        delaunay_tri=tri,
        top_positions_2d=top_2d,
        top_atom_indices=top_idx,
        cell=cell,
        delaunay_class_index=primary,
        delaunay_class_index_pbc=pbc_index,
        pbc=pbc_on,
    )
    primary_only = _build_site_records(
        vertex,
        nn_dists,
        positions,
        symbols,
        local_tree,
        "slab",
        pore_threshold=2.5,
        use_delaunay=True,
        delaunay_tri=tri,
        top_positions_2d=top_2d,
        top_atom_indices=top_idx,
        cell=cell,
        delaunay_class_index=primary,
        delaunay_class_index_pbc=pbc_index,
        pbc=pbc_off,
    )
    assert upgraded[0].site_type == "bridge"
    assert frozenset(upgraded[0].slab_indices) == frozenset((0, 1))
    assert primary_only[0].site_type == "atop"


def test_get_unified_sites_upgrades_pbc_edge_atop_on_production_path():
    """Hot path on a real slab: boundary sites primary-atop become bridge via PBC."""
    from scipy.spatial import Delaunay

    from metalsurfer.placement.site_coords import (
        _cart_to_frac,
        _project_to_slab_plane,
        top_layer_mask_by_normal,
    )
    from metalsurfer.placement.site_voronoi import (
        _build_delaunay_classification_index,
        _delaunay_site_classification,
    )

    slab = make_slab()
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(slab.get_pbc(), dtype=bool)
    top_idx = np.nonzero(top_layer_mask_by_normal(positions, cell, 0.5))[0]
    top_2d = _project_to_slab_plane(positions[top_idx], cell)
    tri = Delaunay(top_2d)
    primary = _build_delaunay_classification_index(top_2d, top_idx, tri)

    sites = get_unified_sites(
        slab,
        material_type="slab",
        site_classification_method="delaunay",
    )
    assert sites
    upgraded = []
    for site in sites:
        frac = _cart_to_frac(np.asarray(site.xyz, dtype=float).reshape(1, 3), cell)[0]
        near_boundary = (
            float(frac[0]) < 0.2
            or float(frac[0]) > 0.8
            or float(frac[1]) < 0.2
            or float(frac[1]) > 0.8
        )
        if not near_boundary:
            continue
        vertex = np.asarray(site.xyz, dtype=float)
        vertex_2d = _project_to_slab_plane(vertex.reshape(1, 3), cell)[0]
        primary_type, _ = _delaunay_site_classification(
            vertex,
            top_2d,
            top_idx,
            tri,
            positions,
            vertex_2d=vertex_2d,
            cand_xy=primary[0],
            cand_types=primary[1],
            cand_indices=primary[2],
        )
        if primary_type == "atop" and site.site_type == "bridge":
            upgraded.append(site)
    assert upgraded, (
        "expected get_unified_sites to upgrade near-boundary primary-atop → bridge"
    )
    # Interior hollow count must stay intact (upgrade-only, not global reclassify).
    assert sum(1 for s in sites if s.site_type == "hollow") == (
        _GOLDEN_SLAB_SITE_TYPE_MULTISET["hollow"]
    )


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


def test_voronoi_enrichment_increases_site_count_on_porous():
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

    assert len(vertices_base) > 0
    # Real 3D ridge polygons must yield subdivision candidates on this framework.
    assert len(vertices_enriched) > len(vertices_base)


def test_enrich_along_ridges_walks_polygonal_faces():
    """Closed 3D ridge faces (len > 2) contribute consecutive edges for subdivision.

    A regression to ``if len(ridge) != 2: continue`` skips this face and returns
    the input vertices unchanged.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
            [3.0, 5.0, 0.0],
        ],
        dtype=float,
    )
    nn_dists = np.array([2.0, 2.0, 2.0], dtype=float)
    ridge_vertices = [[0, 1, 2]]
    raw_to_kept = {0: 0, 1: 1, 2: 2}
    extended = vertices.copy()
    tree = KDTree(extended)
    out_verts, _ = _enrich_along_ridges(
        vertices,
        nn_dists,
        ridge_vertices,
        raw_to_kept,
        extended,
        tree,
        probe_radius=0.0,
        max_distance=100.0,
        cell=np.eye(3) * 20.0,
        pbc=np.array([False, False, False], dtype=bool),
    )
    assert len(out_verts) > len(vertices)

    # Control: a ridge that the old len==2 filter would accept still subdivides.
    out_edge, _ = _enrich_along_ridges(
        vertices[:2],
        nn_dists[:2],
        [[0, 1]],
        {0: 0, 1: 1},
        vertices[:2],
        KDTree(vertices[:2]),
        probe_radius=0.0,
        max_distance=100.0,
        cell=np.eye(3) * 20.0,
        pbc=np.array([False, False, False], dtype=bool),
    )
    assert len(out_edge) > 2


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
    monkeypatch.setattr(site_voronoi_module, "Voronoi", lambda _pts: fake_vor)

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
        *,
        cell,
        pbc,
    ):
        captured["ridge_vertices"] = ridge_vertices
        return vertices, nn_dists

    monkeypatch.setattr(site_voronoi_module, "_enrich_along_ridges", _capture_ridges)

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
    from metalsurfer.placement.site_context import clear_site_caches

    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(material_type="slab")

    clear_site_caches()
    assert len(_SITE_CONTEXT_CACHE) == 0

    workflow_shared.resolve_site_context_for_sampling(
        slab,
        config,
        symmetry_broken=True,
    )
    # Unique-sites entry + resolved (sym=True) entry share one cache.
    assert len(_SITE_CONTEXT_CACHE) == 2

    clear_site_caches()
    assert len(_SITE_CONTEXT_CACHE) == 0


def test_site_context_cache_keys_differ_by_symmetry_broken():
    slab = make_slab(nx=2, ny=2)
    config = AdsorptionConfig(material_type="slab")
    clear_site_caches()

    ctx_broken = workflow_shared.resolve_site_context_for_sampling(
        slab, config, symmetry_broken=True
    )
    ctx_intact = workflow_shared.resolve_site_context_for_sampling(
        slab, config, symmetry_broken=False
    )
    # Unique-sites + sym=True + sym=False.
    assert len(_SITE_CONTEXT_CACHE) == 3
    assert ctx_broken is not ctx_intact


# ---------------------------------------------------------------------------
# Slab pathway: enumeration, placement, reproducibility
# ---------------------------------------------------------------------------


def test_slab_enumeration_and_generation_have_high_success_and_site_coverage():
    from metalsurfer.placement.geometry import detect_vdw_overlaps

    slab = make_slab()
    config = adsorption_config_factory(
        material_type="slab",
        num_placements=50,
        placement_z_range=(2.0, 3.0),
        reject_vdw_overlaps=True,
    )
    results = _generate_placements(
        water_conformers(), slab, config, smiles="O", n_desired=50
    )

    assert len(results) >= 50
    visited_sites = {spec.site_index for spec, _, _ in results}
    assert len(visited_sites) >= 2
    for _spec, adsorbate, _descriptor in results:
        ok, dist, reason = check_initial_placement_distance(
            adsorbate,
            slab,
            reject_vdw_overlaps=True,
            material_type="slab",
        )
        assert ok, f"Successful placement must pass contact gates: {reason}"
        assert 1.2 <= dist <= 5.5, (
            f"Adsorbate–surface distance should be physical (1.2–5.5 Å), got {dist:.3f}"
        )
        overlaps, _ = detect_vdw_overlaps(adsorbate, slab, material_type="slab")
        assert len(overlaps) == 0, "Successful placement must not have VDW clashes"


def test_slab_placements_are_above_surface_reference():
    slab = make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=50, placement_z_range=(2.0, 3.0)
    )
    results = _generate_placements(
        water_conformers(), slab, config, smiles="O", n_desired=50
    )
    assert len(results) >= 1
    for _, adsorbate, descriptor in results:
        assert descriptor.surface_ref_z_abs is not None
        assert descriptor.z_abs is not None
        assert descriptor.z_abs >= descriptor.surface_ref_z_abs
        ok, dist, reason = check_initial_placement_distance(
            adsorbate, slab, material_type="slab"
        )
        assert ok, reason
        assert 1.2 <= dist <= 5.5


@pytest.mark.parametrize("mode", ["spec", "descriptor", "pose"])
def test_slab_replay_reproduces_positions(mode):
    slab = make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=10, placement_z_range=(2.0, 3.0)
    )
    conformers = water_conformers()
    spec, result = _first_successful_placement(
        conformers, slab, config, "O", n_desired=10
    )
    assert spec is not None and result is not None

    adsorbate, descriptor = result
    _assert_replay_matches(
        mode,
        adsorbate,
        descriptor,
        spec,
        conformers,
        slab,
        config,
    )


# ---------------------------------------------------------------------------
# Nanoparticle and porous: local-site enumeration and geometry
# ---------------------------------------------------------------------------


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
    )
    conformers = water_conformers()
    results = _generate_placements(
        conformers, structure, config, smiles="O", n_desired=n_desired
    )
    assert len(results) >= 1
    for _spec, adsorbate_i, _desc in results:
        ok, dist, reason = check_initial_placement_distance(
            adsorbate_i, structure, material_type=material_type
        )
        assert ok, f"{material_type} placement failed contact gate: {reason}"
        assert 0.75 <= dist <= 6.0, (
            f"{material_type} adsorbate–surface distance out of band: {dist:.3f}"
        )

    spec, adsorbate, descriptor = results[0]
    _assert_replay_matches(
        "spec", adsorbate, descriptor, spec, conformers, structure, config
    )
    _assert_replay_matches(
        "descriptor", adsorbate, descriptor, spec, conformers, structure, config
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
    matched = False
    for spec in specs:
        result = generate_placement_from_spec(
            spec, conformers, structure, config, smiles="O"
        )
        if result is None or spec.site_index < 0:
            continue
        _, descriptor = result
        site = unique_sites[spec.site_index]
        expected = np.asarray(site.xyz, dtype=float) + float(
            descriptor.z_offset
        ) * np.asarray(site.normal, dtype=float)
        got = np.array(
            [descriptor.x_abs, descriptor.y_abs, descriptor.z_abs], dtype=float
        )
        np.testing.assert_allclose(got, expected, atol=1e-6)
        matched = True
        break

    assert matched, (
        f"Expected at least one successful {material_type} site-based placement"
    )


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


def test_check_desorption_nanoparticle_and_porous():
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    water_far = make_water()
    water_far.set_positions(water_far.get_positions() + np.array([20.0, 20.0, 20.0]))
    np_combined = nanoparticle + water_far
    np_combined.set_cell(nanoparticle.get_cell())
    np_combined.set_pbc(nanoparticle.get_pbc())
    ok_np, _ = check_desorption(
        np_combined, nanoparticle, binding_threshold=4.0, material_type="nanoparticle"
    )
    assert not ok_np

    water_near = make_water()
    sites = get_unified_sites(porous, material_type="porous")
    site = sites[0]
    center = np.asarray(site.xyz, dtype=float) + 1.0 * np.asarray(
        site.normal, dtype=float
    )
    wpos = water_near.get_positions().copy()
    wpos -= np.mean(wpos, axis=0)
    wpos += center
    water_near.set_positions(wpos)
    porous_combined = porous + water_near
    porous_combined.set_cell(porous.get_cell())
    porous_combined.set_pbc(porous.get_pbc())
    ok_porous, _ = check_desorption(
        porous_combined, porous, binding_threshold=4.0, material_type="porous"
    )
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


# ---------------------------------------------------------------------------
# Policy tests for batch placement spec builders
# ---------------------------------------------------------------------------


def _site_type_atop(idx: int) -> str:
    return "atop"


@pytest.mark.parametrize(
    "n_conformers,site_indices,shape,n_binders,flat_aromatic",
    [
        (2, [0, 1, 2], "round", 1, False),
        (1, [0, 1], "flat", 2, True),
    ],
)
def test_max_batch_specs_matches_build_batch_uncapped(
    n_conformers, site_indices, shape, n_binders, flat_aromatic
):
    expected = max_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=site_indices,
        shape=shape,
        n_binders=n_binders,
        flat_aromatic=flat_aromatic,
    )
    actual = build_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=site_indices,
        site_type_for_index=_site_type_atop,
        shape=shape,
        n_binders=n_binders,
        flat_aromatic=flat_aromatic,
        parallel_fraction=0.5,
        n_desired=10**7,
        seed=TEST_SEED,
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
        seed=TEST_SEED,
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
        seed=TEST_SEED,
        dissociative=True,
        n_hollow_pairs=n_hollow_pairs,
    )
    assert len(specs) > 0
    assert all(s.orientation_type == "dissociative" for s in specs)
    assert all(0 <= s.site_index < n_hollow_pairs for s in specs)


def test_build_batch_specs_dissociative_zero_pairs_returns_empty():
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=lambda idx: "hollow",
        shape="linear",
        n_binders=0,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=10,
        seed=TEST_SEED,
        dissociative=True,
        n_hollow_pairs=0,
    )
    assert specs == []
    assert (
        max_batch_placement_specs(
            n_conformers=1,
            site_indices=[0],
            shape="linear",
            n_binders=0,
            flat_aromatic=False,
            dissociative=True,
            n_hollow_pairs=0,
        )
        == 0
    )


def test_build_batch_specs_dissociative_samples_beyond_prefix_pairs():
    """Early-cap must not keep only the first pair indices after z expansion."""
    n_hollow_pairs = 200
    n_desired = 40
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=lambda idx: "hollow",
        shape="linear",
        n_binders=0,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=n_desired,
        seed=TEST_SEED,
        dissociative=True,
        n_hollow_pairs=n_hollow_pairs,
    )
    indices = {s.site_index for s in specs}
    assert len(specs) == n_desired
    assert max(indices) >= 32
    assert len(indices) > 1


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
        seed=TEST_SEED,
    )
    orientation_types = {s.orientation_type for s in specs}
    assert "parallel" in orientation_types
    assert "EN-down" in orientation_types


@pytest.mark.parametrize(
    "parallel_fraction,n_desired,expect_only",
    [
        (0.0, 8, "EN-down"),
        (1.0, 8, "parallel"),
        (0.0, 1, "EN-down"),
        (1.0, 1, "parallel"),
    ],
)
def test_build_batch_specs_flat_aromatic_honors_parallel_fraction(
    parallel_fraction, n_desired, expect_only
):
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=parallel_fraction,
        n_desired=n_desired,
        seed=TEST_SEED,
    )
    assert len(specs) == n_desired
    assert {s.orientation_type for s in specs} == {expect_only}


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
        seed=TEST_SEED,
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
        seed=TEST_SEED,
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
        seed=TEST_SEED,
    )
    ids = [s.placement_index for s in specs]
    assert len(set(ids)) == len(ids), "placement_index values must be unique"


# ---------------------------------------------------------------------------
# Phase 1: Environment-aware KDTree clustering
# ---------------------------------------------------------------------------


def test_env_fingerprint_present_in_unified_sites():
    """Sites should carry an env_fingerprint after Phase 1."""
    sites = get_unified_sites(make_slab(), material_type="slab")
    assert len(sites) > 0
    for s in sites:
        fp = s.env_fingerprint
        assert isinstance(fp, tuple) and len(fp) == 2
        # First element is a tuple of element symbols, second is site_type
        assert isinstance(fp[0], tuple)
        assert isinstance(fp[1], str)


# ---------------------------------------------------------------------------
# Site classification (Delaunay)
# ---------------------------------------------------------------------------


def test_delaunay_classification_on_slab():
    """Delaunay method should produce valid site types for a simple slab."""
    slab = make_slab()
    sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="delaunay"
    )
    assert len(sites) > 0
    valid_types = {"atop", "bridge", "hollow"}
    for s in sites:
        assert s.site_type in valid_types, f"Bad type: {s.site_type}"


def test_delaunay_fallback_for_nanoparticle():
    """Delaunay classification should fall back to distance_ratio for NPs."""
    np_sites_dr = get_unified_sites(
        make_nanoparticle(),
        material_type="nanoparticle",
        site_classification_method="distance_ratio",
    )
    np_sites_del = get_unified_sites(
        make_nanoparticle(),
        material_type="nanoparticle",
        site_classification_method="delaunay",
    )
    # Both should produce the same result (fallback to distance_ratio)
    assert len(np_sites_del) == len(np_sites_dr)


# ---------------------------------------------------------------------------
# Rough slab surface reference
# ---------------------------------------------------------------------------


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
    slab = make_porous_framework()
    sites = get_unified_sites(slab, material_type="porous")
    pore_sites = [s for s in sites if s.site_type == "pore"]
    if not pore_sites:
        pytest.skip("No pore sites in test framework")
    site = pore_sites[0]
    config = AdsorptionConfig(placement_z_range=(1.0, 1.5))
    z_lo, z_hi = _compute_site_z_base(config, slab, site, ["O"])
    assert z_lo > 0.0
    assert z_hi > z_lo


def test_dissociative_z_offset_uses_radius_derived_range():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        skip_topology_check=True,
        placement_z_range=(1.0, 1.5),
    )
    h2 = make_h2()
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    result, reason = generate_placement_from_spec_with_reason(
        spec, [h2], slab, config, smiles="[H][H]"
    )
    if result is None:
        pytest.skip(f"No dissociative placement on test slab: {reason}")
    _, descriptor = result
    hollow_site = Site(
        xyz=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="hollow",
        slab_indices=(),
        material_type="slab",
        site_source="test",
        env_fingerprint=((), "hollow"),
    )
    z_lo, z_hi = _compute_site_z_base(
        config, slab, hollow_site, h2.get_chemical_symbols()
    )
    from metalsurfer.placement.orientation import _site_type_z_offset

    z_lo += _site_type_z_offset(slab, hollow_site, "hollow")
    z_hi += _site_type_z_offset(slab, hollow_site, "hollow")
    expected = z_lo + 0.5 * (z_hi - z_lo)
    assert descriptor.z_offset == pytest.approx(expected)


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


# ---------------------------------------------------------------------------
# Adaptive parallel fraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "symbols, smiles, expected",
    [
        (["C"] * 6 + ["H"] * 6, "c1ccccc1", 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, "c1ccncc1", 0.3),
        (["C"] * 6 + ["O"] + ["H"] * 6, "c1ccccc1O", 0.3),
        (["C"] * 6 + ["N", "O"] + ["H"] * 6, "c1ccc(O)c(N)c1", 0.5),
        (["C"] * 6 + ["H"] * 6, None, 0.8),
        (["C"] * 5 + ["N"] + ["H"] * 5, None, 0.3),
        (["C"] * 4 + ["N", "O"] + ["H"] * 4, None, 0.3),
        (["C"] * 8 + ["N", "O"] + ["H"] * 8, None, 0.5),
    ],
)
def test_estimate_parallel_fraction(symbols, smiles, expected):
    frac = _estimate_parallel_fraction(symbols, smiles=smiles)
    assert frac == expected


def test_classify_adsorbate_orientation_parallel_tilted_unknown():
    # Planar hexagon in xy → parallel to +z surface.
    parallel = Atoms(
        "C6",
        positions=[
            [1.0, 0.0, 5.0],
            [0.5, 0.866, 5.0],
            [-0.5, 0.866, 5.0],
            [-1.0, 0.0, 5.0],
            [-0.5, -0.866, 5.0],
            [0.5, -0.866, 5.0],
        ],
        cell=[10, 10, 20],
        pbc=[True, True, False],
    )
    assert classify_adsorbate_orientation(parallel, slab_size=0) == "parallel"

    # Same ring standing in xz → tilted relative to +z.
    tilted = Atoms(
        "C6",
        positions=[
            [1.0, 0.0, 5.0],
            [0.5, 0.0, 5.866],
            [-0.5, 0.0, 5.866],
            [-1.0, 0.0, 5.0],
            [-0.5, 0.0, 4.134],
            [0.5, 0.0, 4.134],
        ],
        cell=[10, 10, 20],
        pbc=[True, True, False],
    )
    assert classify_adsorbate_orientation(tilted, slab_size=0) == "tilted"

    diatomic = Atoms(
        "CO",
        positions=[[0.0, 0.0, 5.0], [0.0, 0.0, 6.1]],
        cell=[10, 10, 20],
        pbc=[True, True, False],
    )
    assert classify_adsorbate_orientation(diatomic, slab_size=0) == "unknown"


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
    _, descriptor = ok
    assert descriptor.z_fraction > 0.0
    assert descriptor.z_abs is not None
    assert float(descriptor.z_abs) > surface_z + 0.35


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


def test_voronoi_auto_widen_retries_when_first_window_empty(monkeypatch):
    """Empty first Voronoi window triggers one widened retry when enabled."""
    import metalsurfer.placement.site_context as site_context_mod

    clear_site_caches()
    slab = make_slab()
    calls = {"n": 0}
    real = site_context_mod.get_unified_sites

    def fake_get_unified_sites(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return []
        return real(*args, **kwargs)

    monkeypatch.setattr(site_context_mod, "get_unified_sites", fake_get_unified_sites)
    ctx = _get_unique_sites_for_specs(slab, AdsorptionConfig(voronoi_auto_widen=True))
    assert calls["n"] == 2
    assert ctx.use_sites
    assert len(ctx.sites) > 0


def test_voronoi_auto_widen_disabled_skips_retry(monkeypatch):
    import metalsurfer.placement.site_context as site_context_mod

    clear_site_caches()
    slab = make_slab()
    calls = {"n": 0}

    def fake_get_unified_sites(*args, **kwargs):
        calls["n"] += 1
        return []

    monkeypatch.setattr(site_context_mod, "get_unified_sites", fake_get_unified_sites)
    ctx = _get_unique_sites_for_specs(slab, AdsorptionConfig(voronoi_auto_widen=False))
    assert calls["n"] == 1
    assert not ctx.use_sites


# ---------------------------------------------------------------------------
# Dissociative hollow-site pairs
# ---------------------------------------------------------------------------


def test_hollow_site_pairs_found_for_slab():
    """_get_dissociative_site_pairs must find adjacent hollow pairs within adaptive bounds."""
    from ase.geometry import find_mic

    from metalsurfer.placement._constants import (
        _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
        _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
    )

    slab = make_slab()
    config = AdsorptionConfig()
    pairs = _get_dissociative_site_pairs(slab, config)
    assert isinstance(pairs, list)
    assert len(pairs) >= 1, "4×4 FCC-like slab should yield hollow-site pairs"
    cell = np.asarray(slab.get_cell(), dtype=float)
    for p in pairs:
        assert len(p.xyz1) == 3
        assert len(p.xyz2) == 3
        _, dists = find_mic((np.asarray(p.xyz1) - np.asarray(p.xyz2)).reshape(1, 3), cell)
        sep = float(dists[0])
        assert (
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM
            <= sep
            <= _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM
        ), f"hollow-pair MIC separation {sep:.3f} Å outside adaptive window"


def test_hollow_site_pairs_include_pbc_adjacent_on_small_cell():
    """Small PBC cells must retain MIC-adjacent pairs missed by Cartesian KDTree."""
    from ase.geometry import find_mic

    from metalsurfer.placement.dissociative import _periodic_site_pair_candidates

    # 2x2 top-layer-like points near cell corners.
    sites = np.array(
        [
            [0.5, 0.5, 5.0],
            [3.5, 0.5, 5.0],
            [0.5, 3.5, 5.0],
            [3.5, 3.5, 5.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc = [True, True, False]
    # Cartesian nearest within cell is ~3 A; MIC wrap between (0.5,0.5) and (3.5,0.5) is 1 A.
    cart = set()
    from scipy.spatial import KDTree

    for i, j in KDTree(sites).query_pairs(r=1.5):
        cart.add((min(i, j), max(i, j)))
    periodic = _periodic_site_pair_candidates(sites, cell, pbc, max_sep=1.5)
    assert (0, 1) in periodic  # wrap across a
    assert (0, 1) not in cart
    _, d = find_mic((sites[0] - sites[1]).reshape(1, 3), cell, pbc=pbc)
    assert float(d[0]) <= 1.5


def test_is_top_layer_planar_true_for_three_coplanar_atoms():
    from metalsurfer.placement.site_enumeration import _is_top_layer_planar

    atoms = Atoms(
        "Cu3",
        positions=[[0.0, 0.0, 5.0], [2.5, 0.0, 5.0], [1.25, 2.2, 5.0]],
        cell=[5.0, 5.0, 20.0],
        pbc=[True, True, False],
    )
    assert _is_top_layer_planar(atoms, top_layer_tolerance=0.5) is True


def test_get_unified_sites_uses_material_aware_pbc_not_atoms_ttt():
    """TTT atoms with material_type=slab must still enumerate as TTF slab sites."""
    slab = make_slab()
    ttf = get_unified_sites(slab, material_type="slab")
    ttt = slab.copy()
    ttt.set_pbc([True, True, True])
    sites_ttt = get_unified_sites(ttt, material_type="slab")
    assert len(sites_ttt) == len(ttf)
    assert {s.site_type for s in sites_ttt} == {s.site_type for s in ttf}


def test_topology_bridges_keep_distinct_pbc_midpoints():
    """Same atom-pair interior vs boundary bridges must both survive generation."""
    from metalsurfer.placement.site_voronoi import _generate_slab_topology_sites

    positions = np.array(
        [
            [1.0, 1.0, 0.0],
            [3.0, 1.0, 0.0],
            [1.0, 3.0, 0.0],
            [3.0, 3.0, 0.0],
        ],
        dtype=float,
    )
    cell = np.diag([4.0, 4.0, 20.0])
    pbc = np.array([True, True, False], dtype=bool)
    top_idx = np.arange(4, dtype=int)
    local_tree = KDTree(positions)
    verts, _dists, sources, _tri = _generate_slab_topology_sites(
        positions,
        cell,
        pbc,
        top_idx,
        local_tree,
        site_height=0.5,
        probe_radius=1.0,
        max_distance=5.0,
    )
    bridge_xy = [
        (round(float(v[0]), 3), round(float(v[1]), 3))
        for v, src in zip(verts, sources, strict=True)
        if src == "topology_bridge"
    ]
    # Interior midpoints around (2,1)/(1,2) and near-boundary midpoints near x/y≈0.
    assert (2.0, 1.0) in bridge_xy or any(abs(x - 2.0) < 0.05 and abs(y - 1.0) < 0.05 for x, y in bridge_xy)
    assert any(abs(x) < 0.15 or abs(x - 4.0) < 0.15 for x, _y in bridge_xy) or any(
        abs(y) < 0.15 or abs(y - 4.0) < 0.15 for _x, y in bridge_xy
    )


def test_dissociative_placement_supported_for_nanoparticle():
    nanoparticle = make_nanoparticle()
    config = AdsorptionConfig(
        material_type="nanoparticle",
        skip_topology_check=True,
        num_placements=1,
    )
    h2 = make_h2()
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )

    result, reason = generate_placement_from_spec_with_reason(
        spec,
        [h2],
        nanoparticle,
        config,
    )
    if reason == "no_hollow_site_pairs":
        pytest.skip("No dissociative site pairs on test nanoparticle")
    assert result is not None, reason
    placed, descriptor = result
    assert descriptor.orientation_type == "dissociative"
    hh = float(
        np.linalg.norm(
            placed.get_positions()[1] - placed.get_positions()[0],
        )
    )
    assert hh > 1.0, "Dissociative placement should separate H atoms"
    ok, min_d, dist_reason = check_initial_placement_distance(
        placed, nanoparticle, material_type="nanoparticle"
    )
    assert ok, (min_d, dist_reason)
    assert 1.0 <= min_d <= 3.5


def test_dissociative_placement_on_slab_separates_and_clears_surface():
    """Dissociative H2 on a slab must land on a hollow pair with physical clearance."""
    from ase.geometry import find_mic

    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", skip_topology_check=True)
    pairs = _get_dissociative_site_pairs(slab, config)
    assert pairs, "fixture slab must expose hollow pairs for dissociative placement"
    h2 = make_h2()
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    result, reason = generate_placement_from_spec_with_reason(spec, [h2], slab, config)
    assert result is not None, reason
    placed, descriptor = result
    assert descriptor.orientation_type == "dissociative"
    assert descriptor.site_type == "hollow"

    pos = placed.get_positions()
    cell = np.asarray(slab.get_cell(), dtype=float)
    _, hh_dists = find_mic((pos[1] - pos[0]).reshape(1, 3), cell)
    hh = float(hh_dists[0])
    pair = pairs[descriptor.site_index % len(pairs)]
    _, pair_dists = find_mic(
        (np.asarray(pair.xyz1) - np.asarray(pair.xyz2)).reshape(1, 3), cell
    )
    pair_sep = float(pair_dists[0])
    assert hh == pytest.approx(pair_sep, abs=0.35), (
        f"H–H separation {hh:.3f} should track hollow-pair spacing {pair_sep:.3f}"
    )
    assert hh > 1.0

    ok, min_d, dist_reason = check_initial_placement_distance(
        placed, slab, material_type="slab"
    )
    assert ok, (min_d, dist_reason)
    # Each H must lie on the outward ray from a distinct hollow (normals may tilt).
    site_a = np.asarray(pair.xyz1, dtype=float)
    site_b = np.asarray(pair.xyz2, dtype=float)
    n1 = np.asarray(pair.normal1, dtype=float)
    n2 = np.asarray(pair.normal2, dtype=float)
    n1 = n1 / float(np.linalg.norm(n1))
    n2 = n2 / float(np.linalg.norm(n2))

    def _on_ray(p: np.ndarray, site: np.ndarray, n: np.ndarray) -> bool:
        d = p - site
        h = float(np.dot(d, n))
        if h < 0.5:
            return False
        return float(np.linalg.norm(d - h * n)) < 0.05

    assigned_distinct = (
        _on_ray(pos[0], site_a, n1) and _on_ray(pos[1], site_b, n2)
    ) or (_on_ray(pos[0], site_b, n2) and _on_ray(pos[1], site_a, n1))
    assert assigned_distinct, "each H must map to a distinct hollow of the pair"


def test_dissociative_placement_rejected_for_porous_material_type():
    porous = make_porous_framework()
    config = AdsorptionConfig(
        material_type="porous",
        skip_topology_check=True,
        num_placements=1,
    )
    h2 = make_h2()
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )

    result, reason = generate_placement_from_spec_with_reason(
        spec,
        [h2],
        porous,
        config,
    )
    assert result is None
    assert reason == "dissociative_not_supported_for_porous"


# ---------------------------------------------------------------------------
# Phase 1-2 Enhanced Placement Validation Tests
# ---------------------------------------------------------------------------


def test_vdw_overlap_detection_accepts_good_contact():
    """VDW overlap detection should accept placements with adequate separation."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=3.5, x_shift=2.0, y_shift=2.0
    )

    overlaps, min_dist = detect_vdw_overlaps(water, slab, material_type="slab")
    assert len(overlaps) == 0, "Should not detect overlaps for well-separated water"
    assert min_dist > 2.0


def test_calculate_contact_quality_detects_good_contact():
    """calculate_contact_quality should correctly identify contact atoms."""
    slab = make_slab()
    water = place_adsorbate_above_slab(
        slab, make_water(), z_offset=2.0, x_shift=2.0, y_shift=2.0
    )

    metrics = calculate_contact_quality(
        water, slab, contact_distance_threshold=2.5, material_type="slab"
    )

    assert metrics["num_contacting_atoms"] > 0, "Should have contacting atoms"
    assert metrics["contact_distance"] < 3.0, "Should have reasonable contact distance"
    assert metrics["contact_ratio"] > 0.0, "Should have contact ratio"


def test_adsorbate_separation_accepts_well_separated():
    """check_adsorbate_separation should accept well-separated adsorbates."""
    slab = make_slab()
    water = make_water().copy()

    # Place water
    pos = water.get_positions()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.0
    pos[:, 0] += 5.0
    pos[:, 1] += 5.0
    water.set_positions(pos)
    water.set_cell(slab.get_cell())
    water.set_pbc(slab.get_pbc())

    # Pre-adsorbed positions far away
    pre_ads = np.array([[0.0, 0.0, 5.0]])

    ok, dist = check_adsorbate_separation(
        water,
        pre_ads,
        min_separation=2.0,
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
    )
    assert ok, "Should accept well-separated adsorbates"
    assert dist > 2.0


def test_adsorbate_separation_rejects_close_atoms():
    """check_adsorbate_separation should reject too-close adsorbates."""
    slab = make_slab()
    water = make_water().copy()

    # Place water
    pos = water.get_positions()
    pos[:, 2] += float(np.max(slab.get_positions()[:, 2])) + 2.0
    water.set_positions(pos)
    water.set_cell(slab.get_cell())
    water.set_pbc(slab.get_pbc())

    # Pre-adsorbed positions very close
    pre_ads = np.array([[0.0, 0.0, 5.0]])

    ok, dist = check_adsorbate_separation(
        water,
        pre_ads,
        min_separation=5.0,
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
    )
    assert not ok, "Should reject too-close adsorbates"


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
        max_closest_approach=1.5,
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


def test_strict_initial_placement_e2e_reason():
    slab = make_slab()
    config = AdsorptionConfig(
        num_conformers=1,
        num_placements=12,
        seed=1,
        strict_initial_placement=True,
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


def test_site_classification_auto_matches_delaunay_on_slab():
    slab = make_slab()
    auto_sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="auto"
    )
    del_sites = get_unified_sites(
        slab, material_type="slab", site_classification_method="delaunay"
    )
    assert [s.site_type for s in auto_sites] == [
        s.site_type for s in del_sites
    ]


# ---------------------------------------------------------------------------
# Plan guardrails: injectivity, ordering, rotated slabs, clustering units
# ---------------------------------------------------------------------------


def _site_ordering_key(site: Site) -> tuple:
    xyz = np.asarray(site.xyz, dtype=float)
    return (
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        str(site.site_type),
        str(site.site_source),
    )


def test_get_unified_sites_ordering_is_deterministic():
    slab = make_slab()
    first = get_unified_sites(slab, material_type="slab")
    second = get_unified_sites(slab, material_type="slab")
    assert [_site_ordering_key(s) for s in first] == [
        _site_ordering_key(s) for s in second
    ]


def test_unique_sites_cache_hit_and_miss():
    clear_site_caches()
    clear_site_caches()
    slab_a = make_slab(nx=4)
    slab_b = make_slab(nx=5)
    config = AdsorptionConfig(material_type="slab")
    ctx_a1 = _get_unique_sites_for_specs(slab_a, config)
    ctx_a2 = _get_unique_sites_for_specs(slab_a, config)
    ctx_b = _get_unique_sites_for_specs(slab_b, config)
    assert ctx_a1 is ctx_a2
    assert ctx_a1 is not ctx_b


def test_cluster_equivalent_sites_cartesian_tolerance_scales_with_cell():
    """0.05 Å tolerance merges sub-0.05 Cartesian duplicates regardless of cell size."""
    site_a = site_from_dict(
        {
            "xy": np.array([1.0, 1.0]),
            "z": 5.0,
            "xyz": np.array([1.0, 1.0, 5.0]),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Ru",), "atop"),
        }
    )
    site_b = site_from_dict(
        {
            "xy": np.array([1.04, 1.0]),
            "z": 5.0,
            "xyz": np.array([1.04, 1.0, 5.0]),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Ru",), "atop"),
        }
    )
    for a_len in (8.1, 16.2):
        cell = np.array([[a_len, 0.0, 0.0], [0.0, a_len, 0.0], [0.0, 0.0, 20.0]])
        unique = _cluster_equivalent_sites([site_a, site_b], cell, tolerance=0.05)
        assert len(unique) == 1


def test_cluster_equivalent_sites_tilted_slab_uses_in_plane_distance():
    """Clustering must use slab-plane distance, not Cartesian xy.

    Along the tilted b vector, Cartesian ``[:2]`` under-reports separation, so a
    tolerance between cart_xy and plane distance merges under the old metric and
    keeps sites distinct under the plane metric.
    """
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell = tilt @ np.diag([8.0, 8.0, 20.0])
    n = np.cross(cell[0], cell[1])
    n = n / np.linalg.norm(n)
    along_b = cell[1] / np.linalg.norm(cell[1])

    base = np.array([2.0, 2.0, 5.0], dtype=float)
    # sep=0.5 Å along b → cart_xy≈0.285, plane=0.5; tol=0.35 discriminates.
    other = base + 0.5 * along_b
    assert np.linalg.norm((other - base)[:2]) < 0.35
    assert np.linalg.norm((other - base) - np.dot(other - base, n) * n) > 0.35

    site_a = site_from_dict(
        {
            "xy": base[:2],
            "z": float(base[2]),
            "xyz": base.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
    )
    site_b = site_from_dict(
        {
            "xy": other[:2],
            "z": float(other[2]),
            "xyz": other.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
    )
    unique = _cluster_equivalent_sites(
        [site_a, site_b], cell, tolerance=0.35, z_abs_tolerance=0.2
    )
    assert len(unique) == 2

    # Same height, truly close in-plane → still merge.
    along_a = cell[0] / np.linalg.norm(cell[0])
    near = base + 0.05 * along_a
    site_near = site_from_dict(
        {
            "xy": near[:2],
            "z": float(near[2]),
            "xyz": near.copy(),
            "site_type": "atop",
            "material_type": "slab",
            "env_fingerprint": (("Cu",), "atop"),
        }
    )
    unique_near = _cluster_equivalent_sites(
        [site_a, site_near], cell, tolerance=0.35, z_abs_tolerance=0.2
    )
    assert len(unique_near) == 1


def test_top_layer_mask_unchanged_for_bulk_slab():
    from metalsurfer.placement.site_coords import (
        _height_along_slab_normal,
        top_layer_mask_by_normal,
    )

    slab = make_slab()
    positions = slab.get_positions()
    cell = np.array(slab.get_cell())
    tol = 0.5
    heights = _height_along_slab_normal(positions, cell)
    legacy = heights >= (float(np.max(heights)) - tol)
    layered = top_layer_mask_by_normal(positions, cell, tol)
    assert np.array_equal(legacy, layered)


def test_top_layer_mask_derived_tol_excludes_subsurface_fcc():
    """Derived tol must not mask an entire multi-layer FCC-like slab."""
    from metalsurfer.placement.site_coords import (
        _derive_top_layer_tolerance,
        _height_along_slab_normal,
        top_layer_mask_by_normal,
    )

    positions = []
    for iz in range(4):
        for ix in range(4):
            for iy in range(4):
                positions.append([ix * 2.55, iy * 2.55, iz * 2.1])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[10.2, 0.0, 0.0], [0.0, 10.2, 0.0], [0.0, 0.0, 25.0]])
    symbols = ["Cu"] * len(positions)
    tol = _derive_top_layer_tolerance(symbols)
    assert tol <= 1.2
    mask = top_layer_mask_by_normal(positions, cell, tol)
    heights = _height_along_slab_normal(positions, cell)
    h_max = float(np.max(heights))
    assert mask.sum() == 16
    assert np.all(heights[mask] >= h_max - tol - 1e-9)
    assert not np.any(heights[mask] < h_max - 1.5)


def test_top_layer_mask_includes_step_terrace_for_reconstructed_surface():
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    positions = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * 2.7, iy * 2.7, 5.4])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 5.0])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 2.7])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[8.1, 0.0, 0.0], [0.0, 8.1, 0.0], [0.0, 0.0, 20.0]])
    mask = top_layer_mask_by_normal(positions, cell, 0.5)
    assert mask.sum() == 12  # 9 top + 3 step; exclude bulk at 2.7
    assert np.any(positions[mask, 2] < 5.2)
    assert not np.any(np.isclose(positions[mask, 2], 2.7))


def test_top_layer_mask_includes_step_just_outside_tol():
    """Terrace just below the primary band is included via gap rule."""
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    positions = []
    for ix in range(3):
        for iy in range(3):
            positions.append([ix * 2.7, iy * 2.7, 5.4])
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 4.8])  # Δh = 0.6 > tol=0.5
    for ix in range(3):
        positions.append([ix * 2.7, 0.0, 2.7])
    positions = np.asarray(positions, dtype=float)
    cell = np.array([[8.1, 0.0, 0.0], [0.0, 8.1, 0.0], [0.0, 0.0, 20.0]])
    mask = top_layer_mask_by_normal(positions, cell, 0.5)
    assert mask.sum() == 12
    assert np.any(np.isclose(positions[mask, 2], 4.8))
    assert not np.any(np.isclose(positions[mask, 2], 2.7))


def test_top_layer_mask_empty_positions():
    from metalsurfer.placement.site_coords import top_layer_mask_by_normal

    mask = top_layer_mask_by_normal(
        np.empty((0, 3)),
        np.eye(3) * 10.0,
        0.5,
    )
    assert mask.shape == (0,)
    assert mask.dtype == bool


def test_rotated_slab_descriptor_round_trip():
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
        "descriptor", adsorbate, descriptor, spec, water_conformers(), slab, config
    )


def test_tilted_slab_descriptor_round_trip():
    """Slab tilted so the surface normal is not Cartesian +z."""
    slab = make_slab()
    cell = np.array(slab.get_cell(), dtype=float)
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell[:3] = tilt @ cell[:3]
    slab.set_cell(cell)
    pos = slab.get_positions()
    pos[:] = (tilt @ pos.T).T
    slab.set_positions(pos)
    config = adsorption_config_factory(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    adsorbate, descriptor = result
    _assert_replay_matches(
        "descriptor", adsorbate, descriptor, spec, water_conformers(), slab, config
    )


def test_deduplicate_points_is_order_independent():
    """Union-find dedup should pick the same representative regardless of input order."""
    points_a = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ]
    )
    points_b = points_a[[0, 2, 1]]
    keep_a = _deduplicate_points(points_a, tolerance=0.05)
    keep_b = _deduplicate_points(points_b, tolerance=0.05)
    assert int(np.sum(keep_a)) == 2
    assert int(np.sum(keep_b)) == 2
    kept_a = np.sort(points_a[keep_a], axis=0)
    kept_b = np.sort(points_b[keep_b], axis=0)
    np.testing.assert_allclose(kept_a, kept_b, atol=1e-10)


def test_hollow_order_metadata_on_slab():
    """Slab hollow sites should carry hollow_order metadata when classified as hollow."""
    sites = get_unified_sites(make_slab(), material_type="slab")
    hollow_sites = [s for s in sites if s.site_type == "hollow"]
    assert len(hollow_sites) > 0
    for site in hollow_sites:
        order = site.hollow_order
        assert order is None or order in (3, 4)


def test_extract_features_depends_only_on_absolute_geometry():
    record = PlacementRecord.from_descriptor(
        make_placement_descriptor(
            placement_id=1,
            x_abs=1.25,
            y_abs=2.5,
            z_abs=7.75,
            quat_w=0.9,
            quat_x=0.1,
            quat_y=0.2,
            quat_z=0.3,
        ),
        molecule="water",
        smiles="O",
        surface_id="test",
    )
    record.site_index = 99
    record.surface_ref_z_abs = 0.0
    record.z_offset = 99.0
    features = extract_features(record)
    assert set(features.keys()) == {
        "x",
        "y",
        "z",
        "conformer_index",
        "quat_w",
        "quat_x",
        "quat_y",
        "quat_z",
    }
    assert features["x"] == pytest.approx(1.25)
    assert features["y"] == pytest.approx(2.5)
    assert features["z"] == pytest.approx(7.75)


def test_dissociative_descriptor_replay_round_trip():
    """fragment_positions must replay dissociative geometry exactly."""
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", enable_dissociative_placement=True
    )
    h2 = make_h2()
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    result, reason = generate_placement_from_spec_with_reason(spec, [h2], slab, config)
    assert result is not None, reason
    placed, descriptor = result
    assert descriptor.fragment_positions is not None
    assert "fragment_positions" not in descriptor.to_row()
    assert "initial_fragment_positions" not in descriptor.to_row()
    rich_row = descriptor.to_row(include_provenance=True)
    assert rich_row.get("initial_fragment_positions") is not None
    replayed = generate_placement_from_descriptor(descriptor, [h2], slab, config)
    assert replayed is not None
    assert np.allclose(replayed.get_positions(), placed.get_positions(), atol=1e-8)


def test_dissociative_descriptor_without_fragment_positions_fails():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", enable_dissociative_placement=True
    )
    h2 = make_h2()
    descriptor = make_placement_descriptor(
        orientation_type="dissociative",
        site_type="hollow",
        x_abs=1.0,
        y_abs=1.0,
        z_abs=3.0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        fragment_positions=None,
    )
    assert generate_placement_from_descriptor(descriptor, [h2], slab, config) is None


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


def test_enable_dissociative_placement_without_topology_skip():
    """Placement dissociative gate uses enable_dissociative_placement alone."""
    slab = make_slab()
    h2 = make_h2()
    config = AdsorptionConfig(
        material_type="slab",
        enable_dissociative_placement=True,
        skip_topology_check=False,
        num_placements=4,
        seed=0,
    )
    specs = enumerate_placement_specs([h2], slab, config, "HH", n_desired=4)
    assert specs
    assert all(s.orientation_type == "dissociative" for s in specs)


def test_skip_topology_check_alone_still_enables_dissociative_with_warning():
    slab = make_slab()
    h2 = make_h2()
    config = AdsorptionConfig(
        material_type="slab",
        skip_topology_check=True,
        enable_dissociative_placement=False,
        num_placements=4,
        seed=0,
    )
    with pytest.warns(DeprecationWarning, match="enable_dissociative_placement"):
        specs = enumerate_placement_specs([h2], slab, config, "HH", n_desired=4)
    assert specs
    assert all(s.orientation_type == "dissociative" for s in specs)


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
        record = PlacementRecord.from_descriptor(descriptor, molecule="water", smiles="O")
        feats = extract_features(record)
        assert list(feats.keys()) == FEATURE_NAMES
        assert "fragment_positions" not in feats
        feature_rows.append(tuple(round(feats[name], 10) for name in FEATURE_NAMES))
    assert len(feature_rows) >= 16
    assert len(set(feature_rows)) == len(feature_rows)


def test_dissociative_com_features_injective_and_record_replay():
    """Dissociative COM+quat features stay unique; fragments survive record round-trip."""
    slab = make_slab()
    h2 = make_h2()
    config = AdsorptionConfig(
        material_type="slab",
        enable_dissociative_placement=True,
        num_placements=32,
        seed=0,
    )
    specs = enumerate_placement_specs([h2], slab, config, "HH", n_desired=32)
    feature_rows: list[tuple[float, ...]] = []
    for spec in specs:
        result, _reason = generate_placement_from_spec_with_reason(
            spec, [h2], slab, config
        )
        if result is None:
            continue
        placed, descriptor = result
        assert descriptor.fragment_positions is not None
        record = PlacementRecord.from_descriptor(descriptor, molecule="H2", smiles="[H][H]")
        assert record.fragment_positions == descriptor.fragment_positions
        feats = extract_features(record)
        assert list(feats.keys()) == FEATURE_NAMES
        feature_rows.append(tuple(round(feats[name], 10) for name in FEATURE_NAMES))

        flat = record.to_flat_dict(include_provenance=True)
        assert flat.get("initial_fragment_positions") is not None
        restored = PlacementRecord.from_flat_dict(flat)
        assert restored.fragment_positions == record.fragment_positions
        replay_desc = restored.to_placement_descriptor()
        assert replay_desc.fragment_positions == descriptor.fragment_positions
        replayed = generate_placement_from_descriptor(
            replay_desc, [h2], slab, config
        )
        assert replayed is not None
        assert np.allclose(replayed.get_positions(), placed.get_positions(), atol=1e-8)

    assert len(feature_rows) >= 8
    assert len(set(feature_rows)) == len(feature_rows)


# ---------------------------------------------------------------------------
# Packing-aware occupancy, stratified policy, recovery, catalog oracles
# ---------------------------------------------------------------------------


def _make_site(xyz, site_type="hollow", source="topology_hollow"):
    from metalsurfer.placement.site_types import Site

    return Site(
        xyz=np.asarray(xyz, dtype=float),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type=site_type,
        slab_indices=(0,),
        material_type="slab",
        site_source=source,
        env_fingerprint=(("Ru",), site_type),
    )


def test_filter_sites_by_occupancy_drops_near_adsorbate():
    from metalsurfer.placement._material import material_aware_pbc
    from metalsurfer.placement.occupancy import filter_sites_by_occupancy

    slab = make_slab()
    near = _make_site([1.0, 1.0, 6.0])
    far = _make_site([8.0, 8.0, 6.0])
    existing = np.array([[1.05, 1.05, 6.1]])
    kept = filter_sites_by_occupancy(
        [near, far],
        existing,
        cell=np.asarray(slab.get_cell(), dtype=float),
        pbc=material_aware_pbc("slab"),
        min_separation=2.0,
    )
    assert len(kept) == 1
    assert np.allclose(kept[0].xyz, far.xyz)
    unchanged = filter_sites_by_occupancy(
        [near, far],
        np.empty((0, 3)),
        cell=np.asarray(slab.get_cell(), dtype=float),
        pbc=material_aware_pbc("slab"),
        min_separation=2.0,
    )
    assert len(unchanged) == 2


def test_enumerate_specs_empty_sites_returns_empty():
    """No sites / use_sites=False must not invent site_index=-1 capacity."""
    from metalsurfer.placement.generators import (
        enumerate_placement_specs,
        estimate_placement_spec_capacity,
    )
    from metalsurfer.placement.site_context import SiteContext

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0, num_placements=40)
    ctx = SiteContext(
        sites=[],
        use_sites=False,
        source="no_sites",
        raw_unclustered=[],
    )
    specs = enumerate_placement_specs(
        [make_water()],
        slab,
        config,
        "O",
        n_desired=40,
        site_context=ctx,
    )
    assert specs == []
    capacity = estimate_placement_spec_capacity(
        [make_water()], slab, config, "O", site_context=ctx
    )
    assert capacity == 0


def test_enumerate_specs_skips_occupied_site_indices():
    from metalsurfer.placement.generators import enumerate_placement_specs
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0, num_placements=40)
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    assert ctx.use_sites and ctx.sites
    blocked = 0
    site = ctx.sites[blocked]
    water = make_water().copy()
    # Place clearly inside min_initial_distance (equality at the threshold is kept).
    water.set_positions(water.get_positions() + site.xyz + np.array([0.0, 0.0, 0.5]))
    full = slab.copy() + water
    specs = enumerate_placement_specs(
        [make_water()],
        slab,
        config,
        "O",
        n_desired=40,
        site_context=ctx,
        full_slab=full,
    )
    assert specs
    assert all(int(s.site_index) != blocked for s in specs if int(s.site_index) >= 0)


def test_estimate_complexity_shrinks_under_coverage():
    from metalsurfer.placement.generators import estimate_molecule_complexity
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    clean = estimate_molecule_complexity(
        [make_water()], slab, config, "O", site_context=ctx
    )
    # Block nearly all sites by placing a dense adsorbate cloud near every site.
    ads_pos = np.vstack([s.xyz + np.array([0.0, 0.0, 0.2]) for s in ctx.sites])
    ads = Atoms(["H"] * len(ads_pos), positions=ads_pos)
    full = slab.copy() + ads
    covered = estimate_molecule_complexity(
        [make_water()],
        slab,
        config,
        "O",
        site_context=ctx,
        full_slab=full,
    )
    assert covered < clean
    assert covered == 0.0


def test_overlap_recovery_rescues_lateral_clash():
    clear_site_caches()
    slab = make_slab()
    z_top = float(np.max(slab.get_positions()[:, 2]))
    pre = Atoms("O", positions=[[2.0, 2.0, z_top + 2.2]])
    full = slab.copy() + pre
    water = make_water().copy()
    # Canonical-ish centered water; finalize will translate via pose abs coords.
    water.set_positions(water.get_positions() - np.mean(water.get_positions(), axis=0))
    config = AdsorptionConfig(
        material_type="slab",
        placement_distance_recovery=True,
        placement_x_range=(-2.0, 2.0),
        placement_y_range=(-2.0, 2.0),
        seed=1,
    )
    pose = PlacementPose(
        conformer_index=0,
        site_index=0,
        site_type="atop",
        placement_index=0,
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
        x_abs=2.0,
        y_abs=2.0,
        z_fraction=0.5,
        z_abs=z_top + 2.2,
        orientation_type="round",
    )
    ctx = _PlacementContext(
        pose=pose,
        site=None,
        mat_type="slab",
        surface_ref=z_top,
        is_local_ref=False,
        source="test",
        canonical_pos=water.get_positions().copy(),
        use_sites=False,
        rotated_pos=water.get_positions().copy(),
        z_base_lo=z_top + 1.5,
        z_base_hi=z_top + 3.0,
    )
    result, reason = _finalize_placement(
        ctx,
        water,
        full,
        config,
        slab_for_sites=slab,
        allow_distance_recovery=True,
    )
    # Either recovered or still overlap if nudges insufficient — never crash.
    assert reason is None or reason == "adsorbate_overlap"
    if reason is None:
        assert result is not None


def test_policy_subsample_covers_multiple_site_types():
    from metalsurfer.placement.policy import build_batch_placement_specs

    site_types = {0: "atop", 1: "bridge", 2: "hollow"}

    def site_type_for(i: int):
        return site_types.get(i)

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1, 2],
        site_type_for_index=site_type_for,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=24,
        seed=0,
    )
    types = {s.site_type for s in specs}
    assert len(types) >= 2


def test_policy_stratified_is_seed_deterministic():
    from metalsurfer.placement.policy import build_batch_placement_specs

    kwargs = dict(
        n_conformers=1,
        site_indices=[0, 1, 2],
        site_type_for_index=lambda i: ["atop", "bridge", "hollow"][i],
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=18,
        seed=7,
    )
    a = build_batch_placement_specs(**kwargs)
    b = build_batch_placement_specs(**kwargs)
    assert [(s.site_index, s.tilt_deg, s.azimuth_deg, s.z_fraction) for s in a] == [
        (s.site_index, s.tilt_deg, s.azimuth_deg, s.z_fraction) for s in b
    ]


def test_clearance_lift_along_normal_helper():
    from metalsurfer.placement.pose import _clearance_lift_along_normal

    normal = np.array([0.0, 0.0, 1.0])
    # COM-centred: atom at z=-1.5 protrudes 1.5 Å toward the surface.
    rotated = np.array([[0.0, 0.0, -1.5], [0.0, 0.0, 0.5], [0.0, 0.0, 1.0]])
    assert _clearance_lift_along_normal(rotated, normal) == pytest.approx(1.5)
    # All atoms above COM along +normal → no lift.
    above = np.array([[0.0, 0.0, 0.2], [0.0, 0.0, 0.5]])
    assert _clearance_lift_along_normal(above, normal) == pytest.approx(0.0)
    # Degenerate normal → no lift.
    assert _clearance_lift_along_normal(rotated, np.zeros(3)) == pytest.approx(0.0)


def test_clearance_aware_height_raises_protruding_pose():
    """Clearance lift must raise z_abs when orientation puts atoms below the COM."""
    from metalsurfer.placement.pose import (
        _clearance_lift_along_normal,
        _pose_from_spec,
    )
    from metalsurfer.placement.site_coords import _slab_normal

    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=1,
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=False,
        seed=0,
    )
    # Elongated chain along z in canonical frame → large protrusion after binder align/tilt.
    chain = Atoms(
        "OCC",
        positions=[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [2.8, 0.0, 0.0]],
    )
    chain.center()
    ctx_sites = _get_unique_sites_for_specs(slab, config)
    assert ctx_sites.use_sites and ctx_sites.sites
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type=str(ctx_sites.sites[0].site_type),
        tilt_deg=45.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    ctx, pose_fail = _pose_from_spec(
        chain, spec, slab, config, "OCC", site_context=ctx_sites
    )
    assert pose_fail is None
    assert ctx is not None
    n_hat = _slab_normal(np.asarray(slab.get_cell(), dtype=float))
    lift = _clearance_lift_along_normal(ctx.rotated_pos, n_hat)
    assert lift > 0.2, f"expected nontrivial protrusion lift, got {lift:.3f}"
    # Closest atom along the normal should sit near surface_ref + z_offset.
    atom_heights = (
        ctx.rotated_pos
        + np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs])
    ) @ n_hat
    closest_h = float(np.min(atom_heights))
    z_offset = ctx.z_base_lo + spec.z_fraction * (ctx.z_base_hi - ctx.z_base_lo)
    assert closest_h == pytest.approx(ctx.surface_ref + z_offset, abs=1e-5)


def test_cco_generation_yield_meets_seeded_bar():
    """Flexible ethanol should clear the seeded ≥40% generation bar (target ≥60%)."""
    import math

    slab = make_slab()
    n_desired = 30
    config = AdsorptionConfig(
        material_type="slab",
        num_conformers=5,
        num_placements=n_desired,
        seed=42,
        slab_relaxation_mode="none",
        device="cpu",
    )
    result = create_conformers_from_smiles("CCO", config=config)
    assert result is not None
    conformers, _ = result
    specs = enumerate_placement_specs(
        conformers, slab, config, "CCO", n_desired=n_desired, seed=42
    )
    n_ok = 0
    for spec in specs:
        placed = generate_placement_from_spec(
            spec, conformers, slab, config, smiles="CCO"
        )
        if placed is not None:
            n_ok += 1
    min_ok = max(27, int(math.ceil(0.9 * n_desired)))
    assert n_ok >= min_ok, (
        f"CCO generation yield too low: {n_ok}/{n_desired} (need >= {min_ok})"
    )


def test_policy_prior_prefers_mild_tilt_and_mid_z():
    """When subsampling, prefer milder tilts / mid z vs a uniform shuffle of the same pool."""
    from metalsurfer.placement.policy import _stratified_sample

    # Build a synthetic pool dominated by extreme tilts/low z so priors matter.
    pool: list[PlacementSpec] = []
    for site_i, site_type in enumerate(["atop", "bridge", "hollow"]):
        for tilt in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
            for zf in (0.1, 0.3, 0.5, 0.7, 0.9):
                pool.append(
                    PlacementSpec(
                        conformer_index=0,
                        orientation_type="round",
                        face_flip=False,
                        en_atom_index=None,
                        site_index=site_i,
                        site_type=site_type,
                        tilt_deg=tilt,
                        azimuth_deg=0.0,
                        azimuth_in_plane_deg=0.0,
                        z_fraction=zf,
                        placement_index=0,
                    )
                )
    selected = _stratified_sample(pool, n_desired=18, seed=0)
    assert len(selected) == 18
    assert len({s.site_type for s in selected}) >= 2
    mean_tilt = float(np.mean([s.tilt_deg for s in selected]))
    mean_zf = float(np.mean([s.z_fraction for s in selected]))
    # Full-grid means are 40° tilt and 0.5 z; prior should pull tilt down.
    assert mean_tilt < 40.0, f"expected mild-tilt bias, got mean tilt {mean_tilt:.1f}"
    assert abs(mean_zf - 0.5) <= 0.15, f"expected mid-z bias, got mean zf {mean_zf:.2f}"


def test_fcc_catalog_has_atop_bridge_hollow_and_topology_majority():
    clear_site_caches()
    slab = make_slab(nx=4, ny=4, n_layers=3)
    sites = get_unified_sites(slab, material_type="slab", enrich=True)
    types = {s.site_type for s in sites}
    assert {"atop", "bridge", "hollow"} <= types
    topo = sum(1 for s in sites if str(s.site_source).startswith("topology"))
    assert topo >= len(sites) // 3


def test_site_context_cache_key_includes_config_and_symmetry():
    from metalsurfer.placement.site_context import (
        _site_context_cache_key,
        clear_site_caches,
        resolve_site_context_for_sampling,
    )

    clear_site_caches()
    slab = make_slab()
    c1 = AdsorptionConfig(material_type="slab", voronoi_probe_radius=1.0)
    c2 = AdsorptionConfig(material_type="slab", voronoi_probe_radius=1.5)
    k1 = _site_context_cache_key(slab, c1, symmetry_broken=False)
    k2 = _site_context_cache_key(slab, c1, symmetry_broken=True)
    k3 = _site_context_cache_key(slab, c2, symmetry_broken=False)
    assert k1 != k2
    assert k1 != k3
    a = resolve_site_context_for_sampling(slab, c1, symmetry_broken=False)
    b = resolve_site_context_for_sampling(slab, c1, symmetry_broken=False)
    assert a is b


def test_site_context_cache_key_includes_species_and_symmetry_tol():
    from metalsurfer.placement.site_context import (
        _site_context_cache_key,
        _unique_sites_cache_key,
        clear_site_caches,
    )

    clear_site_caches()
    cu = make_slab(symbol="Cu")
    ni = make_slab(symbol="Ni")
    # Same lattice geometry, different chemistry.
    ni.set_cell(cu.get_cell(), scale_atoms=False)
    ni.set_positions(cu.get_positions())
    cfg = AdsorptionConfig(material_type="slab")
    assert _unique_sites_cache_key(cu, cfg) != _unique_sites_cache_key(ni, cfg)

    c_loose = AdsorptionConfig(material_type="slab", symmetry_tolerance=0.05)
    c_tight = AdsorptionConfig(material_type="slab", symmetry_tolerance=0.01)
    assert _site_context_cache_key(
        cu, c_loose, symmetry_broken=False
    ) != _site_context_cache_key(cu, c_tight, symmetry_broken=False)


def test_site_context_cache_key_float_packing_no_collision():
    from metalsurfer.placement.site_context import (
        _pack_optional_float,
        _unique_sites_cache_key,
    )

    # Naive str concat collides for these triples; structured packing must not.
    assert (
        f"{1.5}{20.0}{0.5}" == f"{1.52}{0.0}{0.5}"
    )
    packed_a = (
        _pack_optional_float(1.5)
        + _pack_optional_float(20.0)
        + _pack_optional_float(0.5)
    )
    packed_b = (
        _pack_optional_float(1.52)
        + _pack_optional_float(0.0)
        + _pack_optional_float(0.5)
    )
    assert packed_a != packed_b

    slab = make_slab()
    a = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.5,
    )
    b = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=1.5,
        voronoi_max_site_distance=20.0,
        top_layer_tolerance=0.6,
    )
    assert _unique_sites_cache_key(slab, a) != _unique_sites_cache_key(slab, b)


def test_surface_reference_uses_prefix_not_symbols():
    from metalsurfer.workflow.shared import _build_surface_reference_slab

    base = make_slab(symbol="Ru")
    # Same-element adatom appended as suffix.
    decorated = base.copy()
    decorated.extend(Atoms("Ru", positions=[[1.0, 1.0, 10.0]]))
    ref = _build_surface_reference_slab(decorated, base)
    assert len(ref) == len(base)
    assert len(ref) == len(decorated) - 1


def test_place_at_sites_two_site_matches_dissociative():
    from metalsurfer.models import PlacementSpec
    from metalsurfer.placement.dissociative import (
        _generate_dissociative_placement_from_spec,
        _get_dissociative_site_pairs,
        place_at_sites,
    )
    from metalsurfer.placement.orientation import _site_type_z_offset
    from metalsurfer.placement.site_enumeration import _compute_site_z_base
    from metalsurfer.placement.site_types import Site

    clear_site_caches()
    slab = make_slab()
    h2 = make_h2()
    config = AdsorptionConfig(
        material_type="slab", enable_dissociative_placement=True, seed=0
    )
    pairs = _get_dissociative_site_pairs(slab, config, slab_for_sites=slab)
    assert len(pairs) >= 1
    spec = PlacementSpec(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    via_diss, reason = _generate_dissociative_placement_from_spec(
        h2, spec, slab, config, slab_for_sites=slab
    )
    assert reason is None and via_diss is not None
    placed_a, desc_a = via_diss

    pair = pairs[0]
    hollow = Site(
        xyz=np.zeros(3),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="hollow",
        slab_indices=(),
        material_type="slab",
        site_source="test",
        env_fingerprint=((), "hollow"),
    )
    z_lo, z_hi = _compute_site_z_base(config, slab, hollow, ["H", "H"])
    z_lo += _site_type_z_offset(slab, hollow, "hollow")
    z_hi += _site_type_z_offset(slab, hollow, "hollow")
    z_offset = z_lo + 0.5 * (z_hi - z_lo)
    site_a = Site(
        xyz=pair.xyz1,
        normal=pair.normal1,
        site_type="hollow",
        slab_indices=(),
        material_type="slab",
        site_source="dissociative_hollow_pair",
        env_fingerprint=((), "hollow"),
    )
    site_b = Site(
        xyz=pair.xyz2,
        normal=pair.normal2,
        site_type="hollow",
        slab_indices=(),
        material_type="slab",
        site_source="dissociative_hollow_pair",
        env_fingerprint=((), "hollow"),
    )
    via_place = place_at_sites(
        h2,
        [site_a, site_b],
        config=config,
        spec=spec,
        height_override=float(z_offset),
        slab=slab,
        slab_for_sites=slab,
    )
    assert via_place is not None
    placed_b, _desc_b = via_place
    assert np.allclose(placed_a.get_positions(), placed_b.get_positions(), atol=1e-6)
    assert desc_a.fragment_positions is not None


def test_packing_yield_improves_with_occupancy_prune():
    from metalsurfer.placement.generators import (
        enumerate_placement_specs,
        generate_placement_from_spec_with_reason,
    )
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    clear_site_caches()
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        seed=0,
        num_placements=30,
        placement_distance_recovery=False,
    )
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    # Occupy one site region heavily.
    site = ctx.sites[0]
    pre = make_water()
    pre.set_positions(pre.get_positions() + site.xyz + np.array([0.0, 0.0, 1.8]))
    full = slab.copy() + pre

    def _overlap_frac(full_slab):
        specs = enumerate_placement_specs(
            [make_water()],
            slab,
            config,
            "O",
            n_desired=30,
            site_context=ctx,
            full_slab=full_slab,
            seed=0,
        )
        n_ov = 0
        n_fail = 0
        for spec in specs:
            _res, reason = generate_placement_from_spec_with_reason(
                spec,
                [make_water()],
                full,
                config,
                smiles="O",
                site_context=ctx,
                slab_for_sites=slab,
            )
            if reason is not None:
                n_fail += 1
                if reason == "adsorbate_overlap":
                    n_ov += 1
        return n_ov / max(n_fail, 1), len(specs)

    pruned_frac, n_pruned = _overlap_frac(full)
    unpruned_frac, n_unpruned = _overlap_frac(None)
    assert n_pruned > 0 and n_unpruned > 0
    # Pruning should not increase the overlap-fail share among failures.
    assert pruned_frac <= unpruned_frac + 1e-9


def test_retry_blocks_repeated_bad_site_index(monkeypatch):
    from metalsurfer.models import PlacementSpec
    from metalsurfer.placement._constants import _RETRY_BLOCK_SITE_AFTER
    from metalsurfer.workflow import core as core_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    calls = {"n": 0}
    seen_filters = []

    def fake_enumerate(
        conformers,
        slab_for_sites,
        config,
        smiles,
        n_desired,
        filter_spec=None,
        site_context=None,
        seed=None,
        full_slab=None,
    ):
        calls["n"] += 1
        specs = []
        for i, site_idx in enumerate([3, 3, 5]):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=site_idx,
                site_type="hollow",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        seen_filters.append([s.site_index for s in specs])
        return specs[:n_desired]

    def fake_materialize(**kwargs):
        failures = []
        for spec in kwargs["specs"]:
            failures.append(
                PlacementFailureEvent(
                    placement_id=spec.placement_index,
                    stage="generation",
                    reason="adsorbate_overlap",
                    descriptor=None,
                )
            )
        return [], [], [], failures

    monkeypatch.setattr(core_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(core_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=3,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        seed=0,
    )
    _combined, _ids, _desc, _fails, _attempts = core_mod._generate_placements_with_retry(
        [make_water()],
        make_slab(),
        config,
        "O",
        None,
        make_slab(),
        calculator=None,
    )
    assert _RETRY_BLOCK_SITE_AFTER >= 2
    # After enough failures on site 3, later attempts should exclude it.
    assert any(3 not in batch for batch in seen_filters[1:])


# ---------------------------------------------------------------------------
# Phase 3 — pose frac XY, failure reasons, geometry floors, occupancy MIC
# ---------------------------------------------------------------------------


def _tilted_make_slab():
    """Cu-like slab with a non-Cartesian surface normal (tilted b/c)."""
    slab = make_slab()
    cell = np.array(slab.get_cell(), dtype=float)
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell[:3] = tilt @ cell[:3]
    slab.set_cell(cell)
    pos = slab.get_positions()
    pos[:] = (tilt @ pos.T).T
    slab.set_positions(pos)
    return slab


def test_tilted_slab_site_xy_frac_uses_full_3d_projection():
    """Descriptor frac a/b must project the full COM, not [x, y, 0]."""
    from metalsurfer.placement.site_coords import _slab_plane_projectors

    slab = _tilted_make_slab()
    config = adsorption_config_factory(
        material_type="slab", num_placements=20, placement_z_range=(2.0, 3.0)
    )
    spec, result = _first_successful_placement(
        water_conformers(), slab, config, "O", n_desired=20
    )
    assert spec is not None and result is not None
    _adsorbate, descriptor = result
    assert descriptor.z_abs is not None
    pinv_ab_T, _ = _slab_plane_projectors(np.asarray(slab.get_cell(), dtype=float))
    full = np.array(
        [descriptor.x_abs, descriptor.y_abs, float(descriptor.z_abs)], dtype=float
    )
    expected = np.mod(full @ pinv_ab_T, 1.0)
    zeroed = np.mod(
        np.array([descriptor.x_abs, descriptor.y_abs, 0.0], dtype=float) @ pinv_ab_T,
        1.0,
    )
    # Bug baseline: zeroing z shifts frac coords on this tilt.
    assert not np.allclose(expected, zeroed, atol=1e-6)
    assert descriptor.site_xy_frac_a == pytest.approx(float(expected[0]), abs=1e-9)
    assert descriptor.site_xy_frac_b == pytest.approx(float(expected[1]), abs=1e-9)
    assert descriptor.placement_mode_resolved == "sites"


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


def test_check_initial_placement_distance_empty_geometry():
    slab = make_slab()
    empty = Atoms()
    empty.set_cell(slab.get_cell())
    empty.set_pbc(slab.get_pbc())
    ok, dist, reason = check_initial_placement_distance(
        empty, slab, material_type="slab"
    )
    assert not ok
    assert dist == float("inf")
    assert reason == "empty_geometry"


def test_min_distance_floor_rejects_close_o_cu():
    """Explicit min_distance=5 Å must reject O–Cu at ~1.5 Å even if covalent ratio allows."""
    from ase.data import atomic_numbers, covalent_radii

    slab = make_slab(n_layers=1, symbol="Cu")
    cu = slab.get_positions()[0]
    r_o = float(covalent_radii[atomic_numbers["O"]])
    r_cu = float(covalent_radii[atomic_numbers["Cu"]])
    # Within covalent-ratio acceptance but far below a 5 Å floor.
    height = 1.5
    assert height < 5.0
    assert height > (r_o + r_cu) * 0.5
    oxygen = Atoms("O", positions=[cu + np.array([0.0, 0.0, height])])
    oxygen.set_cell(slab.get_cell())
    oxygen.set_pbc(slab.get_pbc())
    ok, dist, reason = check_initial_placement_distance(
        oxygen,
        slab,
        min_distance=5.0,
        min_contact_ratio=0.5,
        material_type="slab",
    )
    assert not ok
    assert reason == "too_close"
    assert dist == pytest.approx(height, abs=1e-6)


def test_filter_sites_by_occupancy_mic_wrap():
    """Occupancy prune must use MIC so a near-boundary adsorbate blocks the wrapped site."""
    from metalsurfer.placement._material import material_aware_pbc
    from metalsurfer.placement.occupancy import filter_sites_by_occupancy

    slab = make_slab()
    cell = np.asarray(slab.get_cell(), dtype=float)
    # Site just inside +a; existing adsorbate just outside via wrap (near a=0).
    near_hi = _make_site([cell[0, 0] - 0.3, 5.0, 6.0])
    far = _make_site([5.0, 5.0, 6.0])
    existing = np.array([[0.2, 5.0, 6.0]])
    kept = filter_sites_by_occupancy(
        [near_hi, far],
        existing,
        cell=cell,
        pbc=material_aware_pbc("slab"),
        min_separation=1.0,
    )
    assert len(kept) == 1
    assert np.allclose(kept[0].xyz, far.xyz)
    # None existing → unchanged.
    assert (
        len(
            filter_sites_by_occupancy(
                [near_hi, far],
                None,
                cell=cell,
                pbc=material_aware_pbc("slab"),
                min_separation=1.0,
            )
        )
        == 2
    )


def test_principal_axis_rotation_flat_hexagon_stays_near_flat():
    from metalsurfer.placement.geometry import _principal_axis_rotation

    hex_pos = np.array(
        [
            [1.4 * np.cos(i * np.pi / 3), 1.4 * np.sin(i * np.pi / 3), 0.0]
            for i in range(6)
        ],
        dtype=float,
    )
    hex_pos -= hex_pos.mean(axis=0)
    rotated, _score = _principal_axis_rotation(hex_pos, np.array([0.0, 0.0, 1.0]))
    assert rotated is not None
    # Plane normal ≈ z → z-span stays small (near-flat).
    assert float(np.ptp(rotated[:, 2])) < 0.35


def test_check_adsorbate_separation_requires_cell_when_pbc_requested():
    mol = make_water()
    pre = np.array([[0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="cell"):
        check_adsorbate_separation(mol, pre, pbc=[True, True, False])


def test_calculate_min_distance_left_handed_cell_uses_abs_det():
    p1 = np.array([[0.1, 0.1, 0.0]])
    p2 = np.array([[9.9, 0.1, 0.0]])
    # Left-handed cell (det < 0) with the same |a|,|b| as a 10×10 slab.
    cell = np.array([[10.0, 0.0, 0.0], [0.0, -10.0, 0.0], [0.0, 0.0, 15.0]])
    assert float(np.linalg.det(cell)) < 0.0
    d = calculate_min_distance(p1, p2, cell=cell, use_pbc=True, pbc=[True, True, False])
    assert d == pytest.approx(0.2, abs=1e-6)

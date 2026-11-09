"""Batch placement-spec builder policies."""

import itertools
import math

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.models import PlacementSpec
from metalsurfer.placement import (
    enumerate_placement_specs,
    generate_placement_from_spec,
)
from metalsurfer.placement import (
    geometry as placement_geometry,
)
from metalsurfer.placement._constants import (
    _AZIMUTH,
    _AZIMUTH_IN_PLANE,
    _EARLY_CAP_WORKING_SET_MULTIPLIER,
    _POROUS_SITE_INDEX_WEIGHT,
    _SITE_NEIGHBOR_ABS_TOL_ANGSTROM,
    _TILT_FULL,
    _WALL_Z_FRACTIONS,
    _Z_FRACTIONS,
)
from metalsurfer.placement.policy import (
    _TILT_PARALLEL,
    _flat_aromatic_parallel_total,
    _parallel_aip_values,
    _stratified_sample,
    _unravel_parallel_flat,
    build_batch_placement_specs,
    max_batch_placement_specs,
)

from ..conftest import (
    make_slab,
)
from ._helpers import (
    TEST_SEED,
    _site_type_atop,
)


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
        n_binders=n_binders,
        flat_aromatic=flat_aromatic,
        site_type_for_index=_site_type_atop,
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


def test_build_batch_specs_tilt_zero_drops_redundant_azimuth_in_plane():
    """At tilt=0, azimuth_in_plane is fixed to 0 (azimuth already covers in-plane)."""
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=1,
        flat_aromatic=True,
        parallel_fraction=1.0,
        n_desired=10**7,
        seed=TEST_SEED,
    )
    tilt0 = [s for s in specs if s.orientation_type == "parallel" and s.tilt_deg == 0.0]
    assert tilt0
    assert all(s.azimuth_in_plane_deg == 0.0 for s in tilt0)
    # Nonzero tilt still varies azimuth_in_plane.
    tilted = [
        s for s in specs if s.orientation_type == "parallel" and s.tilt_deg != 0.0
    ]
    assert tilted
    assert {s.azimuth_in_plane_deg for s in tilted} > {0.0}


def test_tilt_zero_azimuth_pairs_collapse_to_eight_geometries():
    """At tilt=0, az × aip commute about the normal → 8 unique of 32 pairs."""
    pos = np.array(
        [
            [1.3, 0.0, 0.0],
            [0.5, 0.866, 0.0],
            [-0.5, 0.866, 0.0],
            [-1.0, 0.0, 0.0],
            [-0.5, -0.866, 0.0],
            [0.5, -0.866, 0.0],
        ],
        dtype=float,
    )
    normal = np.array([0.0, 0.0, 1.0])
    unique: set[bytes] = set()
    for az in _AZIMUTH:
        for aip in _AZIMUTH_IN_PLANE:
            base, _R_base = placement_geometry._flat_orientation_from_principal_axis(
                pos, normal, azimuth_in_plane_deg=aip
            )
            rotated, _R_tilt = placement_geometry._rotation_with_tilt(
                base, normal, 0.0, az
            )
            rounded = np.round(rotated, decimals=6)
            # Canonicalize +0.0/-0.0 so tobytes() uniqueness matches geometry uniqueness.
            rounded[rounded == 0.0] = 0.0
            unique.add(rounded.tobytes())
    assert len(unique) == len(_AZIMUTH)
    assert len(unique) < len(_AZIMUTH) * len(_AZIMUTH_IN_PLANE)


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
        n_binders=0,
        flat_aromatic=False,
        dissociative=True,
        n_hollow_pairs=n_hollow_pairs,
    )
    assert count == n_hollow_pairs * len(_WALL_Z_FRACTIONS)


def test_short_draw_cycles_wall_height_shells():
    """A short batch includes contact, halfway, and van der Waals starts."""
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=3,
        seed=TEST_SEED,
    )
    assert [float(spec.z_fraction) for spec in specs[:3]] == list(_WALL_Z_FRACTIONS)


def test_wall_near_grid_collapses_contact_z_fractions():
    """Wall-near z < 0.5 repeats the contact COM; pores keep every fraction."""
    wall = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=10**6,
        seed=TEST_SEED,
    )
    wall_z = {float(s.z_fraction) for s in wall}
    assert wall_z
    assert wall_z <= {0.5, 0.7, 0.9}
    assert 0.5 in wall_z

    def _pore(_index: int) -> str:
        return "pore"

    pores = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0],
        site_type_for_index=_pore,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=10**6,
        seed=TEST_SEED,
    )
    pore_z = {float(s.z_fraction) for s in pores}
    assert pore_z == set(_Z_FRACTIONS)


def test_mixed_catalog_keeps_each_sites_height_axis():
    """A void site keeps every shell; a wall site keeps contact and above."""

    def _site_type(index: int) -> str:
        return "pore" if index == 0 else "atop"

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1],
        site_type_for_index=_site_type,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=10**6,
        seed=TEST_SEED,
        site_is_void_for_index=lambda index: index == 0,
    )
    by_site: dict[int, set[float]] = {}
    for spec in specs:
        by_site.setdefault(int(spec.site_index), set()).add(float(spec.z_fraction))
    assert by_site[0] == set(_Z_FRACTIONS)
    assert by_site[1] == set(_WALL_Z_FRACTIONS)


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


def test_occupancy_counts_are_equal_and_keep_site_type():
    """Counts alternate. A high count does not lead, and site_type stays natural."""

    def _bin(site_idx: int) -> str:
        return "2" if int(site_idx) in {1, 3} else "0"

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1, 2, 3],
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=8,
        seed=TEST_SEED,
        site_bin_for_index=_bin,
    )
    assert len(specs) == 8
    assert {spec.site_type for spec in specs} == {"atop"}
    counts = [_bin(spec.site_index) for spec in specs]
    assert counts == ["0", "2"] * 4


def test_same_count_does_not_hide_atops_behind_bridges():
    """Bridges and vacant atops that share a count both appear in a short draw."""

    def _type(site_idx: int) -> str:
        return "bridge" if int(site_idx) < 3 else "atop"

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1, 2, 3, 4],
        site_type_for_index=_type,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=8,
        seed=TEST_SEED,
        site_bin_for_index=lambda _site: "1",
    )
    assert len(specs) == 8
    assert "atop" in {spec.site_type for spec in specs[:4]}
    assert "bridge" in {spec.site_type for spec in specs}


def test_many_counts_still_open_on_a_contact_pose():
    """A draw the size of the count list is one contact pose per count.

    Each count holds every binding mode. The opening type rotates, so the
    draw is not one mode repeated.
    """
    modes = ("atop", "bridge", "hollow")
    n_counts = len(modes)
    per_count = len(modes)

    def _type(site_idx: int) -> str:
        return modes[int(site_idx) % len(modes)]

    def _bin(site_idx: int) -> str:
        return str(int(site_idx) // per_count)

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=list(range(n_counts * per_count)),
        site_type_for_index=_type,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=n_counts,
        seed=TEST_SEED,
        site_bin_for_index=_bin,
    )
    assert len(specs) == n_counts
    assert {_bin(spec.site_index) for spec in specs} == {
        str(i) for i in range(n_counts)
    }
    assert {spec.site_type for spec in specs} == set(modes)
    assert {float(spec.z_fraction) for spec in specs} == {float(_WALL_Z_FRACTIONS[0])}
    assert {float(spec.tilt_deg) for spec in specs} == {0.0}


def _coverage_type(site_idx: int) -> str:
    return ("atop", "hollow", "bridge")[int(site_idx)]


def _coverage_clearance(site_idx: int) -> float:
    """Hollow is strictly clearer than atop. The bridge clearance is not finite."""
    window = float(_SITE_NEIGHBOR_ABS_TOL_ANGSTROM)
    if int(site_idx) == 1:
        return window + window
    if int(site_idx) == 2:
        return float("inf")
    return 0.0


@pytest.mark.parametrize(
    ("parallel_fraction", "shape", "flat_aromatic", "n_binders"),
    [(0.0, "round", False, 1), (1.0, "flat", True, 2)],
)
def test_same_count_reserves_each_finite_site_type(
    parallel_fraction: float,
    shape: str,
    flat_aromatic: bool,
    n_binders: int,
):
    """One count's contact shell keeps every finite binding mode.

    The draw is the wall shells of the opening type, then the other finite
    contact. A non-finite clearance is not a substrate site.
    """
    contact = float(_WALL_Z_FRACTIONS[0])
    n_desired = len(_WALL_Z_FRACTIONS) + 1
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1, 2],
        site_type_for_index=_coverage_type,
        shape=shape,
        n_binders=n_binders,
        flat_aromatic=flat_aromatic,
        parallel_fraction=parallel_fraction,
        n_desired=n_desired,
        seed=TEST_SEED,
        site_bin_for_index=lambda _site: "0",
        site_clearance_for_index=_coverage_clearance,
    )
    assert len(specs) == n_desired
    assert specs[0].site_type == "atop"
    assert float(specs[0].z_fraction) == contact
    assert float(specs[0].tilt_deg) == 0.0
    contacts = {
        spec.site_type
        for spec in specs
        if float(spec.z_fraction) == contact and float(spec.tilt_deg) == 0.0
    }
    assert contacts == {"atop", "hollow"}
    assert all(spec.site_index != 2 for spec in specs)


def test_en_down_coverage_uses_the_clearer_site():
    """Flat aromatic EN-down keeps the clearer site of an occupancy count."""

    def _bin(site_idx: int) -> str:
        return "0" if int(site_idx) == 1 else "3"

    def _clearance(site_idx: int) -> float:
        return 4.0 if int(site_idx) == 1 else 0.2

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=[0, 1],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.0,
        n_desired=4,
        seed=TEST_SEED,
        site_bin_for_index=_bin,
        site_clearance_for_index=_clearance,
    )
    en_down = [spec for spec in specs if spec.orientation_type == "EN-down"]
    assert en_down
    assert any(spec.site_index == 1 for spec in en_down)
    clearer = [spec for spec in en_down if spec.site_index == 1]
    assert float(clearer[0].z_fraction) == 0.5
    assert float(clearer[0].tilt_deg) == 0.0


def test_rare_occupancy_count_survives_large_catalog_cap():
    """A single positive-count site stays in the draw without occupying the front."""

    def _bin(site_idx: int) -> str:
        return "1" if int(site_idx) == 777 else "0"

    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=list(range(900)),
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=40,
        seed=TEST_SEED,
        site_bin_for_index=_bin,
    )
    assert len(specs) == 40
    assert any(int(spec.site_index) == 777 for spec in specs)
    assert specs[0].site_index != 777


def test_build_batch_specs_flat_aromatic_large_grid_capped():
    n_desired = 20
    site_indices = list(range(50))
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=site_indices,
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=n_desired,
        seed=TEST_SEED,
    )
    assert len(specs) == n_desired
    n_par = sum(1 for s in specs if s.orientation_type == "parallel")
    n_en = sum(1 for s in specs if s.orientation_type == "EN-down")
    assert n_par == 10
    assert n_en == 10
    assert len({s.placement_index for s in specs}) == n_desired


def test_build_batch_specs_tops_up_when_one_branch_filtered():
    """If filter empties one orientation branch, fill n_desired from the other."""
    n_desired = 20
    specs = build_batch_placement_specs(
        n_conformers=1,
        site_indices=list(range(50)),
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=n_desired,
        seed=TEST_SEED,
        filter_spec=lambda s: s.orientation_type != "parallel",
    )
    assert len(specs) == n_desired
    assert {s.orientation_type for s in specs} == {"EN-down"}


def test_flat_aromatic_binder_count_scales_en_down_capacity():
    """Charged atoms raise n_binders, so the EN-down capacity grows."""
    n_conformers, n_sites = 1, 2
    base = max_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=list(range(n_sites)),
        n_binders=1,
        flat_aromatic=True,
        site_type_for_index=_site_type_atop,
    )
    with_charge = max_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=list(range(n_sites)),
        n_binders=2,
        flat_aromatic=True,
        site_type_for_index=_site_type_atop,
    )
    assert with_charge > base
    # The EN-down axis grows linearly with the binder count while the
    # parallel axis is binder-independent: the delta is one binder block.
    z_total = n_sites * 3  # wall sites keep z_fraction >= 0.5 (3 of 5 values)
    en_delta = n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * z_total
    assert with_charge - base == en_delta


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


def test_cco_generation_yield_meets_seeded_bar():
    """Flexible ethanol should clear the seeded 90% generation bar."""

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


def test_build_batch_specs_else_branch_early_caps_without_prefix_bias():
    """Non-flat path must not materialize the full Cartesian grid before subsample."""
    n_sites = 1000
    n_conformers = 3
    n_desired = 20
    site_indices = list(range(n_sites))
    full_grid = (
        n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * len(_Z_FRACTIONS) * n_sites
    )
    assert full_grid > 100_000

    specs = build_batch_placement_specs(
        n_conformers=n_conformers,
        site_indices=site_indices,
        site_type_for_index=_site_type_atop,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.5,
        n_desired=n_desired,
        seed=TEST_SEED,
    )
    assert len(specs) == n_desired

    # Uniform sample of the working set must not collapse to the product prefix
    # (tilt=0, az=0, z=first, sites 0..n).
    az_vals = {s.azimuth_deg for s in specs}
    tilt_vals = {s.tilt_deg for s in specs}
    conf_vals = {s.conformer_index for s in specs}
    site_vals = {s.site_index for s in specs}
    assert len(az_vals) > 1
    assert len(tilt_vals) > 1
    assert len(conf_vals) >= 2
    # Sites should not be only the first working_cap / n_pose contiguous block.
    assert max(site_vals) > n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER


def test_single_site_type_short_draw_follows_phased_diagonal():
    """One site type: first height cycle is the mild-tilt diagonal of shell ``p``."""
    conformers = (0, 1, 2, 3)
    sites = (0, 1, 2, 3)
    pool: list[PlacementSpec] = []
    for zf, tilt, ci, az, site in itertools.product(
        _Z_FRACTIONS, (0.0, 45.0), conformers, _AZIMUTH, sites
    ):
        pool.append(
            PlacementSpec(
                conformer_index=ci,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=site,
                site_type=None,
                tilt_deg=tilt,
                azimuth_deg=az,
                azimuth_in_plane_deg=0.0,
                z_fraction=zf,
                placement_index=0,
            )
        )

    selected = _stratified_sample(pool, n_desired=10, seed=0)
    assert len(selected) == 10
    n_z = len(_Z_FRACTIONS)
    first, second = selected[:n_z], selected[n_z:]
    assert [s.z_fraction for s in first] == list(_Z_FRACTIONS)
    assert all(s.tilt_deg == 0.0 for s in first)
    for phase, spec in enumerate(first):
        assert spec.conformer_index == conformers[phase % len(conformers)]
        assert spec.azimuth_deg == pytest.approx(_AZIMUTH[phase % len(_AZIMUTH)])
        assert spec.site_index == sites[phase % len(sites)]
    first_keys = {(s.conformer_index, s.azimuth_deg, s.site_index) for s in first}
    second_keys = {(s.conformer_index, s.azimuth_deg, s.site_index) for s in second}
    assert first_keys.isdisjoint(second_keys)
    assert all(s.tilt_deg == 0.0 for s in second)
    contact_z = _Z_FRACTIONS[0]
    contact = [spec for spec in selected if spec.z_fraction == contact_z]
    assert len(contact) >= 2
    assert contact[1].site_index != contact[0].site_index
    assert contact[1].conformer_index == contact[0].conformer_index


def _draw_covering_grid(*, n_conformers: int, n_sites: int, n_z: int) -> int:
    """Smallest draw whose working set is the full kept-height product."""
    n_total = n_conformers * len(_TILT_FULL) * len(_AZIMUTH) * n_z * n_sites
    return (
        n_total + _EARLY_CAP_WORKING_SET_MULTIPLIER - 1
    ) // _EARLY_CAP_WORKING_SET_MULTIPLIER


def _span_draw(
    site_indices: list[int], site_type, **overrides: object
) -> list[PlacementSpec]:
    n_z = (
        len(_Z_FRACTIONS)
        if overrides.get("site_is_void_for_index")
        else len(_WALL_Z_FRACTIONS)
    )
    return build_batch_placement_specs(
        n_conformers=1,
        site_indices=site_indices,
        site_type_for_index=site_type,
        shape="round",
        n_binders=1,
        flat_aromatic=False,
        parallel_fraction=0.0,
        n_desired=_draw_covering_grid(
            n_conformers=1, n_sites=len(site_indices), n_z=n_z
        ),
        seed=TEST_SEED,
        **overrides,
    )


def test_wall_short_draw_follows_span_order():
    """Wall heights open on the spanned sites, not the lowest numeric ids."""
    site_indices = [10, 0, 3]
    prefix = _span_draw(site_indices, _site_type_atop)[: len(_WALL_Z_FRACTIONS)]
    assert [s.site_index for s in prefix] == site_indices
    assert [float(s.z_fraction) for s in prefix] == list(_WALL_Z_FRACTIONS)
    assert all(s.tilt_deg == 0.0 for s in prefix)


def test_mixed_site_types_open_on_span_first_site():
    """Each site type leads with its span-first site before a later numeric id."""
    specs = _span_draw([10, 0, 7], lambda idx: "bridge" if idx == 7 else "hollow")
    hollow = [s.site_index for s in specs if s.site_type == "hollow"]
    bridge = [s.site_index for s in specs if s.site_type == "bridge"]
    assert hollow[:2] == [10, 0]
    assert bridge[0] == 7


def test_porous_short_draw_follows_span_order():
    """Pore round-robin leads with the spanned site, not the lowest id."""
    site_indices = [10, 0, 3]
    specs = _span_draw(
        site_indices,
        lambda _idx: "pore",
        site_index_weight=_POROUS_SITE_INDEX_WEIGHT,
        site_is_void_for_index=lambda _idx: True,
    )
    n_z = len(_Z_FRACTIONS)
    assert [s.site_index for s in specs[:n_z]] == [site_indices[0]] * n_z
    assert specs[n_z].site_index == site_indices[1]


def test_flat_aromatic_large_grid_samples_beyond_prefix():
    """A flat-aromatic cap must not be conformer 0 and the first site indices."""
    n_desired = 12
    specs = build_batch_placement_specs(
        n_conformers=6,
        site_indices=list(range(400)),
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=n_desired,
        seed=TEST_SEED,
    )
    assert len(specs) == n_desired
    assert len({s.conformer_index for s in specs}) > 1
    assert (
        max(s.site_index for s in specs) > n_desired * _EARLY_CAP_WORKING_SET_MULTIPLIER
    )


def test_policy_prior_prefers_mild_tilt_and_mid_z():
    """When subsampling, prefer milder tilts / mid z vs a uniform shuffle of the same pool."""
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
    assert mean_tilt < 40.0, f"mean tilt {mean_tilt:.4f} is not below the 40° grid mean"
    assert {round(s.z_fraction, 1) for s in selected} == {0.1, 0.3, 0.5, 0.7, 0.9}


def test_flat_aromatic_indexed_sampling_matches_seed_without_filter():
    """Unfiltered flat-aromatic specs stay seed-identical under indexed unravel."""
    specs_a = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0, 1, 2],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=24,
        filter_spec=None,
        seed=TEST_SEED,
    )
    specs_b = build_batch_placement_specs(
        n_conformers=2,
        site_indices=[0, 1, 2],
        site_type_for_index=_site_type_atop,
        shape="flat",
        n_binders=2,
        flat_aromatic=True,
        parallel_fraction=0.5,
        n_desired=24,
        filter_spec=None,
        seed=TEST_SEED,
    )
    assert len(specs_a) == len(specs_b) == 24

    def _key(s: PlacementSpec) -> tuple:
        return (
            s.conformer_index,
            s.orientation_type,
            s.face_flip,
            s.en_atom_index,
            s.site_index,
            s.tilt_deg,
            s.azimuth_deg,
            s.azimuth_in_plane_deg,
            s.z_fraction,
        )

    assert [_key(s) for s in specs_a] == [_key(s) for s in specs_b]


def test_parallel_flat_unravel_matches_product_order():
    """Indexed parallel decode matches itertools.product × aip nesting."""
    n_conf = 2
    # Span order: wall shells on site 0, then the void shells on site 3.
    pairs = [
        (0, 0.5),
        (0, 0.7),
        (0, 0.9),
        (3, 0.1),
        (3, 0.3),
        (3, 0.5),
        (3, 0.7),
        (3, 0.9),
    ]

    def product_unweighted():
        for ci, ff, tl, azv, (si, zfv) in itertools.product(
            range(n_conf),
            [False, True],
            _TILT_PARALLEL,
            _AZIMUTH,
            pairs,
        ):
            for aip in _parallel_aip_values(tl):
                yield (
                    ci,
                    ff,
                    float(tl),
                    float(azv),
                    float(zfv),
                    float(aip),
                    si,
                )

    def product_weighted():
        for ff, tl, azv, (si, zfv), ci in itertools.product(
            [False, True],
            _TILT_PARALLEL,
            _AZIMUTH,
            pairs,
            range(n_conf),
        ):
            for aip in _parallel_aip_values(tl):
                yield (
                    ci,
                    ff,
                    float(tl),
                    float(azv),
                    float(zfv),
                    float(aip),
                    si,
                )

    n = _flat_aromatic_parallel_total(n_conformers=n_conf, n_site_z=len(pairs))
    unweighted = list(product_unweighted())
    weighted = list(product_weighted())
    assert len(unweighted) == n == len(weighted)
    for flat, expected in enumerate(unweighted):
        assert (
            _unravel_parallel_flat(
                flat,
                n_conformers=n_conf,
                site_z_pairs=pairs,
                conformer_fastest=False,
            )
            == expected
        )
    for flat, expected in enumerate(weighted):
        assert (
            _unravel_parallel_flat(
                flat,
                n_conformers=n_conf,
                site_z_pairs=pairs,
                conformer_fastest=True,
            )
            == expected
        )

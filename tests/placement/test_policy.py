"""Batch placement-spec builder policies."""

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
from metalsurfer.placement._constants import _Z_FRACTIONS
from metalsurfer.placement.policy import (
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
    assert count == n_hollow_pairs * len(_Z_FRACTIONS)

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
    """Flexible ethanol should clear the seeded ≥40% generation bar (target ≥60%)."""

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


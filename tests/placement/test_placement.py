"""Cross-material orchestration of the universal placement workflow."""

from collections import Counter

import numpy as np
import pytest

from metalsurfer.filters import check_desorption
from metalsurfer.placement import (
    calculate_min_distance,
    get_unified_sites,
)

from ..conftest import (
    adsorption_config_factory,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    make_water,
    water_conformers,
)
from ._helpers import (
    _GOLDEN_SLAB_SITE_TYPE_MULTISET,
    _GOLDEN_SLAB_UNIFIED_SITE_COUNT,
    _assert_replay_matches,
    _first_successful_placement,
)


def test_unified_sites_slab_golden_count_and_type_multiset():
    """Stable site catalog for make_slab() under default classification."""
    slab = make_slab()
    sites = get_unified_sites(slab, material_type="slab")
    assert len(sites) == _GOLDEN_SLAB_UNIFIED_SITE_COUNT
    assert dict(Counter(s.site_type for s in sites)) == _GOLDEN_SLAB_SITE_TYPE_MULTISET


@pytest.mark.parametrize("mode", ["spec", "pose"])
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


def test_check_desorption_nanoparticle_and_porous():
    nanoparticle = make_nanoparticle()
    porous = make_porous_framework()

    water_far = make_water()
    water_far.set_positions(water_far.get_positions() + np.array([20.0, 20.0, 20.0]))
    np_combined = nanoparticle + water_far
    np_combined.set_cell(nanoparticle.get_cell())
    np_combined.set_pbc(nanoparticle.get_pbc())
    ok_np_far, reason_np_far = check_desorption(
        np_combined, nanoparticle, binding_threshold=4.0, material_type="nanoparticle"
    )
    assert not ok_np_far
    assert "too far" in reason_np_far
    dist_np_far = calculate_min_distance(
        water_far.get_positions(),
        nanoparticle.get_positions(),
        nanoparticle.get_cell(),
        use_pbc=True,
        pbc=list(nanoparticle.get_pbc()),
    )
    assert float(dist_np_far) > 4.0

    np_sites = get_unified_sites(nanoparticle, material_type="nanoparticle")
    assert np_sites
    water_near_np = make_water()
    n_hat = np.asarray(np_sites[0].normal, dtype=float)
    n_hat = n_hat / float(np.linalg.norm(n_hat))
    center_np = np.asarray(np_sites[0].xyz, dtype=float) + 1.5 * n_hat
    wpos = water_near_np.get_positions().copy()
    wpos -= np.mean(wpos, axis=0)
    wpos += center_np
    water_near_np.set_positions(wpos)
    np_near_combined = nanoparticle + water_near_np
    np_near_combined.set_cell(nanoparticle.get_cell())
    np_near_combined.set_pbc(nanoparticle.get_pbc())
    ok_np_near, reason_np_near = check_desorption(
        np_near_combined,
        nanoparticle,
        binding_threshold=4.0,
        material_type="nanoparticle",
    )
    assert ok_np_near
    assert "adsorbed" in reason_np_near
    dist_np_near = calculate_min_distance(
        water_near_np.get_positions(),
        nanoparticle.get_positions(),
        nanoparticle.get_cell(),
        use_pbc=True,
        pbc=list(nanoparticle.get_pbc()),
    )
    assert float(dist_np_near) <= 4.0

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
    ok_porous_near, reason_porous_near = check_desorption(
        porous_combined, porous, binding_threshold=4.0, material_type="porous"
    )
    assert ok_porous_near
    assert "adsorbed" in reason_porous_near

    water_far_porous = make_water()
    # Dense 3D PBC: Cartesian translation wraps; sample free volume for a true far pose.
    cell = np.asarray(porous.get_cell(), dtype=float)
    rng = np.random.default_rng(0)
    best_d = -1.0
    best_com = None
    for _ in range(800):
        com = rng.random(3) @ cell
        wpos = water_far_porous.get_positions().copy()
        wpos -= np.mean(wpos, axis=0)
        wpos += com
        d = calculate_min_distance(
            wpos,
            porous.get_positions(),
            cell,
            use_pbc=True,
            pbc=[True, True, True],
        )
        if float(d) > best_d:
            best_d = float(d)
            best_com = com
    assert best_com is not None and best_d > 3.5, (
        f"porous fixture should expose a void beyond desorption threshold, got {best_d:.3f}"
    )
    wpos = water_far_porous.get_positions().copy()
    wpos -= np.mean(wpos, axis=0)
    wpos += best_com
    water_far_porous.set_positions(wpos)
    porous_far_combined = porous + water_far_porous
    porous_far_combined.set_cell(porous.get_cell())
    porous_far_combined.set_pbc(porous.get_pbc())
    # Fixture max water clearance is ~3.8 Å; use a threshold below that so the
    # far pose is still classified as desorbed.
    ok_porous_far, reason_porous_far = check_desorption(
        porous_far_combined, porous, binding_threshold=3.5, material_type="porous"
    )
    assert not ok_porous_far
    assert "too far" in reason_porous_far
    assert best_d > 3.5

"""Dissociative (fragment) placement pathways."""

import numpy as np
import pytest
from scipy.spatial import KDTree

from metalsurfer.config import AdsorptionConfig
from metalsurfer.ml.features import FEATURE_NAMES, extract_features
from metalsurfer.ml.schema import PlacementRecord
from metalsurfer.placement import (
    check_initial_placement_distance,
    enumerate_placement_specs,
    generate_placement_from_spec_with_reason,
    get_hollow_sites_for_adatoms,
    get_unified_sites,
)
from metalsurfer.placement.dissociative import (
    _dissociative_pair_cache_key,
    _get_dissociative_site_pairs,
)
from metalsurfer.placement.site_enumeration import (
    _compute_site_z_base,
)
from metalsurfer.placement.site_types import Site

from ..conftest import (
    make_h2,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
)
from ._helpers import dissoc_placement_spec


def test_dissociative_z_offset_uses_radius_derived_range():
    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        skip_topology_check=True,
        placement_z_range=(1.0, 1.5),
    )
    h2 = make_h2()
    spec = dissoc_placement_spec()
    result, reason = generate_placement_from_spec_with_reason(
        spec, [h2], slab, config, smiles="[H][H]"
    )
    assert result is not None, f"fixture slab must place dissociative H2: {reason}"
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


def test_hollow_site_pairs_found_for_slab():
    """_get_dissociative_site_pairs must find adjacent hollow pairs within adaptive bounds."""
    from ase.geometry import find_mic

    slab = make_slab()
    config = AdsorptionConfig()
    pairs = _get_dissociative_site_pairs(slab, config)
    assert len(pairs) >= 8, (
        f"4×4 FCC-like slab should yield many hollow-site pairs, got {len(pairs)}"
    )
    cell = np.asarray(slab.get_cell(), dtype=float)
    # The dissociative pairs connect adjacent hollow sites, so their separation
    # should be comparable to the hollow-site spacing of the surface (the
    # physically relevant length scale). The slab's flat top layer is a periodic
    # lattice, so we derive this spacing from the slab's own hollow sites via a
    # KDTree rather than hard-coding a constant.
    hollows = get_hollow_sites_for_adatoms(slab, material_type="slab")
    assert hollows, "slab must expose hollow sites for the NN reference"
    hpos = np.array([h.xyz for h in hollows], dtype=float)
    h_d, _ = KDTree(hpos).query(hpos, k=2)
    d_NN = float(np.median(h_d[:, 1]))
    for p in pairs:
        assert len(p.xyz1) == 3
        assert len(p.xyz2) == 3
        _, dists = find_mic(
            (np.asarray(p.xyz1) - np.asarray(p.xyz2)).reshape(1, 3), cell
        )
        sep = float(dists[0])
        assert 0.5 * d_NN <= sep <= 1.5 * d_NN, (
            f"hollow-pair separation {sep:.3f} Å out of physical band "
            f"[0.5, 1.5]×d_NN={d_NN:.3f} Å"
        )


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

    for i, j in KDTree(sites).query_pairs(r=1.5):
        cart.add((min(i, j), max(i, j)))
    periodic = _periodic_site_pair_candidates(sites, cell, pbc, max_sep=1.5)
    assert (0, 1) in periodic  # wrap across a
    assert (0, 1) not in cart
    _, d = find_mic((sites[0] - sites[1]).reshape(1, 3), cell, pbc=pbc)
    assert float(d[0]) <= 1.5


def test_periodic_site_pair_distances_match_find_mic_on_wrap_cell():
    """Distances returned alongside the pairs must be the true MIC distances."""
    from ase.geometry import find_mic

    from metalsurfer.placement.dissociative import _periodic_site_pair_candidates

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
    max_sep = 1.5

    pairs = _periodic_site_pair_candidates(sites, cell, pbc, max_sep=max_sep)
    assert pairs, "wrap-around cell must yield MIC-adjacent pairs"
    for (i, j), d in pairs.items():
        _, dm = find_mic((sites[i] - sites[j]).reshape(1, 3), cell, pbc=pbc)
        assert abs(float(dm[0]) - d) < 1e-9, (
            f"pair ({i},{j}) distance {d:.12f} != find_mic {float(dm[0]):.12f}"
        )
        assert d <= max_sep + 1e-12

    # No MIC-adjacent pair may be dropped.
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            _, dm = find_mic((sites[i] - sites[j]).reshape(1, 3), cell, pbc=pbc)
            if float(dm[0]) <= max_sep:
                assert (i, j) in pairs


def test_mean_nn_separation_mic_matches_find_mic_reference():
    """Offsets + KDTree must reproduce find_mic, including the self-image trap."""
    from ase.build import fcc111
    from ase.geometry import find_mic

    from metalsurfer.placement.dissociative import _mean_nn_separation_mic

    def reference(points, cell, pbc):
        nn = []
        for i in range(len(points)):
            _, dists = find_mic(points - points[i], cell, pbc=pbc)
            dists = np.asarray(dists, dtype=float)
            dists[i] = np.inf
            nn.append(float(np.min(dists)))
        return float(np.mean(nn))

    pbc = [True, True, False]

    # Hexagonal fcc111 cell.
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    slab.set_pbc(pbc)
    hex_points = slab.get_positions()
    hex_cell = np.asarray(slab.get_cell(), dtype=float)
    assert (
        abs(
            _mean_nn_separation_mic(hex_points, hex_cell, pbc)
            - reference(hex_points, hex_cell, pbc)
        )
        < 1e-9
    )

    # Elongated orthorhombic cell: a naive k=2 query returns each site's own
    # image at |a| = 3 A, which is shorter than the true 10 A neighbour.
    elong_cell = np.diag([3.0, 20.0, 30.0])
    elong_points = np.array([[0.0, 0.0, 5.0], [0.0, 10.0, 5.0]], dtype=float)
    got = _mean_nn_separation_mic(elong_points, elong_cell, pbc)
    assert abs(got - reference(elong_points, elong_cell, pbc)) < 1e-9
    assert abs(got - 10.0) < 1e-9, f"self-image leaked into the mean NN: {got}"


def test_dissociative_placement_supported_for_nanoparticle():
    nanoparticle = make_nanoparticle()
    config = AdsorptionConfig(
        material_type="nanoparticle",
        skip_topology_check=True,
        enable_dissociative_placement=True,
        num_placements=1,
    )
    pairs = _get_dissociative_site_pairs(nanoparticle, config)
    assert pairs, "Au₁₃ fixture must expose dissociative site pairs"
    h2 = make_h2()
    spec = dissoc_placement_spec()

    result, reason = generate_placement_from_spec_with_reason(
        spec,
        [h2],
        nanoparticle,
        config,
    )
    assert result is not None, reason
    placed, descriptor = result
    assert descriptor.orientation_type == "dissociative"
    assert descriptor.site_source is not None
    assert "dissociative" in str(descriptor.site_source)
    assert descriptor.surface_ref_z_abs is not None
    hh = float(
        np.linalg.norm(
            placed.get_positions()[1] - placed.get_positions()[0],
        )
    )
    pair = pairs[descriptor.site_index % len(pairs)]
    pair_sep = float(np.linalg.norm(np.asarray(pair.xyz1) - np.asarray(pair.xyz2)))
    assert hh == pytest.approx(pair_sep, abs=1e-5), (
        f"H–H separation {hh:.3f} should track pair spacing {pair_sep:.3f}"
    )
    assert hh > 1.0, "Dissociative placement should separate H atoms"
    ok, min_d, dist_reason = check_initial_placement_distance(
        placed, nanoparticle, material_type="nanoparticle"
    )
    assert ok, (min_d, dist_reason)
    # Lower floor is gated by `assert ok`; only the slack upper tail is checked.
    assert min_d <= descriptor.z_offset + 0.8


def test_dissociative_wrap_pair_cartesian_separation_matches_mic():
    """Boundary-crossing pairs must place fragments at MIC images, not in-cell coords."""
    from ase.build import fcc111
    from ase.geometry import find_mic

    slab = fcc111("Pt", (3, 3, 3), vacuum=10.0)
    slab.set_pbc([True, True, False])
    config = AdsorptionConfig(material_type="slab", skip_topology_check=True)
    pairs = _get_dissociative_site_pairs(slab, config)
    assert pairs, "Pt 3x3 must expose dissociative hollow pairs"

    cell = np.asarray(slab.get_cell(), dtype=float)
    wrap_pairs = []
    for idx, pair in enumerate(pairs):
        xyz1 = np.asarray(pair.xyz1, dtype=float)
        xyz2 = np.asarray(pair.xyz2, dtype=float)
        cart = float(np.linalg.norm(xyz2 - xyz1))
        _, mic_d = find_mic((xyz2 - xyz1).reshape(1, 3), cell, pbc=[True, True, False])
        mic = float(mic_d[0])
        # Stored coords should already realize MIC in Cartesian space.
        assert cart == pytest.approx(mic, abs=1e-9), (
            f"pair {idx}: stored Cartesian sep {cart:.3f} != MIC {mic:.3f}"
        )
        a_len = float(np.linalg.norm(cell[0]))
        if mic < 0.6 * a_len:
            wrap_pairs.append((idx, pair, mic))

    assert wrap_pairs, "expected at least one MIC wrap pair on Pt 3x3"

    h2 = make_h2()
    for idx, pair, mic_sep in wrap_pairs[:3]:
        spec = dissoc_placement_spec(site_index=idx)
        result, reason = generate_placement_from_spec_with_reason(
            spec, [h2], slab, config
        )
        assert result is not None, reason
        placed, descriptor = result
        pos = placed.get_positions()
        cart_hh = float(np.linalg.norm(pos[1] - pos[0]))
        assert cart_hh == pytest.approx(mic_sep, abs=1e-5), (
            f"placed H–H Cartesian {cart_hh:.3f} should match MIC pair {mic_sep:.3f}, "
            "not the wrapped in-cell gap"
        )
        # Centroid must sit near the midpoint of the unfolded sites, not mid-cell void.
        mid = 0.5 * (np.asarray(pair.xyz1) + np.asarray(pair.xyz2))
        centroid = np.array(
            [descriptor.x_abs, descriptor.y_abs, descriptor.z_abs], dtype=float
        )
        assert float(np.linalg.norm(centroid[:2] - mid[:2])) < 0.5


def test_dissociative_placement_on_slab_separates_and_clears_surface():
    """Dissociative H2 on a slab must land on a hollow pair with physical clearance."""
    from ase.geometry import find_mic

    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", skip_topology_check=True)
    pairs = _get_dissociative_site_pairs(slab, config)
    assert pairs, "fixture slab must expose hollow pairs for dissociative placement"
    h2 = make_h2()
    spec = dissoc_placement_spec()
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
    assert hh == pytest.approx(pair_sep, abs=1e-5), (
        f"H–H separation {hh:.3f} should track hollow-pair spacing {pair_sep:.3f}"
    )
    assert hh > 1.0

    ok, min_d, dist_reason = check_initial_placement_distance(
        placed, slab, material_type="slab"
    )
    assert ok, (min_d, dist_reason)
    # Each H must sit above a distinct hollow: same in-plane column as the site,
    # at surface_ref + z_offset (Voronoi vertices may sit above the metal).
    from metalsurfer.placement.site_coords import _slab_normal

    site_a = np.asarray(pair.xyz1, dtype=float)
    site_b = np.asarray(pair.xyz2, dtype=float)
    n_hat = _slab_normal(cell)
    n_hat = n_hat / float(np.linalg.norm(n_hat))
    surface_ref = float(descriptor.surface_ref_z_abs)
    z_off = float(descriptor.z_offset)

    def _above_hollow(p: np.ndarray, site: np.ndarray) -> bool:
        d = p - site
        lateral = float(np.linalg.norm(d - np.dot(d, n_hat) * n_hat))
        if lateral > 0.08:
            return False
        height = float(np.dot(p, n_hat))
        return abs(height - (surface_ref + z_off)) < 1e-3

    assigned_distinct = (
        _above_hollow(pos[0], site_a) and _above_hollow(pos[1], site_b)
    ) or (_above_hollow(pos[0], site_b) and _above_hollow(pos[1], site_a))
    assert assigned_distinct, (
        "each H must map to a distinct hollow column at surface_ref + z_offset"
    )
    # Lower floor is gated by the contact gate; only the slack upper tail is checked.
    assert float(min_d) <= descriptor.z_offset + 0.8, (
        f"dissociative H–surface distance should be chemisorption-like, got {min_d:.3f}"
    )


def test_dissociative_placement_rejected_for_porous_material_type():
    porous = make_porous_framework()
    config = AdsorptionConfig(
        material_type="porous",
        skip_topology_check=True,
        num_placements=1,
    )
    h2 = make_h2()
    spec = dissoc_placement_spec()

    result, reason = generate_placement_from_spec_with_reason(
        spec,
        [h2],
        porous,
        config,
    )
    assert result is None
    assert reason == "dissociative_not_supported_for_porous"


def test_dissociative_fragment_positions_round_trip():
    """fragment_positions must match the placed dissociative geometry."""
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", enable_dissociative_placement=True)
    h2 = make_h2()
    spec = dissoc_placement_spec()
    result, reason = generate_placement_from_spec_with_reason(spec, [h2], slab, config)
    assert result is not None, reason
    placed, descriptor = result
    assert descriptor.fragment_positions is not None
    assert "fragment_positions" not in descriptor.to_row()
    assert "initial_fragment_positions" not in descriptor.to_row()
    rich_row = descriptor.to_row(include_provenance=True)
    assert rich_row.get("initial_fragment_positions") is not None
    assert np.allclose(
        np.asarray(descriptor.fragment_positions, dtype=float),
        placed.get_positions(),
        atol=1e-8,
    )


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


def test_dissociative_com_features_injective_and_record_replay():
    """Dissociative COM+quat features stay diverse; fragments survive record round-trip.

    Symmetry-equivalent hollow pairs share COM at a fixed height above the
    top layer, so the stratified pool need not be fully injective — require
    height diversity within a pair and a large unique-feature count overall.
    """
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
    by_pair: dict[int, list[tuple[float, tuple[float, ...]]]] = {}
    for spec in specs:
        result, _reason = generate_placement_from_spec_with_reason(
            spec, [h2], slab, config
        )
        if result is None:
            continue
        placed, descriptor = result
        assert descriptor.fragment_positions is not None
        record = PlacementRecord.from_descriptor(
            descriptor, molecule="H2", smiles="[H][H]"
        )
        assert record.descriptor.fragment_positions == descriptor.fragment_positions
        feats = extract_features(record)
        assert list(feats.keys()) == FEATURE_NAMES
        row = tuple(round(feats[name], 10) for name in FEATURE_NAMES)
        feature_rows.append(row)
        by_pair.setdefault(int(spec.site_index), []).append(
            (float(spec.z_fraction), row)
        )

        flat = record.to_flat_dict(include_provenance=True)
        assert flat.get("initial_fragment_positions") is not None
        restored = PlacementRecord.from_flat_dict(flat)
        assert (
            restored.descriptor.fragment_positions
            == record.descriptor.fragment_positions
        )
        replay_desc = restored.to_placement_descriptor()
        assert replay_desc.fragment_positions == descriptor.fragment_positions
        assert np.allclose(
            np.asarray(replay_desc.fragment_positions, dtype=float),
            placed.get_positions(),
            atol=1e-8,
        )

    assert len(feature_rows) >= 8
    assert len(set(feature_rows)) >= max(8, int(0.75 * len(feature_rows)))
    # Distinct z_fraction on the same hollow pair must change COM height features.
    for _pair_idx, rows in by_pair.items():
        z_to_feat = {}
        for zf, row in rows:
            z_to_feat.setdefault(zf, row)
        if len(z_to_feat) >= 2:
            assert len(set(z_to_feat.values())) == len(z_to_feat)


def test_fcc_catalog_has_atop_bridge_hollow_and_topology_majority():
    slab = make_slab(nx=4, ny=4, n_layers=3)
    sites = get_unified_sites(slab, material_type="slab", enrich=True)
    types = {s.site_type for s in sites}
    assert {"atop", "bridge", "hollow"} <= types
    topo = sum(1 for s in sites if str(s.site_source).startswith("topology"))
    assert topo >= len(sites) // 3


def test_dissociative_pair_cache_key_includes_voronoi_params():
    slab = make_slab()
    base = AdsorptionConfig(material_type="slab")
    alt = AdsorptionConfig(
        material_type="slab",
        voronoi_probe_radius=(base.voronoi_probe_radius or 1.0) + 0.5,
    )
    assert _dissociative_pair_cache_key(slab, base) != _dissociative_pair_cache_key(
        slab, alt
    )


def test_dissociative_pair_cache_ignores_site_context_calls():
    """site_context catalogs must not poison the clean-slab process cache."""
    from metalsurfer.placement.dissociative import _DISSOCIATIVE_PAIR_CACHE
    from metalsurfer.placement.site_context import SiteContext

    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", skip_topology_check=True)

    lonely = Site(
        xyz=np.array([1.0, 1.0, 6.0]),
        normal=np.array([0.0, 0.0, 1.0]),
        site_type="hollow",
        slab_indices=(0,),
        material_type="slab",
        site_source="test",
        env_fingerprint=((), "hollow"),
    )
    ctx = SiteContext(
        sites=[lonely],
        use_sites=True,
        source="test",
        raw_unclustered=[lonely],
    )
    empty = _get_dissociative_site_pairs(slab, config, site_context=ctx)
    assert empty == []

    full = _get_dissociative_site_pairs(slab, config)
    assert full, "clean-slab path must still discover pairs after a site_context call"
    assert len(_DISSOCIATIVE_PAIR_CACHE) >= 1


def test_dissociative_pair_cache_hits_with_site_context_and_occupancy(monkeypatch):
    """Same site_context + occupancy positions must hit the process cache."""
    from metalsurfer.placement import dissociative as dissoc_mod
    from metalsurfer.placement.dissociative import _DISSOCIATIVE_PAIR_CACHE
    from metalsurfer.placement.site_context import (
        SiteContext,
        _get_unique_sites_for_specs,
    )

    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", skip_topology_check=True)
    core = _get_unique_sites_for_specs(slab, config)
    ctx = SiteContext(
        sites=core.sites,
        use_sites=True,
        source=core.source,
        raw_unclustered=core.raw_unclustered,
    )
    occ = np.array([[0.5, 0.5, 8.0]], dtype=float)

    first = _get_dissociative_site_pairs(
        slab, config, existing_adsorbate_positions=occ, site_context=ctx
    )
    n_cached = len(_DISSOCIATIVE_PAIR_CACHE)
    assert n_cached >= 1

    calls = {"n": 0}
    orig = dissoc_mod._compute_dissociative_site_pairs

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(dissoc_mod, "_compute_dissociative_site_pairs", _counting)
    second = _get_dissociative_site_pairs(
        slab, config, existing_adsorbate_positions=occ, site_context=ctx
    )
    assert calls["n"] == 0
    assert len(second) == len(first)

    other_occ = np.array([[5.0, 5.0, 8.0]], dtype=float)
    _get_dissociative_site_pairs(
        slab, config, existing_adsorbate_positions=other_occ, site_context=ctx
    )
    assert calls["n"] == 1
    assert len(_DISSOCIATIVE_PAIR_CACHE) > n_cached

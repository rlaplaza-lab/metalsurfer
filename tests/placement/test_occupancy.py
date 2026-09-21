"""Occupancy-aware packing, recovery and fill strategies."""

from collections.abc import Callable

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import PlacementPose, PlacementSpec
from metalsurfer.placement import (
    check_initial_placement_distance,
    get_unified_sites,
    material_aware_pbc,
)
from metalsurfer.placement.dissociative import _get_dissociative_site_pairs
from metalsurfer.placement.occupancy import (
    filter_sites_by_occupancy,
    results_mutually_clear,
)
from metalsurfer.placement.pose import (
    _finalize_placement,
    _PlacementContext,
)
from metalsurfer.placement.site_context import _get_unique_sites_for_specs
from metalsurfer.placement.site_enumeration import (
    _compute_site_z_base,
)
from metalsurfer.placement.site_types import Site
from metalsurfer.workflow.shared import PlacementFailureEvent

from ..conftest import (
    make_h2,
    make_nanoparticle,
    make_placement_descriptor,
    make_porous_framework,
    make_slab,
    make_water,
    water_conformers,
)
from ._helpers import (
    _generate_placements,
    _make_site,
    _round_atop_placement_spec,
    dissoc_placement_spec,
)

_SpecFilter = Callable[[PlacementSpec], bool] | None
_SpecFactory = Callable[[int, _SpecFilter], list[PlacementSpec]]


def _filter_specs(
    specs: list[PlacementSpec], filter_spec: _SpecFilter
) -> list[PlacementSpec]:
    if filter_spec is None:
        return specs
    return [s for s in specs if filter_spec(s)]


def _atop_specs(n_desired: int, filter_spec: _SpecFilter) -> list[PlacementSpec]:
    return _filter_specs(
        [_round_atop_placement_spec(i) for i in range(n_desired)], filter_spec
    )


def _enumerate_from(
    make_specs: _SpecFactory,
) -> Callable[..., list[PlacementSpec]]:
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
        conformer_energies=None,
        grid_info=None,
    ):
        return make_specs(n_desired, filter_spec)

    return fake_enumerate


def _patch_fill(
    monkeypatch: pytest.MonkeyPatch,
    fill_mod: object,
    *,
    enumerate_fn: Callable[..., list[PlacementSpec]] | None = None,
    materialize_fn: Callable[..., tuple] | None = None,
) -> None:
    if enumerate_fn is not None:
        monkeypatch.setattr(fill_mod, "enumerate_placement_specs", enumerate_fn)
    if materialize_fn is not None:
        monkeypatch.setattr(fill_mod, "_materialize_spec_placements", materialize_fn)


def _materialize_all_succeed() -> Callable[..., tuple]:
    def fake_materialize(**kwargs: object) -> tuple:
        combined, ids, descs = [], [], []
        for spec in kwargs["specs"]:
            desc = make_placement_descriptor(placement_id=spec.placement_index)
            combined.append(Atoms("H"))
            ids.append(spec.placement_index)
            descs.append(desc)
        return combined, ids, descs, []

    return fake_materialize


def _materialize_all_fail(reason: str = "too_close") -> Callable[..., tuple]:
    def fake_materialize(**kwargs: object) -> tuple:
        return (
            [],
            [],
            [],
            [
                PlacementFailureEvent(
                    placement_id=spec.placement_index,
                    stage="generation",
                    reason=reason,
                    descriptor=None,
                )
                for spec in kwargs["specs"]
            ],
        )

    return fake_materialize


def _materialize_every_other() -> Callable[..., tuple]:
    """Succeed on even indices within each batch (≈50% yield)."""
    succeed = _materialize_all_succeed()

    def fake_materialize(**kwargs: object) -> tuple:
        specs = kwargs["specs"]
        keep = [s for i, s in enumerate(specs) if i % 2 == 0]
        fail = [s for i, s in enumerate(specs) if i % 2 == 1]
        combined, ids, descs, _ = succeed(specs=keep)
        _, _, _, failures = _materialize_all_fail()(specs=fail)
        return combined, ids, descs, failures

    return fake_materialize


def _run_fill(fill_mod: object, config: AdsorptionConfig, *, slab: Atoms | None = None):
    slab_atoms = make_slab() if slab is None else slab
    return fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=slab_atoms,
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=slab_atoms,
        calculator=None,
    )


def test_env_fingerprint_present_in_unified_sites():
    """Sites carry a shared env_fingerprint after classify."""
    sites = get_unified_sites(make_slab(), material_type="slab")
    assert len(sites) > 0
    for s in sites:
        fp = s.env_fingerprint
        assert isinstance(fp, tuple) and len(fp) == 3
        assert isinstance(fp[0], tuple)
        assert isinstance(fp[1], tuple)
        assert isinstance(fp[2], int)
        assert s.tangent_basis is not None
        assert np.asarray(s.tangent_basis).shape == (2, 3)


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


def test_estimate_complexity_shrinks_under_coverage():
    from metalsurfer.placement.generators import estimate_placement_capacity
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    clean = estimate_placement_capacity(
        [make_water()], slab, config, "O", site_context=ctx
    )
    # Block nearly all sites by placing a dense adsorbate cloud near every site.
    ads_pos = np.vstack([s.xyz + np.array([0.0, 0.0, 0.2]) for s in ctx.sites])
    ads = Atoms(["H"] * len(ads_pos), positions=ads_pos)
    full = slab.copy() + ads
    covered = estimate_placement_capacity(
        [make_water()],
        slab,
        config,
        "O",
        site_context=ctx,
        full_slab=full,
    )
    assert covered < clean
    # Full-surface saturation collapses estimated complexity to zero by construction.
    assert covered == 0.0


def test_occupancy_pruning_uses_min_adsorbate_separation_not_min_initial_distance():
    """Site pruning must honour min_adsorbate_separation when defaults diverge."""
    from metalsurfer.placement.generators import estimate_placement_capacity
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    slab = make_slab()
    ctx = resolve_site_context_for_sampling(
        slab, AdsorptionConfig(material_type="slab", seed=0), symmetry_broken=True
    )
    assert ctx.use_sites and ctx.sites
    site = ctx.sites[0]
    # Place an adsorbate atom ~2 A from the catalog anchor.
    ads = Atoms("H", positions=[site.xyz + np.array([2.0, 0.0, 0.0])])
    full = slab.copy() + ads

    loose = AdsorptionConfig(
        material_type="slab",
        seed=0,
        min_initial_distance=1.5,
        min_adsorbate_separation=1.5,
    )
    strict = AdsorptionConfig(
        material_type="slab",
        seed=0,
        min_initial_distance=1.5,
        min_adsorbate_separation=3.0,
    )
    score_loose = estimate_placement_capacity(
        [make_water()], slab, loose, "O", site_context=ctx, full_slab=full
    )
    score_strict = estimate_placement_capacity(
        [make_water()], slab, strict, "O", site_context=ctx, full_slab=full
    )
    assert score_strict < score_loose


def test_footprint_ranks_without_pruning():
    """Vertex gate keeps both sites; footprint clearance ranks the far one higher."""
    from metalsurfer.placement.occupancy import (
        available_site_indices,
        site_footprint_clearances,
    )
    from metalsurfer.placement.site_types import Site

    def _site(xyz, idx):
        return Site(
            xyz=np.array(xyz, dtype=float),
            normal=np.array([0.0, 0.0, 1.0], dtype=float),
            site_type="atop",
            slab_indices=(idx,),
            material_type="slab",
            site_source="topology",
            env_fingerprint=((), (), 0),
        )

    near, far = _site([0.0, 0.0, 0.0], 0), _site([10.0, 0.0, 0.0], 1)
    existing = np.array([[2.2, 0.0, 0.0]], dtype=float)
    existing_radii = np.array([0.7], dtype=float)
    cell = np.eye(3) * 30.0
    pbc = [False, False, False]
    assert available_site_indices(
        [near, far], existing, cell=cell, pbc=pbc, min_separation=1.5
    ) == [0, 1]
    clearances = site_footprint_clearances(
        [near, far],
        existing,
        existing_radii,
        cell=cell,
        pbc=pbc,
        incoming_radius=2.0,
    )
    assert clearances[1] > clearances[0]
    # Negative clearance does not prune: vertex gate still returns both sites.
    assert clearances[0] < 0.0


def test_incoming_inplane_radius_drops_thickness_axis():
    from metalsurfer.placement.occupancy import incoming_inplane_radius

    # Flat molecule in xy: radius ~1.0 with scale 1.0.
    flat = Atoms(
        "CCC",
        positions=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    )
    r = incoming_inplane_radius(flat, footprint_scale=1.0)
    assert 0.9 < r < 1.1


def test_overlap_recovery_rescues_lateral_clash():
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
        z_base_lo=1.5,
        z_base_hi=3.0,
        normal=np.array([0.0, 0.0, 1.0]),
    )
    result, reason = _finalize_placement(
        ctx,
        water,
        full,
        config,
        slab_for_sites=slab,
        allow_distance_recovery=True,
    )
    assert reason is None, f"seeded lateral clash should recover, got {reason}"
    assert result is not None
    adsorbate_ok, descriptor = result
    assert (
        abs(float(descriptor.x_abs) - 2.0) > 1e-3
        or abs(float(descriptor.y_abs) - 2.0) > 1e-3
    ), "overlap recovery should nudge XY away from the clash"
    gate_ok, min_d, gate_reason = check_initial_placement_distance(
        adsorbate_ok, slab, material_type="slab"
    )
    assert gate_ok, (min_d, gate_reason)


def test_clearance_aware_height_raises_protruding_pose():
    """Pairwise contact-solve clears the gate for a tilted protruding chain."""
    from metalsurfer.placement.geometry import check_initial_placement_distance
    from metalsurfer.placement.pose import (
        _contact_atom_index,
        _pose_from_spec,
    )

    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=1,
        placement_z_range=(2.0, 3.0),
        placement_z_scale_by_covalent_radius=False,
        seed=0,
    )
    # Elongated chain; binder-aligned orientation uses O as contact atom.
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
    assert ctx.site is not None
    n_hat = np.asarray(ctx.normal, dtype=float)
    contact_idx = _contact_atom_index(
        ctx.rotated_pos,
        n_hat,
        list(chain.get_chemical_symbols()),
        orientation_type=spec.orientation_type,
        en_atom_index=spec.en_atom_index,
    )
    assert contact_idx == 0  # oxygen binder
    ads = chain.copy()
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
    # Protruding C atoms sit above the binder along the site normal.
    atom_heights = ads.get_positions() @ n_hat
    assert (
        float(np.min(atom_heights))
        == pytest.approx(float(atom_heights[contact_idx]), abs=0.05)
        or float(atom_heights[contact_idx]) <= float(np.min(atom_heights)) + 0.05
    )
    assert float(np.max(atom_heights)) > float(atom_heights[contact_idx]) + 0.2


def test_place_dissociative_two_sites_matches_spec_path():
    from metalsurfer.placement.dissociative import (
        _generate_dissociative_placement_from_spec,
        _place_dissociative_two_sites,
    )
    from metalsurfer.placement.orientation import _site_type_z_offset

    slab = make_slab()
    h2 = make_h2()
    config = AdsorptionConfig(
        material_type="slab", enable_dissociative_placement=True, seed=0
    )
    pairs = _get_dissociative_site_pairs(slab, config, slab_for_sites=slab)
    assert len(pairs) >= 1
    spec = dissoc_placement_spec()
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
        env_fingerprint=((), (), 0),
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
        env_fingerprint=((), (), 0),
    )
    site_b = Site(
        xyz=pair.xyz2,
        normal=pair.normal2,
        site_type="hollow",
        slab_indices=(),
        material_type="slab",
        site_source="dissociative_hollow_pair",
        env_fingerprint=((), (), 0),
    )
    via_place = _place_dissociative_two_sites(
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
    # Both code paths must realize the identical placement bit-for-bit.
    assert np.allclose(placed_a.get_positions(), placed_b.get_positions(), atol=1e-10)
    assert desc_a.fragment_positions is not None


def test_packing_yield_improves_with_occupancy_prune():
    """In-plane occupancy under coverage drops near-site capacity vs bare slab."""
    from metalsurfer.placement.occupancy import (
        available_site_indices,
        existing_adsorbate_cloud,
        incoming_inplane_radius,
        site_footprint_clearances,
    )
    from metalsurfer.placement.site_context import resolve_site_context_for_sampling

    slab = make_slab()
    config = AdsorptionConfig(
        material_type="slab", seed=0, min_adsorbate_separation=1.5
    )
    ctx = resolve_site_context_for_sampling(slab, config, symmetry_broken=True)
    site = ctx.sites[0]
    pre = Atoms(
        "O", positions=[np.asarray(site.xyz, dtype=float) + np.array([0.0, 0.0, 0.4])]
    )
    full = slab.copy() + pre
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = [True, True, False]
    existing_pos, existing_radii = existing_adsorbate_cloud(
        slab, full, min_separation=1.5
    )
    bare = available_site_indices(
        ctx.sites, None, cell=cell, pbc=pbc, min_separation=1.5
    )
    covered = available_site_indices(
        ctx.sites, existing_pos, cell=cell, pbc=pbc, min_separation=1.5
    )
    assert len(covered) < len(bare)
    occupied_idx = next(
        i for i, s in enumerate(ctx.sites) if np.allclose(s.xyz, site.xyz)
    )
    assert occupied_idx not in covered
    # Footprint ranking still scores the occupied anchor worse than survivors.
    clearances = site_footprint_clearances(
        ctx.sites,
        existing_pos,
        existing_radii,
        cell=cell,
        pbc=pbc,
        incoming_radius=incoming_inplane_radius(make_water(), footprint_scale=0.85),
    )
    assert float(np.max(clearances[covered])) > float(clearances[occupied_idx])


@pytest.mark.parametrize(
    "plugin",
    ["topology", "voronoi", "adaptive_grid", "rolling_probe"],
)
def test_inplane_occupancy_rejects_overhead_adsorbate(plugin):
    """Adsorbate above an unlifted anchor along the site normal still occupies it."""
    from metalsurfer.placement.occupancy import available_site_indices

    slab = make_slab()
    sites = get_unified_sites(
        slab, material_type="slab", site_generator=plugin, n_jobs=1
    )
    site = next(s for s in sites if s.slab_indices)
    n_hat = np.asarray(site.normal, dtype=float)
    n_hat = n_hat / float(np.linalg.norm(n_hat))
    # Contact-height adsorbate along the site normal: 3D distance is ~2 Å, so a
    # 3D gate would miss it; in-plane MIC must still reject the column.
    existing = np.asarray(site.xyz, dtype=float) + 2.0 * n_hat
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = [True, True, False]
    covered = available_site_indices(
        sites, existing.reshape(1, 3), cell=cell, pbc=pbc, min_separation=1.5
    )
    occupied_idx = next(i for i, s in enumerate(sites) if np.allclose(s.xyz, site.xyz))
    assert occupied_idx not in covered
    # Laterally far site stays available.
    far = next(
        (
            s
            for s in sites
            if float(
                np.linalg.norm(
                    (s.xyz - site.xyz) - np.dot(s.xyz - site.xyz, n_hat) * n_hat
                )
            )
            > 3.0
        ),
        None,
    )
    assert far is not None
    far_idx = next(i for i, s in enumerate(sites) if np.allclose(s.xyz, far.xyz))
    assert far_idx in covered


def test_fill_oversamples_to_meet_num_placements(monkeypatch):
    """50% materialization yield still fills n_target via one-shot oversampling."""
    from metalsurfer.workflow import placement_fill as fill_mod

    requested = []

    def make_specs(n_desired, filter_spec):
        requested.append(n_desired)
        return _atop_specs(n_desired, filter_spec)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=_materialize_every_other(),
    )
    result = _run_fill(
        fill_mod,
        AdsorptionConfig(
            material_type="slab",
            num_placements=4,
            placement_retry_enabled=False,
            placement_retry_oversample_max=4.0,
            seed=0,
        ),
    )
    assert len(result.combined) == 4
    assert requested[0] >= 4
    assert result.n_attempts == 1


def test_fill_diversity_retry(monkeypatch):
    """Short first pass triggers one failed-key retry with fresh indices."""
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.placement_fill import placement_spec_key

    seen_keys: list[set[tuple]] = []
    attempt_indices: list[list[int]] = []
    round_id = {"n": 0}

    def make_specs(n_desired, filter_spec):
        base = 1000 * round_id["n"]
        round_id["n"] += 1
        specs = [
            _round_atop_placement_spec(i, site_index=base + i) for i in range(n_desired)
        ]
        filtered = _filter_specs(specs, filter_spec)
        seen_keys.append({placement_spec_key(s) for s in filtered})
        return filtered

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        attempt_indices.append([s.placement_index for s in specs])
        # First enumeration uses site_index < 1000; retry uses >= 1000.
        if any(int(s.site_index) < 1000 for s in specs):
            return _materialize_all_fail()(specs=specs)
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    result = _run_fill(
        fill_mod,
        AdsorptionConfig(
            material_type="slab",
            num_placements=4,
            placement_retry_enabled=True,
            placement_retry_oversample_max=2.0,
            seed=0,
        ),
    )
    assert len(result.combined) == 4
    assert result.n_attempts == 2
    assert seen_keys[0].isdisjoint(seen_keys[1])
    flat = [i for batch in attempt_indices for i in batch]
    assert len(flat) == len(set(flat))


def test_fill_retry_keeps_same_fingerprint_on_too_close(monkeypatch):
    """too_close bans the exact spec, not sibling sites that share a fingerprint."""
    from metalsurfer.placement.site_context import SiteContext
    from metalsurfer.placement.site_types import Site
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    shared_fp = (("Ru",), (0,), 1)
    other_fp = (("Ru", "Ru"), (0, 0), 1)
    sites = [
        Site(
            xyz=np.array([0.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(0,),
            material_type="slab",
            site_source="test",
            env_fingerprint=shared_fp,
        ),
        Site(
            xyz=np.array([1.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(1,),
            material_type="slab",
            site_source="test",
            env_fingerprint=shared_fp,
        ),
        Site(
            xyz=np.array([2.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="bridge",
            slab_indices=(0, 1),
            material_type="slab",
            site_source="test",
            env_fingerprint=other_fp,
        ),
    ]
    ctx = SiteContext(sites=sites, use_sites=True, source="test")
    seen_site_indices: list[list[int]] = []
    round_id = {"n": 0}

    def make_specs(n_desired, filter_spec):
        round_id["n"] += 1
        if round_id["n"] == 1:
            specs = [_round_atop_placement_spec(0, site_index=0)]
        else:
            specs = [
                _round_atop_placement_spec(0, site_index=1),
                _round_atop_placement_spec(1, site_index=2),
            ]
        filtered = _filter_specs(specs, filter_spec)
        seen_site_indices.append([s.site_index for s in filtered])
        return filtered[:n_desired]

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        if round_id["n"] == 1:
            return (
                [],
                [],
                [],
                [
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                    for spec in specs
                ],
            )
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    slab = make_slab()
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=slab,
        config=AdsorptionConfig(
            material_type="slab",
            num_placements=1,
            placement_retry_enabled=True,
            placement_retry_oversample_max=2.0,
            seed=0,
            placement_fill_clamp_to_capacity=False,
        ),
        smiles="O",
        site_context=ctx,
        slab_atoms=slab,
        calculator=None,
    )
    assert len(result.combined) == 1
    assert result.n_attempts == 2
    assert seen_site_indices[0] == [0]
    # Same fingerprint remains eligible after a pose failure.
    assert 1 in seen_site_indices[1]
    assert 2 in seen_site_indices[1]


def test_fill_retry_bans_pose_family_after_too_close(monkeypatch):
    """too_close bans same-height z fractions; a higher fraction may redraw."""
    from metalsurfer.placement.site_context import SiteContext
    from metalsurfer.placement.site_types import Site
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    sites = [
        Site(
            xyz=np.array([0.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(0,),
            material_type="slab",
            site_source="test",
            env_fingerprint=(("Ru",), (0,), 1),
        ),
        Site(
            xyz=np.array([2.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(1,),
            material_type="slab",
            site_source="test",
            env_fingerprint=(("Ru",), (0,), 1),
        ),
    ]
    ctx = SiteContext(sites=sites, use_sites=True, source="test")
    seen_specs: list[list[tuple[int, int, float]]] = []
    round_id = {"n": 0}

    def make_specs(n_desired, filter_spec):
        round_id["n"] += 1
        if round_id["n"] == 1:
            specs = [_round_atop_placement_spec(0, site_index=0, z_fraction=0.3)]
        else:
            specs = [
                _round_atop_placement_spec(0, site_index=0, z_fraction=0.1),
                _round_atop_placement_spec(0, site_index=0, z_fraction=0.3),
                _round_atop_placement_spec(0, site_index=0, z_fraction=0.7),
                _round_atop_placement_spec(0, site_index=1, z_fraction=0.5),
            ]
        filtered = _filter_specs(specs, filter_spec)
        seen_specs.append(
            [
                (int(s.conformer_index), int(s.site_index), float(s.z_fraction))
                for s in filtered
            ]
        )
        return filtered[:n_desired]

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        if round_id["n"] == 1:
            return (
                [],
                [],
                [],
                [
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                    for spec in specs
                ],
            )
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    slab = make_slab()
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=slab,
        config=AdsorptionConfig(
            material_type="slab",
            num_placements=1,
            placement_retry_enabled=True,
            placement_retry_oversample_max=2.0,
            seed=0,
            placement_fill_clamp_to_capacity=False,
        ),
        smiles="O",
        site_context=ctx,
        slab_atoms=slab,
        calculator=None,
    )
    assert len(result.combined) == 1
    assert result.n_attempts == 2
    assert (0, 0, 0.3) in seen_specs[0]
    # Contact-height siblings (z <= 0.5) of the failed pose are dropped.
    # A higher fraction and a different site may redraw.
    assert (0, 0, 0.1) not in seen_specs[1]
    assert (0, 0, 0.3) not in seen_specs[1]
    assert (0, 0, 0.7) in seen_specs[1]
    assert (0, 1, 0.5) in seen_specs[1]


def test_fill_retry_bans_site_on_adsorbate_overlap(monkeypatch):
    """adsorbate_overlap bans (conformer, site) on retry, not every conformer."""
    from metalsurfer.placement.site_context import SiteContext
    from metalsurfer.placement.site_types import Site
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    shared_fp = (("Ru",), (0,), 1)
    sites = [
        Site(
            xyz=np.array([0.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(0,),
            material_type="slab",
            site_source="test",
            env_fingerprint=shared_fp,
        ),
        Site(
            xyz=np.array([1.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(1,),
            material_type="slab",
            site_source="test",
            env_fingerprint=shared_fp,
        ),
    ]
    ctx = SiteContext(sites=sites, use_sites=True, source="test")
    seen_pairs: list[list[tuple[int, int]]] = []
    round_id = {"n": 0}

    def make_specs(n_desired, filter_spec):
        round_id["n"] += 1
        if round_id["n"] == 1:
            specs = [_round_atop_placement_spec(0, site_index=0)]
        else:
            # Same site, different conformer must remain eligible; other site too.
            specs = [
                _round_atop_placement_spec(0, site_index=0),
                _round_atop_placement_spec(1, site_index=0),
                _round_atop_placement_spec(0, site_index=1),
            ]
        filtered = _filter_specs(specs, filter_spec)
        seen_pairs.append([(s.conformer_index, s.site_index) for s in filtered])
        return filtered[:n_desired]

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        if round_id["n"] == 1:
            return (
                [],
                [],
                [],
                [
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="adsorbate_overlap",
                        descriptor=None,
                    )
                    for spec in specs
                ],
            )
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    slab = make_slab()
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water(), make_water()],
        slab_for_sites=slab,
        config=AdsorptionConfig(
            material_type="slab",
            num_placements=1,
            placement_retry_enabled=True,
            placement_retry_oversample_max=2.0,
            seed=0,
            placement_fill_clamp_to_capacity=False,
        ),
        smiles="O",
        site_context=ctx,
        slab_atoms=slab,
        calculator=None,
    )
    assert len(result.combined) == 1
    assert result.n_attempts == 2
    assert seen_pairs[0] == [(0, 0)]
    assert (0, 0) not in seen_pairs[1]
    assert (1, 0) in seen_pairs[1] or (0, 1) in seen_pairs[1]


def test_fill_chunked_stops_early(monkeypatch):
    """Chunked materialize does not process the oversampled tail once full."""
    from metalsurfer.workflow import placement_fill as fill_mod

    materialized_counts: list[int] = []

    def make_specs(n_desired, filter_spec):
        specs = [_round_atop_placement_spec(i, site_index=i) for i in range(n_desired)]
        return _filter_specs(specs, filter_spec)

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        materialized_counts.append(len(specs))
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    result = _run_fill(
        fill_mod,
        AdsorptionConfig(
            material_type="slab",
            num_placements=2,
            placement_retry_enabled=False,
            placement_retry_oversample_max=6.0,
            seed=0,
            placement_fill_clamp_to_capacity=False,
        ),
    )
    assert len(result.combined) == 2
    assert result.n_attempts == 1
    # First chunk of size n_target succeeds fully → no further chunks.
    assert materialized_counts == [2]


def test_fill_failure_sets_are_per_call(monkeypatch):
    """Two fill calls with a shared catalog do not leak failed-spec bans."""
    from metalsurfer.placement.site_context import SiteContext
    from metalsurfer.placement.site_types import Site
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    sites = [
        Site(
            xyz=np.array([0.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(0,),
            material_type="slab",
            site_source="test",
            env_fingerprint=(("Ru",), (0,), 1),
        ),
        Site(
            xyz=np.array([1.0, 0.0, 5.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            site_type="atop",
            slab_indices=(1,),
            material_type="slab",
            site_source="test",
            env_fingerprint=(("Ru",), (0,), 1),
        ),
    ]
    ctx = SiteContext(sites=sites, use_sites=True, source="test")
    seen: list[list[int]] = []
    call_n = {"n": 0}

    def make_specs(n_desired, filter_spec):
        call_n["n"] += 1
        specs = [
            _round_atop_placement_spec(0, site_index=0),
            _round_atop_placement_spec(1, site_index=1),
        ]
        filtered = _filter_specs(specs, filter_spec)
        seen.append([s.site_index for s in filtered])
        return filtered[:n_desired]

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        # First fill call fails everything; second call (new fill) succeeds.
        if call_n["n"] <= 1:
            return (
                [],
                [],
                [],
                [
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                    for spec in specs
                ],
            )
        return _materialize_all_succeed()(specs=specs)

    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(make_specs),
        materialize_fn=fake_materialize,
    )
    slab = make_slab()
    cfg = AdsorptionConfig(
        material_type="slab",
        num_placements=1,
        placement_retry_enabled=False,
        placement_retry_oversample_max=1.0,
        seed=0,
        placement_fill_clamp_to_capacity=False,
    )
    first = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=slab,
        config=cfg,
        smiles="O",
        site_context=ctx,
        slab_atoms=slab,
        calculator=None,
    )
    assert first.combined == []
    second = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=slab,
        config=cfg,
        smiles="O",
        site_context=ctx,
        slab_atoms=slab,
        calculator=None,
    )
    assert len(second.combined) == 1
    # Second call still sees site 0 — failure sets do not persist across fills.
    assert 0 in seen[1]


def test_fill_empty_enumeration(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    calls = {"n": 0}

    def make_specs(_n_desired, _filter_spec):
        calls["n"] += 1
        return []

    _patch_fill(monkeypatch, fill_mod, enumerate_fn=_enumerate_from(make_specs))
    result = _run_fill(
        fill_mod,
        AdsorptionConfig(material_type="slab", num_placements=5, seed=0),
    )
    assert result.combined == []
    assert calls["n"] == 1
    assert result.n_attempts == 0


def test_resolve_materialize_workers_joblib_semantics():
    from metalsurfer.placement.generators import resolve_materialize_workers

    assert resolve_materialize_workers(1, cpu_count=8) == 1
    assert resolve_materialize_workers(4, cpu_count=8) == 4
    assert resolve_materialize_workers(-1, cpu_count=8) == 8
    assert resolve_materialize_workers(-2, cpu_count=8) == 7
    assert resolve_materialize_workers(-2, cpu_count=1) == 1
    assert resolve_materialize_workers(4, n_tasks=2, cpu_count=8) == 2
    with pytest.raises(ValueError, match="n_jobs"):
        resolve_materialize_workers(0, cpu_count=8)


def test_placement_workers_inherit_global_n_jobs():
    """placement_materialize_workers=None must inherit the global n_jobs knob."""
    from metalsurfer.placement.generators import (
        generate_placements_from_specs,
        resolve_materialize_workers,
    )

    config = AdsorptionConfig(n_jobs=3, placement_materialize_workers=None)
    inherited = (
        config.n_jobs
        if config.placement_materialize_workers is None
        else config.placement_materialize_workers
    )
    assert inherited == 3
    assert resolve_materialize_workers(inherited, n_tasks=10, cpu_count=8) == 3
    assert (
        AdsorptionConfig(
            n_jobs=3, placement_materialize_workers=1
        ).placement_materialize_workers
        == 1
    )

    slab = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 1.8]])
    slab.set_cell([8.0, 8.0, 20.0])
    slab.set_pbc([True, True, False])
    adsorbate = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    specs = [_round_atop_placement_spec(i) for i in range(2)]
    for i, spec in enumerate(specs):
        spec.placement_index = i
    results_a = generate_placements_from_specs(
        specs, [adsorbate], slab, AdsorptionConfig(n_jobs=1)
    )
    results_b = generate_placements_from_specs(
        specs, [adsorbate], slab, AdsorptionConfig(n_jobs=-1)
    )
    assert len(results_a) == len(results_b) == 2
    for (res_a, _), (res_b, _) in zip(results_a, results_b, strict=True):
        if res_a is None or res_b is None:
            assert res_a is None and res_b is None
            continue
        assert np.allclose(res_a[0].get_positions(), res_b[0].get_positions())


def test_materialize_specs_trims_to_target(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    monkeypatch.setattr(
        fill_mod, "_materialize_spec_placements", _materialize_all_succeed()
    )
    result = fill_mod.materialize_specs(
        specs=[_round_atop_placement_spec(i) for i in range(6)],
        n_target=4,
        conformers=[make_water()],
        slab_atoms=make_slab(),
        calculator=None,
        config=AdsorptionConfig(
            material_type="slab",
            placement_fill_clamp_to_capacity=False,
        ),
        smiles="O",
        site_context=None,
    )
    assert len(result.combined) == 4
    assert result.placement_ids == [0, 1, 2, 3]
    assert result.n_attempts == 1


def test_fill_clamps_target_to_capacity(monkeypatch, caplog):
    import logging

    from metalsurfer.workflow import placement_fill as fill_mod

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", lambda *a, **k: 2)
    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(_atop_specs),
        materialize_fn=_materialize_all_succeed(),
    )
    with caplog.at_level(logging.WARNING):
        result = _run_fill(
            fill_mod,
            AdsorptionConfig(
                material_type="slab",
                num_placements=10,
                placement_retry_enabled=False,
                seed=0,
            ),
        )
    assert len(result.combined) == 2
    assert result.n_attempts == 1
    assert any("clamped from" in r.message for r in caplog.records)


def test_clamp_flag_false_legacy(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    monkeypatch.setattr(fill_mod, "estimate_placement_spec_capacity", lambda *a, **k: 2)
    _patch_fill(
        monkeypatch,
        fill_mod,
        enumerate_fn=_enumerate_from(_atop_specs),
        materialize_fn=_materialize_all_succeed(),
    )
    result = _run_fill(
        fill_mod,
        AdsorptionConfig(
            material_type="slab",
            num_placements=10,
            placement_retry_enabled=False,
            placement_fill_clamp_to_capacity=False,
            seed=0,
        ),
    )
    assert len(result.combined) == 10


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


# ---------------------------------------------------------------------------
# Rec 2a — occupancy packing / blocked-site exclusion across material types.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "material_type, factory",
    [
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
        ("porous", make_porous_framework),
    ],
)
def test_occupancy_filter_excludes_near_adsorbate_each_material(material_type, factory):
    """A site near an existing adsorbate is blocked; a free site is kept (per material)."""
    structure = factory()
    cell = np.asarray(structure.get_cell(), dtype=float)
    pbc = material_aware_pbc(material_type)
    site = _make_site([2.0, 2.0, 5.0], material_type=material_type)
    far = _make_site([8.0, 8.0, 5.0], material_type=material_type)
    existing = np.array([[2.05, 2.05, 5.0]])
    kept = filter_sites_by_occupancy(
        [site, far],
        existing,
        cell=cell,
        pbc=pbc,
        min_separation=2.0,
    )
    assert len(kept) == 1
    assert np.allclose(kept[0].xyz, far.xyz)


@pytest.mark.parametrize(
    "material_type, factory",
    [
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
        ("porous", make_porous_framework),
    ],
)
def test_initial_placement_distance_packs_free_rejects_blocked_each_material(
    material_type, factory
):
    """A real generated placement passes the gate; an overlapping one is blocked."""
    structure = factory()
    config = AdsorptionConfig(material_type=material_type, seed=0)
    accepted = False
    for _spec, adsorbate, _desc in _generate_placements(
        water_conformers(), structure, config, smiles="O", n_desired=8
    ):
        ok_free, _, reason_free = check_initial_placement_distance(
            adsorbate, structure, material_type=material_type
        )
        if ok_free:
            accepted = True
            break
    assert accepted, "expected at least one gate-accepted generated placement"

    blocked = make_water().copy()
    bpos = blocked.get_positions().copy()
    bpos += structure.get_positions()[0]
    blocked.set_positions(bpos)
    blocked.set_cell(structure.get_cell())
    blocked.set_pbc(structure.get_pbc())
    ok_blocked, _, reason_blocked = check_initial_placement_distance(
        blocked, structure, material_type=material_type
    )
    assert not ok_blocked
    assert reason_blocked in ("too_close", "empty_geometry")


# ---------------------------------------------------------------------------
# results_mutually_clear: n-tuplet pairwise adsorbate clearance
# ---------------------------------------------------------------------------


def _water_suffix_at(x_shift: float) -> Atoms:
    """Adsorbate-only water fragment shifted along x (cell-sized slab context)."""
    mol = make_water()
    pos = mol.get_positions().copy()
    pos[:, 0] += x_shift
    mol.set_positions(pos)
    return mol


def test_results_mutually_clear_accepts_separated_fragments():
    """Fragments several Å apart under the slab MIC are mutually clear."""
    slab = make_slab()
    clear = results_mutually_clear(
        _water_suffix_at(3.0),
        _water_suffix_at(7.0),
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
        min_separation=2.0,
    )
    assert clear


def test_results_mutually_clear_rejects_overlapping_fragments():
    """Two fragments at the same site clash below any sane min_separation."""
    slab = make_slab()
    clear = results_mutually_clear(
        _water_suffix_at(5.0),
        _water_suffix_at(5.2),
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
        min_separation=2.0,
    )
    assert not clear


def test_results_mutually_clear_honours_boundary_equality():
    """Distance exactly at min_separation counts as clear (>= semantics)."""
    slab = make_slab()
    a = Atoms("H", positions=[[10.0, 5.4, 5.7]])
    b = Atoms("H", positions=[[12.0, 5.4, 5.7]])
    assert results_mutually_clear(
        a, b, cell=slab.get_cell(), pbc=material_aware_pbc("slab"), min_separation=2.0
    )
    assert not results_mutually_clear(
        a, b, cell=slab.get_cell(), pbc=material_aware_pbc("slab"), min_separation=2.01
    )


def test_results_mutually_clear_wraps_periodic_images():
    """A fragment near +x edge clashes with its -x periodic image."""
    slab = make_slab()
    cell = np.asarray(slab.get_cell(), dtype=float)
    near_edge = Atoms("H", positions=[[cell[0][0] - 0.5, 5.4, 5.7]])
    other_side = Atoms("H", positions=[[0.5, 5.4, 5.7]])
    assert not results_mutually_clear(
        near_edge,
        other_side,
        cell=slab.get_cell(),
        pbc=material_aware_pbc("slab"),
        min_separation=2.0,
    )

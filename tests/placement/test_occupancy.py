"""Occupancy-aware packing, recovery and fill strategies."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import PlacementPose, PlacementSpec
from metalsurfer.placement import (
    check_initial_placement_distance,
    get_unified_sites,
)
from metalsurfer.placement.dissociative import _get_dissociative_site_pairs
from metalsurfer.placement.pose import (
    _finalize_placement,
    _PlacementContext,
)
from metalsurfer.placement.site_context import (
    _get_unique_sites_for_specs,
    clear_site_caches,
)
from metalsurfer.placement.site_enumeration import (
    _compute_site_z_base,
)
from metalsurfer.placement.site_types import Site

from ..conftest import (
    make_h2,
    make_placement_descriptor,
    make_slab,
    make_water,
)
from ._helpers import (
    _make_site,
)


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
        ctx.rotated_pos + np.array([ctx.pose.x_abs, ctx.pose.y_abs, ctx.pose.z_abs])
    ) @ n_hat
    closest_h = float(np.min(atom_heights))
    z_offset = ctx.z_base_lo + spec.z_fraction * (ctx.z_base_hi - ctx.z_base_lo)
    assert closest_h == pytest.approx(ctx.surface_ref + z_offset, abs=1e-5)


def test_place_at_sites_two_site_matches_dissociative():
    from metalsurfer.models import PlacementSpec
    from metalsurfer.placement.dissociative import (
        _generate_dissociative_placement_from_spec,
        place_at_sites,
    )
    from metalsurfer.placement.orientation import _site_type_z_offset

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
    from metalsurfer.workflow import placement_fill as fill_mod
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
        conformer_energies=None,
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

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=3,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        seed=0,
    )
    fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    assert _RETRY_BLOCK_SITE_AFTER >= 2
    # After enough failures on site 3, later attempts should exclude it.
    assert any(3 not in batch for batch in seen_filters[1:])


def test_fill_oversamples_to_meet_num_placements(monkeypatch):
    """50% materialization yield still fills n_target via oversampling."""
    from ase import Atoms

    from metalsurfer.models import PlacementSpec
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    from ..conftest import make_placement_descriptor

    requested = []

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
    ):
        requested.append(n_desired)
        specs = []
        for i in range(n_desired):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=i,
                site_type="atop",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        return specs

    def fake_materialize(**kwargs):
        combined = []
        ids = []
        descs = []
        failures = []
        for i, spec in enumerate(kwargs["specs"]):
            if i % 2 == 0:
                desc = make_placement_descriptor(placement_id=spec.placement_index)
                combined.append(Atoms("H"))
                ids.append(spec.placement_index)
                descs.append(desc)
            else:
                failures.append(
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                )
        return combined, ids, descs, failures

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=4,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        placement_retry_oversample_max=4.0,
        seed=0,
    )
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    assert len(result.combined) == 4
    assert requested[0] >= 4  # oversampled beyond exact remaining
    assert result.n_attempts <= 3


def test_fill_early_stops_on_empty_enumeration(monkeypatch):
    from metalsurfer.workflow import placement_fill as fill_mod

    calls = {"n": 0}

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
    ):
        calls["n"] += 1
        return []

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=5,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        seed=0,
    )
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    assert result.combined == []
    assert calls["n"] == 1  # early exit; no wasted attempts
    assert result.n_attempts == 1


def test_request_count_respects_oversample_cap():
    from metalsurfer.workflow.placement_fill import _request_count

    assert _request_count(10, yield_est=0.5, oversample_max=4.0) == 20
    assert _request_count(10, yield_est=0.05, oversample_max=4.0) == 40
    assert _request_count(10, yield_est=1.0, oversample_max=4.0) == 10


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


def test_fill_yield_floor_keeps_oversampling_after_zero_success(monkeypatch):
    """A zero-success round must not collapse the next request to remaining only."""
    from metalsurfer.models import PlacementSpec
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    requested = []

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
    ):
        requested.append(n_desired)
        specs = []
        for i in range(n_desired):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=i + 1000 * len(requested),
                site_type="atop",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        return specs

    def fake_materialize(**kwargs):
        failures = [
            PlacementFailureEvent(
                placement_id=spec.placement_index,
                stage="generation",
                reason="too_close",
                descriptor=None,
            )
            for spec in kwargs["specs"]
        ]
        # Fail entirely on first attempt; succeed all on later attempts.
        if len(requested) == 1:
            return [], [], [], failures
        from ase import Atoms

        from ..conftest import make_placement_descriptor

        combined = []
        ids = []
        descs = []
        for spec in kwargs["specs"]:
            combined.append(Atoms("H"))
            ids.append(spec.placement_index)
            descs.append(make_placement_descriptor(placement_id=spec.placement_index))
        return combined, ids, descs, []

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=4,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        placement_retry_oversample_max=4.0,
        seed=0,
    )
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    assert len(result.combined) == 4
    assert len(requested) >= 2
    # After zero yield, next round still oversamples (cap = remaining * 4).
    assert requested[1] >= 4


def test_backfill_oversamples_by_yield(monkeypatch):
    """Backfill requests more than remaining when observed yield is low."""
    from ase import Atoms

    from metalsurfer.models import PlacementSpec
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    from ..conftest import make_placement_descriptor

    chunk_sizes = []

    def fake_materialize(**kwargs):
        specs = kwargs["specs"]
        chunk_sizes.append(len(specs))
        combined = []
        ids = []
        descs = []
        failures = []
        for i, spec in enumerate(specs):
            if i % 2 == 0:
                desc = make_placement_descriptor(placement_id=spec.placement_index)
                combined.append(Atoms("H"))
                ids.append(spec.placement_index)
                descs.append(desc)
            else:
                failures.append(
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                )
        return combined, ids, descs, failures

    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    primary = [
        PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=i,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=i,
        )
        for i in range(2)
    ]
    # First primary succeeds once (50% of 2) → need 3 more; yield_est=0.5 → request 6.
    backfill = [
        PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=100 + i,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=100 + i,
        )
        for i in range(20)
    ]
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=4,
        placement_retry_oversample_max=4.0,
        placement_materialize_workers=1,
    )
    result = fill_mod.materialize_specs_filling_target(
        primary_specs=primary,
        backfill_specs=backfill,
        n_target=4,
        conformers=[make_water()],
        slab_atoms=make_slab(),
        calculator=None,
        config=config,
        smiles="O",
        site_context=None,
    )
    assert len(result.combined) == 4
    assert chunk_sizes[0] == 2  # primary
    assert chunk_sizes[1] >= 4  # oversampled backfill for remaining=3 at 50% yield


def test_fill_clamps_target_to_capacity(monkeypatch, caplog):
    """R1: target is clamped to enumerable capacity; warns and bounds attempts."""
    import logging

    from metalsurfer.workflow import placement_fill as fill_mod

    CAPACITY = 2

    def fake_estimate(*args, **kwargs):
        return float(CAPACITY)

    monkeypatch.setattr(fill_mod, "estimate_molecule_complexity", fake_estimate)

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
    ):
        specs = []
        for i in range(n_desired):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=i,
                site_type="atop",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        return specs

    def fake_materialize(**kwargs):
        combined, ids, descs = [], [], []
        for spec in kwargs["specs"]:
            desc = make_placement_descriptor(placement_id=spec.placement_index)
            combined.append(Atoms("H"))
            ids.append(spec.placement_index)
            descs.append(desc)
        return combined, ids, descs, []

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=10,
        placement_retry_enabled=True,
        placement_retry_max_attempts=8,
        seed=0,
    )
    with caplog.at_level(logging.WARNING):
        result = fill_mod.fill_materialized_placements(
            conformers=[make_water()],
            slab_for_sites=make_slab(),
            config=config,
            smiles="O",
            site_context=None,
            slab_atoms=make_slab(),
            calculator=None,
        )
    assert len(result.combined) <= CAPACITY
    assert len(result.combined) == CAPACITY  # fully reachable here
    assert result.n_attempts < config.placement_retry_max_attempts
    assert any("clamped from" in r.message for r in caplog.records)


def test_early_stop_on_plateaued_yield(monkeypatch):
    """R2: give up after `patience` consecutive zero-yield retry attempts."""
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    calls = {"n": 0}

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
    ):
        specs = []
        for i in range(min(n_desired, 4)):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=i,
                site_type="atop",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        return specs

    def fake_materialize(**kwargs):
        calls["n"] += 1
        combined, ids, descs, failures = [], [], [], []
        # Succeed exactly once (first spec of the very first materialization).
        for j, spec in enumerate(kwargs["specs"]):
            if calls["n"] == 1 and j == 0:
                desc = make_placement_descriptor(placement_id=spec.placement_index)
                combined.append(Atoms("H"))
                ids.append(spec.placement_index)
                descs.append(desc)
            else:
                failures.append(
                    PlacementFailureEvent(
                        placement_id=spec.placement_index,
                        stage="generation",
                        reason="too_close",
                        descriptor=None,
                    )
                )
        return combined, ids, descs, failures

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=10,
        placement_retry_enabled=True,
        placement_retry_max_attempts=8,
        placement_retry_early_stop_patience=2,
        seed=0,
    )
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    # 1 successful attempt + 2 zero-yield attempts -> early stop at 3.
    assert result.n_attempts == 3
    assert len(result.combined) == 1


def test_cell_tracking_skips_retried_cells(monkeypatch):
    """R3: the same discrete cell is never materialized more than once across
    retry attempts (the failed-key/partial-unblock filters keep it excluded)."""
    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.placement_fill import placement_cell_key
    from metalsurfer.workflow.shared import PlacementFailureEvent

    # Large fixed pool of distinct cells so the pool never exhausts (no fallback
    # re-materialization); check_cells must then keep every cell in <=1 batch.
    POOL_SIZE = 200
    pool = [
        PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=i,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=i,
        )
        for i in range(POOL_SIZE)
    ]

    seen_cells: list[list] = []

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
    ):
        specs = [s for s in pool if filter_spec is None or filter_spec(s)]
        return specs[:n_desired]

    def spy_materialize(**kwargs):
        specs = kwargs["specs"]
        seen_cells.append([placement_cell_key(s) for s in specs])
        failures = [
            PlacementFailureEvent(
                placement_id=s.placement_index,
                stage="generation",
                reason="too_close",
                descriptor=None,
            )
            for s in specs
        ]
        return [], [], [], failures

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", spy_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=10,
        placement_retry_enabled=True,
        placement_retry_max_attempts=3,
        placement_retry_early_stop_patience=100,
        seed=0,
    )
    fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    # No discrete cell may be materialized in more than one attempt's batch.
    seen_per_cell: dict = {}
    for batch_idx, batch in enumerate(seen_cells):
        for cell in batch:
            assert cell not in seen_per_cell or seen_per_cell[cell] == batch_idx, (
                f"cell {cell} materialized in batches "
                f"{seen_per_cell.get(cell)} and {batch_idx}"
            )
            seen_per_cell[cell] = batch_idx


def test_pool_empty_partial_unblock(monkeypatch, caplog):
    """R4: failed-key filter empties pool -> partial unblock (filter kept),
    with unfiltered fallback only as a last resort."""
    import logging

    from metalsurfer.workflow import placement_fill as fill_mod
    from metalsurfer.workflow.shared import PlacementFailureEvent

    known_specs = [
        PlacementSpec(
            conformer_index=0,
            orientation_type="round",
            face_flip=False,
            en_atom_index=None,
            site_index=3,
            site_type="hollow",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=i,
        )
        for i in range(3)
    ]
    known = known_specs[0]
    # Per enumerate call, record whether the failed-key block is still active.
    filter_accepts_known: list[bool] = []

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
    ):
        if filter_spec is not None:
            filter_accepts_known.append(bool(filter_spec(known)))
        else:
            filter_accepts_known.append(True)
        return [s for s in known_specs if filter_spec is None or filter_spec(s)]

    def fake_materialize(**kwargs):
        failures = [
            PlacementFailureEvent(
                placement_id=spec.placement_index,
                stage="generation",
                reason="too_close",
                descriptor=None,
            )
            for spec in kwargs["specs"]
        ]
        return [], [], [], failures

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    config = AdsorptionConfig(
        material_type="slab",
        num_placements=3,
        placement_retry_enabled=True,
        placement_retry_max_attempts=4,
        placement_retry_early_stop_patience=2,
        seed=0,
    )
    with caplog.at_level(logging.WARNING):
        fill_mod.fill_materialized_placements(
            conformers=[make_water()],
            slab_for_sites=make_slab(),
            config=config,
            smiles="O",
            site_context=None,
            slab_atoms=make_slab(),
            calculator=None,
        )
    # First call: block empty -> accepts. Then pool-empty: partial unblock keeps
    # the failed-key block (rejects), and only the last-resort fallback drops it.
    assert filter_accepts_known[0] is True
    rejecting = [i for i, v in enumerate(filter_accepts_known) if not v]
    accepting_after = [i for i, v in enumerate(filter_accepts_known) if v]
    assert rejecting, "expected a partial-unblock re-enumeration that keeps the filter"
    assert max(rejecting) < max(accepting_after), (
        "unfiltered fallback should be a last resort after the partial unblock"
    )
    assert any("still empty after partial unblock" in r.message for r in caplog.records)


def test_clamp_flag_false_legacy(monkeypatch):
    """Legacy behavior: with clamp disabled, the full target is attempted
    even when enumerable capacity is far smaller."""
    from metalsurfer.workflow import placement_fill as fill_mod

    CAPACITY = 2

    def fake_estimate(*args, **kwargs):
        return float(CAPACITY)

    monkeypatch.setattr(fill_mod, "estimate_molecule_complexity", fake_estimate)

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
    ):
        specs = []
        for i in range(n_desired):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=i,
                site_type="atop",
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=i,
            )
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)
        return specs

    def fake_materialize(**kwargs):
        combined, ids, descs = [], [], []
        for spec in kwargs["specs"]:
            desc = make_placement_descriptor(placement_id=spec.placement_index)
            combined.append(Atoms("H"))
            ids.append(spec.placement_index)
            descs.append(desc)
        return combined, ids, descs, []

    monkeypatch.setattr(fill_mod, "enumerate_placement_specs", fake_enumerate)
    monkeypatch.setattr(fill_mod, "_materialize_spec_placements", fake_materialize)

    n_target = 10
    config = AdsorptionConfig(
        material_type="slab",
        num_placements=n_target,
        placement_retry_enabled=True,
        placement_retry_max_attempts=8,
        placement_fill_clamp_to_capacity=False,
        seed=0,
    )
    result = fill_mod.fill_materialized_placements(
        conformers=[make_water()],
        slab_for_sites=make_slab(),
        config=config,
        smiles="O",
        site_context=None,
        slab_atoms=make_slab(),
        calculator=None,
    )
    assert len(result.combined) == n_target
    assert len(result.combined) > CAPACITY


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

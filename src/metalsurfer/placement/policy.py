"""Sampler policy for batch-based BO placement spec proposals."""

import itertools
import random
from collections.abc import Callable

from ..models import PlacementSpec

TILT_FULL = [0.0, 15.0, 30.0, 45.0, 60.0, 90.0]
TILT_PARALLEL = [0.0, 15.0, 30.0]
AZIMUTH = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
AZIMUTH_IN_PLANE = [0.0, 90.0, 180.0, 270.0]
Z_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 0.9]


def _normalized_site_indices(site_indices: list[int]) -> list[int]:
    return site_indices if site_indices else [-1]


def max_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    shape: str,
    n_binders: int,
    flat_aromatic: bool,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
) -> int:
    """Return total number of specs in the policy search grid.

    Delegates to :func:`build_batch_placement_specs` with an uncapped budget so
    that the count is always consistent with what the builder actually produces,
    without duplicating the combinatorial arithmetic.
    """
    return len(
        build_batch_placement_specs(
            n_conformers=n_conformers,
            site_indices=site_indices,
            site_type_for_index=lambda _: None,
            shape=shape,
            n_binders=n_binders,
            flat_aromatic=flat_aromatic,
            parallel_fraction=0.5,
            n_desired=10**9,
            dissociative=dissociative,
            n_hollow_pairs=n_hollow_pairs,
        )
    )


def build_batch_placement_specs(
    *,
    n_conformers: int,
    site_indices: list[int],
    site_type_for_index: Callable[[int], str | None],
    shape: str,
    n_binders: int,
    flat_aromatic: bool,
    parallel_fraction: float,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    dissociative: bool = False,
    n_hollow_pairs: int = 0,
    seed: int | None = None,
) -> list[PlacementSpec]:
    """Build stratified BO candidate specs from policy priors.

    When *seed* is not None and the full combinatorial grid exceeds
    *n_desired*, specs are sampled uniformly at random (seeded) instead
    of being truncated lexicographically.  This ensures coverage across
    all dimensions (conformers, tilts, azimuths, sites, z-fractions)
    even with small budgets.

    Dissociative specs are always generated sequentially (grid is small).
    When *seed* is None the legacy sequential truncation is used, which
    preserves backward compatibility for :func:`max_batch_placement_specs`.
    """
    normalized_sites = _normalized_site_indices(site_indices)

    def make_spec(
        placement_index: int,
        conf_idx: int,
        orient: str,
        face_flip: bool,
        en_idx: int | None,
        site_idx: int,
        tilt: float,
        az: float,
        az_ip_val: float,
        zf_val: float,
    ) -> PlacementSpec:
        return PlacementSpec(
            conformer_index=conf_idx,
            orientation_type=orient,  # sampler metadata, not geometry semantics
            face_flip=face_flip,
            en_atom_index=en_idx,
            site_index=site_idx,
            site_type=site_type_for_index(site_idx),
            tilt_deg=tilt,
            azimuth_deg=az,
            azimuth_in_plane_deg=az_ip_val,
            z_fraction=zf_val,
            placement_index=placement_index,
        )

    specs: list[PlacementSpec] = []
    pid = 0

    def append_from_iter(
        items,
        build_spec: Callable[..., PlacementSpec],
        *,
        target_size: int,
    ) -> None:
        nonlocal pid
        for item in items:
            if len(specs) >= target_size:
                break
            spec = build_spec(pid, *item)
            pid += 1
            if filter_spec is None or filter_spec(spec):
                specs.append(spec)

    # --- Dissociative: small grid, always fully enumerated, early return ---
    if dissociative:
        pair_indices = list(range(max(n_hollow_pairs, 1)))
        items = itertools.product(pair_indices, Z_FRACTIONS)
        append_from_iter(
            items,
            lambda placement_index, pair_idx, zfv: make_spec(
                placement_index,
                0,
                "dissociative",
                False,
                None,
                pair_idx,
                0.0,
                0.0,
                0.0,
                zfv,
            ),
            target_size=n_desired,
        )
        return specs[:n_desired]

    # For non-dissociative paths: when seed is set, generate the full grid
    # then subsample; when seed is None, use sequential truncation (legacy).
    use_subsampling = seed is not None
    effective_cap = 10**9 if use_subsampling else n_desired

    if flat_aromatic:
        n_par = max(1, int(n_desired * parallel_fraction))
        n_en = max(1, n_desired - n_par)
        par_cap = 10**9 if use_subsampling else n_par
        en_cap = 10**9 if use_subsampling else (n_par + n_en)

        parallel_items = itertools.product(
            range(n_conformers),
            [False, True],
            TILT_PARALLEL,
            AZIMUTH,
            Z_FRACTIONS,
            AZIMUTH_IN_PLANE,
            normalized_sites,
        )
        # Collect parallel specs into their own list for separate subsampling
        par_specs: list[PlacementSpec] = []

        def _append_par(
            items_iter,
            build_fn: Callable[..., PlacementSpec],
            cap: int,
        ) -> None:
            nonlocal pid
            for item in items_iter:
                if len(par_specs) >= cap:
                    break
                spec = build_fn(pid, *item)
                pid += 1
                if filter_spec is None or filter_spec(spec):
                    par_specs.append(spec)

        _append_par(
            parallel_items,
            lambda placement_index, ci, ff, tl, azv, zfv, aip, si: make_spec(
                placement_index,
                ci,
                "parallel",
                ff,
                None,
                si,
                tl,
                azv,
                aip,
                zfv,
            ),
            par_cap,
        )

        en_down_items = itertools.product(
            range(n_conformers),
            range(max(n_binders, 1)),
            TILT_FULL,
            AZIMUTH,
            Z_FRACTIONS,
            normalized_sites,
        )
        en_specs: list[PlacementSpec] = []

        def _append_en(
            items_iter,
            build_fn: Callable[..., PlacementSpec],
            cap: int,
        ) -> None:
            nonlocal pid
            for item in items_iter:
                if len(en_specs) >= cap:
                    break
                spec = build_fn(pid, *item)
                pid += 1
                if filter_spec is None or filter_spec(spec):
                    en_specs.append(spec)

        _append_en(
            en_down_items,
            lambda placement_index, ci, ei, tl, azv, zfv, si: make_spec(
                placement_index,
                ci,
                "EN-down",
                False,
                ei if n_binders > 1 else None,
                si,
                tl,
                azv,
                0.0,
                zfv,
            ),
            en_cap,
        )

        if use_subsampling:
            rng = random.Random(seed)
            if len(par_specs) > n_par:
                par_specs = rng.sample(par_specs, n_par)
            if len(en_specs) > n_en:
                en_specs = rng.sample(en_specs, n_en)
        else:
            par_specs = par_specs[:n_par]
            en_specs = en_specs[:n_en]

        specs = par_specs + en_specs
    else:
        orient = "vertical" if shape == "linear" else "round"
        items = itertools.product(
            range(n_conformers),
            TILT_FULL,
            AZIMUTH,
            Z_FRACTIONS,
            normalized_sites,
        )
        append_from_iter(
            items,
            lambda placement_index, ci, tl, azv, zfv, si: make_spec(
                placement_index,
                ci,
                orient,
                False,
                None,
                si,
                tl,
                azv,
                0.0,
                zfv,
            ),
            target_size=effective_cap,
        )

        if use_subsampling and len(specs) > n_desired:
            specs = random.Random(seed).sample(specs, n_desired)

    # Re-index placement_index after subsampling for monotonic ordering
    for i, spec in enumerate(specs):
        spec.placement_index = i

    return specs[:n_desired]

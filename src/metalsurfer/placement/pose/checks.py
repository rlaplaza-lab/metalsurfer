"""Pose validation and slab distance scratch helpers."""

import logging

import numpy as np
from ase import Atoms

from ...config import AdsorptionConfig
from ...filters import adsorbates_mutually_disconnected
from .. import geometry as geom
from .._material import material_aware_pbc

logger = logging.getLogger(__name__)


def _saturation_exclude_count(
    slab: Atoms,
    slab_for_sites: Atoms | None,
) -> int | None:
    """Prefix length of substrate atoms when *slab* has pre-adsorbed suffix."""
    if slab_for_sites is None:
        return None
    n = len(slab_for_sites)
    if n >= len(slab):
        return None
    return n


def _build_slab_distance_scratch(
    slab: Atoms,
    exclude_n: int | None,
    mat_type: str,
) -> geom._SlabDistanceScratch:
    """Build the invariant slab-side slice used across candidate validations."""
    slab_syms = list(slab.get_chemical_symbols())
    if exclude_n is not None:
        slab_pos = np.asarray(slab.get_positions()[:exclude_n], dtype=float)
        slab_syms = slab_syms[:exclude_n]
        pre_ads_pos = np.asarray(slab.get_positions()[exclude_n:], dtype=float)
    else:
        slab_pos = np.asarray(slab.get_positions(), dtype=float)
        pre_ads_pos = None
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(mat_type)
    slab_cov_r = geom.radii_array(slab_syms, kind="covalent")
    slab_vdw_r = geom.radii_array(slab_syms, kind="vdw")
    return geom._SlabDistanceScratch(
        slab_pos=slab_pos,
        cell=cell,
        pbc=pbc,
        slab_syms=slab_syms,
        slab_cov_r=slab_cov_r,
        slab_vdw_r=slab_vdw_r,
        pre_ads_pos=pre_ads_pos,
    )


def _validate_posed_adsorbate(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
    material_type: str | None = None,
    slab_scratch: geom._SlabDistanceScratch | None = None,
) -> str | None:
    """Run distance, adsorbate–adsorbate disconnect, and optional contact-quality checks.

    Order:

    1. Always: covalent floor ``max(min_initial_distance, covalent_sum *
       min_contact_ratio)``, optional ``max_initial_distance``, then optional
       van der Waals overlap when ``reject_vdw_overlaps`` is set.
    2. Under coverage: adsorbate–adsorbate disconnection via the shared
       connectivity rule (``connectivity_multiplier``).
    3. Only when ``strict_initial_placement`` or ``require_multiple_contact``:
       contact quality (closest approach, contact count, then variance).

    Returns a failure reason token, or ``None`` when the placement is accepted.
    *material_type* defaults to ``config.material_type``; callers with a resolved
    placement context should pass ``ctx.mat_type``.

    The mol↔slab MIC distance matrix is computed **once** and reused for the
    distance gate and the contact-quality gate.  When *slab_scratch* is provided,
    the slab side is reused from it instead of re-slicing the ASE ``Atoms``.
    """
    mat_type = material_type if material_type is not None else config.material_type
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    if slab_scratch is None:
        slab_scratch = _build_slab_distance_scratch(slab, exclude_n, mat_type)

    _mol_pos, _slab_pos, _mol_syms, _slab_syms, _cell, _pbc, dists = (
        geom._mol_slab_contact_arrays(
            adsorbate,
            slab,
            material_type=mat_type,
            exclude_slab_atoms=exclude_n,
            slab_scratch=slab_scratch,
        )
    )
    ok, _, dist_reason = geom.check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
        reject_vdw_overlaps=config.reject_vdw_overlaps,
        vdw_overlap_scale=config.vdw_overlap_scale,
        exclude_slab_atoms=exclude_n,
        material_type=mat_type,
        pairwise_distances=dists,
        slab_scratch=slab_scratch,
    )
    if not ok:
        return dist_reason

    if exclude_n is not None and exclude_n < len(slab):
        pre_ads = slab[exclude_n:]
        if len(pre_ads) > 0 and not adsorbates_mutually_disconnected(
            adsorbate,
            pre_ads,
            float(config.connectivity_multiplier),
            material_type=mat_type,
            cell=np.asarray(slab.get_cell(), dtype=float),
        ):
            return "adsorbate_overlap"

    if config.strict_initial_placement or config.require_multiple_contact:
        contact_ok, contact_reason = geom.check_initial_contact_quality(
            adsorbate,
            slab,
            strict_initial_placement=config.strict_initial_placement,
            require_multiple_contact=config.require_multiple_contact,
            max_closest_approach=float(config.max_closest_approach),
            min_contact_atoms=int(config.min_contact_atoms),
            contact_distance_threshold=config.contact_distance_threshold,
            exclude_slab_atoms=exclude_n,
            material_type=mat_type,
            pairwise_distances=dists,
        )
        if not contact_ok:
            return contact_reason

    return None

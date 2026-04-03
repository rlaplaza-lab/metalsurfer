"""Placement generation logic for adsorbate placement on slab surfaces."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..exceptions import DependencyMissingError
from ..models import PlacementDescriptor, PlacementPose, PlacementSpec
from . import geometry as geom
from . import policy
from . import sites as sts
from ._constants import (
    _DISSOCIATIVE_MAX_ADJACENT_SEP,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP,
    _PARALLEL_Z_FLOOR_ANGSTROM,
    _PARALLEL_Z_HI_SHRINK,
    _PARALLEL_Z_LO_SHRINK,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _SITE_Z_OFFSETS,
)

logger = logging.getLogger(__name__)


@dataclass
class SiteContext:
    """Cached result of Voronoi site detection for a given slab geometry."""

    sites: list[dict[str, object]]
    use_sites: bool
    source: str


@dataclass
class _PoseResult:
    """Internal return type of _pose_from_spec(); named fields replace raw 7-tuple."""

    pose: PlacementPose
    surface_ref: float
    z_offset: float
    z_fraction: float
    site: dict[str, object] | None
    source: str
    site_reference_frame: str


def _is_flat_aromatic(
    shape: str,
    smiles: str | None,
    symbols: list[str],
) -> bool:
    """True if the adsorbate is flat with aromatic EN atoms (parallel-placement candidate)."""
    if shape != "flat":
        return False
    binders = geom._binding_atom_candidates(symbols)
    if smiles is not None:
        return _is_flat_aromatic_with_en(smiles)
    return bool(binders)


def _is_flat_aromatic_with_en(smiles: str) -> bool:
    """True if molecule has aromatic rings and electronegative (binding) atoms."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyMissingError(
            "rdkit",
            "flat aromatic detection for placement",
            "pip install rdkit",
        ) from exc
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    binders = {"O", "N", "S", "F", "Cl", "Br", "I"}
    has_en = any(a.GetSymbol() in binders for a in mol.GetAtoms())
    return bool(aromatic and has_en)


def classify_adsorbate_orientation(
    atoms: Atoms, slab_size: int, threshold: float = 0.7
) -> str:
    """Classify adsorbate as 'parallel' or 'EN-down' from inertia (plane normal vs surface).

    For flat molecules, the plane normal is the axis of largest inertia (eigenvecs[:, 2]),
    per the perpendicular axis theorem. Parallel = ring approximately horizontal;
    EN-down = ring tilted, electronegative atom toward surface.
    """
    pos = atoms.get_positions()[slab_size:]
    if len(pos) < 3:
        return "unknown"
    masses = atoms.get_masses()[slab_size:]
    _, eigenvecs = geom._compute_inertia_tensor(pos, masses)
    plane_normal = eigenvecs[:, 2]
    if plane_normal[2] < 0:
        plane_normal = -plane_normal
    dot = abs(float(np.dot(plane_normal, np.array([0.0, 0.0, 1.0]))))
    return "parallel" if dot > threshold else "EN-down"


def _get_unique_sites_for_specs(
    slab: Atoms,
    config: AdsorptionConfig,
) -> SiteContext:
    """Get unique non-identical sites using unified Voronoi detection.

    Works for slabs, nanoparticles, and porous materials.
    Returns ``SiteContext(sites=[], use_sites=False, source="no_sites")`` when
    site detection yields nothing.
    """
    mat_type = config.material_type
    probe_radius = config.voronoi_probe_radius
    max_site_dist = config.voronoi_max_site_distance

    if len(slab) < 4:
        logger.info(
            "Slab has fewer than 4 atoms (%d); cannot detect adsorption sites",
            len(slab),
        )
        return SiteContext(sites=[], use_sites=False, source="no_sites")

    raw_sites = sts.get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_dist,
        top_layer_tolerance=config.top_layer_tolerance,
        material_type=mat_type,
        enrich=config.voronoi_site_enrichment,
    )
    if not raw_sites:
        logger.info(
            "Unified Voronoi site detection found no sites for %d-atom structure "
            "(probe_radius=%.2f, max_distance=%.2f, material_type=%r)",
            len(slab),
            probe_radius,
            max_site_dist,
            mat_type,
        )
        return SiteContext(sites=[], use_sites=False, source="no_sites")

    cell = np.array(slab.get_cell())
    unique_sites = sts._cluster_equivalent_sites(
        raw_sites,
        cell,
        tolerance=config.site_equivalence_tolerance,
    )
    if not unique_sites:
        logger.info(
            "Site clustering eliminated all %d raw sites for %d-atom structure "
            "(tolerance=%.3f, material_type=%r)",
            len(raw_sites),
            len(slab),
            config.site_equivalence_tolerance,
            mat_type,
        )
        return SiteContext(sites=[], use_sites=False, source="no_sites")

    source = str(unique_sites[0].get("site_source", "voronoi"))
    return SiteContext(sites=unique_sites, use_sites=True, source=source)


def _resolve_site_context(
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None = None,
) -> SiteContext:
    if site_context is not None:
        return site_context
    return _get_unique_sites_for_specs(slab, config)


def _resolve_surface_ref(
    site: dict[str, object] | None,
    slab: Atoms,
    mat_type: str,
) -> tuple[float, bool]:
    """Return *(surface_ref_z, is_local_ref)* for z-offset calculations.

    For slabs the reference is the topmost atomic z so that z_offset is the
    gap above the surface layer.  For nanoparticles and porous materials the
    Voronoi vertex z IS the surface reference (local).
    """
    if mat_type == "slab":
        return float(np.max(slab.get_positions()[:, 2])), False
    if site is not None and "xyz" in site:
        return float(site["xyz"][2]), True
    if site is not None and "z" in site:
        return float(site["z"]), True
    return float(np.max(slab.get_positions()[:, 2])), False


def _pose_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    z_fraction: float | None = None,
    xy_override: tuple[float, float] | None = None,
    site_context: SiteContext | None = None,
) -> _PoseResult | None:
    """Build a universal PlacementPose from a PlacementSpec.

    Returns ``None`` when no adsorption site is available and no
    ``xy_override`` was given (the caller should treat this as a placement
    failure with reason ``"no_sites_found"``).
    """
    ads_pos = adsorbate.get_positions().copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
    normal = np.array([0.0, 0.0, 1.0])  # default; may be overridden for pore sites
    shape, _, _ = geom._classify_molecule_shape(canonical_pos)

    _ctx = _resolve_site_context(slab, config, site_context)
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    resolved_source = _ctx.source
    site = (
        unique_sites[spec.site_index]
        if use_sites and 0 <= spec.site_index < len(unique_sites)
        else None
    )

    # Determine placement normal (local surface normal for pore/NP sites)
    if site is not None and "normal" in site:
        site_normal = np.asarray(site["normal"], dtype=float)
        if np.linalg.norm(site_normal) > 1e-8:
            normal = site_normal / np.linalg.norm(site_normal)

    mat_type = (
        str(site["material_type"]) if site and "material_type" in site else "slab"
    )

    if xy_override is not None:
        x, y = xy_override[0], xy_override[1]
    elif use_sites and spec.site_index >= 0 and spec.site_index < len(unique_sites):
        site_xy = np.asarray(unique_sites[spec.site_index]["xy"])
        x, y = float(site_xy[0]), float(site_xy[1])
    else:
        # No sites and no override: cannot place at a physically motivated
        # position.  Return a degenerate pose so the caller can report a
        # clean failure instead of placing at a random location.
        logger.debug(
            "No sites available for spec placement_index=%d; returning degenerate pose",
            spec.placement_index,
        )
        return None

    z_base_lo, z_base_hi = sts._compute_site_z_base(config, slab, site, symbols)
    if site and spec.site_type and spec.site_type in _SITE_Z_OFFSETS:
        offset = _SITE_Z_OFFSETS[spec.site_type]
        z_base_lo += offset
        z_base_hi += offset

    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)
    # Only apply parallel z-floor for slab/NP; pore geometry already constrains placement
    if flat_aromatic and spec.orientation_type == "parallel" and mat_type != "porous":
        z_base_lo = max(_PARALLEL_Z_FLOOR_ANGSTROM, z_base_lo - _PARALLEL_Z_LO_SHRINK)
        z_base_hi = max(
            z_base_lo + _PARALLEL_Z_MIN_HI_MARGIN, z_base_hi - _PARALLEL_Z_HI_SHRINK
        )

    zf = spec.z_fraction if z_fraction is None else z_fraction
    z_offset = z_base_lo + zf * (z_base_hi - z_base_lo)

    # Surface reference: the datum z from which z_offset is measured.
    # For slabs the reference is the top of the surface layer – Voronoi
    # vertices sit *between* layers, so using their z would place the
    # adsorbate inside the slab.  For nanoparticles and porous materials
    # the vertex z IS on the surface / inside the pore.
    surface_ref, is_local_ref = _resolve_surface_ref(site, slab, mat_type)

    if spec.orientation_type == "parallel":
        base_pos = geom._flat_orientation_from_principal_axis(
            canonical_pos,
            normal,
            azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
            face_flip=spec.face_flip,
        )
    else:
        base_pos = geom._surface_aligned_rotation(
            canonical_pos,
            normal,
            0,
            symbols,
            en_atom_index=spec.en_atom_index,
        )
    rotated_pos = geom._rotation_with_tilt(
        base_pos, normal, spec.tilt_deg, spec.azimuth_deg
    )
    rot_mat = geom.best_fit_rotation(canonical_pos, rotated_pos)
    quat = geom.rotation_matrix_to_quaternion(rot_mat)
    if mat_type != "slab" and site is not None and "xyz" in site:
        placement_center = (
            np.asarray(site["xyz"], dtype=float) + float(z_offset) * normal
        )
    else:
        placement_center = np.array([float(x), float(y), float(surface_ref + z_offset)])

    pose = PlacementPose(
        conformer_index=spec.conformer_index,
        site_index=spec.site_index,
        site_type=spec.site_type,
        placement_index=spec.placement_index,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
        x_abs=float(placement_center[0]),
        y_abs=float(placement_center[1]),
        z_fraction=float(zf),
        z_abs=float(placement_center[2]),
        orientation_type=spec.orientation_type,
        face_flip=spec.face_flip,
        en_atom_index=spec.en_atom_index,
        tilt_deg=spec.tilt_deg,
        azimuth_deg=spec.azimuth_deg,
        azimuth_in_plane_deg=spec.azimuth_in_plane_deg,
    )
    return _PoseResult(
        pose=pose,
        surface_ref=float(surface_ref),
        z_offset=float(z_offset),
        z_fraction=float(zf),
        site=site,
        source=resolved_source if use_sites else "no_sites",
        site_reference_frame="local_site" if is_local_ref else "global_top_layer",
    )


def _apply_pose(
    pose: PlacementPose,
    adsorbate: Atoms,
    canonical_pos: np.ndarray,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    site: dict[str, object] | None,
    mat_type: str,
    surface_ref: float,
    is_local_ref: bool,
    resolved_source: str,
    use_sites: bool,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Core pose application: rotate, translate, validate, build descriptor.

    Accepts pre-resolved geometry values so the caller avoids redundant
    Voronoi look-ups and canonical-frame recomputations.
    """
    raw_q = np.array([pose.quat_w, pose.quat_x, pose.quat_y, pose.quat_z], dtype=float)
    if float(np.linalg.norm(raw_q)) < 1e-10:
        logger.debug(
            "Degenerate quaternion (norm < 1e-10) for placement_index=%d; skipping",
            pose.placement_index,
        )
        return None
    quat = geom.normalize_quaternion(raw_q)
    test = (geom.quaternion_to_rotation_matrix(quat) @ canonical_pos.T).T
    test[:, 0] += pose.x_abs
    test[:, 1] += pose.y_abs

    if pose.z_abs is None:
        logger.debug("Pose replay requires z_abs for deterministic reconstruction")
        return None
    z_abs = float(pose.z_abs)

    if mat_type != "slab" and site is not None and "xyz" in site and "normal" in site:
        site_xyz = np.asarray(site["xyz"], dtype=float)
        site_normal = np.asarray(site["normal"], dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        if nrm > 1e-8:
            site_normal = site_normal / nrm
            displacement = (
                np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float) - site_xyz
            )
            z_offset = float(np.dot(displacement, site_normal))
        else:
            z_offset = float(z_abs - surface_ref)
    else:
        z_offset = float(z_abs - surface_ref)

    test[:, 2] += z_abs
    adsorbate.set_positions(test)
    ok, _ = geom.check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
    )
    if not ok:
        return None

    slab_indices = (
        tuple(site["slab_indices"])
        if site is not None and "slab_indices" in site
        else None
    )
    inv_2d = np.linalg.inv(np.array(slab.get_cell())[:2, :2])
    xy_frac = (inv_2d @ np.array([pose.x_abs, pose.y_abs])) % 1.0
    descriptor = PlacementDescriptor(
        conformer_index=pose.conformer_index,
        orientation_type=pose.orientation_type or "round",
        face_flip=pose.face_flip,
        en_atom_index=pose.en_atom_index,
        site_index=pose.site_index,
        site_type=pose.site_type,
        tilt_deg=pose.tilt_deg,
        azimuth_deg=pose.azimuth_deg,
        azimuth_in_plane_deg=pose.azimuth_in_plane_deg,
        z_fraction=pose.z_fraction,
        placement_index=pose.placement_index,
        x=float(pose.x_abs),
        y=float(pose.y_abs),
        z_offset=float(z_offset),
        x_abs=float(pose.x_abs),
        y_abs=float(pose.y_abs),
        surface_ref_z_abs=surface_ref,
        z_abs=float(z_abs),
        shape=geom._classify_molecule_shape(canonical_pos)[0],
        slab_indices=slab_indices,
        placement_mode_resolved="sites",
        site_source=resolved_source if use_sites else "no_sites",
        site_reference_frame="local_site" if is_local_ref else "global_top_layer",
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
    )
    return adsorbate, descriptor


def generate_placement_from_pose(
    pose: PlacementPose,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement using universal pose semantics."""
    if not conformers:
        return None
    if pose.conformer_index < 0 or pose.conformer_index >= len(conformers):
        logger.debug(
            "Invalid conformer_index=%d for %d conformers",
            pose.conformer_index,
            len(conformers),
        )
        return None
    if not _is_finite_number(pose.x_abs) or not _is_finite_number(pose.y_abs):
        logger.debug("Pose must provide finite x_abs and y_abs")
        return None
    if pose.z_abs is not None and not _is_finite_number(pose.z_abs):
        logger.debug("Pose z_abs must be finite when provided")
        return None

    adsorbate = conformers[pose.conformer_index].copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(
        adsorbate.get_positions(), symbols=symbols
    )

    _ctx = _resolve_site_context(slab, config, site_context)
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    resolved_source = _ctx.source
    site = (
        unique_sites[pose.site_index]
        if use_sites and 0 <= pose.site_index < len(unique_sites)
        else None
    )
    mat_type = (
        str(site["material_type"]) if site and "material_type" in site else "slab"
    )
    surface_ref, is_local_ref = _resolve_surface_ref(site, slab, mat_type)
    return _apply_pose(
        pose,
        adsorbate,
        canonical_pos,
        slab,
        config,
        site=site,
        mat_type=mat_type,
        surface_ref=surface_ref,
        is_local_ref=is_local_ref,
        resolved_source=resolved_source,
        use_sites=use_sites,
    )


# ---------------------------------------------------------------------------
# Shared adsorbate/site grid computation used by enumerate / estimate
# ---------------------------------------------------------------------------


def _spec_grid_info(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None,
) -> dict:
    """Compute the spec-enumeration inputs once for both enumerate and estimate."""
    is_dissociative = config.skip_topology_check and _is_dissociable_diatomic(
        conformers[0]
    )
    _ctx = _resolve_site_context(slab, config, site_context)
    unique_sites = _ctx.sites
    use_sites = _ctx.use_sites
    site_indices = (
        list(range(len(unique_sites))) if use_sites and unique_sites else [-1]
    )

    ads_pos = conformers[0].get_positions() - np.mean(
        conformers[0].get_positions(), axis=0
    )
    shape, _, _ = geom._classify_molecule_shape(ads_pos)
    symbols = conformers[0].get_chemical_symbols()
    binders = geom._binding_atom_candidates(symbols)
    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)

    hollow_pairs: list = []
    if is_dissociative:
        hollow_pairs = _get_hollow_site_pairs(slab, config)

    return {
        "is_dissociative": is_dissociative,
        "unique_sites": unique_sites,
        "use_sites": use_sites,
        "site_indices": site_indices,
        "shape": shape,
        "symbols": symbols,
        "n_binders": len(binders),
        "flat_aromatic": flat_aromatic,
        "hollow_pairs": hollow_pairs,
        "n_hollow_pairs": len(hollow_pairs),
    }


def enumerate_placement_specs(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    site_context: SiteContext | None = None,
) -> list[PlacementSpec]:
    """Enumerate placement specs for diverse sampling.

    Builds a stratified set of specs covering conformers, orientation types,
    face flip, electronegative atoms, sites, tilt, and azimuth.
    When ``config.skip_topology_check`` is True and the molecule is a
    dissociable diatomic, dissociative specs are generated instead.
    """
    if not conformers:
        return []

    info = _spec_grid_info(conformers, slab, config, smiles, site_context)
    unique_sites = info["unique_sites"]
    use_sites = info["use_sites"]

    def site_type_for(site_idx: int) -> str | None:
        if info["is_dissociative"]:
            return "hollow"
        if not use_sites or site_idx < 0 or site_idx >= len(unique_sites):
            return None
        return str(unique_sites[site_idx]["site_type"])

    return policy.build_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info["site_indices"],
        site_type_for_index=site_type_for,
        shape=info["shape"],
        n_binders=info["n_binders"],
        flat_aromatic=info["flat_aromatic"],
        parallel_fraction=config.flat_aromatic_parallel_fraction,
        n_desired=n_desired,
        filter_spec=filter_spec,
        dissociative=info["is_dissociative"],
        n_hollow_pairs=info["n_hollow_pairs"],
    )


def estimate_placement_spec_capacity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
) -> int:
    """Estimate total enumerated specs for current conformers/site grid."""
    if not conformers:
        return 0
    info = _spec_grid_info(conformers, slab, config, smiles, site_context)
    return policy.max_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info["site_indices"],
        shape=info["shape"],
        n_binders=info["n_binders"],
        flat_aromatic=info["flat_aromatic"],
        dissociative=info["is_dissociative"],
        n_hollow_pairs=info["n_hollow_pairs"],
    )


def estimate_molecule_complexity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
) -> float:
    """Estimate a placement-space complexity score for one molecule.

    Wraps :func:`estimate_placement_spec_capacity` so that the score reflects
    the true size of the orientation/site search space: number of conformers,
    molecular shape (linear / flat / round), binding-atom count, and available
    adsorption sites.  Returns 1.0 as a floor so molecules always get at least
    a minimal placement budget share.
    """
    capacity = estimate_placement_spec_capacity(
        conformers, slab, config, smiles, site_context=site_context
    )
    return max(1.0, float(capacity))


def distribute_placement_budget(
    complexities: dict[str, float],
    total_budget: int,
) -> dict[str, int]:
    """Distribute *total_budget* placements across molecules proportionally.

    Molecules with larger complexity scores (more orientations / binding atoms /
    conformers) receive more placements because their search space is wider.
    Every molecule is guaranteed at least one placement.  The allocations sum
    to exactly *total_budget* after largest-remainder rounding.

    Parameters
    ----------
    complexities:
        Mapping of ``{molecule_name: complexity_score}`` where each score is a
        positive float (e.g. from :func:`estimate_molecule_complexity`).
    total_budget:
        Total number of placements to distribute.

    Returns
    -------
    dict[str, int]
        Placement counts per molecule summing to *total_budget*.
    """
    if not complexities:
        return {}
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive, got {total_budget}")

    names = list(complexities)
    if total_budget < len(names):
        raise ValueError(
            f"total_budget ({total_budget}) must be >= number of molecules "
            f"({len(names)}); cannot guarantee every molecule at least 1 placement"
        )

    scores = [max(1.0, float(complexities[n])) for n in names]
    total_score = sum(scores)

    # Initial allocation: floor of proportional share, minimum 1
    raw = [max(1, int(s / total_score * total_budget)) for s in scores]
    remainder = total_budget - sum(raw)

    if remainder != 0:
        # Largest-remainder method: sort by fractional excess descending
        fractions = [
            s / total_score * total_budget - max(1, int(s / total_score * total_budget))
            for s in scores
        ]
        order = sorted(range(len(names)), key=lambda i: fractions[i], reverse=True)
        for i in range(abs(remainder)):
            idx = order[i % len(order)]
            raw[idx] += 1 if remainder > 0 else -1
            # Never drop below 1
            if raw[idx] < 1:
                raw[idx] = 1

    return dict(zip(names, raw, strict=True))


def generate_placement_from_spec(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement from spec. Returns (adsorbate, descriptor) or None."""
    result, _ = generate_placement_from_spec_with_reason(
        spec, conformers, slab, config, smiles=smiles, site_context=site_context
    )
    return result


def generate_placement_from_spec_with_reason(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Generate placement from spec and provide a failure reason when unavailable."""
    if not conformers:
        return None, "no_conformers"

    if spec.orientation_type == "dissociative":
        adsorbate = conformers[spec.conformer_index % len(conformers)].copy()
        return _generate_dissociative_placement_from_spec(adsorbate, spec, slab, config)

    # Resolve site context once to avoid redundant Voronoi computation.
    resolved_ctx = _resolve_site_context(slab, config, site_context)

    adsorbate = conformers[spec.conformer_index % len(conformers)].copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(
        adsorbate.get_positions(), symbols=symbols
    )

    pose_result = _pose_from_spec(
        adsorbate, spec, slab, config, smiles, site_context=resolved_ctx
    )
    if pose_result is None:
        return None, "no_sites_found"

    pose = pose_result.pose
    surface_ref = pose_result.surface_ref
    site = pose_result.site
    source = pose_result.source
    is_local_ref = pose_result.site_reference_frame == "local_site"
    use_sites = resolved_ctx.use_sites
    mat_type = (
        str(site["material_type"]) if site and "material_type" in site else "slab"
    )

    result = _apply_pose(
        pose,
        adsorbate,
        canonical_pos,
        slab,
        config,
        site=site,
        mat_type=mat_type,
        surface_ref=surface_ref,
        is_local_ref=is_local_ref,
        resolved_source=source,
        use_sites=use_sites,
    )
    if result is not None:
        return result, None
    return None, "initial_distance_or_site_constraints"


def generate_placement_from_descriptor(
    descriptor: PlacementDescriptor,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
) -> Atoms | None:
    """Reproduce placement deterministically from descriptor."""
    if not conformers:
        return None
    if descriptor.conformer_index < 0 or descriptor.conformer_index >= len(conformers):
        logger.debug(
            "Descriptor conformer_index=%d out of range for %d conformers",
            descriptor.conformer_index,
            len(conformers),
        )
        return None
    if descriptor.x_abs is None or descriptor.y_abs is None or descriptor.z_abs is None:
        logger.debug(
            "Descriptor replay requires x_abs, y_abs, and z_abs; got x_abs=%s y_abs=%s z_abs=%s",
            descriptor.x_abs,
            descriptor.y_abs,
            descriptor.z_abs,
        )
        return None
    if None in (
        descriptor.quat_w,
        descriptor.quat_x,
        descriptor.quat_y,
        descriptor.quat_z,
    ):
        logger.debug("Descriptor replay requires quaternion components")
        return None
    x_abs = float(descriptor.x_abs)
    y_abs = float(descriptor.y_abs)
    zf = float(descriptor.z_fraction)
    quat = np.array(
        [
            float(descriptor.quat_w),
            float(descriptor.quat_x),
            float(descriptor.quat_y),
            float(descriptor.quat_z),
        ],
        dtype=float,
    )
    pose = PlacementPose(
        conformer_index=descriptor.conformer_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        placement_index=descriptor.placement_index,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
        x_abs=x_abs,
        y_abs=y_abs,
        z_fraction=zf,
        z_abs=float(descriptor.z_abs),
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
    )
    result = generate_placement_from_pose(
        pose, conformers, slab, config, site_context=site_context
    )
    if result is None:
        return None
    adsorbate, _ = result
    return adsorbate


def _is_dissociable_diatomic(adsorbate: Atoms) -> bool:
    """True if molecule is a homonuclear diatomic (e.g. H2, O2, N2)."""
    syms = adsorbate.get_chemical_symbols()
    return len(syms) == 2 and syms[0] == syms[1]


def _get_hollow_site_pairs(
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    existing_adsorbate_positions: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return list of (xy1, xy2) hollow-site pairs suitable for dissociative placement."""
    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    cell = slab.get_cell()
    pbc = list(slab.get_pbc())

    hollow_sites = sts.get_hollow_sites_for_adatoms(
        sites_slab,
        top_layer_tolerance=config.top_layer_tolerance,
        dedup_tolerance=config.hollow_site_dedup_tolerance,
    )
    if len(hollow_sites) < 2:
        return []

    # Filter out occupied sites
    if (
        existing_adsorbate_positions is not None
        and len(existing_adsorbate_positions) > 0
    ):
        surface_z = float(np.max(sites_slab.get_positions()[:, 2]))
        available: list[np.ndarray] = []
        for h_xy in hollow_sites:
            site_pos = np.append(h_xy, surface_z)
            d = geom.calculate_min_distance(
                site_pos.reshape(1, 3),
                existing_adsorbate_positions,
                cell=cell,
                pbc=pbc,
            )
            if d >= config.min_initial_distance:
                available.append(h_xy)
        hollow_sites = available
        if len(hollow_sites) < 2:
            return []

    min_fragment_sep = _DISSOCIATIVE_MIN_FRAGMENT_SEP
    max_adjacent_sep = _DISSOCIATIVE_MAX_ADJACENT_SEP
    surface_z = float(np.max(sites_slab.get_positions()[:, 2]))
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(hollow_sites)):
        for j in range(i + 1, len(hollow_sites)):
            xy_i = np.append(hollow_sites[i], surface_z)
            xy_j = np.append(hollow_sites[j], surface_z)
            _, dists = find_mic((xy_i - xy_j).reshape(1, 3), cell, pbc=pbc)
            d = float(dists[0])
            if min_fragment_sep <= d <= max_adjacent_sep:
                pairs.append((hollow_sites[i], hollow_sites[j]))
    return pairs


def _generate_dissociative_placement_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Place fragments at different hollow sites based on spec pair index."""
    if not _is_dissociable_diatomic(adsorbate):
        return None, "not_dissociable_diatomic"

    sites_slab = slab_for_sites if slab_for_sites is not None else slab

    # Compute existing adsorbate positions for occupied site filtering
    existing_ads_pos = None
    if slab_for_sites is not None and len(slab) > len(slab_for_sites):
        existing_ads_pos = slab.get_positions()[len(slab_for_sites) :]

    pairs = _get_hollow_site_pairs(
        slab,
        config,
        slab_for_sites=slab_for_sites,
        existing_adsorbate_positions=existing_ads_pos,
    )
    if not pairs:
        return None, "no_hollow_site_pairs"

    pair_idx = spec.site_index % len(pairs)
    xy1, xy2 = pairs[pair_idx]

    surface_z = float(np.max(sites_slab.get_positions()[:, 2]))
    z_lo, z_hi = config.placement_z_range
    z_offset = z_lo + spec.z_fraction * (z_hi - z_lo)

    syms = adsorbate.get_chemical_symbols()
    pos1 = np.append(xy1, surface_z + z_offset)
    pos2 = np.append(xy2, surface_z + z_offset)

    result = Atoms(symbols=syms, positions=[pos1, pos2])
    result.set_cell(slab.get_cell())
    result.set_pbc(slab.get_pbc())

    ok, _ = geom.check_initial_placement_distance(
        result,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
    )
    if not ok:
        return None, "initial_distance_or_site_constraints"

    centroid = (pos1 + pos2) / 2.0
    inv_2d = np.linalg.inv(np.array(slab.get_cell())[:2, :2])
    xy_frac = (inv_2d @ np.array([centroid[0], centroid[1]])) % 1.0
    descriptor = PlacementDescriptor(
        conformer_index=0,
        orientation_type="dissociative",
        face_flip=False,
        en_atom_index=None,
        site_index=pair_idx,
        site_type="hollow",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        azimuth_in_plane_deg=0.0,
        z_fraction=spec.z_fraction,
        placement_index=spec.placement_index,
        x=float(centroid[0]),
        y=float(centroid[1]),
        z_offset=float(z_offset),
        x_abs=float(centroid[0]),
        y_abs=float(centroid[1]),
        surface_ref_z_abs=surface_z,
        z_abs=float(centroid[2]),
        shape="linear",
        slab_indices=None,
        placement_mode_resolved="sites",
        site_source="dissociative_hollow_pair",
        site_reference_frame="global_top_layer",
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
    )
    return (result, descriptor), None

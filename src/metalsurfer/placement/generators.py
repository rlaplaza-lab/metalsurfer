"""Build placements from specs: sites, orientations, and validation."""

import dataclasses
import hashlib
import logging
import random
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from scipy.spatial import KDTree

from .._utils import is_finite_number as _is_finite_number
from ..config import AdsorptionConfig
from ..exceptions import DependencyMissingError
from ..models import PlacementDescriptor, PlacementPose, PlacementSpec
from ..symmetry import SymmetryAnalysisError
from . import geometry as geom
from . import policy
from . import sites as sts
from ._constants import (
    _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
    _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE,
    _DISTANCE_RECOVERY_HEIGHT_STEPS,
    _DISTANCE_RECOVERY_XY_ATTEMPTS,
    _MOL_COVALENT_RADIUS_FALLBACK,
    _ORIENTATION_CLASSIFICATION_PARALLEL_DOT_THRESHOLD,
    _PARALLEL_FRACTION_HIGH_BINDER_RATIO,
    _PARALLEL_FRACTION_HIGH_RATIO_CUTOFF,
    _PARALLEL_FRACTION_LOW_BINDER_RATIO,
    _PARALLEL_FRACTION_MEDIUM_BINDER_RATIO,
    _PARALLEL_FRACTION_MEDIUM_RATIO_CUTOFF,
    _PARALLEL_FRACTION_NO_BINDERS,
    _PARALLEL_FRACTION_NO_RING,
    _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
    _PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE,
    _PARALLEL_Z_HI_SHRINK_FALLBACK_ANGSTROM,
    _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE,
    _PARALLEL_Z_LO_SHRINK_FALLBACK_ANGSTROM,
    _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _SITE_Z_OFFSET_FROM_SURFACE_RADIUS,
    _VECTOR_NORM_EPS,
    _VORONOI_AUTO_WIDEN_MAX_SCALE,
    _VORONOI_AUTO_WIDEN_PROBE_SCALE,
)
from ._material import material_aware_pbc, material_type_for_placement

logger = logging.getLogger(__name__)


def _rdkit_chem():
    """Return ``rdkit.Chem`` (lazy import for dependency_behavior tests)."""
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise DependencyMissingError(
            "rdkit",
            "placement aromatic and ring heuristics",
            "pip install rdkit",
        ) from exc
    return Chem


@dataclass
class SiteContext:
    """Cached result of Voronoi site detection for a given slab geometry."""

    sites: list[dict[str, object]]
    use_sites: bool
    source: str
    # Pre-clustering output of :func:`sites.get_unified_sites` (same as used for clustering).
    raw_unclustered: list[dict[str, object]] | None = None


_UNIQUE_SITES_CACHE_MAX_ENTRIES = 16
_UNIQUE_SITES_CACHE: dict[str, SiteContext] = {}
_UNIQUE_SITES_CACHE_LOCK = threading.Lock()

# Post-symmetry SiteContext cache (geometry + symmetry_broken); owned here with unique-sites.
_SITE_CONTEXT_CACHE_MAX_ENTRIES = 16
_SITE_CONTEXT_CACHE: dict[int, SiteContext] = {}
_SITE_CONTEXT_CACHE_LOCK = threading.Lock()


def clear_site_caches() -> None:
    """Clear unique-sites and resolved site-context caches."""
    with _UNIQUE_SITES_CACHE_LOCK:
        _UNIQUE_SITES_CACHE.clear()
    with _SITE_CONTEXT_CACHE_LOCK:
        _SITE_CONTEXT_CACHE.clear()


def clear_unique_sites_cache() -> None:
    """Clear cached unique-site and resolved site-context caches (tests / long runs)."""
    clear_site_caches()


def clear_site_context_cache() -> None:
    """Alias for :func:`clear_site_caches` (workflow / test compatibility)."""
    clear_site_caches()


def _unique_sites_cache_key(slab: Atoms, config: AdsorptionConfig) -> str:
    pos_bytes = slab.get_positions().tobytes()
    cell_bytes = np.asarray(slab.get_cell()).tobytes()
    pbc_bytes = str(list(slab.get_pbc())).encode()
    cfg_bytes = (
        str(config.voronoi_probe_radius).encode()
        + str(config.voronoi_max_site_distance).encode()
        + str(config.top_layer_tolerance).encode()
        + str(config.site_equivalence_tolerance).encode()
        + str(config.voronoi_site_enrichment).encode()
        + str(config.voronoi_auto_widen).encode()
        + str(config.site_classification_method).encode()
        + config.material_type.encode()
    )
    return hashlib.sha256(pos_bytes + cell_bytes + pbc_bytes + cfg_bytes).hexdigest()


def _store_unique_sites_cache(cache_key: str, ctx: SiteContext) -> SiteContext:
    with _UNIQUE_SITES_CACHE_LOCK:
        if len(_UNIQUE_SITES_CACHE) >= _UNIQUE_SITES_CACHE_MAX_ENTRIES:
            _UNIQUE_SITES_CACHE.pop(next(iter(_UNIQUE_SITES_CACHE)))
        _UNIQUE_SITES_CACHE[cache_key] = ctx
    return ctx


def resolve_site_context_for_sampling(
    slab_atoms: Atoms,
    config: AdsorptionConfig,
    *,
    symmetry_broken: bool,
) -> SiteContext:
    """Clustered Voronoi sites, then optional spglib orbit reduction unless *symmetry_broken*."""
    pos_bytes = slab_atoms.get_positions().tobytes()
    cell_bytes = np.asarray(slab_atoms.get_cell()).tobytes()
    pbc_bytes = str(list(slab_atoms.get_pbc())).encode()
    cache_key = hash(pos_bytes + cell_bytes + pbc_bytes + str(symmetry_broken).encode())

    with _SITE_CONTEXT_CACHE_LOCK:
        cached = _SITE_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _core_ctx = _get_unique_sites_for_specs(slab_atoms, config)
    core_sites = _core_ctx.sites
    use_sites = _core_ctx.use_sites
    raw_unclustered = _core_ctx.raw_unclustered

    def _site_context(
        sites: list[dict[str, object]],
        source: str,
    ) -> SiteContext:
        return SiteContext(
            sites=sites,
            use_sites=True,
            source=source,
            raw_unclustered=raw_unclustered,
        )

    if not use_sites or not core_sites:
        result = _core_ctx
    elif symmetry_broken:
        logger.debug("Site context: symmetry broken, using clustered Voronoi set")
        result = _site_context(core_sites, "voronoi")
    else:
        try:
            symmetry_aware_sites = sts.get_symmetry_aware_sites(
                slab_atoms,
                top_layer_tolerance=config.top_layer_tolerance,
                symmetry_tolerance=config.symmetry_tolerance,
                material_type=config.material_type,
                probe_radius=config.voronoi_probe_radius,
                max_site_distance=config.voronoi_max_site_distance,
                enrich=config.voronoi_site_enrichment,
                site_classification_method=config.site_classification_method,
                raw_sites=raw_unclustered,
            )
        except SymmetryAnalysisError as exc:
            logger.info(
                "Symmetry site reduction failed; using clustered Voronoi sites (%s)",
                exc,
            )
            symmetry_aware_sites = []

        if symmetry_aware_sites:
            logger.info(
                "Using symmetry-reduced sites (%d sites)", len(symmetry_aware_sites)
            )
            result = _site_context(symmetry_aware_sites, "symmetry_aware")
        else:
            logger.debug("Using clustered Voronoi sites (no symmetry-reduced set)")
            result = _site_context(core_sites, "voronoi")

    with _SITE_CONTEXT_CACHE_LOCK:
        if len(_SITE_CONTEXT_CACHE) >= _SITE_CONTEXT_CACHE_MAX_ENTRIES:
            _SITE_CONTEXT_CACHE.pop(next(iter(_SITE_CONTEXT_CACHE)))
        _SITE_CONTEXT_CACHE[cache_key] = result
    return result


@dataclass
class _PlacementContext:
    """Inputs for ``_finalize_placement``: pose, site/material refs, canonical and rotated positions."""

    pose: PlacementPose
    site: dict[str, object] | None
    mat_type: str
    surface_ref: float
    is_local_ref: bool
    source: str
    canonical_pos: np.ndarray
    use_sites: bool
    rotated_pos: np.ndarray
    z_base_lo: float = 0.0
    z_base_hi: float = 0.0
    normal: np.ndarray | None = None
    site_xy: tuple[float, float] | None = None


@dataclass(frozen=True)
class _DissociativeSitePair:
    xyz1: np.ndarray
    normal1: np.ndarray
    xyz2: np.ndarray
    normal2: np.ndarray


@dataclass
class _SpecGridInfo:
    is_dissociative: bool
    unique_sites: list[dict[str, object]]
    use_sites: bool
    site_indices: list[int]
    shape: str
    symbols: list[str]
    n_binders: int
    flat_aromatic: bool
    n_hollow_pairs: int


def _require_float(value: float | None, *, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _pose_from_descriptor(descriptor: PlacementDescriptor) -> PlacementPose:
    return PlacementPose(
        conformer_index=descriptor.conformer_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        placement_index=descriptor.placement_index,
        quat_w=_require_float(descriptor.quat_w),
        quat_x=_require_float(descriptor.quat_x),
        quat_y=_require_float(descriptor.quat_y),
        quat_z=_require_float(descriptor.quat_z),
        x_abs=_require_float(descriptor.x_abs),
        y_abs=_require_float(descriptor.y_abs),
        z_fraction=descriptor.z_fraction,
        z_abs=descriptor.z_abs,
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
    )


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
    Chem = _rdkit_chem()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    aromatic = any(a.GetIsAromatic() for a in mol.GetAtoms())
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    binder_indices = geom._binding_atom_candidates(symbols)
    binder_set = {symbols[i] for i in binder_indices}
    has_en = any(a.GetSymbol() in binder_set for a in mol.GetAtoms())
    return bool(aromatic and has_en)


def _mean_molecule_covalent_radius(symbols: list[str]) -> float:
    radii = [geom._get_covalent_radius(s) for s in symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        return _MOL_COVALENT_RADIUS_FALLBACK
    return float(np.mean(valid))


def _radius_sum_for_site(
    slab: Atoms,
    site: dict[str, object] | None,
    mol_symbols: list[str],
) -> float | None:
    r_surface = sts._get_site_surface_radii(slab, site)
    if r_surface is None:
        return None
    return r_surface + _mean_molecule_covalent_radius(mol_symbols)


def _site_type_z_offset(
    slab: Atoms,
    site: dict[str, object] | None,
    site_type: str | None,
) -> float:
    if not site_type or site_type not in _SITE_Z_OFFSET_FROM_SURFACE_RADIUS:
        return 0.0
    r_surface = sts._get_site_surface_radii(slab, site)
    if r_surface is None:
        return 0.0
    return _SITE_Z_OFFSET_FROM_SURFACE_RADIUS[site_type] * r_surface


def _parallel_z_adjustments(
    slab: Atoms,
    site: dict[str, object] | None,
    mol_symbols: list[str],
) -> tuple[float, float, float]:
    radius_sum = _radius_sum_for_site(slab, site, mol_symbols)
    if radius_sum is None:
        return (
            _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
            _PARALLEL_Z_LO_SHRINK_FALLBACK_ANGSTROM,
            _PARALLEL_Z_HI_SHRINK_FALLBACK_ANGSTROM,
        )
    return (
        max(
            _PARALLEL_Z_FLOOR_MIN_ANGSTROM,
            _PARALLEL_Z_FLOOR_RADIUS_SUM_SCALE * radius_sum,
        ),
        _PARALLEL_Z_LO_SHRINK_RADIUS_SUM_SCALE * radius_sum,
        _PARALLEL_Z_HI_SHRINK_RADIUS_SUM_SCALE * radius_sum,
    )


def classify_adsorbate_orientation(
    atoms: Atoms,
    slab_size: int,
    threshold: float = _ORIENTATION_CLASSIFICATION_PARALLEL_DOT_THRESHOLD,
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


def _estimate_parallel_fraction(
    symbols: list[str],
    smiles: str | None,
) -> float:
    """Estimate the fraction of placements that should be parallel (π-stacking).

    Returns a value in [0.3, 0.8] based on the ratio of binding atoms
    (electronegative) to ring atoms.  Pure aromatics without binding atoms
    (e.g. benzene) get 0.8; molecules with strong EN-down binders (e.g.
    pyridine, phenol) get 0.3; mixed cases get 0.5.
    """
    binders = geom._binding_atom_candidates(symbols)
    n_binders = len(binders)
    n_ring = 0
    if smiles is not None:
        Chem = _rdkit_chem()
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            n_ring = sum(1 for a in mol.GetAtoms() if a.GetIsAromatic())
    if n_ring == 0:
        n_ring = sum(1 for s in symbols if s == "C")

    if n_binders == 0:
        return _PARALLEL_FRACTION_NO_BINDERS
    if n_ring == 0:
        return _PARALLEL_FRACTION_NO_RING
    ratio = n_binders / n_ring
    if ratio >= _PARALLEL_FRACTION_HIGH_RATIO_CUTOFF:
        return _PARALLEL_FRACTION_HIGH_BINDER_RATIO
    if ratio >= _PARALLEL_FRACTION_MEDIUM_RATIO_CUTOFF:
        return _PARALLEL_FRACTION_MEDIUM_BINDER_RATIO
    return _PARALLEL_FRACTION_LOW_BINDER_RATIO


def _get_unique_sites_for_specs(
    slab: Atoms,
    config: AdsorptionConfig,
) -> SiteContext:
    """Get unique non-identical sites using unified Voronoi detection.

    Works for slabs, nanoparticles, and porous materials.
    Returns ``SiteContext(sites=[], use_sites=False, source="no_sites")`` when
    site detection yields nothing.
    """
    cache_key = _unique_sites_cache_key(slab, config)
    with _UNIQUE_SITES_CACHE_LOCK:
        cached = _UNIQUE_SITES_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if config.material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            "config.material_type must be 'slab', 'nanoparticle', or 'porous', "
            f"got {config.material_type!r}"
        )

    mat_type = config.material_type
    probe_radius = config.voronoi_probe_radius
    max_site_dist = config.voronoi_max_site_distance

    if len(slab) < 4:
        logger.warning(
            "Slab has fewer than 4 atoms (%d); cannot detect adsorption sites",
            len(slab),
        )
        return _store_unique_sites_cache(
            cache_key,
            SiteContext(
                sites=[], use_sites=False, source="no_sites", raw_unclustered=None
            ),
        )

    raw_sites = sts.get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_dist,
        top_layer_tolerance=config.top_layer_tolerance,
        material_type=mat_type,
        enrich=config.voronoi_site_enrichment,
        site_classification_method=config.site_classification_method,
    )
    if not raw_sites and config.voronoi_auto_widen:
        positions = slab.get_positions()
        symbols = list(slab.get_chemical_symbols())
        cell = np.asarray(slab.get_cell(), dtype=float)
        pbc = material_aware_pbc(mat_type)
        derived_probe, derived_max = sts._derive_voronoi_distance_window(
            positions, symbols, np.asarray(pbc, dtype=bool), cell
        )
        eff_probe = float(probe_radius) if probe_radius is not None else derived_probe
        eff_max = float(max_site_dist) if max_site_dist is not None else derived_max
        wide_probe = float(eff_probe * _VORONOI_AUTO_WIDEN_PROBE_SCALE)
        wide_max = float(max(eff_max * _VORONOI_AUTO_WIDEN_MAX_SCALE, wide_probe))
        logger.info(
            "Voronoi auto-widen: retrying site detection with probe=%.3f max=%.3f "
            "(was probe=%.3f max=%.3f)",
            wide_probe,
            wide_max,
            eff_probe,
            eff_max,
        )
        raw_sites = sts.get_unified_sites(
            slab,
            probe_radius=wide_probe,
            max_site_distance=wide_max,
            top_layer_tolerance=config.top_layer_tolerance,
            material_type=mat_type,
            enrich=config.voronoi_site_enrichment,
            site_classification_method=config.site_classification_method,
        )
    if not raw_sites:
        logger.warning(
            "Unified Voronoi site detection found no sites for %d-atom structure "
            "(probe_radius=%s, max_distance=%s, material_type=%r)",
            len(slab),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_dist:.2f}" if max_site_dist is not None else "auto",
            mat_type,
        )
        return _store_unique_sites_cache(
            cache_key,
            SiteContext(
                sites=[], use_sites=False, source="no_sites", raw_unclustered=None
            ),
        )

    cell = np.array(slab.get_cell())
    unique_sites = sts._cluster_equivalent_sites(
        raw_sites,
        cell,
        tolerance=config.site_equivalence_tolerance,
    )
    if not unique_sites:
        logger.warning(
            "Site clustering eliminated all %d raw sites for %d-atom structure "
            "(tolerance=%.3f, material_type=%r)",
            len(raw_sites),
            len(slab),
            config.site_equivalence_tolerance,
            mat_type,
        )
        return _store_unique_sites_cache(
            cache_key,
            SiteContext(
                sites=[],
                use_sites=False,
                source="no_sites",
                raw_unclustered=raw_sites,
            ),
        )

    source = str(unique_sites[0].get("site_source", "voronoi"))
    return _store_unique_sites_cache(
        cache_key,
        SiteContext(
            sites=unique_sites,
            use_sites=True,
            source=source,
            raw_unclustered=raw_sites,
        ),
    )


def _resolve_surface_ref(
    site: dict[str, object] | None,
    slab: Atoms,
    mat_type: str,
    *,
    rough_slab_local_z: bool = False,
) -> tuple[float, bool]:
    """Return *(surface_ref_z, is_local_ref)* for z-offset calculations.

    For slabs the reference is the topmost height along the slab normal so that
    z_offset is the gap above the surface layer.  For nanoparticles and porous
    materials the Voronoi vertex z IS the surface reference (local).

    When *rough_slab_local_z* is True and the slab is non-planar, use the
    site's own height along the normal instead of the global maximum.  This
    prevents step-edge sites from getting excessive z offsets.
    """
    if mat_type == "slab":
        cell = np.asarray(slab.get_cell(), dtype=float)
        positions = slab.get_positions()
        if (
            rough_slab_local_z
            and site is not None
            and not sts._is_top_layer_planar(slab)
        ):
            if "xyz" in site:
                xyz = np.asarray(site["xyz"], dtype=float)
                if xyz.shape == (3,):
                    return float(sts._height_along_slab_normal(xyz, cell)), True
            if "z" in site:
                z_val = site["z"]
                if isinstance(z_val, (int, float, np.floating)):
                    return float(z_val), True
        return float(np.max(sts._height_along_slab_normal(positions, cell))), False
    if site is not None and "xyz" in site:
        xyz_raw = site["xyz"]
        if isinstance(xyz_raw, (list, tuple, np.ndarray)) and len(xyz_raw) >= 3:
            return float(np.asarray(xyz_raw, dtype=float)[2]), True
    if site is not None and "z" in site:
        z_val = site["z"]
        if isinstance(z_val, (int, float, np.floating)):
            return float(z_val), True
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
    slab_for_sites: Atoms | None = None,
) -> _PlacementContext | None:
    """Build a placement context (pose + resolved geometry) from a spec.

    Returns ``None`` when no adsorption site is available and no
    ``xy_override`` was given (the caller should treat this as a placement
    failure with reason ``"no_sites_found"``).
    """
    ads_pos = adsorbate.get_positions().copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(ads_pos, symbols=symbols)
    normal = np.array([0.0, 0.0, 1.0])
    shape, _, _ = geom._classify_molecule_shape(canonical_pos)

    ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
    site = None
    if ctx.use_sites and 0 <= spec.site_index < len(ctx.sites):
        site = ctx.sites[spec.site_index]

    if site is not None and "normal" in site:
        site_normal = np.asarray(site["normal"], dtype=float)
        if np.linalg.norm(site_normal) > _VECTOR_NORM_EPS:
            normal = site_normal / np.linalg.norm(site_normal)

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    if xy_override is not None:
        x, y = xy_override[0], xy_override[1]
    elif site is not None:
        site_xy = np.asarray(site["xy"])
        x, y = float(site_xy[0]), float(site_xy[1])
    else:
        logger.debug(
            "No sites available for spec placement_index=%d; returning None",
            spec.placement_index,
        )
        return None

    placement_reference_slab = slab_for_sites if slab_for_sites is not None else slab

    z_base_lo, z_base_hi = sts._compute_site_z_base(
        config, placement_reference_slab, site, symbols
    )
    if site and spec.site_type:
        offset = _site_type_z_offset(placement_reference_slab, site, spec.site_type)
        z_base_lo += offset
        z_base_hi += offset

    flat_aromatic = _is_flat_aromatic(shape, smiles, symbols)
    if flat_aromatic and spec.orientation_type == "parallel" and mat_type != "porous":
        z_floor, z_lo_shrink, z_hi_shrink = _parallel_z_adjustments(
            placement_reference_slab, site, symbols
        )
        z_base_lo = max(z_floor, z_base_lo - z_lo_shrink)
        z_base_hi = max(
            z_base_lo + _PARALLEL_Z_MIN_HI_MARGIN,
            z_base_hi - z_hi_shrink,
        )

    zf = spec.z_fraction if z_fraction is None else z_fraction
    z_offset = z_base_lo + zf * (z_base_hi - z_base_lo)

    # Slab: top-layer z (Voronoi vertex z can sit between layers). NP/pore: local vertex.
    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        placement_reference_slab,
        mat_type,
        rough_slab_local_z=config.rough_slab_local_z,
    )

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
            symbols,
            en_atom_index=spec.en_atom_index,
        )
    rotated_pos = geom._rotation_with_tilt(
        base_pos, normal, spec.tilt_deg, spec.azimuth_deg
    )
    rot_mat = geom.best_fit_rotation(canonical_pos, rotated_pos)
    quat = geom.rotation_matrix_to_quaternion(rot_mat)

    if mat_type == "slab":
        cell = np.asarray(placement_reference_slab.get_cell(), dtype=float)
        n_hat = sts._slab_normal(cell)
        if site is not None and "xyz" in site:
            base = np.asarray(site["xyz"], dtype=float)
        else:
            base = np.array([float(x), float(y), float(surface_ref)], dtype=float)
        base_h = float(np.dot(base, n_hat))
        target_h = float(surface_ref + z_offset)
        placement_center = base + (target_h - base_h) * n_hat
    elif site is not None and "xyz" in site:
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
    return _PlacementContext(
        pose=pose,
        site=site,
        mat_type=mat_type,
        surface_ref=float(surface_ref),
        is_local_ref=is_local_ref,
        source=ctx.source if ctx.use_sites else "no_sites",
        canonical_pos=canonical_pos,
        use_sites=ctx.use_sites,
        rotated_pos=rotated_pos,
        z_base_lo=float(z_base_lo),
        z_base_hi=float(z_base_hi),
        normal=np.asarray(normal, dtype=float),
        site_xy=(float(x), float(y)),
    )


def _context_from_pose(
    pose: PlacementPose,
    canonical_pos: np.ndarray,
    slab: Atoms,
    config: AdsorptionConfig,
    site_context: SiteContext | None,
) -> _PlacementContext | None:
    """Replay path: normalize quaternion, rotate *canonical_pos*, resolve site for ``_finalize_placement``."""
    raw_q = np.array([pose.quat_w, pose.quat_x, pose.quat_y, pose.quat_z], dtype=float)
    if float(np.linalg.norm(raw_q)) < _VECTOR_NORM_EPS:
        logger.warning(
            "Degenerate quaternion (norm < %.1e) for placement_index=%d; skipping",
            _VECTOR_NORM_EPS,
            pose.placement_index,
        )
        return None
    quat = geom.normalize_quaternion(raw_q)
    rotated_pos = (geom.quaternion_to_rotation_matrix(quat) @ canonical_pos.T).T

    ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
    site = None
    if ctx.use_sites and 0 <= pose.site_index < len(ctx.sites):
        site = ctx.sites[pose.site_index]
    mat_type = material_type_for_placement(site, when_no_site=config.material_type)
    surface_ref, is_local_ref = _resolve_surface_ref(
        site,
        slab,
        mat_type,
        rough_slab_local_z=config.rough_slab_local_z,
    )

    pose_normalized = dataclasses.replace(
        pose,
        quat_w=float(quat[0]),
        quat_x=float(quat[1]),
        quat_y=float(quat[2]),
        quat_z=float(quat[3]),
    )

    return _PlacementContext(
        pose=pose_normalized,
        site=site,
        mat_type=mat_type,
        surface_ref=float(surface_ref),
        is_local_ref=is_local_ref,
        source=ctx.source if ctx.use_sites else "no_sites",
        canonical_pos=canonical_pos,
        use_sites=ctx.use_sites,
        rotated_pos=rotated_pos,
    )


def _recover_z_offset(
    ctx: _PlacementContext, z_abs: float, slab: Atoms | None = None
) -> float:
    """Recover the gap above the surface reference from absolute z.

    For slabs this is the displacement along the slab normal above
    *surface_ref* (height along normal).  For axis-aligned slabs this equals
    ``z_abs - surface_ref``.  For nanoparticle / pore sites with an oriented
    normal we project the displacement onto the local normal.
    """
    pose = ctx.pose
    site = ctx.site
    if ctx.mat_type == "slab" and slab is not None:
        cell = np.asarray(slab.get_cell(), dtype=float)
        n_hat = sts._slab_normal(cell)
        placement = np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float)
        return float(np.dot(placement, n_hat) - ctx.surface_ref)
    if site is not None and "xyz" in site and "normal" in site:
        site_xyz = np.asarray(site["xyz"], dtype=float)
        site_normal = np.asarray(site["normal"], dtype=float)
        nrm = float(np.linalg.norm(site_normal))
        if nrm > _VECTOR_NORM_EPS:
            site_normal = site_normal / nrm
            displacement = (
                np.array([pose.x_abs, pose.y_abs, z_abs], dtype=float) - site_xyz
            )
            return float(np.dot(displacement, site_normal))
    return float(z_abs - ctx.surface_ref)


def _saturation_exclude_count(
    slab: Atoms,
    slab_for_sites: Atoms | None,
) -> int | None:
    """Return substrate atom count for saturation exclude, else None."""
    if slab_for_sites is None:
        return None
    n_sub = len(slab_for_sites)
    if n_sub < len(slab):
        return n_sub
    return None


def _placement_normal(ctx: _PlacementContext, slab: Atoms) -> np.ndarray:
    """Unit normal used for height/lateral recovery."""
    if ctx.mat_type == "slab":
        return sts._slab_normal(np.asarray(slab.get_cell(), dtype=float))
    normal = ctx.normal
    if normal is None:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    normal = np.asarray(normal, dtype=float)
    nrm = float(np.linalg.norm(normal))
    if nrm > _VECTOR_NORM_EPS:
        return normal / nrm
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _center_with_height_delta(
    center: np.ndarray,
    *,
    old_z_fraction: float,
    new_z_fraction: float,
    ctx: _PlacementContext,
    slab: Atoms,
) -> np.ndarray:
    """Shift *center* along the placement normal by the height-window delta."""
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)
    delta_h = (new_z_fraction - old_z_fraction) * z_span
    if abs(delta_h) < 1e-12:
        return np.asarray(center, dtype=float).copy()
    n_hat = _placement_normal(ctx, slab)
    return np.asarray(center, dtype=float) + float(delta_h) * n_hat


def _xy_recovery_offsets(
    config: AdsorptionConfig,
    *,
    placement_index: int,
    site_index: int,
) -> list[tuple[float, float]]:
    """Deterministic in-plane recovery offsets within configured XY ranges."""
    x_lo, x_hi = config.placement_x_range
    y_lo, y_hi = config.placement_y_range
    if abs(x_hi - x_lo) < 1e-12 and abs(y_hi - y_lo) < 1e-12:
        return []
    rng = random.Random(
        (int(config.seed) * 1_000_003)
        ^ (int(placement_index) * 97)
        ^ (int(site_index) * 1_009)
    )
    return [
        (rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi))
        for _ in range(_DISTANCE_RECOVERY_XY_ATTEMPTS)
    ]


def _apply_lateral_offset(
    center: np.ndarray,
    *,
    dx: float,
    dy: float,
    ctx: _PlacementContext,
    slab: Atoms,
) -> np.ndarray:
    """Apply a lateral recovery offset in the plane perpendicular to the site/slab normal."""
    n_hat = _placement_normal(ctx, slab)
    # Build an orthonormal in-plane basis from Cartesian dx/dy.
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n_hat))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n_hat, ref)
    u = u / float(np.linalg.norm(u))
    v = np.cross(n_hat, u)
    shifted = np.asarray(center, dtype=float) + float(dx) * u + float(dy) * v
    if ctx.mat_type == "slab":
        cell = np.asarray(slab.get_cell(), dtype=float)
        # MIC-wrap the in-plane a–b components; keep height along normal.
        pinv_ab_T, _ = sts._slab_plane_projectors(cell)
        frac2 = shifted @ pinv_ab_T
        frac2 = np.mod(frac2, 1.0)
        # Reconstruct Cartesian from fractional a,b plus original height along normal.
        planar = frac2[0] * cell[0] + frac2[1] * cell[1]
        h = float(np.dot(shifted, n_hat))
        base_h = float(np.dot(planar, n_hat))
        return planar + (h - base_h) * n_hat
    return shifted


def _set_adsorbate_at_center(
    adsorbate: Atoms,
    rotated_pos: np.ndarray,
    center: np.ndarray,
) -> None:
    test = rotated_pos.copy()
    test[:, 0] += float(center[0])
    test[:, 1] += float(center[1])
    test[:, 2] += float(center[2])
    adsorbate.set_positions(test)


def _recover_distance_failure(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    fail_reason: str,
    *,
    slab_for_sites: Atoms | None = None,
) -> tuple[_PlacementContext, str | None]:
    """Nudge height then XY after ``too_close`` / ``too_far``; return updated ctx or last reason."""
    if fail_reason not in ("too_close", "too_far"):
        return ctx, fail_reason

    pose = ctx.pose
    if pose.z_abs is None:
        return ctx, fail_reason

    zf = float(pose.z_fraction)
    origin = np.array([pose.x_abs, pose.y_abs, float(pose.z_abs)], dtype=float)
    z_span = float(ctx.z_base_hi - ctx.z_base_lo)

    height_candidates: list[float] = []
    if z_span > 1e-9:
        for step in range(1, _DISTANCE_RECOVERY_HEIGHT_STEPS + 1):
            frac = step / float(_DISTANCE_RECOVERY_HEIGHT_STEPS + 1)
            if fail_reason == "too_close":
                cand = zf + (1.0 - zf) * frac
            else:
                cand = zf * (1.0 - frac)
            cand = float(min(1.0, max(0.0, cand)))
            if abs(cand - zf) > 1e-9:
                height_candidates.append(cand)

    last_reason: str | None = fail_reason
    for cand_zf in height_candidates:
        center = _center_with_height_delta(
            origin,
            old_z_fraction=zf,
            new_z_fraction=cand_zf,
            ctx=ctx,
            slab=slab,
        )
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, center)
        last_reason = _validate_posed_adsorbate(
            adsorbate, slab, config, slab_for_sites=slab_for_sites
        )
        if last_reason is None:
            new_pose = dataclasses.replace(
                pose,
                x_abs=float(center[0]),
                y_abs=float(center[1]),
                z_abs=float(center[2]),
                z_fraction=float(cand_zf),
            )
            return dataclasses.replace(ctx, pose=new_pose), None
        if last_reason not in ("too_close", "too_far"):
            return ctx, last_reason

    work_zf = zf
    if fail_reason == "too_close" and height_candidates:
        work_zf = max(height_candidates)
    elif fail_reason == "too_far" and height_candidates:
        work_zf = min(height_candidates)

    work_center = _center_with_height_delta(
        origin,
        old_z_fraction=zf,
        new_z_fraction=work_zf,
        ctx=ctx,
        slab=slab,
    )

    for dx, dy in _xy_recovery_offsets(
        config,
        placement_index=pose.placement_index,
        site_index=pose.site_index,
    ):
        center = _apply_lateral_offset(work_center, dx=dx, dy=dy, ctx=ctx, slab=slab)
        _set_adsorbate_at_center(adsorbate, ctx.rotated_pos, center)
        last_reason = _validate_posed_adsorbate(
            adsorbate, slab, config, slab_for_sites=slab_for_sites
        )
        if last_reason is None:
            new_pose = dataclasses.replace(
                pose,
                x_abs=float(center[0]),
                y_abs=float(center[1]),
                z_abs=float(center[2]),
                z_fraction=float(work_zf),
            )
            return dataclasses.replace(ctx, pose=new_pose), None
        if last_reason not in ("too_close", "too_far"):
            return ctx, last_reason

    return ctx, last_reason or fail_reason


def _validate_posed_adsorbate(
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
) -> str | None:
    """Run distance, adsorbate-separation, and optional contact-quality checks.

    Returns a failure reason token, or ``None`` when the placement is accepted.
    """
    exclude_n = _saturation_exclude_count(slab, slab_for_sites)
    ok, _, dist_reason = geom.check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
        max_initial_distance=config.max_initial_distance,
        reject_vdw_overlaps=config.reject_vdw_overlaps,
        vdw_overlap_scale=config.vdw_overlap_scale,
        exclude_slab_atoms=exclude_n,
        material_type=config.material_type,
    )
    if not ok:
        return dist_reason or "distance_check_failed"

    if exclude_n is not None:
        pre_ads = np.asarray(slab.get_positions()[exclude_n:], dtype=float)
        sep_ok, _ = geom.check_adsorbate_separation(
            adsorbate,
            pre_ads,
            cell=np.asarray(slab.get_cell(), dtype=float),
            pbc=material_aware_pbc(config.material_type),
        )
        if not sep_ok:
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
            material_type=config.material_type,
        )
        if not contact_ok:
            return contact_reason

    return None


def _finalize_placement(
    ctx: _PlacementContext,
    adsorbate: Atoms,
    slab: Atoms,
    config: AdsorptionConfig,
    *,
    slab_for_sites: Atoms | None = None,
    allow_distance_recovery: bool = False,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Translate the pre-rotated positions, validate, and build a descriptor."""
    pose = ctx.pose
    if pose.z_abs is None:
        logger.warning("Pose replay requires z_abs for deterministic reconstruction")
        return None, "missing_z_abs"
    z_abs = float(pose.z_abs)

    test = ctx.rotated_pos.copy()
    test[:, 0] += pose.x_abs
    test[:, 1] += pose.y_abs
    test[:, 2] += z_abs
    adsorbate.set_positions(test)

    fail_reason = _validate_posed_adsorbate(
        adsorbate, slab, config, slab_for_sites=slab_for_sites
    )
    if fail_reason is not None:
        if (
            allow_distance_recovery
            and config.placement_distance_recovery
            and fail_reason in ("too_close", "too_far")
        ):
            ctx, fail_reason = _recover_distance_failure(
                ctx,
                adsorbate,
                slab,
                config,
                fail_reason,
                slab_for_sites=slab_for_sites,
            )
            pose = ctx.pose
            if fail_reason is not None:
                return None, fail_reason
            if pose.z_abs is None:
                return None, "missing_z_abs"
            z_abs = float(pose.z_abs)
        else:
            return None, fail_reason

    z_offset = _recover_z_offset(ctx, z_abs, slab)
    slab_indices: tuple[int, ...] | None = None
    if ctx.site is not None and "slab_indices" in ctx.site:
        raw_indices = ctx.site["slab_indices"]
        if isinstance(raw_indices, (list, tuple)):
            slab_indices = tuple(int(i) for i in raw_indices)
    cell = np.asarray(slab.get_cell(), dtype=float)
    pinv_ab_T, _ = sts._slab_plane_projectors(cell)
    placement_xy = np.array([pose.x_abs, pose.y_abs, 0.0], dtype=float)
    frac2 = placement_xy @ pinv_ab_T
    xy_frac = np.mod(frac2, 1.0)

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
        z_offset=z_offset,
        x_abs=float(pose.x_abs),
        y_abs=float(pose.y_abs),
        surface_ref_z_abs=ctx.surface_ref,
        z_abs=z_abs,
        shape=geom._classify_molecule_shape(ctx.canonical_pos)[0],
        slab_indices=slab_indices,
        placement_mode_resolved="sites",
        site_source=ctx.source if ctx.use_sites else "no_sites",
        site_reference_frame="local_site" if ctx.is_local_ref else "global_top_layer",
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        quat_w=float(pose.quat_w),
        quat_x=float(pose.quat_x),
        quat_y=float(pose.quat_y),
        quat_z=float(pose.quat_z),
    )
    return (adsorbate, descriptor), None


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
        logger.warning(
            "Invalid conformer_index=%d for %d conformers",
            pose.conformer_index,
            len(conformers),
        )
        return None
    if not _is_finite_number(pose.x_abs) or not _is_finite_number(pose.y_abs):
        logger.warning("Pose must provide finite x_abs and y_abs")
        return None
    if pose.z_abs is not None and not _is_finite_number(pose.z_abs):
        logger.warning("Pose z_abs must be finite when provided")
        return None

    adsorbate = conformers[pose.conformer_index].copy()
    symbols = adsorbate.get_chemical_symbols()
    canonical_pos = geom.compute_canonical_molecular_frame(
        adsorbate.get_positions(), symbols=symbols
    )

    ctx = _context_from_pose(pose, canonical_pos, slab, config, site_context)
    if ctx is None:
        return None
    result, fail_reason = _finalize_placement(ctx, adsorbate, slab, config)
    if result is None:
        logger.debug("Pose placement rejected: %s", fail_reason)
        return None
    return result


def _spec_grid_info(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None,
    full_slab: Atoms | None = None,
) -> _SpecGridInfo:
    """Compute the spec-enumeration inputs once for both enumerate and estimate."""
    is_dissociative = (
        config.skip_topology_check
        and config.material_type in ("slab", "nanoparticle")
        and _is_dissociable_diatomic(conformers[0])
    )
    _ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )
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

    n_hollow_pairs = 0
    if is_dissociative:
        existing_ads_pos = None
        working_slab = full_slab if full_slab is not None else slab
        if full_slab is not None and len(full_slab) > len(slab):
            existing_ads_pos = full_slab.get_positions()[len(slab) :]
        n_hollow_pairs = len(
            _get_dissociative_site_pairs(
                working_slab,
                config,
                slab_for_sites=slab,
                existing_adsorbate_positions=existing_ads_pos,
                site_context=_ctx,
            )
        )

    return _SpecGridInfo(
        is_dissociative=is_dissociative,
        unique_sites=unique_sites,
        use_sites=use_sites,
        site_indices=site_indices,
        shape=shape,
        symbols=symbols,
        n_binders=len(binders),
        flat_aromatic=flat_aromatic,
        n_hollow_pairs=n_hollow_pairs,
    )


def enumerate_placement_specs(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    n_desired: int,
    filter_spec: Callable[[PlacementSpec], bool] | None = None,
    site_context: SiteContext | None = None,
    seed: int | None = None,
    full_slab: Atoms | None = None,
) -> list[PlacementSpec]:
    """Enumerate placement specs for diverse sampling.

    Builds a stratified set of specs covering conformers, orientation types,
    face flip, electronegative atoms, sites, tilt, and azimuth.
    When ``config.skip_topology_check`` is True and the molecule is a
    dissociable diatomic, dissociative specs are generated instead.

    Uses ``config.seed`` when *seed* is omitted. When the combinatorial grid
    exceeds *n_desired*, specs are subsampled uniformly (reproducible via seed).
    """
    if not conformers:
        return []

    eff_seed = config.seed if seed is None else seed
    info = _spec_grid_info(
        conformers, slab, config, smiles, site_context, full_slab=full_slab
    )
    unique_sites = info.unique_sites
    use_sites = info.use_sites

    def site_type_for(site_idx: int) -> str | None:
        if info.is_dissociative:
            return "hollow"
        if not use_sites or site_idx < 0 or site_idx >= len(unique_sites):
            return None
        return str(unique_sites[site_idx]["site_type"])

    parallel_fraction = config.flat_aromatic_parallel_fraction
    if config.adaptive_parallel_fraction and info.flat_aromatic:
        parallel_fraction = _estimate_parallel_fraction(info.symbols, smiles)

    return policy.build_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info.site_indices,
        site_type_for_index=site_type_for,
        shape=info.shape,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        parallel_fraction=parallel_fraction,
        n_desired=n_desired,
        filter_spec=filter_spec,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
        seed=eff_seed,
    )


def estimate_placement_spec_capacity(
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None,
    site_context: SiteContext | None = None,
    full_slab: Atoms | None = None,
) -> int:
    """Estimate total enumerated specs for current conformers/site grid."""
    if not conformers:
        return 0
    info = _spec_grid_info(
        conformers, slab, config, smiles, site_context, full_slab=full_slab
    )
    return policy.max_batch_placement_specs(
        n_conformers=len(conformers),
        site_indices=info.site_indices,
        shape=info.shape,
        n_binders=info.n_binders,
        flat_aromatic=info.flat_aromatic,
        dissociative=info.is_dissociative,
        n_hollow_pairs=info.n_hollow_pairs,
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
    """Split *total_budget* across molecules in proportion to complexity scores.

    Each molecule gets at least one placement; counts sum to *total_budget*
    (largest-remainder rounding). Scores are typically from
    :func:`estimate_molecule_complexity`.
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

    scores = [max(1.0, float(complexities[name])) for name in names]
    total_score = sum(scores)

    def _share(score: float) -> float:
        return score / total_score * total_budget

    raw = [max(1, int(_share(score))) for score in scores]
    remainder = total_budget - sum(raw)

    if remainder != 0:
        fractions = [_share(score) - max(1, int(_share(score))) for score in scores]
        order = sorted(range(len(names)), key=lambda i: fractions[i], reverse=True)
        for i in range(abs(remainder)):
            idx = order[i % len(order)]
            raw[idx] += 1 if remainder > 0 else -1
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
    slab_for_sites: Atoms | None = None,
) -> tuple[Atoms, PlacementDescriptor] | None:
    """Generate adsorbate placement from spec. Returns (adsorbate, descriptor) or None."""
    result, _ = generate_placement_from_spec_with_reason(
        spec,
        conformers,
        slab,
        config,
        smiles=smiles,
        site_context=site_context,
        slab_for_sites=slab_for_sites,
    )
    return result


def generate_placement_from_spec_with_reason(
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
    smiles: str | None = None,
    site_context: SiteContext | None = None,
    slab_for_sites: Atoms | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Generate placement from spec and provide a failure reason when unavailable."""
    if not conformers:
        return None, "no_conformers"

    if spec.orientation_type == "dissociative":
        adsorbate = conformers[spec.conformer_index % len(conformers)].copy()
        return _generate_dissociative_placement_from_spec(
            adsorbate,
            spec,
            slab,
            config,
            slab_for_sites=slab_for_sites,
            site_context=site_context,
        )

    resolved_ctx = (
        site_context
        if site_context is not None
        else _get_unique_sites_for_specs(slab, config)
    )

    adsorbate = conformers[spec.conformer_index % len(conformers)].copy()

    placement_ctx = _pose_from_spec(
        adsorbate,
        spec,
        slab,
        config,
        smiles,
        site_context=resolved_ctx,
        slab_for_sites=slab_for_sites,
    )
    if placement_ctx is None:
        return None, "no_sites_found"

    result, fail_reason = _finalize_placement(
        placement_ctx,
        adsorbate,
        slab,
        config,
        slab_for_sites=slab_for_sites,
        allow_distance_recovery=True,
    )
    if result is not None:
        return result, None
    return None, fail_reason or "distance_check_failed"


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
        logger.warning(
            "Descriptor conformer_index=%d out of range for %d conformers",
            descriptor.conformer_index,
            len(conformers),
        )
        return None
    if descriptor.x_abs is None or descriptor.y_abs is None or descriptor.z_abs is None:
        logger.warning(
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
        logger.warning("Descriptor replay requires quaternion components")
        return None
    pose = _pose_from_descriptor(descriptor)
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


def _site_outward_normal(
    site_xyz: np.ndarray,
    site_entry: dict[str, object],
    *,
    material_type: str,
    reference_positions: np.ndarray,
    slab_normal: np.ndarray,
) -> np.ndarray:
    """Unit outward normal for dissociative fragment placement."""
    if "normal" in site_entry:
        normal = np.asarray(site_entry["normal"], dtype=float)
        norm = float(np.linalg.norm(normal))
        if norm > _VECTOR_NORM_EPS:
            return normal / norm
    if material_type == "nanoparticle":
        com = np.mean(reference_positions, axis=0)
        outward = np.asarray(site_xyz, dtype=float) - com
        norm = float(np.linalg.norm(outward))
        if norm > _VECTOR_NORM_EPS:
            return outward / norm
    return np.asarray(slab_normal, dtype=float)


def _get_dissociative_site_pairs(
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    existing_adsorbate_positions: np.ndarray | None = None,
    *,
    raw_sites: list[dict[str, object]] | None = None,
    site_context: SiteContext | None = None,
) -> list[_DissociativeSitePair]:
    """Return outward-oriented site pairs for dissociative diatomic placement."""
    if config.material_type not in ("slab", "nanoparticle"):
        return []

    sites_slab = slab_for_sites if slab_for_sites is not None else slab
    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    pbc = material_aware_pbc(config.material_type)
    # Pair uniqueness on slabs uses xy-only MIC (intentional for planar catalogs).
    pbc_xy = [bool(pbc[0]), bool(pbc[1]), False]
    slab_normal = sts._slab_normal(cell_arr)

    if raw_sites is not None:
        raw = list(raw_sites)
    elif site_context is not None and site_context.raw_unclustered is not None:
        raw = list(site_context.raw_unclustered)
    elif site_context is not None and site_context.sites:
        raw = list(site_context.sites)
    else:
        raw = sts.get_unified_sites(
            sites_slab,
            top_layer_tolerance=config.top_layer_tolerance,
            material_type=config.material_type,
            probe_radius=config.voronoi_probe_radius,
            max_site_distance=config.voronoi_max_site_distance,
            enrich=config.voronoi_site_enrichment,
            site_classification_method=config.site_classification_method,
        )
    if config.material_type == "slab":
        site_entries = [s for s in raw if s.get("site_type") in ("hollow", "pore")]
    else:
        site_entries = list(raw)
    if len(site_entries) < 2:
        return []

    site_xyz = np.array(
        [np.asarray(s["xyz"], dtype=float) for s in site_entries], dtype=float
    )
    if config.material_type == "slab":
        keep = sts._deduplicate_points(
            site_xyz,
            config.hollow_site_dedup_tolerance,
            cell=cell_arr,
            pbc=np.asarray(pbc_xy, dtype=bool),
        )
        site_xyz = site_xyz[keep]
        site_entries = [site_entries[i] for i in np.nonzero(keep)[0]]
    if len(site_xyz) < 2:
        return []

    if (
        existing_adsorbate_positions is not None
        and len(existing_adsorbate_positions) > 0
    ):
        available_xyz: list[np.ndarray] = []
        available_entries: list[dict[str, object]] = []
        for entry, site_pos in zip(site_entries, site_xyz, strict=True):
            d = geom.calculate_min_distance(
                site_pos.reshape(1, 3),
                existing_adsorbate_positions,
                cell=cell_arr,
                pbc=pbc,
            )
            if d >= config.min_initial_distance:
                available_xyz.append(site_pos)
                available_entries.append(entry)
        if len(available_xyz) < 2:
            return []
        site_xyz = np.asarray(available_xyz, dtype=float)
        site_entries = available_entries

    symbols = sites_slab.get_chemical_symbols()
    top_positions = sites_slab.get_positions()
    if config.material_type == "nanoparticle":
        radius_indices = list(range(len(top_positions)))
    else:
        top_mask = sts.top_layer_mask_by_normal(
            top_positions,
            cell_arr,
            float(config.top_layer_tolerance),
        )
        radius_indices = list(np.nonzero(top_mask)[0])
    top_radii = [
        r
        for i in radius_indices
        if (r := geom._get_covalent_radius(symbols[int(i)])) is not None
    ]
    mean_top_radius = float(np.mean(top_radii)) if top_radii else 1.0

    site_3d = site_xyz
    if len(site_3d) >= 2:
        _nn_tree = KDTree(site_3d)
        nn_d, _ = _nn_tree.query(site_3d, k=2)
        nn_d_arr = np.asarray(nn_d, dtype=float)
        if nn_d_arr.ndim == 2:
            mean_nn_sep = float(np.mean(nn_d_arr[:, 1]))
        else:
            mean_nn_sep = float(np.mean(nn_d_arr))

        atomic_constraint = _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE * (
            2.0 * mean_top_radius
        )
        surface_constraint = 0.8 * mean_nn_sep
        adaptive_min = min(atomic_constraint, surface_constraint)
        min_fragment_sep = max(
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM, adaptive_min
        )
    else:
        mean_nn_sep = _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM
        min_fragment_sep = max(
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_FLOOR_ANGSTROM,
            _DISSOCIATIVE_MIN_FRAGMENT_SEP_RADIUS_SCALE * (2.0 * mean_top_radius),
        )
    max_adjacent_sep = float(
        np.clip(
            _DISSOCIATIVE_MAX_ADJACENT_SEP_NN_SCALE * mean_nn_sep,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_FLOOR_ANGSTROM,
            _DISSOCIATIVE_MAX_ADJACENT_SEP_CAP_ANGSTROM,
        )
    )

    tree = KDTree(site_3d)
    candidate_pairs = tree.query_pairs(r=max_adjacent_sep)

    pairs: list[_DissociativeSitePair] = []
    for i, j in candidate_pairs:
        if config.material_type == "slab":
            _, dists = find_mic(
                (site_3d[i] - site_3d[j]).reshape(1, 3),
                cell_arr,
                pbc=pbc_xy,
            )
            d = float(dists[0])
        else:
            d = float(np.linalg.norm(site_3d[i] - site_3d[j]))
        if min_fragment_sep <= d <= max_adjacent_sep:
            normal_i = _site_outward_normal(
                site_3d[i],
                site_entries[i],
                material_type=config.material_type,
                reference_positions=top_positions,
                slab_normal=slab_normal,
            )
            normal_j = _site_outward_normal(
                site_3d[j],
                site_entries[j],
                material_type=config.material_type,
                reference_positions=top_positions,
                slab_normal=slab_normal,
            )
            pairs.append(
                _DissociativeSitePair(
                    xyz1=site_3d[i].copy(),
                    normal1=normal_i,
                    xyz2=site_3d[j].copy(),
                    normal2=normal_j,
                )
            )
    return pairs


def _get_hollow_site_pairs(
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    existing_adsorbate_positions: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return hollow-site xyz pairs (slab dissociative placements; legacy API)."""
    return [
        (pair.xyz1, pair.xyz2)
        for pair in _get_dissociative_site_pairs(
            slab,
            config,
            slab_for_sites=slab_for_sites,
            existing_adsorbate_positions=existing_adsorbate_positions,
        )
    ]


def _generate_dissociative_placement_from_spec(
    adsorbate: Atoms,
    spec: PlacementSpec,
    slab: Atoms,
    config: AdsorptionConfig,
    slab_for_sites: Atoms | None = None,
    site_context: SiteContext | None = None,
) -> tuple[tuple[Atoms, PlacementDescriptor] | None, str | None]:
    """Place homonuclear-diatomic fragments at two surface sites."""
    if config.material_type not in ("slab", "nanoparticle"):
        return None, f"dissociative_not_supported_for_{config.material_type}"

    if not _is_dissociable_diatomic(adsorbate):
        return None, "not_dissociable_diatomic"

    sites_slab = slab_for_sites if slab_for_sites is not None else slab

    existing_ads_pos = None
    if slab_for_sites is not None and len(slab) > len(slab_for_sites):
        existing_ads_pos = slab.get_positions()[len(slab_for_sites) :]

    pairs = _get_dissociative_site_pairs(
        slab,
        config,
        slab_for_sites=slab_for_sites,
        existing_adsorbate_positions=existing_ads_pos,
        site_context=site_context,
    )
    if not pairs:
        return None, "no_hollow_site_pairs"

    pair_idx = spec.site_index % len(pairs)
    site_pair = pairs[pair_idx]

    cell_arr = np.asarray(slab.get_cell(), dtype=float)
    if config.material_type == "nanoparticle":
        h_surface = 0.5 * (
            float(np.dot(np.asarray(site_pair.xyz1, dtype=float), site_pair.normal1))
            + float(np.dot(np.asarray(site_pair.xyz2, dtype=float), site_pair.normal2))
        )
        site_reference_frame = "local_site"
    else:
        heights = sts._height_along_slab_normal(sites_slab.get_positions(), cell_arr)
        h_surface = float(np.max(heights))
        site_reference_frame = "global_top_layer"

    syms = adsorbate.get_chemical_symbols()
    hollow_site: dict[str, object] = {"site_type": "hollow"}
    z_lo, z_hi = sts._compute_site_z_base(config, sites_slab, hollow_site, syms)
    site_offset = _site_type_z_offset(sites_slab, hollow_site, "hollow")
    z_lo += site_offset
    z_hi += site_offset
    z_offset = z_lo + spec.z_fraction * (z_hi - z_lo)

    pos1 = np.asarray(site_pair.xyz1, dtype=float) + float(z_offset) * site_pair.normal1
    pos2 = np.asarray(site_pair.xyz2, dtype=float) + float(z_offset) * site_pair.normal2

    result = Atoms(symbols=syms, positions=[pos1, pos2])
    result.set_cell(slab.get_cell())
    result.set_pbc(slab.get_pbc())

    fail_reason = _validate_posed_adsorbate(
        result, slab, config, slab_for_sites=slab_for_sites
    )
    if fail_reason is not None:
        return None, fail_reason

    centroid = (pos1 + pos2) / 2.0
    pinv_ab_T, _ = sts._slab_plane_projectors(np.asarray(slab.get_cell(), dtype=float))
    placement_xy = np.array([centroid[0], centroid[1], 0.0], dtype=float)
    xy_frac = np.mod(placement_xy @ pinv_ab_T, 1.0)
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
        surface_ref_z_abs=h_surface,
        z_abs=float(centroid[2]),
        shape="linear",
        slab_indices=None,
        placement_mode_resolved="sites",
        site_source="dissociative_hollow_pair",
        site_reference_frame=site_reference_frame,
        site_xy_frac_a=float(xy_frac[0]),
        site_xy_frac_b=float(xy_frac[1]),
        quat_w=1.0,
        quat_x=0.0,
        quat_y=0.0,
        quat_z=0.0,
    )
    return (result, descriptor), None

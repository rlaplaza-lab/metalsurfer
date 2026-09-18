"""Unified site enumeration, clustering, symmetry reduction, and z-base helpers."""

import logging
from collections.abc import Callable

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree, QhullError

from .._utils import cell_has_volume
from ..symmetry import SymmetryAnalyzer
from ._constants import (
    _ADSORBATE_COVALENT_RADIUS_FALLBACK,
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    _DEFAULT_SYMMETRY_TOLERANCE,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _VORONOI_AUTO_WIDEN_MAX_SCALE,
    _VORONOI_AUTO_WIDEN_PROBE_SCALE,
)
from ._material import (
    material_aware_pbc,
    material_type_for_placement,
    validate_material_type,
)
from .geometry import _get_covalent_radius
from .site_adaptive_grid import dedupe_adaptive_sites_within_type
from .site_classify import (
    _build_site_records,
    _DelaunayClassifyInputs,
)
from .site_coords import (
    _cart_to_frac,
    _derive_top_layer_tolerance,
    _derive_voronoi_distance_window,
    _frac_to_cart,
    _height_along_slab_normal,
    _mean_covalent_radius,
    _minimum_image_fractional_delta,
    _pbc_merge_pair_set,
    _periodic_image_offsets,
    _pore_threshold_from_mean_radius,
    _project_to_slab_plane,
    _shift_along_slab_normal,
    _slab_normal,
    _slab_plane_projectors,
    _top_layer_tolerance_from_mean_radius,
    _union_find_cluster,
    _wrap_cartesian,
    _wrap_fractional,
    top_layer_mask_by_normal,
)
from .site_np import (
    _convex_hull_surface_mask,
    _surface_atom_normals,
    _try_convex_hull,
)
from .site_plugins import (
    SiteGenerationContext,
    resolve_site_generator,
)
from .site_plugins.helpers import (
    PlanarWidenScratch as _PlanarWidenScratch,
)
from .site_plugins.helpers import (
    apply_site_mask as _apply_site_mask,
)
from .site_plugins.helpers import (
    bounding_box_cell as _bounding_box_cell,
)
from .site_plugins.helpers import (
    median_nn_or_fallback as _median_nn_or_fallback,
)
from .site_plugins.helpers import (
    merge_dedup_site_arrays as _merge_dedup_site_arrays,
)
from .site_plugins.helpers import (
    top_layer_is_planar_from_arrays as _top_layer_is_planar_from_arrays,
)
from .site_types import Site
from .site_voronoi import (
    _build_delaunay_classification_index,
)

logger = logging.getLogger(__name__)

_EMPTY_ATOM_INDICES: tuple[int, ...] = ()


def _delaunay_classify_inputs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    material_type: str,
    site_classification_method: str,
    slab_top_atom_indices: np.ndarray | None,
    topology_primary_delaunay: Delaunay | None,
    expanded_xy: np.ndarray | None = None,
    expanded_origin: list[int] | None = None,
    expanded_tri: Delaunay | None = None,
) -> _DelaunayClassifyInputs | None:
    """Build prebuilt Delaunay classification inputs, or ``None`` when disabled."""
    if site_classification_method == "delaunay" and material_type != "slab":
        logger.warning(
            "site_classification_method='delaunay' is slab-only; "
            "keeping topology labels (nanoparticle) or distance-ratio "
            "(porous) for material_type=%r",
            material_type,
        )
    if material_type != "slab" or site_classification_method not in (
        "delaunay",
        "auto",
    ):
        return None
    if slab_top_atom_indices is None:
        raise ValueError(
            "slab_top_atom_indices must be set for Delaunay classification"
        )
    if len(slab_top_atom_indices) < 3:
        return None
    top_positions_2d = _project_to_slab_plane(positions[slab_top_atom_indices], cell)
    tri = topology_primary_delaunay
    if tri is None:
        try:
            tri = Delaunay(top_positions_2d)
        except (QhullError, ValueError, RuntimeError) as exc:
            logger.debug("Delaunay classification disabled (%s)", exc)
            return None
    class_index = _build_delaunay_classification_index(
        top_positions_2d,
        slab_top_atom_indices,
        tri,
        cell=cell,
        pbc=pbc,
        expanded_xy=expanded_xy,
        expanded_origin=expanded_origin,
        expanded_tri=expanded_tri,
    )
    return _DelaunayClassifyInputs(
        top_positions_2d,
        slab_top_atom_indices,
        class_index,
    )


def _inject_atop_sites(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    *,
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    material_type: str,
    local_tree: KDTree,
    accessibility_tree: KDTree | None,
    median_nn: float | None,
    slab_top_atom_indices: np.ndarray | None,
    has_topology_atop: bool,
    probe_radius: float,
    max_site_distance: float,
    atom_indices: list[tuple[int, ...]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[tuple[int, ...]]]:
    """Inject atop when topology did not already produce any.

    Uses the same height as the topology generator when *median_nn* is supplied.
    *accessibility_tree* (PBC-aware) gates distances under periodic boundaries.
    """
    atoms = (
        list(atom_indices)
        if atom_indices is not None
        else [_EMPTY_ATOM_INDICES for _ in range(len(vertices))]
    )
    if material_type in ("slab", "nanoparticle") and has_topology_atop:
        return vertices, nn_dists, source_hints, atoms
    if material_type not in ("slab", "nanoparticle"):
        return vertices, nn_dists, source_hints, atoms

    if median_nn is None:
        ref = (
            positions[slab_top_atom_indices]
            if material_type == "slab" and slab_top_atom_indices is not None
            else positions
        )
        median_nn = _median_nn_or_fallback(
            nn_dists if material_type == "slab" else np.empty(0, dtype=float),
            reference_positions=ref,
            cell=cell,
            pbc=pbc,
        )
    atop_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn

    if material_type == "slab":
        if slab_top_atom_indices is None:
            return vertices, nn_dists, source_hints, atoms
        top_atom_indices = np.asarray(slab_top_atom_indices, dtype=int)
        atom_normals = None
    else:
        hull = _try_convex_hull(positions)
        if hull is None:
            return vertices, nn_dists, source_hints, atoms
        top_atom_indices = np.nonzero(_convex_hull_surface_mask(positions, hull=hull))[
            0
        ].astype(int)
        if len(top_atom_indices) == 0:
            return vertices, nn_dists, source_hints, atoms
        atom_normals = _surface_atom_normals(positions, top_atom_indices, hull)

    candidate_verts: list[np.ndarray] = []
    candidate_atom_ids: list[int] = []
    for li, ai in enumerate(top_atom_indices):
        atom_pos = positions[int(ai)]
        if material_type == "slab":
            candidate = _shift_along_slab_normal(
                atom_pos.reshape(1, 3), cell, atop_height
            )[0]
            if np.any(pbc):
                candidate = _wrap_cartesian(candidate.reshape(1, 3), cell, pbc)[0]
        else:
            assert atom_normals is not None
            n_hat = atom_normals[li]
            if float(np.linalg.norm(n_hat)) < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
                continue
            candidate = atom_pos + atop_height * n_hat
        candidate_verts.append(candidate)
        candidate_atom_ids.append(int(ai))

    if not candidate_verts:
        return vertices, nn_dists, source_hints, atoms

    candidate_arr = np.asarray(candidate_verts, dtype=float)
    gate_tree = (
        accessibility_tree
        if accessibility_tree is not None and np.any(pbc)
        else local_tree
    )
    d_nn_all = np.asarray(gate_tree.query(candidate_arr, k=1)[0], dtype=float).ravel()
    keep_acc = (d_nn_all >= float(probe_radius)) & (
        d_nn_all <= float(max_site_distance)
    )
    if not np.any(keep_acc):
        return vertices, nn_dists, source_hints, atoms

    candidate_arr = candidate_arr[keep_acc]
    candidate_dist_arr = d_nn_all[keep_acc]
    kept_atom_ids = [candidate_atom_ids[i] for i in np.nonzero(keep_acc)[0]]
    candidate_sources = ["atop_injected"] * len(candidate_arr)
    candidate_atoms = [(ai,) for ai in kept_atom_ids]

    n_existing = len(vertices)
    vertices, nn_dists, source_hints, atoms = _merge_dedup_site_arrays(
        vertices,
        nn_dists,
        source_hints,
        candidate_arr,
        candidate_dist_arr,
        candidate_sources,
        cell=cell,
        pbc=pbc,
        atom_indices=atoms,
        new_atom_indices=candidate_atoms,
    )
    n_injected = len(vertices) - n_existing
    logger.debug(
        "Injected %d atop candidate sites (%d total sites)",
        max(n_injected, 0),
        len(vertices),
    )

    return vertices, nn_dists, source_hints, atoms


def get_unified_sites(
    atoms: Atoms,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    top_layer_tolerance: float | None = None,
    material_type: str | None = None,
    pore_threshold: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
    *,
    auto_widen: bool = True,
    planar_z_variance_threshold: float | None = None,
    site_generator: str = "auto",
    adsorbate: Atoms | None = None,
    grid_spacing_scale: float | None = None,
    n_jobs: int = -2,
) -> list[Site]:
    """Return adsorption/placement sites for *atoms*.

    Candidates come from a plugin selected by *site_generator*
    (``auto`` / ``topology`` / ``voronoi`` / ``adaptive_grid``).
    With ``auto``, slabs and nanoparticles use topology; porous frameworks
    use Voronoi. ``adaptive_grid`` works on all materials but is not selected
    by ``auto``.

    - **slab** (topology): Delaunay atop/bridge/hollow; planar top layers skip
      Voronoi; rough slabs merge Voronoi enrichment.
    - **nanoparticle** (topology): hull + NN only.
    - **porous** (voronoi): free-volume vertices with optional ridge enrichment.
    - **adaptive_grid**: atom-centred Cartesian shells with exposure filtering
      and iterative refinement for every material type.

    Parameters
    ----------
    atoms
        :class:`~ase.Atoms` structure to detect sites on.
    probe_radius
        Accessibility probe radius (auto-derived if None).
    max_site_distance
        Maximum site-to-atom distance (auto-derived if None).
    top_layer_tolerance
        Height tolerance for the top layer (auto-derived if None).
    material_type
        ``"slab"``, ``"nanoparticle"``, or ``"porous"``.
    pore_threshold
        Pore classification threshold (auto-derived if None).
    enrich
        Whether to enrich Voronoi ridge candidates.
    site_classification_method
        Site classification method (``"auto"``, ``"delaunay"``, etc.).
    auto_widen
        When True and the first pass finds no sites, retry once with a widened
        probe / max-distance window.
    planar_z_variance_threshold
        Max top-layer height variance (Å²) for classifying a slab as planar.
        ``None`` uses the library default.
    site_generator
        ``"auto"`` (material default), ``"topology"``, ``"voronoi"``, or
        ``"adaptive_grid"``.
    adsorbate
        Optional representative conformer. Used only by ``adaptive_grid`` when
        *grid_spacing_scale* is omitted; ignored by topology and Voronoi.
    grid_spacing_scale
        Shared adaptive-grid spacing length (Å). Typically
        :func:`~metalsurfer.placement.site_adaptive_grid.min_adsorbate_grid_scale`
        across competing molecules; ignored by topology / Voronoi.
    n_jobs
        Joblib-style CPU workers for ``adaptive_grid`` shell/refine stages
        (default ``-2``).
    """
    scratch = _PlanarWidenScratch()
    sites = _enumerate_unified_sites(
        atoms,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        pore_threshold=pore_threshold,
        enrich=enrich,
        site_classification_method=site_classification_method,
        planar_z_variance_threshold=planar_z_variance_threshold,
        site_generator=site_generator,
        adsorbate=adsorbate,
        grid_spacing_scale=grid_spacing_scale,
        n_jobs=n_jobs,
        _widen_scratch=scratch,
    )
    if sites or not auto_widen:
        return sites

    positions = atoms.get_positions()
    symbols = list(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.get_cell(), dtype=float)
    # Caller (_enumerate_unified_sites) already validates material_type.
    assert material_type is not None
    pbc = np.asarray(material_aware_pbc(material_type), dtype=bool)
    derived_probe, derived_max = _derive_voronoi_distance_window(
        positions, symbols, pbc, cell
    )
    eff_probe = float(probe_radius) if probe_radius is not None else derived_probe
    eff_max = float(max_site_distance) if max_site_distance is not None else derived_max
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
    reuse = (
        scratch if scratch.planar_skip_voronoi and scratch.exp_tri is not None else None
    )
    return _enumerate_unified_sites(
        atoms,
        probe_radius=wide_probe,
        max_site_distance=wide_max,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        pore_threshold=pore_threshold,
        enrich=enrich,
        site_classification_method=site_classification_method,
        planar_z_variance_threshold=planar_z_variance_threshold,
        site_generator=site_generator,
        adsorbate=adsorbate,
        grid_spacing_scale=grid_spacing_scale,
        n_jobs=n_jobs,
        _reuse_topology=reuse,
    )


def _enumerate_unified_sites(
    atoms: Atoms,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    top_layer_tolerance: float | None = None,
    material_type: str | None = None,
    pore_threshold: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
    planar_z_variance_threshold: float | None = None,
    site_generator: str = "auto",
    *,
    adsorbate: Atoms | None = None,
    grid_spacing_scale: float | None = None,
    n_jobs: int = -2,
    _widen_scratch: _PlanarWidenScratch | None = None,
    _reuse_topology: _PlanarWidenScratch | None = None,
) -> list[Site]:
    """Core site enumeration (single pass, no auto-widen)."""
    if len(atoms) == 0:
        raise ValueError("atoms must contain at least one atom")
    if material_type is None:
        raise ValueError(
            "material_type must be explicitly specified: 'slab', 'nanoparticle', or 'porous'"
        )
    validate_material_type(material_type)
    plugin = resolve_site_generator(site_generator, material_type)

    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    atoms_pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    pbc_for_voronoi = np.asarray(material_aware_pbc(material_type), dtype=bool)
    if not np.array_equal(atoms_pbc, pbc_for_voronoi):
        logger.debug(
            "Using material_aware_pbc(%r)=%s for site enumeration "
            "(atoms.get_pbc() was %s)",
            material_type,
            pbc_for_voronoi.tolist(),
            atoms_pbc.tolist(),
        )

    symbols = atoms.get_chemical_symbols()
    if top_layer_tolerance is None or pore_threshold is None:
        # One all-symbols mean for top-layer depth and pore threshold; Voronoi
        # window may use a top-layer subset and is derived separately.
        mean_radius = _mean_covalent_radius(symbols)
        if top_layer_tolerance is None:
            top_layer_tolerance = _top_layer_tolerance_from_mean_radius(mean_radius)
        if pore_threshold is None:
            pore_threshold = _pore_threshold_from_mean_radius(mean_radius)
    z_var_threshold = (
        float(planar_z_variance_threshold)
        if planar_z_variance_threshold is not None
        else _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
    )

    if not cell_has_volume(cell):
        cell = _bounding_box_cell(positions)
        if np.any(pbc_for_voronoi):
            logger.warning(
                "Input cell is degenerate while PBC is enabled; using a padded "
                "bounding-box cell with PBC disabled for site enumeration"
            )
            pbc_for_voronoi[:] = False

    if probe_radius is None or max_site_distance is None:
        derived_probe, derived_max = _derive_voronoi_distance_window(
            positions, symbols, pbc_for_voronoi, cell
        )
        probe_radius = derived_probe if probe_radius is None else probe_radius
        max_site_distance = (
            derived_max if max_site_distance is None else max_site_distance
        )

    ctx = SiteGenerationContext(
        positions=positions,
        cell=cell,
        pbc=pbc_for_voronoi,
        symbols=list(symbols),
        material_type=material_type,
        probe_radius=float(probe_radius),
        max_site_distance=float(max_site_distance),
        top_layer_tolerance=float(top_layer_tolerance),
        enrich=bool(enrich),
        planar_z_variance_threshold=float(z_var_threshold),
        adsorbate=adsorbate,
        grid_spacing_scale=grid_spacing_scale,
        n_jobs=int(n_jobs),
    )
    batch = plugin.generate(ctx, reuse=_reuse_topology)

    if _widen_scratch is not None and isinstance(batch.reuse, _PlanarWidenScratch):
        _widen_scratch.planar_skip_voronoi = batch.reuse.planar_skip_voronoi
        _widen_scratch.primary_delaunay = batch.reuse.primary_delaunay
        _widen_scratch.exp_xy = batch.reuse.exp_xy
        _widen_scratch.exp_origin = batch.reuse.exp_origin
        _widen_scratch.exp_tri = batch.reuse.exp_tri

    if batch.early_empty:
        return []

    vertices = batch.vertices
    nn_dists = batch.nn_dists
    source_hints = list(batch.source_hints)
    atom_indices = list(batch.atom_indices)
    local_tree = KDTree(positions)

    if batch.apply_slab_height_mask and len(vertices) > 0:
        heights = _height_along_slab_normal(positions, cell)
        h_surface = float(np.max(heights))
        nn_margin = (
            float(np.median(nn_dists))
            if len(nn_dists) > 0
            else float(top_layer_tolerance)
        )
        h_min = h_surface - max(float(top_layer_tolerance), nn_margin)
        keep_mask = _height_along_slab_normal(vertices, cell) >= h_min
        vertices, nn_dists, source_hints, atom_indices = _apply_site_mask(
            vertices, nn_dists, source_hints, keep_mask, atom_indices
        )

    if batch.inject_atop:
        vertices, nn_dists, source_hints, atom_indices = _inject_atop_sites(
            vertices,
            nn_dists,
            source_hints,
            positions=positions,
            cell=cell,
            pbc=pbc_for_voronoi,
            material_type=material_type,
            local_tree=local_tree,
            accessibility_tree=batch.accessibility_tree,
            median_nn=batch.topology_median_nn,
            slab_top_atom_indices=batch.slab_top_atom_indices,
            has_topology_atop=batch.has_topology_atop,
            probe_radius=float(probe_radius),
            max_site_distance=float(max_site_distance),
            atom_indices=atom_indices,
        )

    if len(vertices) == 0:
        logger.warning(
            "No accessible sites after atop injection for %d-atom "
            "structure (probe_radius=%s, max_distance=%s, material_type=%r)",
            len(atoms),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_distance:.2f}" if max_site_distance is not None else "auto",
            material_type,
        )
        return []

    # ``auto`` / ``delaunay`` use Delaunay on slabs; ``distance_ratio`` is honored
    # literally (opt-in A/B). Default config ``auto`` preserves catalysis sampling.
    delaunay_inputs = _delaunay_classify_inputs(
        positions,
        cell,
        pbc_for_voronoi,
        material_type=material_type,
        site_classification_method=site_classification_method,
        slab_top_atom_indices=batch.slab_top_atom_indices,
        topology_primary_delaunay=batch.topology_primary_delaunay,
        expanded_xy=batch.topology_expanded_xy,
        expanded_origin=batch.topology_expanded_origin,
        expanded_tri=batch.topology_expanded_tri,
    )

    sites = _build_site_records(
        vertices,
        nn_dists,
        positions,
        symbols,
        local_tree,
        material_type,
        pore_threshold,
        cell=cell,
        source_hints=source_hints,
        pbc=pbc_for_voronoi,
        delaunay=delaunay_inputs,
        atom_indices=atom_indices,
    )

    if (
        sites
        and source_hints
        and any(h == "adaptive_grid" for h in source_hints)
    ):
        sites = dedupe_adaptive_sites_within_type(
            sites,
            cell=cell,
            pbc=pbc_for_voronoi,
            median_nn=float(batch.topology_median_nn or 0.0),
            probe_radius=float(probe_radius),
            max_site_distance=float(max_site_distance),
        )

    if cell_has_volume(cell):
        # Deterministic fractional-xyz order for stable site_index / raw catalog.
        all_xyz = np.asarray([s.xyz for s in sites], dtype=float).reshape(-1, 3)
        all_frac = _wrap_fractional(_cart_to_frac(all_xyz, cell), pbc_for_voronoi)

        def _site_frac_key(i: int) -> tuple:
            frac = all_frac[i]
            return (
                float(frac[0]),
                float(frac[1]),
                float(frac[2]),
                str(sites[i].site_type),
            )

        order = sorted(range(len(sites)), key=_site_frac_key)
        sites = [sites[i] for i in order]

    return sites


def get_hollow_sites_for_adatoms(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    *,
    material_type: str,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
    site_generator: str = "auto",
    site_equivalence_tolerance: float = _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    auto_widen: bool = True,
    planar_z_variance_threshold: float | None = None,
) -> list[Site]:
    """Return hollow/pore sites for adatom placement from the clustered catalog.

    Uniqueness uses :func:`_cluster_equivalent_sites` with
    *site_equivalence_tolerance* (same metric as molecular placement).

    Parameters
    ----------
    slab
        :class:`~ase.Atoms` substrate.
    top_layer_tolerance
        Height tolerance for the top layer (auto-derived if None).
    material_type
        ``"slab"``, ``"nanoparticle"``, or ``"porous"``. Required so that the
        PBC semantics always match the caller's intent (same as
        :func:`get_unified_sites`).
    probe_radius
        Voronoi probe radius (auto-derived if None).
    max_site_distance
        Maximum site-to-atom distance (auto-derived if None).
    enrich
        Whether to enrich Voronoi ridge candidates.
    site_classification_method
        Site classification method (``"auto"``, ``"delaunay"``, etc.).
    site_generator
        Site generator plugin (``"auto"``, ``"topology"``, ``"voronoi"``,
        ``"adaptive_grid"``).
    site_equivalence_tolerance
        Fingerprint-aware clustering tolerance (Å).
    auto_widen
        Retry once with a wider accessibility window if the first pass is empty.
    planar_z_variance_threshold
        Max top-layer height variance (Å²) for planar classification.
    """
    raw = get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        enrich=enrich,
        site_classification_method=site_classification_method,
        site_generator=site_generator,
        auto_widen=auto_widen,
        planar_z_variance_threshold=planar_z_variance_threshold,
    )
    if not raw:
        return []
    cell = np.asarray(slab.get_cell(), dtype=float)
    clustered = _cluster_equivalent_sites(
        raw,
        cell,
        tolerance=site_equivalence_tolerance,
    )
    return [s for s in clustered if s.site_type in ("hollow", "pore")]


# ---------------------------------------------------------------------------
# Environment-aware site clustering
# ---------------------------------------------------------------------------


def _env_fingerprint(site: Site) -> tuple:
    """Return the local-environment fingerprint of *site*."""
    fp = site.env_fingerprint
    if fp is not None:
        return tuple(fp)
    return (str(site.site_type),)


def _cluster_with_metric(
    n: int,
    coords: np.ndarray,
    fps: list[tuple],
    *,
    image_offsets: list[np.ndarray] | None,
    kdtree_radius: float,
    pair_filter: Callable[[int, int], bool] | None = None,
) -> list[int]:
    """KDTree query_pairs + union-find; one representative index per cluster."""
    # Shared PBC-aware pair extraction (same primitive as _deduplicate_points).
    candidates = _pbc_merge_pair_set(coords, kdtree_radius, image_offsets=image_offsets)

    merge_set: set[tuple[int, int]] = set()
    for key in candidates:
        a, b = key
        if fps[a] != fps[b]:
            continue
        if pair_filter is not None and not pair_filter(a, b):
            continue
        merge_set.add(key)

    components = _union_find_cluster(n, list(merge_set))
    return sorted(min(comp) for comp in components)


def _cluster_equivalent_sites(
    sites: list[Site],
    cell: np.ndarray,
    tolerance: float = _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    z_abs_tolerance: float | None = None,
) -> list[Site]:
    """Group equivalent sites; return unique representatives.

    Merges only when spatially close (material-aware metric) and
    ``env_fingerprint`` matches. ``site_source`` is ignored so topology,
    Voronoi, and atop-injected candidates in the same pocket collapse.
    """
    if not sites:
        return []

    n = len(sites)
    mat_type = material_type_for_placement(sites[0], when_no_site="slab")

    def _get_xyz(s: Site) -> np.ndarray:
        return np.asarray(s.xyz, dtype=float)

    def _sort_key(s: Site) -> tuple:
        xyz = _get_xyz(s)
        return (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            str(s.site_type),
        )

    order = sorted(range(n), key=lambda i: _sort_key(sites[i]))
    sorted_sites = [sites[i] for i in order]
    fps = [_env_fingerprint(s) for s in sorted_sites]

    if mat_type == "nanoparticle" or not cell_has_volume(cell):
        coords = np.array([_get_xyz(s) for s in sorted_sites])
        reps = _cluster_with_metric(
            n,
            coords,
            fps,
            image_offsets=None,
            kdtree_radius=tolerance,
        )
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    if mat_type == "porous":
        coords = np.array([_get_xyz(s) for s in sorted_sites])
        pbc_full = np.asarray(material_aware_pbc(mat_type), dtype=bool)
        image_offsets = _periodic_image_offsets(cell, pbc_full, tolerance)
        reps = _cluster_with_metric(
            n,
            coords,
            fps,
            image_offsets=image_offsets,
            kdtree_radius=tolerance,
        )
        result = [sorted_sites[i] for i in reps]

        # Prefer open pore sites (larger nn_distance) so early caps / stratified
        # samples are less likely to start inside framework walls.
        def _porous_priority(s: Site) -> tuple:
            xyz = _get_xyz(s)
            nn = float(s.nn_distance) if s.nn_distance is not None else -1.0
            pore_rank = 0 if s.site_type == "pore" else 1
            return (pore_rank, -nn, float(xyz[0]), float(xyz[1]), float(xyz[2]))

        return sorted(result, key=_porous_priority)

    z_tol = (
        z_abs_tolerance
        if z_abs_tolerance is not None
        else _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM
    )
    pinv_ab_T, _ = _slab_plane_projectors(cell)
    coords = np.array([_get_xyz(s) for s in sorted_sites])
    heights = _height_along_slab_normal(coords, cell)
    pbc_slab = np.asarray(material_aware_pbc(mat_type), dtype=bool)
    r_search = float(np.hypot(tolerance, z_tol))
    image_offsets = _periodic_image_offsets(cell, pbc_slab, r_search)
    n_hat = _slab_normal(cell)
    inv_cell = np.linalg.inv(cell)

    def _slab_pair_filter(a: int, b: int) -> bool:
        xyz_a = coords[a]
        xyz_b = coords[b]
        delta_frac = _minimum_image_fractional_delta(
            (xyz_b - xyz_a).reshape(1, 3) @ inv_cell,
            pbc_slab,
        )[0]
        delta_cart = _frac_to_cart(delta_frac.reshape(1, 3), cell)[0]
        # In-plane distance: drop the component along the slab normal.
        delta_plane = delta_cart - float(np.dot(delta_cart, n_hat)) * n_hat
        dxy = float(np.linalg.norm(delta_plane))
        dz = abs(float(heights[a]) - float(heights[b]))
        return dxy < tolerance and dz < z_tol

    reps = _cluster_with_metric(
        n,
        coords,
        fps,
        image_offsets=image_offsets,
        kdtree_radius=r_search,
        pair_filter=_slab_pair_filter,
    )
    result = [sorted_sites[i] for i in reps]

    # Batched heights for the representatives (one slab-normal + projection
    # instead of a cross product and matrix inverse per site).
    rep_xyz = np.asarray([_get_xyz(s) for s in result], dtype=float).reshape(-1, 3)
    rep_heights = _height_along_slab_normal(rep_xyz, cell)

    def _slab_coord(i: int) -> np.ndarray:
        xyz = rep_xyz[i]
        frac2 = xyz @ pinv_ab_T
        frac2 = frac2 - np.floor(frac2)
        # Use the height along the slab normal (not Cartesian z) so the ordered
        # representatives follow the surface, not an arbitrary tilted-z ordering.
        h = float(rep_heights[i])
        return np.array([float(frac2[0]), float(frac2[1]), h])

    def _slab_key(i: int) -> tuple:
        c = _slab_coord(i)
        return (float(c[0]), float(c[1]), float(c[2]), str(result[i].site_type))

    order_by_slab = sorted(range(len(result)), key=_slab_key)
    return [result[i] for i in order_by_slab]


# ---------------------------------------------------------------------------
# Symmetry-aware site reduction
# ---------------------------------------------------------------------------


def get_symmetry_aware_sites(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    symmetry_tolerance: float = _DEFAULT_SYMMETRY_TOLERANCE,
    *,
    material_type: str,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
    raw_sites: list[Site] | None = None,
    planar_z_variance_threshold: float | None = None,
    site_generator: str = "auto",
) -> list[Site]:
    """Return symmetry-reduced adsorption sites using spglib.

    Production sampling passes the **clustered** catalog via *raw_sites*
    (see :func:`~metalsurfer.placement.site_context.resolve_site_context_for_sampling`).
    Orbits are blocked by classified ``site_type`` only; ``site_source`` does
    not participate.

    Parameters
    ----------
    slab
        :class:`~ase.Atoms` substrate.
    top_layer_tolerance
        Height tolerance for the top layer (auto-derived if None).
    symmetry_tolerance
        Spatial tolerance for symmetry analysis.
    material_type
        ``"slab"``, ``"nanoparticle"``, or ``"porous"``. Required so that the
        symmetry mode and PBC semantics always match the caller's intent (same
        as :func:`get_unified_sites`): ``cluster`` for nanoparticles,
        ``periodic`` for slabs/porous (ignores ``slab.get_pbc()``).
    probe_radius
        Voronoi probe radius (auto-derived if None).
    max_site_distance
        Maximum site-to-atom distance (auto-derived if None).
    enrich
        Whether to enrich Voronoi ridge candidates.
    site_classification_method
        Site classification method (``"auto"``, ``"delaunay"``, etc.).
    raw_sites
        Optional pre-computed site list (typically the clustered catalog).
        When omitted, sites are enumerated via :func:`get_unified_sites`.
    planar_z_variance_threshold
        Max top-layer height variance (Å²) for planar classification.
        ``None`` uses the library default.
    site_generator
        Site generator plugin (``"auto"``, ``"topology"``, ``"voronoi"``).
    """
    validate_material_type(material_type)

    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(
            slab.get_chemical_symbols(),
        )
    z_var_threshold = (
        float(planar_z_variance_threshold)
        if planar_z_variance_threshold is not None
        else _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD
    )

    if raw_sites is not None:
        site_list = raw_sites
    else:
        site_list = get_unified_sites(
            slab,
            probe_radius=probe_radius,
            max_site_distance=max_site_distance,
            top_layer_tolerance=top_layer_tolerance,
            material_type=material_type,
            enrich=enrich,
            site_classification_method=site_classification_method,
            planar_z_variance_threshold=z_var_threshold,
            site_generator=site_generator,
        )
    if not site_list:
        return []

    sym_mode = "cluster" if material_type == "nanoparticle" else "periodic"
    planar_for_symmetry = False
    if material_type == "slab":
        positions = slab.get_positions()
        cell = np.asarray(slab.get_cell(), dtype=float)
        top_mask = top_layer_mask_by_normal(positions, cell, float(top_layer_tolerance))
        planar_for_symmetry = _top_layer_is_planar_from_arrays(
            positions,
            cell,
            float(top_layer_tolerance),
            z_var_threshold,
            top_mask=top_mask,
        )

    symmetry_analyzer = SymmetryAnalyzer(
        slab,
        symmetry_tolerance=symmetry_tolerance,
        mode=sym_mode,
    )
    return symmetry_analyzer.analyze_site_symmetry(
        site_list,
        planar=planar_for_symmetry,
    )


# ---------------------------------------------------------------------------
# Z-base range computation (used by generators.py)
# ---------------------------------------------------------------------------


def _get_site_surface_radii(
    slab: Atoms,
    site: Site | None = None,
    *,
    top_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> float:
    """Mean covalent radius of framework atoms nearest to the placement site."""
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    cell = np.asarray(slab.get_cell(), dtype=float)

    indices: tuple[int, ...] | None = None
    if site is not None and site.slab_indices:
        indices = tuple(int(i) for i in site.slab_indices)

    if indices is None:
        if top_indices is not None:
            indices = tuple(int(i) for i in top_indices)
        else:
            top_depth = _derive_top_layer_tolerance(symbols)
            top_mask = top_layer_mask_by_normal(positions, cell, float(top_depth))
            indices = tuple(int(i) for i in np.nonzero(top_mask)[0])

    site_symbols = [symbols[int(i)] for i in indices]
    radii = [_get_covalent_radius(s) for s in site_symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        logger.debug(
            "No positive covalent radii for site surface symbols %r (indices %r); "
            "using mean surface (framework) fallback %.3f Å",
            site_symbols,
            list(indices),
            _SURFACE_COVALENT_RADIUS_FALLBACK,
        )
        return float(_SURFACE_COVALENT_RADIUS_FALLBACK)
    return float(np.mean(valid))


def _compute_site_z_base(
    config,
    slab: Atoms,
    site: Site | None,
    mol_symbols: list[str],
    r_surface: float | None = None,
) -> tuple[float, float]:
    """Compute z-offset range for placement above *site*.

    When ``placement_z_scale_by_covalent_radius`` is True (default), each bound
    is ``placement_z_range[i] * (r_mol + r_surface)``. Otherwise the config
    tuple is returned as literal Å offsets.

    *r_surface* may be supplied to avoid recomputing the surface radius (e.g.
    when the caller already fetched it once per pose via
    :func:`_get_site_surface_radii`).
    """
    z_lo, z_hi = config.placement_z_range

    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi

    if r_surface is None:
        r_surface = _get_site_surface_radii(slab, site)
    r_mol = _mean_covalent_radius(
        mol_symbols, fallback=_ADSORBATE_COVALENT_RADIUS_FALLBACK
    )
    r_sum = r_mol + r_surface

    z_lo = float(z_lo) * r_sum
    z_hi = float(z_hi) * r_sum
    if z_hi < z_lo + _PARALLEL_Z_MIN_HI_MARGIN:
        z_hi = z_lo + _PARALLEL_Z_MIN_HI_MARGIN
    return z_lo, z_hi

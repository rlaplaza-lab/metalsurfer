"""Unified site enumeration, clustering, symmetry reduction, and z-base helpers."""

import logging
from collections.abc import Callable

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree, QhullError

from .._utils import cell_has_volume
from ..symmetry import SymmetryAnalyzer
from ._constants import (
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _BOUNDING_BOX_CELL_PAD_ANGSTROM,
    _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    _DEFAULT_SYMMETRY_TOLERANCE,
    _KD_RADIUS_SEARCH_PADDING,
    _MOL_COVALENT_RADIUS_FALLBACK,
    _NORMAL_K_NEIGHBOURS,
    _PARALLEL_Z_MIN_HI_MARGIN,
    _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM,
    _VORONOI_AUTO_WIDEN_MAX_SCALE,
    _VORONOI_AUTO_WIDEN_PROBE_SCALE,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_RADIUS_FALLBACK_ANGSTROM,
)
from ._material import (
    material_aware_pbc,
    material_type_for_placement,
    validate_material_type,
)
from .geometry import _get_covalent_radius
from .site_classify import (
    _build_site_records,
    _compute_local_normals_batch,
    _DelaunayClassifyInputs,
)
from .site_coords import (
    _build_periodic_images,
    _cart_to_frac,
    _deduplicate_points,
    _derive_top_layer_tolerance,
    _derive_voronoi_distance_window,
    _filter_non_duplicate_candidates,
    _frac_to_cart,
    _height_along_slab_normal,
    _minimum_image_fractional_delta,
    _periodic_image_offsets,
    _project_to_slab_plane,
    _shift_along_slab_normal,
    _slab_normal,
    _slab_plane_projectors,
    _union_find_cluster,
    _wrap_cartesian,
    _wrap_fractional,
    derive_pore_threshold,
    top_layer_mask_by_normal,
)
from .site_types import Site
from .site_voronoi import (
    _build_delaunay_classification_index,
    _generate_slab_topology_sites,
    _voronoi_sites,
)

logger = logging.getLogger(__name__)


def _merge_dedup_site_arrays(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    new_vertices: np.ndarray,
    new_dists: np.ndarray,
    new_sources: list[str],
    *,
    cell: np.ndarray,
    pbc: np.ndarray | list[bool],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Append *new_* site arrays and deduplicate against the combined set."""
    if len(new_vertices) == 0:
        return vertices, nn_dists, source_hints
    if len(vertices) == 0:
        return new_vertices, new_dists, list(new_sources)
    combined = np.vstack([vertices, new_vertices])
    combined_dists = np.concatenate([nn_dists, new_dists])
    combined_sources = source_hints + list(new_sources)
    keep = _deduplicate_points(combined, _VORONOI_DEDUP_TOLERANCE, cell=cell, pbc=pbc)
    kept_idx = np.nonzero(keep)[0]
    return combined[keep], combined_dists[keep], [combined_sources[i] for i in kept_idx]


def _median_nn_or_fallback(
    nn_dists: np.ndarray,
    *,
    reference_positions: np.ndarray | None = None,
    cell: np.ndarray | None = None,
    pbc: np.ndarray | None = None,
) -> float:
    """Median nearest-neighbour distance, or top-layer / covalent fallback.

    When *nn_dists* is empty (e.g. planar slabs that skip Voronoi), prefer the
    median MIC nearest-neighbour spacing of *reference_positions* (typically the
    top layer) over a fixed covalent-scale constant.
    """
    if len(nn_dists) > 0:
        return float(np.median(nn_dists))
    if (
        reference_positions is not None
        and len(reference_positions) >= 2
        and cell is not None
        and pbc is not None
    ):
        pts = np.asarray(reference_positions, dtype=float)
        pbc_arr = np.asarray(pbc, dtype=bool)
        cell_arr = np.asarray(cell, dtype=float)
        if cell_has_volume(cell_arr) and np.any(pbc_arr):
            margin = float(np.max(np.linalg.norm(cell_arr[pbc_arr], axis=1)))
            ext = _build_periodic_images(pts, cell_arr, pbc_arr, margin=margin)
            tree = KDTree(ext)
            dists, idxs = tree.query(pts, k=min(len(ext), 2))
            dists = np.atleast_2d(np.asarray(dists, dtype=float))
            idxs = np.atleast_2d(np.asarray(idxs))
            n = len(pts)
            valid = (idxs % n) != np.arange(n)[:, None]
            nn = np.where(valid, dists, np.inf).min(axis=1)
            finite = nn[np.isfinite(nn)]
            if len(finite) > 0:
                return float(np.median(finite))
        else:
            tree = KDTree(pts)
            nn_d, _ = tree.query(pts, k=2)
            return float(np.median(np.asarray(nn_d, dtype=float)[:, 1]))
    return _VORONOI_MAX_DISTANCE_COVALENT_SCALE * _VORONOI_RADIUS_FALLBACK_ANGSTROM


def _periodic_accessibility_tree(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    max_distance: float,
) -> KDTree:
    """KDTree over periodic images, for candidate-to-framework distance gating.

    A plain ``KDTree(positions)`` reports the *in-cell* nearest-neighbour
    distance, which overestimates the true minimum-image distance for
    candidates near an a/b boundary. The image margin covers ``max_distance``
    so any candidate that would pass the accessibility window is found.
    """
    if not np.any(pbc) or not cell_has_volume(cell):
        return KDTree(positions)
    margin = float(max_distance) + _VORONOI_DEDUP_TOLERANCE
    return KDTree(_build_periodic_images(positions, cell, pbc, margin=margin))


def _delaunay_classify_inputs(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    material_type: str,
    site_classification_method: str,
    slab_top_atom_indices: np.ndarray,
    topology_primary_delaunay: Delaunay | None,
    expanded_xy: np.ndarray | None = None,
    expanded_origin: list[int] | None = None,
    expanded_tri: Delaunay | None = None,
) -> _DelaunayClassifyInputs | None:
    """Build prebuilt Delaunay classification inputs, or ``None`` when disabled."""
    if site_classification_method == "delaunay" and material_type != "slab":
        logger.warning(
            "site_classification_method='delaunay' is slab-only; "
            "falling back to distance-ratio classification for material_type=%r",
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


def _apply_site_mask(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    source_hints: list[str],
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Keep only vertices selected by boolean *mask* (index arrays stay aligned)."""
    kept = np.nonzero(mask)[0]
    return vertices[mask], nn_dists[mask], [source_hints[i] for i in kept]


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
    slab_top_atom_indices: np.ndarray | None,
    slab_has_topology_atop: bool,
    probe_radius: float,
    max_site_distance: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Atop injection safety net for nanoparticles, and slabs lacking topology atop."""
    if material_type == "slab" and slab_has_topology_atop:
        return vertices, nn_dists, source_hints
    if material_type not in ("slab", "nanoparticle") or len(vertices) == 0:
        return vertices, nn_dists, source_hints

    median_nn = _median_nn_or_fallback(
        nn_dists,
        reference_positions=(
            positions[slab_top_atom_indices]
            if slab_top_atom_indices is not None
            else positions
        ),
        cell=cell,
        pbc=pbc,
    )
    atop_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn

    if material_type == "slab":
        if slab_top_atom_indices is None:
            raise ValueError(
                "slab_top_atom_indices must be set for slab atop injection"
            )
        top_atom_indices = slab_top_atom_indices
        atom_normals: np.ndarray | None = None
    else:
        com = np.mean(positions, axis=0)
        k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
        _, norm_idx_all = local_tree.query(positions, k=k_norm)
        if np.ndim(norm_idx_all) == 1:
            norm_idx_all = np.asarray(norm_idx_all, dtype=int).reshape(-1, 1)
        atom_normals = _compute_local_normals_batch(positions, positions, norm_idx_all)
        outward_dots = np.einsum("ij,ij->i", atom_normals, positions - com)
        top_atom_indices = np.nonzero(outward_dots > 0.0)[0].astype(int)

    candidate_verts: list[np.ndarray] = []
    for ai in top_atom_indices:
        atom_pos = positions[int(ai)]
        if material_type == "slab":
            candidate = _shift_along_slab_normal(
                atom_pos.reshape(1, 3), cell, atop_height
            )[0]
            if np.any(pbc):
                candidate = _wrap_cartesian(candidate.reshape(1, 3), cell, pbc)[0]
        else:
            if atom_normals is None:
                raise ValueError(
                    "atom_normals must be set for nanoparticle atop injection"
                )
            candidate = atom_pos + atop_height * atom_normals[int(ai)]
        candidate_verts.append(candidate)

    if not candidate_verts:
        return vertices, nn_dists, source_hints

    candidate_arr = np.asarray(candidate_verts, dtype=float)
    d_nn_all = np.asarray(local_tree.query(candidate_arr, k=1)[0], dtype=float).ravel()
    keep_acc = (d_nn_all >= float(probe_radius)) & (
        d_nn_all <= float(max_site_distance)
    )
    if not np.any(keep_acc):
        return vertices, nn_dists, source_hints

    candidate_arr = candidate_arr[keep_acc]
    candidate_dist_arr = d_nn_all[keep_acc]
    candidate_sources = ["atop_injected"] * int(np.count_nonzero(keep_acc))

    keep_new = _filter_non_duplicate_candidates(
        candidate_arr, vertices, _VORONOI_DEDUP_TOLERANCE
    )
    candidate_arr = candidate_arr[keep_new]
    candidate_dist_arr = candidate_dist_arr[keep_new]
    candidate_sources = [candidate_sources[i] for i in np.nonzero(keep_new)[0]]
    if len(candidate_arr) > 0:
        n_existing = len(vertices)
        vertices, nn_dists, source_hints = _merge_dedup_site_arrays(
            vertices,
            nn_dists,
            source_hints,
            candidate_arr,
            candidate_dist_arr,
            candidate_sources,
            cell=cell,
            pbc=pbc,
        )
        n_injected = len(vertices) - n_existing
        logger.debug(
            "Injected %d atop candidate sites (%d total sites)",
            max(n_injected, 0),
            len(vertices),
        )

    return vertices, nn_dists, source_hints


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
) -> list[Site]:
    """Return adsorption/placement sites for *atoms*.

    Per-material behaviour
    ----------------------
    - **slab**: sites come from the topology generator
      (:func:`_generate_slab_topology_sites`), which enumerates atop, bridge and
      hollow candidates from the top layer including its ±1 a/b periodic images.
      A flat top layer is coplanar, so Qhull cannot build a 3D Voronoi diagram
      from it (``QH6154 initial simplex is flat``); the Voronoi pass is therefore
      skipped explicitly for planar slabs rather than being attempted and
      swallowed. ``voronoi_probe_radius``, ``voronoi_max_site_distance`` and
      ``auto_widen`` **remain active** on slabs — they drive the
      topology accessibility window and the empty-result retry. Only ``enrich``
      (``voronoi_site_enrichment``) has no effect on planar slabs.
    - **porous** / **nanoparticle**: Voronoi vertices are the primary source,
      with optional ridge enrichment, plus an atop-injection safety net for
      nanoparticles.
    - rotated slabs are handled using the slab normal rather than Cartesian z

    Parameters
    ----------
    atoms
        :class:`~ase.Atoms` structure to detect sites on.
    probe_radius
        Voronoi probe radius (auto-derived if None).
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
    """
    sites = _enumerate_unified_sites(
        atoms,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        pore_threshold=pore_threshold,
        enrich=enrich,
        site_classification_method=site_classification_method,
    )
    if sites or not auto_widen:
        return sites

    positions = atoms.get_positions()
    symbols = list(atoms.get_chemical_symbols())
    cell = np.asarray(atoms.get_cell(), dtype=float)
    mat = material_type if material_type is not None else "slab"
    pbc = np.asarray(material_aware_pbc(mat), dtype=bool)
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
    return _enumerate_unified_sites(
        atoms,
        probe_radius=wide_probe,
        max_site_distance=wide_max,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        pore_threshold=pore_threshold,
        enrich=enrich,
        site_classification_method=site_classification_method,
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
) -> list[Site]:
    """Core site enumeration (single pass, no auto-widen)."""
    if material_type is None:
        raise ValueError(
            "material_type must be explicitly specified: 'slab', 'nanoparticle', or 'porous'"
        )
    validate_material_type(material_type)

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
    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(symbols)
    if pore_threshold is None:
        pore_threshold = derive_pore_threshold(symbols)

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

    voronoi_positions = positions
    slab_top_atom_indices: np.ndarray | None = None
    slab_skip_voronoi = False
    if material_type == "slab":
        # Compute once; reused for Voronoi crop, topology, and atop injection.
        slab_top_mask = top_layer_mask_by_normal(
            positions, cell, float(top_layer_tolerance)
        )
        slab_top_atom_indices = np.nonzero(slab_top_mask)[0]
        top_only = positions[slab_top_mask]
        if len(top_only) >= 4:
            voronoi_positions = top_only
        # A coplanar top layer has no 3D Voronoi diagram: Qhull raises
        # "QH6154 initial simplex is flat" and the generator returns zero
        # vertices. Skip it explicitly instead of raising-and-swallowing.
        slab_skip_voronoi = _top_layer_is_planar_from_arrays(
            positions, cell, float(top_layer_tolerance)
        )
        if slab_skip_voronoi:
            logger.info(
                "Slab top layer is planar; skipping Voronoi vertex generation. "
                "Slab sites come from the topology generator "
                "(atop/bridge/hollow). probe_radius/max_site_distance still "
                "gate accessibility; site enrichment does not apply."
            )

    if slab_skip_voronoi:
        vertices = np.empty((0, 3), dtype=float)
        nn_dists = np.empty(0, dtype=float)
    else:
        vertices, nn_dists = _voronoi_sites(
            voronoi_positions,
            cell,
            pbc_for_voronoi,
            probe_radius=probe_radius,
            max_distance=max_site_distance,
            enrich=enrich,
            symbols=symbols,
        )
    source_hints = ["voronoi"] * len(vertices)

    local_tree = KDTree(positions)

    slab_has_topology_atop = False
    topology_primary_delaunay = None
    topology_expanded_xy: np.ndarray | None = None
    topology_expanded_origin: list[int] | None = None
    topology_expanded_tri = None

    # Slab-specific topology enrichment becomes part of the default generator.
    if material_type == "slab" and slab_top_atom_indices is not None:
        median_nn = _median_nn_or_fallback(
            nn_dists,
            reference_positions=positions[slab_top_atom_indices],
            cell=cell,
            pbc=pbc_for_voronoi,
        )
        site_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn
        # The accessibility gate compares candidate-to-framework distances, so
        # it must respect PBC: a hollow whose defining atoms straddle an a/b
        # boundary otherwise gets an inflated d_nn and is silently dropped.
        accessibility_tree = _periodic_accessibility_tree(
            positions,
            cell,
            pbc_for_voronoi,
            float(max_site_distance),
        )
        (
            topo_vertices,
            topo_dists,
            topo_sources,
            topology_primary_delaunay,
            topology_expanded_xy,
            topology_expanded_origin,
            topology_expanded_tri,
        ) = _generate_slab_topology_sites(
            positions,
            cell,
            pbc_for_voronoi,
            slab_top_atom_indices,
            accessibility_tree,
            site_height,
            float(probe_radius),
            float(max_site_distance),
        )
        slab_has_topology_atop = any(s == "topology_atop" for s in topo_sources)
        if len(topo_vertices) > 0:
            vertices, nn_dists, source_hints = _merge_dedup_site_arrays(
                vertices,
                nn_dists,
                source_hints,
                topo_vertices,
                topo_dists,
                topo_sources,
                cell=cell,
                pbc=pbc_for_voronoi,
            )

    if len(vertices) == 0:
        logger.warning(
            "No accessible sites for %d-atom structure (probe_radius=%s, max_distance=%s, material_type=%r)",
            len(atoms),
            f"{probe_radius:.2f}" if probe_radius is not None else "auto",
            f"{max_site_distance:.2f}" if max_site_distance is not None else "auto",
            material_type,
        )
        return []

    if material_type == "slab":
        heights = _height_along_slab_normal(positions, cell)
        h_surface = float(np.max(heights))
        nn_margin = (
            float(np.median(nn_dists))
            if len(nn_dists) > 0
            else float(top_layer_tolerance)
        )
        h_min = h_surface - max(float(top_layer_tolerance), nn_margin)
        keep_mask = _height_along_slab_normal(vertices, cell) >= h_min
        vertices, nn_dists, source_hints = _apply_site_mask(
            vertices, nn_dists, source_hints, keep_mask
        )

    if material_type == "nanoparticle" and len(vertices) > 0:
        com = np.mean(positions, axis=0)
        k_norm = min(_NORMAL_K_NEIGHBOURS, len(positions))
        _, norm_idx = local_tree.query(vertices, k=k_norm)
        if np.ndim(norm_idx) == 1:
            norm_idx = np.asarray(norm_idx, dtype=int).reshape(-1, 1)
        normals = _compute_local_normals_batch(vertices, positions, norm_idx)
        outward = np.einsum("ij,ij->i", normals, vertices - com) > 0.0
        vertices, nn_dists, source_hints = _apply_site_mask(
            vertices, nn_dists, source_hints, outward
        )

    # Atop injection safety net for nanoparticles; for slabs only when topology
    # did not already produce atop candidates.
    vertices, nn_dists, source_hints = _inject_atop_sites(
        vertices,
        nn_dists,
        source_hints,
        positions=positions,
        cell=cell,
        pbc=pbc_for_voronoi,
        material_type=material_type,
        local_tree=local_tree,
        slab_top_atom_indices=slab_top_atom_indices,
        slab_has_topology_atop=slab_has_topology_atop,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
    )

    if len(vertices) == 0:
        return []

    # ``auto`` / ``delaunay`` use Delaunay on slabs; ``distance_ratio`` is honored
    # literally (opt-in A/B). Default config ``auto`` preserves catalysis sampling.
    delaunay_inputs = _delaunay_classify_inputs(
        positions,
        cell,
        pbc_for_voronoi,
        material_type=material_type,
        site_classification_method=site_classification_method,
        slab_top_atom_indices=slab_top_atom_indices,
        topology_primary_delaunay=topology_primary_delaunay,
        expanded_xy=topology_expanded_xy,
        expanded_origin=topology_expanded_origin,
        expanded_tri=topology_expanded_tri,
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
    )

    if cell_has_volume(cell):

        def _site_frac_key(site: Site) -> tuple:
            frac = _wrap_fractional(
                _cart_to_frac(np.asarray(site.xyz, dtype=float).reshape(1, 3), cell),
                pbc_for_voronoi,
            )[0]
            return (
                float(frac[0]),
                float(frac[1]),
                float(frac[2]),
                str(site.site_type),
            )

        sites.sort(key=_site_frac_key)

    return sites


# ---------------------------------------------------------------------------
# Hollow sites for adatom / dissociative placement
# ---------------------------------------------------------------------------


def get_hollow_sites_for_adatoms(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    dedup_tolerance: float = _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    *,
    material_type: str = "slab",
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
) -> list[Site]:
    """Return hollow/pore sites for adatom placement, deduplicated.

    Parameters
    ----------
    slab
        :class:`~ase.Atoms` substrate.
    top_layer_tolerance
        Height tolerance for the top layer (auto-derived if None).
    dedup_tolerance
        Spatial tolerance for deduplication.
    material_type
        ``"slab"``, ``"nanoparticle"``, or ``"porous"``.
    probe_radius
        Voronoi probe radius (auto-derived if None).
    max_site_distance
        Maximum site-to-atom distance (auto-derived if None).
    enrich
        Whether to enrich Voronoi ridge candidates.
    site_classification_method
        Site classification method (``"auto"``, ``"delaunay"``, etc.).
    """
    raw = get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        enrich=enrich,
        site_classification_method=site_classification_method,
    )
    hollow_sites = [s for s in raw if s.site_type in ("hollow", "pore")]
    if not hollow_sites:
        return []
    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc(material_type), dtype=bool)
    hollow_xyz = np.array([s.xyz for s in hollow_sites], dtype=float)
    keep = _deduplicate_points(hollow_xyz, dedup_tolerance, cell=cell, pbc=pbc)
    return [hollow_sites[i] for i in np.nonzero(keep)[0]]


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
    pair_filter: Callable[[int, int, np.ndarray], bool] | None = None,
) -> list[int]:
    """KDTree query_pairs + union-find; one representative index per cluster."""
    if image_offsets:
        all_coords = np.vstack([coords + off for off in image_offsets])
    else:
        all_coords = coords

    tree = KDTree(all_coords)
    raw_pairs = tree.query_pairs(r=kdtree_radius, output_type="ndarray")

    merge_set: set[tuple[int, int]] = set()
    for a_exp, b_exp in raw_pairs:
        a = int(a_exp) % n
        b = int(b_exp) % n
        if a == b:
            continue
        if fps[a] != fps[b]:
            continue
        if pair_filter is not None and not pair_filter(a, b, coords):
            continue
        merge_set.add((min(a, b), max(a, b)))

    components = _union_find_cluster(n, list(merge_set))
    return sorted(min(comp) for comp in components)


def _cluster_equivalent_sites(
    sites: list[Site],
    cell: np.ndarray,
    tolerance: float = _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    z_abs_tolerance: float | None = None,
) -> list[Site]:
    """Group equivalent sites; return unique representatives."""
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
        pbc_full = np.array([True, True, True])
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
    pbc_slab = np.array([True, True, False])
    image_offsets = _periodic_image_offsets(cell, pbc_slab, tolerance)

    def _slab_pair_filter(a: int, b: int, _coords_arr: np.ndarray) -> bool:
        xyz_a = coords[a]
        xyz_b = coords[b]
        delta_frac = _minimum_image_fractional_delta(
            _cart_to_frac((xyz_b - xyz_a).reshape(1, 3), cell),
            pbc_slab,
        )[0]
        delta_cart = _frac_to_cart(delta_frac.reshape(1, 3), cell)[0]
        # In-plane distance: drop the component along the slab normal.
        n_hat = _slab_normal(cell)
        delta_plane = delta_cart - float(np.dot(delta_cart, n_hat)) * n_hat
        dxy = float(np.linalg.norm(delta_plane))
        dz = abs(float(heights[a]) - float(heights[b]))
        return dxy < tolerance and dz < z_tol

    r_search = tolerance * _KD_RADIUS_SEARCH_PADDING
    reps = _cluster_with_metric(
        n,
        coords,
        fps,
        image_offsets=image_offsets,
        kdtree_radius=r_search,
        pair_filter=_slab_pair_filter,
    )
    result = [sorted_sites[i] for i in reps]

    def _slab_coord(s: Site) -> np.ndarray:
        xyz = _get_xyz(s)
        frac2 = xyz @ pinv_ab_T
        frac2 = frac2 - np.floor(frac2)
        z = float(s.z)
        return np.array([float(frac2[0]), float(frac2[1]), z])

    def _slab_key(s: Site) -> tuple:
        c = _slab_coord(s)
        return (float(c[0]), float(c[1]), float(c[2]), str(s.site_type))

    return sorted(result, key=_slab_key)


# ---------------------------------------------------------------------------
# Symmetry-aware site reduction
# ---------------------------------------------------------------------------


def get_symmetry_aware_sites(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    symmetry_tolerance: float = _DEFAULT_SYMMETRY_TOLERANCE,
    material_type: str = "slab",
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "auto",
    raw_sites: list[Site] | None = None,
) -> list[Site]:
    """Return symmetry-reduced adsorption sites using spglib.

    Parameters
    ----------
    slab
        :class:`~ase.Atoms` substrate.
    top_layer_tolerance
        Height tolerance for the top layer (auto-derived if None).
    symmetry_tolerance
        Spatial tolerance for symmetry analysis.
    material_type
        ``"slab"``, ``"nanoparticle"``, or ``"porous"``.
    probe_radius
        Voronoi probe radius (auto-derived if None).
    max_site_distance
        Maximum site-to-atom distance (auto-derived if None).
    enrich
        Whether to enrich Voronoi ridge candidates.
    site_classification_method
        Site classification method (``"auto"``, ``"delaunay"``, etc.).
    raw_sites
        Optional pre-computed raw site list.
    """
    validate_material_type(material_type)

    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(
            slab.get_chemical_symbols(),
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
        )
    if not site_list:
        return []

    sym_mode = "cluster" if material_type == "nanoparticle" else "auto"
    planar_for_symmetry = (material_type == "slab") and _is_top_layer_planar(
        slab, top_layer_tolerance
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
# Internal helpers
# ---------------------------------------------------------------------------


def _top_layer_is_planar_from_arrays(
    positions: np.ndarray,
    cell: np.ndarray,
    top_layer_tolerance: float = _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
) -> bool:
    """Check whether the topmost atomic layer of *positions* is approximately flat.

    The fit is done in an orientation-aware slab coordinate system rather than
    assuming the slab normal is Cartesian z.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    top_mask = top_layer_mask_by_normal(positions, cell, float(top_layer_tolerance))
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return False
    top_pos = positions[top_indices]
    xy = _project_to_slab_plane(top_pos, cell)
    h = _height_along_slab_normal(top_pos, cell)
    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])
    coeffs, _residuals, rank, _ = np.linalg.lstsq(A, h, rcond=None)
    if rank < 3:
        return False
    h_pred = A @ coeffs
    return float(np.var(h - h_pred)) < z_variance_threshold


def _is_top_layer_planar(
    slab: Atoms,
    top_layer_tolerance: float = _PLANAR_TOP_LAYER_TOLERANCE_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
) -> bool:
    """Check whether the topmost atomic layer is approximately flat."""
    return _top_layer_is_planar_from_arrays(
        slab.get_positions(),
        np.asarray(slab.get_cell(), dtype=float),
        top_layer_tolerance,
        z_variance_threshold,
    )


def _bounding_box_cell(
    positions: np.ndarray,
    pad: float = _BOUNDING_BOX_CELL_PAD_ANGSTROM,
) -> np.ndarray:
    """Orthorhombic cell spanning atomic positions plus padding."""
    lo = positions.min(axis=0)
    hi = positions.max(axis=0)
    span = np.maximum(hi - lo + pad, pad)
    return np.diag(span)


# ---------------------------------------------------------------------------
# Z-base range computation (used by generators.py)
# ---------------------------------------------------------------------------


def _get_site_surface_radii(
    slab: Atoms,
    site: Site | None = None,
) -> float:
    """Mean covalent radius of framework atoms nearest to the placement site."""
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    cell = np.asarray(slab.get_cell(), dtype=float)

    indices: tuple[int, ...] | None = None
    if site is not None and site.slab_indices:
        indices = tuple(int(i) for i in site.slab_indices)

    if indices is None:
        top_depth = _derive_top_layer_tolerance(symbols)
        top_mask = top_layer_mask_by_normal(positions, cell, float(top_depth))
        indices = tuple(int(i) for i in np.nonzero(top_mask)[0])

    radii = [_get_covalent_radius(symbols[int(i)]) for i in indices]
    radii = [r for r in radii if r is not None]
    if not radii:
        site_symbols = [symbols[int(i)] for i in indices]
        logger.warning(
            "No positive covalent radii for site surface symbols %r (indices %r); "
            "using mean adsorbate fallback %.3f Å",
            site_symbols,
            list(indices),
            _MOL_COVALENT_RADIUS_FALLBACK,
        )
        return float(_MOL_COVALENT_RADIUS_FALLBACK)
    return float(np.mean(radii))


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
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else _MOL_COVALENT_RADIUS_FALLBACK
    r_sum = r_mol + r_surface

    z_lo = float(z_lo) * r_sum
    z_hi = float(z_hi) * r_sum
    if z_hi < z_lo + _PARALLEL_Z_MIN_HI_MARGIN:
        z_hi = z_lo + _PARALLEL_Z_MIN_HI_MARGIN
    return z_lo, z_hi

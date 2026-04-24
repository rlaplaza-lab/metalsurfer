"""Voronoi sites, clustering, and optional spglib-based symmetry reduction."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree, QhullError, Voronoi

from ..symmetry import SymmetryAnalyzer
from ._constants import (
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _ATOP_RATIO,
    _BOUNDING_BOX_CELL_PAD_ANGSTROM,
    _BRIDGE_EQ_TOL,
    _BRIDGE_FAR_RATIO,
    _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
    _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
    _DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    _DEFAULT_SYMMETRY_TOLERANCE,
    _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
    _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM,
    _DISTANCE_RATIO_FLOOR_EPS,
    _DISTANCE_ZERO_EPS,
    _ENRICHMENT_MAX_SUBDIVISIONS,
    _ENRICHMENT_SPACING_BETA,
    _HOLLOW_EQ_TOL,
    _KD_RADIUS_SEARCH_PADDING,
    _MOL_COVALENT_RADIUS_FALLBACK,
    _NORMAL_K_NEIGHBOURS,
    _NON_SLAB_Z_HI_FROM_NN_SCALE,
    _NON_SLAB_Z_LO_FROM_NN_SCALE,
    _PORE_THRESHOLD_COVALENT_SCALE,
    _PORE_THRESHOLD_MIN_ANGSTROM,
    _SITE_CLASSIFICATION_NEIGHBOURS,
    _SITE_Z_RADIUS_REFERENCE_ANGSTROM,
    _SITE_Z_RADIUS_SHIFT_SCALE,
    _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM,
    _SURFACE_NORMAL_FALLBACK_NORM_EPS,
    _TOP_LAYER_DEPTH_COVALENT_SCALE,
    _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    _VORONOI_FRACTIONAL_CELL_MARGIN,
    _VORONOI_DEDUP_TOLERANCE,
    _VORONOI_MAX_DISTANCE_COVALENT_SCALE,
    _VORONOI_PROBE_RADIUS_COVALENT_SCALE,
    _VORONOI_RADIUS_FALLBACK_ANGSTROM,
)
from ._material import detect_material_type, material_type_for_placement
from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)

DEFAULT_SYMMETRY_TOLERANCE = _DEFAULT_SYMMETRY_TOLERANCE
DEFAULT_SITE_EQUIVALENCE_TOLERANCE = _DEFAULT_SITE_EQUIVALENCE_TOLERANCE
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE = _DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE


# ---------------------------------------------------------------------------
# Periodic image generation
# ---------------------------------------------------------------------------


def _build_periodic_images(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> np.ndarray:
    """Return extended positions including periodic images.

    slab (pbc=[T,T,F/T]): 3x3x1 -> 9N points.
    porous (pbc=[T,T,T]):  3x3x3 -> 27N points.
    nanoparticle:          no images -> N points.
    """
    extended = []
    ranges = [([-1, 0, 1] if pbc[d] else [0]) for d in range(3)]
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                offset = i * cell[0] + j * cell[1] + k * cell[2]
                extended.append(positions + offset)
    return np.vstack(extended)


# ---------------------------------------------------------------------------
# Voronoi site generation
# ---------------------------------------------------------------------------


def _deduplicate_points(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Return a boolean keep-mask that removes near-duplicate points."""
    if len(points) == 0:
        return np.ones(0, dtype=bool)
    ded_tree = KDTree(points)
    pairs = ded_tree.query_pairs(r=tolerance, output_type="ndarray")
    keep = np.ones(len(points), dtype=bool)
    for i, j in pairs:
        if keep[i] and keep[j]:
            keep[j] = False
    return keep


def _is_duplicate_of(
    candidate: np.ndarray,
    tree: KDTree | None,
    tolerance: float,
) -> bool:
    """True when *candidate* is within *tolerance* of any point in *tree*."""
    if tree is None:
        return False
    nearest = float(tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0])
    return nearest < tolerance


def _mean_covalent_radius(symbols: list[str]) -> float:
    radii = [_get_covalent_radius(s) for s in symbols]
    valid = [r for r in radii if r is not None]
    if not valid:
        return _VORONOI_RADIUS_FALLBACK_ANGSTROM
    return float(np.mean(valid))


def _derive_voronoi_distance_window(
    positions: np.ndarray,
    symbols: list[str],
    pbc: np.ndarray,
) -> tuple[float, float]:
    if len(positions) == 0:
        base_radius = _VORONOI_RADIUS_FALLBACK_ANGSTROM
    elif bool(pbc[2]):
        z_max = float(np.max(positions[:, 2]))
        top_depth = _TOP_LAYER_DEPTH_MIN_ANGSTROM
        top_mask = positions[:, 2] >= (z_max - top_depth)
        top_idx = np.nonzero(top_mask)[0]
        top_symbols = [symbols[int(i)] for i in top_idx] if len(top_idx) else symbols
        base_radius = _mean_covalent_radius(top_symbols)
    else:
        base_radius = _mean_covalent_radius(symbols)

    probe_radius = _VORONOI_PROBE_RADIUS_COVALENT_SCALE * base_radius
    max_distance = _VORONOI_MAX_DISTANCE_COVALENT_SCALE * base_radius
    return float(probe_radius), float(max(max_distance, probe_radius))


def _derive_top_layer_tolerance(
    positions: np.ndarray,
    symbols: list[str],
) -> float:
    mean_radius = _mean_covalent_radius(symbols)
    return max(_TOP_LAYER_DEPTH_MIN_ANGSTROM, _TOP_LAYER_DEPTH_COVALENT_SCALE * mean_radius)


def _derive_pore_threshold(symbols: list[str]) -> float:
    """Return pore classification threshold from mean covalent radius."""
    mean_radius = _mean_covalent_radius(symbols)
    return max(_PORE_THRESHOLD_MIN_ANGSTROM, _PORE_THRESHOLD_COVALENT_SCALE * mean_radius)


def _voronoi_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    probe_radius: float | None = None,
    max_distance: float | None = None,
    enrich: bool = True,
    symbols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Voronoi vertices accessible for adsorption, optionally enriched.

    When *enrich* is True, sparse edges of the Voronoi graph are subdivided
    with intermediate points that pass the same accessibility checks.  This
    improves coverage on rugged surfaces, stepped slabs, and porous
    materials without changing the site representation.

    Returns (vertices, nn_distances) where vertices is (M, 3) and
    nn_distances is (M,) - distance to nearest framework atom for each site.
    """
    if len(positions) < 4:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)
    if symbols is None:
        symbols = ["C"] * len(positions)
    if probe_radius is None or max_distance is None:
        derived_probe, derived_max = _derive_voronoi_distance_window(positions, symbols, pbc)
        probe_radius = derived_probe if probe_radius is None else probe_radius
        max_distance = derived_max if max_distance is None else max_distance

    extended = _build_periodic_images(positions, cell, pbc)
    try:
        vor = Voronoi(extended)
    except (QhullError, ValueError, RuntimeError) as exc:
        logger.debug("Voronoi computation failed (%s); returning no vertices", exc)
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    raw_vertices = vor.vertices

    # Build a map from raw vertex index to unit-cell-filtered index so we
    # can translate Voronoi ridge_vertices into the kept-vertex domain.
    if np.linalg.det(cell) > 0:
        inv_cell = np.linalg.inv(cell)
        frac = raw_vertices @ inv_cell.T
        inside = np.ones(len(frac), dtype=bool)
        for dim in range(3):
            if pbc[dim]:
                inside &= (
                    frac[:, dim] >= -_VORONOI_FRACTIONAL_CELL_MARGIN
                ) & (frac[:, dim] < 1.0 + _VORONOI_FRACTIONAL_CELL_MARGIN)
        inside_indices = np.nonzero(inside)[0]
        raw_vertices = raw_vertices[inside_indices]
    else:
        inside_indices = np.arange(len(raw_vertices))

    if len(raw_vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    # Use periodic images for nearest-neighbour filtering so boundary-adjacent
    # vertices get physically correct nearest distances under PBC.
    tree = KDTree(extended)
    nn_dists, _ = tree.query(raw_vertices, k=1)
    nn_dists = nn_dists.ravel()

    accessible = (nn_dists >= probe_radius) & (nn_dists <= max_distance)
    vertices = raw_vertices[accessible]
    nn_dists = nn_dists[accessible]

    if len(vertices) == 0:
        return np.empty((0, 3), dtype=float), np.empty(0, dtype=float)

    # Deduplicate
    keep = _deduplicate_points(vertices, _VORONOI_DEDUP_TOLERANCE)
    vertices = vertices[keep]
    nn_dists = nn_dists[keep]

    if not enrich or len(vertices) < 2:
        return vertices, nn_dists

    # --- Ridge-based geodesic enrichment ---
    # Map: original Voronoi vertex index -> index in kept vertices (or -1)
    accessible_of_inside = np.nonzero(accessible)[0]
    kept_of_accessible = np.nonzero(keep)[0]
    # Full chain: raw vor index -> inside index -> accessible index -> kept index
    raw_to_kept = {}
    for kept_idx, acc_idx in enumerate(kept_of_accessible):
        inside_idx = accessible_of_inside[acc_idx]
        raw_idx = int(inside_indices[inside_idx])
        raw_to_kept[raw_idx] = kept_idx

    enriched_verts, enriched_dists = _enrich_along_ridges(
        vertices,
        nn_dists,
        vor.ridge_vertices,
        raw_to_kept,
        extended,
        tree,
        probe_radius,
        max_distance,
    )
    return enriched_verts, enriched_dists


# ---------------------------------------------------------------------------
# Ridge-based geodesic enrichment
# ---------------------------------------------------------------------------

# Target spacing factor and subdivision cap are imported from _constants.
# _ENRICHMENT_SPACING_BETA and _ENRICHMENT_MAX_SUBDIVISIONS are used below.


def _enrich_along_ridges(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    ridge_vertices: list[list[int]],
    raw_to_kept: dict[int, int],
    extended_positions: np.ndarray,
    framework_tree: KDTree,
    probe_radius: float,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Subdivide long admissible Voronoi edges and re-check accessibility.

    Only edges whose endpoints are both kept (accessible, in unit cell) are
    considered.  An edge is admissible when the support atom sets of its
    endpoints share at least one atom, preventing interpolation across
    disconnected surfaces or separate pore channels.

    Parameters
    ----------
    vertices : (N, 3) array
        Kept Voronoi vertices.
    nn_dists : (N,) array
        Nearest-framework-atom distance for each kept vertex.
    ridge_vertices : sequence of sequences
        Voronoi ridge endpoint lists (indices into the *original* vor.vertices).
        Unbounded ridges include ``-1`` and are ignored.
    raw_to_kept : dict
        Mapping from original Voronoi vertex index to kept-vertex index.
    extended_positions : (E, 3) array
        Framework atom positions including periodic images.
    framework_tree : KDTree
        KDTree built on *extended_positions*.
    probe_radius, max_distance : float
        Same accessibility window used for original vertices.

    Returns
    -------
    (vertices_enriched, nn_dists_enriched) with the same layout as the input
    but with additional interpolated sites appended.
    """
    n_kept = len(vertices)
    if n_kept < 2:
        return vertices, nn_dists

    # Compute support atom sets for each kept vertex (k=4 nearest framework atoms)
    k_support = min(_SITE_CLASSIFICATION_NEIGHBOURS, len(extended_positions))
    _, support_indices = framework_tree.query(vertices, k=k_support)
    if support_indices.ndim == 1:
        support_indices = support_indices.reshape(-1, 1)
    support_sets = [set(int(j) for j in row) for row in support_indices]

    # Adaptive target spacing: beta * median nearest-neighbour distance
    median_nn = float(np.median(nn_dists))
    target_spacing = _ENRICHMENT_SPACING_BETA * median_nn

    # Collect admissible edges between kept vertices
    edges: set[tuple[int, int]] = set()
    for ridge in ridge_vertices:
        if len(ridge) != 2:
            continue
        r0, r1 = int(ridge[0]), int(ridge[1])
        if r0 < 0 or r1 < 0:
            continue
        if r0 in raw_to_kept and r1 in raw_to_kept:
            k0, k1 = raw_to_kept[r0], raw_to_kept[r1]
            if k0 != k1:
                edges.add((min(k0, k1), max(k0, k1)))

    if not edges:
        return vertices, nn_dists

    new_verts: list[np.ndarray] = []
    new_dists: list[float] = []

    for k0, k1 in sorted(edges):
        # Same-surface check: support atoms must overlap
        if not support_sets[k0] & support_sets[k1]:
            continue

        v0, v1 = vertices[k0], vertices[k1]
        edge_len = float(np.linalg.norm(v1 - v0))
        if edge_len <= target_spacing:
            continue

        n_subdivisions = min(
            int(edge_len / target_spacing),
            _ENRICHMENT_MAX_SUBDIVISIONS,
        )
        if n_subdivisions < 1:
            continue

        for s in range(1, n_subdivisions + 1):
            t = s / (n_subdivisions + 1)
            candidate = v0 + t * (v1 - v0)
            d_nn = float(
                framework_tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0]
            )
            if probe_radius <= d_nn <= max_distance:
                new_verts.append(candidate)
                new_dists.append(d_nn)

    if not new_verts:
        return vertices, nn_dists

    all_verts = np.vstack([vertices, np.array(new_verts)])
    all_dists = np.concatenate([nn_dists, np.array(new_dists)])

    # Final deduplication of the combined set (preserving original points)
    keep = _deduplicate_points(all_verts, _VORONOI_DEDUP_TOLERANCE)
    return all_verts[keep], all_dists[keep]


# ---------------------------------------------------------------------------
# Site classification
# ---------------------------------------------------------------------------


def _classify_voronoi_site(
    vertex: np.ndarray,
    positions: np.ndarray,
    tree: KDTree | None = None,
    pore_threshold: float = _PORE_THRESHOLD_MIN_ANGSTROM,
    k: int = _SITE_CLASSIFICATION_NEIGHBOURS,
) -> tuple[str, tuple[int, ...]]:
    """Classify vertex as atop/bridge/hollow/pore."""
    k = min(k, len(positions))
    if tree is None:
        tree = KDTree(positions)
    dists, idx = tree.query(vertex.reshape(1, 3), k=k)
    dists = dists.ravel()
    idx = idx.ravel()
    if len(dists) == 0:
        return "atop", ()
    d1 = dists[0]
    if d1 < _DISTANCE_ZERO_EPS:
        return "atop", (int(idx[0]),)
    if d1 > pore_threshold:
        return "pore", tuple(int(i) for i in idx)
    if len(dists) >= 2 and dists[1] / d1 > _ATOP_RATIO:
        return "atop", (int(idx[0]),)
    if len(dists) >= 3 and all(
        abs(dists[i] - d1) / max(d1, _DISTANCE_RATIO_FLOOR_EPS) < _HOLLOW_EQ_TOL
        for i in range(1, 3)
    ):
        return "hollow", tuple(int(i) for i in idx[:3])
    if len(dists) >= 2 and abs(dists[1] - d1) / max(
        d1, _DISTANCE_RATIO_FLOOR_EPS
    ) < _BRIDGE_EQ_TOL:
        far3 = len(dists) < 3 or dists[2] / d1 > _BRIDGE_FAR_RATIO
        if far3:
            return "bridge", tuple(int(i) for i in idx[:2])
    return "hollow", tuple(int(i) for i in idx[:3])


def _delaunay_site_classification(
    vertex: np.ndarray,
    top_positions: np.ndarray,
    top_atom_indices: np.ndarray,
    triangulation: Delaunay,
    positions: np.ndarray,
    *,
    bridge_threshold: float = _DELAUNAY_BRIDGE_THRESHOLD_FRACTION,
) -> tuple[str, tuple[int, ...]]:
    """Classify site using Delaunay triangulation of top-layer atoms.

    Generates canonical atop (vertex), bridge (edge midpoint), and hollow
    (face centroid) reference points from the triangulation, then assigns
    the Voronoi vertex to the nearest reference.

    Parameters
    ----------
    vertex : (3,) array
        Voronoi vertex position.
    top_positions : (M, 2) or (M, 3) array
        Top-layer atom positions used for Delaunay.
    top_atom_indices : (M,) array
        Mapping from top_positions index to global positions index.
    triangulation : Delaunay
        Pre-computed Delaunay triangulation of top_positions[:, :2].
    positions : (N, 3) array
        All slab atom positions (nearest-neighbor checks).
    bridge_threshold : float
        Max fractional distance (relative to mean nearest-neighbour
        distance) for a bridge assignment; beyond this a bridge site is
        reclassified as hollow.

    Returns
    -------
    (site_type, atom_indices) with the same interface as
    :func:`_classify_voronoi_site`.
    """
    xy = vertex[:2]

    # Build reference points: atop, bridge (edge midpoints), hollow (centroids)
    simplices = triangulation.simplices
    top_xy = (
        top_positions[:, :2]
        if top_positions.ndim == 2 and top_positions.shape[1] >= 2
        else top_positions
    )

    best_type = "hollow"
    best_dist = float("inf")
    best_indices: tuple[int, ...] = ()

    # Check atop: distance to each top-layer atom
    for li, gi in enumerate(top_atom_indices):
        d = float(np.linalg.norm(xy - top_xy[li]))
        if d < best_dist:
            best_dist = d
            best_type = "atop"
            best_indices = (int(gi),)

    # Characteristic length for thresholding: mean nn distance in top layer
    if len(top_xy) >= 2:
        _top_tree = KDTree(top_xy)
        _nn_d, _ = _top_tree.query(top_xy, k=2)
        char_len = float(np.mean(_nn_d[:, 1]))
    else:
        char_len = (
            best_dist
            if best_dist < float("inf")
            else _DELAUNAY_CHAR_LENGTH_FALLBACK_ANGSTROM
        )

    # Check bridge: edge midpoints
    seen_edges: set[tuple[int, int]] = set()
    for simplex in simplices:
        for e0, e1 in ((0, 1), (1, 2), (0, 2)):
            li0, li1 = int(simplex[e0]), int(simplex[e1])
            edge_key = (min(li0, li1), max(li0, li1))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            mid = (top_xy[li0] + top_xy[li1]) / 2.0
            d = float(np.linalg.norm(xy - mid))
            if d < best_dist:
                best_dist = d
                best_type = "bridge"
                best_indices = (int(top_atom_indices[li0]), int(top_atom_indices[li1]))

    # Check hollow: face centroids
    for simplex in simplices:
        li0, li1, li2 = int(simplex[0]), int(simplex[1]), int(simplex[2])
        centroid = (top_xy[li0] + top_xy[li1] + top_xy[li2]) / 3.0
        d = float(np.linalg.norm(xy - centroid))
        if d < best_dist:
            best_dist = d
            best_type = "hollow"
            best_indices = (
                int(top_atom_indices[li0]),
                int(top_atom_indices[li1]),
                int(top_atom_indices[li2]),
            )

    # Bridge threshold: when close to an edge midpoint but still too far,
    # fall back to hollow using the three nearest framework atoms.  An
    # atop threshold used to live here too but was a no-op ("pass" block)
    # because the main loop already picks the globally nearest reference.
    if best_type == "bridge" and best_dist > bridge_threshold * char_len:
        _tree = KDTree(positions)
        _, idx = _tree.query(vertex.reshape(1, 3), k=min(3, len(positions)))
        idx = idx.ravel()
        best_type = "hollow"
        best_indices = tuple(int(i) for i in idx[:3])

    return best_type, best_indices


# ---------------------------------------------------------------------------
# Local surface normal
# ---------------------------------------------------------------------------


def _compute_local_normal(
    vertex: np.ndarray,
    positions: np.ndarray,
    tree: KDTree | None = None,
    k: int = _NORMAL_K_NEIGHBOURS,
) -> np.ndarray:
    """Outward unit normal at *vertex* from centroid of k nearest atoms."""
    k = min(k, len(positions))
    if tree is None:
        tree = KDTree(positions)
    _, idx = tree.query(vertex.reshape(1, 3), k=k)
    centroid = np.mean(positions[idx.ravel()], axis=0)
    vec = vertex - centroid
    norm = float(np.linalg.norm(vec))
    if norm < _SURFACE_NORMAL_FALLBACK_NORM_EPS:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def _build_site_records(
    vertices: np.ndarray,
    nn_dists: np.ndarray,
    positions: np.ndarray,
    symbols: list[str],
    local_tree: KDTree,
    material_type: str,
    pore_threshold: float,
    *,
    use_delaunay: bool,
    delaunay_tri: Delaunay | None,
    top_positions: np.ndarray | None,
    top_atom_indices: np.ndarray | None,
) -> list[dict[str, object]]:
    sites: list[dict[str, object]] = []
    for i, vertex in enumerate(vertices):
        if (
            use_delaunay
            and delaunay_tri is not None
            and top_positions is not None
            and top_atom_indices is not None
        ):
            site_type, nearest_idx = _delaunay_site_classification(
                vertex, top_positions, top_atom_indices, delaunay_tri, positions
            )
        else:
            site_type, nearest_idx = _classify_voronoi_site(
                vertex, positions, tree=local_tree, pore_threshold=pore_threshold
            )
        env_fingerprint = (
            tuple(sorted(symbols[j] for j in nearest_idx if j < len(symbols))),
            site_type,
        )
        sites.append(
            {
                "xy": vertex[:2].copy(),
                "z": float(vertex[2]),
                "xyz": vertex.copy(),
                "site_type": site_type,
                "slab_indices": nearest_idx,
                "normal": _compute_local_normal(vertex, positions, tree=local_tree),
                "nn_distance": float(nn_dists[i]) if i < len(nn_dists) else None,
                "site_source": "voronoi",
                "material_type": material_type,
                "env_fingerprint": env_fingerprint,
            }
        )
    return sites


# ---------------------------------------------------------------------------
# Unified site dict builder
# ---------------------------------------------------------------------------


def get_unified_sites(
    atoms: Atoms,
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    top_layer_tolerance: float | None = None,
    material_type: str | None = None,
    pore_threshold: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "distance_ratio",
) -> list[dict[str, object]]:
    """Return adsorption/placement site dicts for *atoms*.

    Works for slabs (planar or non-planar), nanoparticles, and porous
    materials (zeolites, MOFs).

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object.
    probe_radius : float | None, optional
        Minimum distance from Voronoi vertex to any atom. When None, derived
        from top-surface mean covalent radius.
    max_site_distance : float | None, optional
        Maximum distance from Voronoi vertex to nearest atom. When None,
        derived from top-surface mean covalent radius.
    top_layer_tolerance : float | None, optional
        For slabs, filter z below top surface by this amount. When None,
        derived from top-surface mean covalent radius.
    material_type : str | None, optional
        One of "slab", "nanoparticle", "porous". If None, auto-detect from structure.
        Config layer requires explicit selection; None here is for internal flexibility.
    pore_threshold : float | None, optional
        Distance threshold for pore site classification. When None, derived
        from mean top-layer covalent radius.
    enrich : bool, optional
        When True (default), subdivide long admissible Voronoi edges to
        improve site coverage on rugged surfaces and porous materials.
    site_classification_method : str, optional
        ``"distance_ratio"`` (default) uses nearest-neighbor distance ratios;
        ``"delaunay"`` uses Delaunay triangulation of top-layer atoms (slabs
        only; falls back to distance_ratio for non-slab materials).

    Returns
    -------
    list[dict[str, object]]
        Site dictionaries with keys:
         ``"xy"`` (x,y) tuple, ``"z"`` float, ``"xyz"`` (3,) array,
         ``"site_type"`` str, ``"slab_indices"`` tuple, ``"normal"`` (3,) array,
         ``"site_source"`` str, ``"material_type"`` str.
    """
    # Auto-detect material type if not specified.
    # Config layer requires explicit selection; internal functions can auto-detect.
    if material_type is None:
        material_type = detect_material_type(atoms)

    if material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
        )

    positions = atoms.get_positions()
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)

    pbc_for_voronoi = pbc.copy()
    if material_type == "nanoparticle":
        pbc_for_voronoi[:] = False

    symbols = atoms.get_chemical_symbols()
    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(positions, symbols)
    if pore_threshold is None:
        pore_threshold = _derive_pore_threshold(symbols)

    if np.linalg.det(cell) <= 0:
        cell = _bounding_box_cell(positions)

    # Single deterministic Voronoi site generation with optional enrichment.
    vertices, nn_dists = _voronoi_sites(
        positions,
        cell,
        pbc_for_voronoi,
        probe_radius=probe_radius,
        max_distance=max_site_distance,
        enrich=enrich,
        symbols=symbols,
    )

    if len(vertices) == 0:
        logger.warning(
            "No accessible Voronoi sites for %d-atom structure "
            "(probe_radius=%.2f, max_distance=%.2f, material_type=%r)",
            len(atoms),
            probe_radius,
            max_site_distance,
            material_type,
        )
        return []

    # Slab: discard vertices below the top surface layer. Voronoi vertices sit
    # between atoms, so keep sites down to z_surface minus median nn_distance
    # among vertices (at least top_layer_tolerance).
    if material_type == "slab":
        z_surface = float(np.max(positions[:, 2]))
        if len(nn_dists) > 0:
            nn_margin = float(np.median(nn_dists))
        else:
            nn_margin = top_layer_tolerance
        z_min = z_surface - max(top_layer_tolerance, nn_margin)
        keep_mask = vertices[:, 2] >= z_min
        vertices = vertices[keep_mask]
        nn_dists = nn_dists[keep_mask]

    local_tree = KDTree(positions)

    # Nanoparticle: keep only exterior sites (normal points away from COM)
    if material_type == "nanoparticle" and len(vertices) > 0:
        com = np.mean(positions, axis=0)
        normals = np.array(
            [_compute_local_normal(v, positions, tree=local_tree) for v in vertices]
        )
        outward = np.array(
            [
                float(np.dot(normals[i], vertices[i] - com)) > 0
                for i in range(len(vertices))
            ],
            dtype=bool,
        )
        vertices = vertices[outward]
        nn_dists = nn_dists[outward]

    # --- Atop site injection ---
    # Voronoi vertices sit between atoms (bridge/hollow) by construction.
    # Inject explicit atop candidates above top-layer / surface atoms so
    # that atop binding (preferred for CO, H₂O, NH₃, etc.) is represented.
    if material_type in ("slab", "nanoparticle") and len(vertices) > 0:
        if probe_radius is None or max_site_distance is None:
            derived_probe, derived_max = _derive_voronoi_distance_window(
                positions, symbols, pbc_for_voronoi
            )
            probe_radius = derived_probe if probe_radius is None else probe_radius
            max_site_distance = (
                derived_max if max_site_distance is None else max_site_distance
            )
        median_nn = (
            float(np.median(nn_dists))
            if len(nn_dists) > 0
            else _VORONOI_MAX_DISTANCE_COVALENT_SCALE * _VORONOI_RADIUS_FALLBACK_ANGSTROM
        )
        atop_height = _ATOP_INJECTION_HEIGHT_FACTOR * median_nn

        if material_type == "slab":
            z_surface = float(np.max(positions[:, 2]))
            top_mask = positions[:, 2] >= z_surface - top_layer_tolerance
            top_atom_indices = np.nonzero(top_mask)[0]
        else:
            # Nanoparticle: use atoms whose outward normal component is positive
            com = np.mean(positions, axis=0)
            top_atom_indices = np.array(
                [
                    i
                    for i in range(len(positions))
                    if float(
                        np.dot(
                            _compute_local_normal(
                                positions[i], positions, tree=local_tree
                            ),
                            positions[i] - com,
                        )
                    )
                    > 0
                ],
                dtype=int,
            )

        # Build KDTree on existing vertices for fast dedup checks against
        # the Voronoi set; duplicates *between* injected atop candidates are
        # handled by one final ``_deduplicate_points`` pass instead of a
        # per-iteration KDTree rebuild (old O(n²) behaviour).
        existing_tree = KDTree(vertices) if len(vertices) > 0 else None

        candidate_verts: list[np.ndarray] = []
        candidate_dists: list[float] = []
        for ai in top_atom_indices:
            atom_pos = positions[ai]
            if material_type == "slab":
                candidate = atom_pos.copy()
                candidate[2] += atop_height
            else:
                normal = _compute_local_normal(atom_pos, positions, tree=local_tree)
                candidate = atom_pos + atop_height * normal

            d_nn = float(local_tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0])
            if d_nn < probe_radius or d_nn > max_site_distance:
                continue

            if _is_duplicate_of(candidate, existing_tree, _VORONOI_DEDUP_TOLERANCE):
                continue

            candidate_verts.append(candidate)
            candidate_dists.append(d_nn)

        if candidate_verts:
            candidate_arr = np.array(candidate_verts)
            candidate_dist_arr = np.array(candidate_dists)
            # Final dedup pass over (existing ∪ candidate) to collapse any
            # injected atop sites that ended up within tolerance of each
            # other.  Existing Voronoi vertices are already pairwise
            # deduplicated, so only inter-candidate merges can trigger here.
            combined = np.vstack([vertices, candidate_arr])
            combined_dists = np.concatenate([nn_dists, candidate_dist_arr])
            keep = _deduplicate_points(combined, _VORONOI_DEDUP_TOLERANCE)
            n_existing = len(vertices)
            n_injected = int(np.count_nonzero(keep[n_existing:]))
            vertices = combined[keep]
            nn_dists = combined_dists[keep]
            logger.debug(
                "Injected %d atop candidate sites (%d total sites)",
                n_injected,
                len(vertices),
            )

    if len(vertices) == 0:
        return []

    # Pre-compute Delaunay triangulation for slab classification if requested.
    _use_delaunay = site_classification_method == "delaunay" and material_type == "slab"
    _delaunay_tri = None
    _top_positions: np.ndarray | None = None
    _top_atom_indices: np.ndarray | None = None
    if _use_delaunay:
        z_surface = float(np.max(positions[:, 2]))
        top_mask = positions[:, 2] >= z_surface - top_layer_tolerance
        _top_atom_indices = np.nonzero(top_mask)[0]
        _top_positions = positions[_top_atom_indices]
        if len(_top_atom_indices) >= 3:
            _delaunay_tri = Delaunay(_top_positions[:, :2])
        else:
            _use_delaunay = False

    sites = _build_site_records(
        vertices,
        nn_dists,
        positions,
        symbols,
        local_tree,
        material_type,
        pore_threshold,
        use_delaunay=_use_delaunay,
        delaunay_tri=_delaunay_tri,
        top_positions=_top_positions,
        top_atom_indices=_top_atom_indices,
    )

    # Deterministic ordering: fractional coordinates then site_type
    if np.linalg.det(cell) > 0:
        inv_cell = np.linalg.inv(cell)
        sites.sort(
            key=lambda s: (
                *((inv_cell @ np.asarray(s["xyz"])) % 1.0).tolist(),
                str(s["site_type"]),
            )
        )

    return sites


# ---------------------------------------------------------------------------
# Hollow sites for adatom / dissociative placement
# ---------------------------------------------------------------------------


def get_hollow_sites_for_adatoms(
    slab: Atoms,
    top_layer_tolerance: float = _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    dedup_tolerance: float = DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
) -> list[np.ndarray]:
    """Hollow/pore site xy positions for adatom placement, deduplicated."""
    raw = get_unified_sites(slab, top_layer_tolerance=top_layer_tolerance)
    hollow_xy = np.array(
        [np.asarray(s["xy"]) for s in raw if s.get("site_type") in ("hollow", "pore")]
    )
    if len(hollow_xy) == 0:
        return []
    keep = _deduplicate_points(hollow_xy, dedup_tolerance)
    return [np.asarray(xy) for xy in hollow_xy[keep]]


# ---------------------------------------------------------------------------
# Union-find for environment-aware site clustering
# ---------------------------------------------------------------------------


def _union_find_cluster(
    n: int,
    merge_pairs: list[tuple[int, int]],
) -> list[list[int]]:
    """Union-find with path compression and union-by-rank.

    Returns a list of connected components (lists of original indices).
    """
    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1

    for a, b in merge_pairs:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _env_fingerprint(site: dict[str, object]) -> tuple:
    """Return the local-environment fingerprint of *site*.

    Falls back to ``(site_type,)`` when no fingerprint was computed.
    """
    fp = site.get("env_fingerprint")
    if fp is not None:
        return fp
    return (str(site.get("site_type", "")),)


def _cluster_with_metric(
    n: int,
    coords: np.ndarray,
    fps: list[tuple],
    *,
    image_offsets: list[np.ndarray] | None,
    kdtree_radius: float,
    pair_filter: Callable[[int, int, np.ndarray], bool] | None = None,
) -> list[int]:
    """KDTree ``query_pairs`` + union-find; one representative index per cluster.

    *coords* may be tiled by *image_offsets* (expanded index mod *n* maps back).
    *pair_filter* runs after the fingerprint match (e.g. slab xy/z checks).
    """
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
    sites: list[dict[str, object]],
    cell: np.ndarray,
    tolerance: float = DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
    z_abs_tolerance: float | None = None,
) -> list[dict[str, object]]:
    """Group equivalent sites; return unique representatives.

    Uses KDTree ``query_pairs`` + union-find instead of greedy O(n·k)
    scanning, with an environment-fingerprint guard so geometrically close
    sites with different chemical neighborhoods are never merged.

    - nanoparticle: Cartesian 3D distance.
    - porous: 3D fractional with periodic wrapping.
    - slab: 2D fractional (a,b) + absolute z (separate tolerance for z).

    Parameters
    ----------
    tolerance : float
        Fractional-coordinate tolerance for xy (slab), full 3D (porous),
        or Cartesian distance (nanoparticle).
    z_abs_tolerance : float | None
        For slabs, absolute z tolerance in Å.  Defaults to 0.5 Å when
        ``None``, avoiding the old bug where the dimensionless *tolerance*
        was reused for absolute z comparisons.
    """
    if not sites:
        return []

    n = len(sites)
    mat_type = material_type_for_placement(sites[0], when_no_site="slab")

    def _get_xyz(s: dict[str, object]) -> np.ndarray:
        if "xyz" in s:
            return np.asarray(s["xyz"], dtype=float)
        return np.array([*np.asarray(s["xy"], dtype=float), float(s.get("z", 0.0))])

    # Stable sort key for deterministic representative selection
    def _sort_key(s: dict[str, object]) -> tuple:
        xyz = _get_xyz(s)
        return (
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            str(s.get("site_type", "")),
        )

    # Sort sites and remember original order for representative selection
    order = sorted(range(n), key=lambda i: _sort_key(sites[i]))
    sorted_sites = [sites[i] for i in order]
    fps = [_env_fingerprint(s) for s in sorted_sites]

    # ------------------------------------------------------------------
    # Nanoparticle: Cartesian 3D distance, no periodic images
    # ------------------------------------------------------------------
    if mat_type == "nanoparticle" or np.linalg.det(cell) <= 0:
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

    inv_cell = np.linalg.inv(cell)

    # ------------------------------------------------------------------
    # Porous: 3D fractional with 3×3×3 periodic tiling
    # ------------------------------------------------------------------
    if mat_type == "porous":
        coords = np.array([(inv_cell @ _get_xyz(s)) % 1.0 for s in sorted_sites])
        image_offsets = [
            np.array([ix, iy, iz], dtype=float)
            for ix in (-1, 0, 1)
            for iy in (-1, 0, 1)
            for iz in (-1, 0, 1)
        ]
        reps = _cluster_with_metric(
            n,
            coords,
            fps,
            image_offsets=image_offsets,
            kdtree_radius=tolerance,
        )
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    # ------------------------------------------------------------------
    # Slab: 2D fractional (a,b) + absolute z with 3×3 periodic tiling in xy
    # ------------------------------------------------------------------
    z_tol = (
        z_abs_tolerance
        if z_abs_tolerance is not None
        else _SLAB_Z_ABS_TOLERANCE_DEFAULT_ANGSTROM
    )
    inv_2d = np.linalg.inv(cell[:2, :2])

    def _slab_coord(s: dict[str, object]) -> np.ndarray:
        xy = np.asarray(s["xy"], dtype=float)
        frac = (inv_2d @ xy) % 1.0
        z = float(cast(float, s.get("z", 0.0)))
        return np.array([frac[0], frac[1], z])

    slab_coords = np.array([_slab_coord(s) for s in sorted_sites])
    image_offsets_xy = [
        np.array([ix, iy, 0.0], dtype=float) for ix in (-1, 0, 1) for iy in (-1, 0, 1)
    ]

    def _slab_pair_filter(a: int, b: int, coords_arr: np.ndarray) -> bool:
        ca = coords_arr[a]
        cb = coords_arr[b]
        diff_xy = np.minimum(np.abs(ca[:2] - cb[:2]), 1.0 - np.abs(ca[:2] - cb[:2]))
        return bool(np.all(diff_xy < tolerance) and abs(ca[2] - cb[2]) < z_tol)

    # Slight padding on the KDTree radius so it catches all candidates under
    # either the fractional-xy or absolute-z tolerance; the pair_filter then
    # applies the per-dimension thresholds strictly.
    r_search = max(tolerance, z_tol) * _KD_RADIUS_SEARCH_PADDING
    reps = _cluster_with_metric(
        n,
        slab_coords,
        fps,
        image_offsets=image_offsets_xy,
        kdtree_radius=r_search,
        pair_filter=_slab_pair_filter,
    )
    result = [sorted_sites[i] for i in reps]

    def _slab_key(s: dict[str, object]) -> tuple:
        c = _slab_coord(s)
        return (float(c[0]), float(c[1]), float(c[2]))

    return sorted(result, key=_slab_key)


# ---------------------------------------------------------------------------
# Symmetry-aware site reduction
# ---------------------------------------------------------------------------


def get_symmetry_aware_sites(
    slab: Atoms,
    top_layer_tolerance: float | None = None,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    material_type: str = "slab",
    probe_radius: float | None = None,
    max_site_distance: float | None = None,
    enrich: bool = True,
    site_classification_method: str = "distance_ratio",
    raw_sites: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Symmetry-reduced adsorption sites using spglib.

    Returns ``[]`` if there are no input sites. Otherwise runs
    :class:`~metalsurfer.symmetry.SymmetryAnalyzer` and may raise
    :class:`~metalsurfer.symmetry.SymmetryAnalysisError` (callers that need
    Voronoi-only sampling should catch it, e.g. workflow site resolution).

    When *raw_sites* is set, it must be the unclustered output of
    :func:`get_unified_sites` for this slab and the same parameters as used
    to build that list; Voronoi is not run again.
    """
    if material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
        )

    if top_layer_tolerance is None:
        top_layer_tolerance = _derive_top_layer_tolerance(
            slab.get_positions(),
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


def _is_top_layer_planar(
    slab: Atoms,
    top_layer_tolerance: float = _TOP_LAYER_DEPTH_MIN_ANGSTROM,
    z_variance_threshold: float = _DEFAULT_PLANAR_Z_VARIANCE_THRESHOLD,
) -> bool:
    """True if the topmost atomic layer is approximately flat."""
    positions = slab.get_positions()
    z_max = float(np.max(positions[:, 2]))
    top_mask = positions[:, 2] >= (z_max - top_layer_tolerance)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return False
    top_pos = positions[top_indices]
    x, y, z = top_pos[:, 0], top_pos[:, 1], top_pos[:, 2]
    A = np.column_stack([x, y, np.ones(len(x))])
    coeffs, residuals, rank, _ = np.linalg.lstsq(A, z, rcond=None)
    if rank < 3 or len(residuals) == 0:
        return False
    z_pred = A @ coeffs
    return float(np.var(z - z_pred)) < z_variance_threshold


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
    site: dict[str, object] | None = None,
) -> float | None:
    """Mean covalent radius of framework atoms nearest to the placement site."""
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z_max = float(np.max(positions[:, 2]))

    if site is not None and "slab_indices" in site:
        indices = site["slab_indices"]
        if not indices:
            indices = None
    else:
        indices = None

    if indices is None:
        top_depth = _derive_top_layer_tolerance(positions, symbols)
        top_mask = positions[:, 2] >= (z_max - top_depth)
        indices = tuple(int(i) for i in np.nonzero(top_mask)[0])

    radii = [_get_covalent_radius(symbols[i]) for i in indices]
    radii = [r for r in radii if r is not None]
    if not radii:
        return None
    return float(np.mean(radii))


def _compute_site_z_base(
    config,
    slab: Atoms,
    site: dict[str, object] | None,
    mol_symbols: list[str],
) -> tuple[float, float]:
    """Compute z-offset range for placement above *site*.

    For non-slab materials (nanoparticles, porous) the surface reference is
    the Voronoi vertex itself, so we centre the z-range around the
    ``nn_distance`` when available.  For slabs the reference is already the
    top surface z, so the config defaults (typically 2–3 Å) directly
    control the adsorbate-surface distance and no nn_distance override is
    needed.
    """
    z_lo, z_hi = config.placement_z_range

    mat_type = material_type_for_placement(site, when_no_site=config.material_type)

    # For non-slab materials whose surface_ref is the Voronoi vertex z,
    # tighten the range around the nn_distance to keep the adsorbate in
    # the accessible zone.
    if (
        mat_type != "slab"
        and site is not None
        and site.get("nn_distance") is not None
        and str(site.get("site_type", "")) != "pore"
    ):
        nn = float(site["nn_distance"])
        nn_lo = nn * _NON_SLAB_Z_LO_FROM_NN_SCALE
        nn_hi = nn * _NON_SLAB_Z_HI_FROM_NN_SCALE
        if nn_hi - nn_lo < z_hi - z_lo:
            z_lo, z_hi = nn_lo, nn_hi

    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi
    if site is not None and str(site.get("site_type", "")) == "pore":
        return z_lo, z_hi

    r_surface = _get_site_surface_radii(slab, site)
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else _MOL_COVALENT_RADIUS_FALLBACK

    if r_surface is None:
        return z_lo, z_hi

    delta = _SITE_Z_RADIUS_SHIFT_SCALE * (
        r_mol + r_surface - _SITE_Z_RADIUS_REFERENCE_ANGSTROM
    )
    return z_lo + delta, z_hi + delta


# ---------------------------------------------------------------------------
# Public API (for external imports)
# ---------------------------------------------------------------------------


def get_symmetry_info(
    slab: Atoms,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
) -> dict[str, object]:
    """Symmetry metadata including spglib space group."""
    symmetry_analyzer = SymmetryAnalyzer(slab, symmetry_tolerance=symmetry_tolerance)
    return symmetry_analyzer.get_symmetry_info()

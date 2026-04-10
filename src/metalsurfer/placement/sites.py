"""Site detection and clustering for adsorbate placement.

Unified Voronoi-based sites for slabs, nanoparticles, and 3D-periodic porous
frameworks. Optional Delaunay-based classification on slab top layers when
``site_classification_method == 'delaunay'``. Symmetry reduction via
``get_symmetry_aware_sites()`` builds on ``get_unified_sites()``.
"""

from __future__ import annotations

import logging
from typing import cast

import numpy as np
from ase import Atoms
from scipy.spatial import Delaunay, KDTree, QhullError, Voronoi

from ..symmetry import SymmetryAnalysisError, SymmetryAnalyzer
from ._constants import (
    _ATOP_INJECTION_HEIGHT_FACTOR,
    _ATOP_RATIO,
    _BRIDGE_EQ_TOL,
    _BRIDGE_FAR_RATIO,
    _ENRICHMENT_MAX_SUBDIVISIONS,
    _ENRICHMENT_SPACING_BETA,
    _HOLLOW_EQ_TOL,
    _NORMAL_K_NEIGHBOURS,
    _PORE_THRESHOLD_ANGSTROM,
    _VORONOI_DEDUP_TOLERANCE,
)
from ._material import (  # noqa: F401
    _SLAB_VACUUM_FRACTION,
    _resolve_material_type,
    detect_material_type,
)
from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)

DEFAULT_SYMMETRY_TOLERANCE = 0.1
DEFAULT_SITE_EQUIVALENCE_TOLERANCE = 0.05
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE = 0.1


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


def _voronoi_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    probe_radius: float = 1.2,
    max_distance: float = 4.0,
    enrich: bool = True,
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

    extended = _build_periodic_images(positions, cell, pbc)
    try:
        vor = Voronoi(extended)
    except (QhullError, ValueError, RuntimeError) as exc:
        logger.warning("Voronoi computation failed (%s); returning no sites", exc)
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
                inside &= (frac[:, dim] >= -0.01) & (frac[:, dim] < 1.01)
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
    k_support = min(4, len(extended_positions))
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
    pore_threshold: float = _PORE_THRESHOLD_ANGSTROM,
    k: int = 4,
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
    if d1 < 1e-12:
        return "atop", (int(idx[0]),)
    if d1 > pore_threshold:
        return "pore", tuple(int(i) for i in idx)
    if len(dists) >= 2 and dists[1] / d1 > _ATOP_RATIO:
        return "atop", (int(idx[0]),)
    if len(dists) >= 3 and all(
        abs(dists[i] - d1) / max(d1, 1e-8) < _HOLLOW_EQ_TOL for i in range(1, 3)
    ):
        return "hollow", tuple(int(i) for i in idx[:3])
    if len(dists) >= 2 and abs(dists[1] - d1) / max(d1, 1e-8) < _BRIDGE_EQ_TOL:
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
    atop_threshold: float = 0.3,
    bridge_threshold: float = 0.3,
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
        All slab atom positions (for fallback nearest-neighbor).
    atop_threshold : float
        Max fractional distance (relative to nearest-neighbor distance)
        to classify as atop.
    bridge_threshold : float
        Max fractional distance to classify as bridge.

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
        char_len = best_dist if best_dist < float("inf") else 1.0

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

    # Apply thresholds: only keep atop/bridge if close enough relative to
    # the characteristic nearest-neighbor distance, otherwise fall back to hollow.
    if best_type == "atop" and best_dist > atop_threshold * char_len:
        # Recheck: is there a better bridge or hollow?
        pass  # keep the assignment — the loop already picked the globally nearest
    if best_type == "bridge" and best_dist > bridge_threshold * char_len:
        # Close to an edge midpoint but not enough; fallback
        # Use full positions k=3 nearest for hollow
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
    if norm < 1e-8:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


# ---------------------------------------------------------------------------
# Unified site dict builder
# ---------------------------------------------------------------------------


def get_unified_sites(
    atoms: Atoms,
    probe_radius: float = 1.2,
    max_site_distance: float = 4.0,
    top_layer_tolerance: float = 0.5,
    material_type: str | None = None,
    pore_threshold: float = _PORE_THRESHOLD_ANGSTROM,
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
    probe_radius : float, optional
        Minimum distance from Voronoi vertex to any atom (default 1.2 Å).
    max_site_distance : float, optional
        Maximum distance from Voronoi vertex to nearest atom (default 4.0 Å).
    top_layer_tolerance : float, optional
        For slabs, filter z below top surface by this amount (default 0.5 Å).
    material_type : str | None, optional
        One of "slab", "nanoparticle", "porous". If None, auto-detect from structure.
        Config layer requires explicit selection; None here is for internal flexibility.
    pore_threshold : float, optional
        Distance threshold for pore site classification (default ~3.0 Å).
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
    )

    if len(vertices) == 0:
        logger.warning(
            "No Voronoi sites found for %d atoms (probe_radius=%.2f, max_distance=%.2f)",
            len(atoms),
            probe_radius,
            max_site_distance,
        )
        return []

    # Slab: discard vertices below the top surface layer.
    # Voronoi vertices sit *between* atoms, so legitimate top-layer sites
    # can be up to ~1 nearest-neighbour distance below the topmost atom z.
    # Use the median nn_distance of all current vertices as a robust lower
    # bound, floored by top_layer_tolerance.
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
        median_nn = float(np.median(nn_dists)) if len(nn_dists) > 0 else 1.5
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

        # Build KDTree on existing vertices for fast dedup checks
        existing_tree = KDTree(vertices) if len(vertices) > 0 else None

        new_verts: list[np.ndarray] = []
        new_dists: list[float] = []
        for ai in top_atom_indices:
            atom_pos = positions[ai]
            if material_type == "slab":
                candidate = atom_pos.copy()
                candidate[2] += atop_height
            else:
                normal = _compute_local_normal(atom_pos, positions, tree=local_tree)
                candidate = atom_pos + atop_height * normal

            # Accessibility check
            d_nn = float(local_tree.query(candidate.reshape(1, 3), k=1)[0].ravel()[0])
            if d_nn < probe_radius or d_nn > max_site_distance:
                continue

            # Dedup: skip if too close to an existing Voronoi vertex
            if _is_duplicate_of(candidate, existing_tree, _VORONOI_DEDUP_TOLERANCE):
                continue

            # Dedup: skip if too close to a previously injected atop site
            if new_verts:
                _injected_tree = KDTree(np.array(new_verts))
                if _is_duplicate_of(
                    candidate,
                    _injected_tree,
                    _VORONOI_DEDUP_TOLERANCE,
                ):
                    continue

            new_verts.append(candidate)
            new_dists.append(d_nn)

        if new_verts:
            vertices = np.vstack([vertices, np.array(new_verts)])
            nn_dists = np.concatenate([nn_dists, np.array(new_dists)])
            logger.debug(
                "Injected %d atop candidate sites (%d total sites)",
                len(new_verts),
                len(vertices),
            )

    if len(vertices) == 0:
        return []

    symbols = atoms.get_chemical_symbols()

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

    sites: list[dict[str, object]] = []
    for i, vertex in enumerate(vertices):
        if (
            _use_delaunay
            and _delaunay_tri is not None
            and _top_positions is not None
            and _top_atom_indices is not None
        ):
            site_type, nearest_idx = _delaunay_site_classification(
                vertex,
                _top_positions,
                _top_atom_indices,
                _delaunay_tri,
                positions,
            )
        else:
            site_type, nearest_idx = _classify_voronoi_site(
                vertex,
                positions,
                tree=local_tree,
                pore_threshold=pore_threshold,
            )
        normal = _compute_local_normal(vertex, positions, tree=local_tree)
        # Local environment fingerprint: sorted element symbols of nearest
        # framework atoms + site type.  Used by _cluster_equivalent_sites to
        # prevent merging geometrically close sites with different chemical
        # environments (e.g. atop-Ni vs atop-Pt on an alloy surface).
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
                "normal": normal,
                "nn_distance": float(nn_dists[i]) if i < len(nn_dists) else None,
                "site_source": "voronoi",
                "material_type": material_type,
                "env_fingerprint": env_fingerprint,
            }
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
    top_layer_tolerance: float = 0.5,
    dedup_tolerance: float = DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE,
) -> list[np.ndarray]:
    """Hollow/pore site xy positions for adatom placement, deduplicated."""
    raw = get_unified_sites(slab, top_layer_tolerance=top_layer_tolerance)
    hollow_xy = [
        np.asarray(s["xy"]) for s in raw if s.get("site_type") in ("hollow", "pore")
    ]
    unique: list[np.ndarray] = []
    for c in hollow_xy:
        if not any(float(np.linalg.norm(c - u)) < dedup_tolerance for u in unique):
            unique.append(c)
    return unique


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
    mat_type = _resolve_material_type(sites[0], fallback="slab")

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
    # Nanoparticle: Cartesian 3D distance
    # ------------------------------------------------------------------
    if mat_type == "nanoparticle" or np.linalg.det(cell) <= 0:
        coords = np.array([_get_xyz(s) for s in sorted_sites])
        tree = KDTree(coords)
        raw_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")
        merge = [(int(a), int(b)) for a, b in raw_pairs if fps[a] == fps[b]]
        components = _union_find_cluster(n, merge)
        reps = sorted([min(comp) for comp in components])
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    inv_cell = np.linalg.inv(cell)

    # ------------------------------------------------------------------
    # Porous: 3D fractional with periodic wrapping
    # ------------------------------------------------------------------
    if mat_type == "porous":
        frac_coords = np.array([(inv_cell @ _get_xyz(s)) % 1.0 for s in sorted_sites])
        # For periodic fractional coords, KDTree can miss pairs across
        # the 0/1 boundary.  Expand into 3^3 = 27 images, query pairs,
        # then map back.
        images = []
        image_map = []  # maps expanded index -> original index
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                for iz in (-1, 0, 1):
                    shifted = frac_coords + np.array([ix, iy, iz])
                    images.append(shifted)
                    image_map.extend(range(n))
        all_coords = np.vstack(images)
        tree = KDTree(all_coords)
        raw_pairs = tree.query_pairs(r=tolerance, output_type="ndarray")
        merge: list[tuple[int, int]] = []
        for a_exp, b_exp in raw_pairs:
            a_orig = image_map[int(a_exp)]
            b_orig = image_map[int(b_exp)]
            if a_orig != b_orig and fps[a_orig] == fps[b_orig]:
                merge.append((min(a_orig, b_orig), max(a_orig, b_orig)))
        # Deduplicate merge pairs
        merge = list(set(merge))
        components = _union_find_cluster(n, merge)
        reps = sorted([min(comp) for comp in components])
        result = [sorted_sites[i] for i in reps]
        return sorted(result, key=_sort_key)

    # ------------------------------------------------------------------
    # Slab: 2D fractional + absolute z (separate tolerance)
    # ------------------------------------------------------------------
    z_tol = z_abs_tolerance if z_abs_tolerance is not None else 0.5
    inv_2d = np.linalg.inv(cell[:2, :2])

    def _slab_coord(s: dict[str, object]) -> np.ndarray:
        xy = np.asarray(s["xy"], dtype=float)
        frac = (inv_2d @ xy) % 1.0
        z = float(cast(float, s.get("z", 0.0)))
        return np.array([frac[0], frac[1], z])

    slab_coords = np.array([_slab_coord(s) for s in sorted_sites])

    # For 2D periodic fractional coords, expand into 3×3 images (z unchanged)
    images_2d = []
    image_map_2d: list[int] = []
    for ix in (-1, 0, 1):
        for iy in (-1, 0, 1):
            shifted = slab_coords.copy()
            shifted[:, 0] += ix
            shifted[:, 1] += iy
            images_2d.append(shifted)
            image_map_2d.extend(range(n))
    all_slab = np.vstack(images_2d)

    # Use max(tolerance, z_tol) as KDTree radius to catch all candidates,
    # then filter more strictly per dimension.
    r_search = max(tolerance, z_tol) * 1.5  # slight padding for safety
    tree = KDTree(all_slab)
    raw_pairs = tree.query_pairs(r=r_search, output_type="ndarray")
    merge = []
    for a_exp, b_exp in raw_pairs:
        a_orig = image_map_2d[int(a_exp)]
        b_orig = image_map_2d[int(b_exp)]
        if a_orig == b_orig:
            continue
        if fps[a_orig] != fps[b_orig]:
            continue
        ca = slab_coords[a_orig]
        cb = slab_coords[b_orig]
        diff_xy = np.minimum(np.abs(ca[:2] - cb[:2]), 1.0 - np.abs(ca[:2] - cb[:2]))
        if np.all(diff_xy < tolerance) and abs(ca[2] - cb[2]) < z_tol:
            merge.append((min(a_orig, b_orig), max(a_orig, b_orig)))
    merge = list(set(merge))
    components = _union_find_cluster(n, merge)
    reps = sorted([min(comp) for comp in components])
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
    top_layer_tolerance: float = 0.5,
    symmetry_tolerance: float = DEFAULT_SYMMETRY_TOLERANCE,
    material_type: str = "slab",
    probe_radius: float = 1.2,
    max_site_distance: float = 4.0,
    enrich: bool = True,
    site_classification_method: str = "distance_ratio",
) -> list[dict[str, object]] | None:
    """Symmetry-reduced adsorption sites using spglib.

    Returns None on failure. Nanoparticles use cluster-in-a-box symmetry.
    Material type must be explicitly specified.
    """
    if material_type not in ("slab", "nanoparticle", "porous"):
        raise ValueError(
            f"material_type must be 'slab', 'nanoparticle', or 'porous', got {material_type!r}"
        )

    raw_sites = get_unified_sites(
        slab,
        probe_radius=probe_radius,
        max_site_distance=max_site_distance,
        top_layer_tolerance=top_layer_tolerance,
        material_type=material_type,
        enrich=enrich,
        site_classification_method=site_classification_method,
    )
    if not raw_sites:
        return None

    sym_mode = "cluster" if material_type == "nanoparticle" else "auto"
    planar_for_symmetry = (material_type == "slab") and _is_top_layer_planar(
        slab, top_layer_tolerance
    )

    try:
        symmetry_analyzer = SymmetryAnalyzer(
            slab,
            symmetry_tolerance=symmetry_tolerance,
            mode=sym_mode,
        )
        return symmetry_analyzer.analyze_site_symmetry(
            raw_sites,
            planar=planar_for_symmetry,
        )
    except SymmetryAnalysisError as exc:
        logger.warning(
            "Symmetry analysis failed, skipping symmetry-aware sites: %s", exc
        )
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_top_layer_planar(
    slab: Atoms,
    top_layer_tolerance: float = 0.5,
    z_variance_threshold: float = 0.01,
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


def _bounding_box_cell(positions: np.ndarray, pad: float = 5.0) -> np.ndarray:
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
        top_mask = positions[:, 2] >= (z_max - 0.5)
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

    mat_type = _resolve_material_type(site, fallback="slab")

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
        nn_lo = nn * 0.7
        nn_hi = nn * 1.2
        if nn_hi - nn_lo < z_hi - z_lo:
            z_lo, z_hi = nn_lo, nn_hi

    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi
    if site is not None and str(site.get("site_type", "")) == "pore":
        return z_lo, z_hi

    r_surface = _get_site_surface_radii(slab, site)
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else 0.77

    if r_surface is None:
        return z_lo, z_hi

    r_ref = 2.0
    scale = 0.5
    delta = scale * (r_mol + r_surface - r_ref)
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

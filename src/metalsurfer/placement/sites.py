"""Site detection and clustering for adsorbate placement.

Unified Voronoi-based sites for slabs, nanoparticles, and 3D-periodic porous
materials. ``get_unified_sites()`` is the main entry; optional
``get_symmetry_aware_sites()`` applies spglib reduction on top.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

import numpy as np
from ase import Atoms
from scipy.spatial import KDTree, QhullError, Voronoi

from ..symmetry import SymmetryAnalyzer
from ._constants import (
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
from ._material import _SLAB_VACUUM_FRACTION, detect_material_type  # noqa: F401
from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)

DEFAULT_SYMMETRY_TOLERANCE = 0.1
DEFAULT_SITE_EQUIVALENCE_TOLERANCE = 0.05
DEFAULT_HOLLOW_SITE_DEDUP_TOLERANCE = 0.2


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
        logger.debug("Voronoi computation failed: %s", exc)
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
    ded_tree = KDTree(vertices)
    pairs = ded_tree.query_pairs(r=_VORONOI_DEDUP_TOLERANCE, output_type="ndarray")
    keep = np.ones(len(vertices), dtype=bool)
    for i, j in pairs:
        if keep[i] and keep[j]:
            keep[j] = False
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

    # Final deduplication of the combined set
    ded_tree = KDTree(all_verts)
    pairs = ded_tree.query_pairs(r=_VORONOI_DEDUP_TOLERANCE, output_type="ndarray")
    keep = np.ones(len(all_verts), dtype=bool)
    for i, j in pairs:
        if keep[i] and keep[j]:
            # Prefer originals over enriched: drop the higher-index one
            keep[j] = False
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

    if len(vertices) == 0:
        return []

    sites: list[dict[str, object]] = []
    for i, vertex in enumerate(vertices):
        site_type, nearest_idx = _classify_voronoi_site(
            vertex,
            positions,
            tree=local_tree,
            pore_threshold=pore_threshold,
        )
        normal = _compute_local_normal(vertex, positions, tree=local_tree)
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
# Site deduplication / clustering
# ---------------------------------------------------------------------------


def _cluster_by_distance(
    sites: list[dict[str, object]],
    coord_fn: Callable[[dict[str, object]], np.ndarray],
    key_fn: Callable[[dict[str, object]], tuple],
    dist_fn: Callable[[np.ndarray, np.ndarray], bool],
) -> list[dict[str, object]]:
    """Generic greedy clustering: keep one representative per equivalence class.

    Sorts *sites* by *key_fn* for deterministic output, then greedily selects
    representatives: a site is kept unless *dist_fn(coord_fn(site), coord_fn(rep))*
    returns True for some already-kept representative.

    Parameters
    ----------
    sites:
        Input site dicts.
    coord_fn:
        Maps a site dict to its comparison coordinate vector.
    key_fn:
        Stable sort key (must be consistent with coord_fn).
    dist_fn:
        Returns True when two coordinate vectors are close enough to be duplicates.
    """
    sorted_sites = sorted(sites, key=key_fn)
    reps: list[dict[str, object]] = []
    for s in sorted_sites:
        cs = coord_fn(s)
        if not any(dist_fn(cs, coord_fn(r)) for r in reps):
            reps.append(s)
    return sorted(reps, key=key_fn)


def _cluster_equivalent_sites(
    sites: list[dict[str, object]],
    cell: np.ndarray,
    tolerance: float = DEFAULT_SITE_EQUIVALENCE_TOLERANCE,
) -> list[dict[str, object]]:
    """Group equivalent sites by fractional coords; return unique representatives.

    - nanoparticle: Cartesian 3D.
    - porous: 3D fractional with periodic wrapping.
    - slab: 2D fractional (a,b) + absolute z.

    All three paths use the shared :func:`_cluster_by_distance` primitive.
    """
    if not sites:
        return []

    mat_type = str(sites[0].get("material_type", "slab"))

    def _get_xyz(s: dict[str, object]) -> np.ndarray:
        if "xyz" in s:
            return np.asarray(s["xyz"], dtype=float)
        return np.array([*np.asarray(s["xy"], dtype=float), float(s.get("z", 0.0))])

    # ------------------------------------------------------------------
    # Nanoparticle: Cartesian 3D distance
    # ------------------------------------------------------------------
    if mat_type == "nanoparticle" or np.linalg.det(cell) <= 0:
        return _cluster_by_distance(
            sites,
            coord_fn=_get_xyz,
            key_fn=lambda s: tuple(_get_xyz(s).tolist()),
            dist_fn=lambda a, b: float(np.linalg.norm(a - b)) < tolerance,
        )

    inv_cell = np.linalg.inv(cell)

    # ------------------------------------------------------------------
    # Porous: 3D fractional with periodic wrapping
    # ------------------------------------------------------------------
    if mat_type == "porous":

        def _frac3(s: dict[str, object]) -> np.ndarray:
            return (inv_cell @ _get_xyz(s)) % 1.0

        def _frac3_key(s: dict[str, object]) -> tuple:
            f = _frac3(s)
            return (float(f[0]), float(f[1]), float(f[2]))

        def _frac3_close(a: np.ndarray, b: np.ndarray) -> bool:
            diff = np.minimum(np.abs(a - b), 1.0 - np.abs(a - b))
            return bool(np.all(diff < tolerance))

        return _cluster_by_distance(sites, _frac3, _frac3_key, _frac3_close)

    # ------------------------------------------------------------------
    # Slab: 2D fractional + absolute z
    # ------------------------------------------------------------------
    inv_2d = np.linalg.inv(cell[:2, :2])

    def _frac_slab(s: dict[str, object]) -> np.ndarray:
        xy = np.asarray(s["xy"], dtype=float)
        frac = (inv_2d @ xy) % 1.0
        z = float(cast(float, s.get("z", 0.0)))
        return np.array([frac[0], frac[1], z])

    def _frac_slab_key(s: dict[str, object]) -> tuple:
        c = _frac_slab(s)
        return (float(c[0]), float(c[1]), float(c[2]))

    def _slab_close(a: np.ndarray, b: np.ndarray) -> bool:
        diff_xy = np.minimum(np.abs(a[:2] - b[:2]), 1.0 - np.abs(a[:2] - b[:2]))
        return bool(np.all(diff_xy < tolerance) and abs(a[2] - b[2]) < tolerance)

    return _cluster_by_distance(sites, _frac_slab, _frac_slab_key, _slab_close)


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
) -> list[dict[str, object]]:
    """Symmetry-reduced adsorption sites using spglib.

    Returns an empty list when no raw sites are found. On spglib or orbit
    verification failure, raises :exc:`~metalsurfer.symmetry.SymmetryAnalysisError`.
    Callers that need a soft fallback (e.g. workflow) should catch that exception.

    Nanoparticles use cluster-in-a-box symmetry. *material_type* must be explicit.
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
    )
    if not raw_sites:
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
        raw_sites,
        planar=planar_for_symmetry,
    )


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
        raw = site["slab_indices"]
        indices = raw if raw else None
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

    mat_type = str(site.get("material_type", "slab")) if site else "slab"

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

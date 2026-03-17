"""Site detection and clustering for adsorbate placement."""

import logging

import numpy as np
from ase import Atoms
from scipy.spatial import ConvexHull, Delaunay

from .geometry import _get_covalent_radius

logger = logging.getLogger(__name__)


def is_surface_planar(
    slab: Atoms,
    top_layer_tolerance: float = 0.5,
    z_variance_threshold: float = 0.01,
) -> bool:
    """Return True if the top layer of the slab is approximately planar.

    Identifies top layer via z >= z_max - top_layer_tolerance, fits a plane
    z = ax + by + c, and checks residual variance. Returns False for
    non-periodic slabs, too few top atoms, or variance above threshold.
    """
    positions = slab.get_positions()
    cell = slab.get_cell()
    if cell is None or len(cell) == 0 or np.linalg.det(cell) <= 0:
        return False
    pbc = slab.get_pbc()
    if not (pbc[0] and pbc[1]):
        return False

    z_max = float(np.max(positions[:, 2]))
    top_mask = positions[:, 2] >= (z_max - top_layer_tolerance)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return False

    top_pos = positions[top_indices]
    x = top_pos[:, 0]
    y = top_pos[:, 1]
    z = top_pos[:, 2]
    # Least-squares plane: z = ax + by + c
    A = np.column_stack([x, y, np.ones(len(x))])
    coeffs, residuals, rank, _ = np.linalg.lstsq(A, z, rcond=None)
    if rank < 3 or len(residuals) == 0:
        return False
    z_pred = A @ coeffs
    variance = float(np.var(z - z_pred))
    return variance < z_variance_threshold


def get_hollow_sites_for_adatoms(
    slab: Atoms,
    top_layer_tolerance: float = 0.5,
    dedup_tolerance: float = 0.2,
) -> list[np.ndarray]:
    """Get hollow site xy positions for adatom placement, with near-duplicate removal.

    Used by deposit_adatoms. Returns list of (2,) arrays.
    """
    raw = get_adsorption_sites(slab, top_layer_tolerance)
    if not raw:
        return []
    hollow_xy = [np.asarray(s["xy"]) for s in raw if s.get("site_type") == "hollow"]
    unique: list[np.ndarray] = []
    for c in hollow_xy:
        if not any(np.linalg.norm(c - u) < dedup_tolerance for u in unique):
            unique.append(c)
    return unique


def _get_hollow_sites_from_top_xy(
    top_xy: np.ndarray, top_z: float
) -> list[dict[str, object]]:
    """Get hollow sites from top-layer xy coordinates via Delaunay or sklearn fallback."""
    sites: list[dict[str, object]] = []
    try:
        tri = Delaunay(top_xy)
        for simplex in tri.simplices:
            centroid = np.mean(top_xy[simplex], axis=0)
            sites.append(
                {"xy": centroid, "site_type": "hollow", "z": top_z, "indices": simplex}
            )
        return sites
    except (RuntimeError, ValueError):
        logger.debug("Delaunay failed for hollow sites, using sklearn fallback")
        return _get_hollow_sites_from_top_xy_sklearn(top_xy, top_z)


def _get_hollow_sites_from_top_xy_sklearn(
    top_xy: np.ndarray, top_z: float
) -> list[dict[str, object]]:
    """Hollow sites via sklearn NearestNeighbors when scipy Delaunay is unavailable or fails."""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=6).fit(top_xy)
    _, indices = nbrs.kneighbors(top_xy)
    sites: list[dict[str, object]] = []
    seen: set = set()
    for nbr_idx in indices:
        centroid = np.mean(top_xy[nbr_idx[:3]], axis=0)
        key = tuple(np.round(centroid, 4))
        if key not in seen:
            seen.add(key)
            sites.append(
                {
                    "xy": centroid,
                    "site_type": "hollow",
                    "z": top_z,
                    "indices": nbr_idx[:3],
                }
            )
    return sites


def _get_bridge_sites(
    top_xy: np.ndarray, top_z: float, top_indices: np.ndarray
) -> list[dict[str, object]]:
    """Get bridge sites (midpoints of nearest-neighbor pairs) from top layer."""
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=4).fit(top_xy)
    distances, indices = nbrs.kneighbors(top_xy)
    sites: list[dict[str, object]] = []
    seen: set = set()
    for i in range(len(top_xy)):
        for j_idx in range(1, min(4, len(indices[i]))):
            j = indices[i][j_idx]
            if i >= j:
                continue
            pair = (min(i, j), max(i, j))
            if pair in seen:
                continue
            seen.add(pair)
            midpoint = 0.5 * (top_xy[i] + top_xy[j])
            sites.append(
                {
                    "xy": midpoint,
                    "site_type": "bridge",
                    "z": top_z,
                    "indices": (int(top_indices[i]), int(top_indices[j])),
                }
            )
    return sites


def get_adsorption_sites(
    slab: Atoms,
    top_layer_tolerance: float = 0.5,
) -> list[dict[str, object]] | None:
    """Detect adsorption sites (atop, bridge, hollow) on the top layer.

    Returns a list of site dicts with keys: 'xy' (2D array), 'site_type', 'z', 'indices'.
    Returns None if site detection fails (e.g. non-periodic, too few atoms).
    """
    positions = slab.get_positions()
    cell = slab.get_cell()
    if cell is None or len(cell) == 0 or np.linalg.det(cell) <= 0:
        return None
    pbc = slab.get_pbc()
    if not (pbc[0] and pbc[1]):
        return None

    z_max = float(np.max(positions[:, 2]))
    top_mask = positions[:, 2] >= (z_max - top_layer_tolerance)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return None

    top_xy = positions[top_indices, :2]

    all_sites: list[dict[str, object]] = []

    # Atop: top-layer atom positions (indices are slab indices)
    for idx in top_indices:
        all_sites.append(
            {
                "xy": positions[idx, :2].copy(),
                "site_type": "atop",
                "z": z_max,
                "indices": (int(idx),),
                "slab_indices": (int(idx),),
            }
        )

    # Bridge: midpoints of nearest-neighbor pairs
    bridge_sites = _get_bridge_sites(top_xy, z_max, top_indices)
    for s in bridge_sites:
        s["slab_indices"] = s["indices"]
    all_sites.extend(bridge_sites)

    # Hollow: Delaunay centroids (indices are top_xy indices)
    hollow_sites = _get_hollow_sites_from_top_xy(top_xy, z_max)
    for s in hollow_sites:
        s["slab_indices"] = tuple(int(top_indices[i]) for i in s["indices"])
    all_sites.extend(hollow_sites)

    return all_sites


def _sort_envelope_sites(
    sites: list[dict[str, object]],
    inv_2d: np.ndarray,
) -> list[dict[str, object]]:
    """Sort envelope sites by fractional coordinates for deterministic ordering."""

    def key(s: dict[str, object]) -> tuple[float, float, float]:
        xy = np.asarray(s["xy"])
        frac = (inv_2d @ xy) % 1.0
        z = float(s.get("z", 0.0))
        return (float(frac[0]), float(frac[1]), z)

    return sorted(sites, key=key)


def get_envelope_placement_sites(
    slab: Atoms,
    top_layer_tolerance: float = 0.5,
    cell: np.ndarray | None = None,
) -> list[dict[str, object]] | None:
    """Get placement sites from the upper envelope (convex hull) of the top layer.

    For non-planar surfaces (e.g. with adatoms), uses scipy ConvexHull to find
    upper facets and returns their centroids as placement sites. No symbol-based
    filtering; works for defects, alloys, and pre-adsorbed surfaces.

    Returns list of site dicts with keys: 'xy', 'z', 'site_type'='envelope'.
    Returns None if ConvexHull fails; fallback uses a grid-based z_surface.
    """
    positions = slab.get_positions()
    if cell is None:
        cell = slab.get_cell()
    if cell is None or len(cell) == 0 or np.linalg.det(cell) <= 0:
        return None
    pbc = slab.get_pbc()
    if not (pbc[0] and pbc[1]):
        return None

    z_max = float(np.max(positions[:, 2]))
    top_mask = positions[:, 2] >= (z_max - top_layer_tolerance)
    top_indices = np.nonzero(top_mask)[0]
    if len(top_indices) < 3:
        return None

    top_pos = positions[top_indices]

    try:
        hull = ConvexHull(top_pos)
        sites: list[dict[str, object]] = []
        for simplex in hull.simplices:
            face = top_pos[simplex]
            v1 = face[1] - face[0]
            v2 = face[2] - face[0]
            normal = np.cross(v1, v2)
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            if normal[2] > 0:
                centroid = np.mean(face, axis=0)
                slab_indices = tuple(sorted(int(top_indices[i]) for i in simplex))
                sites.append(
                    {
                        "xy": centroid[:2].copy(),
                        "z": float(centroid[2]),
                        "site_type": "envelope",
                        "slab_indices": slab_indices,
                    }
                )
        if sites:
            cell_arr = np.asarray(cell)
            inv_2d = np.linalg.inv(cell_arr[:2, :2])
            sites = _sort_envelope_sites(sites, inv_2d)
            return sites
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "ConvexHull failed for envelope sites (e.g. planar slab), using grid fallback: %s",
            exc,
        )

    # Fallback: grid-based z_surface with covalent-radius scaling from nearby atoms
    cell_arr = np.asarray(cell)
    inv_2d = np.linalg.inv(cell_arr[:2, :2])
    n_grid = 5
    sites = []
    # Use covalent-radius-scaled radius: ~2 * max covalent radius for nearby search
    symbols = slab.get_chemical_symbols()
    max_r = 0.0
    for s in symbols:
        r = _get_covalent_radius(s)
        if r is not None:
            max_r = max(max_r, r)
    radius = max(3.0, 2.5 * max_r)  # At least 3 A, or 2.5 * max covalent radius
    for i in range(n_grid):
        for j in range(n_grid):
            frac = np.array([(i + 0.5) / n_grid, (j + 0.5) / n_grid])
            xy = (cell_arr[:2, :2] @ frac).flatten()
            dxy = positions[:, :2] - xy
            frac_diff = (inv_2d @ dxy.T).T
            frac_diff = frac_diff - np.round(frac_diff)
            dxy_wrapped = (cell_arr[:2, :2] @ frac_diff.T).T
            dists_xy = np.linalg.norm(dxy_wrapped, axis=1)
            nearby = dists_xy < radius
            if np.any(nearby):
                nearby_idx = np.nonzero(nearby)[0]
                # Use top 3–5 atoms by z for slab_indices (covalent radius scaling)
                z_at_nearby = positions[nearby_idx, 2]
                top_k = min(5, len(nearby_idx))
                top_local = np.argpartition(z_at_nearby, -top_k)[-top_k:]
                slab_indices = tuple(sorted(int(nearby_idx[k]) for k in top_local))
                z_surf = float(np.max(positions[nearby_idx, 2]))
                sites.append(
                    {
                        "xy": xy.copy(),
                        "z": z_surf,
                        "site_type": "envelope",
                        "slab_indices": slab_indices,
                    }
                )
    if sites:
        sites = _sort_envelope_sites(sites, inv_2d)
    return sites if sites else None


def _get_site_surface_radii(
    slab: Atoms,
    site: dict[str, object] | None = None,
) -> float | None:
    """Mean covalent radius of surface atoms at the placement site.

    When site is provided (has slab_indices), use those atoms. Otherwise
    use the top layer (for random xy placement).
    """
    positions = slab.get_positions()
    symbols = slab.get_chemical_symbols()
    z_max = float(np.max(positions[:, 2]))

    if site is not None and "slab_indices" in site:
        indices = site["slab_indices"]
        if not indices:  # Empty tuple (e.g. envelope fallback edge case)
            indices = None
    else:
        indices = None
    if indices is None:
        top_mask = positions[:, 2] >= (z_max - 0.5)
        indices = np.nonzero(top_mask)[0]

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
) -> tuple:
    """Compute z range for placement, optionally scaled by covalent radii.

    When placement_z_scale_by_covalent_radius is True, adjusts the base range
    by the sum of molecule and surface covalent radii relative to a reference.
    """
    z_lo, z_hi = config.placement_z_range
    if not config.placement_z_scale_by_covalent_radius:
        return z_lo, z_hi

    r_surface = _get_site_surface_radii(slab, site)
    mol_radii = [_get_covalent_radius(s) for s in mol_symbols]
    mol_radii = [r for r in mol_radii if r is not None]
    r_mol = float(np.mean(mol_radii)) if mol_radii else 0.77  # fallback C

    if r_surface is None:
        return z_lo, z_hi

    # Reference: C + Ni ~ 0.77 + 1.24 = 2.01; typical z ~ 2.5
    r_ref = 2.0
    scale = 0.5  # how strongly to scale with radius
    delta = scale * (r_mol + r_surface - r_ref)
    return z_lo + delta, z_hi + delta


def _cluster_equivalent_sites(
    sites: list[dict[str, object]],
    cell: np.ndarray,
    tolerance: float = 0.05,
) -> list[dict[str, object]]:
    """Group symmetry-equivalent sites by fractional coordinates; return unique representatives."""
    if not sites:
        return []
    inv = np.linalg.inv(cell[:2, :2])

    def _frac_key(s: dict[str, object]) -> tuple[float, float, float]:
        xy = np.asarray(s["xy"])
        frac = (inv @ xy) % 1.0
        z = float(s.get("z", 0.0))
        return (float(frac[0]), float(frac[1]), z)

    sorted_sites = sorted(sites, key=_frac_key)
    representatives: list[dict[str, object]] = []
    for s in sorted_sites:
        xy = np.asarray(s["xy"])
        frac = (inv @ xy) % 1.0
        is_duplicate = False
        for rep in representatives:
            rep_xy = np.asarray(rep["xy"])
            rep_frac = (inv @ rep_xy) % 1.0
            diff = np.abs(frac - rep_frac)
            diff = np.minimum(diff, 1.0 - diff)
            if np.all(diff < tolerance):
                is_duplicate = True
                break
        if not is_duplicate:
            representatives.append(s)
    return sorted(representatives, key=_frac_key)

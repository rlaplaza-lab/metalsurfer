"""Rolling-probe (Connolly / SAS) wall-near adsorption-site candidates.

A probe sphere of soft radius equal to the accessibility target clearance sits
tangent to one, two, or three framework spheres. Those contacts are exactly
atop / bridge / hollow sites — the same geometry topology approximates on a
planar top layer, but without a lattice or Cartesian shell spray.

Geometric supports from the contact order are retained (not re-expanded), so
bridge / hollow typing and catalog density stay comparable to
``adaptive_grid``. This is wall-near sampling on every accessible face —
external surfaces, nanoparticle facets, and internal pore walls — not
free-volume / pore-centre enumeration (Voronoi).
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.spatial import KDTree

from .._utils import cell_has_volume
from ._constants import (
    _ADAPTIVE_GRID_DEFAULT_SPACING,
    _ADAPTIVE_GRID_EXPOSURE_N_STEPS,
    _ADAPTIVE_GRID_EXPOSURE_STEP,
    _ROLLING_PROBE_ATOP_SAMPLES,
    _ROLLING_PROBE_BRIDGE_SAMPLES,
    _ROLLING_PROBE_BURY_CN,
    _ROLLING_PROBE_CLASH_TOL,
    _ROLLING_PROBE_NEIGHBOR_K,
    _ROLLING_PROBE_POROUS_EXPOSURE_N_STEPS,
    _SURFACE_COVALENT_RADIUS_FALLBACK,
    _VECTOR_NORM_EPS,
)
from ._parallel import resolve_materialize_workers
from .geometry import _get_covalent_radius
from .site_adaptive_grid import (
    AdaptiveGridResult,
    AdaptiveGridSpacing,
    CandidateSite,
    SidePolicy,
    _best_per_support_key,
    _candidate_geometry_score,
    _candidates_to_arrays,
    _framework_median_nn,
    _local_surface_normal,
    _merge_by_radius,
    _ray_exposure_mask,
    _shell_target_clearance,
    _side_policy_mask,
    _support_positions_for_candidate,
    adaptive_grid_spacing,
    resolve_side_policy_for_pbc,
)
from .site_coords import (
    _minimum_image_cartesian_delta,
    _wrap_cartesian,
    project_anchor_to_support_plane,
)
from .site_plugins.helpers import periodic_accessibility_tree

_SOURCE_HINT = "rolling_probe"
_DEFAULT_N_JOBS = -2


@dataclass(frozen=True)
class _Contact:
    """Clash-free rolling-probe contact with geometric support images."""

    position: np.ndarray
    support_images: tuple[int, ...]


def _framework_radii_from_symbols(symbols: Sequence[str]) -> np.ndarray:
    """Per-atom covalent radii with a shared fallback for missing tables."""
    radii = []
    for sym in symbols:
        r = _get_covalent_radius(str(sym))
        radii.append(
            float(r)
            if r is not None and r > 0.0
            else float(_SURFACE_COVALENT_RADIUS_FALLBACK)
        )
    return np.asarray(radii, dtype=float)


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Return unit vectors approximately uniform on the sphere (Fibonacci lattice)."""
    n = max(1, int(n))
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]], dtype=float)
    i = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * i
    return np.column_stack([r * np.cos(theta), y, r * np.sin(theta)])


def _orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors spanning the plane perpendicular to *axis*."""
    a = np.asarray(axis, dtype=float)
    nrm = float(np.linalg.norm(a))
    if nrm < _VECTOR_NORM_EPS:
        return (
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 1.0, 0.0], dtype=float),
        )
    a = a / nrm
    ref = np.array([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(a, ref))) > 0.9:
        ref = np.array([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(a, ref)
    u /= float(np.linalg.norm(u))
    v = np.cross(a, u)
    return u, v


def _three_sphere_intersections(
    c1: np.ndarray,
    r1: float,
    c2: np.ndarray,
    r2: float,
    c3: np.ndarray,
    r3: float,
) -> list[np.ndarray]:
    """Return 0–2 points at distance *ri* from centre *ci* (Apollonius)."""
    p1 = np.asarray(c1, dtype=float)
    p2 = np.asarray(c2, dtype=float)
    p3 = np.asarray(c3, dtype=float)
    d12 = p2 - p1
    d = float(np.linalg.norm(d12))
    if d < _VECTOR_NORM_EPS:
        return []
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return []
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < -1e-10:
        return []
    h = float(np.sqrt(max(0.0, h2)))
    ex = d12 / d
    mid = p1 + a * ex
    d13 = p3 - p1
    ey_raw = d13 - float(np.dot(d13, ex)) * ex
    ey_nrm = float(np.linalg.norm(ey_raw))
    if ey_nrm < _VECTOR_NORM_EPS:
        dist3 = float(np.linalg.norm(mid - p3))
        if abs(dist3 - r3) <= 1e-6 and h <= 1e-8:
            return [mid.copy()]
        return []
    ey = ey_raw / ey_nrm
    ez = np.cross(ex, ey)
    x3 = float(np.dot(d13, ex))
    y3 = float(np.dot(d13, ey))
    denom = 2.0 * h * y3 if abs(y3) > _VECTOR_NORM_EPS or h > _VECTOR_NORM_EPS else 0.0
    if abs(denom) < _VECTOR_NORM_EPS:
        lhs = (a - x3) ** 2 + h * h
        if abs(lhs - r3 * r3) > 1e-6:
            return []
        if h < _VECTOR_NORM_EPS:
            return [mid.copy()]
        return [mid + h * ez, mid - h * ez]
    cos_t = ((a - x3) ** 2 + h * h + y3 * y3 - r3 * r3) / denom
    if abs(cos_t) > 1.0 + 1e-8:
        return []
    cos_t = float(np.clip(cos_t, -1.0, 1.0))
    sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t * cos_t)))
    y = h * cos_t
    out = [mid + y * ey + (h * sin_t) * ez]
    if sin_t > 1e-8:
        out.append(mid + y * ey - (h * sin_t) * ez)
    return out


def _two_sphere_circle_samples(
    c1: np.ndarray,
    r1: float,
    c2: np.ndarray,
    r2: float,
    n_samples: int,
) -> np.ndarray:
    """Sample the circle of centres tangent to two expanded spheres."""
    p1 = np.asarray(c1, dtype=float)
    p2 = np.asarray(c2, dtype=float)
    d12 = p2 - p1
    d = float(np.linalg.norm(d12))
    if d < _VECTOR_NORM_EPS:
        return np.empty((0, 3), dtype=float)
    if d > r1 + r2 + 1e-9 or d < abs(r1 - r2) - 1e-9:
        return np.empty((0, 3), dtype=float)
    a = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d)
    h2 = r1 * r1 - a * a
    if h2 < -1e-10:
        return np.empty((0, 3), dtype=float)
    h = float(np.sqrt(max(0.0, h2)))
    mid = p1 + a * (d12 / d)
    if h < _VECTOR_NORM_EPS:
        return mid.reshape(1, 3)
    u, v = _orthonormal_basis(d12)
    n = max(1, int(n_samples))
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return mid + h * (
        np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :]
    )


def _image_shift(
    image_index: int,
    n_atoms: int,
    positions: np.ndarray,
    tree_data: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
) -> tuple[int, tuple[int, int, int]]:
    """Map a KDTree image index to (base_atom, lattice shift)."""
    base = int(image_index) % n_atoms
    if not np.any(pbc) or not cell_has_volume(cell):
        return base, (0, 0, 0)
    img_pos = np.asarray(tree_data[int(image_index)], dtype=float)
    primary = np.asarray(positions[base], dtype=float)
    delta = img_pos - primary
    inv = np.linalg.inv(cell)
    frac = delta @ inv
    shift = (
        int(np.rint(frac[0])) if bool(pbc[0]) else 0,
        int(np.rint(frac[1])) if bool(pbc[1]) else 0,
        int(np.rint(frac[2])) if bool(pbc[2]) else 0,
    )
    return base, shift


def _clash_free(
    points: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    skin: float,
    *,
    ignore_image_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Return True where no non-support atom is closer than ``R_k + skin``."""
    if len(points) == 0:
        return np.ones(0, dtype=bool)
    k = min(max(4, _ROLLING_PROBE_NEIGHBOR_K), len(np.asarray(tree.data)))
    dists, idxs = tree.query(points, k=k)
    dists = np.atleast_2d(np.asarray(dists, dtype=float))
    idxs = np.atleast_2d(np.asarray(idxs, dtype=int))
    atom_idx = idxs % n_atoms
    thresholds = (
        framework_radii[atom_idx] + float(skin) - float(_ROLLING_PROBE_CLASH_TOL)
    )
    ignore = {int(i) for i in (ignore_image_indices or ())}
    keep = np.ones(len(points), dtype=bool)
    for row in range(len(points)):
        for col in range(dists.shape[1]):
            img = int(idxs[row, col])
            if img in ignore:
                continue
            if float(dists[row, col]) < float(thresholds[row, col]):
                keep[row] = False
                break
    return keep


def _atom_contacts(
    atom_i: int,
    *,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    skin: float,
    atop_dirs: np.ndarray,
) -> list[_Contact]:
    """Collect clash-free contacts seeded from framework atom *atom_i*."""
    n_atoms = len(positions)
    tree_data = np.asarray(tree.data, dtype=float)
    pos_i = np.asarray(positions[atom_i], dtype=float)
    r_i = float(framework_radii[atom_i]) + float(skin)
    k = min(_ROLLING_PROBE_NEIGHBOR_K + 1, len(tree_data))
    dists, idxs = tree.query(pos_i.reshape(1, 3), k=k)
    dists = np.atleast_1d(np.asarray(dists, dtype=float).ravel())
    idxs = np.atleast_1d(np.asarray(idxs, dtype=int).ravel())
    keep_nb: list[tuple[float, int, int]] = []
    self_imgs: set[int] = set()
    for d, img in zip(dists, idxs, strict=True):
        base = int(img) % n_atoms
        if base == atom_i and float(d) < 1e-6:
            self_imgs.add(int(img))
            continue
        keep_nb.append((float(d), int(img), base))
    self_img = next(iter(self_imgs)) if self_imgs else atom_i

    atop_pts = pos_i + r_i * atop_dirs
    atop_ok = _clash_free(
        atop_pts,
        tree,
        framework_radii,
        n_atoms,
        skin,
        ignore_image_indices=list(self_imgs),
    )
    collected: list[_Contact] = []
    if np.any(atop_ok):
        for pt in atop_pts[atop_ok]:
            collected.append(_Contact(position=pt.copy(), support_images=(self_img,)))

    nbrs: list[tuple[int, int, np.ndarray]] = []
    for dist_i, img_i, base_i in keep_nb:
        dist_f = float(dist_i)
        img_n = int(img_i)
        base_n = int(base_i)
        cutoff = float(framework_radii[atom_i] + framework_radii[base_n] + 2.0 * skin)
        if dist_f > cutoff + 1e-6:
            continue
        nbrs.append((img_n, base_n, np.asarray(tree_data[img_n], dtype=float).copy()))

    # Skip pair/triplet work only when nothing is exposed, CN is high, and
    # there are no contact-range neighbours (true bulk).
    if not collected and len(keep_nb) >= _ROLLING_PROBE_BURY_CN and not nbrs:
        return []

    for img_j, base_j, pos_j in nbrs:
        if base_j < atom_i:
            continue
        r_j = float(framework_radii[base_j]) + float(skin)
        samples = _two_sphere_circle_samples(
            pos_i, r_i, pos_j, r_j, _ROLLING_PROBE_BRIDGE_SAMPLES
        )
        if len(samples) == 0:
            continue
        ignore = list(self_imgs) + [img_j]
        ok = _clash_free(
            samples,
            tree,
            framework_radii,
            n_atoms,
            skin,
            ignore_image_indices=ignore,
        )
        for pt in samples[ok]:
            collected.append(
                _Contact(position=pt.copy(), support_images=(self_img, img_j))
            )

    for a in range(len(nbrs)):
        img_j, base_j, pos_j = nbrs[a]
        if base_j < atom_i:
            continue
        r_j = float(framework_radii[base_j]) + float(skin)
        for b in range(a + 1, len(nbrs)):
            img_k, base_k, pos_k = nbrs[b]
            if base_k < atom_i:
                continue
            d_jk = float(np.linalg.norm(pos_j - pos_k))
            if cell_has_volume(cell) and np.any(pbc):
                d_jk = float(
                    np.linalg.norm(
                        _minimum_image_cartesian_delta(pos_j - pos_k, cell, pbc)
                    )
                )
            cutoff_jk = float(
                framework_radii[base_j] + framework_radii[base_k] + 2.0 * skin
            )
            if d_jk > cutoff_jk + 1e-6:
                continue
            r_k = float(framework_radii[base_k]) + float(skin)
            sols = _three_sphere_intersections(pos_i, r_i, pos_j, r_j, pos_k, r_k)
            if not sols:
                continue
            pts = np.asarray(sols, dtype=float)
            ignore = list(self_imgs) + [img_j, img_k]
            ok = _clash_free(
                pts,
                tree,
                framework_radii,
                n_atoms,
                skin,
                ignore_image_indices=ignore,
            )
            for pt in pts[ok]:
                collected.append(
                    _Contact(
                        position=pt.copy(),
                        support_images=(self_img, img_j, img_k),
                    )
                )

    return collected


def _contacts_to_candidates(
    contacts: list[_Contact],
    *,
    positions: np.ndarray,
    framework_radii: np.ndarray,
    tree: KDTree,
    cell: np.ndarray,
    pbc: np.ndarray,
    min_clearance: float,
    max_clearance: float,
    target_clearance: float,
) -> list[CandidateSite]:
    """Build CandidateSites with geometric supports (not re-expanded)."""
    if not contacts:
        return []
    n_atoms = len(positions)
    tree_data = np.asarray(tree.data, dtype=float)
    out: list[CandidateSite] = []
    seen: set[
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...], tuple[float, ...]]
    ] = set()
    for contact in contacts:
        vert = _wrap_cartesian(
            np.asarray(contact.position, dtype=float).reshape(1, 3),
            cell,
            pbc,
        )[0]
        bases: list[int] = []
        shifts: list[tuple[int, int, int]] = []
        supp_dists: list[float] = []
        for img in contact.support_images:
            base, shift = _image_shift(img, n_atoms, positions, tree_data, cell, pbc)
            bases.append(base)
            shifts.append(shift)
            img_pos = tree_data[int(img)]
            d = float(np.linalg.norm(vert - img_pos))
            if cell_has_volume(cell) and np.any(pbc):
                d = float(
                    np.linalg.norm(
                        _minimum_image_cartesian_delta(vert - img_pos, cell, pbc)
                    )
                )
            supp_dists.append(d - float(framework_radii[base]))
        clearance = float(min(supp_dists)) if supp_dists else 0.0
        if (
            clearance < float(min_clearance) - 1e-9
            or clearance > float(max_clearance) + 1e-9
        ):
            continue
        order = sorted(range(len(bases)), key=lambda i: (supp_dists[i], bases[i]))
        bases_t = tuple(bases[i] for i in order)
        shifts_t = tuple(shifts[i] for i in order)
        dists_t = tuple(float(supp_dists[i]) for i in order)
        key = (bases_t, shifts_t, tuple(np.round(vert, 3)))
        if key in seen:
            continue
        seen.add(key)
        supp_pos = _support_positions_for_candidate(bases_t, shifts_t, positions, cell)
        normal = _local_surface_normal(vert, supp_pos, np.asarray(dists_t, dtype=float))
        score = _candidate_geometry_score(
            clearance=clearance,
            target_clearance=float(target_clearance),
            gradient=0.0,
            support_distances=dists_t,
            support_positions=supp_pos,
        )
        anchor = project_anchor_to_support_plane(vert, normal, supp_pos)
        nrm = float(np.linalg.norm(np.asarray(normal, dtype=float)))
        if (
            nrm > _VECTOR_NORM_EPS
            and len(supp_pos) > 0
            and np.any(pbc)
            and cell_has_volume(cell)
        ):
            anchor = _wrap_cartesian(anchor.reshape(1, 3), cell, pbc)[0]
        out.append(
            CandidateSite(
                position=np.asarray(anchor, dtype=float).copy(),
                clearance=clearance,
                support_indices=bases_t,
                support_image_shifts=shifts_t,
                support_distances=dists_t,
                normal=normal.copy(),
                score=float(score),
            )
        )
    return out


def _finalize_rolling_contacts(
    candidates: list[CandidateSite],
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    merge_radius: float,
) -> list[CandidateSite]:
    """One site per geometric support key, then merge_radius NMS.

    Contacts are already at the target skin clearance, so lateral snap (used by
    adaptive_grid shells) is skipped — snapping collapsed distinct bridge /
    hollow basins on flat metals.
    """
    unique = _best_per_support_key(candidates)
    return _merge_by_radius(unique, cell=cell, pbc=pbc, merge_radius=merge_radius)


def _exposure_for_pbc(
    vertices: np.ndarray,
    normals: np.ndarray,
    tree: KDTree,
    framework_radii: np.ndarray,
    n_atoms: int,
    *,
    cell: np.ndarray,
    pbc: np.ndarray,
    side_policy: SidePolicy,
    positions: np.ndarray,
) -> np.ndarray:
    """Ray exposure + side policy; shorter rays under full 3D PBC."""
    n_periodic = int(np.count_nonzero(np.asarray(pbc, dtype=bool).reshape(3)))
    n_steps = (
        int(_ROLLING_PROBE_POROUS_EXPOSURE_N_STEPS)
        if n_periodic == 3
        else int(_ADAPTIVE_GRID_EXPOSURE_N_STEPS)
    )
    keep = _ray_exposure_mask(
        vertices,
        normals,
        tree,
        framework_radii,
        n_atoms,
        step=float(_ADAPTIVE_GRID_EXPOSURE_STEP),
        n_steps=n_steps,
    )
    keep &= _side_policy_mask(
        normals,
        cell=cell,
        pbc=pbc,
        side_policy=side_policy,
        vertices=vertices,
        positions=positions,
    )
    return keep


def generate_rolling_probe_sites(
    positions: np.ndarray,
    cell: np.ndarray,
    pbc: np.ndarray,
    *,
    probe_radius: float,
    max_site_distance: float,
    n_jobs: int = _DEFAULT_N_JOBS,
    framework_radii: np.ndarray | None = None,
    symbols: Sequence[str] | None = None,
    side_policy: SidePolicy | Literal["all", "positive", "negative", "external"] = (
        "positive"
    ),
) -> AdaptiveGridResult:
    """Enumerate rolling-probe wall-near candidates with support metadata.

    *probe_radius* / *max_site_distance* map onto the clearance window
    (element-dependent surface), matching every other site plugin. Catalog
    density uses the same NMS floor as default ``adaptive_grid`` (spacing
    ``0.70`` Å). Face / exposure policy follows *pbc* geometry, not material
    labels.
    """
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    n_atoms = len(positions)
    side_policy = resolve_side_policy_for_pbc(pbc, side_policy)
    if framework_radii is None:
        if symbols is not None and len(symbols) == n_atoms:
            framework_radii = _framework_radii_from_symbols(symbols)
        else:
            framework_radii = np.full(
                n_atoms, float(_SURFACE_COVALENT_RADIUS_FALLBACK), dtype=float
            )
    else:
        framework_radii = np.asarray(framework_radii, dtype=float)
        if len(framework_radii) != n_atoms:
            raise ValueError(
                f"framework_radii length {len(framework_radii)} != n_atoms {n_atoms}"
            )

    mean_r = float(np.mean(framework_radii)) if n_atoms else 0.0
    min_clearance = float(probe_radius) - mean_r
    max_clearance = float(max_site_distance) - mean_r
    if max_clearance < min_clearance:
        max_clearance = min_clearance

    median_nn = _framework_median_nn(positions, cell, pbc)
    spacing = adaptive_grid_spacing(
        initial_spacing=float(_ADAPTIVE_GRID_DEFAULT_SPACING),
        max_levels=0,
        framework_median_nn=median_nn,
    )
    merge_r = float(spacing.merge_radius)
    tree = periodic_accessibility_tree(
        positions,
        cell,
        pbc,
        float(max_site_distance) + (float(np.max(framework_radii)) if n_atoms else 0.0),
    )
    empty = AdaptiveGridResult(
        vertices=np.empty((0, 3), dtype=float),
        nn_dists=np.empty(0, dtype=float),
        clearances=np.empty(0, dtype=float),
        atom_indices=[],
        normals=np.empty((0, 3), dtype=float),
        spacing=spacing,
        accessibility_tree=tree,
    )
    if n_atoms == 0:
        return empty

    skin = _shell_target_clearance(min_clearance, max_clearance, median_nn)
    skin = max(float(skin), max(float(min_clearance), 0.05))
    atop_dirs = _fibonacci_sphere(_ROLLING_PROBE_ATOP_SAMPLES)

    workers = resolve_materialize_workers(n_jobs, n_tasks=n_atoms)
    if workers <= 1 or n_atoms < 4:
        chunks = [
            _atom_contacts(
                i,
                positions=positions,
                framework_radii=framework_radii,
                tree=tree,
                cell=cell,
                pbc=pbc,
                skin=skin,
                atop_dirs=atop_dirs,
            )
            for i in range(n_atoms)
        ]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = [
                pool.submit(
                    _atom_contacts,
                    i,
                    positions=positions,
                    framework_radii=framework_radii,
                    tree=tree,
                    cell=cell,
                    pbc=pbc,
                    skin=skin,
                    atop_dirs=atop_dirs,
                )
                for i in range(n_atoms)
            ]
            chunks = [f.result() for f in futs]

    contacts = [c for chunk in chunks for c in chunk]
    if not contacts:
        return empty

    candidates = _contacts_to_candidates(
        contacts,
        positions=positions,
        framework_radii=framework_radii,
        tree=tree,
        cell=cell,
        pbc=pbc,
        min_clearance=min_clearance,
        max_clearance=max_clearance,
        target_clearance=skin,
    )
    if not candidates:
        return empty

    verts, _nn, _clears, _, normals = _candidates_to_arrays(candidates, tree=tree)
    keep = _exposure_for_pbc(
        verts,
        normals,
        tree,
        framework_radii,
        n_atoms,
        cell=cell,
        pbc=pbc,
        side_policy=side_policy,
        positions=positions,
    )
    candidates = [c for c, k in zip(candidates, keep, strict=True) if k]
    if not candidates:
        return empty

    clustered = _finalize_rolling_contacts(
        candidates,
        cell=cell,
        pbc=pbc,
        merge_radius=merge_r,
    )
    if not clustered:
        return empty
    verts, nn_dists, clears, atoms, normals = _candidates_to_arrays(
        clustered, tree=tree
    )
    return AdaptiveGridResult(
        vertices=verts,
        nn_dists=nn_dists,
        clearances=clears,
        atom_indices=atoms,
        normals=normals,
        spacing=AdaptiveGridSpacing(
            characteristic_length=spacing.characteristic_length,
            initial_spacing=spacing.initial_spacing,
            fine_spacing=spacing.fine_spacing,
            max_levels=0,
            merge_radius=merge_r,
        ),
        accessibility_tree=tree,
    )

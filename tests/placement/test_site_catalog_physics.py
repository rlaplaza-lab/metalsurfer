"""Catalog-physics matrix: plugins must emit physically consistent sites."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from scipy.spatial import ConvexHull, KDTree

from metalsurfer._geom_pbc import minimum_image_cartesian_delta
from metalsurfer.placement._constants import _ADAPTIVE_GRID_DEFAULT_SPACING
from metalsurfer.placement._material import material_aware_pbc
from metalsurfer.placement.geometry import framework_radii_from_symbols
from metalsurfer.placement.site_adaptive_grid import _shell_offsets
from metalsurfer.placement.site_classify import (
    _build_site_records,
    _support_locality_cutoff,
)
from metalsurfer.placement.site_coords import (
    _cart_to_frac,
    _height_along_slab_normal,
    _pore_threshold_from_mean_radius,
    _slab_normal,
    _wrap_cartesian,
    derive_pore_threshold,
    mic_support_positions,
    top_layer_mask_by_normal,
)
from metalsurfer.placement.site_enumeration import (
    _cluster_equivalent_sites,
    get_symmetry_aware_sites,
    get_unified_sites,
)
from metalsurfer.placement.site_np import _hull_signed_distances
from metalsurfer.placement.site_plugins import resolved_site_generator_name
from metalsurfer.placement.site_plugins.helpers import periodic_accessibility_tree
from metalsurfer.placement.site_types import Site, site_kind_from_type_and_supports
from metalsurfer.placement.site_wall_near_common import wall_near_clearance_window
from metalsurfer.symmetry import SymmetryAnalysisError

from ._site_systems import (
    _CHANNEL_AXIS_XY,
    _CHANNEL_R_O,
    SiteSystem,
    catalog_physics_cases,
    make_sio_channel,
)

_ALLOWED_TYPES = frozenset({"atop", "bridge", "hollow", "pore"})
_WALL_NEAR = frozenset({"adaptive_grid", "rolling_probe"})
_MOTIF_PLUGINS = frozenset(
    {"auto", "topology", "adaptive_grid", "rolling_probe", "voronoi"}
)
_OUTWARD_STEP = 0.35
_CHANNEL_AXIS_NEAR = 1.5
_CHANNEL_WALL_NEAR = 1.0


def _label(system: SiteSystem, plugin: str, sites: list[Site]) -> str:
    return (
        f"{system.name}/{plugin} n={len(sites)} "
        f"types={dict(Counter(str(s.site_type) for s in sites))}"
    )


def _tree(atoms: Atoms, material_type: str) -> KDTree:
    return periodic_accessibility_tree(
        np.asarray(atoms.get_positions(), dtype=float),
        np.asarray(atoms.get_cell(), dtype=float),
        np.asarray(material_aware_pbc(material_type), dtype=bool),
        max_distance=6.0,
    )


def _nn(tree: KDTree, point: np.ndarray) -> float:
    return float(tree.query(np.asarray(point, dtype=float).reshape(3), k=1)[0])


def _mean_covalent(symbols: list[str]) -> float:
    radii = [
        float(covalent_radii[atomic_numbers[s]]) for s in symbols if s in atomic_numbers
    ]
    return float(np.mean(radii)) if radii else 1.3


def _bond_cover_radius(symbols: list[str], *, median_nn: float | None = None) -> float:
    base = 1.35 * _mean_covalent(symbols)
    if median_nn is not None and float(median_nn) > 0.0:
        return max(base, 1.25 * float(median_nn))
    return base


def _exposed_atom_indices(atoms: Atoms, material_type: str) -> np.ndarray:
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    if material_type == "slab":
        return np.nonzero(top_layer_mask_by_normal(pos, cell, 0.6))[0]
    if material_type == "nanoparticle":
        hull = ConvexHull(pos)
        return np.unique(hull.vertices)
    return np.arange(len(atoms), dtype=int)


def _assert_shared(
    sites: list[Site], atoms: Atoms, system: SiteSystem, plugin: str
) -> None:
    label = _label(system, plugin, sites)
    assert len(sites) >= 3, f"{label}: empty/too-small catalog"
    n_atoms = len(atoms)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc(system.material_type), dtype=bool)
    tree = _tree(atoms, system.material_type)
    symbols = list(atoms.get_chemical_symbols())
    pore_thr = _pore_threshold_from_mean_radius(_mean_covalent(symbols))
    if len(pos) >= 2:
        nn_d, _ = tree.query(pos, k=2)
        median_nn = float(np.median(np.asarray(nn_d, dtype=float)[:, 1]))
    else:
        median_nn = None
    outward_ok = outward_n = 0

    for s in sites:
        st = str(s.site_type)
        assert st in _ALLOWED_TYPES, label
        assert s.material_type == system.material_type, label
        xyz = np.asarray(s.xyz, dtype=float)
        normal = np.asarray(s.normal, dtype=float)
        assert np.all(np.isfinite(xyz)) and np.all(np.isfinite(normal)), label
        assert float(np.linalg.norm(normal)) == pytest.approx(1.0, abs=1e-5), label
        tb = np.asarray(s.tangent_basis, dtype=float)
        assert tb.shape == (2, 3), label
        assert abs(float(np.linalg.norm(tb[0])) - 1.0) < 1e-5, label
        assert abs(float(np.linalg.norm(tb[1])) - 1.0) < 1e-5, label
        assert abs(float(tb[0] @ tb[1])) < 1e-5, label
        assert abs(float(tb[0] @ normal)) < 1e-5, label
        assert abs(float(tb[1] @ normal)) < 1e-5, label
        assert s.kind == site_kind_from_type_and_supports(st, s.slab_indices), label
        assert all(0 <= int(i) < n_atoms for i in s.slab_indices), label

        nn_here = _nn(tree, xyz)
        assert s.nn_distance is not None
        assert float(s.nn_distance) == pytest.approx(nn_here, abs=1e-5), label

        if st == "pore" or s.kind == "void":
            assert st == "pore" and not s.slab_indices, (
                f"{label}: void must be pore with empty supports "
                f"(got type={st!r} supports={s.slab_indices})"
            )
            assert nn_here > pore_thr, (
                f"{label}: pore nn={nn_here:.3f} <= threshold {pore_thr:.3f}"
            )
            continue

        assert s.slab_indices, f"{label}: wall site missing supports (type={st})"
        assert s.clearance is not None and float(s.clearance) > 0.0, label
        n_hat = normal / float(np.linalg.norm(normal))
        supp = mic_support_positions(xyz, pos, s.slab_indices, cell=cell, pbc=pbc)
        assert len(supp) == len(s.slab_indices), label
        # Bridges sit on the MIC bond midpoint (may be below max-support height
        # when the two atoms sit at different depths along the normal).
        if st != "bridge":
            assert float(np.dot(xyz, n_hat)) == pytest.approx(
                float(np.max(supp @ n_hat)), abs=1e-5
            ), label
        d_supp = np.asarray(
            [float(np.linalg.norm(xyz - sp)) for sp in supp], dtype=float
        )
        cutoff = _support_locality_cutoff(symbols, s.slab_indices, median_nn=median_nn)
        assert float(np.min(d_supp)) <= cutoff, (
            f"{label}: no near support (min={float(np.min(d_supp)):.3f} "
            f"cutoff={cutoff:.3f})"
        )

        # Motif geometry at the final anchor.
        if st == "atop":
            assert len(s.slab_indices) == 1, label
            assert float(np.min(d_supp)) < 0.35, label
        elif st == "bridge":
            assert len(s.slab_indices) == 2, label
            # Not collapsed onto one coordinating atom.
            assert float(np.min(d_supp)) > 0.15, label
            elems = [symbols[int(j)] for j in s.slab_indices]
            if elems[0] == elems[1]:
                assert abs(float(d_supp[0] - d_supp[1])) <= 0.40 * max(
                    float(np.mean(d_supp)), 1e-6
                ), label
        elif st == "hollow":
            assert len(s.slab_indices) >= 3, label
            assert float(np.min(d_supp)) > 0.25, (
                f"{label}: hollow coincident with an atom (min={float(np.min(d_supp)):.3f})"
            )

        step = max(_OUTWARD_STEP, 0.35 * float(s.clearance))
        outward_n += 1
        if _nn(tree, xyz + step * n_hat) >= _nn(tree, xyz - step * n_hat) - 1e-6:
            outward_ok += 1

    if outward_n:
        if system.order != "regular":
            max_bad = max(2, outward_n // 30)
        elif system.material_type == "porous":
            max_bad = max(2, outward_n // 100)
        else:
            max_bad = 0
        assert outward_n - outward_ok <= max_bad, (
            f"{label}: outward {outward_ok}/{outward_n}"
        )
    if plugin in _WALL_NEAR:
        assert all(str(s.site_type) != "pore" for s in sites), label
        assert any(str(s.site_type) == t for t in ("atop", "bridge", "hollow")), label

    if plugin in _WALL_NEAR or plugin in ("auto", "topology", "voronoi"):
        exposed = _exposed_atom_indices(atoms, system.material_type)
        wall = [s for s in sites if str(s.site_type) != "pore"]
        assert wall, label
        wxyz = np.asarray([s.xyz for s in wall], dtype=float)
        inv = np.linalg.inv(cell) if np.linalg.det(cell) != 0 else None
        cover_r = _bond_cover_radius(symbols, median_nn=median_nn)
        for ai in exposed:
            delta = wxyz - pos[int(ai)]
            if inv is not None and np.any(pbc):
                frac = delta @ inv.T
                frac[:, pbc] -= np.round(frac[:, pbc])
                delta = frac @ cell
            dmin = float(np.min(np.linalg.norm(delta, axis=1)))
            assert dmin <= cover_r, (
                f"{label}: exposed atom {int(ai)} ({symbols[int(ai)]}) "
                f"has no wall site within {cover_r:.2f} Å (dmin={dmin:.2f})"
            )

    if plugin == "adaptive_grid":
        wall = [s for s in sites if str(s.site_type) != "pore"]
        radii = framework_radii_from_symbols(symbols)
        mean_r = _mean_covalent(symbols)
        min_c, max_c = wall_near_clearance_window(
            radii, probe_radius=1.25 * mean_r, max_site_distance=4.25 * mean_r
        )
        offsets = _shell_offsets(
            float(max_c) + float(np.max(radii)),
            float(_ADAPTIVE_GRID_DEFAULT_SPACING),
            r_min=max(0.0, float(min_c) + float(np.min(radii))),
        )
        assert len(np.unique(np.round(np.linalg.norm(offsets, axis=1), 6))) >= 2, label
        clears = np.asarray(
            [float(s.clearance) for s in wall if s.clearance is not None], dtype=float
        )
        assert len(clears) == len(wall) and float(np.min(clears)) > 0.0, label
        nns = np.asarray([float(s.nn_distance) for s in wall], dtype=float)
        void_scale = (
            float(system.free_radius)
            if system.free_radius is not None
            else max(float(pore_thr) * 1.1, float(pore_thr) + 0.25)
        )
        assert float(np.median(nns)) <= pore_thr + 1e-6, label
        assert float(np.max(nns)) < void_scale + 1e-6, label


def _assert_slab(
    sites: list[Site], atoms: Atoms, system: SiteSystem, plugin: str
) -> None:
    label = _label(system, plugin, sites)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    n_hat = _slab_normal(cell)
    heights = _height_along_slab_normal(pos, cell)
    h_min, h_max = float(np.min(heights)), float(np.max(heights))
    wall = [s for s in sites if str(s.site_type) != "pore"]
    pores = [s for s in sites if str(s.site_type) == "pore"]
    assert wall, label
    # Slabs are not porous frameworks: no free-volume pore centres.
    assert not pores, f"{label}: unexpected pores on a metal slab"
    wall_h = np.asarray(
        [_height_along_slab_normal(np.asarray(s.xyz, dtype=float), cell) for s in wall]
    ).ravel()
    depth = max(2.5, 0.45 * (h_max - h_min))
    near_face = (np.abs(wall_h - h_max) <= depth) | (np.abs(wall_h - h_min) <= depth)
    near_face |= (wall_h < h_min) & ((h_min - wall_h) <= depth)
    assert float(np.mean(near_face)) >= 0.95, label
    assert float(np.max(wall_h)) >= h_max - depth, label
    dots = np.asarray([float(np.dot(np.asarray(s.normal), n_hat)) for s in wall])
    if plugin in _WALL_NEAR:
        min_med = 0.75 if system.order == "disordered" else 0.85
        assert float(np.median(np.abs(dots))) >= min_med, label
        # Stepped walls can host a few nearly in-plane contacts; require the
        # bulk of the catalog to stay face-aligned.
        assert float(np.percentile(np.abs(dots), 10)) >= 0.25, label
    else:
        min_dot = 0.35 if system.order == "disordered" else 0.85
        assert float(np.median(dots)) >= min_dot, label
        assert float(np.min(wall_h)) >= h_min + 0.35 * (h_max - h_min), label
    if system.order == "disordered" and plugin in (
        "auto",
        "topology",
        "adaptive_grid",
        "rolling_probe",
    ):
        assert np.count_nonzero(wall_h <= h_max - 1.5) >= 3, label
        assert np.count_nonzero(wall_h >= h_max - 0.8) >= 3, label
        frac_a = (
            np.asarray([s.xyz for s in wall], dtype=float) @ np.linalg.inv(cell).T
        )[:, 0]
        assert np.count_nonzero(frac_a < 0.5) >= 3, label
        assert np.count_nonzero(frac_a >= 0.5) >= 3, label
        atom_h = heights.ravel()
        n_hi = n_mid = 0
        for s in wall:
            sh = atom_h[list(s.slab_indices)]
            if np.any(sh >= h_max - 0.9):
                n_hi += 1
            if np.any((sh >= h_min + 0.35 * (h_max - h_min)) & (sh <= h_max - 1.2)):
                n_mid += 1
        assert n_hi >= 3 and n_mid >= 3, (
            f"{label}: terrace supports hi={n_hi} mid={n_mid}"
        )
    elif system.order == "disordered":
        assert float(np.max(wall_h)) >= h_max - 0.8, label
        frac_a = (
            np.asarray([s.xyz for s in wall], dtype=float) @ np.linalg.inv(cell).T
        )[:, 0]
        assert np.count_nonzero(frac_a < 0.5) >= 1, label
        assert np.count_nonzero(frac_a >= 0.5) >= 1, label
    if system.order == "regular" and plugin in ("auto", "topology", "voronoi"):
        n_top = int(np.count_nonzero(top_layer_mask_by_normal(pos, cell, 0.5)))
        by = Counter(str(s.site_type) for s in sites)
        assert by["atop"] == n_top, label
        assert by["bridge"] == 3 * n_top, label
        assert by["hollow"] == 2 * n_top, label


def _assert_nanoparticle(
    sites: list[Site], atoms: Atoms, system: SiteSystem, plugin: str
) -> None:
    label = _label(system, plugin, sites)
    pos = np.asarray(atoms.get_positions(), dtype=float)
    com = pos.mean(axis=0)
    r_max = float(np.max(np.linalg.norm(pos - com, axis=1)))
    site_r = np.asarray(
        [float(np.linalg.norm(np.asarray(s.xyz, dtype=float) - com)) for s in sites]
    )
    assert float(np.median(site_r)) >= 0.50 * r_max, label
    min_r_frac = 0.05 if system.order == "disordered" else 0.15
    assert float(np.min(site_r)) >= min_r_frac * r_max, label
    dirs = [
        (np.asarray(s.xyz) - com) / n
        for s in sites
        if (n := float(np.linalg.norm(np.asarray(s.xyz) - com))) > 1e-8
    ]
    assert dirs and float(np.linalg.norm(np.mean(dirs, axis=0))) < 0.45, label
    # Both hemispheres along the lab z axis (rolling probe must not keep one cap).
    z_off = np.asarray([float(np.asarray(s.xyz)[2] - com[2]) for s in sites])
    assert np.any(z_off > 0.15 * r_max) and np.any(z_off < -0.15 * r_max), label
    hull = ConvexHull(pos)
    hull_ok = 0
    for s in sites:
        n_hat = np.asarray(s.normal, dtype=float)
        n_hat /= float(np.linalg.norm(n_hat))
        eps = float(s.clearance) if s.clearance is not None else 0.5
        probe = np.asarray(s.xyz, dtype=float) + max(eps, 0.1) * n_hat
        exits = float(np.max(_hull_signed_distances(probe.reshape(1, 3), hull))) > 0.0
        outward_com = float(np.dot(n_hat, np.asarray(s.xyz) - com)) > 0.0
        if exits and outward_com:
            hull_ok += 1
    if plugin in ("auto", "topology") or system.order == "regular":
        assert hull_ok == len(sites) or (
            plugin in _WALL_NEAR and len(sites) - hull_ok <= 2
        ), f"{label}: hull/COM outward {hull_ok}/{len(sites)}"
    else:
        assert hull_ok >= len(sites) - 4, f"{label}: hull/COM {hull_ok}/{len(sites)}"


def _assert_porous(
    sites: list[Site], atoms: Atoms, system: SiteSystem, plugin: str
) -> None:
    label = _label(system, plugin, sites)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(material_aware_pbc(system.material_type), dtype=bool)
    tree = _tree(atoms, system.material_type)
    symbols = list(atoms.get_chemical_symbols())
    pore_thr = _pore_threshold_from_mean_radius(_mean_covalent(symbols))
    for s in sites:
        xyz = np.asarray(s.xyz, dtype=float)
        wrapped = _wrap_cartesian(xyz.reshape(1, 3), cell, pbc)[0]
        assert np.all(np.abs(_cart_to_frac(xyz - wrapped, cell)) < 1.5 + 1e-6), label

    wall = [s for s in sites if str(s.site_type) != "pore"]
    pores = [s for s in sites if str(s.site_type) == "pore"]
    if plugin in _WALL_NEAR:
        assert wall and not pores, label
        wall_types = {str(s.site_type) for s in wall}
        if plugin == "adaptive_grid":
            assert {"atop", "bridge", "hollow"} <= wall_types, label
        else:
            # Rolling probe may be atop-heavy in narrow pores.
            assert "atop" in wall_types and len(wall_types) >= 1, label
            assert len(wall) >= 3, label
    elif plugin in ("auto", "voronoi"):
        assert wall and pores, label
        assert len(pores) >= 2, label
        assert all(_nn(tree, np.asarray(s.xyz)) > pore_thr for s in pores), label

    if system.channel_axis_xy is None:
        return
    axis = np.asarray(system.channel_axis_xy, dtype=float)
    free_r = float(system.free_radius or 2.0)

    def d_axis(site: Site) -> float:
        return float(np.linalg.norm(np.asarray(site.xyz, dtype=float)[:2] - axis))

    if plugin in _WALL_NEAR:
        assert float(np.median([d_axis(s) for s in sites])) >= 0.55 * free_r, label
        assert all(d_axis(s) >= 0.35 for s in sites), label
        pos = np.asarray(atoms.get_positions(), dtype=float)
        site_z = np.asarray([float(np.asarray(s.xyz)[2]) for s in wall])
        for rz in (float(np.min(pos[:, 2])), float(np.max(pos[:, 2]))):
            assert np.any(np.abs(site_z - rz) <= 1.25), label
        into_z: list[float] = []
        for s in wall:
            da = d_axis(s)
            if not (0.5 * free_r <= da <= 1.15 * float(_CHANNEL_R_O) + 0.75):
                continue
            xy = np.asarray(s.xyz, dtype=float)[:2]
            radial = axis - xy
            rn = float(np.linalg.norm(radial))
            n_xy = np.asarray(s.normal, dtype=float)[:2]
            nn_xy = float(np.linalg.norm(n_xy))
            if rn < 1e-8 or nn_xy < 0.25:
                continue
            if float(np.dot(n_xy / nn_xy, radial / rn)) >= 0.35:
                into_z.append(float(np.asarray(s.xyz)[2]))
        assert len(into_z) >= 4, label
        assert float(np.min(into_z)) <= float(np.min(pos[:, 2])) + 1.25, label
        assert float(np.max(into_z)) >= float(np.max(pos[:, 2])) - 1.25, label
        return
    if plugin in ("auto", "voronoi"):
        near = [s for s in pores if d_axis(s) <= _CHANNEL_AXIS_NEAR]
        assert len(near) >= 2, label
        assert all(d_axis(s) <= free_r for s in near), label
        assert any(
            _nn(tree, np.asarray(s.xyz)) >= 0.55 * free_r
            or (s.nn_distance is not None and float(s.nn_distance) >= 0.55 * free_r)
            for s in near
        ), label
        wall_d = np.asarray([d_axis(s) for s in wall], dtype=float)
        assert float(np.median(wall_d)) >= 0.7 * free_r, label
        assert float(np.min(wall_d)) >= free_r * 0.5, label
        if system.order == "regular":
            near_d = np.asarray([d_axis(s) for s in near], dtype=float)
            assert float(np.median(near_d)) < float(np.median(wall_d)) - 0.3, label
            assert float(np.min(wall_d)) >= _CHANNEL_WALL_NEAR * 0.5, label


def _assert_auto_bar(sites: list[Site], system: SiteSystem, plugin: str) -> None:
    types = {str(s.site_type) for s in sites}
    label = _label(system, plugin, sites)
    if plugin in _MOTIF_PLUGINS and system.material_type in ("slab", "nanoparticle"):
        by = Counter(str(s.site_type) for s in sites)
        if plugin == "voronoi" and system.order == "disordered":
            # Explicit Voronoi on rough slabs is sparse (no topology mesh).
            assert {"atop", "bridge"} <= types, label
            assert by["atop"] >= 2 and by["bridge"] >= 1, label
            return
        assert {"atop", "bridge", "hollow"} <= types, label
        assert by["atop"] >= 2 and by["bridge"] >= 2 and by["hollow"] >= 2, label


def _assert_auto_matches(
    atoms: Atoms, system: SiteSystem, auto_sites: list[Site]
) -> None:
    resolved = resolved_site_generator_name("auto", system.material_type)
    named = get_unified_sites(
        atoms, material_type=system.material_type, site_generator=resolved, n_jobs=1
    )
    label = f"{system.name}/auto↔{resolved}"
    assert len(auto_sites) == len(named) > 0, label
    assert [s.site_type for s in auto_sites] == [s.site_type for s in named], label
    np.testing.assert_allclose(
        np.asarray([s.xyz for s in auto_sites], dtype=float),
        np.asarray([s.xyz for s in named], dtype=float),
        atol=1e-6,
        err_msg=label,
    )


def _assert_symmetry(
    sites: list[Site], atoms: Atoms, system: SiteSystem, plugin: str
) -> None:
    if system.order != "regular":
        return
    label = _label(system, plugin, sites)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    clustered = _cluster_equivalent_sites(sites, cell)
    assert clustered, label
    try:
        sym = get_symmetry_aware_sites(
            atoms,
            material_type=system.material_type,
            raw_sites=clustered,
            site_generator=plugin if plugin != "auto" else "auto",
            n_jobs=1,
        )
    except SymmetryAnalysisError:
        if system.material_type == "porous" and system.channel_axis_xy is None:
            return
        raise
    assert sym, label
    mult_sum = sum(int(s.symmetry_multiplicity or 0) for s in sym)
    assert mult_sum == len(clustered), label
    if system.name != "slab_regular":
        return
    by = Counter(str(s.site_type) for s in sym)
    assert by["atop"] == 1 and by["bridge"] == 1, label
    if plugin == "adaptive_grid":
        assert by["hollow"] >= 2, label
    else:
        assert by["hollow"] == 2, label


_GEOMETRY = {
    "slab": _assert_slab,
    "nanoparticle": _assert_nanoparticle,
    "porous": _assert_porous,
}


@pytest.mark.parametrize(
    ("system", "plugin"),
    [
        pytest.param(
            system,
            plugin,
            id=f"{system.name}-{plugin}",
            marks=[pytest.mark.slow] if system.slow else [],
        )
        for system, plugin in catalog_physics_cases()
    ],
)
def test_site_catalog_physics(system: SiteSystem, plugin: str):
    atoms = system.factory()
    sites = get_unified_sites(
        atoms, material_type=system.material_type, site_generator=plugin, n_jobs=1
    )
    _assert_shared(sites, atoms, system, plugin)
    _assert_auto_bar(sites, system, plugin)
    _GEOMETRY[system.material_type](sites, atoms, system, plugin)
    _assert_symmetry(sites, atoms, system, plugin)
    if plugin == "auto":
        _assert_auto_matches(atoms, system, sites)


def test_support_locality_keeps_hollow_drops_ghost_support():
    cell = np.diag([20.0, 20.0, 20.0])
    pbc = np.array([True, True, True], dtype=bool)
    positions = np.array(
        [[0.0, 0.0, 0.0], [2.77, 0.0, 0.0], [1.385, 2.40, 0.0], [10.0, 10.0, 10.0]],
        dtype=float,
    )
    symbols = ["Pt", "Pt", "Pt", "Pt"]
    vertex = np.mean(positions[:3], axis=0).reshape(1, 3)
    normal = np.array([[0.0, 0.0, 1.0]], dtype=float)
    clearance = np.array([1.0], dtype=float)
    tree = KDTree(positions)
    assert 2.0 < _support_locality_cutoff(symbols, (0, 1, 2), median_nn=2.77) < 6.0
    common = dict(
        cell=cell,
        pbc=pbc,
        source_hints=["adaptive_grid"],
        normals=normal,
        clearances=clearance,
    )
    kept = _build_site_records(
        vertex,
        positions,
        symbols,
        tree,
        "porous",
        2.5,
        atom_indices=[(0, 1, 2)],
        **common,
    )
    assert len(kept) == 1 and str(kept[0].site_type) == "hollow"
    assert (
        _build_site_records(
            vertex,
            positions,
            symbols,
            tree,
            "porous",
            2.5,
            atom_indices=[(3, 3, 3)],
            **common,
        )
        == []
    )


def test_sio_channel_has_tetrahedral_bond_length():
    atoms = make_sio_channel()
    pos = np.asarray(atoms.get_positions(), dtype=float)
    cell = np.asarray(atoms.get_cell(), dtype=float)
    pbc = np.asarray(atoms.get_pbc(), dtype=bool)
    symbols = list(atoms.get_chemical_symbols())
    si = [i for i, s in enumerate(symbols) if s == "Si"]
    ox = [i for i, s in enumerate(symbols) if s == "O"]
    bonds = []
    for oi in ox:
        ds = [
            float(
                np.linalg.norm(
                    minimum_image_cartesian_delta(pos[oi] - pos[si_i], cell, pbc)
                )
            )
            for si_i in si
        ]
        bonds.append(min(ds))
    assert bonds
    assert float(np.mean(bonds)) == pytest.approx(1.62, abs=0.02)
    axis = np.asarray(_CHANNEL_AXIS_XY, dtype=float)
    axis_nn = float(np.min(np.linalg.norm(pos[:, :2] - axis, axis=1)))
    assert axis_nn == pytest.approx(float(_CHANNEL_R_O), abs=0.02)
    pore_thr = derive_pore_threshold(symbols)
    assert axis_nn > pore_thr, (
        f"channel axis nn {axis_nn:.3f} must exceed pore threshold {pore_thr:.3f}"
    )

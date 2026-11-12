"""Slab site paths explore the vacuum-richer face."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import fcc111

from metalsurfer._geom_pbc import (
    height_along_slab_normal,
    slab_outward_normal,
    slab_vacuum_face_sign,
)
from metalsurfer.placement.site_coords import top_layer_mask_by_normal
from metalsurfer.placement.site_enumeration import get_unified_sites
from metalsurfer.surface_prep._surfaces import ensure_slab_z_alignment


def _inverted_pt_slab() -> Atoms:
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=12.0, periodic=True)
    slab.set_pbc([True, True, False])
    pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    c_len = float(np.linalg.norm(cell[2]))
    z_min = float(np.min(pos[:, 2]))
    z_max = float(np.max(pos[:, 2]))
    # ~1 Å vacuum above; the rest below.
    pos[:, 2] = pos[:, 2] - z_min + (c_len - (z_max - z_min) - 1.0)
    slab.set_positions(pos)
    return slab


def test_slab_vacuum_face_sign_aligned_vs_inverted():
    aligned = fcc111("Pt", size=(3, 3, 3), vacuum=12.0, periodic=True)
    aligned.set_pbc([True, True, False])
    aligned = ensure_slab_z_alignment(aligned)
    a_pos = np.asarray(aligned.get_positions(), dtype=float)
    a_cell = np.asarray(aligned.get_cell(), dtype=float)
    assert slab_vacuum_face_sign(a_pos, a_cell) == 1.0

    inv = _inverted_pt_slab()
    pos = np.asarray(inv.get_positions(), dtype=float)
    cell = np.asarray(inv.get_cell(), dtype=float)
    assert slab_vacuum_face_sign(pos, cell) == -1.0
    assert float(slab_outward_normal(pos, cell)[2]) < 0.0


def test_near_equal_vacuum_keeps_positive_ab_face():
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    slab.set_pbc([True, True, False])
    pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    assert slab_vacuum_face_sign(pos, cell) == 1.0
    top = np.nonzero(top_layer_mask_by_normal(pos, cell, 0.5))[0]
    pos = pos.copy()
    pos[top[1], 2] += 0.45
    assert slab_vacuum_face_sign(pos, cell) == 1.0


def test_top_layer_mask_follows_vacuum_richer_face():
    slab = _inverted_pt_slab()
    pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    mask = top_layer_mask_by_normal(pos, cell, 0.5)
    heights = height_along_slab_normal(pos, cell)
    assert float(np.max(heights[mask])) <= float(np.min(heights[~mask])) + 1e-9


def test_all_site_plugins_explore_vacuum_richer_face_on_inverted_slab():
    slab = _inverted_pt_slab()
    pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    heights = height_along_slab_normal(pos, cell)
    h_mid = 0.5 * (float(np.min(heights)) + float(np.max(heights)))
    outward = slab_outward_normal(pos, cell)
    for plugin in ("topology", "voronoi", "adaptive_grid", "rolling_probe"):
        sites = get_unified_sites(
            slab,
            site_generator=plugin,
            n_jobs=1,
            material_type="slab",
        )
        assert sites
        site_h = height_along_slab_normal(
            np.asarray([s.xyz for s in sites], dtype=float), cell
        )
        assert float(np.mean(site_h <= h_mid)) >= 0.95
        dots = np.asarray(
            [float(np.dot(np.asarray(s.normal, dtype=float), outward)) for s in sites]
        )
        assert float(np.mean(dots >= -1e-6)) >= 0.95

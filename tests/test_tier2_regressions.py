"""Regression tests for the Tier 2 correctness fixes.

Each test pins a defect that was silent: it produced no exception and no failing
test, but changed scientific output (site geometry, site labels, orbit counts,
saturation bookkeeping, substrate constraints). Keep these specific to the
failure mode rather than to the surrounding implementation.
"""

import logging
from collections import Counter

import numpy as np
import pytest
from ase.build import fcc100, fcc111, hcp0001

from metalsurfer.placement.site_coords import (
    _slab_normal,
    top_layer_mask_by_normal,
)
from metalsurfer.placement.site_enumeration import get_unified_sites

# ---------------------------------------------------------------------------
# #1 slab site normals are the exact a×b surface normal
# ---------------------------------------------------------------------------


def _rotated(atoms, angle_deg: float):
    rotated = atoms.copy()
    rotated.rotate(angle_deg, "x", rotate_cell=True)
    return rotated


@pytest.mark.parametrize(
    "slab_factory",
    [
        lambda: fcc111("Pt", (3, 3, 4), vacuum=10.0),
        lambda: fcc100("Cu", (3, 3, 4), vacuum=10.0),
        lambda: hcp0001("Ru", (3, 3, 4), vacuum=10.0),
    ],
    ids=["fcc111", "fcc100", "hcp0001"],
)
def test_slab_site_normals_are_exactly_the_slab_normal(slab_factory):
    """A k-nearest centroid tilted slab normals by up to 56°; use a×b instead."""
    slab = slab_factory()
    cell = np.asarray(slab.get_cell(), dtype=float)
    expected = _slab_normal(cell)

    sites = get_unified_sites(slab, material_type="slab")
    assert sites

    normals = np.array([s.normal for s in sites], dtype=float)
    assert np.allclose(normals, expected, atol=1e-12)
    # Unrotated ASE slabs have the surface normal along +z.
    assert np.allclose(normals[:, 2], 1.0, atol=1e-12)


def test_rotated_slab_site_normals_follow_the_rotated_cell():
    """Normals track the cell, not Cartesian z, when the slab is rotated."""
    slab = _rotated(fcc111("Pt", (3, 3, 4), vacuum=10.0), 30.0)
    cell = np.asarray(slab.get_cell(), dtype=float)
    expected = _slab_normal(cell)
    assert abs(float(expected[2])) < 0.99  # genuinely tilted

    sites = get_unified_sites(slab, material_type="slab")
    assert sites
    normals = np.array([s.normal for s in sites], dtype=float)
    assert np.allclose(normals, expected, atol=1e-12)


@pytest.mark.parametrize("size", [3, 4, 5])
def test_hollow_count_is_twice_the_top_layer_on_fcc111(size):
    """Accessibility gating is periodic: no hollow is dropped at an a/b boundary."""
    slab = fcc111("Pt", (size, size, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))
    assert n_top == size * size

    sites = get_unified_sites(slab, material_type="slab")
    n_hollow = sum(1 for s in sites if s.site_source == "topology_hollow")
    assert n_hollow == 2 * n_top


# ---------------------------------------------------------------------------
# #2 cross-boundary hollows keep their hollow label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slab_factory",
    [
        lambda: fcc111("Pt", (3, 3, 4), vacuum=10.0),
        lambda: fcc111("Pt", (4, 4, 4), vacuum=10.0),
        lambda: fcc111("Pt", (5, 5, 4), vacuum=10.0),
        lambda: hcp0001("Ru", (3, 3, 4), vacuum=10.0),
    ],
    ids=["fcc111-3", "fcc111-4", "fcc111-5", "hcp0001-3"],
)
def test_site_type_matches_topology_source_on_close_packed_slabs(slab_factory):
    """Every generated site keeps the label its generator intended."""
    slab = slab_factory()
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    n_top = int(np.count_nonzero(top_layer_mask_by_normal(positions, cell, 0.5)))

    sites = get_unified_sites(slab, material_type="slab")
    assert sites
    mismatches = [
        (s.site_type, s.site_source)
        for s in sites
        if s.site_type != s.site_source.removeprefix("topology_")
    ]
    assert mismatches == []
    assert Counter(s.site_type for s in sites) == Counter(
        {"atop": n_top, "bridge": 3 * n_top, "hollow": 2 * n_top}
    )


def test_expanded_classification_index_carries_hollow_candidates():
    """The ±1 a/b index must emit hollows, not only cross-boundary bridges."""
    from scipy.spatial import Delaunay

    from metalsurfer.placement.site_coords import _project_to_slab_plane
    from metalsurfer.placement.site_voronoi import (
        _build_delaunay_classification_index,
    )

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    top_idx = np.nonzero(top_layer_mask_by_normal(positions, cell, 0.5))[0]
    top_2d = _project_to_slab_plane(positions[top_idx], cell)
    tri = Delaunay(top_2d)

    _, types_primary, _ = _build_delaunay_classification_index(top_2d, top_idx, tri)
    _, types_pbc, _ = _build_delaunay_classification_index(
        top_2d,
        top_idx,
        tri,
        cell=cell,
        pbc=np.array([True, True, False]),
    )
    counts_primary = Counter(types_primary)
    counts_pbc = Counter(types_pbc)
    assert counts_pbc["hollow"] > counts_primary["hollow"]
    assert counts_pbc["bridge"] > counts_primary["bridge"]


# ---------------------------------------------------------------------------
# #3 the Voronoi pass on planar slabs is skipped explicitly, not swallowed
# ---------------------------------------------------------------------------


def test_planar_slab_skips_voronoi_and_says_so(caplog, monkeypatch):
    """Qhull used to raise QH6154 on the coplanar top layer and be swallowed."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    def _fail(*_args, **_kwargs):
        raise AssertionError("_voronoi_sites must not run on a planar slab")

    monkeypatch.setattr(site_enumeration, "_voronoi_sites", _fail)

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.INFO, logger="metalsurfer.placement.site_enumeration"):
        sites = site_enumeration.get_unified_sites(slab, material_type="slab")

    assert sites
    assert any(
        "skipping Voronoi vertex generation" in record.getMessage()
        for record in caplog.records
    )


def test_non_planar_slab_still_runs_voronoi():
    """Only coplanar top layers skip the Voronoi pass."""
    import metalsurfer.placement.site_enumeration as site_enumeration

    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    positions = slab.get_positions()
    positions[-1, 2] += 1.5  # adatom-like bump breaks coplanarity
    slab.set_positions(positions)

    cell = np.asarray(slab.get_cell(), dtype=float)
    assert not site_enumeration._top_layer_is_planar_from_arrays(
        np.asarray(slab.get_positions(), dtype=float), cell, 0.5
    )


def test_slab_only_enrichment_flag_warns(caplog):
    """voronoi_site_enrichment is porous/nanoparticle-only; slabs get a warning."""
    from metalsurfer.config import AdsorptionConfig
    from metalsurfer.placement.site_context import (
        _get_unique_sites_for_specs,
        clear_site_caches,
    )

    clear_site_caches()
    config = AdsorptionConfig(material_type="slab", voronoi_site_enrichment=False)
    slab = fcc111("Pt", (3, 3, 4), vacuum=10.0)
    with caplog.at_level(logging.WARNING, logger="metalsurfer.placement.site_context"):
        _get_unique_sites_for_specs(slab, config)
    clear_site_caches()

    assert any(
        "voronoi_site_enrichment=False has no effect" in record.getMessage()
        for record in caplog.records
    )

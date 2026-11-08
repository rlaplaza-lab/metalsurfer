"""Even wall heights and sparse sites on defects and ligands."""

import numpy as np
import pytest
from ase import Atoms
from ase.cluster import Icosahedron

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import PlacementSpec
from metalsurfer.placement import enumerate_placement_specs, get_unified_sites
from metalsurfer.placement.geometry import min_pair_clearance_angstrom
from metalsurfer.placement.pose import _pose_from_spec
from metalsurfer.placement.pose.height import _vdw_pair_clearance
from metalsurfer.placement.site_context import site_context_for_sampling
from metalsurfer.placement.site_coords import exposed_lower_band_indices

from ..conftest import make_nanoparticle, make_porous_framework, make_slab, make_water


def _square_layer(n: int, z: float, spacing: float = 2.7) -> list[list[float]]:
    return [[ix * spacing, iy * spacing, z] for ix in range(n) for iy in range(n)]


def _slab_from_positions(positions: list[list[float]], symbols: str = "Cu") -> Atoms:
    pos = np.asarray(positions, dtype=float)
    span = float(np.max(pos[:, :2]) - np.min(pos[:, :2]) + 2.7)
    return Atoms(
        symbols=[symbols] * len(pos),
        positions=pos,
        cell=[span, span, 20.0],
        pbc=[True, True, False],
    )


def _supports(sites) -> set[int]:
    out: set[int] = set()
    for site in sites:
        out.update(int(i) for i in site.slab_indices)
    return out


def test_flat_multilayer_does_not_mesh_the_covered_layer():
    spacing = 2.7
    top = _square_layer(4, 4.2, spacing)
    buried = _square_layer(4, 2.1, spacing)
    flat = _slab_from_positions(top + buried)
    sites = get_unified_sites(
        flat, material_type="slab", top_layer_tolerance=0.5, site_generator="topology"
    )
    n_top = len(top)
    assert sites
    assert _supports(sites).isdisjoint(set(range(n_top, len(flat))))
    top_only = get_unified_sites(
        _slab_from_positions(top),
        material_type="slab",
        top_layer_tolerance=0.5,
        site_generator="topology",
    )
    assert len(sites) == len(top_only)


def test_monatomic_step_adds_lower_terrace_atop_not_buried_atoms():
    spacing = 2.7
    lower = [[ix * spacing, iy * spacing, 0.0] for ix in range(4) for iy in range(2)]
    upper = [
        [ix * spacing, (iy + 2) * spacing, 2.2] for ix in range(4) for iy in range(2)
    ]
    buried = [
        [ix * spacing, (iy + 2) * spacing, 0.0] for ix in range(4) for iy in range(2)
    ]
    n_lower = len(lower)
    n_upper = len(upper)
    slab = _slab_from_positions(lower + upper + buried)
    sites = get_unified_sites(
        slab, material_type="slab", top_layer_tolerance=0.5, site_generator="topology"
    )
    supports = _supports(sites)
    assert supports & set(range(n_lower))
    buried_idx = set(range(n_lower + n_upper, len(slab)))
    assert supports.isdisjoint(buried_idx)
    atop_supports = {
        int(site.slab_indices[0])
        for site in sites
        if site.site_type == "atop" and len(site.slab_indices) == 1
    }
    assert atop_supports & set(range(n_lower))


def test_vacancy_floor_atom_is_a_site_support():
    spacing = 2.7
    top = _square_layer(4, 4.4, spacing)
    # Drop the centre of the top layer. The atom below that hole is the floor.
    hole = 4 * 1 + 1  # ix=1, iy=1
    top_kept = [p for i, p in enumerate(top) if i != hole]
    floor_xy = top[hole][:2]
    buried = _square_layer(4, 2.2, spacing)
    n_top = len(top_kept)
    floor_index = n_top + hole
    slab = _slab_from_positions(top_kept + buried)
    assert np.allclose(slab.positions[floor_index][:2], floor_xy)
    sites = get_unified_sites(
        slab, material_type="slab", top_layer_tolerance=0.5, site_generator="topology"
    )
    assert floor_index in _supports(sites)
    # Atoms still directly under a remaining top atom stay out.
    covered = n_top  # ix=0, iy=0 of the buried layer, aligned with a kept top atom
    assert covered not in _supports(sites)


def test_exposed_lower_band_empty_on_flat_stack():
    positions = np.asarray(_square_layer(3, 4.2) + _square_layer(3, 2.1), dtype=float)
    cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
    primary = np.arange(9)
    extra = exposed_lower_band_indices(
        positions,
        primary,
        ["Cu"] * len(positions),
        cell,
        np.array([True, True, False]),
        2.7,
    )
    assert len(extra) == 0


def _com_heights(atoms, material: str, *, full=None):
    mol = make_water()
    config = AdsorptionConfig(material_type=material, seed=0, num_conformers=1)
    ctx = site_context_for_sampling(atoms, config, full_slab=full)
    wall = next(i for i, site in enumerate(ctx.sites) if site.kind != "void")
    site = ctx.sites[wall]
    o_idx = list(mol.get_chemical_symbols()).index("O")
    heights = []
    for zf in (0.5, 0.7, 0.9):
        spec = PlacementSpec(
            conformer_index=0,
            orientation_type="EN-down",
            face_flip=False,
            en_atom_index=o_idx,
            site_index=wall,
            site_type=str(site.site_type),
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=zf,
            placement_index=0,
        )
        posed, fail = _pose_from_spec(
            mol,
            spec,
            full if full is not None else atoms,
            config,
            "O",
            site_context=ctx,
            slab_for_sites=atoms if full is not None else None,
        )
        assert fail is None and posed is not None
        n_hat = np.asarray(posed.normal, dtype=float)
        com = np.array(
            [posed.pose.x_abs, posed.pose.y_abs, posed.pose.z_abs], dtype=float
        )
        heights.append(float(np.dot(com, n_hat)))
    return heights


@pytest.mark.parametrize(
    ("material", "factory"),
    [
        ("slab", make_slab),
        ("nanoparticle", make_nanoparticle),
        ("porous", make_porous_framework),
    ],
)
def test_wall_z_fractions_are_evenly_spaced(material, factory):
    heights = _com_heights(factory(), material)
    step = heights[1] - heights[0]
    assert step == pytest.approx(heights[2] - heights[1], abs=1e-6)
    assert step > 0.3


def test_wall_ends_match_covalent_and_vdw_gates():
    slab = make_slab()
    mol = make_water()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx = site_context_for_sampling(slab, config)
    wall = next(i for i, site in enumerate(ctx.sites) if site.site_type == "atop")
    site = ctx.sites[wall]
    o_idx = list(mol.get_chemical_symbols()).index("O")
    poses = {}
    for zf in (0.5, 0.7, 0.9):
        spec = PlacementSpec(
            conformer_index=0,
            orientation_type="EN-down",
            face_flip=False,
            en_atom_index=o_idx,
            site_index=wall,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=zf,
            placement_index=0,
        )
        posed, fail = _pose_from_spec(mol, spec, slab, config, "O", site_context=ctx)
        assert fail is None and posed is not None
        poses[zf] = posed
    support = int(site.slab_indices[0])
    metal = slab.get_chemical_symbols()[support]
    covalent = min_pair_clearance_angstrom(
        "O",
        metal,
        min_distance=config.min_initial_distance,
        min_contact_ratio=config.min_contact_ratio,
    )
    vdw = _vdw_pair_clearance(config, "O", metal)
    standoff = max(vdw, covalent + 0.8)
    distances = {}
    for zf in (0.5, 0.7, 0.9):
        com = np.array(
            [poses[zf].pose.x_abs, poses[zf].pose.y_abs, poses[zf].pose.z_abs]
        )
        oxygen = com + poses[zf].rotated_pos[o_idx]
        distances[zf] = float(np.linalg.norm(oxygen - slab.positions[support]))
    # The contact pad is 1e-6 Å. Nearest-atom distance matches the gate.
    assert distances[0.5] == pytest.approx(covalent, abs=1e-3)
    assert distances[0.9] == pytest.approx(standoff, abs=1e-3)
    assert distances[0.7] == pytest.approx(
        0.5 * (distances[0.5] + distances[0.9]), abs=1e-3
    )


def test_ligand_sites_and_occupancy_and_clearance():
    top = _square_layer(4, 0.0)
    slab = _slab_from_positions(top)
    # Ligand above a hollow-side metal, inside the neighbour window of a neighbour.
    ligand_pos = np.array([1.3, 0.0, 2.3], dtype=float)
    full = slab.copy()
    ligand = Atoms("O", positions=[ligand_pos])
    covered = full + ligand
    config = AdsorptionConfig(material_type="slab", seed=0, num_placements=4)
    ctx = site_context_for_sampling(slab, config, full_slab=covered)
    ligand_sites = [
        site
        for site in ctx.sites
        if any(int(i) >= len(slab) for i in site.slab_indices)
    ]
    assert any(site.site_type == "atop" for site in ligand_sites)
    assert any(site.site_type == "bridge" for site in ligand_sites)
    mol = make_water()
    # The first type/height pass stays on the open terrace. Ligand sites are
    # in the occupancy-count fill after that pass.
    specs = enumerate_placement_specs(
        [mol],
        slab,
        config,
        "O",
        n_desired=16,
        site_context=ctx,
        full_slab=covered,
    )
    assert specs
    kinds = []
    for spec in specs:
        site = ctx.sites[spec.site_index]
        kinds.append(any(int(i) >= len(slab) for i in site.slab_indices))
    assert any(kinds)
    assert any(not flag for flag in kinds)
    # Metal column under the ligand is occupancy-pruned: no substrate atop at xy ~ 0.
    substrate_atops = [
        site
        for site in ctx.sites
        if site.site_type == "atop"
        and site.slab_indices
        and int(site.slab_indices[0]) < len(slab)
    ]
    # The enumerate path drops anchors within 1.5 Å in-plane. Index 0 is at the origin.
    used = {spec.site_index for spec in specs}
    origin_atops = [
        i
        for i, site in enumerate(ctx.sites)
        if site in substrate_atops and float(np.linalg.norm(site.xyz[:2])) < 1.0
    ]
    assert origin_atops
    assert used.isdisjoint(origin_atops)


def test_pose_beside_a_ligand_is_lifted_clear_of_it():
    top = _square_layer(4, 0.0)
    slab = _slab_from_positions(top)
    ligand_pos = np.array([1.3, 0.0, 2.3], dtype=float)
    covered = slab + Atoms("O", positions=[ligand_pos])
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx = site_context_for_sampling(slab, config)
    origin = next(
        i
        for i, site in enumerate(ctx.sites)
        if site.site_type == "atop" and float(np.linalg.norm(site.xyz[:2])) < 0.2
    )
    mol = make_water()
    o_idx = list(mol.get_chemical_symbols()).index("O")

    def _height(host: Atoms, *, with_ligand: bool) -> float:
        spec = PlacementSpec(
            conformer_index=0,
            orientation_type="EN-down",
            face_flip=False,
            en_atom_index=o_idx,
            site_index=origin,
            site_type="atop",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=0.5,
            placement_index=0,
        )
        posed, fail = _pose_from_spec(
            mol,
            spec,
            host,
            config,
            "O",
            site_context=ctx,
            slab_for_sites=slab if with_ligand else None,
        )
        assert fail is None and posed is not None
        n_hat = np.asarray(posed.normal, dtype=float)
        com = np.array([posed.pose.x_abs, posed.pose.y_abs, posed.pose.z_abs])
        return float(np.dot(com, n_hat))

    assert _height(covered, with_ligand=True) > _height(slab, with_ligand=False) + 0.2


def test_flat_ring_parallel_heights_are_evenly_spaced():
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    ring = Atoms(
        "C6",
        positions=[[1.4 * np.cos(a), 1.4 * np.sin(a), 0.0] for a in angles],
    )
    slab = make_slab()
    config = AdsorptionConfig(material_type="slab", seed=0)
    ctx = site_context_for_sampling(slab, config)
    site_index = next(
        i for i, site in enumerate(ctx.sites) if site.site_type == "hollow"
    )
    heights = []
    for zf in (0.5, 0.7, 0.9):
        spec = PlacementSpec(
            conformer_index=0,
            orientation_type="parallel",
            face_flip=False,
            en_atom_index=None,
            site_index=site_index,
            site_type="hollow",
            tilt_deg=0.0,
            azimuth_deg=0.0,
            azimuth_in_plane_deg=0.0,
            z_fraction=zf,
            placement_index=0,
        )
        posed, fail = _pose_from_spec(
            ring, spec, slab, config, "c1ccccc1", site_context=ctx
        )
        assert fail is None and posed is not None
        n_hat = np.asarray(posed.normal, dtype=float)
        com = np.array([posed.pose.x_abs, posed.pose.y_abs, posed.pose.z_abs])
        heights.append(float(np.dot(com, n_hat)))
    step = heights[1] - heights[0]
    assert step == pytest.approx(heights[2] - heights[1], abs=1e-6)
    assert step > 0.3


def test_nanoparticle_dent_is_meshed_and_a_buried_centre_is_not():
    angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    rim = [[6.0 * np.cos(a), 6.0 * np.sin(a), 1.0] for a in angles]
    positions = rim + [[0.0, 0.0, -1.2], [0.0, 0.0, 0.85]]
    dent = len(positions) - 1
    cone = Atoms(
        ["Cu"] * len(positions),
        positions=positions,
        cell=[30.0, 30.0, 30.0],
        pbc=False,
    )
    sites = get_unified_sites(
        cone, material_type="nanoparticle", site_generator="topology"
    )
    assert any(site.site_type == "atop" and dent in site.slab_indices for site in sites)
    cluster = Icosahedron("Au", 2)
    cluster_sites = get_unified_sites(
        cluster, material_type="nanoparticle", site_generator="topology"
    )
    assert 0 not in _supports(cluster_sites)

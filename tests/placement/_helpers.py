"""Shared helpers and golden constants for the placement test package."""

from dataclasses import replace

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import (
    PlacementDescriptor,
    PlacementPose,
    PlacementSpec,
)
from metalsurfer.placement import (
    enumerate_placement_specs,
    generate_placement_from_pose,
    generate_placement_from_spec,
)
from metalsurfer.placement.site_types import Site

from ..conftest import (
    make_nanoparticle,
    make_porous_framework,
    make_slab,
)

TEST_SEED = 0

_GOLDEN_SLAB_UNIFIED_SITE_COUNT = 126

_GOLDEN_SLAB_SITE_TYPE_MULTISET = {"atop": 16, "bridge": 78, "hollow": 32}

_LOCAL_SITE_MATERIAL_PARAMS = [
    ("nanoparticle", make_nanoparticle, 20, (1.5, 2.5), 20),
    ("porous", make_porous_framework, 12, (1.5, 3.0), 12),
]


def _first_successful_placement(conformers, slab, config, smiles, n_desired=20):
    specs = enumerate_placement_specs(
        conformers,
        slab,
        config,
        smiles,
        n_desired=n_desired,
    )
    for spec in specs:
        result = generate_placement_from_spec(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
        )
        if result is not None:
            return spec, result
    return None, None


def _generate_placements(conformers, slab, config, smiles=None, n_desired=30):
    specs = enumerate_placement_specs(conformers, slab, config, smiles, n_desired)
    results = []
    for spec in specs:
        result = generate_placement_from_spec(
            spec,
            conformers,
            slab,
            config,
            smiles=smiles,
        )
        if result is not None:
            results.append((spec, result[0], result[1]))
    return results


def _pose_from_descriptor(descriptor: PlacementDescriptor) -> PlacementPose:
    return PlacementPose(
        conformer_index=descriptor.conformer_index,
        site_index=descriptor.site_index,
        site_type=descriptor.site_type,
        placement_index=descriptor.placement_index,
        quat_w=float(descriptor.quat_w),
        quat_x=float(descriptor.quat_x),
        quat_y=float(descriptor.quat_y),
        quat_z=float(descriptor.quat_z),
        x_abs=float(descriptor.x_abs),
        y_abs=float(descriptor.y_abs),
        z_fraction=float(descriptor.z_fraction),
        z_abs=float(descriptor.z_abs),
        orientation_type=descriptor.orientation_type,
        face_flip=descriptor.face_flip,
        en_atom_index=descriptor.en_atom_index,
        tilt_deg=descriptor.tilt_deg,
        azimuth_deg=descriptor.azimuth_deg,
        azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
    )


def _assert_replay_matches(
    mode: str,
    adsorbate: Atoms,
    descriptor: PlacementDescriptor,
    spec: PlacementSpec,
    conformers: list[Atoms],
    slab: Atoms,
    config: AdsorptionConfig,
) -> None:
    if mode == "spec":
        replay = generate_placement_from_spec(
            spec, conformers, slab, config, smiles="O"
        )
        assert replay is not None
        replayed = replay[0]
    else:
        pose = _pose_from_descriptor(descriptor)
        replay = generate_placement_from_pose(pose, conformers, slab, config)
        assert replay is not None
        replayed = replay[0]
    np.testing.assert_allclose(
        adsorbate.get_positions(), replayed.get_positions(), atol=1e-10
    )


def _site_type_atop(idx: int) -> str:
    return "atop"


def _site_ordering_key(site: Site) -> tuple:
    xyz = np.asarray(site.xyz, dtype=float)
    return (
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        str(site.site_type),
        str(site.site_source),
    )


def _make_site(
    xyz,
    site_type="hollow",
    source="topology_hollow",
    *,
    normal=None,
    slab_indices=(0,),
    material_type="slab",
    env_fingerprint=None,
):
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
    if env_fingerprint is None:
        env_fingerprint = (("Ru",), site_type)
    return Site(
        xyz=np.asarray(xyz, dtype=float),
        normal=np.asarray(normal, dtype=float),
        site_type=site_type,
        slab_indices=tuple(slab_indices),
        material_type=material_type,
        site_source=source,
        env_fingerprint=tuple(env_fingerprint),
    )


_DISSOC_PLACEMENT_SPEC = PlacementSpec(
    conformer_index=0,
    orientation_type="dissociative",
    face_flip=False,
    en_atom_index=None,
    site_index=0,
    site_type="hollow",
    tilt_deg=0.0,
    azimuth_deg=0.0,
    azimuth_in_plane_deg=0.0,
    z_fraction=0.5,
    placement_index=0,
)


def dissoc_placement_spec(**overrides) -> PlacementSpec:
    return replace(_DISSOC_PLACEMENT_SPEC, **overrides)


def _tilted_make_slab():
    """Cu-like slab with a non-Cartesian surface normal (tilted b/c)."""
    slab = make_slab()
    cell = np.array(slab.get_cell(), dtype=float)
    tilt = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.866, -0.5],
            [0.0, 0.5, 0.866],
        ],
        dtype=float,
    )
    cell[:3] = tilt @ cell[:3]
    slab.set_cell(cell)
    pos = slab.get_positions()
    pos[:] = (tilt @ pos.T).T
    slab.set_positions(pos)
    return slab

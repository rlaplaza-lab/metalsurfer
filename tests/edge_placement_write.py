"""Regenerate edge-placement JSON. Imported only by the snapshot script.

Running ``python tests/edge_placement_snapshots.py`` loads this module. The
coverage tests import the loader and do not.
"""

from __future__ import annotations

import tempfile
from dataclasses import replace

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.models import ScreeningResult
from metalsurfer.placement.generators import _spec_grid_info
from metalsurfer.placement.site_context import resolve_site_context_for_sampling
from metalsurfer.surface_prep import prepare_substrate
from tests.edge_chemistry_helpers import (
    _COVERAGE_PLACEMENTS,
    _OLEIC_SMILES,
    _PME3_SMILES,
    _PYRIDONE_SMILES,
    au4_tetrahedron,
    cu111_for,
    materialize_pose_pool,
)
from tests.edge_placement_snapshots import (
    _PYRIDONE_PAIR_CONFIGS,
    _PYRIDONE_TRIMER_CONFIGS,
    SCHEMA_VERSION,
    _catalog_record,
    _frame_record,
    _pose_record,
    _write,
    joint_pose_pool_size,
)


def _bare_payload(
    *,
    slab: Atoms,
    poses: list[ScreeningResult],
    info,
) -> dict[str, object]:
    counts = info.occupied_neighbor_counts or {}
    return {
        "schema_version": SCHEMA_VERSION,
        "slab": _frame_record(slab),
        "catalog": _catalog_record(info),
        "anchor": None,
        "placements": [
            _pose_record(
                pose,
                int(counts.get(int(pose.placement_descriptor.site_index), 0))
                if info.occupied_neighbor_counts is not None
                else None,
            )
            for pose in poses
        ],
    }


def _cpu_config(*, num_conformers: int) -> AdsorptionConfig:
    return AdsorptionConfig(
        seed=42,
        device="cpu",
        slab_relaxation_mode="none",
        num_conformers=num_conformers,
        num_placements=_COVERAGE_PLACEMENTS,
    )


def write_all() -> None:
    """Materialize the edge pools once and store them under ``tests/data``."""
    pme3 = _cpu_config(num_conformers=1)
    cluster = prepare_substrate(
        slab=au4_tetrahedron(),
        config=pme3,
        results_dir=tempfile.mkdtemp(prefix="metalsurfer-snapshots-"),
        material_type="nanoparticle",
    )
    packed = create_conformers_from_smiles(_PME3_SMILES, config=pme3)
    assert packed is not None
    conformers, _energies = packed
    pme3_sites = resolve_site_context_for_sampling(
        cluster.atoms,
        pme3,
        symmetry_broken=False,
        material_type="nanoparticle",
    )
    for n_desired, name in ((100, "pme3_au4_100"), (8, "pme3_au4_8")):
        poses = materialize_pose_pool(
            smiles=_PME3_SMILES,
            conformers=conformers,
            slab=cluster.atoms,
            config=pme3,
            site_context=pme3_sites,
            n_desired=n_desired,
        )
        info = _spec_grid_info(
            conformers,
            cluster.atoms,
            pme3,
            _PME3_SMILES,
            pme3_sites,
            material_type="nanoparticle",
        )
        _write(name, _bare_payload(slab=cluster.atoms, poses=poses, info=info))

    slab_config = _cpu_config(num_conformers=4)
    pyridone_slab, pyridone_conformers = cu111_for(
        _PYRIDONE_SMILES,
        slab_config,
        tempfile.mkdtemp(prefix="metalsurfer-snapshots-"),
    )
    pyridone_sites = resolve_site_context_for_sampling(
        pyridone_slab, slab_config, symmetry_broken=False, material_type="slab"
    )
    poses = materialize_pose_pool(
        smiles=_PYRIDONE_SMILES,
        conformers=pyridone_conformers,
        slab=pyridone_slab,
        config=slab_config,
        site_context=pyridone_sites,
        n_desired=16,
    )
    info = _spec_grid_info(
        pyridone_conformers,
        pyridone_slab,
        slab_config,
        _PYRIDONE_SMILES,
        pyridone_sites,
        material_type="slab",
    )
    _write(
        "pyridone_pool_16",
        _bare_payload(slab=pyridone_slab, poses=poses, info=info),
    )

    seed = materialize_pose_pool(
        smiles=_PYRIDONE_SMILES,
        conformers=pyridone_conformers,
        slab=pyridone_slab,
        config=slab_config,
        site_context=pyridone_sites,
        n_desired=1,
    )
    covered = seed[0].atoms
    anchor = covered[len(pyridone_slab) :]
    for n_desired, name in ((100, "pyridone_step2_100"), (16, "pyridone_step2_16")):
        poses = materialize_pose_pool(
            smiles=_PYRIDONE_SMILES,
            conformers=pyridone_conformers,
            slab=pyridone_slab,
            config=slab_config,
            site_context=pyridone_sites,
            full_slab=covered,
            n_desired=n_desired,
        )
        info = _spec_grid_info(
            pyridone_conformers,
            pyridone_slab,
            slab_config,
            _PYRIDONE_SMILES,
            pyridone_sites,
            full_slab=covered,
            material_type="slab",
        )
        payload = _bare_payload(slab=pyridone_slab, poses=poses, info=info)
        payload["anchor"] = _frame_record(anchor)
        _write(name, payload)

    _write_pyridone_joint_pools(pyridone_slab, pyridone_conformers)

    oleic_slab, oleic_conformers = cu111_for(
        _OLEIC_SMILES,
        slab_config,
        tempfile.mkdtemp(prefix="metalsurfer-snapshots-"),
    )
    oleic_sites = resolve_site_context_for_sampling(
        oleic_slab, slab_config, symmetry_broken=False, material_type="slab"
    )
    oleic_poses = materialize_pose_pool(
        smiles=_OLEIC_SMILES,
        conformers=oleic_conformers,
        slab=oleic_slab,
        config=slab_config,
        site_context=oleic_sites,
        n_desired=100,
    )
    oleic_info = _spec_grid_info(
        oleic_conformers,
        oleic_slab,
        slab_config,
        _OLEIC_SMILES,
        oleic_sites,
        material_type="slab",
    )
    _write(
        "oleic_100",
        _bare_payload(slab=oleic_slab, poses=oleic_poses, info=oleic_info),
    )
    oleic_seed = materialize_pose_pool(
        smiles=_OLEIC_SMILES,
        conformers=oleic_conformers,
        slab=oleic_slab,
        config=slab_config,
        site_context=oleic_sites,
        n_desired=1,
    )
    oleic_covered = oleic_seed[0].atoms
    oleic_anchor = oleic_covered[len(oleic_slab) :]
    oleic_step2 = materialize_pose_pool(
        smiles=_OLEIC_SMILES,
        conformers=oleic_conformers,
        slab=oleic_slab,
        config=slab_config,
        site_context=oleic_sites,
        full_slab=oleic_covered,
        n_desired=16,
    )
    oleic_step2_info = _spec_grid_info(
        oleic_conformers,
        oleic_slab,
        slab_config,
        _OLEIC_SMILES,
        oleic_sites,
        full_slab=oleic_covered,
        material_type="slab",
    )
    step2_payload = _bare_payload(
        slab=oleic_slab, poses=oleic_step2, info=oleic_step2_info
    )
    step2_payload["anchor"] = _frame_record(oleic_anchor)
    _write("oleic_step2_16", step2_payload)


def _write_pyridone_joint_pools(slab: Atoms, conformers: list[Atoms]) -> None:
    """Full-lattice pools at the pair and trimer budgets a run would screen."""
    for n_per, n_configs, name in (
        (2, _PYRIDONE_PAIR_CONFIGS, "pyridone_pair"),
        (3, _PYRIDONE_TRIMER_CONFIGS, "pyridone_trimer"),
    ):
        pool = joint_pose_pool_size(n_per, n_configs)
        config = replace(
            _cpu_config(num_conformers=4),
            num_placements=pool,
            saturation_molecules_per_step=n_per,
        )
        sites = resolve_site_context_for_sampling(
            slab, config, symmetry_broken=False, material_type="slab"
        )
        poses = materialize_pose_pool(
            smiles=_PYRIDONE_SMILES,
            conformers=conformers,
            slab=slab,
            config=config,
            site_context=sites,
            n_desired=pool,
        )
        info = _spec_grid_info(
            conformers, slab, config, _PYRIDONE_SMILES, sites, material_type="slab"
        )
        payload = _bare_payload(slab=slab, poses=poses, info=info)
        payload["n_per_config"] = n_per
        payload["n_configs"] = n_configs
        payload["catalog_positions"] = [
            np.asarray(site.xyz, dtype=float).tolist() for site in sites.sites
        ]
        _write(name, payload)


def write_pyridone_joint_pools() -> None:
    """Materialize the pyridone pair and trimer pools without the other edges."""
    config = _cpu_config(num_conformers=4)
    slab, conformers = cu111_for(
        _PYRIDONE_SMILES,
        config,
        tempfile.mkdtemp(prefix="metalsurfer-snapshots-"),
    )
    _write_pyridone_joint_pools(slab, conformers)

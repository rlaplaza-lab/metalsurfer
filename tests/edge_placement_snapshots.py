"""Materialized edge-placement snapshots.

The edge-chemistry tests load these files and check the poses. They do not
rebuild conformers, enumerate sites, or relax anything. Regenerate from the
repository root with ``PYTHONPATH=. python tests/edge_placement_snapshots.py``.
``joint`` rewrites only the pyridone pair and trimer pools.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.models import PlacementDescriptor, ScreeningResult
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

SNAPSHOT_DIR = Path(__file__).resolve().parent / "data" / "edge_placements"

# Joint-config counts a homogeneous run screens. The pose pool is
# n * configs * placement_retry_oversample_max. 25 pair configs (100 poses)
# and 33 trimer configs (198 poses) are the budgets whose assembled groups
# contain an NH···O dimer and a connected trimer.
_PYRIDONE_PAIR_CONFIGS = 25
_PYRIDONE_TRIMER_CONFIGS = 33

_DESCRIPTOR_TUPLES = ("slab_indices",)


def _jsonable(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _positions(atoms: Atoms) -> list[list[float]]:
    return np.round(np.asarray(atoms.get_positions(), dtype=float), 6).tolist()


def _pose_record(
    pose: ScreeningResult, occupancy_count: int | None
) -> dict[str, object]:
    adsorbate = pose.atoms[pose.slab_size :]
    return {
        "placement_id": int(pose.placement_id),
        "occupancy_count": occupancy_count,
        "symbols": list(adsorbate.get_chemical_symbols()),
        "positions": _positions(adsorbate),
        "descriptor": _jsonable(asdict(pose.placement_descriptor)),
    }


def _frame_record(atoms: Atoms) -> dict[str, object]:
    return {
        "symbols": list(atoms.get_chemical_symbols()),
        "positions": _positions(atoms),
        "cell": np.round(np.asarray(atoms.get_cell(), dtype=float), 6).tolist(),
        "pbc": [bool(flag) for flag in atoms.get_pbc()],
    }


def _catalog_record(info) -> dict[str, object]:
    counts = info.occupied_neighbor_counts or {}
    site_types = sorted(
        {str(info.unique_sites[int(index)].site_type) for index in info.site_indices}
    )
    return {
        "site_types": site_types,
        "positive_sites": [
            int(index)
            for index in info.site_indices
            if int(counts.get(int(index), 0)) >= 1
        ],
        "open_atops": [
            int(index)
            for index in info.site_indices
            if info.unique_sites[int(index)].site_type == "atop"
        ],
    }


def _write(name: str, payload: dict[str, object]) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def joint_pose_pool_size(n_per_config: int, n_configs: int) -> int:
    """Pose pool a one-species n-tuple run materializes for this joint budget."""
    oversample = float(AdsorptionConfig().placement_retry_oversample_max)
    slots = int(n_per_config) * int(n_configs)
    return max(1, int(math.ceil(slots * oversample)))


def load_snapshot(name: str) -> dict[str, object]:
    """Load one materialized-placement snapshot."""
    path = SNAPSHOT_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def frame_atoms(record: dict[str, object]) -> Atoms:
    """Rebuild a slab or adsorbate stored in a snapshot."""
    return Atoms(
        symbols=list(record["symbols"]),
        positions=record["positions"],
        cell=record["cell"],
        pbc=record["pbc"],
    )


def pose_atoms(snapshot: dict[str, object], pose: dict[str, object]) -> Atoms:
    """Adsorbate geometry for one stored placement."""
    slab = snapshot["slab"]
    assert isinstance(slab, dict)
    return Atoms(
        symbols=list(pose["symbols"]),
        positions=pose["positions"],
        cell=slab["cell"],
        pbc=slab["pbc"],
    )


def screening_result(
    snapshot: dict[str, object], pose: dict[str, object]
) -> ScreeningResult:
    """Screening stub whose adsorbate is the stored materialized placement."""
    slab_record = snapshot["slab"]
    assert isinstance(slab_record, dict)
    slab = frame_atoms(slab_record)
    adsorbate = pose_atoms(snapshot, pose)
    combined = slab + adsorbate
    descriptor = dict(pose["descriptor"])
    for key in _DESCRIPTOR_TUPLES:
        if descriptor.get(key) is not None:
            descriptor[key] = tuple(descriptor[key])
    fragment = descriptor.get("fragment_positions")
    if fragment is not None:
        descriptor["fragment_positions"] = tuple(tuple(point) for point in fragment)
    return ScreeningResult(
        molecule="adsorbate",
        placement_id=int(pose["placement_id"]),
        energy_adslab=0.0,
        energy_slab=0.0,
        energy_adsorbate=0.0,
        energy_adsorption=0.0,
        atoms=combined,
        slab_size=len(slab),
        distance=0.0,
        placement_descriptor=PlacementDescriptor(**descriptor),
    )


def _bare_payload(
    *,
    slab: Atoms,
    poses: list[ScreeningResult],
    info,
) -> dict[str, object]:
    counts = info.occupied_neighbor_counts or {}
    return {
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


def _cpu_config(*, material_type: str, num_conformers: int) -> AdsorptionConfig:
    return AdsorptionConfig(
        material_type=material_type,
        seed=42,
        device="cpu",
        slab_relaxation_mode="none",
        num_conformers=num_conformers,
        num_placements=_COVERAGE_PLACEMENTS,
    )


def write_all() -> None:
    """Materialize the edge pools once and store them under ``tests/data``."""
    pme3 = _cpu_config(material_type="nanoparticle", num_conformers=1)
    cluster = prepare_substrate(
        slab=au4_tetrahedron(),
        config=pme3,
        results_dir="results_snapshot_au4_pme3",
    )
    packed = create_conformers_from_smiles(_PME3_SMILES, config=pme3)
    assert packed is not None
    conformers, _energies = packed
    pme3_sites = resolve_site_context_for_sampling(
        cluster.atoms, pme3, symmetry_broken=False
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
            conformers, cluster.atoms, pme3, _PME3_SMILES, pme3_sites
        )
        _write(name, _bare_payload(slab=cluster.atoms, poses=poses, info=info))

    slab_config = _cpu_config(material_type="slab", num_conformers=4)
    pyridone_slab, pyridone_conformers = cu111_for(
        _PYRIDONE_SMILES, slab_config, "results_snapshot_pyridone"
    )
    pyridone_sites = resolve_site_context_for_sampling(
        pyridone_slab, slab_config, symmetry_broken=False
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
        )
        payload = _bare_payload(slab=pyridone_slab, poses=poses, info=info)
        payload["anchor"] = _frame_record(anchor)
        _write(name, payload)

    _write_pyridone_joint_pools(pyridone_slab, pyridone_conformers)

    oleic_slab, oleic_conformers = cu111_for(
        _OLEIC_SMILES, slab_config, "results_snapshot_oleic"
    )
    oleic_sites = resolve_site_context_for_sampling(
        oleic_slab, slab_config, symmetry_broken=False
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
        oleic_conformers, oleic_slab, slab_config, _OLEIC_SMILES, oleic_sites
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
            _cpu_config(material_type="slab", num_conformers=4),
            num_placements=pool,
            saturation_molecules_per_step=n_per,
        )
        sites = resolve_site_context_for_sampling(slab, config, symmetry_broken=False)
        poses = materialize_pose_pool(
            smiles=_PYRIDONE_SMILES,
            conformers=conformers,
            slab=slab,
            config=config,
            site_context=sites,
            n_desired=pool,
        )
        info = _spec_grid_info(conformers, slab, config, _PYRIDONE_SMILES, sites)
        payload = _bare_payload(slab=slab, poses=poses, info=info)
        payload["n_per_config"] = n_per
        payload["n_configs"] = n_configs
        payload["catalog_positions"] = [
            np.asarray(site.xyz, dtype=float).tolist() for site in sites.sites
        ]
        _write(name, payload)


def write_pyridone_joint_pools() -> None:
    """Materialize the pyridone pair and trimer pools without the other edges."""
    config = _cpu_config(material_type="slab", num_conformers=4)
    slab, conformers = cu111_for(_PYRIDONE_SMILES, config, "results_snapshot_pyridone")
    _write_pyridone_joint_pools(slab, conformers)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "joint":
        write_pyridone_joint_pools()
    else:
        write_all()

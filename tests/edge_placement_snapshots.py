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
from dataclasses import asdict
from pathlib import Path

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import PlacementDescriptor, ScreeningResult

SNAPSHOT_DIR = Path(__file__).resolve().parent / "data" / "edge_placements"
SCHEMA_VERSION = 1

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
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = int(payload.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"{path.name} schema_version {version} != {SCHEMA_VERSION}")
    return payload


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


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tests.edge_placement_write import write_all, write_pyridone_joint_pools

    if len(sys.argv) > 1 and sys.argv[1] == "joint":
        write_pyridone_joint_pools()
    else:
        write_all()

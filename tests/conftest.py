"""Shared fixtures for metalsurfer test suites."""

# Use 'spawn' for multiprocessing so CUDA can be used in tests. The default
# 'fork' causes "Cannot re-initialize CUDA in forked subprocess" when FairChem
# or other GPU code runs after a fork (e.g. with pytest-xdist -n).
import contextlib
import multiprocessing

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.models import PlacementDescriptor


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item):
    """Run tests with no_fork marker in main process when --forked is used.

    pytest-forked uses os.fork(), which breaks CUDA. Tests marked no_fork
    (e.g. GPU tests) run in the main process instead.
    """
    if item.config.getoption("forked", default=False) and item.get_closest_marker(
        "no_fork"
    ):
        from _pytest.runner import runtestprotocol

        return runtestprotocol(item)
    return None


# ---------------------------------------------------------------------------
# slab builders
# ---------------------------------------------------------------------------


def make_slab(
    nx: int = 4,
    ny: int = 4,
    n_layers: int = 3,
    spacing: float = 2.7,
    symbol: str = "Ru",
) -> Atoms:
    """Build a simple FCC(111)-like slab."""
    a = 2.7
    positions = []
    for lz in range(n_layers):
        for ix in range(nx):
            for iy in range(ny):
                x = ix * a + (lz % 2) * a / 2
                y = iy * a + (lz % 2) * a / 2
                z = lz * spacing
                positions.append([x, y, z])
    atoms = Atoms(
        symbols=[symbol] * len(positions),
        positions=positions,
        cell=[nx * a, ny * a, n_layers * spacing + 15.0],
        pbc=[True, True, True],
    )
    return atoms


# ---------------------------------------------------------------------------
# molecule builders
# ---------------------------------------------------------------------------


def make_water() -> Atoms:
    return Atoms(
        "OH2",
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.24],
            [-0.24, 0.93, 0.24],
        ],
    )


def make_ethanol() -> Atoms:
    return Atoms(
        "C2H6O",
        positions=[
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-0.5, 0.89, 0.0],
            [-0.5, -0.89, 0.0],
            [0.0, 0.0, -1.09],
            [2.04, 0.89, 0.0],
            [2.04, -0.89, 0.0],
            [1.54, 0.0, 1.09],
            [2.54, 0.0, -0.5],
        ],
    )


# ---------------------------------------------------------------------------
# placement descriptor helper
# ---------------------------------------------------------------------------


def make_placement_descriptor(
    placement_id: int = 0,
    conformer_index: int = 0,
    orientation_type: str = "round",
    site_type: str | None = "atop",
    x: float = 0.0,
    y: float = 0.0,
    z: float = 2.5,
    shape: str = "round",
    **kwargs: object,
) -> PlacementDescriptor:
    """Minimal PlacementDescriptor for tests. Override any field via kwargs."""
    defaults: dict[str, object] = {
        "conformer_index": conformer_index,
        "orientation_type": orientation_type,
        "face_flip": False,
        "en_atom_index": None,
        "site_index": placement_id if site_type else -1,
        "site_type": site_type,
        "tilt_deg": 0.0,
        "azimuth_deg": 0.0,
        "azimuth_in_plane_deg": 0.0,
        "z_fraction": 0.5,
        "placement_index": placement_id,
        "x": x,
        "y": y,
        "z": z,
        "shape": shape,
        "slab_indices": None,
    }
    defaults.update(kwargs)
    return PlacementDescriptor(**defaults)


# ---------------------------------------------------------------------------
# placement helpers
# ---------------------------------------------------------------------------


def place_molecule_on_slab(
    slab: Atoms,
    mol: Atoms,
    z_offset: float = 3.0,
    x_shift: float = 5.0,
    y_shift: float = 5.0,
) -> Atoms:
    """Place *mol* above *slab* and return combined system."""
    slab_z = max(slab.get_positions()[:, 2])
    mol = mol.copy()
    pos = mol.get_positions().copy()
    pos -= np.mean(pos, axis=0)
    pos[:, 0] += x_shift
    pos[:, 1] += y_shift
    pos[:, 2] += slab_z + z_offset
    mol.set_positions(pos)
    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    return combined

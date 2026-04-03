"""Shared fixtures for metalsurfer test suites."""

# Use 'spawn' for multiprocessing so CUDA can be used in tests. The default
# 'fork' causes "Cannot re-initialize CUDA in forked subprocess" when FairChem
# or other GPU code runs after a fork (e.g. with pytest-xdist -n).
import contextlib
import multiprocessing
from pathlib import Path

with contextlib.suppress(RuntimeError):
    multiprocessing.set_start_method("spawn", force=True)

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.models import PlacementDescriptor, ScreeningResult


def _clear_cuda_for_gpu_test() -> None:
    """Evict TorchSim autobatchers and release CUDA allocations between GPU tests."""
    try:
        from metalsurfer.optimization import clear_autobatcher_cache

        clear_autobatcher_cache()
    except (ImportError, RuntimeError):
        pass


@pytest.fixture(autouse=True)
def _release_cuda_after_gpu_test(request):
    """Free GPU memory before and after @pytest.mark.gpu tests to reduce OOM skips."""
    if request.node.get_closest_marker("gpu"):
        _clear_cuda_for_gpu_test()
    yield
    if request.node.get_closest_marker("gpu"):
        _clear_cuda_for_gpu_test()


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


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run a test in an isolated temporary working directory."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


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
    z_offset: float = 2.5,
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
        "z_offset": z_offset,
        "shape": shape,
        "slab_indices": None,
        "quat_w": None,
        "quat_x": None,
        "quat_y": None,
        "quat_z": None,
    }
    defaults.update(kwargs)
    return PlacementDescriptor(**defaults)


def make_screening_result(
    molecule: str = "water",
    placement_id: int = 0,
    energy_adsorption: float = -1.0,
    energy_slab: float = -200.0,
    energy_adsorbate: float = -10.0,
    atoms: Atoms | None = None,
    slab_size: int | None = None,
    distance: float = 2.5,
    placement_descriptor: PlacementDescriptor | None = None,
    **kwargs: object,
) -> ScreeningResult:
    """Build a compact ScreeningResult with sensible test defaults."""
    slab = make_slab() if (atoms is None or slab_size is None) else None
    result_atoms = (
        atoms if atoms is not None else place_molecule_on_slab(slab, make_water())
    )
    result_slab_size = slab_size if slab_size is not None else len(slab)
    descriptor = (
        placement_descriptor
        if placement_descriptor is not None
        else make_placement_descriptor(placement_id=placement_id)
    )
    defaults: dict[str, object] = {
        "molecule": molecule,
        "placement_id": placement_id,
        "energy_adslab": energy_slab + energy_adsorbate + energy_adsorption,
        "energy_slab": energy_slab,
        "energy_adsorbate": energy_adsorbate,
        "energy_adsorption": energy_adsorption,
        "atoms": result_atoms,
        "slab_size": result_slab_size,
        "distance": distance,
        "placement_descriptor": descriptor,
    }
    defaults.update(kwargs)
    return ScreeningResult(**defaults)


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


def assert_lines_contain(text: str, expected_lines: list[str]) -> None:
    """Assert that every expected line appears in *text*.

    Keeps output-format tests focused on semantic lines and avoids brittle
    punctuation/order coupling for unrelated lines.
    """
    lines = set(text.splitlines())
    missing = [line for line in expected_lines if line not in lines]
    assert not missing, f"Missing lines: {missing}\n---\n{text}"


# ---------------------------------------------------------------------------
# nanoparticle / porous material builders
# ---------------------------------------------------------------------------


def make_nanoparticle() -> Atoms:
    """Au₁₃ icosahedral cluster (no PBC)."""
    # Central atom + 12 vertices of an icosahedron at bond length ~2.88 Å
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    d = 2.88 / np.sqrt(1 + phi**2)  # scale so nn distance ≈ 2.88 Å
    verts = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            verts.append([0, s1 * d, s2 * phi * d])
            verts.append([s1 * d, s2 * phi * d, 0])
            verts.append([s2 * phi * d, 0, s1 * d])
    positions = [[0.0, 0.0, 0.0]] + verts
    atoms = Atoms("Au13", positions=positions)
    atoms.set_cell([25, 25, 25])
    atoms.set_pbc([False, False, False])
    return atoms


def make_porous_framework() -> Atoms:
    """Load SiO₂ porous framework from bundled CIF (mp-1195265, Pnma)."""
    from ase.io import read

    cif_path = Path(__file__).parent / "test_files" / "SiO2.cif"
    atoms = read(str(cif_path))
    return atoms

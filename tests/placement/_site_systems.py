"""Seeded regular / disordered systems for site-catalog physics checks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.build import fcc111
from ase.data import atomic_numbers, covalent_radii

from metalsurfer._geom_pbc import minimum_image_cartesian_delta
from metalsurfer.site_plugin_ids import PLUGIN_ALLOWED_MATERIALS

from ..conftest import make_nanoparticle, make_porous_framework

Order = Literal["regular", "disordered"]
Material = Literal["slab", "nanoparticle", "porous"]

# Square Si–O channel: O at 3.2 Å from axis so axis NN exceeds pore threshold.
_CHANNEL_A = 12.0
_CHANNEL_C = 8.0
_SIO_BOND = 1.62
_CHANNEL_AXIS_XY = (_CHANNEL_A / 2.0, _CHANNEL_A / 2.0)
_CHANNEL_R_O = 3.2
_CHANNEL_R_SI = _CHANNEL_R_O + _SIO_BOND
_CHANNEL_FREE_RADIUS = _CHANNEL_R_O - float(covalent_radii[atomic_numbers["O"]])


@dataclass(frozen=True)
class SiteSystem:
    name: str
    material_type: Material
    order: Order
    factory: Callable[[], Atoms]
    channel_axis_xy: tuple[float, float] | None = None
    free_radius: float | None = None
    slow: bool = False


def _push_apart_close_pairs(atoms: Atoms, *, min_scale: float = 0.85) -> Atoms:
    """Nudge atoms so every MIC pair is at least *min_scale* × covalent sum."""
    out = atoms.copy()
    pos = np.asarray(out.get_positions(), dtype=float)
    cell = np.asarray(out.get_cell(), dtype=float)
    pbc = np.asarray(out.get_pbc(), dtype=bool)
    radii = np.asarray(
        [float(covalent_radii[atomic_numbers[s]]) for s in out.get_chemical_symbols()],
        dtype=float,
    )
    for _ in range(20):
        moved = False
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                dvec = minimum_image_cartesian_delta(pos[j] - pos[i], cell, pbc)
                dist = float(np.linalg.norm(dvec))
                min_d = min_scale * (radii[i] + radii[j])
                if 1e-8 < dist < min_d:
                    push = 0.5 * (min_d - dist) * dvec / dist
                    pos[i] -= push
                    pos[j] += push
                    moved = True
        if not moved:
            break
    out.set_positions(pos)
    return out


def make_regular_slab() -> Atoms:
    slab = fcc111("Pt", size=(3, 3, 3), vacuum=10.0, periodic=True)
    slab.set_pbc([True, True, False])
    return slab


def make_disordered_slab() -> Atoms:
    slab = fcc111("Pt", size=(4, 4, 3), vacuum=10.0, periodic=True)
    slab.set_pbc([True, True, False])
    pos = np.asarray(slab.get_positions(), dtype=float)
    cell = np.asarray(slab.get_cell(), dtype=float)
    top = pos[:, 2] > float(np.max(pos[:, 2])) - 0.8
    frac_a = (pos @ np.linalg.inv(cell).T)[:, 0]
    slab = slab[~(top & (frac_a > 0.5))]
    slab.set_pbc([True, True, False])
    slab.rattle(0.15, seed=1)
    return slab


def make_disordered_nanoparticle() -> Atoms:
    """Vacancy-defected Au13 icosahedron (rattled surface, one vertex removed)."""
    atoms = make_nanoparticle()
    keep = [0] + list(range(2, len(atoms)))
    atoms = atoms[keep]
    atoms.rattle(0.12, seed=3)
    atoms.center()
    return atoms


def make_sio_channel() -> Atoms:
    """Periodic square Si–O channel with tetrahedral Si–O bond length."""
    a = float(_CHANNEL_A)
    c = float(_CHANNEL_C)
    cx, cy = a / 2.0, a / 2.0
    r_si = float(_CHANNEL_R_SI)
    r_o = float(_CHANNEL_R_O)
    symbols: list[str] = []
    positions: list[list[float]] = []
    directions = (
        (1.0, 0.0),
        (0.0, 1.0),
        (-1.0, 0.0),
        (0.0, -1.0),
    )
    for z in (c * 0.25, c * 0.75):
        for dx, dy in directions:
            symbols.append("Si")
            positions.append([cx + r_si * dx, cy + r_si * dy, z])
            symbols.append("O")
            positions.append([cx + r_o * dx, cy + r_o * dy, z])
    return Atoms(symbols=symbols, positions=positions, cell=[a, a, c], pbc=True)


def make_disordered_sio_channel() -> Atoms:
    atoms = make_sio_channel()
    atoms.rattle(stdev=0.12, seed=2)
    atoms = _push_apart_close_pairs(atoms, min_scale=0.80)
    axis = np.asarray(_CHANNEL_AXIS_XY, dtype=float)
    xy = np.asarray(atoms.get_positions(), dtype=float)[:, :2]
    assert float(np.min(np.linalg.norm(xy - axis, axis=1))) >= 1.2
    return atoms


SITE_SYSTEMS: tuple[SiteSystem, ...] = (
    SiteSystem("slab_regular", "slab", "regular", make_regular_slab),
    SiteSystem("slab_disordered", "slab", "disordered", make_disordered_slab),
    SiteSystem("np_regular", "nanoparticle", "regular", make_nanoparticle),
    SiteSystem(
        "np_disordered", "nanoparticle", "disordered", make_disordered_nanoparticle
    ),
    SiteSystem(
        "porous_channel",
        "porous",
        "regular",
        make_sio_channel,
        channel_axis_xy=_CHANNEL_AXIS_XY,
        free_radius=_CHANNEL_FREE_RADIUS,
    ),
    SiteSystem(
        "porous_channel_disordered",
        "porous",
        "disordered",
        make_disordered_sio_channel,
        channel_axis_xy=_CHANNEL_AXIS_XY,
        free_radius=_CHANNEL_FREE_RADIUS,
    ),
    SiteSystem("porous_sio2", "porous", "regular", make_porous_framework, slow=True),
)


def catalog_physics_cases() -> list[tuple[SiteSystem, str]]:
    return [
        (system, plugin)
        for system in SITE_SYSTEMS
        for plugin in ("auto",)
        + tuple(
            sorted(
                p
                for p, mats in PLUGIN_ALLOWED_MATERIALS.items()
                if system.material_type in mats
            )
        )
    ]

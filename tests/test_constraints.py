"""Tests for top-layer identification and frozen-index computation."""

from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import (
    compute_frozen_indices,
    identify_top_layer_indices,
)
from metalsurfer.workflow.shared import _resolve_base_slab_for_frozen


def _make_slab(n_layers: int = 4, atoms_per_layer: int = 4, spacing: float = 2.0):
    """Build a synthetic slab with *n_layers* layers along z."""
    positions = []
    symbols = []
    for layer in range(n_layers):
        for i in range(atoms_per_layer):
            x = float(i % 2) * 1.5
            y = float(i // 2) * 1.5
            z = layer * spacing
            positions.append([x, y, z])
            symbols.append("Ru")
    atoms = Atoms(symbols=symbols, positions=positions)
    atoms.set_cell([5.0, 5.0, n_layers * spacing + 10.0])
    atoms.set_pbc([True, True, True])
    return atoms


def test_identify_top_layer():
    slab = _make_slab(n_layers=4, atoms_per_layer=4, spacing=2.0)
    top = identify_top_layer_indices(slab, tolerance=0.5)
    # only the 4 atoms in the topmost layer (z=6.0) should be returned
    expected_z = 6.0
    for idx in top:
        assert abs(slab.get_positions()[idx, 2] - expected_z) <= 0.5
    assert len(top) == 4


def test_frozen_indices_default_relax_top():
    slab = _make_slab(n_layers=4, atoms_per_layer=4, spacing=2.0)
    config = AdsorptionConfig(relax_top_layer=True, top_layer_tolerance=0.5)
    frozen = compute_frozen_indices(slab, config)
    # bottom 3 layers should be frozen (12 atoms), top layer free (4 atoms)
    assert len(frozen) == 12
    top_z = max(slab.get_positions()[:, 2])
    for idx in frozen:
        assert slab.get_positions()[idx, 2] < top_z - 0.5 + 1e-6


def test_frozen_indices_all_frozen():
    slab = _make_slab(n_layers=4, atoms_per_layer=4, spacing=2.0)
    config = AdsorptionConfig(relax_top_layer=False)
    frozen = compute_frozen_indices(slab, config)
    assert len(frozen) == len(slab)


def test_resolve_base_slab_for_frozen_after_auto_resize():
    """In-plane repeat must freeze the full resized substrate, not one tile."""
    slab = _make_slab(n_layers=2, atoms_per_layer=4, spacing=2.0)
    base = slab.copy()
    resized = slab.repeat((2, 2, 1))
    config = AdsorptionConfig(relax_top_layer=False)

    effective = _resolve_base_slab_for_frozen(resized, base, slab_was_resized=True)
    assert effective is not None
    assert len(effective) == len(resized)
    frozen = compute_frozen_indices(effective, config)
    assert len(frozen) == len(resized)

    unchanged = _resolve_base_slab_for_frozen(slab, base, slab_was_resized=False)
    assert unchanged is base
    assert len(compute_frozen_indices(unchanged, config)) == len(base)


def test_resolve_base_slab_for_frozen_relax_top_layer_after_resize():
    """Top-layer freeze must span every repeated tile after in-plane resize."""
    slab = _make_slab(n_layers=4, atoms_per_layer=4, spacing=2.0)
    base = slab.copy()
    resized = slab.repeat((2, 2, 1))
    config = AdsorptionConfig(relax_top_layer=True, top_layer_tolerance=0.5)

    effective = _resolve_base_slab_for_frozen(resized, base, slab_was_resized=True)
    assert effective is not None
    assert len(effective) == len(resized)
    frozen = compute_frozen_indices(effective, config)
    # 4 tiles x 12 frozen atoms per tile (3 of 4 layers)
    assert len(frozen) == 48


def test_frozen_indices_by_symbol():
    slab = _make_slab(n_layers=2, atoms_per_layer=4, spacing=2.0)
    syms = slab.get_chemical_symbols()
    # mark half the atoms as Cu
    for i in range(0, len(syms), 2):
        syms[i] = "Cu"
    slab.set_chemical_symbols(syms)

    config = AdsorptionConfig(freeze_symbols=["Ru"])
    frozen = compute_frozen_indices(slab, config)
    for idx in frozen:
        assert slab.get_chemical_symbols()[idx] == "Ru"
    # Cu atoms should not be frozen
    cu_indices = [i for i, s in enumerate(slab.get_chemical_symbols()) if s == "Cu"]
    assert all(ci not in frozen for ci in cu_indices)

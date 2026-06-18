"""Tests for top-layer identification and frozen-index computation."""

from ase.constraints import FixAtoms

from metalsurfer.optimization import (
    compute_frozen_indices,
    format_atom_index_ranges,
    frozen_indices_from_constraints,
    identify_top_layer_indices,
    log_substrate_freeze_policy,
)
from metalsurfer.surface_prep import apply_surface_constraints

from .conftest import make_slab


def test_identify_top_layer():
    slab = make_slab(nx=2, ny=2, n_layers=4, spacing=2.0)
    top = identify_top_layer_indices(slab, tolerance=0.5)
    expected_z = 6.0
    for idx in top:
        assert abs(slab.get_positions()[idx, 2] - expected_z) <= 0.5
    assert len(top) == 4


def test_frozen_indices_default_relax_top():
    slab = make_slab(nx=2, ny=2, n_layers=4, spacing=2.0)
    frozen = compute_frozen_indices(slab, relax_top_layer=True, top_layer_tolerance=0.5)
    assert len(frozen) == 12
    top_z = max(slab.get_positions()[:, 2])
    for idx in frozen:
        assert slab.get_positions()[idx, 2] < top_z - 0.5 + 1e-6


def test_frozen_indices_all_frozen():
    slab = make_slab(nx=2, ny=2, n_layers=4, spacing=2.0)
    frozen = compute_frozen_indices(slab, relax_top_layer=False)
    assert len(frozen) == len(slab)


def test_frozen_indices_by_symbol():
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    syms = slab.get_chemical_symbols()
    for i in range(0, len(syms), 2):
        syms[i] = "Cu"
    slab.set_chemical_symbols(syms)

    frozen = compute_frozen_indices(slab, freeze_symbols=["Ru"])
    for idx in frozen:
        assert slab.get_chemical_symbols()[idx] == "Ru"
    cu_indices = [i for i, s in enumerate(slab.get_chemical_symbols()) if s == "Cu"]
    assert all(ci not in frozen for ci in cu_indices)


def test_apply_surface_constraints_round_trip():
    slab = make_slab(nx=2, ny=2, n_layers=4, spacing=2.0)
    slab.set_constraint()
    constrained = apply_surface_constraints(
        slab, relax_top_layer=True, top_layer_tolerance=0.5
    )
    assert frozen_indices_from_constraints(constrained) == compute_frozen_indices(
        slab, relax_top_layer=True, top_layer_tolerance=0.5
    )


def test_frozen_indices_from_constraints_reads_fixatoms():
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    slab.set_constraint(FixAtoms(indices=[0, 1, 2]))
    assert frozen_indices_from_constraints(slab) == [0, 1, 2]


def test_format_atom_index_ranges():
    assert format_atom_index_ranges([]) == "(none)"
    assert format_atom_index_ranges([0, 1, 2, 5, 7, 8]) == "0-2, 5, 7-8"


def test_log_substrate_freeze_policy_all_frozen(caplog):
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    with caplog.at_level("INFO"):
        log_substrate_freeze_policy(slab, context="Test slab")
    assert "all 8 substrate atoms frozen" in caplog.text
    assert "Ru×8" in caplog.text


def test_log_substrate_freeze_policy_partial_freeze(caplog):
    slab = make_slab(nx=2, ny=2, n_layers=4, spacing=2.0)
    slab.set_constraint()
    slab = apply_surface_constraints(
        slab, relax_top_layer=True, top_layer_tolerance=0.5
    )
    with caplog.at_level("INFO"):
        log_substrate_freeze_policy(slab, context="Test slab")
    assert "12/16 substrate atoms frozen" in caplog.text
    assert "4 free to move" in caplog.text
    assert "moving" in caplog.text

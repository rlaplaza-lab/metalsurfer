"""Tests for top-layer identification and frozen-index computation."""

import logging

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from metalsurfer.surface_prep import (
    apply_surface_constraints,
    check_frozen_substrate_displacement,
    compute_frozen_indices,
    format_atom_index_ranges,
    frozen_indices_from_constraints,
    identify_relaxable_surface_indices,
    identify_top_layer_indices,
    log_substrate_freeze_policy,
    max_frozen_substrate_displacement,
)

from .conftest import make_nanoparticle, make_porous_framework, make_slab


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


def test_relaxable_surface_nanoparticle_outer_shell():
    nanoparticle = make_nanoparticle()
    free = identify_relaxable_surface_indices(
        nanoparticle,
        material_type="nanoparticle",
        tolerance=0.5,
    )
    positions = nanoparticle.get_positions()
    com = positions.mean(axis=0)
    dists = np.linalg.norm(positions - com, axis=1)
    r_max = float(np.max(dists))
    assert free
    for idx in free:
        assert dists[idx] >= r_max - 0.5 - 1e-6
    frozen = compute_frozen_indices(
        nanoparticle,
        relax_top_layer=True,
        material_type="nanoparticle",
        top_layer_tolerance=0.5,
    )
    assert 0 in frozen  # central atom stays frozen on Au13


def test_relaxable_surface_porous_pore_boundary():
    porous = make_porous_framework()
    free = identify_relaxable_surface_indices(
        porous,
        material_type="porous",
        tolerance=0.5,
    )
    assert free
    assert len(free) < len(porous)
    frozen = compute_frozen_indices(
        porous,
        relax_top_layer=True,
        material_type="porous",
        top_layer_tolerance=0.5,
    )
    assert len(frozen) + len(free) == len(porous)
    constrained = apply_surface_constraints(
        porous,
        relax_top_layer=True,
        material_type="porous",
        top_layer_tolerance=0.5,
    )
    assert frozen_indices_from_constraints(constrained) == frozen


def test_check_frozen_substrate_displacement_detects_drift():
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    slab = apply_surface_constraints(slab, relax_top_layer=False)
    combined = slab.copy()
    combined += Atoms("H", positions=[[0.0, 0.0, 8.0]])
    drifted = combined.copy()
    drifted.set_constraint()
    pos = drifted.get_positions()
    pos[0, 0] += 0.05
    drifted.set_positions(pos)
    ok, reason = check_frozen_substrate_displacement(drifted, slab, slab_size=len(slab))
    assert not ok
    assert "displaced" in reason
    assert max_frozen_substrate_displacement(drifted, slab, slab_size=len(slab)) > 0.04


def test_check_frozen_substrate_displacement_passes_when_fixed():
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    slab = apply_surface_constraints(slab, relax_top_layer=False)
    combined = slab.copy()
    combined += Atoms("H", positions=[[0.0, 0.0, 8.0]])
    ok, _ = check_frozen_substrate_displacement(combined, slab, slab_size=len(slab))
    assert ok


def test_relax_top_layer_two_layer_tolerance_freezes_bottom_half():
    """Large top_layer_tolerance must free only the top band, not the whole slab.

    Multi-layer Cu(111)-like slabs use ~2 Å interlayer spacing; a 2.1 Å tolerance
    should leave two layers free and freeze the rest (camphor demo policy).
    """
    slab = make_slab(nx=4, ny=4, n_layers=4, spacing=2.08)
    free = identify_relaxable_surface_indices(slab, material_type="slab", tolerance=2.1)
    frozen = compute_frozen_indices(
        slab,
        relax_top_layer=True,
        top_layer_tolerance=2.1,
        material_type="slab",
    )
    assert len(free) == 32  # two layers × 4×4
    assert len(frozen) == 32
    constrained = apply_surface_constraints(
        slab,
        relax_top_layer=True,
        top_layer_tolerance=2.1,
        material_type="slab",
    )
    assert frozen_indices_from_constraints(constrained) == frozen


def test_relax_top_layer_empty_freeze_falls_back_to_full_substrate(caplog):
    """If every atom would be free, apply_surface_constraints freezes all atoms."""
    slab = make_slab(nx=2, ny=2, n_layers=2, spacing=2.0)
    # Tolerance larger than slab thickness → simple band frees everyone.
    with caplog.at_level(logging.WARNING, logger="metalsurfer.surface_prep._surfaces"):
        constrained = apply_surface_constraints(
            slab,
            relax_top_layer=True,
            top_layer_tolerance=100.0,
            material_type="slab",
        )
    frozen = frozen_indices_from_constraints(constrained)
    assert frozen == list(range(len(slab)))
    assert any("no atoms frozen" in r.message for r in caplog.records)


def test_deposit_adatoms_refreshes_fixatoms_for_new_atoms(tmp_path):
    """Appending adatoms must not leave them free under stale base FixAtoms."""
    from metalsurfer.surface_prep import SlabContainer, deposit_adatoms

    base = SlabContainer(make_slab(nx=4, ny=4, n_layers=3))
    n_base = len(base.atoms)
    assert frozen_indices_from_constraints(base.atoms) == list(range(n_base))

    deposited = deposit_adatoms(
        base,
        "Sn",
        coverage_fraction=0.2,
        seed=42,
        results_dir=str(tmp_path),
        relaxation_mode="none",
    )
    n_total = len(deposited.atoms)
    assert n_total > n_base
    assert frozen_indices_from_constraints(deposited.atoms) == list(range(n_total))

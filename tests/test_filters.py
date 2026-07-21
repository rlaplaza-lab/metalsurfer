"""Comprehensive tests for decomposition, desorption, and duplicate filters.

The decomposition filter must catch:
- Complete fragmentation (molecule splits into pieces)
- Atom loss / gain (H migration to surface, formula mismatch)
- Bond breaking (C-O, C-C, O-H cleavage)
- Ring opening
- Subtle rearrangements (H shift changing per-atom coordination)
- Bond formation (spurious new bonds)

The filter must NOT reject:
- Intact molecules at various heights above the surface
- Intact molecules with multiple surface element types
"""

import numpy as np
import pytest
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii

from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import (
    _bond_counts_from_atoms,
    _bond_counts_from_smiles,
    _coordination_fingerprint_from_atoms,
    _coordination_fingerprint_from_smiles,
    _formula_from_atoms,
    _formula_from_smiles,
    _is_molecule_connected,
    check_decomposition,
    check_desorption,
    filter_results,
)
from metalsurfer.models import ScreeningResult

from .conftest import (
    make_placement_descriptor,
    make_slab,
    make_water,
    place_molecule_on_slab,
)


def _sr(atoms, energy_adsorption, placement_id, molecule="test"):
    """Shorthand to build a ScreeningResult for testing."""
    return ScreeningResult(
        molecule=molecule,
        placement_id=placement_id,
        energy_adslab=energy_adsorption,
        energy_slab=0.0,
        energy_adsorbate=0.0,
        energy_adsorption=energy_adsorption,
        atoms=atoms,
        slab_size=16,
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=placement_id),
    )


# ---------------------------------------------------------------------------
# test helpers
# ---------------------------------------------------------------------------


def _make_alloy_slab():
    """A Ru/Cu alloy slab at z=0 using the shared reference layout."""
    atoms = make_slab(n_layers=1)
    syms = atoms.get_chemical_symbols()
    for i in range(len(syms)):
        if i % 2 == 1:
            syms[i] = "Cu"
    atoms.set_chemical_symbols(syms)
    return atoms


def _make_ethanol():
    """C2H5OH – 9 atoms with correct connectivity (covalent_radii geometry for bond-count tests)."""
    r_CC = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["C"]]
    r_CO = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["O"]]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    r_OH = covalent_radii[atomic_numbers["O"]] + covalent_radii[atomic_numbers["H"]]
    return Atoms(
        "CCOH6",
        positions=[
            [0.0, 0.0, 0.0],  # C1
            [r_CC, 0.0, 0.0],  # C2
            [r_CC + r_CO, 0.0, 0.0],  # O
            [r_CC + r_CO + r_OH, 0.0, 0.0],  # H (on O)
            [-r_CH * 0.33, r_CH * 0.94, 0.0],  # H (on C1)
            [-r_CH * 0.33, -r_CH * 0.47, r_CH * 0.82],  # H (on C1)
            [-r_CH * 0.33, -r_CH * 0.47, -r_CH * 0.82],  # H (on C1)
            [r_CC + r_CH * 0.33, r_CH * 0.94, 0.0],  # H (on C2)
            [r_CC + r_CH * 0.33, -r_CH * 0.94, 0.0],  # H (on C2)
        ],
    )


def _make_methanol():
    """CH3OH – 6 atoms."""
    r_CO = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["O"]]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    r_OH = covalent_radii[atomic_numbers["O"]] + covalent_radii[atomic_numbers["H"]]
    return Atoms(
        "COH4",
        positions=[
            [0.0, 0.0, 0.0],
            [r_CO, 0.0, 0.0],
            [r_CO + r_OH, 0.0, 0.0],
            [-r_CH * 0.33, r_CH * 0.94, 0.0],
            [-r_CH * 0.33, -r_CH * 0.47, r_CH * 0.82],
            [-r_CH * 0.33, -r_CH * 0.47, -r_CH * 0.82],
        ],
    )


def _make_acetic_acid():
    """CH3COOH – 8 atoms with C-C, C=O, C-O, O-H, C-H bonds.

    Uses 120-degree O-C-O angle so the two oxygens are far enough apart
    that no spurious O-O bond is detected at multiplier=1.3.
    """
    r_CC = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["C"]]
    r_CO = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["O"]]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    r_OH = covalent_radii[atomic_numbers["O"]] + covalent_radii[atomic_numbers["H"]]
    cos60 = 0.5
    sin60 = 0.866
    return Atoms(
        "CCO2H4",
        positions=[
            [0.0, 0.0, 0.0],  # C1 (methyl)
            [r_CC, 0.0, 0.0],  # C2 (carboxyl)
            [r_CC + r_CO * cos60, r_CO * sin60, 0.0],  # O (=O)
            [r_CC + r_CO * cos60, -r_CO * sin60, 0.0],  # O (-OH)
            [r_CC + r_CO * cos60 + r_OH, -r_CO * sin60, 0.0],  # H (on OH)
            [-r_CH * 0.33, r_CH * 0.94, 0.0],  # H (on C1)
            [-r_CH * 0.33, -r_CH * 0.47, r_CH * 0.82],  # H (on C1)
            [-r_CH * 0.33, -r_CH * 0.47, -r_CH * 0.82],  # H (on C1)
        ],
    )


# ---------------------------------------------------------------------------
# SMILES-derived reference helpers
# ---------------------------------------------------------------------------


def test_formula_from_smiles_water():
    f = _formula_from_smiles("O")
    assert f == {"O": 1, "H": 2}


def test_formula_from_smiles_ethanol():
    f = _formula_from_smiles("CCO")
    assert f == {"C": 2, "H": 6, "O": 1}


def test_formula_from_smiles_acetic_acid():
    f = _formula_from_smiles("CC(=O)O")
    assert f == {"C": 2, "H": 4, "O": 2}


def test_formula_from_smiles_invalid():
    assert _formula_from_smiles("not_a_smiles_xyz!!!") is None


def test_bond_counts_from_smiles_water():
    b = _bond_counts_from_smiles("O")
    assert b[frozenset({"O", "H"})] == 2
    assert len(b) == 1


def test_bond_counts_from_smiles_ethanol():
    b = _bond_counts_from_smiles("CCO")
    assert b[frozenset({"C"})] == 1
    assert b[frozenset({"C", "O"})] == 1
    assert b[frozenset({"C", "H"})] == 5
    assert b[frozenset({"O", "H"})] == 1


def test_coordination_fingerprint_from_smiles_water():
    fp = _coordination_fingerprint_from_smiles("O")
    assert fp["O"] == [2]
    assert fp["H"] == [1, 1]


def test_coordination_fingerprint_from_smiles_ethanol():
    fp = _coordination_fingerprint_from_smiles("CCO")
    # Ethanol (CCO): two carbons with degree 4, oxygen with degree 2.
    assert sorted(fp["C"]) == [4, 4]
    assert fp["O"] == [2]
    assert sorted(fp["H"]) == [1, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# formula check on combined slab+molecule
# ---------------------------------------------------------------------------


def test_formula_from_atoms_intact():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water())
    f = _formula_from_atoms(combined, surface_symbols=["Ru"])
    assert f == {"O": 1, "H": 2}


def test_formula_from_atoms_alloy_surface():
    slab = _make_alloy_slab()
    combined = place_molecule_on_slab(slab, make_water())
    f = _formula_from_atoms(combined, surface_symbols=["Ru", "Cu"])
    assert f == {"O": 1, "H": 2}


def test_formula_detects_missing_hydrogen():
    """Simulate H migrating away from the molecule."""
    slab = make_slab(n_layers=1)
    water = make_water()
    combined = place_molecule_on_slab(slab, water)
    # remove the last H by keeping only slab + O + one H
    pos = combined.get_positions()
    syms = combined.get_chemical_symbols()
    keep = list(range(len(slab))) + [len(slab), len(slab) + 1]
    new_atoms = Atoms(
        symbols=[syms[i] for i in keep],
        positions=pos[keep],
    )
    new_atoms.set_cell(slab.get_cell())
    new_atoms.set_pbc(slab.get_pbc())
    f = _formula_from_atoms(new_atoms, surface_symbols=["Ru"])
    ref = _formula_from_smiles("O")
    assert f != ref  # one H missing


# ---------------------------------------------------------------------------
# connectivity checks
# ---------------------------------------------------------------------------


def test_is_connected_intact_water():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water())
    assert _is_molecule_connected(combined, surface_symbols=["Ru"], multiplier=1.3)


def test_is_connected_fragmented():
    slab = make_slab(n_layers=1)
    slab_z = max(slab.get_positions()[:, 2])
    mol = Atoms(
        "OH2",
        positions=[
            [5.0, 5.0, slab_z + 3.0],
            [0.0, 0.0, slab_z + 11.0],
            [9.0, 9.0, slab_z + 11.0],
        ],
    )
    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    assert not _is_molecule_connected(combined, surface_symbols=["Ru"], multiplier=1.3)


def test_is_connected_single_atom():
    """A single non-surface atom is always 'connected'."""
    slab = make_slab(n_layers=1)
    slab_z = max(slab.get_positions()[:, 2])
    mol = Atoms("O", positions=[[5.0, 5.0, slab_z + 3.0]])
    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    assert _is_molecule_connected(combined, surface_symbols=["Ru"], multiplier=1.3)


def test_connectivity_multiplier_sensitivity():
    """Tight multiplier can flag structures that a loose one accepts."""
    slab = make_slab(n_layers=1)
    water = make_water()
    combined = place_molecule_on_slab(slab, water)
    # stretch one O-H bond to 1.8x covalent sum
    pos = combined.get_positions().copy()
    slab_size = len(slab)
    o_pos = pos[slab_size]
    h_pos = pos[slab_size + 1]
    r_OH = covalent_radii[atomic_numbers["O"]] + covalent_radii[atomic_numbers["H"]]
    direction = h_pos - o_pos
    direction /= np.linalg.norm(direction)
    pos[slab_size + 1] = o_pos + direction * r_OH * 1.5
    combined.set_positions(pos)

    assert _is_molecule_connected(combined, surface_symbols=["Ru"], multiplier=1.6)
    assert not _is_molecule_connected(combined, surface_symbols=["Ru"], multiplier=1.2)


# ---------------------------------------------------------------------------
# bond-count checks
# ---------------------------------------------------------------------------


def test_bond_counts_intact_ethanol():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, _make_ethanol())
    actual = _bond_counts_from_atoms(combined, surface_symbols=["Ru"], multiplier=1.3)
    ref = _bond_counts_from_smiles("CCO")
    assert actual == ref


def test_bond_counts_detect_cc_break():
    """Breaking the C-C bond should change bond counts."""
    slab = make_slab(n_layers=1)
    ethanol = _make_ethanol()
    combined = place_molecule_on_slab(slab, ethanol)
    pos = combined.get_positions().copy()
    slab_size = len(slab)
    # push C2 far from C1 to break C-C bond
    pos[slab_size + 1, 0] += 5.0
    # also move atoms bonded to C2 (O, H_O, H_C2_1, H_C2_2)
    for idx in [slab_size + 2, slab_size + 3, slab_size + 7, slab_size + 8]:
        pos[idx, 0] += 5.0
    combined.set_positions(pos)
    actual = _bond_counts_from_atoms(combined, surface_symbols=["Ru"], multiplier=1.3)
    ref = _bond_counts_from_smiles("CCO")
    assert actual != ref


# ---------------------------------------------------------------------------
# coordination fingerprint checks
# ---------------------------------------------------------------------------


def test_coordination_fingerprint_intact_ethanol():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, _make_ethanol())
    actual = _coordination_fingerprint_from_atoms(
        combined, surface_symbols=["Ru"], multiplier=1.3
    )
    ref = _coordination_fingerprint_from_smiles("CCO")
    assert actual == ref


def test_coordination_fingerprint_detects_h_shift():
    """Simulate a 1,2-H shift: move one H from C1 to C2.

    Total bond-pair counts may stay the same (both C-H), but the
    per-atom coordination changes: C1 goes from 4 to 3, C2 goes from
    3-ish to 4-ish (or vice versa), changing the sorted fingerprint.
    """
    slab = make_slab(n_layers=1)
    ethanol = _make_ethanol()
    combined = place_molecule_on_slab(slab, ethanol)
    pos = combined.get_positions().copy()
    slab_size = len(slab)

    # H at index slab_size+4 is bonded to C1 (index slab_size+0).
    # Move it close to C2 (index slab_size+1) instead.
    c2_pos = pos[slab_size + 1]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    pos[slab_size + 4] = c2_pos + np.array([0.0, 0.0, r_CH * 0.9])
    combined.set_positions(pos)

    actual = _coordination_fingerprint_from_atoms(
        combined, surface_symbols=["Ru"], multiplier=1.3
    )
    ref = _coordination_fingerprint_from_smiles("CCO")
    assert actual != ref, (
        f"H-shift should change coordination fingerprint: ref={ref}, actual={actual}"
    )


# ---------------------------------------------------------------------------
# check_decomposition — intact molecules
# ---------------------------------------------------------------------------


def test_decomposition_intact_water():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.2, 1.3],
    )
    assert ok, reason


def test_decomposition_intact_ethanol():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, _make_ethanol())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CCO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert ok, reason


def test_decomposition_intact_methanol():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, _make_methanol())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert ok, reason


def test_decomposition_intact_acetic_acid():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, _make_acetic_acid())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CC(=O)O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert ok, reason


def test_decomposition_intact_on_alloy():
    """Intact molecule on a Ru/Cu alloy surface."""
    slab = _make_alloy_slab()
    combined = place_molecule_on_slab(slab, make_water())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru", "Cu"],
        connectivity_multipliers=[1.2, 1.3],
    )
    assert ok, reason


def test_decomposition_no_smiles_reference():
    """Without reference SMILES, only the fragment check runs."""
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water())
    ok, reason = check_decomposition(
        combined,
        reference_smiles=None,
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert ok
    assert "no SMILES reference" in reason


# ---------------------------------------------------------------------------
# check_decomposition — fragmentation
# ---------------------------------------------------------------------------


def test_decomposition_fragmented_water():
    slab = make_slab(n_layers=1)
    slab_z = max(slab.get_positions()[:, 2])
    mol = Atoms(
        "OH2",
        positions=[
            [5.0, 5.0, slab_z + 3.0],
            [0.0, 0.0, slab_z + 11.0],
            [9.0, 9.0, slab_z + 11.0],
        ],
    )
    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.2, 1.3],
    )
    assert not ok
    assert "not connected" in reason


def _combined_slab_two_waters_far_apart():
    """slab + H2O + H2O (two disconnected intact waters) for saturation-style checks."""
    slab = make_slab(n_layers=1)
    first = place_molecule_on_slab(
        slab, make_water(), z_offset=3.0, x_shift=2.0, y_shift=2.0
    )
    w2 = make_water().copy()
    slab_z = float(np.max(slab.get_positions()[:, 2]))
    z_offset = 3.0
    pos2 = w2.get_positions().copy()
    pos2 -= np.mean(pos2, axis=0)
    pos2[:, 0] += 7.0
    pos2[:, 1] += 2.0
    pos2[:, 2] += slab_z + z_offset
    w2.set_positions(pos2)
    combined = first + w2
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    return slab, first, combined


def test_check_decomposition_prefix_ignores_prior_adsorbate():
    """Only the trailing adsorbate is validated (sequential saturation)."""
    slab, slab_plus_first, combined = _combined_slab_two_waters_far_apart()
    prefix = len(slab_plus_first)
    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
        adsorbate_prefix_atoms=prefix,
    )
    assert ok, reason

    ok_legacy, reason_legacy = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok_legacy
    assert "not connected" in reason_legacy


def test_check_decomposition_prefix_invalid():
    ok, reason = check_decomposition(
        make_slab(n_layers=1),
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
        adsorbate_prefix_atoms=999,
    )
    assert not ok
    assert "invalid adsorbate_prefix_atoms" in reason


def test_filter_results_uses_slab_prefix_for_decomposition():
    """filter_results passes len(slab) so the second molecule is checked alone."""
    slab, slab_plus_first, combined = _combined_slab_two_waters_far_apart()
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    results = [_sr(combined, -1.0, 0)]
    filtered = filter_results(
        results,
        slab=slab_plus_first,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 1

    filtered_wrong = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered_wrong) == 0


def test_decomposition_fragmented_ethanol_two_pieces():
    """Ethanol splits into CH3 + CH2OH.

    With MIC-aware distances in a 10 A cell, an 8 A displacement wraps
    to ~2 A, so the connectivity check may not trigger — but the bond
    pattern / coordination fingerprint checks still catch the
    rearrangement.  We only assert the decomposition is detected.
    """
    slab = make_slab(n_layers=1)
    ethanol = _make_ethanol()
    combined = place_molecule_on_slab(slab, ethanol)
    pos = combined.get_positions().copy()
    slab_size = len(slab)
    # move C2 and everything bonded to it far away
    for idx in [
        slab_size + 1,
        slab_size + 2,
        slab_size + 3,
        slab_size + 7,
        slab_size + 8,
    ]:
        pos[idx, 0] += 8.0
    combined.set_positions(pos)
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CCO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok, f"Fragmented ethanol should be detected, got: {reason}"


# ---------------------------------------------------------------------------
# check_decomposition — formula mismatch (atom loss / gain)
# ---------------------------------------------------------------------------


def test_decomposition_h_loss():
    """One H migrated away and is no longer part of the system."""
    slab = make_slab(n_layers=1)
    water = make_water()
    combined = place_molecule_on_slab(slab, water)
    # remove the last H entirely
    pos = combined.get_positions()
    syms = combined.get_chemical_symbols()
    keep = list(range(len(slab))) + [len(slab), len(slab) + 1]
    truncated = Atoms(
        symbols=[syms[i] for i in keep],
        positions=pos[keep],
    )
    truncated.set_cell(slab.get_cell())
    truncated.set_pbc(slab.get_pbc())
    ok, reason = check_decomposition(
        truncated,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok
    assert "formula mismatch" in reason


def test_decomposition_extra_atom():
    """An extra O appeared among the adsorbate atoms (e.g. surface O migrated)."""
    slab = make_slab(n_layers=1)
    water = make_water()
    combined = place_molecule_on_slab(slab, water)
    slab_z = max(slab.get_positions()[:, 2])
    extra = Atoms("O", positions=[[4.0, 4.0, slab_z + 3.5]])
    combined = combined + extra
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok
    assert "formula mismatch" in reason


# ---------------------------------------------------------------------------
# check_decomposition — bond-pair mismatch
# ---------------------------------------------------------------------------


def test_decomposition_bond_mismatch_wrong_molecule():
    """Bond-count check catches wrong molecule (CC vs CCO)."""
    slab = make_slab(n_layers=1)
    slab_z = max(slab.get_positions()[:, 2])
    r_CC = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["C"]]
    mol = Atoms(
        "CC",
        positions=[
            [5.0, 5.0, slab_z + 3.0],
            [5.0 + r_CC * 0.9, 5.0, slab_z + 3.0],
        ],
    )
    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CCO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok
    # formula check will fire first since CC vs CCO differs in atom count
    assert "formula mismatch" in reason or "bond pattern mismatch" in reason


def test_decomposition_oh_bond_break():
    """Break the O-H bond in methanol while keeping everything connected via C."""
    slab = make_slab(n_layers=1)
    methanol = _make_methanol()
    combined = place_molecule_on_slab(slab, methanol)
    pos = combined.get_positions().copy()
    slab_size = len(slab)
    # methanol atom order: C, O, H(O), H(C), H(C), H(C)
    # move H(O) far from O but close to C to keep it "connected"
    c_pos = pos[slab_size + 0]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    pos[slab_size + 2] = c_pos + np.array([0.0, 0.0, r_CH * 0.9])
    combined.set_positions(pos)
    ok, reason = check_decomposition(
        combined,
        reference_smiles="CO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok
    # the H moved from O to C: coordination changes
    assert "mismatch" in reason


# ---------------------------------------------------------------------------
# check_decomposition — subtle rearrangement (coordination fingerprint)
# ---------------------------------------------------------------------------


def test_decomposition_h_shift_caught_by_coordination():
    """1,2-H shift in ethanol: H moves from C1 to C2.

    Bond-pair counts (C-H total) could stay the same, but the per-atom
    coordination fingerprint must change because C1 loses a neighbour
    and C2 gains one.
    """
    slab = make_slab(n_layers=1)
    ethanol = _make_ethanol()
    combined = place_molecule_on_slab(slab, ethanol)
    pos = combined.get_positions().copy()
    slab_size = len(slab)

    # move H#4 (on C1) close to C2
    c2_pos = pos[slab_size + 1]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    pos[slab_size + 4] = c2_pos + np.array([0.0, 0.0, r_CH * 0.9])
    combined.set_positions(pos)

    ok, reason = check_decomposition(
        combined,
        reference_smiles="CCO",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert not ok, (
        "H-shift should be caught by coordination fingerprint or bond mismatch"
    )
    assert "mismatch" in reason


# ---------------------------------------------------------------------------
# check_desorption
# ---------------------------------------------------------------------------


def test_desorption_adsorbed():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    ok, reason = check_desorption(
        combined, slab, binding_threshold=4.0, material_type="slab"
    )
    assert ok, reason


def test_desorption_too_far():
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water(), z_offset=10.0)
    ok, reason = check_desorption(
        combined, slab, binding_threshold=4.0, material_type="slab"
    )
    assert not ok
    assert "too far" in reason


def test_desorption_borderline():
    """Molecule right at the threshold should be marked desorbed (> not >=)."""
    slab = make_slab(n_layers=1)
    combined = place_molecule_on_slab(slab, make_water(), z_offset=4.5)
    ok, reason = check_desorption(
        combined, slab, binding_threshold=4.0, material_type="slab"
    )
    assert not ok
    assert "too far" in reason


def test_desorption_threshold_is_strict_greater_than():
    """Exactly at binding_threshold remains adsorbed; epsilon beyond is desorbed."""
    from metalsurfer.placement import calculate_min_distance

    slab = make_slab(n_layers=1)
    threshold = 3.0
    # Monoatomic adsorbate directly above a surface atom → MIC min == Δz.
    surface_atom = slab.get_positions()[0]
    ads = Atoms(
        "He",
        positions=[[surface_atom[0], surface_atom[1], surface_atom[2] + threshold]],
    )
    combined = slab + ads
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())
    cell = combined.get_cell()
    min_at = calculate_min_distance(
        combined[len(slab) :].get_positions(),
        slab.get_positions(),
        cell,
        use_pbc=True,
        pbc=[True, True, False],
    )
    assert min_at == pytest.approx(threshold, abs=1e-6)

    ok_eq, reason_eq = check_desorption(
        combined, slab, binding_threshold=threshold, material_type="slab"
    )
    assert ok_eq, reason_eq
    assert "adsorbed" in reason_eq

    pos = combined.get_positions().copy()
    pos[len(slab) :, 2] += 1e-3
    combined.set_positions(pos)
    ok_gt, reason_gt = check_desorption(
        combined, slab, binding_threshold=threshold, material_type="slab"
    )
    assert not ok_gt
    assert "too far" in reason_gt


def test_desorption_no_adsorbate():
    slab = make_slab(n_layers=1)
    ok, reason = check_desorption(
        slab, slab, binding_threshold=4.0, material_type="slab"
    )
    assert not ok
    assert "no adsorbate" in reason


def test_desorption_ignores_pre_adsorbed_atoms_when_surface_symbols_provided():
    """Regression: in saturation, slab may include previously adsorbed atoms.

    Distance-to-surface checks must ignore those and consider only the true
    substrate atoms (identified by surface_symbols).
    """
    slab_metal = make_slab(n_layers=1, symbol="Ru")
    x_shift = 5.0
    y_shift = 5.0
    z_offset = 10.0

    # Build the placement we want first, then pin a fake "pre-adsorbed" atom
    # directly under the new molecule so it can incorrectly mask desorption.
    slab_metal_z = float(np.max(slab_metal.get_positions()[:, 2]))
    water = make_water().copy()
    pos = water.get_positions().copy()
    pos -= np.mean(pos, axis=0)
    pos[:, 0] += x_shift
    pos[:, 1] += y_shift
    pos[:, 2] += slab_metal_z + z_offset
    water.set_positions(pos)

    # Place the pre-adsorbed atom at the oxygen position (very close contact).
    o_pos = water.get_positions()[0].copy()
    pre_adsorbed = Atoms("C", positions=[o_pos])
    slab_with_pre_adsorbate = slab_metal + pre_adsorbed
    slab_with_pre_adsorbate.set_cell(slab_metal.get_cell())
    slab_with_pre_adsorbate.set_pbc(slab_metal.get_pbc())

    combined = slab_with_pre_adsorbate + water
    combined.set_cell(slab_with_pre_adsorbate.get_cell())
    combined.set_pbc(slab_with_pre_adsorbate.get_pbc())

    ok, _ = check_desorption(
        combined,
        slab_with_pre_adsorbate,
        binding_threshold=4.0,
        material_type="slab",
    )
    assert ok, "Without surface_symbols, pre-adsorbed atoms can mask desorption"

    ok, reason = check_desorption(
        combined,
        slab_with_pre_adsorbate,
        binding_threshold=4.0,
        surface_symbols=["Ru"],
        material_type="slab",
    )
    assert not ok
    assert "too far" in reason


def test_filter_results_desorption_uses_surface_symbols_masking():
    """filter_results should pass surface_symbols into desorption filtering."""
    slab_metal = make_slab(n_layers=1, symbol="Ru")
    slab_z = float(np.max(slab_metal.get_positions()[:, 2]))

    pre_adsorbed = Atoms("C", positions=[[5.0, 5.0, slab_z + 9.8]])
    slab_with_pre_adsorbate = slab_metal + pre_adsorbed
    slab_with_pre_adsorbate.set_cell(slab_metal.get_cell())
    slab_with_pre_adsorbate.set_pbc(slab_metal.get_pbc())

    combined = place_molecule_on_slab(
        slab_with_pre_adsorbate,
        make_water(),
        z_offset=10.0,
        x_shift=5.0,
        y_shift=5.0,
    )
    results = [_sr(combined, -1.0, 0)]

    config = AdsorptionConfig(skip_topology_check=True, connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab_with_pre_adsorbate,
        surface_symbols=["Ru"],
        reference_smiles=None,
        config=config,
    )
    assert filtered == []


# ---------------------------------------------------------------------------
# duplicate detection
# ---------------------------------------------------------------------------


def test_duplicate_removal():
    slab = make_slab(n_layers=1)
    combined1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    combined2 = combined1.copy()

    results = [
        _sr(combined1, -1.0, 0),
        _sr(combined2, -1.01, 1),
    ]
    config = AdsorptionConfig(
        energy_dedup_threshold=0.05,
        rmsd_dedup_threshold=0.1,
        connectivity_multipliers=[1.3],
    )
    filtered = filter_results(results, slab=slab, surface_symbols=["Ru"], config=config)
    assert len(filtered) == 1


def test_duplicate_removal_tracks_removed_duplicates():
    slab = make_slab(n_layers=1)
    combined1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    combined2 = combined1.copy()

    results = [
        _sr(combined1, -1.0, 0),
        _sr(combined2, -1.01, 1),
    ]
    config = AdsorptionConfig(
        energy_dedup_threshold=0.05,
        rmsd_dedup_threshold=0.1,
        connectivity_multipliers=[1.3],
    )
    removed: list[ScreeningResult] = []
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        config=config,
        duplicate_results_out=removed,
    )

    assert len(filtered) == 1
    assert len(removed) == 1
    assert removed[0].placement_id != filtered[0].placement_id


def test_distinct_kept():
    slab = make_slab(n_layers=1)
    combined1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    combined2 = combined1.copy()
    pos = combined2.get_positions().copy()
    slab_size = len(slab)
    pos[slab_size:, 0] += 3.0
    combined2.set_positions(pos)

    results = [
        _sr(combined1, -1.0, 0),
        _sr(combined2, -2.0, 1),
    ]
    config = AdsorptionConfig(
        energy_dedup_threshold=0.05,
        rmsd_dedup_threshold=0.1,
        connectivity_multipliers=[1.3],
    )
    filtered = filter_results(results, slab=slab, surface_symbols=["Ru"], config=config)
    assert len(filtered) == 2


def test_duplicate_different_energy_kept():
    """Same positions but very different energies -> not duplicates."""
    slab = make_slab(n_layers=1)
    combined1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    combined2 = combined1.copy()

    results = [
        _sr(combined1, -1.0, 0),
        _sr(combined2, -3.0, 1),
    ]
    config = AdsorptionConfig(
        energy_dedup_threshold=0.05,
        rmsd_dedup_threshold=0.1,
        connectivity_multipliers=[1.3],
    )
    filtered = filter_results(results, slab=slab, surface_symbols=["Ru"], config=config)
    assert len(filtered) == 2


# ---------------------------------------------------------------------------
# filter_results pipeline — integration
# ---------------------------------------------------------------------------


def test_filter_pipeline_removes_decomposed_and_desorbed():
    """Unified filter drops both decomposed and desorbed."""
    slab = make_slab(n_layers=1)
    good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    slab_z = max(slab.get_positions()[:, 2])
    decomposed_mol = Atoms(
        "OH2",
        positions=[
            [5.0, 5.0, slab_z + 3.0],
            [0.0, 0.0, slab_z + 11.0],
            [9.0, 9.0, slab_z + 11.0],
        ],
    )
    decomposed = slab + decomposed_mol
    decomposed.set_cell(slab.get_cell())
    decomposed.set_pbc(slab.get_pbc())
    desorbed = place_molecule_on_slab(slab, make_water(), z_offset=10.0)

    results = [
        _sr(good, -1.0, 0),
        _sr(decomposed, -0.5, 1),
        _sr(desorbed, -0.8, 2),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 0


def test_filter_pipeline_catches_rearranged():
    """Pipeline must reject rearranged molecules (H-shift), not just
    fragmented ones."""
    slab = make_slab(n_layers=1)
    good = place_molecule_on_slab(slab, _make_ethanol(), z_offset=2.5)

    rearranged_ethanol = _make_ethanol()
    rearranged = place_molecule_on_slab(slab, rearranged_ethanol, z_offset=2.5)
    pos = rearranged.get_positions().copy()
    slab_size = len(slab)
    c2_pos = pos[slab_size + 1]
    r_CH = covalent_radii[atomic_numbers["C"]] + covalent_radii[atomic_numbers["H"]]
    pos[slab_size + 4] = c2_pos + np.array([0.0, 0.0, r_CH * 0.9])
    rearranged.set_positions(pos)

    results = [
        _sr(good, -1.5, 0),
        _sr(rearranged, -2.0, 1),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="CCO",
        config=config,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 0


def test_filter_pipeline_catches_atom_loss():
    """A molecule that lost an H must be filtered."""
    slab = make_slab(n_layers=1)
    good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)

    water = make_water()
    full = place_molecule_on_slab(slab, water, z_offset=2.5)
    pos = full.get_positions()
    syms = full.get_chemical_symbols()
    keep = list(range(len(slab))) + [len(slab), len(slab) + 1]
    truncated = Atoms(
        symbols=[syms[i] for i in keep],
        positions=pos[keep],
    )
    truncated.set_cell(slab.get_cell())
    truncated.set_pbc(slab.get_pbc())

    results = [
        _sr(good, -1.0, 0),
        _sr(truncated, -2.0, 1),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 0


def test_filter_pipeline_empty_input():
    slab = make_slab(n_layers=1)
    filtered = filter_results([], slab=slab, surface_symbols=["Ru"])
    assert filtered == []


def test_filter_pipeline_all_rejected():
    """When every result is bad, return an empty list."""
    slab = make_slab(n_layers=1)
    decomposed_mol = Atoms(
        "OH2",
        positions=[
            [5.0, 5.0, 3.0],
            [0.0, 0.0, 11.0],
            [9.0, 9.0, 11.0],
        ],
    )
    combined = slab + decomposed_mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())

    results = [_sr(combined, -1.0, 0)]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 0


def test_filter_pipeline_keep_best():
    """keep_best=True should return only the best configuration."""
    slab = make_slab(n_layers=1)
    c1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5, x_shift=3.0)
    c2 = place_molecule_on_slab(slab, make_water(), z_offset=2.5, x_shift=7.0)

    results = [
        _sr(c1, -0.5, 0),
        _sr(c2, -2.5, 1),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
        keep_best=True,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 1


def test_filter_pipeline_skip_topology_check_allows_decomposed():
    """When skip_topology_check=True, decomposed structures pass through."""
    slab = make_slab(n_layers=1)
    good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    slab_z = max(slab.get_positions()[:, 2])
    decomposed_mol = Atoms(
        "OH2",
        positions=[
            [5.0, 5.0, slab_z + 3.0],
            [0.0, 0.0, slab_z + 11.0],
            [9.0, 9.0, slab_z + 11.0],
        ],
    )
    decomposed = slab + decomposed_mol
    decomposed.set_cell(slab.get_cell())
    decomposed.set_pbc(slab.get_pbc())

    results = [
        _sr(good, -1.0, 0),
        _sr(decomposed, -0.5, 1),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3], skip_topology_check=True)
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 2
    assert {r.placement_id for r in filtered} == {0, 1}


def test_filter_pipeline_skip_desorption_check_allows_desorbed():
    """When skip_desorption_check=True, desorbed structures pass through."""
    slab = make_slab(n_layers=1)
    good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    desorbed = place_molecule_on_slab(slab, make_water(), z_offset=10.0)

    results = [
        _sr(good, -1.0, 0),
        _sr(desorbed, -0.8, 1),
    ]
    config = AdsorptionConfig(skip_desorption_check=True)
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 2
    assert {r.placement_id for r in filtered} == {0, 1}


def test_filter_pipeline_alloy_surface():
    """Filters work correctly when surface has multiple element types."""
    slab = _make_alloy_slab()
    good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
    desorbed = place_molecule_on_slab(slab, make_water(), z_offset=10.0)

    results = [
        _sr(good, -1.0, 0),
        _sr(desorbed, -0.2, 1),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru", "Cu"],
        reference_smiles="O",
        config=config,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 0


# ---------------------------------------------------------------------------
# keep_best selects minimum E_ads (most negative), not max |E_ads|
# ---------------------------------------------------------------------------


def test_keep_best_selects_minimum_energy():
    """keep_best=True must select the most negative E_ads, not largest |E_ads|."""
    slab = make_slab(n_layers=1)
    c1 = place_molecule_on_slab(slab, make_water(), z_offset=2.5, x_shift=3.0)
    c2 = place_molecule_on_slab(slab, make_water(), z_offset=2.5, x_shift=7.0)
    c3 = place_molecule_on_slab(slab, make_water(), z_offset=2.5, x_shift=5.0)

    results = [
        _sr(c1, -0.5, 0),
        _sr(c2, -2.5, 1),
        _sr(c3, 3.0, 2),
    ]
    config = AdsorptionConfig(connectivity_multipliers=[1.3])
    filtered = filter_results(
        results,
        slab=slab,
        surface_symbols=["Ru"],
        reference_smiles="O",
        config=config,
        keep_best=True,
    )
    assert len(filtered) == 1
    assert filtered[0].placement_id == 1
    assert filtered[0].energy_adsorption == -2.5


# ---------------------------------------------------------------------------
# PBC-crossing molecule should not be flagged as decomposed
# ---------------------------------------------------------------------------


def test_connected_molecule_across_pbc():
    """A molecule straddling the periodic boundary must not be flagged decomposed."""
    slab = make_slab(n_layers=1)
    mol = make_water()
    pos = mol.get_positions().copy()
    pos -= np.mean(pos, axis=0)
    surface_z = max(slab.get_positions()[:, 2])
    pos[:, 2] += surface_z + 2.5
    lx = float(slab.get_cell()[0, 0])
    pos[0, 0] = 0.03 * lx
    pos[1, 0] = 0.97 * lx
    pos[2, 0] = 0.01 * lx
    pos[:, 1] += 5.0
    mol.set_positions(pos)

    combined = slab + mol
    combined.set_cell(slab.get_cell())
    combined.set_pbc(slab.get_pbc())

    ok, reason = check_decomposition(
        combined,
        reference_smiles="O",
        surface_symbols=["Ru"],
        connectivity_multipliers=[1.3],
    )
    assert ok, f"Molecule crossing PBC should be seen as connected, got: {reason}"

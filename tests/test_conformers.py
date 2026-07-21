"""Tests for conformer generation, deduplication, and Boltzmann selection."""

import random

import numpy as np
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import (
    create_conformers_from_smiles,
    remove_duplicate_conformers,
    select_conformer_boltzmann,
    select_conformer_for_placement,
)

# ---------------------------------------------------------------------------
# create_conformers_from_smiles
# ---------------------------------------------------------------------------


def _pairwise_distances(atoms: Atoms) -> np.ndarray:
    pos = atoms.get_positions()
    diffs = pos[:, None, :] - pos[None, :, :]
    return np.linalg.norm(diffs, axis=2)


class TestCreateConformers:
    def test_valid_smiles_returns_conformers_and_energies(self):
        config = AdsorptionConfig(num_conformers=3, seed=42)
        result = create_conformers_from_smiles("O", config=config)
        assert result is not None
        conformers, energies = result
        assert len(conformers) >= 1
        assert len(energies) == len(conformers)
        assert all(np.isfinite(e) for e in energies)
        for c in conformers:
            assert isinstance(c, Atoms)
            assert len(c) == 3  # H2O
            assert sorted(c.get_chemical_symbols()) == ["H", "H", "O"]
            assert np.all(np.isfinite(c.get_positions()))
            # Physical water geometry: O–H ~0.96 Å, H–H ~1.5 Å.
            syms = c.get_chemical_symbols()
            o_idx = syms.index("O")
            h_idxs = [i for i, s in enumerate(syms) if s == "H"]
            pos = c.get_positions()
            oh1 = float(np.linalg.norm(pos[h_idxs[0]] - pos[o_idx]))
            oh2 = float(np.linalg.norm(pos[h_idxs[1]] - pos[o_idx]))
            hh = float(np.linalg.norm(pos[h_idxs[1]] - pos[h_idxs[0]]))
            assert 0.85 <= oh1 <= 1.15, f"O–H1={oh1:.3f}"
            assert 0.85 <= oh2 <= 1.15, f"O–H2={oh2:.3f}"
            assert 1.20 <= hh <= 1.80, f"H–H={hh:.3f}"

    def test_ethanol_produces_multiple_conformers(self):
        config = AdsorptionConfig(num_conformers=5, seed=42)
        result = create_conformers_from_smiles("CCO", config=config)
        assert result is not None
        conformers, energies = result
        assert len(conformers) >= 1
        assert all(np.isfinite(e) for e in energies)
        for c in conformers:
            assert len(c) == 9
            assert sorted(c.get_chemical_symbols()) == sorted(
                ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
            )
            dmat = _pairwise_distances(c)
            # No unphysical atomic overlap; C–C / C–O should appear in bond window.
            triu = dmat[np.triu_indices(len(c), k=1)]
            assert float(np.min(triu)) > 0.7, f"atoms too close: {np.min(triu):.3f}"
            # At least one C–C (~1.54 Å) and one C–O (~1.43 Å) contact.
            syms = c.get_chemical_symbols()
            cc = [
                dmat[i, j]
                for i in range(len(c))
                for j in range(i + 1, len(c))
                if syms[i] == "C" and syms[j] == "C"
            ]
            co = [
                dmat[i, j]
                for i in range(len(c))
                for j in range(i + 1, len(c))
                if {syms[i], syms[j]} == {"C", "O"}
            ]
            assert any(1.4 <= d <= 1.7 for d in cc), cc
            assert any(1.2 <= d <= 1.6 for d in co), co

    def test_invalid_smiles_returns_none(self):
        config = AdsorptionConfig(num_conformers=3, seed=42)
        result = create_conformers_from_smiles("not_valid!!!", config=config)
        assert result is None

    def test_different_seed_may_differ(self):
        """Different seeds can produce different conformers for flexible molecules."""
        cfg1 = AdsorptionConfig(num_conformers=10, seed=1)
        cfg2 = AdsorptionConfig(num_conformers=10, seed=999)
        r1 = create_conformers_from_smiles("CCCC", config=cfg1)
        r2 = create_conformers_from_smiles("CCCC", config=cfg2)
        assert r1 is not None and r2 is not None
        c1, e1 = r1
        c2, e2 = r2
        assert len(c1) >= 1 and len(c2) >= 1
        assert len(e1) == len(c1) and len(e2) == len(c2)
        # For butane (CCCC), different seeds typically yield different conformers
        pos1 = frozenset(tuple(c.get_positions().flatten().round(6)) for c in c1)
        pos2 = frozenset(tuple(c.get_positions().flatten().round(6)) for c in c2)
        assert pos1 != pos2, (
            "Different seeds (1 vs 999) should produce different conformer sets for CCCC"
        )

    def test_single_atom_smiles(self):
        config = AdsorptionConfig(num_conformers=3, seed=42)
        result = create_conformers_from_smiles("[He]", config=config)
        assert result is not None
        conformers, _ = result
        assert len(conformers) >= 1
        assert len(conformers[0]) == 1

    def test_deduplication_is_applied(self):
        """Water is rigid; 20 conformers should deduplicate to far fewer."""
        config = AdsorptionConfig(
            num_conformers=20,
            seed=42,
            rmsd_dedup_threshold=0.5,
            energy_dedup_threshold=1.0,
        )
        result = create_conformers_from_smiles("O", config=config)
        assert result is not None
        conformers, _ = result
        assert len(conformers) >= 1
        assert len(conformers) < 20, (
            "Water is rigid; deduplication should reduce 20 conformers to fewer"
        )


# ---------------------------------------------------------------------------
# remove_duplicate_conformers
# ---------------------------------------------------------------------------


class TestRemoveDuplicateConformers:
    def _make_conformer(self, offset: float = 0.0, energy: float = 0.0):
        return Atoms(
            "OH2",
            positions=[
                [0.0 + offset, 0.0, 0.0],
                [0.96 + offset, 0.0, 0.24],
                [-0.24 + offset, 0.93, 0.24],
            ],
        ), energy

    def test_single_conformer_unchanged(self):
        c, e = self._make_conformer()
        result_c, result_e = remove_duplicate_conformers([c], [e])
        assert len(result_c) == 1

    def test_empty_list_unchanged(self):
        result_c, result_e = remove_duplicate_conformers([], [])
        assert result_c == []
        assert result_e == []

    def test_identical_conformers_deduplicated(self):
        c1, e1 = self._make_conformer(0.0, 0.0)
        c2, e2 = self._make_conformer(0.0, 0.0)
        result_c, _ = remove_duplicate_conformers(
            [c1, c2], [e1, e2], distance_threshold=0.5, energy_threshold=0.1
        )
        assert len(result_c) == 1

    def test_different_geometry_kept(self):
        """Genuinely different geometry (not just a translation) must survive dedup."""
        c1 = Atoms(
            "OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.24], [-0.24, 0.93, 0.24]]
        )
        c2 = Atoms(
            "OH2", positions=[[0.0, 0.0, 0.0], [0.96, 0.0, -0.50], [-0.24, -1.2, 0.24]]
        )
        result_c, _ = remove_duplicate_conformers(
            [c1, c2], [0.0, 0.0], distance_threshold=0.5, energy_threshold=0.1
        )
        assert len(result_c) == 2

    def test_different_energies_kept(self):
        c1, e1 = self._make_conformer(0.0, 0.0)
        c2, e2 = self._make_conformer(0.0, 5.0)
        result_c, _ = remove_duplicate_conformers(
            [c1, c2], [e1, e2], distance_threshold=0.5, energy_threshold=0.1
        )
        assert len(result_c) == 2

    def test_threshold_boundary(self):
        c1, e1 = self._make_conformer(0.0, 0.0)
        c2, e2 = self._make_conformer(0.0, 0.04)
        result_c, _ = remove_duplicate_conformers(
            [c1, c2], [e1, e2], distance_threshold=10.0, energy_threshold=0.05
        )
        assert len(result_c) == 1

    def test_sorted_by_energy(self):
        c1, e1 = self._make_conformer(0.0, 2.0)
        c2, e2 = self._make_conformer(5.0, 1.0)
        c3, e3 = self._make_conformer(10.0, 3.0)
        result_c, result_e = remove_duplicate_conformers(
            [c1, c2, c3], [e1, e2, e3], distance_threshold=0.5, energy_threshold=0.01
        )
        assert result_e == sorted(result_e)


# ---------------------------------------------------------------------------
# select_conformer_boltzmann
# ---------------------------------------------------------------------------


class TestSelectConformerBoltzmann:
    def _make_conformers(self, n: int = 3):
        conformers = []
        energies = []
        for i in range(n):
            c = Atoms("H", positions=[[float(i), 0.0, 0.0]])
            conformers.append(c)
            energies.append(float(i))
        return conformers, energies

    def test_single_conformer_returned(self):
        c = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        result = select_conformer_boltzmann([c], [0.0])
        assert isinstance(result, Atoms)
        assert np.allclose(result.get_positions(), c.get_positions())

    def test_returns_copy(self):
        c = Atoms("H", positions=[[0.0, 0.0, 0.0]])
        result = select_conformer_boltzmann([c], [0.0])
        assert result is not c

    def test_mismatched_lengths_uses_random(self):
        conformers, _ = self._make_conformers(3)
        result = select_conformer_boltzmann(conformers, [0.0])
        assert isinstance(result, Atoms)

    def test_no_valid_energies_uses_random(self):
        conformers, _ = self._make_conformers(3)
        energies = [float("nan"), float("inf"), float("-inf")]
        result = select_conformer_boltzmann(conformers, energies)
        assert isinstance(result, Atoms)

    def test_deterministic_with_seeded_rng(self):
        conformers, energies = self._make_conformers(5)
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        r1 = select_conformer_boltzmann(conformers, energies, rng=rng1)
        r2 = select_conformer_boltzmann(conformers, energies, rng=rng2)
        assert np.allclose(r1.get_positions(), r2.get_positions())

    def test_different_seeds_may_differ(self):
        conformers, energies = self._make_conformers(10)
        # with spread energies, different seeds should sometimes pick different conformers
        picks = set()
        for seed in range(100):
            rng = random.Random(seed)
            r = select_conformer_boltzmann(
                conformers, energies, temperature=1e6, rng=rng
            )
            picks.add(tuple(r.get_positions().flatten()))
        assert len(picks) > 1, (
            "Different seeds should pick different conformers at high T"
        )

    def test_low_temperature_prefers_lowest_energy(self):
        conformers, energies = self._make_conformers(5)
        picks = []
        for seed in range(50):
            rng = random.Random(seed)
            r = select_conformer_boltzmann(
                conformers, energies, temperature=1.0, rng=rng
            )
            picks.append(r.get_positions()[0, 0])
        # at very low temperature, nearly all picks should be the lowest energy (index 0)
        assert picks.count(0.0) > 40

    def test_all_nan_energies_still_returns(self):
        conformers, _ = self._make_conformers(3)
        energies = [float("nan")] * 3
        result = select_conformer_boltzmann(conformers, energies)
        assert isinstance(result, Atoms)


# ---------------------------------------------------------------------------
# select_conformer_for_placement
# ---------------------------------------------------------------------------


class TestSelectConformerForPlacement:
    def _make_conformers(self, n: int = 4):
        conformers = []
        energies = []
        for i in range(n):
            c = Atoms("H", positions=[[float(i), 0.0, 0.0]])
            conformers.append(c)
            energies.append(float(i))
        return conformers, energies

    def test_cycle_uses_all_conformers(self):
        """cycle sampling ensures all conformers are used across placement_ids."""
        conformers, energies = self._make_conformers(4)
        picked = set()
        for pid in range(8):
            r = select_conformer_for_placement(
                conformers, energies, pid, sampling="cycle"
            )
            picked.add(tuple(r.get_positions().flatten()))
        assert len(picked) == 4, "All 4 conformers should appear in 8 placements"

    def test_cycle_deterministic(self):
        conformers, energies = self._make_conformers(5)
        r1 = select_conformer_for_placement(conformers, energies, 3, sampling="cycle")
        r2 = select_conformer_for_placement(conformers, energies, 3, sampling="cycle")
        assert np.allclose(r1.get_positions(), r2.get_positions())

    def test_boltzmann_fallback(self):
        conformers, energies = self._make_conformers(3)
        r = select_conformer_for_placement(
            conformers, energies, 0, sampling="boltzmann", rng=random.Random(42)
        )
        assert isinstance(r, Atoms)

    def test_single_conformer_returned(self):
        c = Atoms("H", positions=[[1.0, 2.0, 3.0]])
        r = select_conformer_for_placement([c], [0.0], 99, sampling="cycle")
        assert np.allclose(r.get_positions(), c.get_positions())

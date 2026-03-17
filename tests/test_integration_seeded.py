"""Deterministic integration tests: same seed → same output."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement import generate_conformer_placement

from .conftest import make_slab

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# conformer generation determinism
# ---------------------------------------------------------------------------


class TestConformerDeterminism:
    @pytest.mark.parametrize("smiles", ["O", "CCO", "CC(=O)O", "c1ccccc1"])
    def test_same_seed_same_conformers(self, smiles):
        cfg = AdsorptionConfig(num_conformers=5, seed=42)
        r1 = create_conformers_from_smiles(smiles, config=cfg)
        r2 = create_conformers_from_smiles(smiles, config=cfg)
        if r1 is None or r2 is None:
            pytest.skip(f"Cannot generate conformers for {smiles}")
        c1, e1 = r1
        c2, e2 = r2
        assert len(c1) == len(c2)
        for a, b in zip(c1, c2, strict=False):
            assert np.allclose(a.get_positions(), b.get_positions(), atol=1e-10)
        assert np.allclose(e1, e2, atol=1e-10)


# ---------------------------------------------------------------------------
# placement determinism
# ---------------------------------------------------------------------------


class TestPlacementDeterminism:
    def _run_placements(self, seed, n=20):
        """Generate n placements with the given seed and return positions."""
        slab = make_slab()
        cfg = AdsorptionConfig(seed=seed, num_conformers=3, num_placements=n)
        result = create_conformers_from_smiles("O", config=cfg)
        assert result is not None
        conformers, energies = result

        positions = []
        for pid in range(n):
            placed = generate_conformer_placement(
                conformers, energies, slab, pid, config=cfg
            )
            if placed is not None:
                positions.append(placed.get_positions().copy())
            else:
                positions.append(None)
        return positions

    def test_same_seed_same_placements(self):
        pos1 = self._run_placements(seed=42)
        pos2 = self._run_placements(seed=42)
        assert len(pos1) == len(pos2)
        for p1, p2 in zip(pos1, pos2, strict=False):
            if p1 is None:
                assert p2 is None
            else:
                assert np.allclose(p1, p2, atol=1e-10)

    def test_different_seed_different_placements(self):
        pos1 = self._run_placements(seed=1)
        pos2 = self._run_placements(seed=999)
        # at least some successful placements should differ
        diffs = 0
        for p1, p2 in zip(pos1, pos2, strict=False):
            if p1 is not None and p2 is not None and not np.allclose(p1, p2, atol=1e-6):
                diffs += 1
        assert diffs > 0, "Different seeds should produce different placements"


# ---------------------------------------------------------------------------
# end-to-end pipeline determinism (conformers → placement → validation)
# ---------------------------------------------------------------------------


class TestEndToEndDeterminism:
    def _pipeline(self, seed):
        """Run conformer generation + placement + basic validation loop."""
        slab = make_slab()
        cfg = AdsorptionConfig(seed=seed, num_conformers=5, num_placements=30)
        result = create_conformers_from_smiles("CCO", config=cfg)
        assert result is not None
        conformers, energies = result

        placement_results = []
        for pid in range(cfg.num_placements):
            placed = generate_conformer_placement(
                conformers, energies, slab, pid, config=cfg
            )
            if placed is not None:
                placement_results.append(
                    (pid, placed.get_positions().copy(), len(placed))
                )
        return placement_results

    def test_full_pipeline_deterministic(self):
        r1 = self._pipeline(seed=42)
        r2 = self._pipeline(seed=42)
        assert len(r1) == len(r2)
        for (pid1, pos1, n1), (pid2, pos2, n2) in zip(r1, r2, strict=False):
            assert pid1 == pid2
            assert n1 == n2
            assert np.allclose(pos1, pos2, atol=1e-10)

    def test_pipeline_seed_variation(self):
        r1 = self._pipeline(seed=7)
        r2 = self._pipeline(seed=123)
        if len(r1) > 0 and len(r2) > 0:
            # count how many positions differ
            min_len = min(len(r1), len(r2))
            diffs = sum(
                1
                for (_, p1, _), (_, p2, _) in zip(
                    r1[:min_len], r2[:min_len], strict=False
                )
                if not np.allclose(p1, p2, atol=1e-6)
            )
            assert diffs > 0 or len(r1) != len(r2)

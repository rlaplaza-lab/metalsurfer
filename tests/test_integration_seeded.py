"""Deterministic integration tests: same seed → same output."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement import (
    enumerate_placement_specs,
    generate_placement_from_spec,
)

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
        for a, b in zip(c1, c2, strict=True):
            assert np.allclose(a.get_positions(), b.get_positions(), atol=1e-8)
        assert np.allclose(e1, e2, atol=1e-8)


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

        specs = enumerate_placement_specs(conformers, slab, cfg, "O", n_desired=n)
        positions = []
        for spec in specs:
            placed = generate_placement_from_spec(
                spec, conformers, slab, cfg, smiles="O"
            )
            if placed is not None:
                positions.append(placed[0].get_positions().copy())
            else:
                positions.append(None)
        return positions

    def test_same_seed_same_placements(self):
        pos1 = self._run_placements(seed=42)
        pos2 = self._run_placements(seed=42)
        assert len(pos1) == len(pos2)
        for p1, p2 in zip(pos1, pos2, strict=True):
            if p1 is None:
                assert p2 is None
            else:
                assert np.allclose(p1, p2, atol=1e-8)

    def test_different_specs_different_placements(self):
        """Different spec indices produce spatially distinct placements."""
        positions = self._run_placements(seed=42)
        # at least some successful placements at different specs should differ
        valid = [p for p in positions if p is not None]
        assert len(valid) >= 2, "Need at least 2 successful placements"
        diffs = 0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if not np.allclose(valid[i], valid[j], atol=1e-6):
                    diffs += 1
        assert diffs > 0, "Different specs should produce different placements"


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

        specs = enumerate_placement_specs(
            conformers, slab, cfg, "CCO", n_desired=cfg.num_placements
        )
        placement_results = []
        for spec in specs:
            placed = generate_placement_from_spec(
                spec, conformers, slab, cfg, smiles="CCO"
            )
            if placed is not None:
                adsorbate, descriptor = placed
                placement_results.append(
                    (
                        spec.placement_index,
                        adsorbate.get_positions().copy(),
                        len(adsorbate),
                    )
                )
        return placement_results

    def test_full_pipeline_deterministic(self):
        r1 = self._pipeline(seed=42)
        r2 = self._pipeline(seed=42)
        assert len(r1) == len(r2)
        for (pid1, pos1, n1), (pid2, pos2, n2) in zip(r1, r2, strict=True):
            assert pid1 == pid2
            assert n1 == n2
            assert np.allclose(pos1, pos2, atol=1e-8)

    def test_pipeline_seed_variation(self):
        r1 = self._pipeline(seed=7)
        r2 = self._pipeline(seed=123)
        if len(r1) > 0 and len(r2) > 0:
            # count how many positions differ
            min_len = min(len(r1), len(r2))
            diffs = sum(
                1
                for (_, p1, _), (_, p2, _) in zip(
                    r1[:min_len], r2[:min_len], strict=True
                )
                if not np.allclose(p1, p2, atol=1e-6)
            )
            assert diffs > 0 or len(r1) != len(r2)

    def test_descriptor_quaternions_always_finite(self):
        """Every descriptor from the pipeline must have finite quaternion fields."""
        slab = make_slab()
        cfg = AdsorptionConfig(seed=42, num_conformers=3, num_placements=20)
        result = create_conformers_from_smiles("O", config=cfg)
        assert result is not None
        conformers, _ = result
        specs = enumerate_placement_specs(conformers, slab, cfg, "O", n_desired=20)
        n_ok = 0
        for spec in specs:
            placed = generate_placement_from_spec(
                spec, conformers, slab, cfg, smiles="O"
            )
            if placed is None:
                continue
            _, desc = placed
            n_ok += 1
            for field in (
                "quat_w",
                "quat_x",
                "quat_y",
                "quat_z",
                "x_abs",
                "y_abs",
                "z_abs",
            ):
                val = getattr(desc, field)
                assert val is not None, (
                    f"{field} is None for spec {spec.placement_index}"
                )
                assert np.isfinite(val), f"{field}={val} is not finite"
        assert n_ok >= 5, f"Expected >=5 valid placements, got {n_ok}"

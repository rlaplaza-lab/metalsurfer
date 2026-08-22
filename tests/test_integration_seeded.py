"""Deterministic integration tests: same seed → same output.

Also assert physical placement invariants (clearance, height, intramolecular
geometry) so reproducibility is not the only gate that must pass.
"""

import math

import numpy as np
import pytest

from metalsurfer._numeric_defaults import (
    MIN_CONTACT_RATIO_DEFAULT,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from metalsurfer.config import AdsorptionConfig
from metalsurfer.conformers import create_conformers_from_smiles
from metalsurfer.placement import (
    check_initial_placement_distance,
    enumerate_placement_specs,
    generate_placement_from_spec,
)
from metalsurfer.placement.geometry import calculate_min_distance

from .conftest import assert_no_intramolecular_clashes, make_slab

pytestmark = pytest.mark.integration


def _assert_physical_placement(adsorbate, slab, *, material_type: str = "slab") -> None:
    """Shared physics gates for successful initial placements."""
    ok, min_d, reason = check_initial_placement_distance(
        adsorbate,
        slab,
        min_distance=MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type=material_type,
    )
    assert ok, f"placement fails distance gate: min_d={min_d:.3f} reason={reason}"
    assert min_d >= MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM * MIN_CONTACT_RATIO_DEFAULT

    cell = np.asarray(slab.get_cell(), dtype=float)
    pbc = list(slab.get_pbc())
    # Adsorbate must sit above the top-layer atoms (slab normal ≈ +z here).
    ads_z = adsorbate.get_positions()[:, 2]
    slab_z = slab.get_positions()[:, 2]
    assert float(np.min(ads_z)) > float(np.max(slab_z)) - 0.05, (
        f"adsorbate not above surface: min_ads_z={ads_z.min():.3f}, "
        f"max_slab_z={slab_z.max():.3f}"
    )

    assert_no_intramolecular_clashes(adsorbate, slab)

    # Explicit MIC adsorbate–slab clearance via the public distance helper.
    mic_min = calculate_min_distance(
        adsorbate.get_positions(),
        slab.get_positions(),
        cell=cell,
        use_pbc=True,
        pbc=[True, True, False] if material_type == "slab" else list(pbc),
    )
    assert mic_min == pytest.approx(min_d, abs=1e-6)


# ---------------------------------------------------------------------------
# conformer generation determinism
# ---------------------------------------------------------------------------


class TestConformerDeterminism:
    @pytest.fixture(autouse=True)
    def _require_rdkit(self):
        pytest.importorskip("rdkit")

    @pytest.mark.parametrize("smiles", ["O", "CCO", "CC(=O)O", "c1ccccc1"])
    def test_same_seed_same_conformers(self, smiles):
        cfg = AdsorptionConfig(num_conformers=5, seed=42)
        r1 = create_conformers_from_smiles(smiles, config=cfg)
        r2 = create_conformers_from_smiles(smiles, config=cfg)
        assert r1 is not None, f"conformer generation failed for {smiles}"
        assert r2 is not None, f"conformer generation failed for {smiles}"
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
        n_pairs = len(valid) * (len(valid) - 1) // 2
        diffs = 0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                if not np.allclose(valid[i], valid[j], atol=1e-6):
                    diffs += 1
        assert diffs >= max(1, n_pairs // 2), (
            f"Different specs should usually produce different placements "
            f"({diffs}/{n_pairs} differing pairs)"
        )


# ---------------------------------------------------------------------------
# end-to-end pipeline determinism (conformers → placement → validation)
# ---------------------------------------------------------------------------


class TestEndToEndDeterminism:
    def _pipeline(self, seed, *, assert_physics: bool = False):
        """Run conformer generation + placement (+ optional physics gates)."""
        slab = make_slab()
        cfg = AdsorptionConfig(seed=seed, num_conformers=5, num_placements=30)
        result = create_conformers_from_smiles("CCO", config=cfg)
        assert result is not None
        conformers, energies = result
        assert len(energies) == len(conformers)
        assert all(np.isfinite(e) for e in energies)

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
                if assert_physics:
                    _assert_physical_placement(adsorbate, slab)
                    assert descriptor.z_abs is not None
                    assert descriptor.surface_ref_z_abs is not None
                    assert descriptor.z_abs > float(np.max(slab.get_positions()[:, 2]))
                    assert (
                        float(descriptor.z_abs)
                        >= float(descriptor.surface_ref_z_abs) - 0.05
                    )
                    assert 0.0 <= float(descriptor.z_fraction) <= 1.0
                    assert descriptor.orientation_type in {
                        "round",
                        "parallel",
                        "en_down",
                        "vertical",
                    }
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

    def test_pipeline_placements_are_physically_plausible(self):
        """Successful CCO placements clear the surface and keep intramolecular bonds."""
        results = self._pipeline(seed=42, assert_physics=True)
        n_requested = 30
        min_ok = max(27, int(math.ceil(0.9 * n_requested)))
        assert len(results) >= min_ok, (
            f"Expected >= {min_ok}/{n_requested} valid placements, got {len(results)}"
        )
        # Ethanol has 9 atoms; every successful placement must preserve stoichiometry.
        assert all(n == 9 for _, _, n in results)

    def test_pipeline_seed_variation(self):
        r1 = self._pipeline(seed=7)
        r2 = self._pipeline(seed=123)
        min_ok = max(18, int(math.ceil(0.9 * 30)))
        assert len(r1) >= min_ok, f"seed=7 yield too low: {len(r1)}"
        assert len(r2) >= min_ok, f"seed=123 yield too low: {len(r2)}"
        min_len = min(len(r1), len(r2))
        diffs = sum(
            1
            for (_, p1, _), (_, p2, _) in zip(r1[:min_len], r2[:min_len], strict=True)
            if not np.allclose(p1, p2, atol=1e-6)
        )
        assert diffs >= 1, "Different seeds should produce distinct placements"

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
            adsorbate, desc = placed
            _assert_physical_placement(adsorbate, slab)
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
            # Unit quaternion (up to floating-point noise).
            qnorm = np.sqrt(
                desc.quat_w**2 + desc.quat_x**2 + desc.quat_y**2 + desc.quat_z**2
            )
            assert qnorm == pytest.approx(1.0, abs=1e-5)
        min_ok = max(18, int(math.ceil(0.9 * 20)))
        assert n_ok >= min_ok, f"Expected >= {min_ok}/20 valid placements, got {n_ok}"

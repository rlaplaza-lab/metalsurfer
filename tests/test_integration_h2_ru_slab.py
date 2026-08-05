"""Integration test: H2 on Ru(0001) – dissociative placements and geometries."""

import math

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import setup_single_model
from metalsurfer.surface_prep import prepare_substrate
from metalsurfer.workflow import (
    calculate_reference_energies,
    process_molecule,
)
from tests.conftest import GPU_MLIP_MARKS, adsorbate_symbol_pair_distance

pytestmark = GPU_MLIP_MARKS

_GPU_AUTOBATCH = {
    "autobatcher_max_memory_padding": 0.8,
    "autobatcher_max_memory_scaler": 500,
    "autobatcher_max_atoms_to_try": 5000,
}


def _hh_bond_length(atoms, slab_size: int) -> float:
    """H–H distance in adsorbate (H2 has exactly 2 H atoms)."""
    return adsorbate_symbol_pair_distance(atoms, slab_size, "H")


def _run_h2_on_ru():
    """Create Ru(0001) slab and run a single H2 screening flow."""
    num_placements = 10
    # Match demo near-defaults: dissociative hollow pairs + skip connectivity.
    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=1,
        num_placements=num_placements,
        device="cuda",
        enable_dissociative_placement=True,
        skip_topology_check=True,
        **_GPU_AUTOBATCH,
    )
    assert config.stage1_steps == 50
    assert config.stage2_steps == 150
    assert config.skip_desorption_check is False

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(2, 2, 1),
        config=config,
        results_dir="results_test_h2_ru_slab",
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab, calculator, ["H2"], ["[H][H]"], ts_model=ts_model, config=config
    )
    results = process_molecule(
        "[H][H]",
        "H2",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="h2_ru_slab",
    ).results
    return results, num_placements


class TestH2OnRu0001:
    def test_h2_ru_dissociative_placements_and_geometries(self):
        results, num_placements = _run_h2_on_ru()
        min_ok = max(5, int(math.ceil(0.5 * num_placements)))
        assert len(results) >= min_ok, (
            f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
        )

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(np.isfinite(e_ads))
        assert float(e_ads.min()) < 0.0, (
            f"Best E_ads should be negative for H2 on Ru, got {e_ads}"
        )
        assert np.all(e_ads < 1.0), (
            f"E_ads should stay below 1 eV for H2 on Ru, got {e_ads}"
        )
        assert np.all(e_ads >= -2.5), (
            f"E_ads should be >= -2.5 eV for H2 on Ru, got min {e_ads.min():.3f}"
        )

        site_ids = set()
        for r in results:
            assert r.energy_adsorption == pytest.approx(
                r.energy_adslab - r.energy_slab - r.energy_adsorbate,
                abs=1e-4,
            )
            assert r.placement_descriptor.orientation_type == "dissociative"
            assert r.placement_descriptor.site_source == "dissociative_hollow_pair"
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            if r.placement_descriptor.site_index is not None:
                site_ids.add(int(r.placement_descriptor.site_index))

        if len(results) >= 2:
            assert len(site_ids) >= 2, f"Expected multi-site coverage, got {site_ids}"

        slab_size = len(results[0].atoms) - 2
        hh_lengths = [_hh_bond_length(r.atoms, slab_size) for r in results]
        # Initial placements are dissociative hollow pairs; UMA on Ru(0001) often
        # recombines to molecular H₂ (~0.75 Å). Reject mid-bond mush.
        for hh in hh_lengths:
            assert (0.7 <= hh <= 0.9) or (1.5 <= hh <= 4.0), (
                f"H–H should be molecular or dissociated, got {hh:.3f} "
                f"(all={hh_lengths})"
            )

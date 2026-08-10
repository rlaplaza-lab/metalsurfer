"""Integration test: ethene on Ru(0001) – negative E_ads, reasonable geometries."""

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


def _cc_bond_length(atoms, slab_size: int) -> float:
    """C–C distance in adsorbate (ethene has exactly 2 C atoms)."""
    return adsorbate_symbol_pair_distance(atoms, slab_size, "C")


def _run_ethene_on_ru():
    """Create Ru(0001) slab and run a single ethene screening flow."""
    num_placements = 12
    # Near-default slab campaign (default stage steps / gates); modest N for CI time.
    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=3,
        num_placements=num_placements,
        device="cuda",
        **_GPU_AUTOBATCH,
    )
    assert config.stage1_steps == 50
    assert config.stage2_steps == 150
    assert config.skip_topology_check is False
    assert config.skip_desorption_check is False

    slab = prepare_substrate(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(2, 2, 1),
        config=config,
        results_dir="results_test_ethene",
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab, calculator, ["ethene"], ["C=C"], ts_model=ts_model, config=config
    )
    results = process_molecule(
        "C=C",
        "ethene",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="Ru_001",
    ).results
    return results, num_placements


class TestEtheneOnRu0001:
    def test_ethene_ru_negative_adsorption_energies_and_reasonable_geometries(
        self, workdir
    ):
        results, num_placements = _run_ethene_on_ru()
        min_ok = max(6, int(math.ceil(0.5 * num_placements)))
        assert len(results) >= min_ok, (
            f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
        )

        e_ads = np.array([r.energy_adsorption for r in results])
        assert e_ads.min() < 0, (
            f"Best E_ads should be negative (favorable binding), got min {e_ads.min():.3f}"
        )
        assert np.median(e_ads) < 0, (
            f"Median E_ads should be negative, got {np.median(e_ads):.3f}; all: {e_ads}"
        )
        assert np.all(e_ads < 0.5), (
            f"E_ads should stay below 0.5 eV for ethene on Ru, got {e_ads}"
        )
        assert np.all(e_ads >= -2.5), (
            f"E_ads should be >= -2.5 eV for ethene on Ru, got min {e_ads.min():.3f}"
        )

        spread = float(e_ads.max() - e_ads.min())
        assert spread >= 0.03, (
            f"Expected distribution of E_ads (spread >= 0.03 eV), got spread {spread:.4f}"
        )

        slab_size = len(results[0].atoms) - 6  # ethene C2H4
        for r in results:
            assert r.energy_adsorption == pytest.approx(
                r.energy_adslab - r.energy_slab - r.energy_adsorbate,
                abs=1e-4,
            )
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            ads = r.atoms[r.slab_size :]
            assert len(ads) == 6
            assert sorted(ads.get_chemical_symbols()) == ["C", "C", "H", "H", "H", "H"]
            assert r.placement_descriptor is not None
            assert r.placement_descriptor.surface_ref_z_abs is not None
            cc = _cc_bond_length(r.atoms, slab_size)
            assert 1.25 <= cc <= 1.50, (
                f"C=C bond length should be ~1.34 Å (1.25–1.50), got {cc:.3f}"
            )

"""Integration test: ethene on Ru(0001) – negative E_ads, reasonable geometries."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import setup_single_model
from metalsurfer.surfaces import create_slab_from_bulk
from metalsurfer.workflow import (
    calculate_reference_energies,
    process_molecule,
)
from tests.optional_deps import cuda_available, has_mlip_stack

pytestmark = [
    pytest.mark.slow,
    pytest.mark.mlip,
    pytest.mark.gpu,
    pytest.mark.no_fork,  # CUDA incompatible with pytest-forked
    pytest.mark.skipif(
        not has_mlip_stack,
        reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
    ),
    pytest.mark.skipif(
        not cuda_available,
        reason="CUDA GPU required; skipped in CI (no GPU)",
    ),
]


def _cc_bond_length(atoms, slab_size: int) -> float:
    """C–C distance in adsorbate (ethene has exactly 2 C atoms)."""
    ads = atoms[slab_size:]
    syms = ads.get_chemical_symbols()
    c_indices = [i for i, s in enumerate(syms) if s == "C"]
    if len(c_indices) != 2:
        return float("nan")
    pos = ads.get_positions()
    cell = atoms.get_cell()
    d = pos[c_indices[1]] - pos[c_indices[0]]
    d = d - np.round(d @ np.linalg.inv(cell)) @ cell
    return float(np.linalg.norm(d))


def _run_ethene_on_ru():
    """Create Ru(0001) slab and run a single ethene screening flow."""
    slab = create_slab_from_bulk(
        bulk_id="mp-33",
        miller_indices=(0, 0, 1),
        supercell=(2, 2, 1),
        results_dir="results_test_ethene",
    )
    config = AdsorptionConfig(
        material_type="slab",
        seed=42,
        num_conformers=3,
        num_placements=25,
        device="cuda",
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab, calculator, ["ethene"], ["C=C"], ts_model=ts_model, config=config
    )
    return process_molecule(
        "C=C",
        "ethene",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="Ru_001",
    )


class TestEtheneOnRu0001:
    def test_ethene_ru_negative_adsorption_energies_and_reasonable_geometries(self):
        results = _run_ethene_on_ru()
        assert len(results) >= 3, f"Expected >= 3 valid placements, got {len(results)}"

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(e_ads < 0), f"All E_ads should be negative, got {e_ads}"
        assert np.all(e_ads >= -2.5), (
            f"E_ads should be >= -2.5 eV for ethene on Ru, got min {e_ads.min():.3f}"
        )

        spread = float(e_ads.max() - e_ads.min())
        assert spread >= 0.03, (
            f"Expected distribution of E_ads (spread >= 0.03 eV), got spread {spread:.4f}"
        )

        slab_size = len(results[0].atoms) - 6  # ethene C2H4
        for r in results:
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            cc = _cc_bond_length(r.atoms, slab_size)
            assert 1.1 <= cc <= 1.6, (
                f"C=C bond length should be ~1.34 Å (1.1–1.6), got {cc:.3f}"
            )

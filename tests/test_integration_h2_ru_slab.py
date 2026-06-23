"""Integration test: H2 on Ru(0001) – dissociative placements and geometries."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import setup_single_model
from metalsurfer.surface_prep import prepare_substrate
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


def _hh_bond_length(atoms, slab_size: int) -> float:
    """H–H distance in adsorbate (H2 has exactly 2 H atoms)."""
    ads = atoms[slab_size:]
    syms = ads.get_chemical_symbols()
    h_indices = [i for i, s in enumerate(syms) if s == "H"]
    if len(h_indices) != 2:
        return float("nan")
    pos = ads.get_positions()
    d = pos[h_indices[1]] - pos[h_indices[0]]
    if np.any(atoms.get_pbc()):
        cell = atoms.get_cell()
        d = d - np.round(d @ np.linalg.inv(cell)) @ cell
    return float(np.linalg.norm(d))


def _run_h2_on_ru():
    """Create Ru(0001) slab and run a single H2 screening flow."""
    config = AdsorptionConfig(
        material_type="slab",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,
        num_placements=10,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )
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
    return process_molecule(
        "[H][H]",
        "H2",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="h2_ru_slab",
    )


class TestH2OnRu0001:
    def test_h2_ru_dissociative_placements_and_geometries(self):
        results = _run_h2_on_ru()
        assert len(results) >= 1, f"Expected >= 1 valid placement, got {len(results)}"

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(np.isfinite(e_ads))
        assert np.all(e_ads < 5.0), (
            f"E_ads should stay in a smoke window (< 5 eV), got {e_ads}"
        )

        for r in results:
            assert r.placement_descriptor.orientation_type == "dissociative"
            assert r.placement_descriptor.site_source == "dissociative_hollow_pair"
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )

        slab_size = len(results[0].atoms) - 2
        hh_lengths = [_hh_bond_length(r.atoms, slab_size) for r in results]
        assert all(0.7 <= hh <= 5.0 for hh in hh_lengths), (
            f"H–H separation should be molecular or dissociated, got {hh_lengths}"
        )

"""Integration test: CO2 in MOF – negative E_ads, reasonable geometries."""

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


def _co_bond_length(atoms, slab_size: int) -> float:
    """C–O distance in adsorbate (CO2 has exactly 2 C–O bonds)."""
    ads = atoms[slab_size:]
    syms = ads.get_chemical_symbols()
    c_indices = [i for i, s in enumerate(syms) if s == "C"]
    o_indices = [i for i, s in enumerate(syms) if s == "O"]
    if len(c_indices) != 1 or len(o_indices) != 2:
        return float("nan")
    pos = ads.get_positions()
    cell = atoms.get_cell()
    d1 = pos[o_indices[0]] - pos[c_indices[0]]
    d1 = d1 - np.round(d1 @ np.linalg.inv(cell)) @ cell
    d2 = pos[o_indices[1]] - pos[c_indices[0]]
    d2 = d2 - np.round(d2 @ np.linalg.inv(cell)) @ cell
    return float(np.linalg.norm(d1)), float(np.linalg.norm(d2))


def _run_co2_in_mof():
    """Load MOF structure and run a single CO2 screening flow."""
    import os

    from ase.io import read

    cif_path = os.path.join("examples", "mof_structures", "RUBTAK01.cif")
    mof_atoms = read(cif_path)
    config = AdsorptionConfig(
        material_type="porous",
        slab_relaxation_mode="none",
        seed=42,
        num_conformers=1,
        num_placements=5,
        device="cuda",
        skip_topology_check=False,
        skip_desorption_check=False,
        stage1_steps=16,
        stage2_steps=80,
        placement_z_range=(2.0, 6.0),
        min_initial_distance=1.8,
    )
    mof_slab = prepare_substrate(
        slab=mof_atoms,
        config=config,
        results_dir="results_test_co2_mof",
        align=False,
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        mof_slab, calculator, ["CO2"], ["O=C=O"], ts_model=ts_model, config=config
    )
    return process_molecule(
        "O=C=O",
        "CO2",
        mof_slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="co2_mof",
    )


class TestCO2InMOF:
    def test_co2_mof_negative_adsorption_energies_and_reasonable_geometries(self):
        results = _run_co2_in_mof()
        assert len(results) >= 3, f"Expected >= 3 valid placements, got {len(results)}"

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(e_ads < 0), f"All E_ads should be negative, got {e_ads}"
        assert np.all(e_ads >= -10.0), (
            f"E_ads should be >= -10.0 eV for CO2 in MOF, got min {e_ads.min():.3f}"
        )

        spread = float(e_ads.max() - e_ads.min())
        assert spread >= 0.03, (
            f"Expected distribution of E_ads (spread >= 0.03 eV), got spread {spread:.4f}"
        )

        slab_size = len(results[0].atoms) - 3  # CO2
        for r in results:
            assert 1.5 <= r.distance <= 6.0, (
                f"Adsorbate–surface distance should be 1.5–6 Å, got {r.distance:.2f}"
            )
            co1, co2 = _co_bond_length(r.atoms, slab_size)
            assert 1.1 <= co1 <= 1.4, (
                f"C–O bond length should be ~1.16 Å (1.1–1.4), got {co1:.3f}"
            )
            assert 1.1 <= co2 <= 1.4, (
                f"C–O bond length should be ~1.16 Å (1.1–1.4), got {co2:.3f}"
            )

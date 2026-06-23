"""Integration test: H2 on Pt12 nanocluster – pipeline smoke test and geometries.

Hand-built Pt₁₂ skips unrestricted prep relaxation (``slab_relaxation_mode="none"``).
H2 may adsorb dissociatively on Pt (large H–H separation is valid). UMA E_ads on
clusters are qualitative smoke-test signals rather than strict thermochemistry; many
initial placements can relax to the same dissociative minimum and deduplicate to one
unique configuration.
"""

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


def _run_h2_on_pt12():
    """Create Pt12 nanocluster and run a single H2 screening flow."""
    from ase import Atoms

    pt_atoms = Atoms(
        symbols=["Pt"] * 12,
        positions=[
            # Bottom layer (z=0)
            [0.0, 0.0, 0.0],
            [2.8, 0.0, 0.0],
            [1.4, 2.425, 0.0],
            [4.2, 2.425, 0.0],
            # Middle layer (z=2.0)
            [1.4, 0.808, 2.0],
            [4.2, 0.808, 2.0],
            [0.0, 2.425, 2.0],
            [2.8, 2.425, 2.0],
            # Top layer (z=4.0)
            [1.4, 1.617, 4.0],
            [4.2, 1.617, 4.0],
            [0.0, 0.808, 4.0],
            [2.8, 0.808, 4.0],
        ],
        cell=[20, 20, 20],
        pbc=False,
    )

    config = AdsorptionConfig(
        material_type="nanoparticle",
        model_name="uma-s-1p2",
        seed=42,
        num_conformers=1,
        num_placements=5,
        autobatcher_max_memory_padding=0.8,
        autobatcher_max_memory_scaler=500,
        autobatcher_max_atoms_to_try=5000,
        device="cuda",
        slab_relaxation_mode="none",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=50,
        stage2_steps=500,
    )

    nanocluster = prepare_substrate(
        slab=pt_atoms,
        config=config,
        results_dir="results_test_h2_pt12",
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        nanocluster, calculator, ["H2"], ["[H][H]"], ts_model=ts_model, config=config
    )
    return process_molecule(
        "[H][H]",
        "H2",
        nanocluster,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="h2_pt12",
    )


class TestH2OnPt12:
    def test_h2_pt12_pipeline_smoke_and_reasonable_geometries(self):
        results = _run_h2_on_pt12()
        assert len(results) >= 1, f"Expected >= 1 valid placement, got {len(results)}"

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(np.isfinite(e_ads))
        assert np.all(e_ads < 2.0), (
            f"E_ads should stay in a weak-binding smoke window (< 2 eV), got {e_ads}"
        )
        assert np.all(e_ads >= -5.0), (
            f"E_ads should be >= -5.0 eV for H2 on Pt12, got min {e_ads.min():.3f}"
        )

        if len(results) >= 2:
            spread = float(e_ads.max() - e_ads.min())
            assert spread >= 0.01, (
                f"Expected distinct E_ads when multiple unique configs remain, "
                f"got spread {spread:.4f}"
            )

        slab_size = len(results[0].atoms) - 2  # H2
        for r in results:
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            hh = _hh_bond_length(r.atoms, slab_size)
            # Molecular (~0.74 Å) or dissociated H on Pt are both valid.
            assert 0.7 <= hh <= 5.0, (
                f"H–H separation should be molecular or dissociated on cluster, got {hh:.3f}"
            )

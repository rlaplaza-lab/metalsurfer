"""Integration test: H2 on Pt12 nanocluster – negative E_ads, reasonable geometries."""

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
        seed=42,
        num_conformers=1,
        num_placements=5,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        placement_z_range=(1.5, 4.0),
        min_initial_distance=1.5,
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
    def test_h2_pt12_negative_adsorption_energies_and_reasonable_geometries(self):
        results = _run_h2_on_pt12()
        assert len(results) >= 3, f"Expected >= 3 valid placements, got {len(results)}"

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(e_ads < 0), f"All E_ads should be negative, got {e_ads}"
        assert np.all(e_ads >= -5.0), (
            f"E_ads should be >= -5.0 eV for H2 on relaxed Pt12, got min {e_ads.min():.3f}"
        )

        spread = float(e_ads.max() - e_ads.min())
        assert spread >= 0.03, (
            f"Expected distribution of E_ads (spread >= 0.03 eV), got spread {spread:.4f}"
        )

        slab_size = len(results[0].atoms) - 2  # H2
        for r in results:
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            hh = _hh_bond_length(r.atoms, slab_size)
            # H2 may dissociate on Pt with topology checks disabled; allow
            # either molecular (~0.74 Å) or dissociated pair separations.
            assert 0.7 <= hh <= 5.0, (
                f"H–H separation should be molecular or dissociated on cluster, got {hh:.3f}"
            )

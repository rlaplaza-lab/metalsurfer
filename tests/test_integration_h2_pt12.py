"""Integration test: H2 on Pt12 nanocluster – pipeline smoke test and geometries.

Hand-built Pt₁₂ skips unrestricted prep relaxation (``slab_relaxation_mode="none"``).
H2 may adsorb dissociatively on Pt (large H–H separation is valid). UMA E_ads on
clusters are qualitative smoke-test signals rather than strict thermochemistry; many
initial placements can relax to the same dissociative minimum and deduplicate to one
unique configuration.
"""

import math

import numpy as np
import pytest
from ase import Atoms

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


def _run_h2_on_pt12():
    """Create Pt12 nanocluster and run a single H2 screening flow."""
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

    num_placements = 5
    config = AdsorptionConfig(
        material_type="nanoparticle",
        seed=42,
        num_conformers=1,
        num_placements=num_placements,
        device="cuda",
        slab_relaxation_mode="none",
        enable_dissociative_placement=True,
        skip_topology_check=True,
        **_GPU_AUTOBATCH,
    )
    assert config.stage1_steps == 50
    assert config.stage2_steps == 150

    nanocluster = prepare_substrate(
        slab=pt_atoms,
        config=config,
        results_dir="results_test_h2_pt12",
    )
    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        nanocluster, calculator, ["H2"], ["[H][H]"], ts_model=ts_model, config=config
    )
    results = process_molecule(
        "[H][H]",
        "H2",
        nanocluster,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="h2_pt12",
    ).results
    return results, num_placements


class TestH2OnPt12:
    def test_h2_pt12_pipeline_smoke_and_reasonable_geometries(self):
        results, num_placements = _run_h2_on_pt12()
        # Dedup can collapse equivalent dissociative minima; still require a
        # non-trivial survivor fraction of the requested budget.
        min_ok = max(2, int(math.ceil(0.4 * num_placements)))
        assert len(results) >= min_ok, (
            f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
        )

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(np.isfinite(e_ads))
        assert float(e_ads.min()) < 0.5, (
            f"Best E_ads should be near-binding for H2 on Pt12, got {e_ads}"
        )
        # NP hollow-pair sampling can retain a weakly unbound local minimum;
        # keep a generous ceiling while requiring a clearly binding best.
        assert np.all(e_ads < 1.5), (
            f"E_ads should stay below a weak-binding ceiling (< 1.5 eV), got {e_ads}"
        )
        assert np.all(e_ads >= -3.5), (
            f"E_ads should be >= -3.5 eV for H2 on Pt12, got min {e_ads.min():.3f}"
        )

        if len(results) >= 2:
            spread = float(e_ads.max() - e_ads.min())
            assert spread >= 0.01, (
                f"Expected distinct E_ads when multiple unique configs remain, "
                f"got spread {spread:.4f}"
            )

        slab_size = len(results[0].atoms) - 2  # H2
        for r in results:
            assert r.energy_adsorption == pytest.approx(
                r.energy_adslab - r.energy_slab - r.energy_adsorbate,
                abs=1e-4,
            )
            assert r.placement_descriptor is not None
            assert r.placement_descriptor.orientation_type == "dissociative"
            assert 1.5 <= r.distance <= 4.0, (
                f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
            )
            hh = _hh_bond_length(r.atoms, slab_size)
            # Molecular (~0.74 Å) or dissociated H on Pt are both valid.
            assert (0.7 <= hh <= 0.9) or (1.5 <= hh <= 4.0), (
                f"H–H should be molecular or dissociated on cluster, got {hh:.3f}"
            )

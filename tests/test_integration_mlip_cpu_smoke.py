"""Thin CPU integration smoke: real MLIP via process_molecule."""

import numpy as np
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import setup_single_model
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow import calculate_reference_energies, process_molecule
from tests.conftest import MLIP_CPU_MARKS, make_slab

pytestmark = MLIP_CPU_MARKS


def test_process_molecule_water_on_small_slab_cpu():
    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        material_type="slab",
        seed=42,
        num_conformers=1,
        num_placements=2,
        device="cpu",
        stage1_steps=8,
        stage2_steps=8,
        reference_optimization_steps=8,
        optimize_isolated_sequentially=True,
        skip_topology_check=True,
        max_force_convergence=1.0,
        slab_relaxation_mode="none",
    )
    slab = SlabContainer(make_slab(nx=4, ny=4, n_layers=2))

    calculator, ts_model = setup_single_model(config.model_name, config.device)
    ref = calculate_reference_energies(
        slab,
        calculator,
        ["water"],
        ["O"],
        ts_model=ts_model,
        config=config,
    )
    assert np.isfinite(ref.slab_energy)

    outcome = process_molecule(
        "O",
        "water",
        slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="cpu_smoke",
        skip_workload_autotune=True,
    )
    results = outcome.results
    assert len(results) >= 1

    for r in results:
        assert np.isfinite(r.energy_adsorption)
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=1e-3,
        )

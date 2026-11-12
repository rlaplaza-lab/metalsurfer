"""Integration test: water on Cu(111) via run_adsorption(campaign API + MLIP)."""

import numpy as np
import pytest
from ase.build import fcc111

from metalsurfer.campaigns import run_adsorption
from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement.geometry import detect_vdw_overlaps
from metalsurfer.surface_prep import apply_surface_constraints, prepare_substrate
from metalsurfer.workflow import calculate_reference_energies, process_molecule
from tests.conftest import (
    E_ADS_MLIP_TOL,
    GPU_AUTOBATCH,
    assert_water_oh_hh_geometry,
)
from tests.mlip_marks import GPU_MLIP_MARKS, load_uma

pytestmark = GPU_MLIP_MARKS


@pytest.mark.gpu_smoke
def test_run_adsorption_water_on_cu111(workdir):
    """Campaign-level e2e: prepare_substrate → run_adsorption for water/Cu."""
    # Height-shell stratification samples related poses on few sites; a short
    # draw can collapse under energy/RMSD dedup, so keep enough budget for
    # unique survivors while still locking yield and E_ads.
    num_placements = 16
    config = AdsorptionConfig(
        seed=42,
        num_conformers=1,
        num_placements=num_placements,
        device="cuda",
        min_pbc_image_separation=4.5,
        slab_relaxation_mode="none",
        **GPU_AUTOBATCH,
    )
    assert config.stage1_steps == 50
    assert config.stage2_steps == 150
    assert config.skip_topology_check is False
    assert config.skip_desorption_check is False
    assert config.max_force_convergence == 0.05

    slab = prepare_substrate(
        slab=apply_surface_constraints(
            fcc111("Cu", size=(3, 3, 3), a=3.6, vacuum=10.0), material_type="slab"
        ),
        config=config,
        results_dir="results_test_water_cu_slab",
        material_type="slab",
    )
    campaign = run_adsorption(
        substrate=slab,
        molecules=[("O", "water")],
        config=config,
        surface_type="water_cu_slab",
        skip_existing=False,
        save_results=True,
        write_settings=True,
    )

    assert campaign.n_molecules == 1
    assert len(campaign.run_results) == 1
    results = campaign.run_results[0].results
    # Dedup collapses height-shell near-duplicates; keep a floor of unique hits.
    min_ok = 6
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )
    assert campaign.total_configurations == len(results)

    e_ads = np.array([r.energy_adsorption for r in results])
    assert np.all(np.isfinite(e_ads))
    # QC (uma-s-1p2 + oc25): E_ads in about [-0.45, -0.28], median ~−0.35.
    assert float(e_ads.min()) < -0.35, (
        f"Best E_ads regression lock (<-0.35 eV) failed for water on Cu, got {e_ads}"
    )
    # Height-shell draws include vdW standoffs; require a binding majority.
    n_binding = int(np.sum(e_ads < -0.25))
    assert n_binding >= max(3, len(e_ads) // 2), (
        f"Expected a binding majority (E_ads < -0.25 eV), got {n_binding}/{len(e_ads)} "
        f"from {e_ads}"
    )
    assert np.all(e_ads < 0.0), f"All E_ads should be favorable (<0 eV), got {e_ads}"
    assert np.all(e_ads >= -0.55), (
        f"E_ads should be >= -0.55 eV for water on Cu, got min {e_ads.min():.3f}"
    )

    site_ids = set()
    for r in results:
        assert r.energy_adsorption == pytest.approx(
            r.energy_adslab - r.energy_slab - r.energy_adsorbate,
            abs=E_ADS_MLIP_TOL,
        )
        assert 1.5 <= r.distance <= 4.0, (
            f"Adsorbate–surface distance should be 1.5–4 Å, got {r.distance:.2f}"
        )
        ads = r.atoms[r.slab_size :]
        slab_part = r.atoms[: r.slab_size]
        assert len(ads) == 3
        assert sorted(ads.get_chemical_symbols()) == ["H", "H", "O"]
        overlaps, _ = detect_vdw_overlaps(
            ads, slab_part, material_type="slab", vdw_scale=0.5
        )
        assert len(overlaps) == 0, f"hard VDW clash after relaxation: {overlaps[:3]}"
        desc = r.placement_descriptor
        assert desc is not None
        assert desc.orientation_type == "round"
        assert desc.surface_ref_z_abs is not None and desc.z_abs is not None
        assert float(desc.z_abs) >= float(desc.surface_ref_z_abs) - 0.05
        # Intact water should sit above the Cu surface.
        assert (
            float(np.min(ads.get_positions()[:, 2]))
            > float(np.max(slab_part.get_positions()[:, 2])) - 0.05
        )
        assert_water_oh_hh_geometry(ads)
        if desc.site_index is not None:
            site_ids.add(int(desc.site_index))

    assert len(site_ids) >= 2, f"Expected multi-site coverage, got {site_ids}"

    results_dir = workdir / "results_water_cu_slab"
    assert (results_dir / "adsorption_energies_detailed.csv").is_file()
    assert (results_dir / "run_metadata.json").is_file()


def test_water_cu_run_twice_reproducible(workdir):
    """Same seed twice → E_ads within GPU noise (atol 1e-2 eV)."""
    num_placements = 4
    config = AdsorptionConfig(
        seed=42,
        num_conformers=1,
        num_placements=num_placements,
        device="cuda",
        min_pbc_image_separation=4.5,
        slab_relaxation_mode="none",
        **GPU_AUTOBATCH,
    )
    slab = prepare_substrate(
        slab=apply_surface_constraints(
            fcc111("Cu", size=(3, 3, 3), a=3.6, vacuum=10.0), material_type="slab"
        ),
        config=config,
        results_dir="results_water_cu_repro",
        material_type="slab",
    )

    calculator, ts_model = load_uma(config.model_name, config.device, config.task_name)
    reference = calculate_reference_energies(
        slab,
        calculator,
        ["water"],
        ["O"],
        ts_model=ts_model,
        config=config,
    )

    def _run():
        outcome = process_molecule(
            "O",
            "water",
            slab,
            calculator,
            reference,
            ts_model=ts_model,
            config=config,
            surface_type="water_cu_repro",
        )
        return outcome.results

    first = _run()
    second = _run()
    assert first and second
    # Sampling is seed-deterministic; GPU relaxation can land in nearby basins.
    first_keys = {
        int(result.placement_id): (
            int(result.placement_descriptor.site_index),
            round(float(result.placement_descriptor.z_fraction), 5),
            float(result.placement_descriptor.azimuth_deg),
            float(result.placement_descriptor.tilt_deg),
        )
        for result in first
    }
    second_keys = {
        int(result.placement_id): (
            int(result.placement_descriptor.site_index),
            round(float(result.placement_descriptor.z_fraction), 5),
            float(result.placement_descriptor.azimuth_deg),
            float(result.placement_descriptor.tilt_deg),
        )
        for result in second
    }
    assert first_keys == second_keys
    best1 = min(float(result.energy_adsorption) for result in first)
    best2 = min(float(result.energy_adsorption) for result in second)
    assert best1 == pytest.approx(best2, abs=1e-2)

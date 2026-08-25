"""Integration test: water on Cu(111) via run_adsorption (campaign API + MLIP)."""

import math

import numpy as np
import pytest
from ase.build import fcc111

from metalsurfer.campaigns import run_adsorption
from metalsurfer.config import AdsorptionConfig
from metalsurfer.placement.geometry import detect_vdw_overlaps
from metalsurfer.surface_prep import apply_surface_constraints, prepare_substrate
from tests.conftest import (
    E_ADS_MLIP_TOL,
    GPU_AUTOBATCH,
    GPU_MLIP_MARKS,
    assert_water_oh_hh_geometry,
)

pytestmark = GPU_MLIP_MARKS


def test_run_adsorption_water_on_cu111(tmp_path, monkeypatch):
    """Campaign-level e2e: prepare_substrate → run_adsorption for water/Cu."""
    monkeypatch.chdir(tmp_path)
    num_placements = 8
    config = AdsorptionConfig(
        material_type="slab",
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
            fcc111("Cu", size=(3, 3, 3), a=3.6, vacuum=10.0)
        ),
        config=config,
        results_dir="results_test_water_cu_slab",
    )
    campaign = run_adsorption(
        slab=slab,
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
    min_ok = max(6, int(math.ceil(0.75 * num_placements)))
    assert len(results) >= min_ok, (
        f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
    )
    assert campaign.total_configurations == len(results)

    e_ads = np.array([r.energy_adsorption for r in results])
    assert np.all(np.isfinite(e_ads))
    # Bounds tightened against the uma-s-1p2 + oc25 reference run
    # (observed: E_ads in [-0.39, -0.18], median -0.34).
    assert float(e_ads.min()) < -0.15, (
        f"Best E_ads should be clearly binding (<-0.15 eV) for water on Cu, got {e_ads}"
    )
    assert float(np.median(e_ads)) < -0.15, (
        f"Median E_ads should be binding (<-0.15 eV) for water on Cu, "
        f"got {np.median(e_ads):.3f}"
    )
    assert np.all(e_ads < 0.0), f"All E_ads should be favorable (<0 eV), got {e_ads}"
    assert np.all(e_ads >= -0.8), (
        f"E_ads should be >= -0.8 eV for water on Cu, got min {e_ads.min():.3f}"
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

    results_dir = tmp_path / "results_water_cu_slab"
    assert (results_dir / "adsorption_energies_detailed.csv").is_file()
    assert (results_dir / "run_metadata.json").is_file()


def test_water_cu_run_twice_reproducible(tmp_path, monkeypatch):
    """Same seed twice → E_ads within GPU noise (atol 1e-2 eV)."""
    monkeypatch.chdir(tmp_path)
    num_placements = 4
    config = AdsorptionConfig(
        material_type="slab",
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
            fcc111("Cu", size=(3, 3, 3), a=3.6, vacuum=10.0)
        ),
        config=config,
        results_dir="results_water_cu_repro",
    )

    def _energies():
        campaign = run_adsorption(
            slab=slab,
            molecules=[("O", "water")],
            config=config,
            surface_type="water_cu_repro",
            skip_existing=False,
            save_results=False,
            write_settings=False,
        )
        results = campaign.run_results[0].results
        return sorted(float(r.energy_adsorption) for r in results)

    e1 = _energies()
    e2 = _energies()
    assert len(e1) == len(e2) >= 1
    assert e1 == pytest.approx(e2, abs=1e-2)

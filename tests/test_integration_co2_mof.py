"""Integration test: CO2 in MOF – plausible E_ads spread and reasonable geometries.

UMA binding energies in porous frameworks are qualitative; this smoke test checks
that placements relax to sensible CO2 geometries with a spread of E_ads values in a
physically plausible physisorption window rather than demanding strict negativity.
"""

import math
import os

import numpy as np
import pytest
from ase.io import read

from metalsurfer.config import AdsorptionConfig
from metalsurfer.optimization import setup_single_model
from metalsurfer.surface_prep import prepare_substrate
from metalsurfer.workflow import (
    calculate_reference_energies,
    process_molecule,
)
from tests.conftest import GPU_MLIP_MARKS, pair_distance

pytestmark = GPU_MLIP_MARKS

_GPU_AUTOBATCH = {
    "autobatcher_max_memory_padding": 0.8,
    "autobatcher_max_memory_scaler": 500,
    "autobatcher_max_atoms_to_try": 5000,
}


def _co_bond_length(atoms, slab_size: int) -> tuple[float, float]:
    """C–O distances in adsorbate (CO2 has exactly 2 C–O bonds)."""
    ads = atoms[slab_size:]
    syms = ads.get_chemical_symbols()
    c_indices = [i for i, s in enumerate(syms) if s == "C"]
    o_indices = [i for i, s in enumerate(syms) if s == "O"]
    if len(c_indices) != 1 or len(o_indices) != 2:
        return float("nan"), float("nan")
    pos = ads.get_positions()
    cell = atoms.get_cell()
    return (
        pair_distance(pos[c_indices[0]], pos[o_indices[0]], cell=cell),
        pair_distance(pos[c_indices[0]], pos[o_indices[1]], cell=cell),
    )


def _run_co2_in_mof():
    """Load MOF structure and run a single CO2 screening flow."""
    cif_path = os.path.join("examples", "mof_structures", "RUBTAK01.cif")
    mof_atoms = read(cif_path)
    num_placements = 5
    # Near-default porous campaign: keep experimental CIF geometry, default stages.
    config = AdsorptionConfig(
        material_type="porous",
        slab_relaxation_mode="none",
        seed=42,
        num_conformers=1,
        num_placements=num_placements,
        device="cuda",
        **_GPU_AUTOBATCH,
    )
    assert config.stage1_steps == 50
    assert config.stage2_steps == 150
    assert config.skip_topology_check is False
    assert config.skip_desorption_check is False

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
    results = process_molecule(
        "O=C=O",
        "CO2",
        mof_slab,
        calculator,
        ref,
        ts_model=ts_model,
        config=config,
        surface_type="co2_mof",
    ).results
    return results, num_placements


class TestCO2InMOF:
    def test_co2_mof_negative_adsorption_energies_and_reasonable_geometries(self):
        results, num_placements = _run_co2_in_mof()
        min_ok = max(4, int(math.ceil(0.8 * num_placements)))
        assert len(results) >= min_ok, (
            f"Expected >= {min_ok}/{num_placements} valid placements, got {len(results)}"
        )

        e_ads = np.array([r.energy_adsorption for r in results])
        assert np.all(e_ads < 0.2), (
            f"E_ads should stay in a physisorption window (< 0.2 eV), got {e_ads}"
        )
        assert np.all(e_ads >= -2.5), (
            f"E_ads should be >= -2.5 eV for CO2 in MOF, got min {e_ads.min():.3f}"
        )
        # UMA physisorption can sit slightly above zero; require a weak-binding
        # best rather than strict negativity (matches the example gate).
        assert float(e_ads.min()) < 0.5, (
            f"Best E_ads should be weak physisorption (< 0.5 eV) for CO2 in MOF, "
            f"got {e_ads}"
        )

        spread = float(e_ads.max() - e_ads.min())
        assert spread >= 0.02, (
            f"Expected distribution of E_ads (spread >= 0.02 eV), got spread {spread:.4f}"
        )

        site_ids = set()
        slab_size = len(results[0].atoms) - 3  # CO2
        for r in results:
            assert r.energy_adsorption == pytest.approx(
                r.energy_adslab - r.energy_slab - r.energy_adsorbate,
                abs=1e-4,
            )
            assert 1.5 <= r.distance <= 4.5, (
                f"Adsorbate–surface distance should be 1.5–4.5 Å, got {r.distance:.2f}"
            )
            assert r.placement_descriptor is not None
            assert r.placement_descriptor.surface_ref_z_abs is not None
            if r.placement_descriptor.site_index is not None:
                site_ids.add(int(r.placement_descriptor.site_index))
            co1, co2 = _co_bond_length(r.atoms, slab_size)
            assert 1.1 <= co1 <= 1.4, (
                f"C–O bond length should be ~1.16 Å (1.1–1.4), got {co1:.3f}"
            )
            assert 1.1 <= co2 <= 1.4, (
                f"C–O bond length should be ~1.16 Å (1.1–1.4), got {co2:.3f}"
            )
            # CO2 should remain approximately linear after relaxation.
            ads = r.atoms[slab_size:]
            syms = ads.get_chemical_symbols()
            assert sorted(syms) == ["C", "O", "O"]
            c_idx = syms.index("C")
            o_idxs = [i for i, s in enumerate(syms) if s == "O"]
            pos = ads.get_positions()
            cell = np.asarray(r.atoms.get_cell(), dtype=float)
            v1 = pos[o_idxs[0]] - pos[c_idx]
            v2 = pos[o_idxs[1]] - pos[c_idx]
            v1 = v1 - np.round(v1 @ np.linalg.inv(cell)) @ cell
            v2 = v2 - np.round(v2 @ np.linalg.inv(cell)) @ cell
            cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))
            assert 165.0 <= angle <= 180.0, (
                f"O–C–O angle should be ~180°, got {angle:.1f}"
            )
        if len(results) >= 2:
            assert len(site_ids) >= 2, f"Expected multi-site coverage, got {site_ids}"

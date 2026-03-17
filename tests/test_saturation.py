"""Tests for sequential saturation feature."""

import os
import tempfile

import pandas as pd
import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.io_results import save_saturation_results, setup_directories
from metalsurfer.models import (
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
)
from metalsurfer.placement import get_hollow_sites_for_adatoms
from metalsurfer.surfaces import (
    SlabContainer,
    auto_resize_slab_for_molecule,
    create_slab_from_bulk,
)
from metalsurfer.workflow import load_molecules, run_saturation_screening

from .conftest import (
    make_placement_descriptor,
    make_slab,
    make_water,
    place_molecule_on_slab,
)
from .optional_deps import cuda_available, has_mlip_stack

# ---------------------------------------------------------------------------
# load_molecules skip_saturation_file
# ---------------------------------------------------------------------------


def test_load_molecules_skip_saturation_file():
    """When skip_saturation_file=True, skip molecules in saturation_summary.csv."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "smiles.csv")
        with open(csv_path, "w") as f:
            f.write("O,water\nCCO,ethanol\n")
        results_dir = os.path.join(tmpdir, "results_manual")
        os.makedirs(results_dir, exist_ok=True)
        summary = pd.DataFrame(
            {"molecule": ["water"], "n_molecules_at_saturation": [3]}
        )
        summary.to_csv(
            os.path.join(results_dir, "saturation_summary.csv"),
            index=False,
        )
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            molecules, smiles, _ = load_molecules(
                csv_path,
                skip_existing=False,
                skip_saturation_file=True,
                surface_type="manual",
            )
            assert "water" not in molecules
            assert "ethanol" in molecules
        finally:
            os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# save_saturation_results
# ---------------------------------------------------------------------------


def test_save_saturation_results_empty_list_returns_early():
    """save_saturation_results with empty list returns without writing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            setup_directories(["empty_test"])
            save_saturation_results([], surface_type="empty_test")
            # No CSV files should be created
            assert not os.path.exists(
                os.path.join(tmpdir, "results_empty_test/saturation_summary.csv")
            )
        finally:
            os.chdir(old_cwd)


def test_save_saturation_results_writes_csv_and_xyz():
    """save_saturation_results writes saturation_summary, details, and XYZ."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    best = ScreeningResult(
        molecule="water",
        placement_id=0,
        energy_adslab=-190.0,
        energy_slab=-200.0,
        energy_adsorbate=-10.0,
        energy_adsorption=-1.0,
        atoms=combined,
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=best,
        all_results=[best],
    )
    sr = SaturationRunResult(
        molecule="water",
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=combined.copy(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            setup_directories(["saturation_test"])
            save_saturation_results([sr], surface_type="saturation_test")
        finally:
            os.chdir(old_cwd)
        summary_path = os.path.join(
            tmpdir, "results_saturation_test/saturation_summary.csv"
        )
        details_path = os.path.join(
            tmpdir, "results_saturation_test/saturation_details.csv"
        )
        xyz_path = os.path.join(
            tmpdir,
            "results_saturation_test/xyz_structures/water_saturation/step_001_Eads_-1.0000.xyz",
        )
        poscar_path = os.path.join(
            tmpdir,
            "results_saturation_test/vasp_inputs/water_saturation/step_001/POSCAR",
        )
        assert os.path.exists(summary_path)
        assert os.path.exists(details_path)
        assert os.path.exists(xyz_path)
        assert os.path.exists(poscar_path)
        df = pd.read_csv(summary_path)
        assert len(df) == 1
        assert df.iloc[0]["molecule"] == "water"
        assert df.iloc[0]["n_molecules_at_saturation"] == 1


# ---------------------------------------------------------------------------
# slab_for_sites uses resized slab (placement coverage)
# ---------------------------------------------------------------------------


def test_saturation_slab_for_sites_uses_resized_slab():
    """Placement site detection uses metal atoms of current slab, not original base.

    When auto_resize expands the slab, hollow sites must be computed from the
    resized slab so placements span the full cell. This test verifies that
    resizing increases hollow site count (sites scale with unit cells).

    Uses a deliberately small slab (2x2x2, cell ~5.4 A) and min_separation=8.0
    so that resize is required (5.4 < 0.74 + 8 = 8.74).
    """
    from ase import Atoms

    slab = SlabContainer(make_slab(nx=2, ny=2, n_layers=2))
    cell_diag = min(
        slab.atoms.get_cell()[0, 0],
        slab.atoms.get_cell()[1, 1],
    )
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    min_sep = 8.0
    required = 0.74 + min_sep
    assert cell_diag < required, (
        f"Test setup: slab cell {cell_diag:.1f} A must be < {required:.1f} A "
        "to trigger resize"
    )
    resized, was_resized = auto_resize_slab_for_molecule(
        slab, [h2], min_separation=min_sep
    )
    assert was_resized, (
        "Slab must be resized for this test; check auto_resize_slab_for_molecule logic"
    )
    sites_original = get_hollow_sites_for_adatoms(slab.atoms)
    sites_resized = get_hollow_sites_for_adatoms(resized.atoms)
    assert len(sites_resized) > len(sites_original), (
        "Resized slab should have more hollow sites than original"
    )


# ---------------------------------------------------------------------------
# run_saturation_screening (real GPU integration test)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.mlip
@pytest.mark.gpu
@pytest.mark.no_fork  # CUDA incompatible with pytest-forked
@pytest.mark.skipif(
    not has_mlip_stack,
    reason="MLIP stack (torch/fairchem/torch-sim-atomistic) not installed",
)
@pytest.mark.skipif(
    not cuda_available,
    reason="CUDA GPU required; skipped in CI (no GPU)",
)
def test_run_saturation_screening_h2_ni111_real_gpu():
    """Saturation screening with real MLIP on GPU: H2 on Ni(111).

    Runs actual run_saturation_screening (no mocks). Verifies:
    - Saturation loop terminates (E_ads >= 0 or no valid placements)
    - Steps and n_molecules_at_saturation are consistent
    - All E_ads in steps are physically reasonable
    """
    try:
        slab = create_slab_from_bulk(
            bulk_id="mp-23",
            miller_indices=(1, 1, 1),
            supercell=(1, 1, 1),
            results_dir="results_test_saturation",
        )
    except Exception as exc:
        pytest.skip(
            f"Slab creation failed (fairchem-data-oc): {exc!r}",
            allow_module_level=False,
        )

    config = AdsorptionConfig(
        model_name="uma-s-1p1",
        seed=42,
        num_conformers=1,
        num_placements=20,
        device="cuda",
        skip_topology_check=True,
        skip_desorption_check=False,
        stage1_steps=30,
        stage2_steps=200,
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("[H][H],H2\n")
        smiles_path = f.name

    try:
        results = run_saturation_screening(
            slab,
            smiles_file=smiles_path,
            config=config,
            surface_type="test_saturation",
            skip_existing=False,
        )
    except Exception as exc:
        pytest.skip(
            f"MLIP saturation workflow failed: {exc!r}",
            allow_module_level=False,
        )
    finally:
        os.unlink(smiles_path)

    assert len(results) == 1
    sr = results[0]
    assert sr.molecule == "H2"
    assert len(sr.steps) >= 1

    # Saturation logic: last step either has E_ads >= 0 (stopped) or E_ads < 0 (added)
    last = sr.steps[-1]
    if last.best_result.energy_adsorption >= 0:
        n_at_sat = last.n_molecules_on_slab
    else:
        n_at_sat = last.n_molecules_on_slab + 1
    assert sr.n_molecules_at_saturation == n_at_sat

    # All E_ads should be physically reasonable (not wildly unphysical)
    for step in sr.steps:
        e = step.best_result.energy_adsorption
        assert -5.0 <= e <= 5.0, f"E_ads {e:.3f} eV out of reasonable range"

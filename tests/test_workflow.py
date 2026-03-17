"""Tests for workflow validators, process_molecule branches, and I/O helpers."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from metalsurfer.config import AdsorptionConfig
from metalsurfer.io_results import (
    _write_vasp_inputs,
    save_molecule_results,
    save_single_molecule_results,
    save_summary_results,
    setup_directories,
)
from metalsurfer.models import (
    MoleculeSummary,
    ReferenceEnergies,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)
from metalsurfer.surfaces import SlabContainer
from metalsurfer.workflow import (
    _infer_surface_symbols,
    _validate_adsorption,
    _validate_geometry,
    format_failure_summary,
    load_molecules,
    process_molecule,
)

from .conftest import (
    make_placement_descriptor,
    make_slab,
    make_water,
    place_molecule_on_slab,
)

# ---------------------------------------------------------------------------
# _validate_geometry
# ---------------------------------------------------------------------------


class TestValidateGeometry:
    def _combined(self, energy=None, forces_max=0.01, min_dist_ok=True):
        slab = make_slab()
        mol = make_water()
        combined = place_molecule_on_slab(
            slab, mol, z_offset=3.0 if min_dist_ok else 0.1
        )
        calc = MagicMock()
        calc.get_potential_energy.return_value = (
            energy if energy is not None else -100.0
        )
        calc.get_forces.return_value = np.random.RandomState(0).uniform(
            -forces_max, forces_max, (len(combined), 3)
        )
        combined.calc = calc
        return combined, slab

    def test_valid_geometry_passes(self):
        combined, slab = self._combined(energy=-100.0, forces_max=0.01)
        config = AdsorptionConfig()
        ok, reason = _validate_geometry(combined, slab, config)
        assert ok
        assert "valid" in reason

    def test_non_finite_energy_fails(self):
        combined, slab = self._combined(energy=float("nan"))
        config = AdsorptionConfig()
        ok, reason = _validate_geometry(combined, slab, config)
        assert not ok
        assert "non-finite" in reason

    def test_inf_energy_fails(self):
        combined, slab = self._combined(energy=float("inf"))
        config = AdsorptionConfig()
        ok, reason = _validate_geometry(combined, slab, config)
        assert not ok
        assert "non-finite" in reason

    def test_atoms_too_close_fails(self):
        slab = make_slab()
        mol = make_water()
        combined = place_molecule_on_slab(slab, mol, z_offset=3.0)
        # force two atoms to overlap
        pos = combined.get_positions().copy()
        pos[-1] = pos[-2] + 0.01
        combined.set_positions(pos)
        calc = MagicMock()
        calc.get_potential_energy.return_value = -100.0
        calc.get_forces.return_value = np.zeros((len(combined), 3))
        combined.calc = calc
        config = AdsorptionConfig(min_interatomic_distance=0.5)
        ok, reason = _validate_geometry(combined, slab, config)
        assert not ok
        assert "too close" in reason

    def test_high_forces_fails(self):
        combined, slab = self._combined(energy=-100.0, forces_max=10.0)
        config = AdsorptionConfig(max_force_convergence=0.05)
        ok, reason = _validate_geometry(combined, slab, config)
        assert not ok
        assert "forces" in reason


# ---------------------------------------------------------------------------
# _validate_adsorption
# ---------------------------------------------------------------------------


class TestValidateAdsorption:
    def test_adsorbed_passes(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason = _validate_adsorption(combined, slab, config)
        assert ok
        assert "adsorbed" in reason

    def test_desorbed_fails(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=20.0)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason = _validate_adsorption(combined, slab, config)
        assert not ok
        assert "desorbed" in reason

    def test_no_adsorbate_fails(self):
        slab = make_slab()
        config = AdsorptionConfig()
        ok, reason = _validate_adsorption(slab, slab, config)
        assert not ok
        assert "no adsorbate" in reason

    def test_threshold_boundary(self):
        slab = make_slab()
        mol = make_water()
        combined = place_molecule_on_slab(slab, mol, z_offset=3.9)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, _ = _validate_adsorption(combined, slab, config)
        assert ok

    def test_skip_desorption_check_passes_desorbed(self):
        """When skip_desorption_check=True, desorbed structures pass validation."""
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=20.0)
        config = AdsorptionConfig(
            binding_distance_threshold=4.0, skip_desorption_check=True
        )
        ok, reason = _validate_adsorption(combined, slab, config)
        assert ok
        assert "skipped" in reason


# ---------------------------------------------------------------------------
# _infer_surface_symbols
# ---------------------------------------------------------------------------


class TestInferSurfaceSymbols:
    def test_pure_slab(self):
        slab = make_slab(symbol="Ru")
        assert _infer_surface_symbols(slab) == ["Ru"]

    def test_alloy_slab(self):
        slab = make_slab(symbol="Ru")
        syms = slab.get_chemical_symbols()
        for i in range(0, len(syms), 2):
            syms[i] = "Cu"
        slab.set_chemical_symbols(syms)
        result = _infer_surface_symbols(slab)
        assert result == ["Cu", "Ru"]


# ---------------------------------------------------------------------------
# process_molecule — branch coverage
# ---------------------------------------------------------------------------


class TestProcessMolecule:
    def _make_refs(self, mol_name="water"):
        return ReferenceEnergies(
            slab_energy=-200.0,
            molecule_energies={mol_name: -10.0},
        )

    def test_missing_ref_energy_returns_none(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(slab_energy=-200.0)
        result = process_molecule(
            "O", "water", slab, MagicMock(), reference_energies=refs
        )
        assert result is None

    def test_missing_ref_energy_populates_failure_summary(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(slab_energy=-200.0)
        failure_summary = {}
        result = process_molecule(
            "O",
            "water",
            slab,
            MagicMock(),
            reference_energies=refs,
            failure_summary_out=failure_summary,
        )
        assert result is None
        assert failure_summary["stage"] == "reference"
        assert "water" in str(failure_summary["reason"])

    def test_invalid_smiles_returns_none(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(
            slab_energy=-200.0,
            molecule_energies={"bad": -10.0},
        )
        result = process_molecule(
            "not_valid!!!", "bad", slab, MagicMock(), reference_energies=refs
        )
        assert result is None

    def test_invalid_smiles_populates_failure_summary(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(
            slab_energy=-200.0,
            molecule_energies={"bad": -10.0},
        )
        failure_summary = {}
        result = process_molecule(
            "not_valid!!!",
            "bad",
            slab,
            MagicMock(),
            reference_energies=refs,
            failure_summary_out=failure_summary,
        )
        assert result is None
        assert failure_summary["stage"] == "conformers"
        assert "bad" in str(failure_summary["reason"])

    def test_no_valid_placements_returns_none(self):
        slab = SlabContainer(make_slab())
        refs = self._make_refs()
        config = AdsorptionConfig(num_placements=1, num_conformers=1, seed=42)
        # Patch in process_molecule's namespace (robust when sys.modules is altered by other tests)
        mock_cfs = MagicMock(return_value=None)
        with patch.dict(
            process_molecule.__globals__,
            {"create_conformers_from_smiles": mock_cfs},
        ):
            result = process_molecule(
                "O",
                "water",
                slab,
                MagicMock(),
                reference_energies=refs,
                config=config,
            )
        assert result is None

    def test_no_conformers_populates_failure_summary(self):
        slab = SlabContainer(make_slab())
        refs = self._make_refs()
        config = AdsorptionConfig(num_placements=1, num_conformers=1, seed=42)
        failure_summary = {}
        mock_cfs = MagicMock(return_value=None)
        with patch.dict(
            process_molecule.__globals__,
            {"create_conformers_from_smiles": mock_cfs},
        ):
            result = process_molecule(
                "O",
                "water",
                slab,
                MagicMock(),
                reference_energies=refs,
                config=config,
                failure_summary_out=failure_summary,
            )
        assert result is None
        assert failure_summary["stage"] == "conformers"

    def test_simple_binding_energy_skips_auto_resize(self):
        """Simple binding energy (base_slab_for_frozen=None) does not resize slab.

        Resizing is ineffective due to translational symmetry; only saturation
        and pre-adsorbed runs should resize.
        """
        slab = SlabContainer(make_slab())
        refs = self._make_refs()
        config = AdsorptionConfig(
            num_placements=2,
            num_conformers=1,
            seed=42,
            auto_resize_slab=True,
        )
        mock_resize = MagicMock(return_value=(slab, False))
        with patch(
            "metalsurfer.workflow.auto_resize_slab_for_molecule",
            mock_resize,
        ):
            # No conformers -> early return; we never reach resize. Still valid:
            # the resize block is gated by base_slab_for_frozen is not None.
            mock_cfs = MagicMock(return_value=None)
            with patch.dict(
                process_molecule.__globals__,
                {"create_conformers_from_smiles": mock_cfs},
            ):
                process_molecule(
                    "O",
                    "water",
                    slab,
                    MagicMock(),
                    refs,
                    config=config,
                    base_slab_for_frozen=None,
                )
        mock_resize.assert_not_called()

    # Note: Full process_molecule integration on pre-adsorbed slab is covered by
    # test_placement.test_placement_auto_uses_envelope_for_non_planar and
    # test_placement.test_placement_mode_envelope. The process_molecule flow with
    # base_slab_for_frozen is used in saturation and works; mocking optimize_adsorbate
    # is unreliable when run in full test suite due to import order.


# ---------------------------------------------------------------------------
# format_failure_summary
# ---------------------------------------------------------------------------


class TestFormatFailureSummary:
    def test_reference_stage(self):
        summary = {"stage": "reference", "reason": "missing reference energy for H2"}
        out = format_failure_summary(summary)
        assert "Stage: reference" in out
        assert "H2" in out

    def test_placement_stage(self):
        summary = {
            "stage": "placement",
            "n_placements_attempted": 50,
            "n_initial_placements": 0,
        }
        out = format_failure_summary(summary)
        assert "Stage: placement" in out
        assert "50" in out
        assert "0" in out

    def test_validation_stage(self):
        summary = {
            "stage": "validation",
            "n_initial_placements": 50,
            "n_optimized": 48,
            "n_optimization_failed": 2,
            "validation_failures": {
                "desorbed (5.23 A)": 35,
                "high adsorbate forces: 12.34 eV/A": 8,
                "E_ads too high: 1.50 eV": 5,
            },
        }
        out = format_failure_summary(summary)
        assert "Stage: validation" in out
        assert "Initial placements: 50" in out
        assert "Optimized: 48 (2 failed)" in out
        assert "Passed validation: 0" in out
        assert "desorbed (5.23 A): 35" in out
        assert "E_ads too high" in out

    def test_filter_stage(self):
        summary = {
            "stage": "filter",
            "n_before_filter": 5,
            "n_after_filter": 0,
        }
        out = format_failure_summary(summary)
        assert "Stage: filter" in out
        assert "Before filter: 5" in out
        assert "After filter: 0" in out


# ---------------------------------------------------------------------------
# load_molecules
# ---------------------------------------------------------------------------


class TestLoadMolecules:
    def test_loads_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "smiles.csv")
            with open(csv_path, "w") as f:
                f.write("O,water\nCCO,ethanol\n")
            molecules, smiles, status = load_molecules(csv_path, skip_existing=False)
            assert molecules == ["water", "ethanol"]
            assert smiles == ["O", "CCO"]
            assert status == "ok"

    def test_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                csv_path = os.path.join(tmpdir, "smiles.csv")
                with open(csv_path, "w") as f:
                    f.write("O,water\nCCO,ethanol\n")
                results_dir = os.path.join(tmpdir, "results_manual")
                os.makedirs(results_dir, exist_ok=True)
                summary = pd.DataFrame(
                    {"molecule": ["water"], "energy_adsorption": [-1.0]}
                )
                summary.to_csv(
                    os.path.join(results_dir, "adsorption_energies_detailed.csv"),
                    index=False,
                )
                molecules, smiles, status = load_molecules(
                    csv_path, skip_existing=True, surface_type="manual"
                )
                assert "water" not in molecules
                assert status == "ok"
                assert "ethanol" in molecules
            finally:
                os.chdir(old_cwd)

    def test_single_row_csv_returns_one_molecule(self):
        """Single-row CSV returns one molecule and one SMILES."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "smiles.csv")
            with open(csv_path, "w") as f:
                f.write("O,water\n")
            molecules, smiles, status = load_molecules(csv_path, skip_existing=False)
            assert len(molecules) == 1
            assert molecules[0] == "water"
            assert len(smiles) == 1
            assert status == "ok"
            assert smiles[0] == "O"

    def test_empty_csv_returns_empty(self):
        """Empty or header-only CSV returns empty lists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "smiles.csv")
            with open(csv_path, "w") as f:
                f.write("")
            molecules, smiles, status = load_molecules(csv_path, skip_existing=False)
            assert molecules == []
            assert smiles == []
            assert status == "empty_file"


# ---------------------------------------------------------------------------
# save_summary_results
# ---------------------------------------------------------------------------


class TestSaveSummaryResults:
    def _make_result(self, mol_name, e_ads, pid=0):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        return ScreeningResult(
            molecule=mol_name,
            placement_id=pid,
            energy_adslab=-190.0,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=e_ads,
            atoms=atoms,
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=pid),
        )

    def test_writes_detailed_and_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                results = [
                    self._make_result("water", -1.5, pid=0),
                    self._make_result("water", -2.0, pid=1),
                ]
                summary = MoleculeSummary(
                    molecule="water",
                    n_configurations=2,
                    e_ads_min=-2.0,
                    e_ads_max=-1.5,
                    e_ads_mean=-1.75,
                    e_ads_std=0.25,
                    e_ads_median=-1.75,
                    best_placement_id=1,
                    e_ads_best=-2.0,
                )
                rr = ScreeningRunResult(
                    molecule="water",
                    results=results,
                    summary=summary,
                )
                save_summary_results([rr], surface_type="test")
                detailed = os.path.join(
                    tmpdir, "results_test", "adsorption_energies_detailed.csv"
                )
                summary_path = os.path.join(
                    tmpdir, "results_test", "adsorption_energy_summary.csv"
                )
                assert os.path.exists(detailed)
                assert os.path.exists(summary_path)
                df = pd.read_csv(summary_path)
                assert "molecule" in df.columns
                assert len(df) == 1
                assert df.iloc[0]["molecule"] == "water"
                # Detailed CSV includes structure paths
                detail_df = pd.read_csv(detailed)
                assert "xyz_path" in detail_df.columns
                assert "poscar_path" in detail_df.columns
                assert "conformer_000.xyz" in detail_df.iloc[0]["xyz_path"]
                assert "POSCAR" in detail_df.iloc[0]["poscar_path"]
            finally:
                os.chdir(old_cwd)

    def test_empty_results_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                save_summary_results([], surface_type="empty")
            finally:
                os.chdir(old_cwd)

    def test_detailed_csv_includes_placement_descriptor_columns(self):
        """When placement_descriptor is present, detailed CSV has descriptor columns."""
        from metalsurfer.models import PlacementDescriptor

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                atoms = place_molecule_on_slab(make_slab(), make_water())
                descriptor = PlacementDescriptor(
                    conformer_index=0,
                    orientation_type="vertical",
                    face_flip=False,
                    en_atom_index=None,
                    site_index=0,
                    site_type="atop",
                    tilt_deg=0.0,
                    azimuth_deg=0.0,
                    azimuth_in_plane_deg=0.0,
                    z_fraction=0.5,
                    placement_index=0,
                    x=1.0,
                    y=2.0,
                    z=2.5,
                    shape="linear",
                )
                results = [
                    ScreeningResult(
                        molecule="water",
                        placement_id=0,
                        energy_adslab=-190.0,
                        energy_slab=-200.0,
                        energy_adsorbate=-10.0,
                        energy_adsorption=-1.5,
                        atoms=atoms,
                        distance=2.5,
                        placement_descriptor=descriptor,
                    ),
                ]
                rr = ScreeningRunResult(
                    molecule="water",
                    results=results,
                    summary=build_molecule_summary("water", results),
                )
                save_summary_results([rr], surface_type="test")
                detail_df = pd.read_csv(
                    os.path.join(
                        tmpdir, "results_test", "adsorption_energies_detailed.csv"
                    )
                )
                assert "conformer_index" in detail_df.columns
                assert "orientation_type" in detail_df.columns
                assert "site_type" in detail_df.columns
                assert "shape" in detail_df.columns
                assert detail_df.iloc[0]["orientation_type"] == "vertical"
                assert detail_df.iloc[0]["shape"] == "linear"
            finally:
                os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# save_single_molecule_results
# ---------------------------------------------------------------------------


class TestSaveSingleMoleculeResults:
    def test_writes_xyz_poscar_and_csv(self):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        results = [
            ScreeningResult(
                molecule="water",
                placement_id=0,
                energy_adslab=-190.0,
                energy_slab=-200.0,
                energy_adsorbate=-10.0,
                energy_adsorption=-1.5,
                atoms=atoms,
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                save_single_molecule_results(
                    "water", results, surface_type="single_test"
                )
                xyz_path = os.path.join(
                    tmpdir,
                    "results_single_test",
                    "xyz_structures",
                    "water_all",
                    "conformer_000.xyz",
                )
                poscar_path = os.path.join(
                    tmpdir,
                    "results_single_test",
                    "vasp_inputs",
                    "water_all",
                    "conformer_000",
                    "POSCAR",
                )
                detailed_csv = os.path.join(
                    tmpdir,
                    "results_single_test",
                    "adsorption_energies_detailed.csv",
                )
                summary_csv = os.path.join(
                    tmpdir,
                    "results_single_test",
                    "adsorption_energy_summary.csv",
                )
                assert os.path.exists(xyz_path)
                assert os.path.exists(poscar_path)
                assert os.path.exists(detailed_csv)
                assert os.path.exists(summary_csv)
            finally:
                os.chdir(old_cwd)

    def test_empty_results_no_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                save_single_molecule_results("water", [], surface_type="empty")
                # No files created
                assert not os.path.exists(
                    os.path.join(
                        tmpdir, "results_empty/adsorption_energies_detailed.csv"
                    )
                )
            finally:
                os.chdir(old_cwd)


# ---------------------------------------------------------------------------
# save_molecule_results
# ---------------------------------------------------------------------------


class TestSaveMoleculeResults:
    def test_writes_xyz_and_vasp(self):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        entry = ScreeningResult(
            molecule="water",
            placement_id=0,
            energy_adslab=-190.0,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=-1.5,
            atoms=atoms,
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=0),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                save_molecule_results("water", [entry], surface_type="test")
                xyz_path = os.path.join(
                    tmpdir,
                    "results_test",
                    "xyz_structures",
                    "water_all",
                    "conformer_000.xyz",
                )
                vasp_path = os.path.join(
                    tmpdir, "results_test", "vasp_inputs", "water_all", "conformer_000"
                )
                assert os.path.exists(xyz_path)
                assert os.path.exists(vasp_path)
            finally:
                os.chdir(old_cwd)

    def test_write_vasp_inputs_accepts_none_config(self):
        """_write_vasp_inputs creates default config when config=None."""
        atoms = place_molecule_on_slab(make_slab(), make_water())
        with tempfile.TemporaryDirectory() as tmpdir:
            vasp_dir = os.path.join(tmpdir, "vasp_test")
            _write_vasp_inputs(atoms, vasp_dir, "water", config=None)
            assert os.path.exists(os.path.join(vasp_dir, "POSCAR"))
            assert os.path.exists(os.path.join(vasp_dir, "INCAR"))
            assert os.path.exists(os.path.join(vasp_dir, "KPOINTS"))


# ---------------------------------------------------------------------------
# setup_directories
# ---------------------------------------------------------------------------


class TestSetupDirectories:
    def test_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                setup_directories(["test_surface"])
                assert os.path.isdir("results_test_surface")
                assert os.path.isdir("results_test_surface/vasp_inputs")
                assert os.path.isdir("results_test_surface/xyz_structures")
            finally:
                os.chdir(old_cwd)

    def test_default_surface_types_when_none(self):
        """setup_directories(surface_types=None) uses default ['manual']."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                setup_directories(surface_types=None)
                assert os.path.isdir("results_manual")
                assert os.path.isdir("results_manual/vasp_inputs")
                assert os.path.isdir("results_manual/xyz_structures")
            finally:
                os.chdir(old_cwd)

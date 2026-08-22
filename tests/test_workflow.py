"""Tests for workflow validators, process_molecule branches, and I/O helpers."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from ase import Atoms
from scipy.spatial.distance import pdist

from metalsurfer.config import AdsorptionConfig, BOConfig
from metalsurfer.io_results import (
    save_molecule_results,
    save_single_molecule_results,
    save_summary_results,
    screening_run_result,
    setup_directories,
)
from metalsurfer.models import (
    BindingCampaignResult,
    MoleculeSummary,
    PlacementSpec,
    ReferenceEnergies,
    ScreeningRunResult,
    build_molecule_summary,
)
from metalsurfer.placement._material import calculator_pbc_for_atoms
from metalsurfer.placement.geometry import calculate_min_distance
from metalsurfer.reporting import (
    BOValidationFailure,
    FilterFailure,
    OptimizationFailure,
    PlacementFailure,
    ReferenceFailure,
    ValidationFailure,
)
from metalsurfer.surface_prep import SlabContainer, apply_surface_constraints
from metalsurfer.workflow import (
    load_molecules,
    process_molecule,
    process_molecule_bayesian,
)
from metalsurfer.workflow.placement_fill import placement_spec_key
from metalsurfer.workflow.shared import (
    PlacementFailureEvent,
    _build_surface_reference_slab,
    _infer_surface_symbols,
    _validate_adsorption,
    _validate_geometry,
)

from .conftest import (
    assert_lines_contain,
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    place_molecule_on_slab,
)

# ---------------------------------------------------------------------------
# Placement retry diversity key
# ---------------------------------------------------------------------------


def test_placement_spec_key_distinguishes_azimuth_in_plane():
    base = dict(
        conformer_index=0,
        orientation_type="round",
        face_flip=False,
        en_atom_index=None,
        site_index=0,
        site_type="atop",
        tilt_deg=0.0,
        azimuth_deg=0.0,
        z_fraction=0.5,
        placement_index=0,
    )
    a = PlacementSpec(**base, azimuth_in_plane_deg=0.0)
    b = PlacementSpec(**base, azimuth_in_plane_deg=90.0)
    assert placement_spec_key(a) != placement_spec_key(b)
    assert placement_spec_key(a) == placement_spec_key(
        PlacementSpec(**base, azimuth_in_plane_deg=0.0)
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
        # Deterministic force field: only the last adsorbate atom carries |F|.
        forces = np.zeros((len(combined), 3))
        forces[-1, 2] = float(forces_max)
        calc.get_forces.return_value = forces
        combined.calc = calc
        return combined, slab

    def test_valid_geometry_passes(self):
        combined, slab = self._combined(energy=-100.0, forces_max=0.01)
        config = AdsorptionConfig()
        ok, reason = _validate_geometry(combined, slab, config)
        assert ok
        assert reason == ""

    @pytest.mark.parametrize("bad_energy", [float("nan"), float("inf")])
    def test_non_finite_energy_fails(self, bad_energy):
        combined, slab = self._combined(energy=bad_energy)
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
        clash = float(np.min(pdist(combined.get_positions())))
        assert f"{clash:.3f}" in reason  # reports the actual clash distance

    def test_atoms_too_close_across_periodic_boundary_fails(self):
        """MIC clash across opposite a-faces must fail (Cartesian NN misses it)."""
        slab = make_slab()
        cell = np.asarray(slab.get_cell(), dtype=float)
        a_len = float(np.linalg.norm(cell[0]))
        # Two atoms near opposite a-faces: Cartesian distance ≈ a_len, MIC ≈ 0.2 Å.
        atoms = Atoms(
            "HH",
            positions=[[0.1, 2.0, 8.0], [a_len - 0.1, 2.0, 8.0]],
            cell=cell,
            pbc=[True, True, False],
        )
        calc = MagicMock()
        calc.get_potential_energy.return_value = -100.0
        calc.get_forces.return_value = np.zeros((2, 3))
        atoms.calc = calc
        config = AdsorptionConfig(
            material_type="slab",
            min_interatomic_distance=0.5,
        )
        ok, reason = _validate_geometry(atoms, slab[:0], config)
        assert not ok
        assert "too close" in reason
        # Cartesian NN would see ~a_len and pass; MIC must report ~0.2 Å.
        assert float(reason.split(":")[1].split()[0]) < 0.5

    def test_high_forces_fails(self):
        injected = 10.0
        combined, slab = self._combined(energy=-100.0, forces_max=injected)
        config = AdsorptionConfig(max_force_convergence=0.05)
        ok, reason = _validate_geometry(combined, slab, config)
        assert not ok
        assert "forces" in reason
        assert f"{injected:.3f}" in reason


# ---------------------------------------------------------------------------
# _validate_adsorption
# ---------------------------------------------------------------------------


class TestValidateAdsorption:
    def test_adsorbed_passes(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason, _ = _validate_adsorption(combined, slab, config)
        assert ok
        assert reason == ""

    def test_desorbed_fails(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=20.0)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason, _ = _validate_adsorption(combined, slab, config)
        assert not ok
        assert "desorbed" in reason

    def test_desorbed_fails_without_wrapping_through_vacuum(self):
        """Regression: material PBC must not MIC-wrap a lifted adsorbate to the image slab."""
        slab = make_slab(n_layers=3)
        cell = slab.get_cell()
        cell[2, 2] = 18.0
        slab.set_cell(cell)

        combined = place_molecule_on_slab(slab, make_water(), z_offset=12.0)
        combined.set_cell(cell)

        wrapped_d = calculate_min_distance(
            combined[len(slab) :].get_positions(),
            slab.get_positions(),
            cell,
            use_pbc=True,
            pbc=calculator_pbc_for_atoms(combined),
        )
        assert wrapped_d < 4.0, "3D PBC would falsely score this as bound"

        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason, min_d = _validate_adsorption(combined, slab, config)
        assert not ok, f"lifted adsorbate should be desorbed, got: {reason}"
        assert "desorbed" in reason
        assert min_d is not None and min_d > 4.0

    def test_no_adsorbate_fails(self):
        slab = make_slab()
        config = AdsorptionConfig()
        ok, reason, _ = _validate_adsorption(slab, slab, config)
        assert not ok
        assert "no adsorbate" in reason

    def test_threshold_boundary(self):
        slab = make_slab()
        mol = make_water()
        combined = place_molecule_on_slab(slab, mol, z_offset=3.9)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, _, _ = _validate_adsorption(combined, slab, config)
        assert ok

    def test_skip_desorption_check_passes_desorbed(self):
        """When skip_desorption_check=True, desorbed structures pass validation."""
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water(), z_offset=20.0)
        config = AdsorptionConfig(
            binding_distance_threshold=4.0, skip_desorption_check=True
        )
        ok, reason, _ = _validate_adsorption(combined, slab, config)
        assert ok
        assert reason == ""

    def test_validate_adsorption_ignores_pre_adsorbed_atoms_with_surface_symbols(self):
        """Regression: saturation slabs may include previously adsorbed atoms.

        Validation must check adsorption distance to substrate atoms only
        (selected by surface_symbols), not to pre-adsorbed atoms.
        """
        slab_metal = make_slab(symbol="Ru")
        x_shift = 5.0
        y_shift = 5.0
        z_offset = 10.0
        slab_metal_z = float(np.max(slab_metal.get_positions()[:, 2]))

        water = make_water().copy()
        pos = water.get_positions().copy()
        pos -= np.mean(pos, axis=0)
        pos[:, 0] += x_shift
        pos[:, 1] += y_shift
        pos[:, 2] += slab_metal_z + z_offset
        water.set_positions(pos)

        # Place fake pre-adsorbate at oxygen position (close contact).
        o_pos = water.get_positions()[0].copy()
        pre_adsorbed = Atoms("C", positions=[o_pos])
        slab_with_pre_adsorbate = slab_metal + pre_adsorbed
        slab_with_pre_adsorbate.set_cell(slab_metal.get_cell())
        slab_with_pre_adsorbate.set_pbc(slab_metal.get_pbc())

        combined = slab_with_pre_adsorbate + water
        combined.set_cell(slab_with_pre_adsorbate.get_cell())
        combined.set_pbc(slab_with_pre_adsorbate.get_pbc())
        config = AdsorptionConfig(binding_distance_threshold=4.0)

        ok, _, _ = _validate_adsorption(combined, slab_with_pre_adsorbate, config)
        assert ok, "Without surface_symbols, pre-adsorbed atoms can mask desorption"

        ok, reason, _ = _validate_adsorption(
            combined,
            slab_with_pre_adsorbate,
            config,
            surface_symbols=["Ru"],
        )
        assert not ok
        assert "desorbed" in reason


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
# _build_surface_reference_slab
# ---------------------------------------------------------------------------


class TestBuildSurfaceReferenceSlab:
    def test_returns_substrate_only_and_preserves_cell_pbc(self):
        slab_metal = make_slab(symbol="Ru")
        slab_z = float(np.max(slab_metal.get_positions()[:, 2]))
        pre_adsorbed = Atoms("C", positions=[[1.0, 2.0, slab_z + 3.0]])
        slab_with_pre_adsorbate = slab_metal + pre_adsorbed
        slab_with_pre_adsorbate.set_cell(slab_metal.get_cell())
        slab_with_pre_adsorbate.set_pbc(slab_metal.get_pbc())

        ref = _build_surface_reference_slab(
            slab_with_pre_adsorbate, base_slab_for_frozen=slab_metal
        )
        assert len(ref) == len(slab_metal)
        assert set(ref.get_chemical_symbols()) == {"Ru"}
        assert np.allclose(ref.get_cell(), slab_with_pre_adsorbate.get_cell())
        assert np.all(ref.get_pbc() == slab_with_pre_adsorbate.get_pbc())


# ---------------------------------------------------------------------------
# process_molecule — branch coverage
# ---------------------------------------------------------------------------


class TestProcessMolecule:
    def _make_refs(self, mol_name="water"):
        return ReferenceEnergies(
            slab_energy=-200.0,
            molecule_energies={mol_name: -10.0},
        )

    @pytest.mark.parametrize(
        ("smiles", "name", "refs", "expected_stage"),
        [
            (
                "O",
                "water",
                ReferenceEnergies(slab_energy=-200.0),
                "reference",
            ),
            (
                "not_valid!!!",
                "bad",
                ReferenceEnergies(
                    slab_energy=-200.0,
                    molecule_energies={"bad": -10.0},
                ),
                "conformers",
            ),
        ],
    )
    def test_early_failure_paths(self, smiles, name, refs, expected_stage):
        slab = SlabContainer(make_slab())
        config = AdsorptionConfig(num_placements=1, num_conformers=1, seed=42)
        outcome = process_molecule(
            smiles,
            name,
            slab,
            MagicMock(),
            reference_energies=refs,
            config=config,
        )
        assert outcome.results == []
        assert outcome.failure_summary is not None
        assert outcome.failure_summary.stage == expected_stage
        assert name in str(outcome.failure_summary.reason)

    def test_no_conformers_populates_failure_summary(self):
        slab = SlabContainer(make_slab())
        refs = self._make_refs()
        config = AdsorptionConfig(num_placements=1, num_conformers=1, seed=42)
        mock_cfs = MagicMock(return_value=None)
        with patch(
            "metalsurfer.workflow.shared.create_conformers_from_smiles", mock_cfs
        ):
            outcome = process_molecule(
                "O",
                "water",
                slab,
                MagicMock(),
                reference_energies=refs,
                config=config,
            )
        assert outcome.results == []
        assert outcome.failure_summary is not None
        assert outcome.failure_summary.stage == "conformers"

    def test_prepare_substrate_validates_image_separation(self):
        from metalsurfer.exceptions import GeometryValidationError
        from metalsurfer.workflow.shared import prepare_substrate_for_screening

        tiny = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
            cell=[3, 3, 20],
            pbc=[True, True, False],
        )
        slab = SlabContainer(apply_surface_constraints(tiny))
        config = AdsorptionConfig(num_placements=2, num_conformers=1, seed=42)
        big_mol = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]])
        with pytest.raises(
            GeometryValidationError, match="auto_resize_substrate_for_molecule"
        ):
            prepare_substrate_for_screening(
                slab,
                [big_mol],
                None,
                config,
            )

    def test_bo_failure_events_emit_negative_records(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(slab_energy=-200.0, molecule_energies={"water": -10.0})
        config = AdsorptionConfig(
            bo=BOConfig(initial_random=1, batch_size=1, total_budget=1),
            num_placements=2,
            num_conformers=1,
            seed=7,
        )
        specs = [
            PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=-1,
                site_type=None,
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=0,
            )
        ]
        success = make_screening_result(
            molecule="water",
            placement_id=0,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=-1.0,
            atoms=place_molecule_on_slab(make_slab(), make_water()),
            slab_size=len(make_slab()),
            distance=2.0,
            placement_descriptor=make_placement_descriptor(placement_id=0),
        )
        mock_conformers = MagicMock(return_value=([Atoms("H")], [0.0]))
        mock_capacity = MagicMock(return_value=1)
        mock_specs = MagicMock(return_value=specs)
        mock_eval = MagicMock(
            return_value=(
                [success],
                [
                    PlacementFailureEvent(
                        placement_id=0,
                        stage="validation",
                        reason="desorbed (6.20 A)",
                        descriptor=make_placement_descriptor(placement_id=0),
                    )
                ],
            )
        )
        with (
            patch(
                "metalsurfer.workflow.shared.create_conformers_from_smiles",
                mock_conformers,
            ),
            patch(
                "metalsurfer.workflow.bayesian.estimate_placement_spec_capacity",
                mock_capacity,
            ),
            patch(
                "metalsurfer.workflow.bayesian.enumerate_placement_specs", mock_specs
            ),
            patch("metalsurfer.workflow.bayesian._evaluate_placement_batch", mock_eval),
            patch(
                "metalsurfer.workflow.shared.filter_results",
                side_effect=lambda results, **_: results,
            ),
            patch(
                "metalsurfer.workflow.bayesian.build_spec_features_geometry_aware",
                return_value=(
                    pd.DataFrame([{"x": 0.0, "y": 0.0, "z": 7.0}]),
                    [0],
                ),
            ),
        ):
            outcome = process_molecule_bayesian(
                "O",
                "water",
                slab,
                MagicMock(),
                refs,
                ts_model=MagicMock(),
                config=config,
            )
        assert outcome.results is not None
        assert len(outcome.results) == 1
        assert len(outcome.ml_records) == 1
        assert outcome.ml_records[0].is_penalty_label is True
        assert outcome.ml_records[0].failure_stage == "validation"
        assert outcome.ml_records[0].energy_adsorption == pytest.approx(
            config.bo.failure_penalty_overrides["validation"]
        )

    def test_bo_deduplicated_results_are_tracked_for_ml(self):
        slab = SlabContainer(make_slab())
        refs = ReferenceEnergies(slab_energy=-200.0, molecule_energies={"water": -10.0})
        config = AdsorptionConfig(
            bo=BOConfig(initial_random=1, batch_size=1, total_budget=1),
            num_placements=2,
            num_conformers=1,
            seed=7,
        )
        specs = [
            PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=-1,
                site_type=None,
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=0,
            )
        ]
        unique = make_screening_result(
            molecule="water",
            placement_id=0,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=-1.0,
            atoms=place_molecule_on_slab(make_slab(), make_water()),
            slab_size=len(make_slab()),
            distance=2.0,
            placement_descriptor=make_placement_descriptor(placement_id=0),
        )
        duplicate = make_screening_result(
            molecule="water",
            placement_id=1,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=-1.01,
            atoms=place_molecule_on_slab(make_slab(), make_water()),
            slab_size=len(make_slab()),
            distance=2.0,
            placement_descriptor=make_placement_descriptor(placement_id=1),
        )

        mock_conformers = MagicMock(return_value=([Atoms("H")], [0.0]))
        mock_capacity = MagicMock(return_value=1)
        mock_specs = MagicMock(return_value=specs)
        mock_eval = MagicMock(return_value=([unique, duplicate], []))

        def _mock_filter(results, **kwargs):
            kwargs["duplicate_results_out"].append(results[1])
            return [results[0]]

        with (
            patch(
                "metalsurfer.workflow.shared.create_conformers_from_smiles",
                mock_conformers,
            ),
            patch(
                "metalsurfer.workflow.bayesian.estimate_placement_spec_capacity",
                mock_capacity,
            ),
            patch(
                "metalsurfer.workflow.bayesian.enumerate_placement_specs", mock_specs
            ),
            patch("metalsurfer.workflow.bayesian._evaluate_placement_batch", mock_eval),
            patch(
                "metalsurfer.workflow.shared.filter_results", side_effect=_mock_filter
            ),
            patch(
                "metalsurfer.workflow.bayesian.build_spec_features_geometry_aware",
                return_value=(
                    pd.DataFrame([{"x": 0.0, "y": 0.0, "z": 7.0}]),
                    [0],
                ),
            ),
        ):
            outcome = process_molecule_bayesian(
                "O",
                "water",
                slab,
                MagicMock(),
                refs,
                ts_model=MagicMock(),
                config=config,
            )

        assert outcome.results is not None
        assert len(outcome.results) == 1
        assert len(outcome.ml_records) == 1
        assert outcome.ml_records[0].placement_id == 1
        assert outcome.ml_records[0].label_source == "deduplicated_duplicate"
        assert outcome.ml_records[0].is_penalty_label is False


# ---------------------------------------------------------------------------
# format_failure_summary
# ---------------------------------------------------------------------------


class TestFormatFailureSummary:
    def test_reference_stage(self):
        summary = ReferenceFailure(reason="missing reference energy for H2")
        out = BindingCampaignResult.format_failure_summary(summary)
        assert_lines_contain(
            out,
            [
                "  Stage: reference",
                "  Reason: missing reference energy for H2",
            ],
        )

    def test_placement_stage(self):
        summary = PlacementFailure(
            n_placements_attempted=50,
            n_initial_placements=0,
        )
        out = BindingCampaignResult.format_failure_summary(summary)
        assert_lines_contain(
            out,
            [
                "  Stage: placement",
                "  Placements attempted: 50",
                "  Initial placements: 0",
            ],
        )

    def test_validation_stage(self):
        summary = ValidationFailure(
            n_initial_placements=50,
            n_optimized=48,
            n_optimization_failed=2,
            validation_failures={
                "desorbed (5.23 A)": 35,
                "high adsorbate forces: 12.34 eV/A": 8,
                "E_ads too high: 1.50 eV": 5,
            },
        )
        out = BindingCampaignResult.format_failure_summary(summary)
        assert_lines_contain(
            out,
            [
                "  Stage: validation",
                "  Initial placements: 50",
                "  Optimized: 48 (2 failed)",
                "  Passed validation: 0",
                "    desorbed (5.23 A): 35",
            ],
        )
        assert "E_ads too high" in out

    def test_validation_stage_bo(self):
        summary = BOValidationFailure(
            n_evaluated=12,
            n_valid_results=0,
            n_candidate_specs=40,
            n_valid_pool=30,
        )
        out = BindingCampaignResult.format_failure_summary(summary)
        assert_lines_contain(
            out,
            [
                "  Stage: validation",
                "  BO evaluated: 12",
                "  BO valid results: 0",
            ],
        )
        assert "Initial placements" not in out
        assert "Optimized:" not in out
        assert "?" not in out

    def test_filter_stage(self):
        summary = FilterFailure(
            n_before_filter=5,
            n_after_filter=0,
        )
        out = BindingCampaignResult.format_failure_summary(summary)
        assert_lines_contain(
            out,
            [
                "  Stage: filter",
                "  Before filter: 5",
                "  After filter: 0",
            ],
        )

    def test_optimization_stage(self):
        failed = OptimizationFailure(
            n_placements_attempted=10,
            n_initial_placements=10,
            n_optimized=0,
            n_optimization_failed=10,
            validation_failures={"optimizer_crash": 10},
        )
        out = BindingCampaignResult.format_failure_summary(failed)
        assert_lines_contain(
            out,
            [
                "  Stage: optimization",
                "  Placements attempted: 10",
                "  Initial placements: 10",
                "  Optimized: 0 (10 failed)",
                "    optimizer_crash: 10",
            ],
        )
        empty = OptimizationFailure(
            n_placements_attempted=20,
            n_initial_placements=20,
        )
        assert "Optimized:" not in BindingCampaignResult.format_failure_summary(empty)


# ---------------------------------------------------------------------------
# load_molecules
# ---------------------------------------------------------------------------


class TestLoadMolecules:
    def test_loads_csv(self, workdir):
        csv_path = workdir / "smiles.csv"
        csv_path.write_text("O,water\nCCO,ethanol\n")
        molecules, smiles, status = load_molecules(str(csv_path), skip_existing=False)
        assert molecules == ["water", "ethanol"]
        assert smiles == ["O", "CCO"]
        assert status == "ok"

    def test_skip_existing(self, workdir):
        csv_path = workdir / "smiles.csv"
        csv_path.write_text("O,water\nCCO,ethanol\n")
        results_dir = workdir / "results_manual"
        results_dir.mkdir(exist_ok=True)
        summary = pd.DataFrame({"molecule": ["water"], "energy_adsorption": [-1.0]})
        summary.to_csv(results_dir / "adsorption_energies_detailed.csv", index=False)
        molecules, smiles, status = load_molecules(
            str(csv_path), skip_existing=True, surface_type="manual"
        )
        assert "water" not in molecules
        assert status == "ok"
        assert "ethanol" in molecules

    def test_single_row_csv_returns_one_molecule(self, workdir):
        """Single-row CSV returns one molecule and one SMILES."""
        csv_path = workdir / "smiles.csv"
        csv_path.write_text("O,water\n")
        molecules, smiles, status = load_molecules(str(csv_path), skip_existing=False)
        assert len(molecules) == 1
        assert molecules[0] == "water"
        assert len(smiles) == 1
        assert status == "ok"
        assert smiles[0] == "O"

    def test_loads_csv_with_header(self, workdir):
        csv_path = workdir / "smiles.csv"
        csv_path.write_text("smiles,molecule\nO,water\n")
        molecules, smiles, status = load_molecules(str(csv_path), skip_existing=False)
        assert molecules == ["water"]
        assert smiles == ["O"]
        assert status == "ok"

    def test_empty_csv_returns_empty(self, workdir):
        """Empty or header-only CSV returns empty lists."""
        csv_path = workdir / "smiles.csv"
        csv_path.write_text("")
        molecules, smiles, status = load_molecules(str(csv_path), skip_existing=False)
        assert molecules == []
        assert smiles == []
        assert status == "empty_file"


# ---------------------------------------------------------------------------
# save_summary_results
# ---------------------------------------------------------------------------


class TestSaveSummaryResults:
    def _make_result(self, mol_name, e_ads, pid=0):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        return make_screening_result(
            molecule=mol_name,
            placement_id=pid,
            energy_adsorption=e_ads,
            atoms=atoms,
            slab_size=len(make_slab()),
            distance=2.5,
        )

    def test_writes_detailed_and_summary_csv(self, workdir):
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
        rr = ScreeningRunResult(molecule="water", results=results, summary=summary)
        save_summary_results([rr], surface_type="test")
        detailed = workdir / "results_test" / "adsorption_energies_detailed.csv"
        summary_path = workdir / "results_test" / "adsorption_energy_summary.csv"
        assert detailed.exists()
        assert summary_path.exists()
        df = pd.read_csv(summary_path)
        assert "molecule" in df.columns
        assert len(df) == 1
        assert df.iloc[0]["molecule"] == "water"
        detail_df = pd.read_csv(detailed)
        assert "xyz_path" in detail_df.columns
        assert "poscar_path" not in detail_df.columns
        assert "conformer_000.xyz" in detail_df.iloc[0]["xyz_path"]

    def test_empty_results_no_crash(self, workdir):
        save_summary_results([], surface_type="empty")

    def test_detailed_csv_includes_placement_descriptor_columns(self, workdir):
        """Lean detailed CSV keeps pose features; provenance needs the knob."""
        atoms = place_molecule_on_slab(make_slab(), make_water())
        descriptor = make_placement_descriptor(
            placement_id=0,
            orientation_type="vertical",
            site_type="atop",
            x=1.0,
            y=2.0,
            z_offset=2.5,
            shape="linear",
        )
        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.5,
                atoms=atoms,
                slab_size=len(make_slab()),
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
            workdir / "results_test" / "adsorption_energies_detailed.csv"
        )
        assert "conformer_index" in detail_df.columns
        assert "x_abs" in detail_df.columns
        assert "quat_w" in detail_df.columns
        assert "orientation_type" not in detail_df.columns
        assert "site_type" not in detail_df.columns
        assert "shape" not in detail_df.columns
        assert "initial_orientation_type" not in detail_df.columns

        save_summary_results(
            [rr],
            surface_type="test_rich",
            config=AdsorptionConfig(export_placement_provenance=True),
        )
        rich_df = pd.read_csv(
            workdir / "results_test_rich" / "adsorption_energies_detailed.csv"
        )
        assert "initial_orientation_type" in rich_df.columns
        assert "initial_site_type" in rich_df.columns
        assert "initial_shape" in rich_df.columns
        assert rich_df.iloc[0]["initial_orientation_type"] == "vertical"
        assert rich_df.iloc[0]["initial_shape"] == "linear"


# ---------------------------------------------------------------------------
# save_single_molecule_results
# ---------------------------------------------------------------------------


class TestSaveSingleMoleculeResults:
    def test_writes_xyz_and_csv_by_default(self, workdir):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.5,
                atoms=atoms,
                slab_size=len(make_slab()),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        save_single_molecule_results("water", results, surface_type="single_test")
        xyz_path = (
            workdir / "results_single_test/xyz_structures/water_all/conformer_000.xyz"
        )
        poscar_path = (
            workdir / "results_single_test/vasp_inputs/water_all/conformer_000/POSCAR"
        )
        detailed_csv = workdir / "results_single_test/adsorption_energies_detailed.csv"
        summary_csv = workdir / "results_single_test/adsorption_energy_summary.csv"
        assert xyz_path.exists()
        assert not poscar_path.exists()
        assert detailed_csv.exists()
        assert summary_csv.exists()

    def test_writes_vasp_when_enabled(self, workdir):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.5,
                atoms=atoms,
                slab_size=len(make_slab()),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        config = AdsorptionConfig(write_vasp_inputs=True)
        save_single_molecule_results(
            "water", results, surface_type="vasp_test", config=config
        )
        poscar_path = (
            workdir / "results_vasp_test/vasp_inputs/water_all/conformer_000/POSCAR"
        )
        assert poscar_path.exists()

    def test_write_csv_false_writes_structures_not_csv(self, workdir):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.5,
                atoms=atoms,
                slab_size=len(make_slab()),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        save_single_molecule_results(
            "water",
            results,
            surface_type="no_csv_test",
            write_csv=False,
        )
        xyz_path = (
            workdir / "results_no_csv_test/xyz_structures/water_all/conformer_000.xyz"
        )
        detailed_csv = workdir / "results_no_csv_test/adsorption_energies_detailed.csv"
        assert xyz_path.exists()
        assert not detailed_csv.exists()

    def test_campaign_two_molecules_combined_csv(self, workdir):
        slab = make_slab()
        water_atoms = place_molecule_on_slab(slab, make_water())
        other_atoms = place_molecule_on_slab(slab, make_water())
        water_results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.5,
                atoms=water_atoms,
                slab_size=len(slab),
                distance=2.5,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        other_results = [
            make_screening_result(
                molecule="other",
                placement_id=0,
                energy_adsorption=-1.0,
                atoms=other_atoms,
                slab_size=len(slab),
                distance=2.6,
                placement_descriptor=make_placement_descriptor(placement_id=0),
            ),
        ]
        st = "campaign_test"
        config = AdsorptionConfig()
        save_single_molecule_results(
            "water", water_results, surface_type=st, write_csv=False
        )
        save_single_molecule_results(
            "other", other_results, surface_type=st, write_csv=False
        )
        combined = [
            screening_run_result("water", water_results),
            screening_run_result("other", other_results),
        ]
        save_summary_results(combined, surface_type=st, config=config)
        detailed_csv = workdir / f"results_{st}" / "adsorption_energies_detailed.csv"
        summary_csv = workdir / f"results_{st}" / "adsorption_energy_summary.csv"
        df = pd.read_csv(detailed_csv)
        sdf = pd.read_csv(summary_csv)
        assert set(df["molecule"].unique()) == {"water", "other"}
        assert len(sdf) == 2
        assert set(sdf["molecule"]) == {"water", "other"}

    def test_empty_results_no_crash(self, workdir):
        save_single_molecule_results("water", [], surface_type="empty")
        assert not (workdir / "results_empty/adsorption_energies_detailed.csv").exists()


# ---------------------------------------------------------------------------
# save_molecule_results
# ---------------------------------------------------------------------------


class TestSaveMoleculeResults:
    def test_writes_xyz_by_default(self, workdir):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        entry = make_screening_result(
            molecule="water",
            placement_id=0,
            energy_adsorption=-1.5,
            atoms=atoms,
            slab_size=len(make_slab()),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=0),
        )
        save_molecule_results("water", [entry], surface_type="test")
        xyz_path = workdir / "results_test/xyz_structures/water_all/conformer_000.xyz"
        vasp_path = workdir / "results_test/vasp_inputs/water_all/conformer_000"
        assert xyz_path.exists()
        assert not vasp_path.exists()

    def test_writes_vasp_when_enabled(self, workdir):
        atoms = place_molecule_on_slab(make_slab(), make_water())
        entry = make_screening_result(
            molecule="water",
            placement_id=0,
            energy_adsorption=-1.5,
            atoms=atoms,
            slab_size=len(make_slab()),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=0),
        )
        config = AdsorptionConfig(write_vasp_inputs=True)
        save_molecule_results("water", [entry], surface_type="vasp_test", config=config)
        vasp_dir = workdir / "results_vasp_test/vasp_inputs/water_all/conformer_000"
        assert (vasp_dir / "POSCAR").exists()
        assert (vasp_dir / "INCAR").exists()
        assert (vasp_dir / "KPOINTS").exists()


# ---------------------------------------------------------------------------
# setup_directories
# ---------------------------------------------------------------------------


class TestSetupDirectories:
    def test_creates_directories(self, workdir):
        setup_directories(["test_surface"])
        assert os.path.isdir("results_test_surface")
        assert not os.path.isdir("results_test_surface/vasp_inputs")
        assert os.path.isdir("results_test_surface/xyz_structures")

    def test_creates_vasp_directories_when_enabled(self, workdir):
        setup_directories(["test_surface"], write_vasp_inputs=True)
        assert os.path.isdir("results_test_surface/vasp_inputs")

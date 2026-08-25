"""Tests for typed domain models."""

import dataclasses

import pytest

from metalsurfer.models import (
    BindingCampaignResult,
    MoleculeCampaignSummary,
    MoleculeSummary,
    PlacementDescriptor,
    ReferenceEnergies,
    SaturationCampaignResult,
    SaturationRunResult,
    SaturationStepResult,
    ScreeningResult,
    ScreeningRunResult,
    build_molecule_summary,
)
from metalsurfer.reporting import ConformerFailure

from .conftest import (
    make_h2,
    make_placement_descriptor,
    make_slab,
    make_water,
    place_molecule_on_slab,
)


def test_reference_energies():
    ref = ReferenceEnergies(
        slab_energy=-100.0,
        molecule_energies={"water": -10.0, "ethanol": -20.0},
    )
    assert ref.slab_energy == -100.0
    assert ref.get_molecule_energy("water") == -10.0
    assert ref.get_molecule_energy("missing") is None


def test_reference_energies_empty():
    ref = ReferenceEnergies(slab_energy=-50.0)
    assert ref.molecule_energies == {}
    assert ref.get_molecule_energy("any") is None


def test_screening_result():
    atoms = make_h2()
    sr = ScreeningResult(
        molecule="water",
        placement_id=5,
        energy_adslab=-110.0,
        energy_slab=-100.0,
        energy_adsorbate=-10.0,
        energy_adsorption=-0.5,
        atoms=atoms,
        slab_size=0,
        distance=2.3,
        placement_descriptor=make_placement_descriptor(placement_id=5),
    )
    assert sr.molecule == "water"
    assert sr.placement_id == 5
    assert sr.energy_adsorption == pytest.approx(-0.5)
    assert len(sr.atoms) == 2
    row = sr.to_row(xyz_path="results/x.xyz", poscar_path="results/POSCAR")
    assert row["molecule"] == "water"
    assert row["placement_id"] == 5
    assert row["xyz_path"] == "results/x.xyz"
    assert row["poscar_path"] == "results/POSCAR"
    assert "orientation_type" not in row
    assert "z_fraction" not in row
    # No absolute pose on this descriptor: x_abs must fall back to relative x.
    assert sr.placement_descriptor.x_abs is None
    assert row["x_abs"] == pytest.approx(0.0)
    # With an absolute pose present, x_abs must win over relative x.
    abs_desc = make_placement_descriptor(x=1.5, x_abs=-2.25)
    abs_row = dataclasses.replace(sr, placement_descriptor=abs_desc).to_row(
        xyz_path="results/x.xyz", poscar_path="results/POSCAR"
    )
    assert abs_row["x_abs"] == pytest.approx(-2.25)
    for field in ("quat_w", "quat_x", "quat_y", "quat_z"):
        assert field in row
    rich = sr.to_row(include_provenance=True)
    assert rich["initial_orientation_type"] == sr.placement_descriptor.orientation_type
    assert rich["initial_z_fraction"] == sr.placement_descriptor.z_fraction


def test_molecule_summary():
    s = MoleculeSummary(
        molecule="water",
        n_configurations=5,
        e_ads_min=-2.0,
        e_ads_max=-0.5,
        e_ads_mean=-1.2,
        e_ads_std=0.5,
        e_ads_median=-1.1,
        best_placement_id=3,
        e_ads_best=-2.0,
    )
    assert s.molecule == "water"
    assert s.e_ads_best == -2.0


def test_build_molecule_summary_empty_raises():
    """build_molecule_summary raises ValueError for empty results."""
    with pytest.raises(
        ValueError, match="Cannot build molecule summary from empty results"
    ):
        build_molecule_summary("water", [])


def test_build_molecule_summary():
    """build_molecule_summary computes aggregate statistics from results."""
    slab = make_slab()
    combined = place_molecule_on_slab(slab, make_water())
    results = [
        ScreeningResult(
            molecule="water",
            placement_id=i,
            energy_adslab=-190.0 - i * 0.1,
            energy_slab=-200.0,
            energy_adsorbate=-10.0,
            energy_adsorption=-1.5 - i * 0.1,
            atoms=combined,
            slab_size=len(slab),
            distance=2.5,
            placement_descriptor=make_placement_descriptor(placement_id=i),
        )
        for i in range(3)
    ]
    summary = build_molecule_summary("water", results)
    assert summary.molecule == "water"
    assert summary.n_configurations == 3
    assert summary.e_ads_min == -1.7  # -1.5 - 0.2
    assert summary.e_ads_max == -1.5
    assert summary.best_placement_id == 2
    assert summary.e_ads_best == -1.7


def test_screening_run_result():
    atoms = make_h2()
    sr = ScreeningResult(
        molecule="water",
        placement_id=0,
        energy_adslab=-110.0,
        energy_slab=-100.0,
        energy_adsorbate=-10.0,
        energy_adsorption=-0.5,
        atoms=atoms,
        slab_size=0,
        distance=2.3,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    summary = MoleculeSummary(
        molecule="water",
        n_configurations=1,
        e_ads_min=-0.5,
        e_ads_max=-0.5,
        e_ads_mean=-0.5,
        e_ads_std=0.0,
        e_ads_median=-0.5,
        best_placement_id=0,
        e_ads_best=-0.5,
    )
    rr = ScreeningRunResult(
        molecule="water",
        results=[sr],
        summary=summary,
    )
    assert rr.molecule == "water"
    assert len(rr.results) == 1
    assert rr.summary.e_ads_best == -0.5
    rows = rr.to_rows(results_dir="results_test")
    assert len(rows) == 1
    assert rows[0]["xyz_path"].endswith("water_all/conformer_000.xyz")
    assert "poscar_path" not in rows[0]
    rows_vasp = rr.to_rows(results_dir="results_test", write_vasp_inputs=True)
    assert rows_vasp[0]["poscar_path"].endswith("water_all/conformer_000/POSCAR")
    summary_row = rr.to_summary_row()
    assert summary_row is not None
    assert summary_row["best_placement_id"] == 0


def test_screening_run_result_uses_run_level_molecule_in_to_rows():
    """Flattened exports must use ScreeningRunResult.molecule, not inner placement names."""
    from tests.conftest import make_screening_result

    inner = make_screening_result(
        molecule="demo", placement_id=0, energy_adsorption=-1.0
    )
    run = ScreeningRunResult(
        molecule="demo_step_002",
        results=[inner],
        summary=build_molecule_summary("demo_step_002", [inner]),
    )
    rows = run.to_rows()
    assert len(rows) == 1
    assert rows[0]["molecule"] == "demo_step_002"
    assert inner.molecule == "demo"


def test_saturation_step_result():
    """SaturationStepResult holds step metadata and best result."""
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
        slab_size=len(slab),
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
    assert step.step == 1
    assert step.n_molecules_on_slab == 0
    assert step.best_result.energy_adsorption == pytest.approx(-1.0)
    detail_row = step.to_detail_row(
        results_dir="results_test",
        saturation_molecule="water",
    )
    assert detail_row["step"] == 1
    assert detail_row["molecule"] == "water"
    assert detail_row["step_structure_path"].endswith(
        "water_saturation/step_001_best_slab.xyz"
    )
    placement_rows = step.to_rows(
        results_dir="results_test", saturation_molecule="water"
    )
    assert len(placement_rows) == 1
    assert placement_rows[0]["step"] == 1
    assert placement_rows[0]["xyz_path"].endswith(
        "water_saturation/step_001_placements/conformer_000.xyz"
    )


def test_saturation_step_result_n_added_defaults_and_committed():
    """Legacy steps default to one committed placement; unbound steps to none."""
    slab = make_slab()
    bound = ScreeningResult(
        molecule="water",
        placement_id=0,
        energy_adslab=-190.0,
        energy_slab=-200.0,
        energy_adsorbate=-10.0,
        energy_adsorption=-1.0,
        atoms=place_molecule_on_slab(slab, make_water()),
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=0),
    )
    legacy = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=bound,
        all_results=[bound],
    )
    assert legacy.n_added == 1
    assert legacy.committed_results == []
    assert legacy.committed() == [bound]

    unbound_best = ScreeningResult(
        molecule="water",
        placement_id=1,
        energy_adslab=-199.0,
        energy_slab=-200.0,
        energy_adsorbate=-10.0,
        energy_adsorption=1.0,
        atoms=place_molecule_on_slab(slab, make_water()),
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=1),
    )
    unbound = SaturationStepResult(
        step=2,
        molecule="water",
        n_molecules_on_slab=1,
        best_result=unbound_best,
        all_results=[unbound_best],
        n_added=0,
    )
    assert unbound.committed() == []

    explicit = [bound, bound]
    multi = SaturationStepResult(
        step=2,
        molecule="water",
        n_molecules_on_slab=1,
        best_result=bound,
        all_results=[bound],
        n_added=2,
        committed_results=explicit,
    )
    assert multi.committed() == explicit


def test_saturation_run_result():
    """SaturationRunResult holds steps and saturation count."""
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
        slab_size=len(slab),
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
    assert sr.molecule == "water"
    assert len(sr.steps) == 1
    assert sr.n_molecules_at_saturation == 1
    flattened = sr.to_flattened_runs()
    assert len(flattened) == 1
    assert flattened[0].molecule == "water_step_001"
    text = sr.format_completion(label="Water saturation", results_dir="results_water")
    assert "Water saturation complete:" in text
    assert "Molecules at saturation: 1" in text
    assert "Results saved to results_water/" in text
    assert "(XYZ, CSV)" in text
    assert "POSCAR" not in text
    assert "POSCAR" in sr.format_completion(
        label="Water saturation",
        results_dir="results_water",
        write_vasp_inputs=True,
    )


def test_saturation_campaign_result_format_completion():
    step = SaturationStepResult(
        step=1,
        molecule="water",
        n_molecules_on_slab=0,
        best_result=None,
        all_results=[],
    )
    run = SaturationRunResult(
        molecule="water",
        steps=[step],
        n_molecules_at_saturation=1,
        final_slab_atoms=make_slab(),
    )
    campaign = SaturationCampaignResult(
        mode="non_bo",
        surface_type="water",
        runs=[run],
    )
    text = campaign.format_completion(
        label="Water saturation",
        results_dir="results_water",
    )
    assert "(XYZ, CSV)" in text
    assert "POSCAR" not in text

    multi = SaturationCampaignResult(
        mode="non_bo",
        surface_type="multi",
        runs=[run, run],
    )
    multi_text = multi.format_completion(
        label="Multi saturation",
        results_dir="results_multi",
        write_vasp_inputs=True,
    )
    assert "Multi saturation complete:" in multi_text
    assert "Molecules at saturation: 2" in multi_text
    assert "(XYZ, POSCAR, CSV)" in multi_text


def test_binding_campaign_result_formatters():
    campaign = BindingCampaignResult(
        mode="non_bo",
        surface_type="manual",
        run_results=[],
        molecule_summaries=[
            MoleculeCampaignSummary(
                molecule="water",
                n_valid_placements=3,
                best_adsorption_energy=-1.23,
            )
        ],
        total_configurations=42,
        n_molecules=1,
        t_ref_s=0.1,
        t_total_s=0.2,
    )
    assert campaign.format_results_saved_line(results_dir="results_manual").startswith(
        "Results saved to results_manual/"
    )
    assert "(XYZ, CSV)" in campaign.format_results_saved_line(
        results_dir="results_manual"
    )
    assert "POSCAR" not in campaign.format_results_saved_line(
        results_dir="results_manual"
    )
    assert "POSCAR" in campaign.format_results_saved_line(
        results_dir="results_manual",
        write_vasp_inputs=True,
    )
    summary = campaign.format_summary(
        title="Binding summary",
        results_dir="results_manual",
    )
    assert "Binding summary" in summary
    assert "water" in summary


def test_binding_campaign_format_summary_default_title():
    campaign = BindingCampaignResult(
        mode="non_bo",
        surface_type="manual",
        run_results=[],
        molecule_summaries=[
            MoleculeCampaignSummary(
                molecule="H2",
                n_valid_placements=1,
                best_adsorption_energy=-0.5,
            )
        ],
        total_configurations=1,
        n_molecules=1,
        t_ref_s=0.0,
        t_total_s=0.0,
    )
    summary = campaign.format_summary(results_dir="results_manual")
    assert "Binding energy summary" in summary
    assert "H2" in summary


def test_binding_campaign_format_summary_includes_failures():
    campaign = BindingCampaignResult(
        mode="non_bo",
        surface_type="manual",
        run_results=[],
        molecule_summaries=[
            MoleculeCampaignSummary(
                molecule="bad",
                n_valid_placements=0,
                best_adsorption_energy=None,
            )
        ],
        total_configurations=0,
        n_molecules=1,
        t_ref_s=0.0,
        t_total_s=0.0,
        failure_summaries={
            "bad": ConformerFailure(reason="could not generate conformers")
        },
    )
    summary = campaign.format_summary(
        title="Binding summary",
        results_dir="results_manual",
    )
    assert "(no valid placements)" in summary
    assert "Failures for bad" in summary
    assert "conformers" in summary


def test_binding_campaign_format_summary_empty_run():
    campaign = BindingCampaignResult(
        mode="non_bo",
        surface_type="manual",
        run_results=[],
        molecule_summaries=[],
        total_configurations=0,
        n_molecules=0,
        t_ref_s=0.0,
        t_total_s=0.0,
    )
    summary = campaign.format_summary(results_dir="results_manual")
    assert "No molecules processed" in summary
    assert "(XYZ, CSV)" in summary


# ---------------------------------------------------------------------------
# PlacementDescriptor row coercion / round-trip (QC #5)
# ---------------------------------------------------------------------------


class TestPlacementDescriptorRow:
    def test_to_row_from_row_preserves_fields(self):
        desc = make_placement_descriptor(
            placement_id=3,
            conformer_index=2,
            tilt_deg=15.0,
            azimuth_deg=45.0,
            x=1.2,
            y=2.3,
            z_offset=2.5,
            shape="flat",
            site_type="fcc",
        )
        row = desc.to_row(include_provenance=True)
        restored = PlacementDescriptor.from_row(row, placement_index=3)
        for attr in (
            "conformer_index",
            "orientation_type",
            "face_flip",
            "en_atom_index",
            "site_index",
            "site_type",
            "tilt_deg",
            "azimuth_deg",
            "azimuth_in_plane_deg",
            "z_fraction",
            "placement_index",
            "x",
            "y",
            "z_offset",
            "shape",
        ):
            assert getattr(restored, attr) == getattr(desc, attr), attr

    def test_to_row_from_row_preserves_absolute_pose(self):
        """x_abs/y_abs/z_abs and quat_* must survive lean CSV round-trip."""
        desc = make_placement_descriptor(
            placement_id=5,
            x=1.0,
            y=2.0,
            x_abs=9.1,
            y_abs=8.2,
            z_abs=12.3,
            quat_w=0.6,
            quat_x=0.1,
            quat_y=0.2,
            quat_z=0.3,
        )
        assert desc.x_abs != desc.x
        row = desc.to_row(include_provenance=False)
        restored = PlacementDescriptor.from_row(row, placement_index=5)
        for attr in (
            "x_abs",
            "y_abs",
            "z_abs",
            "quat_w",
            "quat_x",
            "quat_y",
            "quat_z",
        ):
            assert getattr(restored, attr) == pytest.approx(getattr(desc, attr)), attr

    def test_from_row_handles_missing_columns(self):
        """Missing optional columns fall back to sane defaults without raising."""
        row = {"conformer_index": 0, "x": 1.0, "y": 2.0}
        desc = PlacementDescriptor.from_row(row)
        assert desc.conformer_index == 0
        assert desc.orientation_type == "round"
        assert desc.site_index == -1
        assert desc.z_fraction == 0.5
        assert desc.face_flip is False

    def test_from_row_rejects_missing_conformer_index(self):
        """A missing required conformer_index raises a clear error, not KeyError."""
        with pytest.raises(ValueError, match="conformer_index"):
            PlacementDescriptor.from_row({"x": 1.0})

    def test_from_row_handles_nan_string_cells(self):
        # Provenance fields arrive as ``initial_*`` columns; values may be
        # "nan" strings or empty cells when read back from CSV.
        row = {
            "conformer_index": "0",
            "initial_en_atom_index": "nan",
            "initial_x": "1.5",
            "initial_face_flip": "false",
            "initial_site_index": "",
        }
        desc = PlacementDescriptor.from_row(row)
        assert desc.conformer_index == 0
        assert desc.en_atom_index is None
        assert desc.x == pytest.approx(1.5)
        assert desc.face_flip is False
        assert desc.site_index == -1

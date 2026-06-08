"""Unit tests for campaign facade symbols and result contracts."""

from metalsurfer import (
    BindingCampaignResult,
    MoleculeCampaignSummary,
    SaturationCampaignResult,
)
from metalsurfer.campaigns import (
    run_adsorption,
    run_adsorption_bo,
    run_saturation,
    run_saturation_bo,
)
from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import (
    MultiMolSaturationRunResult,
    SaturationRunResult,
    SaturationStepResult,
)
from tests.conftest import make_screening_result


def test_campaign_exports_are_callable():
    assert callable(run_adsorption)
    assert callable(run_adsorption_bo)
    assert callable(run_saturation)
    assert callable(run_saturation_bo)


def test_campaign_result_dataclasses():
    summary = MoleculeCampaignSummary(
        molecule="ethane",
        n_valid_placements=10,
        best_adsorption_energy=-0.5,
        n_parallel=6,
        n_endown=4,
    )
    result = BindingCampaignResult(
        mode="non_bo",
        surface_type="demo",
        run_results=[],
        molecule_summaries=[summary],
        total_configurations=10,
        n_molecules=1,
        t_ref_s=1.0,
        t_total_s=2.0,
    )
    assert result.mode == "non_bo"
    assert result.molecule_summaries[0].molecule == "ethane"


def test_run_saturation_bo_forces_bo_and_preserves_multi_molecule(monkeypatch):
    captured: dict[str, object] = {}

    def _fake_run_saturation_screening(*, config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs
        return [
            MultiMolSaturationRunResult(
                molecules=["demo"],
                steps=[],
                n_molecules_at_saturation=0,
            )
        ]

    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        _fake_run_saturation_screening,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_saturation_results",
        lambda *args, **kwargs: None,
    )

    campaign = run_saturation_bo(
        slab=object(),
        molecules="demo.csv",
        config=AdsorptionConfig(
            bo_enabled=False,
            multi_molecule_saturation=True,
        ),
        surface_type="demo",
        skip_existing=False,
        save_results=False,
        write_settings=False,
    )

    assert isinstance(campaign, SaturationCampaignResult)
    assert len(campaign.runs) == 1
    config = captured["config"]
    assert isinstance(config, AdsorptionConfig)
    assert config.bo_enabled is True
    assert config.multi_molecule_saturation is True


def test_run_saturation_passes_config_to_save_saturation_results(monkeypatch):
    """save_saturation_results receives the same AdsorptionConfig as the run."""
    captured: dict[str, object] = {}

    def fake_save(
        results: object,
        surface_type: str = "manual",
        config: AdsorptionConfig | None = None,
    ) -> None:
        captured["save_config"] = config
        captured["surface_type"] = surface_type
        captured["n_results"] = len(results)

    def fake_screening(**_kwargs):
        return [
            SaturationRunResult(
                molecule="demo",
                steps=[],
                n_molecules_at_saturation=0,
            )
        ]

    monkeypatch.setattr("metalsurfer.campaigns.save_saturation_results", fake_save)
    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        fake_screening,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )

    cfg = AdsorptionConfig(seed=999, saturation_save_all_placements=False)
    campaign = run_saturation(
        slab=object(),
        molecules=[("C", "demo")],
        config=cfg,
        surface_type="st_save_cfg",
        skip_existing=False,
        write_settings=False,
    )
    assert isinstance(campaign, SaturationCampaignResult)
    assert captured["save_config"] is cfg
    assert captured["surface_type"] == "st_save_cfg"
    assert captured["n_results"] == 1


def test_run_saturation_save_benchmark_dataset(monkeypatch):
    captured: dict[str, object] = {}

    def fake_save_summary(run_results, surface_type="manual", config=None):
        captured["summary_runs"] = run_results
        captured["surface_type"] = surface_type

    placement = make_screening_result(energy_adsorption=-1.0)

    def fake_screening(**_kwargs):
        return [
            SaturationRunResult(
                molecule="demo",
                steps=[
                    SaturationStepResult(
                        step=1,
                        molecule="demo",
                        n_molecules_on_slab=1,
                        best_result=placement,
                        all_results=[placement],
                    )
                ],
                n_molecules_at_saturation=1,
            )
        ]

    monkeypatch.setattr(
        "metalsurfer.campaigns.save_summary_results",
        fake_save_summary,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_saturation_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        fake_screening,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )

    cfg = AdsorptionConfig(save_benchmark_dataset=True)
    run_saturation(
        slab=object(),
        molecules=[("C", "demo")],
        config=cfg,
        surface_type="st_bench",
        skip_existing=False,
        write_settings=False,
    )
    assert "summary_runs" in captured
    assert captured["surface_type"] == "st_bench"
    flattened = captured["summary_runs"]
    assert len(flattened) == 1
    assert flattened[0].molecule == "demo_step_001"
    rows = flattened[0].to_rows()
    assert rows[0]["molecule"] == "demo_step_001"


def test_run_saturation_write_metadata_persists_json(tmp_path, monkeypatch):
    """write_metadata=True must call write_run_metadata_from_out, not pass a dict."""
    import json

    monkeypatch.chdir(tmp_path)
    run_metadata: dict[str, float] = {}

    def fake_screening(*, run_metadata_out=None, **_kwargs):
        if run_metadata_out is not None:
            run_metadata_out.update(
                n_molecules=2.0,
                total_configs=7.0,
                t_ref_s=1.0,
                t_total_s=3.5,
            )
        return [
            SaturationRunResult(
                molecule="demo",
                steps=[],
                n_molecules_at_saturation=0,
            )
        ]

    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        fake_screening,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_saturation_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )

    campaign = run_saturation(
        slab=object(),
        molecules=[("C", "demo")],
        config=AdsorptionConfig(seed=1),
        surface_type="st_meta",
        skip_existing=False,
        write_settings=False,
        write_metadata=True,
        run_metadata_out=run_metadata,
    )
    assert isinstance(campaign, SaturationCampaignResult)

    path = tmp_path / "results_st_meta" / "run_metadata.json"
    assert path.exists()
    with open(path) as f:
        meta = json.load(f)
    assert meta["input"]["n_molecules"] == 2
    assert meta["results"]["total_configurations"] == 7
    assert meta["timing"]["reference_energies_s"] == 1.0
    assert meta["timing"]["total_wall_clock_s"] == 3.5

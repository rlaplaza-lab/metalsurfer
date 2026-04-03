"""Unit tests for campaign facade symbols and result contracts."""

from metalsurfer import BindingCampaignResult, MoleculeCampaignSummary
from metalsurfer.campaigns import (
    run_adsorption,
    run_adsorption_bo,
    run_saturation,
    run_saturation_bo,
)
from metalsurfer.config import AdsorptionConfig


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
        return ["ok"]

    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        _fake_run_saturation_screening,
    )

    out = run_saturation_bo(
        slab=object(),
        molecules="demo.csv",
        config=AdsorptionConfig(
            bo_enabled=False,
            multi_molecule_saturation=True,
        ),
        surface_type="demo",
        skip_existing=False,
    )

    assert out == ["ok"]
    config = captured["config"]
    assert isinstance(config, AdsorptionConfig)
    assert config.bo_enabled is True
    assert config.multi_molecule_saturation is True

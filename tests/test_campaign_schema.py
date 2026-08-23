"""Tests for YAML campaign schema and run_campaign dispatch."""

from pathlib import Path

import pytest

from metalsurfer.campaign_schema import load_campaign_yaml, parse_campaign_dict
from metalsurfer.campaigns import run_campaign

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "campaigns"


def test_load_smoke_saturation_yaml():
    doc = load_campaign_yaml(FIXTURES / "smoke_saturation.yaml")
    assert doc.campaign == "saturation"
    assert doc.surface_type == "smoke_saturation"
    assert doc.molecules == [("O", "water")]
    assert doc.config.num_placements == 4
    assert doc.config.device == "cpu"
    assert doc.substrate["bulk_id"] == "mp-30"
    assert doc.substrate["miller_indices"] == (1, 1, 1)


def test_load_campaign_yaml_missing_file():
    with pytest.raises(ValueError, match="Campaign file not found"):
        load_campaign_yaml("/nonexistent/campaign.yaml")


def test_load_campaign_yaml_empty_file(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Campaign file is empty"):
        load_campaign_yaml(empty)


def test_parse_campaign_rejects_unknown_substrate_key():
    with pytest.raises(ValueError, match="unknown keys"):
        parse_campaign_dict(
            {
                "campaign": "saturation",
                "surface_type": "bad",
                "substrate": {"bulk_id": "mp-30", "unexpected": True},
                "molecules": [{"smiles": "C", "name": "methane"}],
            }
        )


def test_parse_campaign_rejects_non_atoms_slab():
    with pytest.raises(ValueError, match="substrate.slab must be"):
        parse_campaign_dict(
            {
                "campaign": "saturation",
                "surface_type": "bad",
                "substrate": {"slab": "not-an-atoms"},
                "molecules": [{"smiles": "C", "name": "methane"}],
            }
        )


def test_parse_campaign_rejects_unknown_root_key():
    with pytest.raises(ValueError, match="unknown keys"):
        parse_campaign_dict(
            {
                "campaign": "saturation",
                "surface_type": "bad",
                "confgi": "oops",
                "substrate": {"bulk_id": "mp-30"},
                "molecules": [{"smiles": "C", "name": "methane"}],
            }
        )


def test_parse_campaign_rejects_unknown_config_key():
    with pytest.raises(ValueError, match="unknown key"):
        parse_campaign_dict(
            {
                "campaign": "saturation",
                "surface_type": "bad",
                "substrate": {"bulk_id": "mp-30"},
                "molecules": [{"smiles": "C", "name": "methane"}],
                "config": {"num_placementz": 5},
            }
        )


def test_parse_campaign_accepts_nested_bo_config():
    doc = parse_campaign_dict(
        {
            "campaign": "saturation_bo",
            "surface_type": "ok",
            "substrate": {"bulk_id": "mp-30"},
            "molecules": [{"smiles": "C", "name": "methane"}],
            "config": {"bo": {"transfer": {"enabled": False, "weight_cap": 0.25}}},
        }
    )
    assert doc.config.bo.transfer.enabled is False
    assert doc.config.bo.transfer.weight_cap == pytest.approx(0.25)


def test_run_campaign_dispatches_with_mocks(monkeypatch):
    doc = load_campaign_yaml(FIXTURES / "smoke_saturation.yaml")
    calls: dict[str, object] = {}

    class _FakeSlab:
        pass

    def fake_prepare_substrate(**kwargs):
        calls["substrate"] = kwargs
        return _FakeSlab()

    def fake_run_saturation(**kwargs):
        calls["run"] = kwargs
        return type("Result", (), {"runs": []})()

    import metalsurfer.campaigns as campaigns_module

    monkeypatch.setattr(
        "metalsurfer.campaigns.prepare_substrate",
        fake_prepare_substrate,
    )
    monkeypatch.setitem(campaigns_module._RUNNERS, "saturation", fake_run_saturation)

    run_campaign(doc, skip_existing=False)

    assert calls["substrate"]["bulk_id"] == "mp-30"
    assert calls["substrate"]["config"] is doc.config
    assert calls["run"]["molecules"] == doc.molecules
    assert calls["run"]["skip_existing"] is False

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


_VALID_BASE = {
    "campaign": "saturation",
    "surface_type": "ok",
    "substrate": {"bulk_id": "mp-30"},
    "molecules": [{"smiles": "C", "name": "methane"}],
}


@pytest.mark.parametrize(
    "kind",
    ["adsorption", "adsorption_bo", "saturation", "saturation_bo"],
)
def test_parse_campaign_accepts_all_valid_kinds(kind):
    doc = parse_campaign_dict({**_VALID_BASE, "campaign": kind})
    assert doc.campaign == kind


def test_parse_campaign_rejects_invalid_kind():
    with pytest.raises(ValueError, match="campaign must be one of"):
        parse_campaign_dict({**_VALID_BASE, "campaign": "vibrations"})


@pytest.mark.parametrize("bad", [None, "", "   "])
def test_parse_campaign_rejects_missing_or_empty_surface_type(bad):
    # Missing key and explicit null both reach the same isinstance guard.
    with pytest.raises(ValueError, match="surface_type must be a non-empty"):
        parse_campaign_dict({**_VALID_BASE, "surface_type": bad})


@pytest.mark.parametrize(
    "molecules_raw",
    [
        None,  # key missing and explicit null both hit the same guard
        [],
        "water",
        [{"name": "no-smiles"}],
        [{"smiles": "C", "name": ""}],
    ],
)
def test_parse_campaign_rejects_invalid_molecules(molecules_raw):
    with pytest.raises(ValueError, match="molecules"):
        parse_campaign_dict({**_VALID_BASE, "molecules": molecules_raw})


def test_parse_campaign_accepts_task_name_config_key():
    doc = parse_campaign_dict({**_VALID_BASE, "config": {"task_name": "oc20"}})
    assert doc.config.task_name == "oc20"


def test_parse_campaign_rejects_non_mapping_config():
    with pytest.raises(ValueError, match="config must be a mapping"):
        parse_campaign_dict({**_VALID_BASE, "config": 5})


def test_parse_campaign_rejects_non_mapping_root(tmp_path):
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="campaign root must be a mapping"):
        load_campaign_yaml(list_yaml)


def test_parse_campaign_requires_exactly_one_substrate_source():
    with pytest.raises(ValueError, match="exactly one of bulk_id"):
        parse_campaign_dict({**_VALID_BASE, "substrate": {}})


def test_parse_campaign_rejects_wrong_length_miller_indices():
    with pytest.raises(
        ValueError, match="substrate.miller_indices must be a 3-element"
    ):
        parse_campaign_dict(
            {
                **_VALID_BASE,
                "substrate": {"bulk_id": "mp-30", "miller_indices": [1, 1]},
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

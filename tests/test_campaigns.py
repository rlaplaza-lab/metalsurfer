"""Unit tests for campaign facade symbols and result contracts."""

import pytest

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
from metalsurfer.workflow.shared import MoleculeScreenOutcome
from metalsurfer.models import (
    MultiMolSaturationRunResult,
    SaturationRunResult,
    SaturationStepResult,
)
from tests.conftest import make_screening_result


def _patch_binding_bootstrap(monkeypatch, slab_container, ref=None):
    from metalsurfer.workflow.shared import ScreeningRunBootstrap

    if ref is None:
        ref = object()

    monkeypatch.setattr(
        "metalsurfer.campaigns._bootstrap_screening_run",
        lambda _slab, pairs, _config: ScreeningRunBootstrap(
            calculator=object(),
            ts_model=object(),
            molecule_pairs=pairs,
            ref=ref,
            t_ref_s=0.0,
            slab=slab_container,
        ),
    )


def test_campaign_result_dataclasses():
    summary = MoleculeCampaignSummary(
        molecule="ethane",
        n_valid_placements=10,
        best_adsorption_energy=-0.5,
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


def test_run_saturation_bo_passes_bo_enabled_and_preserves_multi_molecule(monkeypatch):
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
    assert captured["kwargs"]["bo_enabled"] is True
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


def test_run_saturation_write_settings_persists_json(tmp_path, monkeypatch):
    """write_settings=True must call write_run_metadata_from_out, not pass a dict."""
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
        write_settings=True,
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


def test_run_adsorption_csv_path_unified_with_inline(tmp_path, monkeypatch):
    """CSV input uses the same binding path as in-memory lists."""
    from metalsurfer.surface_prep import SlabContainer
    from tests.conftest import make_slab

    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text("C,demo\n")

    placement = make_screening_result(molecule="demo", energy_adsorption=-1.0)
    saved_summary: dict[str, object] = {}
    slab_container = SlabContainer(make_slab())

    def fake_process(_smi, mol, *_args, **_kwargs):
        return MoleculeScreenOutcome(results=[placement])

    def fake_save_summary(run_results, surface_type="manual", config=None):
        saved_summary["run_results"] = run_results
        saved_summary["surface_type"] = surface_type

    _patch_binding_bootstrap(monkeypatch, slab_container)
    monkeypatch.setattr("metalsurfer.campaigns.process_molecule", fake_process)
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_single_molecule_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_summary_results",
        fake_save_summary,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.DatasetLogger.flush",
        lambda self: None,
    )

    campaign = run_adsorption(
        slab=slab_container,
        molecules=str(csv_path),
        config=AdsorptionConfig(seed=1),
        surface_type="csv_demo",
        skip_existing=False,
        write_settings=False,
    )

    assert isinstance(campaign, BindingCampaignResult)
    assert len(campaign.molecule_summaries) == 1
    assert campaign.molecule_summaries[0].molecule == "demo"
    assert campaign.molecule_summaries[0].best_adsorption_energy == -1.0
    assert campaign.n_molecules == 1
    assert saved_summary["surface_type"] == "csv_demo"
    assert len(saved_summary["run_results"]) == 1


def test_run_adsorption_skip_existing_inline_list(tmp_path, monkeypatch):
    """In-memory molecule lists honor skip_existing via summary CSV."""
    import pandas as pd

    from metalsurfer.surface_prep import SlabContainer
    from tests.conftest import make_slab

    monkeypatch.chdir(tmp_path)
    results_dir = tmp_path / "results_skip_inline"
    results_dir.mkdir(parents=True)
    pd.DataFrame({"molecule": ["water"]}).to_csv(
        results_dir / "adsorption_energies_detailed.csv",
        index=False,
    )

    captured: dict[str, int] = {"count": 0}
    slab_container = SlabContainer(make_slab())

    def fake_process(_smi, mol, *_args, **_kwargs):
        captured["count"] += 1
        return MoleculeScreenOutcome(results=[])

    _patch_binding_bootstrap(monkeypatch, slab_container)
    monkeypatch.setattr("metalsurfer.campaigns.process_molecule", fake_process)
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.DatasetLogger.flush",
        lambda self: None,
    )

    campaign = run_adsorption(
        slab=slab_container,
        molecules=[("O", "water"), ("CCO", "ethanol")],
        config=AdsorptionConfig(seed=1),
        surface_type="skip_inline",
        skip_existing=True,
        save_results=False,
        write_settings=False,
    )

    assert campaign.n_molecules == 1
    assert captured["count"] == 1


def test_run_adsorption_warns_when_all_skipped(monkeypatch):
    monkeypatch.setattr(
        "metalsurfer.campaigns._normalize_molecules_input",
        lambda *args, **kwargs: ([], "all_skipped", "demo.csv"),
    )
    with pytest.warns(UserWarning, match="adsorption_energies_detailed"):
        campaign = run_adsorption(
            slab=object(),
            molecules="demo.csv",
            config=AdsorptionConfig(seed=1),
            surface_type="skip_all",
            save_results=False,
            write_settings=False,
        )
    assert campaign.n_molecules == 0
    assert "No molecules processed" in campaign.format_summary(
        results_dir="results_skip_all"
    )


def test_run_adsorption_warns_when_input_empty(monkeypatch):
    monkeypatch.setattr(
        "metalsurfer.campaigns._normalize_molecules_input",
        lambda *args, **kwargs: ([], "empty_file", "empty.csv"),
    )
    with pytest.warns(UserWarning, match="no valid rows"):
        campaign = run_adsorption(
            slab=object(),
            molecules="empty.csv",
            config=AdsorptionConfig(seed=1),
            surface_type="empty_input",
            save_results=False,
            write_settings=False,
        )
    assert campaign.n_molecules == 0


def test_run_saturation_write_settings_includes_campaign_metadata(
    monkeypatch, tmp_path
):
    captured: dict[str, object] = {}

    def fake_write_settings(surface_type, config, **run_info):
        captured["surface_type"] = surface_type
        captured["run_info"] = run_info

    monkeypatch.setattr(
        "metalsurfer.campaigns.write_run_settings",
        fake_write_settings,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.run_saturation_screening",
        lambda **kwargs: [
            SaturationRunResult(
                molecule="demo",
                steps=[],
                n_molecules_at_saturation=0,
            )
        ],
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_saturation_results",
        lambda *args, **kwargs: None,
    )

    run_saturation(
        slab=object(),
        molecules=[("C", "demo")],
        config=AdsorptionConfig(seed=1),
        surface_type="st_meta_settings",
        skip_existing=False,
        write_settings=True,
    )

    run_info = captured["run_info"]
    assert run_info["campaign"] == "saturation"
    assert run_info["mode"] == "non_bo"
    assert run_info["n_molecules"] == 1
    assert run_info["molecules"] == ["demo"]


def test_run_adsorption_save_results_false_skips_disk_writes(tmp_path, monkeypatch):
    """save_results=False skips structure and summary writes for CSV input."""
    from metalsurfer.surface_prep import SlabContainer
    from tests.conftest import make_slab

    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "demo.csv"
    csv_path.write_text("C,demo\n")

    placement = make_screening_result(molecule="demo", energy_adsorption=-1.0)
    saved: dict[str, bool] = {"single": False, "summary": False}
    slab_container = SlabContainer(make_slab())

    def fake_process(_smi, mol, *_args, **_kwargs):
        return MoleculeScreenOutcome(results=[placement])

    _patch_binding_bootstrap(monkeypatch, slab_container)
    monkeypatch.setattr("metalsurfer.campaigns.process_molecule", fake_process)
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_single_molecule_results",
        lambda *args, **kwargs: saved.__setitem__("single", True),
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_summary_results",
        lambda *args, **kwargs: saved.__setitem__("summary", True),
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.DatasetLogger.flush",
        lambda self: None,
    )

    campaign = run_adsorption(
        slab=slab_container,
        molecules=str(csv_path),
        config=AdsorptionConfig(seed=1),
        surface_type="no_save_csv",
        skip_existing=False,
        save_results=False,
        write_settings=False,
    )

    assert isinstance(campaign, BindingCampaignResult)
    assert campaign.n_molecules == 1
    assert saved == {"single": False, "summary": False}


def test_bootstrap_screening_run_validates_substrate_before_model(monkeypatch):
    from metalsurfer.workflow.shared import _bootstrap_screening_run

    model_called = {"value": False}

    def fake_accept(*_args, **_kwargs):
        raise ValueError("bad substrate")

    def fake_setup(*_args, **_kwargs):
        model_called["value"] = True
        return object(), object()

    monkeypatch.setattr(
        "metalsurfer.workflow.shared.accept_substrate_for_api",
        fake_accept,
    )
    monkeypatch.setattr(
        "metalsurfer.workflow.shared.setup_single_model",
        fake_setup,
    )

    with pytest.raises(ValueError, match="bad substrate"):
        _bootstrap_screening_run(
            object(),
            [("C", "demo")],
            AdsorptionConfig(),
        )
    assert model_called["value"] is False


def test_write_settings_alone_writes_timing_metadata(tmp_path, monkeypatch):
    """write_settings=True (default) also persists timing into run_metadata.json."""
    import json

    from metalsurfer.surface_prep import SlabContainer
    from tests.conftest import make_slab

    monkeypatch.chdir(tmp_path)
    placement = make_screening_result(molecule="demo", energy_adsorption=-1.0)
    slab_container = SlabContainer(make_slab())

    def fake_process(_smi, mol, *_args, **_kwargs):
        return MoleculeScreenOutcome(results=[placement])

    _patch_binding_bootstrap(monkeypatch, slab_container)
    monkeypatch.setattr("metalsurfer.campaigns.process_molecule", fake_process)
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_single_molecule_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.save_summary_results",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.setup_directories",
        lambda surface_types, **kwargs: None,
    )
    monkeypatch.setattr(
        "metalsurfer.campaigns.DatasetLogger.flush",
        lambda self: None,
    )

    run_adsorption(
        slab=slab_container,
        molecules=[("C", "demo")],
        config=AdsorptionConfig(seed=1),
        surface_type="meta_or",
        skip_existing=False,
        write_settings=True,
    )

    path = tmp_path / "results_meta_or" / "run_metadata.json"
    assert path.exists()
    with open(path) as f:
        meta = json.load(f)
    assert meta["campaign"] == "multi_molecule_binding"
    assert "timing" in meta
    assert "config" in meta

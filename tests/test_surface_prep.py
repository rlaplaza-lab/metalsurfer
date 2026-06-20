"""Tests for prepare_substrate relaxation option forwarding."""

from metalsurfer.config import AdsorptionConfig
from metalsurfer.surface_prep import SlabContainer, prepare_substrate

from .conftest import make_slab


def test_prepare_substrate_passes_slab_relaxation_options(monkeypatch):
    captured: dict = {}
    fake_calc = object()

    def _fake_setup_single_model(model_name, device):
        captured["setup"] = (model_name, device)
        return fake_calc, None

    def _fake_create_slab_from_bulk(**kwargs):
        captured["create_kwargs"] = kwargs
        return SlabContainer(make_slab(symbol="Ru"))

    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        _fake_setup_single_model,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.create_slab_from_bulk",
        _fake_create_slab_from_bulk,
    )

    prepare_substrate(
        bulk_id="mp-33",
        config=AdsorptionConfig(model_name="uma-s-1p1", device="cpu"),
        slab_relaxation_mode="full",
        slab_relaxation_optimizer="bfgs",
        slab_relaxation_fmax=0.03,
        slab_relaxation_steps=111,
    )

    assert captured["setup"] == ("uma-s-1p1", "cpu")
    assert captured["create_kwargs"]["calculator"] is fake_calc
    assert captured["create_kwargs"]["relaxation_mode"] == "full"
    assert captured["create_kwargs"]["relaxation_optimizer"] == "bfgs"
    assert captured["create_kwargs"]["relaxation_fmax"] == 0.03
    assert captured["create_kwargs"]["relaxation_steps"] == 111


def test_prepare_substrate_passes_adatom_relaxation_options(monkeypatch):
    captured: dict = {}
    fake_calc = object()

    def _fake_setup_single_model(model_name, device):
        return fake_calc, None

    def _fake_create_slab_from_bulk(**kwargs):
        return SlabContainer(make_slab(symbol="Ru"))

    def _fake_deposit_adatoms(
        slab,
        adatom_symbol,
        coverage_fraction,
        calculator=None,
        n_variants=5,
        adsorption_height=1.8,
        seed=None,
        results_dir="results",
        config=None,
        relaxation_mode=None,
        relaxation_optimizer=None,
        relaxation_fmax=None,
        relaxation_steps=None,
    ):
        captured["kwargs"] = {
            "calculator": calculator,
            "relaxation_mode": relaxation_mode,
            "relaxation_optimizer": relaxation_optimizer,
            "relaxation_fmax": relaxation_fmax,
            "relaxation_steps": relaxation_steps,
        }
        return slab

    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        _fake_setup_single_model,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.create_slab_from_bulk",
        _fake_create_slab_from_bulk,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.deposit_adatoms",
        _fake_deposit_adatoms,
    )

    prepare_substrate(
        bulk_id="mp-33",
        adatom_symbol="Au",
        adatom_coverage=0.2,
        config=AdsorptionConfig(device="cpu"),
        adatom_relaxation_mode="full",
        adatom_relaxation_optimizer="fire",
        adatom_relaxation_fmax=0.04,
        adatom_relaxation_steps=77,
    )

    assert captured["kwargs"]["calculator"] is fake_calc
    assert captured["kwargs"]["relaxation_mode"] == "full"
    assert captured["kwargs"]["relaxation_optimizer"] == "fire"
    assert captured["kwargs"]["relaxation_fmax"] == 0.04
    assert captured["kwargs"]["relaxation_steps"] == 77


def test_prepare_substrate_relaxes_loaded_slab_before_finalize(monkeypatch):
    captured: dict = {}
    fake_calc = object()
    base = SlabContainer(make_slab(symbol="Ru"))

    def _fake_setup_single_model(model_name, device):
        captured["setup"] = (model_name, device)
        return fake_calc, None

    def _fake_relax_substrate(slab, calculator, config=None, **kwargs):
        captured["relax_kwargs"] = {
            "calculator": calculator,
            "config": config,
            **kwargs,
        }
        return slab

    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        _fake_setup_single_model,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.relax_substrate",
        _fake_relax_substrate,
    )

    prepare_substrate(
        slab=base,
        config=AdsorptionConfig(model_name="uma-s-1p1", device="cpu"),
        slab_relaxation_mode="ionic_only",
        slab_relaxation_optimizer="bfgs",
        slab_relaxation_fmax=0.03,
        slab_relaxation_steps=99,
    )

    assert captured["setup"] == ("uma-s-1p1", "cpu")
    assert captured["relax_kwargs"]["calculator"] is fake_calc
    assert captured["relax_kwargs"]["relaxation_mode"] == "ionic_only"
    assert captured["relax_kwargs"]["relaxation_optimizer"] == "bfgs"
    assert captured["relax_kwargs"]["relaxation_fmax"] == 0.03
    assert captured["relax_kwargs"]["relaxation_steps"] == 99


def test_prepare_substrate_passes_enforce_top_layer_fraction(monkeypatch):
    captured: dict = {}
    fake_calc = object()

    def _fake_setup_single_model(model_name, device):
        return fake_calc, None

    def _fake_create_slab_from_bulk(**kwargs):
        return SlabContainer(make_slab(symbol="Ru"))

    def _fake_substitute_alloy(
        slab,
        host_symbol,
        guest_symbol,
        guest_fraction,
        calculator=None,
        n_variants=5,
        seed=None,
        relax=True,
        enforce_top_layer_fraction=False,
        top_layer_tolerance=None,
        config=None,
        results_dir="results",
    ):
        captured["enforce_top_layer_fraction"] = enforce_top_layer_fraction
        return slab

    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        _fake_setup_single_model,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.create_slab_from_bulk",
        _fake_create_slab_from_bulk,
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.substitute_alloy",
        _fake_substitute_alloy,
    )

    prepare_substrate(
        bulk_id="mp-33",
        alloy_host="Ru",
        alloy_guest="Cu",
        alloy_fraction=0.5,
        enforce_top_layer_fraction=True,
        config=AdsorptionConfig(device="cpu"),
    )

    assert captured["enforce_top_layer_fraction"] is True


def test_prepare_substrate_slab_input_skips_bulk_load(monkeypatch):
    captured: dict = {}
    base = SlabContainer(make_slab(symbol="Ru"))

    def _fake_create_slab_from_bulk(**kwargs):
        captured["create_called"] = True
        return base

    def _fake_deposit_adatoms(slab, adatom_symbol, coverage_fraction, **kwargs):
        captured["deposit_called"] = True
        return slab

    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.create_slab_from_bulk",
        _fake_create_slab_from_bulk,
    )
    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        lambda *a, **k: (object(), None),
    )
    monkeypatch.setattr(
        "metalsurfer.surface_prep.prep.deposit_adatoms",
        _fake_deposit_adatoms,
    )

    prepare_substrate(
        slab=base,
        adatom_symbol="Sn",
        adatom_coverage=0.1,
        config=AdsorptionConfig(slab_relaxation_mode="none"),
    )

    assert "create_called" not in captured
    assert captured["deposit_called"] is True


def test_finalize_substrate_applies_pbc_and_constraints():
    from metalsurfer.surface_prep import apply_material_pbc, finalize_substrate

    base = make_slab()
    base.set_pbc([True, True, True])
    config = AdsorptionConfig(material_type="slab")

    finalized = finalize_substrate(
        base,
        config,
        align=False,
        relax_top_layer=True,
    )

    assert list(finalized.atoms.get_pbc()) == [True, True, False]
    assert finalized.atoms.constraints
    apply_material_pbc(base, "porous")
    assert list(base.get_pbc()) == [True, True, True]

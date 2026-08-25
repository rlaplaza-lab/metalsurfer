"""Tests for prepare_substrate relaxation option forwarding."""

import pytest

from metalsurfer.config import AdsorptionConfig
from metalsurfer.surface_prep import SlabContainer, prepare_substrate

from .conftest import make_slab


def _patch_prepare_substrate(
    monkeypatch,
    *,
    create=True,
    deposit=None,
    alloy=None,
    relax=None,
):
    """Common monkeypatches for prepare_substrate forwarding tests."""
    captured: dict = {}
    fake_calc = object()

    def _fake_setup(model_name, device, task_name="oc25"):
        captured["setup"] = (model_name, device)
        captured["setup_task_name"] = task_name
        return fake_calc, None

    monkeypatch.setattr(
        "metalsurfer.optimization.setup_single_model",
        _fake_setup,
    )

    if create is True:

        def _fake_create(**kwargs):
            captured["create_kwargs"] = kwargs
            return SlabContainer(make_slab(symbol="Ru"))

        monkeypatch.setattr(
            "metalsurfer.surface_prep.prep.create_slab_from_bulk",
            _fake_create,
        )
    elif callable(create):
        monkeypatch.setattr(
            "metalsurfer.surface_prep.prep.create_slab_from_bulk",
            create,
        )

    if deposit is not None:
        monkeypatch.setattr(
            "metalsurfer.surface_prep.prep.deposit_adatoms",
            deposit,
        )
    if alloy is not None:
        monkeypatch.setattr(
            "metalsurfer.surface_prep.prep.substitute_alloy",
            alloy,
        )
    if relax is not None:
        monkeypatch.setattr(
            "metalsurfer.surface_prep.prep.relax_substrate",
            relax,
        )
    return captured, fake_calc


def test_prepare_substrate_passes_slab_relaxation_options(monkeypatch):
    captured, fake_calc = _patch_prepare_substrate(monkeypatch)

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

    def _fake_deposit(
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

    _patch_prepare_substrate(monkeypatch, deposit=_fake_deposit)

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

    assert captured["kwargs"]["calculator"] is not None
    assert captured["kwargs"]["relaxation_mode"] == "full"
    assert captured["kwargs"]["relaxation_optimizer"] == "fire"
    assert captured["kwargs"]["relaxation_fmax"] == 0.04
    assert captured["kwargs"]["relaxation_steps"] == 77


def test_prepare_substrate_forwards_relax_kwargs_for_loaded_slab(monkeypatch):
    captured: dict = {}
    base = SlabContainer(make_slab(symbol="Ru"))

    def _fake_relax(slab, calculator, config=None, **kwargs):
        captured["relax_kwargs"] = {
            "calculator": calculator,
            "config": config,
            **kwargs,
        }
        return slab

    shared, fake_calc = _patch_prepare_substrate(
        monkeypatch,
        create=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("should not load bulk")
        ),
        relax=_fake_relax,
    )

    prepare_substrate(
        slab=base,
        config=AdsorptionConfig(model_name="uma-s-1p1", device="cpu"),
        slab_relaxation_mode="ionic_only",
        slab_relaxation_optimizer="bfgs",
        slab_relaxation_fmax=0.03,
        slab_relaxation_steps=99,
    )

    assert shared["setup"] == ("uma-s-1p1", "cpu")
    assert captured["relax_kwargs"]["calculator"] is fake_calc
    assert captured["relax_kwargs"]["relaxation_mode"] == "ionic_only"
    assert captured["relax_kwargs"]["relaxation_optimizer"] == "bfgs"
    assert captured["relax_kwargs"]["relaxation_fmax"] == 0.03
    assert captured["relax_kwargs"]["relaxation_steps"] == 99


def test_prepare_substrate_passes_enforce_top_layer_fraction(monkeypatch):
    captured: dict = {}

    def _fake_alloy(
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

    _patch_prepare_substrate(monkeypatch, alloy=_fake_alloy)

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

    def _fake_create(**kwargs):
        captured["create_called"] = True
        return base

    def _fake_deposit(slab, adatom_symbol, coverage_fraction, **kwargs):
        captured["deposit_called"] = True
        return slab

    _patch_prepare_substrate(monkeypatch, create=_fake_create, deposit=_fake_deposit)

    prepare_substrate(
        slab=base,
        adatom_symbol="Sn",
        adatom_coverage=0.1,
        config=AdsorptionConfig(slab_relaxation_mode="none"),
    )

    assert "create_called" not in captured
    assert captured["deposit_called"] is True


def test_finalize_substrate_applies_pbc_and_constraints():
    from ase.constraints import FixAtoms

    from metalsurfer.surface_prep import (
        apply_material_pbc,
        finalize_substrate,
        frozen_indices_from_constraints,
        identify_relaxable_surface_indices,
    )

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
    assert any(isinstance(c, FixAtoms) for c in finalized.atoms.constraints)

    frozen = set(frozen_indices_from_constraints(finalized.atoms))
    top = set(
        identify_relaxable_surface_indices(
            finalized.atoms, material_type="slab", tolerance=0.5
        )
    )
    assert frozen, "relax_top_layer=True must still freeze subsurface atoms"
    assert frozen.isdisjoint(top), "top-layer atoms must remain free to relax"
    assert frozen | top == set(range(len(finalized.atoms)))

    apply_material_pbc(base, "porous")
    assert list(base.get_pbc()) == [True, True, True]


def test_prepare_substrate_multi_element_alloy_requires_host(monkeypatch):
    alloy_slab = make_slab(n_layers=1)
    syms = alloy_slab.get_chemical_symbols()
    for i in range(len(syms)):
        if i % 2 == 1:
            syms[i] = "Cu"
    alloy_slab.set_chemical_symbols(syms)

    _patch_prepare_substrate(monkeypatch)

    with pytest.raises(ValueError, match="alloy_host must be set"):
        prepare_substrate(
            slab=SlabContainer(alloy_slab),
            alloy_guest="Au",
            alloy_fraction=0.25,
            config=AdsorptionConfig(device="cpu"),
        )


def test_prepare_substrate_single_element_alloy_infers_host(monkeypatch):
    captured: dict = {}

    def _fake_alloy(
        slab,
        host_symbol,
        guest_symbol,
        guest_fraction,
        calculator=None,
        **kwargs,
    ):
        captured["host_symbol"] = host_symbol
        return slab

    _patch_prepare_substrate(monkeypatch, alloy=_fake_alloy)

    prepare_substrate(
        bulk_id="mp-33",
        alloy_guest="Cu",
        alloy_fraction=0.25,
        config=AdsorptionConfig(device="cpu"),
    )

    assert captured["host_symbol"] == "Ru"

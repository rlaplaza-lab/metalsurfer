"""CI-fast e2e integration: public run modes + substrate×adsorbate matrix.

Exercises real Voronoi site discovery → placement → filter → I/O.
Only the MLIP boundary is stubbed (bootstrap / optimize / conformers).
Orbit reduction is skipped and site counts are capped for CI speed.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ase import Atoms

from metalsurfer._numeric_defaults import (
    MIN_CONTACT_RATIO_DEFAULT,
    MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
)
from metalsurfer.campaign_schema import load_campaign_yaml
from metalsurfer.campaigns import (
    run_adsorption,
    run_adsorption_bo,
    run_campaign,
    run_saturation,
    run_saturation_bo,
)
from metalsurfer.config import AdsorptionConfig, BOConfig, resolved_bo_eval_budget
from metalsurfer.models import (
    BindingCampaignResult,
    ReferenceEnergies,
    SaturationCampaignResult,
)
from metalsurfer.placement import check_initial_placement_distance
from metalsurfer.placement.geometry import check_adsorbate_separation
from metalsurfer.placement.site_context import (
    SiteContext,
    resolve_site_context_for_sampling,
)
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow.shared import ScreeningRunBootstrap

from .conftest import (
    E_ADS_IDENTITY_TOL,
    assert_no_intramolecular_clashes,
    assert_paths_exist,
    assert_water_oh_hh_geometry,
    make_h2,
    make_nanoparticle,
    make_porous_framework,
    make_slab,
    make_water,
    mock_calculator,
    pair_distance,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "campaigns"

E_SLAB = -200.0
E_MOL = -10.0
E_ADS_BINDING = -0.8
E_ADS_NONBINDING = 0.5
MOL_NAME = "water"
MOL_SMILES = "O"


def _tiny_config(**overrides: Any) -> AdsorptionConfig:
    """Near-default config for stubbed-MLIP CI e2e.

    Only CI-speed overrides differ from :class:`AdsorptionConfig` defaults:
    ``device="cpu"``, tiny conformer/placement counts, short stage steps, and a
    relaxed ``max_force_convergence`` (stub optimizer does not attach forces).
    Physics gates (desorption, topology, energy cap, contact) stay at defaults.
    """
    defaults: dict[str, Any] = {
        "material_type": "slab",
        "seed": 42,
        "num_conformers": 1,
        "num_placements": 8,
        "device": "cpu",
        "stage1_steps": 2,
        "stage2_steps": 2,
        "max_force_convergence": 1.0,
    }
    defaults.update(overrides)
    return AdsorptionConfig(**defaults)


def _bo_config(**overrides: Any) -> AdsorptionConfig:
    bo = overrides.pop("bo", None)
    if bo is None:
        bo_kwargs = {
            "initial_random": overrides.pop("initial_random", 3),
            "batch_size": overrides.pop("batch_size", 2),
            "total_budget": overrides.pop("total_budget", 2),
        }
        extra_bo = {
            k: overrides.pop(k)
            for k in list(overrides)
            if k
            in {
                "initial_random",
                "batch_size",
                "total_budget",
                "acquisition",
                "surrogate",
                "initial_sampling",
                "ucb_kappa",
                "transfer",
            }
        }
        bo_kwargs.update(extra_bo)
        bo = BOConfig(**bo_kwargs)
    return _tiny_config(
        bo=bo, num_placements=overrides.pop("num_placements", 8), **overrides
    )


def _substrate(material_type: str) -> SlabContainer:
    if material_type == "slab":
        # Default 4×4×3 passes image-separation checks; orbit reduction is
        # skipped in the harness so this stays CI-fast.
        return SlabContainer(make_slab())
    if material_type == "nanoparticle":
        return SlabContainer(make_nanoparticle())
    if material_type == "porous":
        atoms = make_porous_framework()
        # Calculator requires c ≥ 18 Å for porous cells.
        cell = atoms.get_cell().array.copy()
        if float(np.linalg.norm(cell[2])) < 18.0:
            cell[2] = cell[2] / np.linalg.norm(cell[2]) * 20.0
            atoms.set_cell(cell, scale_atoms=False)
        return SlabContainer(atoms)
    raise ValueError(f"unknown material_type: {material_type}")


def _conformers_for_smiles(smiles: str) -> tuple[list[Atoms], list[float]]:
    if smiles in {"[H][H]", "H2"}:
        return [make_h2()], [E_MOL]
    return [make_water()], [E_MOL]


class _StubHarness:
    """MLIP-boundary stubs: real Voronoi sites/placement; preserve geometry on optimize."""

    def __init__(
        self,
        slab: SlabContainer,
        *,
        molecule_name: str = MOL_NAME,
        saturation_schedule: bool = False,
    ) -> None:
        self.slab = slab
        self.molecule_name = molecule_name
        self.saturation_schedule = saturation_schedule
        self.optimize_calls = 0
        self.calculator = mock_calculator(energy=E_SLAB, n_atoms=len(slab.atoms))
        self.ref = ReferenceEnergies(
            slab_energy=E_SLAB,
            molecule_energies={molecule_name: E_MOL},
        )

    def e_ads_for_call(self) -> float:
        if not self.saturation_schedule:
            return E_ADS_BINDING
        if self.optimize_calls <= 1:
            return E_ADS_BINDING
        return E_ADS_NONBINDING

    def fake_optimize(
        self,
        combined_atoms_list,
        _slab,
        _ts_model,
        config=None,
        base_slab_for_frozen=None,
        saturation_reuse=False,
    ):
        self.optimize_calls += 1
        e_ads = self.e_ads_for_call()
        out: list[Atoms | None] = []
        for atoms in combined_atoms_list:
            # Keep real placement geometry; only attach a stub energy.
            placed = atoms.copy()
            placed.calc = mock_calculator(
                energy=E_SLAB + E_MOL + e_ads,
                n_atoms=len(placed),
            )
            out.append(placed)
        return out

    def apply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def bootstrap(_slab, pairs, _config):
            mol_energies = {name: E_MOL for _smiles, name in pairs}
            return ScreeningRunBootstrap(
                calculator=self.calculator,
                ts_model=None,
                ref=ReferenceEnergies(
                    slab_energy=E_SLAB,
                    molecule_energies=mol_energies,
                ),
                t_ref_s=0.01,
                slab=self.slab,
            )

        def resolve_sites_ci_fast(slab_atoms, config, *, symmetry_broken: bool):
            # Keep real Voronoi clustering; skip orbit reduction (pathological on
            # hand-built / large porous fixtures in CI). Cap site count so BO /
            # enumeration stay fast with large porous cells.
            ctx = resolve_site_context_for_sampling(
                slab_atoms,
                config,
                symmetry_broken=True,
            )
            max_sites = 32
            if ctx.use_sites and len(ctx.sites) > max_sites:
                return SiteContext(
                    sites=list(ctx.sites[:max_sites]),
                    use_sites=True,
                    source=ctx.source,
                    raw_unclustered=ctx.raw_unclustered,
                )
            return ctx

        monkeypatch.setattr(
            "metalsurfer.campaigns._bootstrap_screening_run",
            bootstrap,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.saturation._bootstrap_screening_run",
            bootstrap,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.optimize_adsorbate_slab_batched",
            self.fake_optimize,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.clear_autobatcher_cache",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.create_conformers_from_smiles",
            lambda smiles, *_a, **_k: _conformers_for_smiles(smiles),
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.resolve_site_context_for_sampling",
            resolve_sites_ci_fast,
        )
        # Saturation recomputes slab energy each step; keep E_slab fixed for stub math.
        monkeypatch.setattr(
            "metalsurfer.workflow.saturation._compute_slab_energy",
            lambda *_a, **_k: E_SLAB,
        )


def _distance_window(material_type: str) -> tuple[float, float]:
    # Match production MIN_INITIAL_DISTANCE_DEFAULT; upper is desorption threshold.
    _ = material_type
    return MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM, 4.0


def _assert_no_intramolecular_clashes(adsorbate: Atoms, slab: Atoms) -> None:
    assert_no_intramolecular_clashes(adsorbate, slab)


def _assert_dissociative_h2_geometry(ads: Atoms, slab: Atoms) -> None:
    assert len(ads) == 2
    pos = ads.get_positions()
    hh = pair_distance(
        pos[0],
        pos[1],
        cell=np.asarray(slab.get_cell(), dtype=float),
        pbc=list(slab.get_pbc()),
    )
    # Dissociative hollow-pair starts are stretched vs molecular H2 (~0.74 Å).
    # Fixture slab hollow pairs sit near a*√2/3 ≈ 1.27 Å (above the 1.0 Å floor).
    assert hh >= 1.0, f"dissociative H–H should be non-molecular, got {hh:.3f} Å"
    assert hh <= 3.5, f"dissociative H–H unphysically far: {hh:.3f} Å"


def _assert_survivor_physics(
    result,
    *,
    material_type: str = "slab",
    expected_symbols: list[str] | None = None,
    dissociative: bool = False,
    base_slab_size: int | None = None,
) -> None:
    """Critical physics gates for campaign survivors (stubbed or real MLIP)."""
    assert np.isfinite(result.energy_adsorption)
    assert result.energy_adsorption == pytest.approx(
        result.energy_adslab - result.energy_slab - result.energy_adsorbate,
        abs=E_ADS_IDENTITY_TOL,
    )
    d_lo, d_hi = _distance_window(material_type)
    assert d_lo <= result.distance <= d_hi, (
        f"Survivor must remain adsorbed ({d_lo}–{d_hi} Å), got {result.distance:.2f}"
    )
    # Must not be desorbed relative to the default binding threshold.
    assert result.distance <= 4.0 + 1e-6, (
        f"Survivor distance {result.distance:.2f} Å exceeds desorption threshold"
    )

    ads = result.atoms[result.slab_size :]
    slab_part = result.atoms[: result.slab_size]
    assert len(ads) >= 1

    if expected_symbols is not None:
        assert sorted(ads.get_chemical_symbols()) == sorted(expected_symbols)

    # Same covalent-contact gate used at placement time (production path).
    # Saturation slabs carry co-adsorbates: production excludes them from the
    # substrate gate (``exclude_slab_atoms``) and validates them with the looser
    # adsorbate-separation gate instead, so mirror that split here.
    n_substrate = (
        int(base_slab_size) if base_slab_size is not None else int(result.slab_size)
    )
    substrate = slab_part[:n_substrate]
    ok, min_d, reason = check_initial_placement_distance(
        ads,
        substrate,
        min_distance=MIN_INITIAL_DISTANCE_DEFAULT_ANGSTROM,
        min_contact_ratio=MIN_CONTACT_RATIO_DEFAULT,
        material_type=material_type,
    )
    assert ok, (
        f"survivor fails placement distance gate: min_d={min_d:.3f} reason={reason}"
    )
    if n_substrate < len(slab_part):
        sep_ok, sep_d = check_adsorbate_separation(
            ads,
            np.asarray(slab_part.get_positions()[n_substrate:], dtype=float),
            cell=np.asarray(slab_part.get_cell(), dtype=float),
            pbc=list(slab_part.get_pbc()),
        )
        assert sep_ok, f"survivor overlaps a co-adsorbate: min_d={sep_d:.3f}"
    # result.distance and the gate share MIC semantics; allow small path differences
    # (e.g. saturation slabs that already contain co-adsorbates).
    assert abs(float(result.distance) - float(min_d)) < 0.15, (
        f"result.distance={result.distance:.3f} vs gate min_d={min_d:.3f}"
    )

    if not dissociative:
        _assert_no_intramolecular_clashes(ads, slab_part)

    if material_type == "slab" and not dissociative:
        # Compare against metal (or framework) atoms only — ignore co-adsorbates.
        light = {"H", "C", "N", "O", "F", "S", "Cl", "P"}
        metal_idx = [
            i
            for i, sym in enumerate(slab_part.get_chemical_symbols())
            if sym not in light
        ]
        if metal_idx:
            ads_z = ads.get_positions()[:, 2]
            slab_z = slab_part.get_positions()[metal_idx, 2]
            assert float(np.min(ads_z)) > float(np.max(slab_z)) - 0.05, (
                f"adsorbate not above surface: min_ads_z={ads_z.min():.3f}, "
                f"max_slab_z={slab_z.max():.3f}"
            )

    desc = result.placement_descriptor
    assert desc is not None
    for field in ("x_abs", "y_abs", "z_abs", "surface_ref_z_abs"):
        val = getattr(desc, field)
        assert val is not None and np.isfinite(val), f"{field}={val}"
    if (
        not dissociative
        and desc.z_abs is not None
        and desc.surface_ref_z_abs is not None
    ):
        if material_type == "slab":
            assert float(desc.z_abs) >= float(desc.surface_ref_z_abs) - 0.05, (
                f"z_abs={desc.z_abs:.3f} below surface_ref_z_abs={desc.surface_ref_z_abs:.3f}"
            )
        else:
            # Local-normal materials: COM height along the site normal equals
            # surface_ref + z_offset (clearance lift included for NP).
            assert desc.z_offset is not None
            assert float(desc.z_offset) > 0.0

    if dissociative:
        assert desc.orientation_type == "dissociative"
        assert len(ads) == 2
        assert sorted(ads.get_chemical_symbols()) == ["H", "H"]
        _assert_dissociative_h2_geometry(ads, slab_part)
    elif expected_symbols is not None and sorted(expected_symbols) == ["H", "H", "O"]:
        assert desc.orientation_type == "round"
        assert_water_oh_hh_geometry(
            ads,
            cell=np.asarray(slab_part.get_cell(), dtype=float),
            pbc=list(slab_part.get_pbc()),
        )


def _assert_binding_yield(
    results: list,
    *,
    n_requested: int,
    min_success_rate: float,
    min_absolute: int = 2,
) -> None:
    """Require a decent survivor count vs requested placements (not a single lucky hit)."""
    assert n_requested >= 1
    n_ok = len(results)
    min_ok = max(min_absolute, int(math.ceil(min_success_rate * n_requested)))
    assert n_ok >= min_ok, (
        f"Placement success too low: {n_ok}/{n_requested} survivors "
        f"(need >= {min_ok}, rate {min_success_rate:.0%})"
    )
    rate = n_ok / float(n_requested)
    assert rate + 1e-12 >= min_success_rate, (
        f"Placement success rate {rate:.0%} < required {min_success_rate:.0%} "
        f"({n_ok}/{n_requested})"
    )
    site_ids = {
        r.placement_descriptor.site_index
        for r in results
        if r.placement_descriptor is not None
        and r.placement_descriptor.site_index is not None
        and int(r.placement_descriptor.site_index) >= 0
    }
    if n_ok >= 2:
        assert len(site_ids) >= 2, (
            f"Survivors should cover multiple sites, got site_ids={sorted(site_ids)}"
        )
    placement_ids = {int(r.placement_id) for r in results}
    assert len(placement_ids) == n_ok, "Survivor placement_id values must be unique"


def _assert_binding_artifacts(results_dir: Path) -> None:
    assert_paths_exist(
        results_dir,
        [
            "adsorption_energies_detailed.csv",
            "run_metadata.json",
        ],
    )
    xyz_root = results_dir / "xyz_structures"
    assert xyz_root.is_dir(), f"Missing {xyz_root}"
    assert any(xyz_root.rglob("*.xyz")), "Expected at least one XYZ structure"


def _assert_saturation_artifacts(results_dir: Path) -> None:
    assert_paths_exist(
        results_dir,
        [
            "saturation_summary.csv",
            "run_metadata.json",
        ],
    )


# ---------------------------------------------------------------------------
# Substrate × adsorbate matrix (run_adsorption)
# ---------------------------------------------------------------------------

_MATRIX_CASES = (
    # material, smiles, name, config overrides, symbols, dissociative, min_success_rate
    ("slab", "O", "water", {}, ["H", "H", "O"], False, 0.75),
    (
        "slab",
        "[H][H]",
        "H2",
        {
            # Near-default dissociative slab settings (same as demos).
            "enable_dissociative_placement": True,
            "skip_topology_check": True,
        },
        ["H", "H"],
        True,
        1.0,
    ),
    ("nanoparticle", "O", "water", {}, ["H", "H", "O"], False, 0.75),
    ("porous", "O", "water", {}, ["H", "H", "O"], False, 0.75),
)


@pytest.mark.parametrize(
    (
        "material_type",
        "smiles",
        "mol_name",
        "config_overrides",
        "expected_symbols",
        "dissociative",
        "min_success_rate",
    ),
    _MATRIX_CASES,
    ids=["slab_water", "slab_h2", "np_water", "porous_water"],
)
def test_run_adsorption_substrate_matrix(
    tmp_path,
    monkeypatch,
    material_type,
    smiles,
    mol_name,
    config_overrides,
    expected_symbols,
    dissociative,
    min_success_rate,
):
    monkeypatch.chdir(tmp_path)
    slab = _substrate(material_type)
    harness = _StubHarness(slab, molecule_name=mol_name)
    harness.apply(monkeypatch)
    surface_type = f"e2e_{material_type}_{mol_name}"
    config = _tiny_config(material_type=material_type, **config_overrides)
    n_requested = int(config.num_placements or 8)
    assert config.skip_desorption_check is False
    assert config.max_adsorption_energy == 5.0
    if not dissociative:
        assert config.skip_topology_check is False

    campaign = run_adsorption(
        slab=slab,
        molecules=[(smiles, mol_name)],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
        save_results=True,
        write_settings=True,
    )

    assert isinstance(campaign, BindingCampaignResult)
    assert campaign.mode == "non_bo"
    assert campaign.n_molecules == 1
    assert len(campaign.run_results) == 1
    results = campaign.run_results[0].results
    _assert_binding_yield(
        results,
        n_requested=n_requested,
        min_success_rate=min_success_rate,
    )
    assert campaign.total_configurations == len(results)
    assert campaign.molecule_summaries[0].n_valid_placements == len(results)
    for r in results:
        _assert_survivor_physics(
            r,
            material_type=material_type,
            expected_symbols=expected_symbols,
            dissociative=dissociative,
        )
        assert r.energy_adsorption == pytest.approx(
            E_ADS_BINDING, abs=E_ADS_IDENTITY_TOL
        )
    _assert_binding_artifacts(tmp_path / f"results_{surface_type}")


def test_run_adsorption_rejects_crushed_geometries(tmp_path, monkeypatch):
    """Critical: overlapping post-relax geometries must not become survivors."""
    monkeypatch.chdir(tmp_path)
    slab = _substrate("slab")
    harness = _StubHarness(slab)
    harness.apply(monkeypatch)

    def crush_optimize(combined_atoms_list, _slab, *_a, **_k):
        harness.optimize_calls += 1
        out: list[Atoms | None] = []
        for atoms in combined_atoms_list:
            crushed = atoms.copy()
            pos = crushed.get_positions().copy()
            n_ads = len(crushed) - len(slab.atoms)
            # Drive adsorbate into the substrate → geometry / desorption fail.
            pos[-n_ads:] = pos[:n_ads] + 0.02
            crushed.set_positions(pos)
            crushed.calc = mock_calculator(
                energy=E_SLAB + E_MOL + E_ADS_BINDING,
                n_atoms=len(crushed),
            )
            out.append(crushed)
        return out

    monkeypatch.setattr(
        "metalsurfer.workflow.shared.optimize_adsorbate_slab_batched",
        crush_optimize,
    )

    campaign = run_adsorption(
        slab=slab,
        molecules=[(MOL_SMILES, MOL_NAME)],
        config=_tiny_config(num_placements=4),
        surface_type="e2e_crush_reject",
        skip_existing=False,
        save_results=False,
        write_settings=False,
    )
    assert campaign.total_configurations == 0
    assert campaign.run_results == []
    assert campaign.molecule_summaries
    assert campaign.molecule_summaries[0].n_valid_placements == 0
    assert "water" in campaign.failure_summaries


# ---------------------------------------------------------------------------
# API e2e — BO adsorption + saturation modes (slab + water)
# Non-BO adsorption mode/n_molecules covered by substrate matrix above.
# ---------------------------------------------------------------------------


def _stubbed_campaign(
    tmp_path,
    monkeypatch,
    *,
    api_fn,
    config: AdsorptionConfig,
    surface_type: str,
    saturation_schedule: bool = False,
):
    monkeypatch.chdir(tmp_path)
    slab = _substrate("slab")
    harness = _StubHarness(slab, saturation_schedule=saturation_schedule)
    harness.apply(monkeypatch)
    return api_fn(
        slab=slab,
        molecules=[(MOL_SMILES, MOL_NAME)],
        config=config,
        surface_type=surface_type,
        skip_existing=False,
        save_results=True,
        write_settings=True,
    )


def _assert_binding_api_campaign(
    campaign,
    config: AdsorptionConfig,
    tmp_path: Path,
    surface_type: str,
    *,
    mode: str,
    min_success_rate: float,
    min_absolute: int | None = None,
) -> None:
    assert isinstance(campaign, BindingCampaignResult)
    assert campaign.mode == mode
    assert campaign.n_molecules == 1
    assert len(campaign.run_results) == 1
    results = campaign.run_results[0].results
    # BO evaluates resolved_bo_eval_budget candidates, not necessarily num_placements.
    n_requested = (
        resolved_bo_eval_budget(config)
        if mode == "bo"
        else int(config.num_placements or 8)
    )
    kwargs: dict[str, Any] = {
        "n_requested": n_requested,
        "min_success_rate": min_success_rate,
    }
    if min_absolute is not None:
        kwargs["min_absolute"] = min_absolute
    _assert_binding_yield(results, **kwargs)
    assert campaign.total_configurations == len(results)
    for r in results:
        _assert_survivor_physics(r, expected_symbols=["H", "H", "O"])
        assert r.energy_adsorption == pytest.approx(
            E_ADS_BINDING, abs=E_ADS_IDENTITY_TOL
        )
    _assert_binding_artifacts(tmp_path / f"results_{surface_type}")


def _assert_saturation_api_campaign(
    campaign,
    tmp_path: Path,
    surface_type: str,
    *,
    mode: str,
    require_nonbinding_later_step: bool = False,
) -> None:
    assert isinstance(campaign, SaturationCampaignResult)
    assert campaign.mode == mode
    assert len(campaign.runs) == 1
    run = campaign.runs[0]
    assert len(run.steps) >= 1
    base_slab_size = int(run.steps[0].best_result.slab_size)
    for step in run.steps:
        _assert_survivor_physics(
            step.best_result,
            expected_symbols=["H", "H", "O"],
            base_slab_size=base_slab_size,
        )
    assert run.steps[0].best_result.energy_adsorption == pytest.approx(
        E_ADS_BINDING, abs=1e-6
    )
    if require_nonbinding_later_step and len(run.steps) >= 2:
        assert run.steps[-1].best_result.energy_adsorption >= 0.0
    _assert_saturation_artifacts(tmp_path / f"results_{surface_type}")


class TestRunModeApiE2E:
    def test_run_adsorption_bo(self, tmp_path, monkeypatch):
        surface_type = "e2e_adsorption_bo"
        config = _bo_config()
        campaign = _stubbed_campaign(
            tmp_path,
            monkeypatch,
            api_fn=run_adsorption_bo,
            config=config,
            surface_type=surface_type,
        )
        _assert_binding_api_campaign(
            campaign,
            config,
            tmp_path,
            surface_type,
            mode="bo",
            min_success_rate=1.0,
            min_absolute=2,
        )

    def test_run_saturation(self, tmp_path, monkeypatch):
        surface_type = "e2e_saturation"
        config = _tiny_config(saturation_max_steps=2)
        campaign = _stubbed_campaign(
            tmp_path,
            monkeypatch,
            api_fn=run_saturation,
            config=config,
            surface_type=surface_type,
            saturation_schedule=True,
        )
        _assert_saturation_api_campaign(
            campaign,
            tmp_path,
            surface_type,
            mode="non_bo",
            require_nonbinding_later_step=True,
        )

    def test_run_saturation_bo(self, tmp_path, monkeypatch):
        surface_type = "e2e_saturation_bo"
        config = _bo_config(saturation_max_steps=2)
        campaign = _stubbed_campaign(
            tmp_path,
            monkeypatch,
            api_fn=run_saturation_bo,
            config=config,
            surface_type=surface_type,
            saturation_schedule=True,
        )
        _assert_saturation_api_campaign(
            campaign,
            tmp_path,
            surface_type,
            mode="bo",
        )


# ---------------------------------------------------------------------------
# YAML API e2e — load_campaign_yaml → run_campaign → real run_* (stubbed MLIP)
# ---------------------------------------------------------------------------

_YAML_CASES = (
    ("smoke_adsorption.yaml", "smoke_adsorption", "adsorption"),
    ("smoke_adsorption_bo.yaml", "smoke_adsorption_bo", "adsorption_bo"),
    ("smoke_saturation.yaml", "smoke_saturation", "saturation"),
    ("smoke_saturation_bo.yaml", "smoke_saturation_bo", "saturation_bo"),
)


@pytest.mark.parametrize(
    ("yaml_name", "surface_type", "campaign_kind"),
    _YAML_CASES,
    ids=[c[2] for c in _YAML_CASES],
)
def test_run_campaign_yaml_run_mode(
    tmp_path, monkeypatch, yaml_name, surface_type, campaign_kind
):
    monkeypatch.chdir(tmp_path)
    slab = _substrate("slab")
    harness = _StubHarness(
        slab,
        saturation_schedule=campaign_kind.startswith("saturation"),
    )
    harness.apply(monkeypatch)

    monkeypatch.setattr(
        "metalsurfer.campaigns.prepare_substrate",
        lambda **_kwargs: slab,
    )

    doc = load_campaign_yaml(FIXTURES / yaml_name)
    run_campaign(doc, skip_existing=False)

    results_dir = tmp_path / f"results_{surface_type}"
    assert results_dir.is_dir(), f"Expected results dir for {campaign_kind}"
    assert (results_dir / "run_metadata.json").is_file()
    if campaign_kind.startswith("saturation"):
        assert (results_dir / "saturation_summary.csv").is_file()
    else:
        assert (results_dir / "adsorption_energies_detailed.csv").is_file()

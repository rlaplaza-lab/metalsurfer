"""CI integration tests: physical gates must apply (not just run without crashing).

These tests exercise real placement, validation, filters, and E_ads accounting with a
stubbed TorchSim optimizer so CI can falsify skipped physics without a GPU/MLIP.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import filter_results
from metalsurfer.models import PlacementSpec, ReferenceEnergies
from metalsurfer.placement import (
    check_initial_placement_distance,
    generate_placement_from_spec_with_reason,
)
from metalsurfer.placement._constants import _SITE_Z_OFFSET_FROM_SURFACE_RADIUS
from metalsurfer.placement.geometry import detect_vdw_overlaps
from metalsurfer.placement.orientation import _site_type_z_offset
from metalsurfer.placement.site_context import SiteContext, _get_unique_sites_for_specs
from metalsurfer.placement.site_types import Site
from metalsurfer.surface_prep import SlabContainer
from metalsurfer.workflow import process_molecule
from metalsurfer.workflow.shared import (
    _evaluate_optimized_candidate,
    _validate_adsorption,
)

from .conftest import (
    E_ADS_IDENTITY_TOL,
    assert_water_oh_hh_geometry,
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    mock_calculator,
    place_molecule_on_slab,
)

pytestmark = pytest.mark.integration


def _attach_calc(atoms: Atoms, energy: float) -> Atoms:
    atoms = atoms.copy()
    atoms.calc = mock_calculator(energy=energy, n_atoms=len(atoms))
    return atoms


# ---------------------------------------------------------------------------
# Placement physics
# ---------------------------------------------------------------------------


class TestPlacementPhysicsGates:
    def test_overlap_into_surface_is_rejected(self):
        """Adsorbate driven into the surface must fail contact / VDW gates."""
        slab = make_slab()
        water = make_water().copy()
        pos = water.get_positions().copy()
        pos -= np.mean(pos, axis=0)
        surface_z = float(np.max(slab.get_positions()[:, 2]))
        pos[:, 2] += surface_z + 0.35
        pos[:, 0] += 5.0
        pos[:, 1] += 5.0
        water.set_positions(pos)
        water.set_cell(slab.get_cell())
        water.set_pbc(slab.get_pbc())

        ok, _dist, reason = check_initial_placement_distance(
            water,
            slab,
            reject_vdw_overlaps=True,
            vdw_overlap_scale=1.0,
            material_type="slab",
        )
        assert not ok, "Physics gate must reject an adsorbate embedded in the surface"
        assert reason in {"too_close", "vdw_overlap"}

        overlaps, min_dist = detect_vdw_overlaps(water, slab, material_type="slab")
        assert len(overlaps) > 0
        assert min_dist < 1.5

    def test_hollow_sits_lower_than_atop_for_same_site(self):
        """Site-type z physics: hollow offset is more negative than atop."""
        slab = make_slab()
        config = AdsorptionConfig(
            material_type="slab",
            seed=42,
            num_conformers=1,
            num_placements=4,
            placement_z_range=(2.0, 3.0),
            placement_distance_recovery=False,
        )
        ctx = _get_unique_sites_for_specs(slab, config)
        assert ctx.use_sites and ctx.sites, "Need site discovery for height physics"
        site = ctx.sites[0]
        site_index = 0
        water = make_water()

        def _place(site_type: str):
            spec = PlacementSpec(
                conformer_index=0,
                orientation_type="round",
                face_flip=False,
                en_atom_index=None,
                site_index=site_index,
                site_type=site_type,
                tilt_deg=0.0,
                azimuth_deg=0.0,
                azimuth_in_plane_deg=0.0,
                z_fraction=0.5,
                placement_index=0 if site_type == "atop" else 1,
            )
            return generate_placement_from_spec_with_reason(
                spec, [water], slab, config, smiles="O", site_context=ctx
            )

        atop_result, atop_reason = _place("atop")
        hollow_result, hollow_reason = _place("hollow")
        assert atop_result is not None, f"atop placement failed: {atop_reason}"
        assert hollow_result is not None, f"hollow placement failed: {hollow_reason}"

        _, atop_desc = atop_result
        _, hollow_desc = hollow_result
        assert atop_desc.z_abs is not None and hollow_desc.z_abs is not None

        expected_delta = _site_type_z_offset(
            slab, site, "hollow"
        ) - _site_type_z_offset(slab, site, "atop")
        assert expected_delta < 0.0
        assert (
            _SITE_Z_OFFSET_FROM_SURFACE_RADIUS["hollow"]
            < (_SITE_Z_OFFSET_FROM_SURFACE_RADIUS["atop"])
        )
        assert hollow_desc.z_abs == pytest.approx(
            atop_desc.z_abs + expected_delta, abs=1e-5
        )


# ---------------------------------------------------------------------------
# Validation / filter physics
# ---------------------------------------------------------------------------


class TestValidationAndFilterPhysics:
    def test_desorption_gate_rejects_far_molecule(self):
        slab = make_slab()
        desorbed = place_molecule_on_slab(slab, make_water(), z_offset=10.0)
        config = AdsorptionConfig(binding_distance_threshold=4.0)
        ok, reason, _ = _validate_adsorption(desorbed, slab, config)
        assert not ok
        assert "desorb" in reason.lower()

        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-0.5,
                atoms=desorbed,
                slab_size=len(slab),
                distance=10.0,
            )
        ]
        filtered = filter_results(
            results,
            slab=slab,
            surface_symbols=["Ru"],
            reference_smiles="O",
            config=config,
        )
        assert filtered == []

    def test_decomposition_gate_rejects_split_water(self):
        slab = make_slab(n_layers=1)
        slab_z = float(np.max(slab.get_positions()[:, 2]))
        decomposed_mol = Atoms(
            "OH2",
            positions=[
                [5.0, 5.0, slab_z + 3.0],
                [0.0, 0.0, slab_z + 11.0],
                [9.0, 9.0, slab_z + 11.0],
            ],
        )
        decomposed = slab + decomposed_mol
        decomposed.set_cell(slab.get_cell())
        decomposed.set_pbc(slab.get_pbc())
        good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)

        results = [
            make_screening_result(
                molecule="water",
                placement_id=0,
                energy_adsorption=-1.0,
                atoms=good,
                slab_size=len(slab),
            ),
            make_screening_result(
                molecule="water",
                placement_id=1,
                energy_adsorption=-2.0,
                atoms=decomposed,
                slab_size=len(slab),
            ),
        ]
        config = AdsorptionConfig(
            connectivity_multiplier=1.3,
            skip_topology_check=False,
            skip_desorption_check=False,
        )
        filtered = filter_results(
            results,
            slab=slab,
            surface_symbols=["Ru"],
            reference_smiles="O",
            config=config,
        )
        assert len(filtered) == 1
        assert filtered[0].placement_id == 0


# ---------------------------------------------------------------------------
# Energy formula + cap
# ---------------------------------------------------------------------------


class TestAdsorptionEnergyPhysics:
    def test_e_ads_formula_and_energy_cap(self):
        slab = make_slab()
        good = place_molecule_on_slab(slab, make_water(), z_offset=2.5)
        e_slab = -200.0
        e_mol = -10.0
        e_ads_ok = -0.75
        e_ads_cap = 6.0  # above default max_adsorption_energy (5 eV)

        descriptor = make_placement_descriptor(placement_id=0)
        config = AdsorptionConfig(max_adsorption_energy=5.0)

        ok_atoms = _attach_calc(good, energy=e_slab + e_mol + e_ads_ok)
        result, failure = _evaluate_optimized_candidate(
            opt_atoms=ok_atoms,
            placement_id=0,
            descriptor=descriptor,
            molecule_name="water",
            slab_atoms=slab,
            config=config,
            E_slab=e_slab,
            E_mol=e_mol,
            surface_symbols=["Ru"],
        )
        assert failure is None
        assert result is not None
        assert result.energy_adsorption == pytest.approx(
            e_ads_ok, abs=E_ADS_IDENTITY_TOL
        )
        assert result.energy_adsorption == pytest.approx(
            result.energy_adslab - result.energy_slab - result.energy_adsorbate,
            abs=E_ADS_IDENTITY_TOL,
        )

        capped = _attach_calc(good, energy=e_slab + e_mol + e_ads_cap)
        result_cap, failure_cap = _evaluate_optimized_candidate(
            opt_atoms=capped,
            placement_id=1,
            descriptor=make_placement_descriptor(placement_id=1),
            molecule_name="water",
            slab_atoms=slab,
            config=config,
            E_slab=e_slab,
            E_mol=e_mol,
            surface_symbols=["Ru"],
        )
        assert result_cap is None
        assert failure_cap is not None
        assert failure_cap.stage == "energy_cap"
        assert "E_ads too high" in failure_cap.reason


# ---------------------------------------------------------------------------
# End-to-end process_molecule with stubbed optimizer
# ---------------------------------------------------------------------------


class TestProcessMoleculePhysicsSurvival:
    def test_only_physically_valid_geometries_survive(self, monkeypatch):
        """Stub optimizer returns good / overlapping / desorbed; only good survives."""
        slab_atoms = make_slab()
        # Give the slab realistic z-vacuum: the desorption check now uses the
        # calculator's 3D PBC (matching the energy geometry), so a lifted
        # molecule must be genuinely far along c to be detected as desorbed
        # rather than MIC-wrapped back near the surface on a thin cell.
        _cell = slab_atoms.get_cell()
        _cell[2, 2] = 60.0
        slab_atoms.set_cell(_cell)
        slab = SlabContainer(slab_atoms)
        e_slab = -200.0
        e_mol = -10.0
        e_ads_good = -0.8
        refs = ReferenceEnergies(
            slab_energy=e_slab,
            molecule_energies={"water": e_mol},
        )
        config = AdsorptionConfig(
            material_type="slab",
            seed=42,
            num_conformers=1,
            num_placements=6,
            # Stub optimizer: no real forces; keep physics gates at defaults.
            max_force_convergence=1.0,
        )

        calc = MagicMock()
        calc.get_potential_energy.return_value = e_mol
        calc.get_forces.side_effect = lambda: np.zeros((3, 3), dtype=float)

        top_z = float(np.max(slab_atoms.get_positions()[:, 2]))
        top_idx = int(np.argmax(slab_atoms.get_positions()[:, 2]))
        xy = slab_atoms.get_positions()[top_idx, :2]
        cheap_sites = SiteContext(
            sites=[
                Site(
                    xyz=np.array([xy[0], xy[1], top_z]),
                    normal=np.array([0.0, 0.0, 1.0]),
                    site_type="atop",
                    slab_indices=(top_idx,),
                    material_type="slab",
                    site_source="topology",
                    env_fingerprint=(),
                )
            ],
            use_sites=True,
            source="test",
        )

        def _fake_optimize(
            combined_atoms_list,
            _slab,
            _ts_model,
            config=None,
            base_slab_for_frozen=None,
            saturation_reuse=False,
        ):
            slab_size = len(slab.atoms)
            out: list[Atoms | None] = []
            for i, atoms in enumerate(combined_atoms_list):
                mode = i % 3
                if mode == 0:
                    # Known adsorbed geometry (do not rely on placer height vs threshold)
                    a = place_molecule_on_slab(
                        atoms[:slab_size], make_water(), z_offset=2.5
                    )
                elif mode == 1:
                    # Crush adsorbate onto slab atoms → geometry fail
                    a = atoms.copy()
                    pos = a.get_positions().copy()
                    n_ads = len(a) - slab_size
                    pos[slab_size:] = pos[:n_ads] + 0.02
                    a.set_positions(pos)
                else:
                    # Lift far above surface → desorption fail
                    a = atoms.copy()
                    pos = a.get_positions().copy()
                    pos[slab_size:, 2] += 20.0
                    a.set_positions(pos)
                a.calc = mock_calculator(
                    energy=e_slab + e_mol + e_ads_good, n_atoms=len(a)
                )
                out.append(a)
            return out

        monkeypatch.setattr(
            "metalsurfer.workflow.shared.optimize_adsorbate_slab_batched",
            _fake_optimize,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.clear_autobatcher_cache",
            lambda *a, **k: None,
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.create_conformers_from_smiles",
            lambda *_a, **_k: ([make_water()], [e_mol]),
        )
        monkeypatch.setattr(
            "metalsurfer.workflow.shared.resolve_site_context_for_sampling",
            lambda *_a, **_k: cheap_sites,
        )

        outcome = process_molecule(
            "O",
            "water",
            slab,
            calc,
            refs,
            ts_model=None,  # avoid TorchSim conformer scoring; optimizer is stubbed
            config=config,
            surface_type="physics_stub",
        )
        results = outcome.results

        # Identical stub "good" geometries collapse under RMSD dedup → one survivor.
        assert len(results) == 1, (
            f"Expected the identical 'good' stub modes to dedup to 1 survivor, "
            f"got {len(results)}"
        )
        for r in results:
            assert r.energy_adsorption == pytest.approx(
                e_ads_good, abs=E_ADS_IDENTITY_TOL
            )
            assert r.energy_adsorption == pytest.approx(
                r.energy_adslab - r.energy_slab - r.energy_adsorbate,
                abs=E_ADS_IDENTITY_TOL,
            )
            assert 1.5 <= r.distance <= 4.0, (
                f"Survivor must remain adsorbed (1.5–4 Å), got {r.distance:.2f}"
            )
            ads = r.atoms[r.slab_size :]
            slab_part = r.atoms[: r.slab_size]
            overlaps, _ = detect_vdw_overlaps(
                ads, slab_part, material_type="slab", vdw_scale=0.7
            )
            assert len(overlaps) == 0, "Survivors must not have hard VDW clashes"

            # Topology filter still on: water connectivity preserved
            assert len(ads) == 3
            assert sorted(ads.get_chemical_symbols()) == ["H", "H", "O"]
            assert_water_oh_hh_geometry(ads)
            assert r.placement_descriptor is not None
            assert r.placement_descriptor.orientation_type == "round"
            assert r.placement_descriptor.surface_ref_z_abs is not None
            assert r.placement_descriptor.z_abs is not None
            assert (
                float(r.placement_descriptor.z_abs)
                >= float(r.placement_descriptor.surface_ref_z_abs) - 0.05
            )

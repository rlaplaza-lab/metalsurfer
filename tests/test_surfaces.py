"""Tests for surface creation, alloy substitution, and adatom deposition."""

import os
import sys
import tempfile
import types

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from metalsurfer.config import AdsorptionConfig
from metalsurfer.exceptions import GeometryValidationError
from metalsurfer.io_results import _write_clean_xyz
from metalsurfer.surfaces import (
    DEFAULT_SLAB_TOP_VACUUM_ANG,
    SlabContainer,
    _molecule_diameter,
    _perpendicular_heights_2d,
    apply_surface_constraints,
    auto_resize_substrate_for_molecule,
    coerce_slab_container,
    compute_minimum_supercell,
    create_slab_from_atoms,
    create_slab_from_bulk,
    deposit_adatoms,
    ensure_slab_z_alignment,
    substitute_alloy,
    validate_substrate,
)

from .conftest import make_slab, make_water

# ---------------------------------------------------------------------------
# SlabContainer
# ---------------------------------------------------------------------------


class TestSlabContainer:
    def test_wraps_atoms(self):
        atoms = make_slab()
        sc = SlabContainer(atoms)
        assert sc.atoms is atoms

    def test_create_slab_from_atoms_copies(self):
        atoms = make_slab()
        sc = create_slab_from_atoms(atoms)
        assert isinstance(sc, SlabContainer)
        # should be a copy, not the same object
        assert sc.atoms is not atoms
        assert np.allclose(sc.atoms.get_positions(), atoms.get_positions())
        assert float(np.min(sc.atoms.get_positions()[:, 2])) == pytest.approx(0.0)


class TestSlabZAlignment:
    def test_ensure_slab_z_alignment_bottom_anchors_fcc111(self):
        centered = fcc111("Al", size=(2, 2, 2), vacuum=7.0)
        aligned = ensure_slab_z_alignment(centered)
        z_min = float(np.min(aligned.get_positions()[:, 2]))
        z_max = float(np.max(aligned.get_positions()[:, 2]))
        c_len = float(np.linalg.norm(aligned.get_cell()[2]))
        assert z_min == pytest.approx(0.0)
        assert c_len >= max(18.0, z_max + DEFAULT_SLAB_TOP_VACUUM_ANG)

    def test_coerce_slab_container_is_pure_wrap(self):
        centered = fcc111("Al", size=(2, 2, 2), vacuum=7.0)
        z_min_before = float(np.min(centered.get_positions()[:, 2]))
        container = coerce_slab_container(centered)
        assert float(np.min(container.atoms.get_positions()[:, 2])) == pytest.approx(
            z_min_before
        )
        assert container.atoms is not centered

    def test_create_slab_from_atoms_aligns_when_requested(self):
        centered = fcc111("Al", size=(2, 2, 2), vacuum=7.0)
        container = create_slab_from_atoms(centered, material_type="slab", align=True)
        assert float(np.min(container.atoms.get_positions()[:, 2])) == pytest.approx(
            0.0
        )

    def test_create_slab_from_atoms_skips_alignment_when_disabled(self):
        centered = fcc111("Al", size=(2, 2, 2), vacuum=7.0)
        z_min_before = float(np.min(centered.get_positions()[:, 2]))
        container = create_slab_from_atoms(
            centered, material_type="nanoparticle", align=False
        )
        assert float(np.min(container.atoms.get_positions()[:, 2])) == pytest.approx(
            z_min_before
        )

    def test_ensure_slab_z_alignment_with_fixatoms_constraint(self):
        from ase.constraints import FixAtoms

        atoms = make_slab(n_layers=2)
        z_min_before = float(np.min(atoms.get_positions()[:, 2]))
        assert z_min_before == pytest.approx(0.0)
        atoms.set_positions(
            atoms.get_positions() + np.array([0.0, 0.0, 5.0]), apply_constraint=False
        )
        atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))
        aligned = ensure_slab_z_alignment(atoms)
        assert float(np.min(aligned.get_positions()[:, 2])) == pytest.approx(0.0)

    def test_create_slab_from_bulk_applies_alignment(self, monkeypatch, tmp_path):
        core = types.ModuleType("fairchem.data.oc.core")

        class _FakeBulk:
            def __init__(self, bulk_src_id_from_db):
                self.bulk_src_id_from_db = bulk_src_id_from_db

        class _FakeSlab:
            atoms: Atoms

            @staticmethod
            def from_bulk_get_specific_millers(bulk, specific_millers):
                atoms = fcc111("Al", size=(2, 2, 2), vacuum=8.0)
                slab = types.SimpleNamespace()
                slab.atoms = atoms
                return [slab]

        core.Bulk = _FakeBulk
        core.Slab = _FakeSlab
        monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
        monkeypatch.setitem(
            sys.modules, "fairchem.data", types.ModuleType("fairchem.data")
        )
        monkeypatch.setitem(
            sys.modules, "fairchem.data.oc", types.ModuleType("fairchem.data.oc")
        )
        monkeypatch.setitem(sys.modules, "fairchem.data.oc.core", core)

        slab = create_slab_from_bulk(
            bulk_id="mp-33",
            miller_indices=(0, 0, 1),
            supercell=(1, 1, 1),
            results_dir=str(tmp_path),
            relaxation_mode="none",
        )
        assert float(np.min(slab.atoms.get_positions()[:, 2])) == pytest.approx(0.0)

    def test_create_slab_from_bulk_empty_candidates_raises(self, monkeypatch):
        core = types.ModuleType("fairchem.data.oc.core")

        class _FakeBulk:
            def __init__(self, bulk_src_id_from_db):
                self.bulk_src_id_from_db = bulk_src_id_from_db

        class _FakeSlab:
            @staticmethod
            def from_bulk_get_specific_millers(bulk, specific_millers):
                return []

        core.Bulk = _FakeBulk
        core.Slab = _FakeSlab
        monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
        monkeypatch.setitem(
            sys.modules, "fairchem.data", types.ModuleType("fairchem.data")
        )
        monkeypatch.setitem(
            sys.modules, "fairchem.data.oc", types.ModuleType("fairchem.data.oc")
        )
        monkeypatch.setitem(sys.modules, "fairchem.data.oc.core", core)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(GeometryValidationError, match="No slabs were generated"),
        ):
            create_slab_from_bulk(
                bulk_id="mp-23",
                miller_indices=(1, 1, 1),
                supercell=(1, 1, 1),
                results_dir=tmpdir,
            )

    def test_create_slab_from_bulk_relaxation_requires_calculator(self, monkeypatch):
        core = types.ModuleType("fairchem.data.oc.core")

        class _FakeBulk:
            def __init__(self, bulk_src_id_from_db):
                self.bulk_src_id_from_db = bulk_src_id_from_db

        class _FakeSlabObj:
            def __init__(self):
                self.atoms = make_slab(symbol="Ru")

        class _FakeSlab:
            @staticmethod
            def from_bulk_get_specific_millers(bulk, specific_millers):
                return [_FakeSlabObj()]

        core.Bulk = _FakeBulk
        core.Slab = _FakeSlab
        monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
        monkeypatch.setitem(
            sys.modules, "fairchem.data", types.ModuleType("fairchem.data")
        )
        monkeypatch.setitem(
            sys.modules, "fairchem.data.oc", types.ModuleType("fairchem.data.oc")
        )
        monkeypatch.setitem(sys.modules, "fairchem.data.oc.core", core)

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="requires a calculator"),
        ):
            create_slab_from_bulk(
                bulk_id="mp-23",
                miller_indices=(1, 1, 1),
                supercell=(1, 1, 1),
                results_dir=tmpdir,
                relaxation_mode="full",
            )


# ---------------------------------------------------------------------------
# substitute_alloy
# ---------------------------------------------------------------------------


class TestSubstituteAlloy:
    def _ru_slab(self):
        return SlabContainer(make_slab(symbol="Ru"))

    def test_zero_fraction_returns_base(self):
        slab = self._ru_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = substitute_alloy(
                slab, "Ru", "Cu", guest_fraction=0.0, results_dir=tmpdir
            )
        syms = result.atoms.get_chemical_symbols()
        assert all(s == "Ru" for s in syms)

    def test_full_replacement(self):
        slab = self._ru_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = substitute_alloy(
                slab, "Ru", "Cu", guest_fraction=1.0, results_dir=tmpdir
            )
        syms = result.atoms.get_chemical_symbols()
        assert all(s == "Cu" for s in syms)

    def test_partial_replacement_count(self):
        slab = self._ru_slab()
        n_total = len(slab.atoms)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = substitute_alloy(
                slab, "Ru", "Cu", guest_fraction=0.25, results_dir=tmpdir
            )
        syms = result.atoms.get_chemical_symbols()
        n_cu = syms.count("Cu")
        expected = int(round(n_total * 0.25))
        assert n_cu == expected

    def test_deterministic_with_same_seed(self):
        slab1 = self._ru_slab()
        slab2 = self._ru_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = substitute_alloy(
                slab1, "Ru", "Cu", guest_fraction=0.33, seed=123, results_dir=tmpdir
            )
            r2 = substitute_alloy(
                slab2, "Ru", "Cu", guest_fraction=0.33, seed=123, results_dir=tmpdir
            )
        assert r1.atoms.get_chemical_symbols() == r2.atoms.get_chemical_symbols()

    def test_different_seed_differs(self):
        slab1 = self._ru_slab()
        slab2 = self._ru_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = substitute_alloy(
                slab1, "Ru", "Cu", guest_fraction=0.33, seed=1, results_dir=tmpdir
            )
            r2 = substitute_alloy(
                slab2, "Ru", "Cu", guest_fraction=0.33, seed=999, results_dir=tmpdir
            )
        assert r1.atoms.get_chemical_symbols() != r2.atoms.get_chemical_symbols()

    def test_uses_config_seed_by_default(self):
        slab1 = self._ru_slab()
        slab2 = self._ru_slab()
        cfg = AdsorptionConfig(seed=77)
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = substitute_alloy(
                slab1, "Ru", "Cu", guest_fraction=0.33, config=cfg, results_dir=tmpdir
            )
            r2 = substitute_alloy(
                slab2, "Ru", "Cu", guest_fraction=0.33, seed=77, results_dir=tmpdir
            )
        assert r1.atoms.get_chemical_symbols() == r2.atoms.get_chemical_symbols()

    def test_writes_output_files(self):
        slab = self._ru_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            substitute_alloy(slab, "Ru", "Cu", guest_fraction=0.5, results_dir=tmpdir)
            assert os.path.exists(os.path.join(tmpdir, "clean_Ru_Cu_50_slab.xyz"))
            assert not os.path.exists(
                os.path.join(tmpdir, "clean_Ru_Cu_50_slab_POSCAR")
            )

    def test_writes_poscar_when_enabled(self):
        slab = self._ru_slab()
        cfg = AdsorptionConfig(write_vasp_inputs=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            substitute_alloy(
                slab,
                "Ru",
                "Cu",
                guest_fraction=0.5,
                config=cfg,
                results_dir=tmpdir,
            )
            assert os.path.exists(os.path.join(tmpdir, "clean_Ru_Cu_50_slab_POSCAR"))

    def test_no_host_atoms_raises(self):
        slab = SlabContainer(make_slab(symbol="Cu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            # Trying to replace "Ru" atoms in a pure Cu slab → 0 replacements
            result = substitute_alloy(
                slab, "Ru", "Sn", guest_fraction=0.5, results_dir=tmpdir
            )
            # guest_fraction of 0 host atoms = 0 replacements → returns base
            assert all(s == "Cu" for s in result.atoms.get_chemical_symbols())

    @pytest.mark.parametrize("bad_fraction", [-0.1, 1.01, -1.0, 2.0])
    def test_out_of_range_fraction_raises(self, bad_fraction):
        slab = self._ru_slab()
        with pytest.raises(ValueError, match="guest_fraction must be between 0 and 1"):
            substitute_alloy(slab, "Ru", "Cu", guest_fraction=bad_fraction)

    def test_accepts_plain_atoms_input(self):
        atoms = make_slab(symbol="Ru")
        with tempfile.TemporaryDirectory() as tmpdir:
            result = substitute_alloy(
                atoms,
                "Ru",
                "Cu",
                guest_fraction=0.25,
                seed=42,
                results_dir=tmpdir,
            )
        assert isinstance(result, SlabContainer)
        assert result.atoms.get_chemical_symbols().count("Cu") > 0


# ---------------------------------------------------------------------------
# deposit_adatoms
# ---------------------------------------------------------------------------


class TestDepositAdatoms:
    _NO_RELAX_CFG = AdsorptionConfig(slab_relaxation_mode="none")

    def _flat_slab(self, n: int = 16, symbol: str = "Ru"):
        """Single-layer test slab."""
        return SlabContainer(make_slab(nx=4, ny=4, n_layers=1, symbol=symbol))

    def _layered_slab(self):
        return SlabContainer(make_slab(nx=4, ny=4, n_layers=3))

    def test_adatoms_placed_above_surface(self):
        slab = self._layered_slab()
        z_max = float(np.max(slab.atoms.get_positions()[:, 2]))
        n_before = len(slab.atoms)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = deposit_adatoms(
                slab,
                "Sn",
                coverage_fraction=0.2,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
        assert len(result.atoms) > n_before
        new_positions = result.atoms.get_positions()[n_before:]
        assert all(z > z_max for z in new_positions[:, 2])

    def test_deterministic_with_same_seed(self):
        slab1 = self._layered_slab()
        slab2 = self._layered_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = deposit_adatoms(
                slab1,
                "Sn",
                coverage_fraction=0.3,
                seed=42,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
            r2 = deposit_adatoms(
                slab2,
                "Sn",
                coverage_fraction=0.3,
                seed=42,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
        assert np.allclose(
            r1.atoms.get_positions(), r2.atoms.get_positions(), atol=1e-10
        )

    def test_different_seed_differs(self):
        slab1 = self._layered_slab()
        slab2 = self._layered_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = deposit_adatoms(
                slab1,
                "Sn",
                coverage_fraction=0.3,
                seed=1,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
            r2 = deposit_adatoms(
                slab2,
                "Sn",
                coverage_fraction=0.3,
                seed=999,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
        # positions should differ (site selection is random)
        assert not np.allclose(
            r1.atoms.get_positions(), r2.atoms.get_positions(), atol=1e-6
        )

    def test_too_few_top_atoms_raises(self):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atoms.set_cell([5.0, 5.0, 10.0])
        atoms.set_pbc([True, True, True])
        slab = SlabContainer(atoms)
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(
                GeometryValidationError, match="Cannot identify top surface layer"
            ),
        ):
            deposit_adatoms(
                slab,
                "Sn",
                coverage_fraction=0.5,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )

    def test_writes_output_files(self):
        slab = self._layered_slab()
        with tempfile.TemporaryDirectory() as tmpdir:
            deposit_adatoms(
                slab,
                "Sn",
                coverage_fraction=0.2,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
            assert os.path.exists(os.path.join(tmpdir, "clean_slab_Sn20.xyz"))
            assert not os.path.exists(os.path.join(tmpdir, "clean_slab_Sn20_POSCAR"))

    def test_writes_poscar_when_enabled(self):
        slab = self._layered_slab()
        cfg = AdsorptionConfig(write_vasp_inputs=True, slab_relaxation_mode="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            deposit_adatoms(
                slab,
                "Sn",
                coverage_fraction=0.2,
                config=cfg,
                results_dir=tmpdir,
            )
            assert os.path.exists(os.path.join(tmpdir, "clean_slab_Sn20_POSCAR"))

    def test_uses_config_seed_by_default(self):
        slab1 = self._layered_slab()
        slab2 = self._layered_slab()
        cfg = AdsorptionConfig(seed=77, slab_relaxation_mode="none")
        with tempfile.TemporaryDirectory() as tmpdir:
            r1 = deposit_adatoms(
                slab1, "Sn", coverage_fraction=0.3, config=cfg, results_dir=tmpdir
            )
            r2 = deposit_adatoms(
                slab2,
                "Sn",
                coverage_fraction=0.3,
                seed=77,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
        assert np.allclose(
            r1.atoms.get_positions(), r2.atoms.get_positions(), atol=1e-10
        )

    @pytest.mark.parametrize("bad_fraction", [-0.1, 1.01, -1.0, 2.0])
    def test_out_of_range_coverage_raises(self, bad_fraction):
        slab = self._layered_slab()
        with pytest.raises(
            ValueError, match="coverage_fraction must be between 0 and 1"
        ):
            deposit_adatoms(slab, "Sn", coverage_fraction=bad_fraction)

    def test_zero_coverage_returns_unmodified(self):
        slab = self._layered_slab()
        n_before = len(slab.atoms)
        syms_before = slab.atoms.get_chemical_symbols()
        result = deposit_adatoms(slab, "Sn", coverage_fraction=0.0)
        assert len(result.atoms) == n_before
        assert result.atoms.get_chemical_symbols() == syms_before

    def test_accepts_plain_atoms_input(self):
        slab = self._layered_slab()
        n_before = len(slab.atoms)
        with tempfile.TemporaryDirectory() as tmpdir:
            result = deposit_adatoms(
                slab.atoms,
                "Sn",
                coverage_fraction=0.2,
                seed=42,
                config=self._NO_RELAX_CFG,
                results_dir=tmpdir,
            )
        assert isinstance(result, SlabContainer)
        assert len(result.atoms) > n_before

    def test_relaxation_mode_requires_calculator(self):
        slab = self._layered_slab()
        with pytest.raises(ValueError, match="requires a calculator"):
            deposit_adatoms(slab, "Sn", coverage_fraction=0.2, relaxation_mode="full")

    def test_relaxation_invoked_for_variants(self, monkeypatch):
        slab = self._layered_slab()

        calls: list[tuple[str, int, int]] = []

        def _fake_relax(
            atoms,
            calculator,
            *,
            mode,
            optimizer_name,
            fmax,
            steps,
            context,
        ):
            calls.append((mode, int(steps), len(atoms)))
            return atoms

        class _FakeCalculator:
            def __init__(self):
                self.results = {}

            def get_potential_energy(self, atoms=None, force_consistent=False):
                return float(len(atoms) if atoms is not None else 0.0)

        monkeypatch.setattr("metalsurfer.surfaces._relax_slab_structure", _fake_relax)
        with tempfile.TemporaryDirectory() as tmpdir:
            deposit_adatoms(
                slab,
                "Sn",
                coverage_fraction=0.2,
                calculator=_FakeCalculator(),
                results_dir=tmpdir,
                n_variants=3,
                relaxation_mode="full",
                relaxation_steps=12,
            )
        assert len(calls) == 3
        assert all(mode == "full" for mode, _, _ in calls)
        assert all(steps == 12 for _, steps, _ in calls)


# ---------------------------------------------------------------------------
# Combined modifier smoke test
# ---------------------------------------------------------------------------


class TestCombinedModifiers:
    """Verify alloy substitution followed by adatom deposition composes correctly."""

    def test_alloy_then_adatom(self):
        slab = SlabContainer(make_slab(nx=4, ny=4, n_layers=3, symbol="Ru"))
        n_original = len(slab.atoms)
        with tempfile.TemporaryDirectory() as tmpdir:
            alloyed = substitute_alloy(
                slab, "Ru", "Cu", guest_fraction=0.25, seed=42, results_dir=tmpdir
            )
            n_cu = alloyed.atoms.get_chemical_symbols().count("Cu")
            assert n_cu > 0
            assert len(alloyed.atoms) == n_original

            decorated = deposit_adatoms(
                alloyed,
                "Sn",
                coverage_fraction=0.2,
                seed=42,
                relaxation_mode="none",
                results_dir=tmpdir,
            )
            assert len(decorated.atoms) > n_original
            syms = set(decorated.atoms.get_chemical_symbols())
            assert syms == {"Ru", "Cu", "Sn"}


# ---------------------------------------------------------------------------
# _molecule_diameter
# ---------------------------------------------------------------------------


class TestMoleculeDiameter:
    def test_single_atom(self):
        mol = Atoms("C", positions=[[0, 0, 0]])
        assert _molecule_diameter([mol]) == 0.0

    def test_diatomic(self):
        mol = Atoms("O2", positions=[[0, 0, 0], [1.21, 0, 0]])
        assert np.isclose(_molecule_diameter([mol]), 1.21, atol=1e-6)

    def test_takes_max_across_conformers(self):
        small = Atoms("O2", positions=[[0, 0, 0], [1.0, 0, 0]])
        large = Atoms("O2", positions=[[0, 0, 0], [3.0, 0, 0]])
        assert np.isclose(_molecule_diameter([small, large]), 3.0, atol=1e-6)

    def test_empty_conformer_list(self):
        assert _molecule_diameter([]) == 0.0

    def test_water(self):
        water = make_water()
        d = _molecule_diameter([water])
        assert d > 1.0
        assert d < 3.0


# ---------------------------------------------------------------------------
# _perpendicular_heights_2d
# ---------------------------------------------------------------------------


class TestPerpendicularHeights2D:
    def test_orthogonal(self):
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 20.0]])
        h_a, h_b = _perpendicular_heights_2d(cell)
        assert np.isclose(h_a, 8.0)
        assert np.isclose(h_b, 10.0)

    def test_skewed_60_degree(self):
        cell = np.array([[10.0, 0.0, 0.0], [5.0, 8.66, 0.0], [0.0, 0.0, 20.0]])
        h_a, h_b = _perpendicular_heights_2d(cell)
        area = abs(10.0 * 8.66)
        assert np.isclose(h_a, area / 10.0, atol=0.01)
        assert np.isclose(h_b, area / np.sqrt(25 + 8.66**2), atol=0.01)

    def test_degenerate_zero_vector_raises(self):
        cell = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]])
        with pytest.raises(GeometryValidationError, match="degenerate"):
            _perpendicular_heights_2d(cell)


# ---------------------------------------------------------------------------
# compute_minimum_supercell
# ---------------------------------------------------------------------------


class TestComputeMinimumSupercell:
    def test_already_sufficient(self):
        cell = np.array([[20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0]])
        nx, ny = compute_minimum_supercell(
            cell, molecule_diameter=3.0, min_separation=8.0
        )
        assert (nx, ny) == (1, 1)

    def test_needs_symmetric_expansion(self):
        cell = np.array([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]])
        nx, ny = compute_minimum_supercell(
            cell, molecule_diameter=3.0, min_separation=8.0
        )
        assert nx == 3
        assert ny == 3

    def test_asymmetric_expansion(self):
        cell = np.array([[15.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]])
        nx, ny = compute_minimum_supercell(
            cell, molecule_diameter=3.0, min_separation=8.0
        )
        assert nx == 1
        assert ny == 3

    def test_skewed_cell_expansion(self):
        cell = np.array([[4.0, 0.0, 0.0], [2.0, 3.46, 0.0], [0.0, 0.0, 20.0]])
        nx, ny = compute_minimum_supercell(
            cell, molecule_diameter=2.0, min_separation=8.0
        )
        assert nx >= 1
        assert ny >= 1
        h_a, h_b = _perpendicular_heights_2d(cell)
        assert ny * h_a >= 10.0
        assert nx * h_b >= 10.0

    def test_zero_diameter(self):
        cell = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 20.0]])
        nx, ny = compute_minimum_supercell(
            cell, molecule_diameter=0.0, min_separation=8.0
        )
        assert nx == 1
        assert ny == 1

    def test_degenerate_cell_raises(self):
        cell = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 20.0]])
        with pytest.raises(GeometryValidationError, match="degenerate"):
            compute_minimum_supercell(cell, molecule_diameter=3.0, min_separation=8.0)


# ---------------------------------------------------------------------------
# validate_substrate
# ---------------------------------------------------------------------------


class TestValidateSubstrate:
    def test_accepts_prepared_slab(self):
        atoms = make_slab()
        validate_substrate(atoms, material_type="slab")

    def test_rejects_misaligned_slab(self):
        atoms = make_slab()
        pos = atoms.get_positions().copy()
        pos[:, 2] += 2.0
        atoms.set_positions(pos, apply_constraint=False)
        with pytest.raises(GeometryValidationError, match="bottom-anchored"):
            validate_substrate(atoms, material_type="slab")

    def test_rejects_wrong_pbc(self):
        atoms = make_slab()
        atoms.set_pbc([True, True, True])
        with pytest.raises(GeometryValidationError, match="PBC"):
            validate_substrate(atoms, material_type="slab")

    def test_rejects_small_image_separation(self):
        atoms = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
            cell=[3, 3, 20],
            pbc=[True, True, False],
        )
        atoms = apply_surface_constraints(atoms)
        mol = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]])
        with pytest.raises(
            GeometryValidationError, match="auto_resize_substrate_for_molecule"
        ):
            validate_substrate(
                atoms,
                material_type="slab",
                conformers=[mol],
            )

    def test_rejects_degenerate_slab_in_plane(self):
        atoms = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [0, 1.5, 0], [0, 3.0, 0], [0, 4.5, 0]],
            cell=[[0.0, 0.0, 0.0], [0, 3, 0], [0, 0, 20]],
            pbc=[True, True, False],
        )
        atoms = apply_surface_constraints(atoms)
        with pytest.raises(GeometryValidationError, match="degenerate"):
            validate_substrate(atoms, material_type="slab")

    def test_accepts_nanoparticle_with_vacuum_box(self):
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
            cell=[20, 20, 20],
            pbc=[False, False, False],
        )
        validate_substrate(atoms, material_type="nanoparticle")

    def test_accepts_nanoparticle_not_bottom_anchored(self):
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 5], [2, 0, 5], [0, 2, 5], [2, 2, 5]],
            cell=[20, 20, 20],
            pbc=[False, False, False],
        )
        validate_substrate(atoms, material_type="nanoparticle")

    def test_rejects_nanoparticle_with_tight_box(self):
        atoms = Atoms(
            "Pt4",
            positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [2, 2, 0]],
            cell=[3, 3, 3],
            pbc=[False, False, False],
        )
        with pytest.raises(GeometryValidationError, match="too tight"):
            validate_substrate(atoms, material_type="nanoparticle")


class TestAutoResizeSlabForMolecule:
    def test_no_resize_when_already_sufficient(self):
        slab = SlabContainer(make_slab())
        water = make_water()
        result, was_resized = auto_resize_substrate_for_molecule(
            slab, [water], min_separation=8.0
        )
        assert not was_resized
        assert len(result.atoms) == len(slab.atoms)

    def test_resize_expands_slab(self):
        atoms = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
            cell=[3, 3, 20],
            pbc=True,
        )
        slab = SlabContainer(atoms)
        mol = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]])
        result, was_resized = auto_resize_substrate_for_molecule(
            slab, [mol], min_separation=8.0
        )
        assert was_resized
        assert len(result.atoms) > len(slab.atoms)
        new_cell = np.array(result.atoms.get_cell())
        h_a, h_b = _perpendicular_heights_2d(new_cell)
        assert min(h_a, h_b) >= 5.0 + 8.0

    def test_does_not_modify_original(self):
        atoms = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
            cell=[3, 3, 20],
            pbc=True,
        )
        slab = SlabContainer(atoms)
        original_len = len(slab.atoms)
        mol = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]])
        _, was_resized = auto_resize_substrate_for_molecule(
            slab, [mol], min_separation=8.0
        )
        assert was_resized
        assert len(slab.atoms) == original_len

    def test_preserves_pbc(self):
        atoms = Atoms(
            "Ru4",
            positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [1.5, 1.5, 0]],
            cell=[3, 3, 20],
            pbc=[True, True, True],
        )
        slab = SlabContainer(atoms)
        mol = Atoms("C2", positions=[[0, 0, 0], [5, 0, 0]])
        result, _ = auto_resize_substrate_for_molecule(slab, [mol], min_separation=8.0)
        assert list(result.atoms.get_pbc()) == [True, True, True]


# ---------------------------------------------------------------------------
# _write_clean_xyz
# ---------------------------------------------------------------------------


class TestWriteCleanXyz:
    def test_roundtrip(self):
        atoms = make_slab(nx=2, ny=2, n_layers=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.xyz")
            _write_clean_xyz(atoms, path)
            assert os.path.exists(path)
            with open(path) as f:
                lines = f.readlines()
            assert int(lines[0].strip()) == len(atoms)

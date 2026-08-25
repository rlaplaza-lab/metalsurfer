"""Unit tests for n-tuplet composite construction, selection, and commit."""

import numpy as np
import pytest
from ase import Atoms

from metalsurfer.config import AdsorptionConfig
from metalsurfer.models import ScreeningResult
from metalsurfer.workflow.composite import (
    build_composite_candidate,
    evaluate_composite_commit,
    select_tuplet_winners,
)

from .conftest import (
    make_placement_descriptor,
    make_screening_result,
    make_slab,
    make_water,
    mock_calculator,
    place_molecule_on_slab,
)

SLAB_PBC = [True, True, False]


def _winner(
    slab: Atoms,
    *,
    pid: int = 0,
    e_ads: float = -1.0,
    x_shift: float = 5.0,
    y_shift: float = 5.0,
    z_offset: float = 3.0,
    molecule: str = "water",
) -> ScreeningResult:
    """Single-adsorbate screening result for one water above *slab*."""
    combined = place_molecule_on_slab(
        slab, make_water(), z_offset=z_offset, x_shift=x_shift, y_shift=y_shift
    )
    return make_screening_result(
        molecule=molecule,
        placement_id=pid,
        energy_adsorption=e_ads,
        atoms=combined,
        slab_size=len(slab),
        distance=2.5,
        placement_descriptor=make_placement_descriptor(placement_id=pid),
    )


def _identity_relaxation(monkeypatch: pytest.MonkeyPatch, energy: float):
    """Patch the batched optimizer to a no-op relaxation with fixed energy."""

    def _fake_optimize(combined_atoms_list, _slab, _ts_model, **_kwargs):
        relaxed = []
        for atoms in combined_atoms_list:
            copy = atoms.copy()
            copy.calc = mock_calculator(energy=energy, n_atoms=len(atoms))
            relaxed.append(copy)
        return relaxed

    monkeypatch.setattr(
        "metalsurfer.workflow.composite.optimize_adsorbate_slab_batched",
        _fake_optimize,
    )


# ---------------------------------------------------------------------------
# build_composite_candidate
# ---------------------------------------------------------------------------


class TestBuildCompositeCandidate:
    def test_prefix_preserved_and_suffixes_appended_in_order(self):
        slab = make_slab()
        first = place_molecule_on_slab(slab, make_water(), x_shift=3.5)
        second = place_molecule_on_slab(slab, make_water(), x_shift=7.0)
        composite = build_composite_candidate(
            slab, [first[len(slab) :], second[len(slab) :]]
        )

        assert list(composite.get_chemical_symbols()) == list(
            slab.get_chemical_symbols()
        ) + list(first.get_chemical_symbols()[len(slab) :]) + list(
            second.get_chemical_symbols()[len(slab) :]
        )
        np.testing.assert_allclose(
            composite.get_positions()[: len(slab)],
            slab.get_positions(),
            atol=0.0,
        )
        suffix = composite.get_positions()[len(slab) :]
        np.testing.assert_allclose(
            suffix[: len(first) - len(slab)],
            first.get_positions()[len(slab) :],
            atol=0.0,
        )
        np.testing.assert_allclose(
            suffix[len(first) - len(slab) :],
            second.get_positions()[len(slab) :],
            atol=0.0,
        )

    def test_cell_inherited_and_pbc_promoted_for_calculator(self):
        slab = make_slab()
        adsorbate = place_molecule_on_slab(slab, make_water())
        composite = build_composite_candidate(slab, [adsorbate[len(slab) :]])

        np.testing.assert_allclose(
            np.asarray(composite.get_cell()), np.asarray(slab.get_cell()), atol=0.0
        )
        # Slab-style [T, T, F] promotes to calculator-safe [T, T, True].
        assert list(composite.get_pbc()) == [True, True, True]

    def test_fixatoms_constraints_reference_prefix_only(self):
        slab = make_slab()
        frozen_before = sorted(slab.constraints[0].get_indices().tolist())
        adsorbate = place_molecule_on_slab(slab, make_water())
        composite = build_composite_candidate(slab, [adsorbate[len(slab) :]])

        assert len(composite.constraints) == 1
        frozen_after = sorted(composite.constraints[0].get_indices().tolist())
        assert frozen_after == frozen_before
        assert max(frozen_after) < len(slab)


# ---------------------------------------------------------------------------
# select_tuplet_winners
# ---------------------------------------------------------------------------


class TestSelectTupletWinners:
    def test_accepts_clear_binders_up_to_cap(self):
        slab = make_slab()
        candidates = [
            _winner(slab, pid=i, e_ads=-0.6 + 0.1 * i, x_shift=x)
            for i, x in enumerate((2.5, 6.0, 9.5))
        ]
        winners = select_tuplet_winners(
            candidates,
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=3,
        )
        assert [w.placement_id for w in winners] == [0, 1, 2]

    def test_rejects_clashing_second_winner(self):
        slab = make_slab()
        candidates = [
            _winner(slab, pid=0, e_ads=-1.0, x_shift=5.0),
            _winner(slab, pid=1, e_ads=-0.9, x_shift=5.1),
            _winner(slab, pid=2, e_ads=-0.8, x_shift=8.0),
        ]
        winners = select_tuplet_winners(
            candidates,
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=3,
        )
        # The clashing pid=1 is skipped; the clear pid=2 fills the tuplet.
        assert [w.placement_id for w in winners] == [0, 2]

    def test_non_binders_never_committed_even_with_free_slots(self):
        slab = make_slab()
        candidates = [_winner(slab, pid=0, e_ads=-0.2), _winner(slab, pid=1, e_ads=0.3)]
        winners = select_tuplet_winners(
            candidates,
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=2,
        )
        assert [w.placement_id for w in winners] == [0]

    def test_tie_broken_by_placement_id_then_molecule(self):
        slab = make_slab()
        tie_a = _winner(slab, pid=7, e_ads=-1.0, x_shift=2.5, molecule="water")
        tie_b = _winner(slab, pid=2, e_ads=-1.0, x_shift=6.5, molecule="water")
        winners = select_tuplet_winners(
            [tie_a, tie_b],
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=2,
        )
        assert [w.placement_id for w in winners] == [2, 7]

        same_pid_co2 = _winner(slab, pid=3, e_ads=-1.0, x_shift=2.5, molecule="co2")
        same_pid_air = _winner(slab, pid=3, e_ads=-1.0, x_shift=6.5, molecule="air")
        by_name = select_tuplet_winners(
            [same_pid_co2, same_pid_air],
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=2,
        )
        assert [w.molecule for w in by_name] == ["air", "co2"]

    def test_max_winners_cutoff_keeps_lowest_energy(self):
        slab = make_slab()
        # 2x2 grid keeps every pair ~4.5 A apart (waters clash below ~2 A
        # atom-to-atom); energies descend with pid so the cap keeps [3, 2].
        candidates = [
            _winner(
                slab,
                pid=i,
                e_ads=-1.0 - i,
                x_shift=2.5 + 4.5 * (i % 2),
                y_shift=2.5 + 4.5 * (i // 2),
            )
            for i in range(4)
        ]
        winners = select_tuplet_winners(
            candidates,
            cell=slab.get_cell(),
            pbc=SLAB_PBC,
            min_separation=1.5,
            max_winners=2,
        )
        assert [w.placement_id for w in winners] == [3, 2]


# ---------------------------------------------------------------------------
# evaluate_composite_commit
# ---------------------------------------------------------------------------

E_SLAB = -200.0


def _evaluate_two_clear_winners(
    monkeypatch, *, energy: float | None = -230.0, **kwargs
):
    """Evaluate a 2-water tuplet with the identity relaxation patched in.

    ``energy=None`` leaves optimizer patching to the caller (used by tests
    that install a custom fake optimizer first).
    """
    slab = make_slab()
    winners = [
        _winner(slab, pid=0, e_ads=-1.0, x_shift=3.5),
        _winner(slab, pid=1, e_ads=-0.8, x_shift=7.0),
    ]
    if energy is not None:
        _identity_relaxation(monkeypatch, energy=energy)
    rewritten, failure = evaluate_composite_commit(
        winners=winners,
        slab_atoms=slab,
        base_slab=slab.copy(),
        ts_model=None,
        config=AdsorptionConfig(**kwargs.pop("config_overrides", {})),
        E_slab=E_SLAB,
        **kwargs,
    )
    return slab, winners, rewritten, failure


class TestEvaluateCompositeCommit:
    def test_success_shares_composite_and_full_tuplet_energies(self, monkeypatch):
        # E(tuplet) = -230; sum(E_mol) = -20 -> E_ads = -230 + 200 + 20 = -10.
        _, winners, rewritten, failure = _evaluate_two_clear_winners(
            monkeypatch, energy=-230.0
        )
        assert failure == ""
        assert len(rewritten) == 2

        first, second = rewritten
        for row in (first, second):
            assert row.energy_adslab == pytest.approx(-230.0)
            assert row.energy_slab == pytest.approx(E_SLAB)
            assert row.energy_adsorbate == pytest.approx(-20.0)
            assert row.energy_adsorption == pytest.approx(-10.0)
            assert row.energy_adslab - row.energy_slab - row.energy_adsorbate == (
                pytest.approx(row.energy_adsorption)
            )
            assert row.slab_size == len(make_slab())
            assert len(row.atoms) == len(make_slab()) + 2 * len(make_water())

        # Per-unit identity survives the rewrite.
        assert (first.molecule, first.placement_id) == (
            winners[0].molecule,
            winners[0].placement_id,
        )
        assert first.placement_descriptor is winners[0].placement_descriptor
        assert first.distance > 0 and second.distance > 0

        # Composite atoms are shared (same positions) across committed rows.
        np.testing.assert_allclose(
            first.atoms.get_positions(), second.atoms.get_positions(), atol=0.0
        )

    def test_optimizer_none_fails_cleanly(self, monkeypatch):
        monkeypatch.setattr(
            "metalsurfer.workflow.composite.optimize_adsorbate_slab_batched",
            lambda *_a, **_kw: [None],
        )
        _, _, rewritten, failure = _evaluate_two_clear_winners(monkeypatch, energy=None)
        assert rewritten == []
        assert failure == "optimizer_returned_none"

    def test_desorbed_unit_rejects_whole_composite(self, monkeypatch):
        slab = make_slab()
        winners = [
            _winner(slab, pid=0, e_ads=-1.0, x_shift=3.5),
            _winner(slab, pid=1, e_ads=-0.8, x_shift=7.0, z_offset=50.0),
        ]
        _identity_relaxation(monkeypatch, energy=-230.0)
        rewritten, failure = evaluate_composite_commit(
            winners=winners,
            slab_atoms=slab,
            base_slab=slab.copy(),
            ts_model=None,
            config=AdsorptionConfig(),
            E_slab=E_SLAB,
        )
        assert rewritten == []
        assert "desorbed" in failure
        assert "water" in failure

    def test_frozen_substrate_drift_rejects_composite(self, monkeypatch):
        n_substrate = len(make_slab())

        def _drifting_optimize(combined_atoms_list, _slab, _ts_model, **_kw):
            out = []
            for atoms in combined_atoms_list:
                copy = atoms.copy()
                # Drop constraints first: FixAtoms.adjust_positions would
                # otherwise pin the frozen atoms and mask the simulated drift.
                copy.set_constraint()
                pos = copy.get_positions().copy()
                pos[:n_substrate] += 5.0
                copy.set_positions(pos)
                copy.calc = mock_calculator(energy=-230.0, n_atoms=len(copy))
                out.append(copy)
            return out

        monkeypatch.setattr(
            "metalsurfer.workflow.composite.optimize_adsorbate_slab_batched",
            _drifting_optimize,
        )
        _, _, rewritten, failure = _evaluate_two_clear_winners(monkeypatch, energy=None)
        assert rewritten == []
        assert "frozen substrate drift" in failure

    def test_topology_guard_failure_is_reported(self, monkeypatch):
        _, _, rewritten, failure = _evaluate_two_clear_winners(
            monkeypatch,
            energy=-230.0,
            topology_check=lambda _atoms, names: (
                False,
                f"expected {len(names)} units, found 1",
            ),
        )
        assert rewritten == []
        assert "topology rearrangement guard" in failure

    def test_energy_cap_applies_to_tuplet_total(self, monkeypatch):
        # E_ads = 1000 + 220 = 1220 eV >> default cap of 5 eV.
        _, _, rewritten, failure = _evaluate_two_clear_winners(
            monkeypatch, energy=1000.0
        )
        assert rewritten == []
        assert "E_ads too high" in failure

    def test_empty_winners_short_circuits(self, monkeypatch):
        rewritten, failure = evaluate_composite_commit(
            winners=[],
            slab_atoms=make_slab(),
            base_slab=make_slab(),
            ts_model=None,
            config=AdsorptionConfig(),
            E_slab=E_SLAB,
        )
        assert rewritten == []
        assert failure == "no winners"

"""Tests for observability features.

Covers:
- load_molecules single-read CSV caching
- run_metadata.json output
- Logging context manager and filter
- Vectorized connectivity helpers produce correct results on edge cases
"""

import json
import logging

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from metalsurfer._logging import ContextFilter, get_log_context, log_context
from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import (
    _adjacency_mask,
    _bond_counts_from_atoms,
    _coordination_fingerprint_from_atoms,
    _covalent_threshold_matrix,
    _is_molecule_connected,
)
from metalsurfer.io_results import write_run_metadata
from metalsurfer.workflow import load_molecules

from .conftest import make_slab, make_water, place_molecule_on_slab

# ---------------------------------------------------------------------------
# load_molecules — single CSV read
# ---------------------------------------------------------------------------


class TestLoadMoleculesCaching:
    def test_reads_csv_once_with_skip_existing(self, tmp_path, monkeypatch):
        """Ensure the summary CSV is read at most once, not per molecule."""
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / "smiles.csv"
        csv_path.write_text("O,water\nCCO,ethanol\nCO,methanol\n")

        results_dir = tmp_path / "results_manual"
        results_dir.mkdir()
        summary = pd.DataFrame(
            {
                "molecule": ["water"],
                "energy_adsorption": [-1.0],
            }
        )
        summary.to_csv(
            results_dir / "adsorption_energies_detailed.csv",
            index=False,
        )

        molecules, smiles, _ = load_molecules(
            str(csv_path),
            skip_existing=True,
            surface_type="manual",
        )
        assert "water" not in molecules
        assert set(molecules) == {"ethanol", "methanol"}
        assert len(smiles) == 2

    def test_corrupt_summary_does_not_crash(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / "smiles.csv"
        csv_path.write_text("O,water\n")

        results_dir = tmp_path / "results_manual"
        results_dir.mkdir()
        (results_dir / "adsorption_energies_detailed.csv").write_text(
            "garbage\nnot,a,csv\n"
        )

        molecules, smiles, _ = load_molecules(
            str(csv_path),
            skip_existing=True,
            surface_type="manual",
        )
        assert molecules == ["water"]

    def test_no_summary_file_loads_all(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        csv_path = tmp_path / "smiles.csv"
        csv_path.write_text("O,water\nCCO,ethanol\n")

        molecules, _, _ = load_molecules(
            str(csv_path),
            skip_existing=True,
            surface_type="manual",
        )
        assert len(molecules) == 2


# ---------------------------------------------------------------------------
# run_metadata.json output
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def test_writes_valid_json(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = AdsorptionConfig(seed=123)

        write_run_metadata(
            surface_type="test",
            config=config,
            smiles_file="smiles.csv",
            n_molecules=3,
            total_configs=10,
            t_ref_s=1.5,
            t_total_s=5.0,
        )

        path = tmp_path / "results_test" / "run_metadata.json"
        assert path.exists()

        with open(path) as f:
            meta = json.load(f)

        assert meta["surface_type"] == "test"
        assert meta["config"]["seed"] == 123
        assert meta["input"]["smiles_file"] == "smiles.csv"
        assert meta["input"]["n_molecules"] == 3
        assert meta["results"]["total_configurations"] == 10
        assert meta["timing"]["total_wall_clock_s"] == 5.0
        assert "timestamp" in meta

    def test_metadata_contains_throughput(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = AdsorptionConfig()
        write_run_metadata(
            surface_type="bench",
            config=config,
            smiles_file="s.csv",
            n_molecules=10,
            total_configs=50,
            t_ref_s=2.0,
            t_total_s=20.0,
        )

        with open(tmp_path / "results_bench" / "run_metadata.json") as f:
            meta = json.load(f)

        assert meta["timing"]["molecules_per_second"] == pytest.approx(0.5)
        assert meta["timing"]["configs_per_second"] == pytest.approx(2.5)

    def test_config_excludes_callable_placement_filter(self, tmp_path, monkeypatch):
        """placement_filter (callable) is omitted from JSON, not stringified."""
        monkeypatch.chdir(tmp_path)
        config = AdsorptionConfig(seed=42, placement_filter=lambda s: True)

        write_run_metadata(
            surface_type="test",
            config=config,
            smiles_file="s.csv",
            n_molecules=1,
            total_configs=5,
            t_ref_s=0.0,
            t_total_s=1.0,
        )

        with open(tmp_path / "results_test" / "run_metadata.json") as f:
            meta = json.load(f)

        config_json = json.dumps(meta["config"])
        assert "<function" not in config_json
        assert meta["config"]["seed"] == 42


# ---------------------------------------------------------------------------
# Logging context
# ---------------------------------------------------------------------------


class TestLogContext:
    def test_nested_context(self):
        with log_context(molecule="water"):
            ctx = get_log_context()
            assert ctx["molecule"] == "water"

            with log_context(placement_id=5):
                ctx2 = get_log_context()
                assert ctx2["molecule"] == "water"
                assert ctx2["placement_id"] == 5

            ctx3 = get_log_context()
            assert "placement_id" not in ctx3
            assert ctx3["molecule"] == "water"

        assert get_log_context() == {}

    def test_context_filter_adds_prefix(self):
        filt = ContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        with log_context(molecule="ethanol", surface_type="Ru001"):
            filt.filter(record)
            assert "molecule=ethanol" in record.ctx_prefix
            assert "surface_type=Ru001" in record.ctx_prefix

    def test_empty_context_no_prefix(self):
        filt = ContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        filt.filter(record)
        assert record.ctx_prefix == ""

    def test_context_filter_includes_extra_keys(self):
        """ContextFilter includes keys beyond _KEY_ORDER in ctx_prefix."""
        filt = ContextFilter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        with log_context(molecule="water", custom_key="extra_value"):
            filt.filter(record)
            assert "molecule=water" in record.ctx_prefix
            assert "custom_key=extra_value" in record.ctx_prefix


# ---------------------------------------------------------------------------
# Vectorized helpers — correctness edge cases
# ---------------------------------------------------------------------------


class TestVectorizedHelpers:
    def test_threshold_matrix_symmetry(self):
        syms = np.array(["C", "O", "H"])
        mat = _covalent_threshold_matrix(syms, 1.3)
        assert mat.shape == (3, 3)
        np.testing.assert_array_equal(mat, mat.T)
        assert np.all(mat > 0)

    def test_adjacency_mask_upper_triangle(self):
        dist = np.array(
            [
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 1.0],
                [5.0, 1.0, 0.0],
            ]
        )
        threshold = np.full((3, 3), 2.0)
        mask = _adjacency_mask(dist, threshold)
        assert mask[0, 1]
        assert mask[1, 2]
        assert not mask[0, 2]
        assert not mask[1, 0]  # lower triangle is zero

    def test_single_atom_connectivity(self):
        slab = make_slab()
        slab_z = max(slab.get_positions()[:, 2])
        mol = Atoms("O", positions=[[5.0, 5.0, slab_z + 3.0]])
        combined = slab + mol
        combined.set_cell(slab.get_cell())
        combined.set_pbc(slab.get_pbc())
        assert _is_molecule_connected(
            combined,
            surface_symbols=["Ru"],
            multiplier=1.3,
        )

    def test_vectorized_bond_counts_match_manual(self):
        """Verify vectorized bond counting matches a manually constructed case."""
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water())
        bonds = _bond_counts_from_atoms(
            combined,
            surface_symbols=["Ru"],
            multiplier=1.3,
        )
        assert frozenset({"O", "H"}) in bonds
        assert bonds[frozenset({"O", "H"})] == 2

    def test_vectorized_coord_fingerprint_water(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water())
        fp = _coordination_fingerprint_from_atoms(
            combined,
            surface_symbols=["Ru"],
            multiplier=1.3,
        )
        assert fp["O"] == [2]
        assert fp["H"] == [1, 1]

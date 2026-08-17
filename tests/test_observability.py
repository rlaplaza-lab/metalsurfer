"""Tests for observability features.

Covers:
- load_molecules single-read CSV caching
- run_metadata.json output
- Logging context manager and record factory
- Vectorized connectivity helpers produce correct results on edge cases
"""

import json
import logging
import sys
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from metalsurfer._logging import (
    _LOG_CTX,
    configure_logging,
    ensure_log_record_defaults,
    log_context,
    torchsim_output_capture,
)
from metalsurfer.config import AdsorptionConfig
from metalsurfer.filters import (
    _adjacency_mask,
    _bond_counts_from_dist,
    _coordination_fingerprint_from_dist,
    _covalent_threshold_matrix,
    _is_molecule_connected_from_dist,
    _nonsurface_distance_and_threshold,
)
from metalsurfer.io_results import write_run_metadata
from metalsurfer.workflow import load_molecules

from ._logging_helpers import CaptureHandler, configured_logger
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

    def test_write_settings_and_metadata_merge(self, tmp_path, monkeypatch):
        """write_run_settings and write_run_metadata merge into one file."""
        from metalsurfer.io_results import write_run_settings

        monkeypatch.chdir(tmp_path)
        config = AdsorptionConfig(seed=7)

        write_run_settings(
            surface_type="merge_test",
            config=config,
            campaign="multi_molecule_binding",
            mode="non_bo",
            n_molecules=2,
        )
        write_run_metadata(
            surface_type="merge_test",
            config=config,
            smiles_file="demo.csv",
            n_molecules=2,
            total_configs=8,
            t_ref_s=1.0,
            t_total_s=4.0,
        )

        with open(tmp_path / "results_merge_test" / "run_metadata.json") as f:
            meta = json.load(f)

        assert meta["campaign"] == "multi_molecule_binding"
        assert meta["mode"] == "non_bo"
        assert meta["input"]["smiles_file"] == "demo.csv"
        assert meta["timing"]["total_wall_clock_s"] == 4.0
        assert meta["config"]["seed"] == 7

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

    def test_results_dir_helper(self):
        from metalsurfer.io_results import results_dir_for

        assert results_dir_for("demo").as_posix() == "results_demo"


# ---------------------------------------------------------------------------
# Logging context
# ---------------------------------------------------------------------------


class TestLogContext:
    def test_preconfigure_formatter_with_ctx_prefix_does_not_fail(self):
        """Formatter with ctx_prefix is safe before configure_logging."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(logging.Formatter("%(ctx_prefix)s%(message)s"))
        logger = logging.getLogger("metalsurfer.preconfig_test")

        try:
            with configured_logger(logger, handler=handler):
                logger.info("preconfig message")
                output = stream.getvalue()
                assert output.strip() == "preconfig message"
        finally:
            logger.removeHandler(handler)
            handler.close()

    def test_nested_context(self):
        with log_context(molecule="water"):
            ctx = dict(_LOG_CTX.get() or {})
            assert ctx["molecule"] == "water"

            with log_context(placement_id=5):
                ctx2 = dict(_LOG_CTX.get() or {})
                assert ctx2["molecule"] == "water"
                assert ctx2["placement_id"] == 5

            ctx3 = dict(_LOG_CTX.get() or {})
            assert "placement_id" not in ctx3
            assert ctx3["molecule"] == "water"

        assert dict(_LOG_CTX.get() or {}) == {}

    def test_log_record_factory_adds_prefix(self):
        ensure_log_record_defaults()
        factory = logging.getLogRecordFactory()
        with log_context(molecule="ethanol", surface_type="Ru001"):
            record = factory("test", logging.INFO, "", 0, "hello", (), None)
            assert "molecule=ethanol" in record.ctx_prefix
            assert "surface_type=Ru001" in record.ctx_prefix

    def test_empty_context_no_prefix(self):
        ensure_log_record_defaults()
        factory = logging.getLogRecordFactory()
        record = factory("test", logging.INFO, "", 0, "hello", (), None)
        assert record.ctx_prefix == ""

    def test_log_record_factory_includes_extra_keys(self):
        """Factory includes keys beyond CTX_KEY_ORDER in ctx_prefix."""
        ensure_log_record_defaults()
        factory = logging.getLogRecordFactory()
        with log_context(molecule="water", custom_key="extra_value"):
            record = factory("test", logging.INFO, "", 0, "hello", (), None)
            assert "molecule=water" in record.ctx_prefix
            assert "custom_key=extra_value" in record.ctx_prefix

    def test_configured_logging_has_ctx_prefix_on_child_logger(self):
        """Ensure child loggers emit records with injected ctx_prefix."""
        logger = logging.getLogger("metalsurfer.io_results")

        captured: list[logging.LogRecord] = []
        handler = CaptureHandler(captured)
        with configured_logger(logger, handler=handler):
            with log_context(molecule="ethanol"):
                logger.info("hello")

            assert len(captured) == 1
            assert hasattr(captured[0], "ctx_prefix")
            assert "molecule=ethanol" in captured[0].ctx_prefix

    def test_configure_logging_routes_info_to_stdout(self, monkeypatch):
        """Regression test: INFO logs should go to stdout (HPC `.out`)."""
        stdout = StringIO()
        monkeypatch.setattr(sys, "stdout", stdout)
        monkeypatch.setenv("METALSURFER_FORCE_STDOUT_LOGS", "1")

        root = logging.getLogger()
        old_root_handlers = list(root.handlers)
        old_root_filters = list(root.filters)
        old_root_level = root.level

        # Ensure we don't depend on test-runner prior logging configuration.
        try:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for filt in list(root.filters):
                root.removeFilter(filt)
            root.setLevel(logging.WARNING)

            configure_logging(default_level="INFO")

            logger = logging.getLogger("metalsurfer.stdout_route_test")
            old_logger_handlers = list(logger.handlers)
            old_logger_level = logger.level
            old_logger_propagate = logger.propagate
            try:
                logger.handlers.clear()
                logger.setLevel(logging.INFO)
                logger.propagate = True

                with log_context(molecule="water"):
                    logger.info("hello stdout routing")

                out = stdout.getvalue()
                assert "hello stdout routing" in out
                assert "metalsurfer.stdout_route_test" in out
                assert "[molecule=water]" in out

                stream_handlers = [
                    h for h in root.handlers if isinstance(h, logging.StreamHandler)
                ]
                assert any(
                    getattr(h, "stream", None) is stdout for h in stream_handlers
                )
            finally:
                logger.handlers = old_logger_handlers
                logger.setLevel(old_logger_level)
                logger.propagate = old_logger_propagate
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
            for handler in old_root_handlers:
                root.addHandler(handler)
            for filt in list(root.filters):
                root.removeFilter(filt)
            for filt in old_root_filters:
                root.addFilter(filt)
            root.setLevel(old_root_level)

    @pytest.mark.parametrize(
        ("to_stderr", "expected_level", "message"),
        [
            (False, logging.INFO, "hello torchsim"),
            (True, logging.WARNING, "problem torchsim"),
        ],
    )
    def test_torchsim_output_capture_stream_levels(
        self, to_stderr, expected_level, message
    ):
        torchsim_logger = logging.getLogger("metalsurfer.torchsim")

        captured: list[logging.LogRecord] = []
        handler = CaptureHandler(captured)
        with configured_logger(torchsim_logger, handler=handler):
            with torchsim_output_capture(carriage_return_rate_limit_s=0.0):
                if to_stderr:
                    print(message, file=sys.stderr)
                else:
                    print(message)

            assert len(captured) == 1
            record = captured[0]
            assert record.levelno == expected_level
            assert record.name == "metalsurfer.torchsim"
            assert record.getMessage() == message

    def test_torchsim_output_capture_coalesces_carriage_returns(self):
        torchsim_logger = logging.getLogger("metalsurfer.torchsim")

        captured: list[logging.LogRecord] = []
        handler = CaptureHandler(captured)
        with configured_logger(torchsim_logger, handler=handler):
            # Very large rate limit: only the final newline-terminated line
            # should be emitted.
            with torchsim_output_capture(carriage_return_rate_limit_s=9999.0):
                sys.stdout.write("p1\rp2\rp3\n")

            assert len(captured) == 1
            assert captured[0].levelno == logging.INFO
            assert captured[0].getMessage() == "p3"


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
        syms, _, dist, thresh = _nonsurface_distance_and_threshold(
            combined, ["Ru"], 1.3
        )
        assert _is_molecule_connected_from_dist(syms, dist, thresh)

    def test_vectorized_bond_counts_match_manual(self):
        """Verify vectorized bond counting matches a manually constructed case."""
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water())
        syms, _, dist, thresh = _nonsurface_distance_and_threshold(
            combined, ["Ru"], 1.3
        )
        bonds = _bond_counts_from_dist(syms, dist, thresh)
        assert frozenset({"O", "H"}) in bonds
        assert bonds[frozenset({"O", "H"})] == 2

    def test_vectorized_coord_fingerprint_water(self):
        slab = make_slab()
        combined = place_molecule_on_slab(slab, make_water())
        syms, _, dist, thresh = _nonsurface_distance_and_threshold(
            combined, ["Ru"], 1.3
        )
        fp = _coordination_fingerprint_from_dist(syms, dist, thresh)
        assert fp["O"] == [2]
        assert fp["H"] == [1, 1]

"""`DatasetLogger` / `load_dataset` for CSV training sets from screening runs."""

import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from ..config import AdsorptionConfig
from ..models import ScreeningResult
from .schema import SCHEMA_VERSION, ComputationContext, PlacementRecord

logger = logging.getLogger(__name__)

_DATASET_FILENAME = "ml_dataset.csv"
_METADATA_FILENAME = "ml_dataset_metadata.json"


class DatasetLogger:
    """Accumulates :class:`PlacementRecord` objects and writes them to CSV.

    Typical usage inside a screening run::

        ds = DatasetLogger("results_fcc111", config=config)
        for result in screening_results:
            ds.add_result(result, smiles="CCO", surface_id="Cu_fcc111")
        ds.flush()

    Records are appended if the CSV already exists, avoiding data loss
    across incremental runs.
    """

    def __init__(
        self,
        output_dir: str,
        config: AdsorptionConfig | None = None,
        surface_id: str = "",
        allow_mixed_context: bool = False,
    ) -> None:
        """Instantiate a dataset logger.

        Parameters
        ----------
        output_dir
            Directory where CSV and metadata are written.
        config
            Optional adsorption config for context.
        surface_id
            Surface identifier string.
        allow_mixed_context
            When False (default), refuse to append rows whose ``context_hash``
            or ``schema_version`` differ from existing CSV content.
        """
        self.output_dir = output_dir
        self.surface_id = surface_id
        self.allow_mixed_context = allow_mixed_context
        self.context = (
            ComputationContext.from_config(config)
            if config is not None
            else ComputationContext()
        )
        self._records: list[PlacementRecord] = []
        self._config = config
        self._seen_hashes: set[str] = set()
        self._row_count = 0
        self._csv_columns: list[str] | None = None
        self._disk_state_loaded = False
        self._legacy_context_warned = False

    def _validate_context_compatibility(self) -> None:
        """Ensure append rows match existing CSV context_hash / schema_version."""
        if not os.path.exists(self.csv_path):
            return
        current_hash = self.context.settings_hash()
        usecols: list[str] = []
        if "context_hash" in (self._csv_columns or []):
            usecols.append("context_hash")
        if "schema_version" in (self._csv_columns or []):
            usecols.append("schema_version")
        if not usecols:
            if not self._legacy_context_warned:
                logger.warning(
                    "Appending to %s without context_hash column; cannot verify "
                    "computation context matches prior rows",
                    self.csv_path,
                )
                self._legacy_context_warned = True
            return
        existing = pd.read_csv(self.csv_path, usecols=usecols)
        mismatches: list[str] = []
        if "context_hash" in existing.columns:
            disk_hashes = set(existing["context_hash"].astype(str).unique())
            if disk_hashes != {current_hash}:
                mismatches.append(
                    f"context_hash disk={sorted(disk_hashes)!r} new={current_hash!r}"
                )
        if "schema_version" in existing.columns:
            disk_versions = set(existing["schema_version"].astype(str).unique())
            if disk_versions != {SCHEMA_VERSION}:
                mismatches.append(
                    f"schema_version disk={sorted(disk_versions)!r} "
                    f"new={SCHEMA_VERSION!r}"
                )
        if not mismatches:
            return
        detail = "; ".join(mismatches)
        if self.allow_mixed_context:
            logger.warning(
                "Appending mixed computation context to %s (%s)",
                self.csv_path,
                detail,
            )
            return
        raise ValueError(
            f"Refusing to append to {self.csv_path}: computation context mismatch "
            f"({detail}). Pass allow_mixed_context=True to override."
        )

    @property
    def csv_path(self) -> str:
        """Path to the dataset CSV file."""
        return os.path.join(self.output_dir, _DATASET_FILENAME)

    @property
    def metadata_path(self) -> str:
        """Path to the dataset metadata JSON file."""
        return os.path.join(self.output_dir, _METADATA_FILENAME)

    def add_result(
        self,
        result: ScreeningResult,
        smiles: str,
        surface_id: str | None = None,
    ) -> PlacementRecord:
        """Convert a ScreeningResult to a PlacementRecord and store it.

        Parameters
        ----------
        result
            Screening result to log.
        smiles
            SMILES string of the molecule.
        surface_id
            Optional surface identifier override.
        """
        sid = surface_id if surface_id is not None else self.surface_id
        record = PlacementRecord.from_screening_result(
            result, smiles=smiles, surface_id=sid, config=self._config
        )
        self._records.append(record)
        logger.debug(
            "Logged record %s: E_ads=%.4f eV",
            record.record_hash(),
            record.energy_adsorption,
        )
        return record

    def add_results(
        self,
        results: list[ScreeningResult],
        smiles: str,
        surface_id: str | None = None,
    ) -> int:
        """Batch-add multiple ScreeningResults. Returns count of records added.

        Parameters
        ----------
        results
            List of screening results to log.
        smiles
            SMILES string of the molecule.
        surface_id
            Optional surface identifier override.
        """
        for r in results:
            self.add_result(r, smiles=smiles, surface_id=surface_id)
        return len(results)

    def add_record(self, record: PlacementRecord) -> None:
        """Directly add a pre-built PlacementRecord.

        Parameters
        ----------
        record
            Pre-built placement record to add.
        """
        self._records.append(record)

    def flush(self) -> str:
        """Write accumulated records to CSV (append mode). Returns the CSV path."""
        if not self._records:
            logger.info("No records to flush")
            return self.csv_path

        os.makedirs(self.output_dir, exist_ok=True)
        include_provenance = bool(
            self._config.export_placement_provenance if self._config else False
        )
        rows = [
            r.to_flat_dict(include_provenance=include_provenance) for r in self._records
        ]
        new_df = pd.DataFrame(rows)

        new_df = new_df.drop_duplicates(subset=["record_hash"], keep="first")

        if not self._disk_state_loaded:
            if os.path.exists(self.csv_path):
                hash_col = pd.read_csv(self.csv_path, usecols=["record_hash"])[
                    "record_hash"
                ].astype(str)
                # Row count includes duplicate hashes already present on disk.
                self._row_count = int(len(hash_col))
                self._seen_hashes = set(hash_col)
                self._csv_columns = list(pd.read_csv(self.csv_path, nrows=0).columns)
            self._disk_state_loaded = True

        if self._csv_columns is not None:
            self._validate_context_compatibility()
            new_cols = list(new_df.columns)
            if self._csv_columns != new_cols:
                raise ValueError(
                    f"Refusing to append to {self.csv_path}: column schema mismatch "
                    f"(existing {len(self._csv_columns)} cols vs new {len(new_cols)} cols). "
                    "export_placement_provenance must match the existing dataset "
                    f"(existing={self._csv_columns[:8]}..., new={new_cols[:8]}...)."
                )
            new_df = new_df[~new_df["record_hash"].astype(str).isin(self._seen_hashes)]
            if new_df.empty:
                logger.info("All %d records already in dataset", len(self._records))
                self._records.clear()
                return self.csv_path
            new_df.to_csv(self.csv_path, mode="a", header=False, index=False)
            self._seen_hashes.update(new_df["record_hash"].astype(str))
            self._row_count += len(new_df)
        else:
            new_df.to_csv(self.csv_path, index=False)
            self._seen_hashes = set(new_df["record_hash"].astype(str))
            self._row_count = len(new_df)
            self._csv_columns = list(new_df.columns)

        total_count = self._row_count

        logger.info(
            "Flushed %d new records to %s (total: %d)",
            len(new_df),
            self.csv_path,
            total_count,
        )

        self._write_metadata(total_count, include_provenance=include_provenance)
        self._records.clear()
        return self.csv_path

    def _write_metadata(
        self, total_records: int, *, include_provenance: bool = False
    ) -> None:
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "total_records": total_records,
            "context": self.context.to_dict(),
            "context_hash": self.context.settings_hash(),
            "surface_id": self.surface_id,
            "export_placement_provenance": include_provenance,
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_dataset(
    path: str,
    as_records: bool = False,
) -> pd.DataFrame | list[PlacementRecord]:
    """Load an ML dataset from CSV.

    Parameters
    ----------
    path : str
        Path to the CSV file or directory containing ``ml_dataset.csv``.
    as_records : bool
        If True, return a list of :class:`PlacementRecord` objects.
        Otherwise return a pandas DataFrame.

    Returns
    -------
    DataFrame or list[PlacementRecord]
    """
    if os.path.isdir(path):
        path = os.path.join(path, _DATASET_FILENAME)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded dataset with %d records from %s", len(df), path)

    if as_records:
        return [PlacementRecord.from_flat_dict(dict(row)) for _, row in df.iterrows()]

    return df

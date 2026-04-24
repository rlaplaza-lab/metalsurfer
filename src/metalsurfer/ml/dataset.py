"""`DatasetLogger` / `load_dataset` for CSV training sets from screening runs."""

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
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
    ) -> None:
        self.output_dir = output_dir
        self.surface_id = surface_id
        self.context = (
            ComputationContext.from_config(config)
            if config is not None
            else ComputationContext()
        )
        self._records: list[PlacementRecord] = []
        self._config = config

    @property
    def csv_path(self) -> str:
        return os.path.join(self.output_dir, _DATASET_FILENAME)

    @property
    def metadata_path(self) -> str:
        return os.path.join(self.output_dir, _METADATA_FILENAME)

    def add_result(
        self,
        result: ScreeningResult,
        smiles: str,
        surface_id: str | None = None,
    ) -> PlacementRecord:
        """Convert a ScreeningResult to a PlacementRecord and store it."""
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
        """Batch-add multiple ScreeningResults. Returns count of records added."""
        for r in results:
            self.add_result(r, smiles=smiles, surface_id=surface_id)
        return len(results)

    def add_record(self, record: PlacementRecord) -> None:
        """Directly add a pre-built PlacementRecord."""
        self._records.append(record)

    @property
    def n_records(self) -> int:
        return len(self._records)

    def flush(self) -> str:
        """Write accumulated records to CSV (append mode). Returns the CSV path."""
        if not self._records:
            logger.info("No records to flush")
            return self.csv_path

        os.makedirs(self.output_dir, exist_ok=True)
        rows = [r.to_flat_dict() for r in self._records]
        new_df = pd.DataFrame(rows)

        new_df = new_df.drop_duplicates(subset=["record_hash"], keep="first")

        if os.path.exists(self.csv_path):
            existing_hashes = set(
                pd.read_csv(self.csv_path, usecols=["record_hash"])["record_hash"]
            )
            new_df = new_df[~new_df["record_hash"].isin(existing_hashes)]
            if new_df.empty:
                logger.info("All %d records already in dataset", len(self._records))
                self._records.clear()
                return self.csv_path
            new_df.to_csv(self.csv_path, mode="a", header=False, index=False)
            total_count = len(existing_hashes) + len(new_df)
        else:
            new_df.to_csv(self.csv_path, index=False)
            total_count = len(new_df)

        logger.info(
            "Flushed %d new records to %s (total: %d)",
            len(new_df),
            self.csv_path,
            total_count,
        )

        self._write_metadata(total_count)
        self._records.clear()
        return self.csv_path

    def _write_metadata(self, total_records: int) -> None:
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "timestamp": datetime.now(UTC).isoformat(),
            "total_records": total_records,
            "context": self.context.to_dict(),
            "context_hash": self.context.settings_hash(),
            "surface_id": self.surface_id,
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
        return [PlacementRecord.from_flat_dict(row) for _, row in df.iterrows()]

    return df


def merge_datasets(*paths: str, output_path: str | None = None) -> pd.DataFrame:
    """Merge multiple ML dataset CSVs, deduplicating by record_hash.

    Parameters
    ----------
    *paths : str
        Paths to CSV files or directories.
    output_path : str, optional
        If set, write the merged dataset to this path.

    Returns
    -------
    DataFrame
        The merged, deduplicated dataset.
    """
    if not paths:
        raise ValueError("merge_datasets requires at least one path")
    frames = [load_dataset(p, as_records=False) for p in paths]

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["record_hash"], keep="first")
    merged = merged.sort_values("record_hash").reset_index(drop=True)

    logger.info("Merged %d datasets -> %d unique records", len(frames), len(merged))

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        logger.info("Saved merged dataset to %s", output_path)

    return merged

"""`BindingEnergyPredictor`: load a fitted pipeline and score `PlacementRecord` rows."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor
from .features import extract_features
from .regression import load_model
from .schema import PlacementRecord


@dataclass
class PredictionResult:
    """Predicted binding energy with optional uncertainty estimate."""

    energy: float
    uncertainty: float | None = None
    record_hash: str | None = None
    model_type: str | None = None


class BindingEnergyPredictor:
    """Predict binding energy from placement descriptors using a trained model.

    Supports single predictions, batch predictions, and ensemble
    uncertainty estimation (when loaded with multiple models).
    """

    def __init__(
        self,
        model: Pipeline,
        metadata: dict[str, Any] | None = None,
        ensemble: list[Pipeline] | None = None,
    ) -> None:
        self._model = model
        self._metadata = metadata or {}
        self._ensemble = ensemble

    @classmethod
    def load(cls, model_dir: str) -> "BindingEnergyPredictor":
        """Load a predictor from a saved model directory."""
        model, metadata = load_model(model_dir)
        return cls(model=model, metadata=metadata)

    @property
    def model_type(self) -> str:
        return self._metadata.get("model_type", "unknown")

    def predict_record(self, record: PlacementRecord) -> PredictionResult:
        """Predict binding energy for a single PlacementRecord."""
        return self.predict_batch([record])[0]

    def predict_batch(
        self,
        records: list[PlacementRecord],
    ) -> list[PredictionResult]:
        """Predict binding energies for multiple records efficiently."""
        if not records:
            return []

        feature_dicts = [extract_features(r) for r in records]
        X = pd.DataFrame(feature_dicts)
        y_pred = self._model.predict(X)

        uncertainties: list[float | None] = [None] * len(records)
        if self._ensemble is not None:
            all_preds = np.array([m.predict(X) for m in self._ensemble])
            std_preds = np.std(all_preds, axis=0)
            uncertainties = [float(s) for s in std_preds]

        return [
            PredictionResult(
                energy=float(y_pred[i]),
                uncertainty=uncertainties[i],
                record_hash=record.record_hash(),
                model_type=self.model_type,
            )
            for i, record in enumerate(records)
        ]

    def predict_descriptor(
        self,
        descriptor: PlacementDescriptor,
        molecule: str = "",
        smiles: str = "",
        surface_id: str = "",
        config: AdsorptionConfig | None = None,
    ) -> PredictionResult:
        """Predict from a PlacementDescriptor (convenience wrapper).

        Builds a temporary PlacementRecord with zero energies, extracts
        features, and returns the prediction.
        """
        record = PlacementRecord.from_descriptor(
            descriptor,
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            config=config,
        )
        return self.predict_record(record)

    def rank_placements(
        self,
        records: list[PlacementRecord],
        top_k: int | None = None,
    ) -> list[tuple[PlacementRecord, PredictionResult]]:
        """Rank placements by predicted binding energy (most negative first).

        Returns a sorted list of (record, prediction) tuples.
        Useful as a pre-filter before running expensive optimizations.
        """
        predictions = self.predict_batch(records)
        paired = list(zip(records, predictions, strict=True))
        paired.sort(key=lambda p: p[1].energy)
        if top_k is not None:
            paired = paired[:top_k]
        return paired

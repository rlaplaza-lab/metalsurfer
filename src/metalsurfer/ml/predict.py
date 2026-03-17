"""Binding energy prediction from placement descriptors via a trained regression model.

Example: predictor = BindingEnergyPredictor.load(\"model_dir/\");
pred = predictor.predict_record(record); use pred.energy as a screening pre-filter.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ..config import AdsorptionConfig
from ..models import PlacementDescriptor
from .features import extract_features
from .regression import load_model
from .schema import ComputationContext, PlacementRecord

logger = logging.getLogger(__name__)


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
        self._feature_names: list[str] = self._metadata.get("feature_names", [])

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
        features = extract_features(record)
        X = pd.DataFrame([features])
        y_pred = float(self._model.predict(X)[0])

        uncertainty = None
        if self._ensemble is not None:
            preds = [float(m.predict(X)[0]) for m in self._ensemble]
            uncertainty = float(np.std(preds))

        return PredictionResult(
            energy=y_pred,
            uncertainty=uncertainty,
            record_hash=record.record_hash(),
            model_type=self.model_type,
        )

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

        results = []
        for i, record in enumerate(records):
            results.append(
                PredictionResult(
                    energy=float(y_pred[i]),
                    uncertainty=uncertainties[i],
                    record_hash=record.record_hash(),
                    model_type=self.model_type,
                )
            )
        return results

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
        ctx = (
            ComputationContext.from_config(config)
            if config is not None
            else ComputationContext()
        )
        record = PlacementRecord(
            molecule=molecule,
            smiles=smiles,
            surface_id=surface_id,
            placement_id=descriptor.placement_index,
            conformer_index=descriptor.conformer_index,
            orientation_type=descriptor.orientation_type,
            face_flip=descriptor.face_flip,
            en_atom_index=descriptor.en_atom_index,
            site_index=descriptor.site_index,
            site_type=descriptor.site_type,
            tilt_deg=descriptor.tilt_deg,
            azimuth_deg=descriptor.azimuth_deg,
            azimuth_in_plane_deg=descriptor.azimuth_in_plane_deg,
            z_fraction=descriptor.z_fraction,
            x=descriptor.x,
            y=descriptor.y,
            z=descriptor.z,
            shape=descriptor.shape,
            slab_indices=descriptor.slab_indices,
            context=ctx,
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

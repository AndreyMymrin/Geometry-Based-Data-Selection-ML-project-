from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def compute_error_rate_scores(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    predictions shape: [n_models, n_samples]
    labels shape: [n_samples]
    """
    if predictions.ndim != 2:
        raise ValueError(f"Expected 2D predictions, got shape={predictions.shape}")
    if labels.ndim != 1:
        raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
    if predictions.shape[1] != labels.shape[0]:
        raise ValueError("Second dimension of predictions must match labels length.")
    errors = predictions != labels[None, :]
    return errors.mean(axis=0).astype(np.float32)


class ErrorRateEnsembleScorer(SampleScorer):
    @property
    def name(self) -> str:
        return "error_rate_ensemble"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "predictions" not in metadata:
            raise ValueError("metadata['predictions'] is required for error-rate scoring.")

        predictions = np.asarray(metadata["predictions"])
        labels_np = np.asarray(labels, dtype=np.int64)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)

        scores = compute_error_rate_scores(predictions=predictions, labels=labels_np)
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame(
            {
                "sample_id": sample_ids_np,
                "label": labels_np,
                "score": scores.astype(float),
                "rank": ranks.astype(int),
                "method": self.name,
            }
        )
        return df.sort_values("rank", kind="stable").reset_index(drop=True)


"""LLM-Based Classifier scorer for text data.

Uses a trained language model to classify and filter data based on
prediction confidence. Computes per-sample mean prediction entropy:
low entropy = model is confident = easy/clean data,
high entropy = model is uncertain = noisy/hard data.

Samples with the lowest entropy (easiest) are removed first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


class LLMClassifierScorer(SampleScorer):
    """Score text samples by LLM prediction entropy.

    Requires metadata['entropy']: np.ndarray of shape (N,) with
    per-sample mean prediction entropy from a trained language model.
    """

    @property
    def name(self) -> str:
        return "llm_classifier"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "entropy" not in metadata:
            raise ValueError(
                "metadata['entropy'] is required for LLM classifier scoring."
            )

        entropy = np.asarray(metadata["entropy"], dtype=np.float32)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        # Score = entropy (lower entropy = easier = removed first)
        scores = entropy
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame({
            "sample_id": sample_ids_np,
            "label": labels_np,
            "score": scores.astype(float),
            "rank": ranks.astype(int),
            "method": self.name,
        })
        return df.sort_values("rank", kind="stable").reset_index(drop=True)

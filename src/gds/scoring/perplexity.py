"""Perplexity Filtering scorer for text data.

Ranks samples by their perplexity under a trained language model.
High perplexity = hard/noisy sample, low perplexity = easy/clean sample.
Samples with the lowest perplexity (easiest) are removed first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


class PerplexityFilteringScorer(SampleScorer):
    """Score text samples by per-sample perplexity.

    Requires metadata['perplexity']: np.ndarray of shape (N,) with
    per-sample perplexity values computed from a trained language model.
    """

    @property
    def name(self) -> str:
        return "perplexity_filtering"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "perplexity" not in metadata:
            raise ValueError(
                "metadata['perplexity'] is required for perplexity filtering."
            )

        perplexity = np.asarray(metadata["perplexity"], dtype=np.float32)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        # Score = perplexity (lower perplexity = easier = removed first)
        scores = perplexity
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame({
            "sample_id": sample_ids_np,
            "label": labels_np,
            "score": scores.astype(float),
            "rank": ranks.astype(int),
            "method": self.name,
        })
        return df.sort_values("rank", kind="stable").reset_index(drop=True)

"""Semantic Deduplication scorer for text data.

Removes redundant samples by finding near-duplicates in embedding space.
Each sample is scored by how redundant it is: samples with many close
neighbours (high max cosine similarity) are considered duplicates and
get low scores (removed first).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def compute_semantic_dedup_scores(
    features: np.ndarray,
    similarity_threshold: float = 0.9,
) -> np.ndarray:
    """Score samples by redundancy in embedding space.

    For each sample, compute max cosine similarity to all other samples.
    Score = 1 - max_similarity: lower score = more redundant = removed first.

    Parameters
    ----------
    features : np.ndarray, shape (N, d)
        Embedding vectors.
    similarity_threshold : float
        Not used for scoring, but logged for reference.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Per-sample uniqueness score. Lower = more redundant.
    """
    features = features.astype(np.float64)

    # L2-normalise for cosine similarity
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = features / norms

    # Cosine similarity matrix
    sim_matrix = normed @ normed.T

    # Zero out self-similarity
    np.fill_diagonal(sim_matrix, -1.0)

    # Max similarity to any other sample
    max_sim = sim_matrix.max(axis=1)

    # Score = 1 - max_sim (more unique = higher score = kept)
    scores = (1.0 - max_sim).astype(np.float32)

    return scores


class SemanticDedupScorer(SampleScorer):
    """Score text samples by semantic redundancy.

    Requires metadata['features']: np.ndarray of shape (N, d).
    """

    @property
    def name(self) -> str:
        return "semantic_dedup"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "features" not in metadata:
            raise ValueError(
                "metadata['features'] is required for semantic dedup."
            )

        features = np.asarray(metadata["features"], dtype=np.float32)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        scores = compute_semantic_dedup_scores(features)
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame({
            "sample_id": sample_ids_np,
            "label": labels_np,
            "score": scores.astype(float),
            "rank": ranks.astype(int),
            "method": self.name,
        })
        return df.sort_values("rank", kind="stable").reset_index(drop=True)

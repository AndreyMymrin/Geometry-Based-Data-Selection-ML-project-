"""Heuristic Filtering scorer for text data.

Basic data cleaning based on character composition of each text chunk.
Computes quality features (alphabetic ratio, character diversity,
whitespace ratio, punctuation density) and combines them into a single
quality score. Lower quality samples are removed first.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def compute_heuristic_scores(chunks: list[np.ndarray], itos: dict[int, str]) -> np.ndarray:
    """Compute heuristic quality scores for text chunks.

    Quality features per chunk:
    - alpha_ratio:     fraction of alphabetic characters
    - diversity:       unique characters / total characters
    - whitespace_ratio: fraction of whitespace characters
    - punctuation_ratio: fraction of punctuation characters

    Score = alpha_ratio * 0.4 + diversity * 0.3
            + (1 - whitespace_ratio) * 0.15 + (1 - punctuation_ratio) * 0.15

    Higher score = higher quality text = kept.
    Lower score = noisy/low-quality = removed first.

    Parameters
    ----------
    chunks : list of np.ndarray
        Each chunk is an int64 array of token IDs (length = block_size + 1).
    itos : dict mapping int -> str
        Index-to-character mapping.

    Returns
    -------
    scores : np.ndarray, shape (len(chunks),)
    """
    n = len(chunks)
    scores = np.empty(n, dtype=np.float32)

    for i, chunk in enumerate(chunks):
        text = "".join(itos.get(int(t), "") for t in chunk)
        length = max(len(text), 1)

        n_alpha = sum(c.isalpha() for c in text)
        n_unique = len(set(text))
        n_whitespace = sum(c.isspace() for c in text)
        n_punct = sum(not c.isalnum() and not c.isspace() for c in text)

        alpha_ratio = n_alpha / length
        diversity = n_unique / length
        ws_ratio = n_whitespace / length
        punct_ratio = n_punct / length

        scores[i] = (
            alpha_ratio * 0.4
            + diversity * 0.3
            + (1.0 - ws_ratio) * 0.15
            + (1.0 - punct_ratio) * 0.15
        )

    return scores


class HeuristicFilteringScorer(SampleScorer):
    """Score text samples by heuristic text quality.

    Requires metadata['chunks'] and metadata['itos'].
    """

    @property
    def name(self) -> str:
        return "heuristic_filtering"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "chunks" not in metadata or "itos" not in metadata:
            raise ValueError(
                "metadata['chunks'] and metadata['itos'] are required "
                "for heuristic filtering."
            )

        chunks = metadata["chunks"]
        itos = metadata["itos"]
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        # Only score chunks corresponding to sample_ids
        selected_chunks = [chunks[sid] for sid in sample_ids]
        scores = compute_heuristic_scores(selected_chunks, itos)
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame({
            "sample_id": sample_ids_np,
            "label": labels_np,
            "score": scores.astype(float),
            "rank": ranks.astype(int),
            "method": self.name,
        })
        return df.sort_values("rank", kind="stable").reset_index(drop=True)

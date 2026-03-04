from __future__ import annotations

import numpy as np


def stable_rank_from_scores(sample_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
    if sample_ids.shape[0] != scores.shape[0]:
        raise ValueError("sample_ids and scores must have equal lengths.")
    order = np.lexsort((sample_ids, scores))
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return rank


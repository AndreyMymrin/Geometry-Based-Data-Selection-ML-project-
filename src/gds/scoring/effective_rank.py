# from __future__ import annotations

# from typing import Any

# import numpy as np
# import pandas as pd

# from gds.scoring.base import SampleScorer
# from gds.scoring.utils import stable_rank_from_scores


# def compute_effective_scores(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray:
#     """
#     Compute effective scores as the complement of error rates.
#     Higher score = more effective (correctly classified by more models).
    
#     predictions shape: [n_models, n_samples]
#     labels shape: [n_samples]
#     """
#     if predictions.ndim != 2:
#         raise ValueError(f"Expected 2D predictions, got shape={predictions.shape}")
#     if labels.ndim != 1:
#         raise ValueError(f"Expected 1D labels, got shape={labels.shape}")
#     if predictions.shape[1] != labels.shape[0]:
#         raise ValueError("Second dimension of predictions must match labels length.")
    
#     # Compute accuracy (fraction of models that classify correctly)
#     correct = predictions == labels[None, :]
#     return correct.mean(axis=0).astype(np.float32)


# class EffectiveRankScorer(SampleScorer):
#     @property
#     def name(self) -> str:
#         return "effective_rank"

#     def score(
#         self,
#         sample_ids: list[int],
#         labels: list[int],
#         metadata: dict[str, Any] | None = None,
#     ) -> pd.DataFrame:
#         if metadata is None or "predictions" not in metadata:
#             raise ValueError("metadata['predictions'] is required for effective-rank scoring.")

#         predictions = np.asarray(metadata["predictions"])
#         labels_np = np.asarray(labels, dtype=np.int64)
#         sample_ids_np = np.asarray(sample_ids, dtype=np.int64)

#         scores = compute_effective_scores(predictions=predictions, labels=labels_np)
#         # For effective rank, we want higher scores (more effective) to have lower ranks
#         # So we rank in descending order of effectiveness
#         ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=-scores)

#         df = pd.DataFrame(
#             {
#                 "sample_id": sample_ids_np,
#                 "label": labels_np,
#                 "score": scores.astype(float),
#                 "rank": ranks.astype(int),
#                 "method": self.name,
#             }
#         )
#         return df.sort_values("rank", kind="stable").reset_index(drop=True)

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def compute_effective_rank(features: np.ndarray) -> np.ndarray:
    """
    Compute per-sample effective rank scores from a feature matrix.

    Uses the Diff-eRank approach (Wei et al., NeurIPS 2024):
        1. Build a trace-normalised covariance matrix from all N representations.
        2. Per-sample score = erank(all) - erank(all \\ {i})  (leave-one-out).

    A higher score means the sample contributes more geometric diversity
    to the representation space and is therefore more informative.

    Parameters
    ----------
    features : np.ndarray, shape (N, d)
        Feature vectors extracted from a model's internal representations.

    Returns
    -------
    scores : np.ndarray, shape (N,), dtype float32
        Per-sample effective rank contribution. Higher = more informative.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={features.shape}")
    if features.shape[0] < 3:
        raise ValueError("compute_effective_rank requires at least 3 samples.")

    features = features.astype(np.float64)

    # --- Step 1: mean-centre and L2-normalise rows ---
    Z = features - features.mean(axis=0, keepdims=True)
    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    Z = Z / row_norms

    # --- Step 2: full covariance (un-trace-normalised, for rank-1 updates) ---
    C_full = Z.T @ Z   # shape (d, d)

    # --- Step 3: global erank ---
    global_er = _erank_from_covariance(C_full)

    # --- Step 4: leave-one-out via rank-1 downdate ---
    N = features.shape[0]
    scores = np.empty(N, dtype=np.float64)
    for i in range(N):
        zi    = Z[i]
        C_loo = C_full - np.outer(zi, zi)
        scores[i] = global_er - _erank_from_covariance(C_loo)

    return scores.astype(np.float32)


def _erank_from_covariance(C: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank of a PSD covariance matrix after trace-normalisation."""
    trace = np.trace(C)
    if trace < eps:
        return 1.0
    C_norm = C / trace
    sv = np.linalg.svd(C_norm, compute_uv=False)
    l1 = sv.sum()
    if l1 < eps:
        return 1.0
    p = sv / l1
    nonzero = p > eps
    H = float(-np.nansum(p[nonzero] * np.log(p[nonzero])))
    return float(np.exp(H))


class EffectiveRankScorer(SampleScorer):
    """
    Geometry-based effective rank scorer (Diff-eRank, Wei et al. NeurIPS 2024).

    Scores each sample by its leave-one-out contribution to the effective rank
    of the population's trace-normalised covariance matrix:

        Score(i) = erank(all N samples) - erank(all N samples except i)

    Higher score → sample adds more geometric diversity → more informative.

    Requires metadata['features']: np.ndarray of shape (N, d).
    """

    def __init__(self) -> None:
        self._last_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "effective_rank"

    def build_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "features" not in metadata:
            raise ValueError("metadata['features'] is required for effective-rank scoring.")

        features      = np.asarray(metadata["features"], dtype=np.float32)
        labels_np     = np.asarray(labels,     dtype=np.int64)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)

        if features.shape[0] != sample_ids_np.shape[0]:
            raise ValueError("Feature rows must match the number of sample ids.")

        scores = compute_effective_rank(features)
        ranks  = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        self._last_metadata = {
            "method":        self.name,
            "feature_shape": [int(features.shape[0]), int(features.shape[1])],
            "mean_score":    float(scores.mean()),
            "min_score":     float(scores.min()),
            "max_score":     float(scores.max()),
        }

        df = pd.DataFrame(
            {
                "sample_id": sample_ids_np,
                "label":     labels_np,
                "score":     scores.astype(float),
                "rank":      ranks.astype(int),
                "method":    self.name,
            }
        )
        return df.sort_values("rank", kind="stable").reset_index(drop=True)
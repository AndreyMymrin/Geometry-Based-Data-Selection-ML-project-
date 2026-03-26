"""Effective rank scorer (Roy & Vetterli, EUSIPCO 2007).

Reference:
    O. Roy and M. Vetterli, "The Effective Rank: A Measure of Effective
    Dimensionality," 15th European Signal Processing Conference (EUSIPCO),
    Poznan, Poland, 2007, pp. 606-610.

Definition:
    Given matrix X of size (n_tokens, d_hidden) with singular values
    sigma_1 >= ... >= sigma_Q:
        p_k  = sigma_k / sum(sigma_j)      (L1-normalised singular values)
        H    = -sum(p_k * log(p_k))         (Shannon entropy)
        erank(X) = exp(H)

    Per-sample score = average erank across all layers of hidden states.
    Higher score -> richer internal representation -> more informative sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def _entropy_from_sigma(sigma: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of the normalised singular-value distribution."""
    total = sigma.sum()
    if total < eps:
        return 0.0
    p = sigma / total
    p_safe = np.maximum(p, eps)
    return float(-np.sum(p * np.log(p_safe)))


def erank(X: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank of a 2-D matrix (Roy & Vetterli, 2007).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Any real-valued matrix (e.g. hidden states of shape
        (n_tokens, d_hidden) for one layer of one sample).

    Returns
    -------
    float
        Effective rank in [1, min(n, d)].
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}.")
    if X.shape[0] < 2:
        raise ValueError("ERank requires at least 2 rows (tokens).")

    X64 = X.astype(np.float64)
    n, d = X64.shape

    if n < d:
        G = X64 @ X64.T                              # (n, n), symmetric
        eigs = np.linalg.eigvalsh(G)                  # ascending order, O(n³)
        eigs = np.maximum(eigs, 0.0)
        sigma = np.sqrt(eigs)
    else:
        sigma = np.linalg.svd(X64, compute_uv=False)

    H = _entropy_from_sigma(sigma, eps)
    return float(np.exp(H))


def layerwise_erank(
    hidden_states: list[np.ndarray],
) -> np.ndarray:
    """Compute erank for each layer's hidden-state matrix.

    Parameters
    ----------
    hidden_states : list of np.ndarray
        L arrays, each of shape (n_tokens, d_hidden).

    Returns
    -------
    np.ndarray, shape (L,)
    """
    return np.array([erank(X) for X in hidden_states], dtype=np.float64)


def average_erank(hidden_states: list[np.ndarray]) -> float:
    """Average erank across all layers for one sample."""
    scores = layerwise_erank(hidden_states)
    return float(scores.mean())


class EffectiveRankScorer:
    """Per-sample effective rank scorer.

    Each sample is scored by the average erank of its hidden-state
    matrices across L transformer layers:

        Score(i) = (1/L) * sum_l  erank( H_i^l )

    where H_i^l has shape (n_tokens, d_hidden) for sample i, layer l.

    Higher score -> richer internal representation -> more informative.

    Requires ``metadata['hidden_states']``: list of N per-sample
    hidden-state lists, each containing L arrays of shape
    (n_tokens, d_hidden).
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
        if metadata is None or "hidden_states" not in metadata:
            raise ValueError(
                "metadata['hidden_states'] is required. "
                "Expected a list of N per-sample hidden-state lists, "
                "each containing L arrays of shape (n_tokens, d_hidden)."
            )

        all_hidden: list[list[np.ndarray]] = metadata["hidden_states"]
        n_samples = len(sample_ids)

        if len(all_hidden) != n_samples:
            raise ValueError(
                f"hidden_states has {len(all_hidden)} entries "
                f"but sample_ids has {n_samples}."
            )

        print(f"  Computing per-sample erank ({n_samples} samples)...")
        scores = np.array(
            [average_erank(hs) for hs in tqdm(all_hidden, desc="  erank")],
            dtype=np.float64,
        )

        # Lower score = less informative = removed first (rank 1).
        # Use stable_rank_from_scores for consistency with other scorers.
        ranks = np.argsort(np.argsort(scores)) + 1

        n_layers = len(all_hidden[0]) if all_hidden else 0
        self._last_metadata = {
            "method":     self.name,
            "n_samples":  n_samples,
            "n_layers":   n_layers,
            "mean_score": float(scores.mean()),
            "min_score":  float(scores.min()),
            "max_score":  float(scores.max()),
        }
        print(f"  erank: mean={scores.mean():.2f}, range=[{scores.min():.2f}, {scores.max():.2f}]")

        return (
            pd.DataFrame(
                {
                    "sample_id": np.asarray(sample_ids, dtype=np.int64),
                    "label":     np.asarray(labels, dtype=np.int64),
                    "score":     scores,
                    "rank":      ranks.astype(int),
                    "method":    self.name,
                }
            )
            .sort_values("rank", kind="stable")
            .reset_index(drop=True)
        )

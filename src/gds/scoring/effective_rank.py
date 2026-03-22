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

    This implementation uses vectorised first-order eigenvalue perturbation
    instead of N separate SVD calls, reducing complexity from O(N * d^3) to
    O(d^3 + N * d^2).

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

    N, d = features.shape
    features = features.astype(np.float64)

    # --- Step 1: mean-centre and L2-normalise rows ---
    Z = features - features.mean(axis=0, keepdims=True)
    row_norms = np.linalg.norm(Z, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-12, 1.0, row_norms)
    Z = Z / row_norms

    # --- Step 2: full covariance ---
    C_full = Z.T @ Z  # (d, d)

    # --- Step 3: eigendecomposition of full covariance (once) ---
    eigenvalues, eigenvectors = np.linalg.eigh(C_full)  # eigenvalues sorted ascending
    eigenvalues = np.maximum(eigenvalues, 0.0)  # clamp numerical negatives

    # --- Step 4: global erank ---
    global_er = _erank_from_eigenvalues(eigenvalues)
    print(f"  Global effective rank: {global_er:.2f}  (d={d}, N={N})")

    # --- Step 5: vectorised leave-one-out via first-order perturbation ---
    # Project each sample onto the eigenbasis: W[i,k] = z_i^T u_k
    W = Z @ eigenvectors  # (N, d)
    W2 = W ** 2           # (N, d) — squared projections

    # LOO eigenvalues: lambda_k^{(i)} ≈ lambda_k - w_{ik}^2
    # shape: (N, d)
    loo_eigs = eigenvalues[np.newaxis, :] - W2
    loo_eigs = np.maximum(loo_eigs, 0.0)  # clamp negatives

    # Compute erank for each LOO set (vectorised)
    loo_eranks = _erank_from_eigenvalues_batch(loo_eigs)

    scores = global_er - loo_eranks
    print(f"  Score range: [{scores.min():.6f}, {scores.max():.6f}]")
    return scores.astype(np.float32)


def _erank_from_eigenvalues(eigs: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank from eigenvalue array (single set)."""
    total = eigs.sum()
    if total < eps:
        return 1.0
    p = eigs / total
    mask = p > eps
    H = float(-np.sum(p[mask] * np.log(p[mask])))
    return float(np.exp(H))


def _erank_from_eigenvalues_batch(eigs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Effective rank from eigenvalue arrays (batch of N sets).

    Parameters
    ----------
    eigs : np.ndarray, shape (N, d)
        Each row is a set of eigenvalues.

    Returns
    -------
    eranks : np.ndarray, shape (N,)
    """
    totals = eigs.sum(axis=1, keepdims=True)  # (N, 1)
    totals = np.maximum(totals, eps)
    p = eigs / totals  # (N, d)
    # Clamp for log stability
    p_safe = np.maximum(p, eps)
    # Zero out negligible eigenvalues so they don't contribute to entropy
    H = -np.sum(np.where(p > eps, p * np.log(p_safe), 0.0), axis=1)
    return np.exp(H)


# Keep the old function for compatibility (used by global erank calculation)
def _erank_from_covariance(C: np.ndarray, eps: float = 1e-12) -> float:
    """Effective rank of a PSD covariance matrix after trace-normalisation."""
    eigs = np.linalg.eigvalsh(C)
    eigs = np.maximum(eigs, 0.0)
    return _erank_from_eigenvalues(eigs, eps)


class EffectiveRankScorer(SampleScorer):
    """
    Geometry-based effective rank scorer (Diff-eRank, Wei et al. NeurIPS 2024).

    Scores each sample by its leave-one-out contribution to the effective rank
    of the population's trace-normalised covariance matrix:

        Score(i) = erank(all N samples) - erank(all N samples except i)

    Higher score -> sample adds more geometric diversity -> more informative.

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

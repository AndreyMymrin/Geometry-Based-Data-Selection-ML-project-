"""Correlation Integral (CorrInt) scorer.

Reference:
    P. Grassberger and I. Procaccia, "Measuring the Strangeness of Strange
    Attractors," Physica 9D (1983), pp. 189-208.

The correlation integral is defined as:

    C(l) = lim_{N->inf} (1/N^2) * |{(i,j) : ||X_i - X_j|| < l, i != j}|

For data lying on a manifold of intrinsic dimension nu:

    C(l) ~ l^nu   for small l

The correlation dimension nu is estimated via a log-log linear fit
of C(l) vs l.

Following Yusupov et al. (2025), the CorrInt score is computed
per-sample on hidden-state matrices from a pretrained model, then
averaged across layers.  Higher CorrInt -> richer representation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm


def estimate_correlation_dimension(
    X: np.ndarray,
    n_radii: int = 20,
    eps: float = 1e-12,
) -> float:
    """Estimate correlation dimension nu of a point cloud.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, d)
        Point cloud (e.g. token representations from one layer).
    n_radii : int
        Number of log-spaced radii for the C(l) curve.

    Returns
    -------
    nu : float
        Estimated correlation dimension.
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {X.shape}.")

    N, d = X.shape
    if N < 5:
        return 1.0

    X64 = X.astype(np.float64)

    # Use kNN to approximate the distance distribution
    k = min(50, N - 1)
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", algorithm="auto")
    nbrs.fit(X64)
    dists, _ = nbrs.kneighbors(X64)
    knn_dists = dists[:, 1:]  # exclude self, shape (N, k)

    all_dists = knn_dists.ravel()
    r_min = max(np.quantile(all_dists, 0.02), eps)
    r_max = np.quantile(all_dists, 0.90)
    if r_max <= r_min:
        return 1.0

    radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)

    # C(l) ~ fraction of pairs within radius l
    C_vals = np.empty(n_radii)
    for i, r in enumerate(radii):
        counts = (knn_dists < r).sum(axis=1)  # (N,)
        C_vals[i] = counts.mean() / max(N - 1, 1)

    valid = C_vals > eps
    if valid.sum() < 3:
        return 1.0

    log_r = np.log(radii[valid])
    log_C = np.log(C_vals[valid])

    # Linear fit in the scaling region (middle 60%)
    n_pts = len(log_r)
    lo = n_pts // 5
    hi = n_pts - n_pts // 5
    if hi - lo < 3:
        lo, hi = 0, n_pts

    coeffs = np.polyfit(log_r[lo:hi], log_C[lo:hi], 1)
    nu = max(float(coeffs[0]), 0.1)
    return nu


def corrint_per_sample(
    hidden_states: list[np.ndarray],
    max_points: int = 500,
) -> float:
    """Compute average correlation dimension across layers for one sample.

    Parameters
    ----------
    hidden_states : list of np.ndarray
        L arrays, each of shape (n_tokens, d_hidden).
    max_points : int
        Subsample spatial points if a layer has more than this many rows.

    Returns
    -------
    float
        Average correlation dimension across layers.
    """
    rng = np.random.RandomState(0)
    dims = []
    for X in hidden_states:
        if X.shape[0] < 5:
            continue
        # Subsample large layers for speed
        if X.shape[0] > max_points:
            idx = rng.choice(X.shape[0], max_points, replace=False)
            X = X[idx]
        dims.append(estimate_correlation_dimension(X))
    if not dims:
        return 1.0
    return float(np.mean(dims))


class CorrIntScorer:
    """Per-sample Correlation Integral scorer (Grassberger & Procaccia, 1983).

    Each sample is scored by the average correlation dimension of its
    hidden-state matrices across L layers:

        Score(i) = (1/L) * sum_l  nu( H_i^l )

    where H_i^l has shape (n_tokens, d_hidden) for sample i, layer l,
    and nu is the estimated correlation dimension.

    Higher score -> richer internal representation -> more informative.

    Requires ``metadata['hidden_states']``: list of N per-sample
    hidden-state lists, each containing L arrays of shape
    (n_tokens, d_hidden).
    """

    def __init__(self) -> None:
        self._last_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "corr_integral"

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

        print(f"  Computing per-sample CorrInt ({n_samples} samples)...")
        scores = np.array(
            [corrint_per_sample(hs) for hs in tqdm(all_hidden, desc="  CorrInt")],
            dtype=np.float64,
        )

        # Lower score = less informative = removed first (rank 1).
        # Use stable_rank_from_scores for consistency with other scorers.
        ranks = np.argsort(np.argsort(scores)) + 1

        n_layers = len(all_hidden[0]) if all_hidden else 0
        self._last_metadata = {
            "method": self.name,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "mean_score": float(scores.mean()),
            "min_score": float(scores.min()),
            "max_score": float(scores.max()),
        }
        print(f"  CorrInt: mean={scores.mean():.2f}, range=[{scores.min():.2f}, {scores.max():.2f}]")

        return (
            pd.DataFrame(
                {
                    "sample_id": np.asarray(sample_ids, dtype=np.int64),
                    "label": np.asarray(labels, dtype=np.int64),
                    "score": scores,
                    "rank": ranks.astype(int),
                    "method": self.name,
                }
            )
            .sort_values("rank", kind="stable")
            .reset_index(drop=True)
        )

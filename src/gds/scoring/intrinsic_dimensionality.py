from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def estimate_intrinsic_dimension_twonn(features: np.ndarray) -> tuple[float, dict[str, float]]:
    """
    Estimate intrinsic dimensionality with the TwoNN method.

    The estimator uses the ratio between the second and first nearest-neighbor
    distances and fits the expected linear relation in log space.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={features.shape}")
    if features.shape[0] < 3:
        raise ValueError("TwoNN requires at least 3 samples.")

    nbrs = NearestNeighbors(n_neighbors=3, metric="euclidean")
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)

    r1 = np.maximum(distances[:, 1].astype(np.float64, copy=False), 1e-12)
    r2 = np.maximum(distances[:, 2].astype(np.float64, copy=False), 1e-12)
    ratios = np.maximum(r2 / r1, 1.0 + 1e-12)

    ratios_sorted = np.sort(ratios)
    n = ratios_sorted.shape[0]
    empirical_cdf = np.arange(1, n + 1, dtype=np.float64) / (n + 1.0)
    x = np.log(ratios_sorted)
    y = -np.log(1.0 - empirical_cdf)

    denom = float(np.dot(x, x))
    if denom <= 0.0:
        raise ValueError("TwoNN failed because the neighbor-distance ratios are degenerate.")

    estimated_dimension = float(np.dot(x, y) / denom)
    estimated_dimension = max(estimated_dimension, 1.0)

    summary = {
        "mean_ratio": float(np.mean(ratios)),
        "median_ratio": float(np.median(ratios)),
        "min_ratio": float(np.min(ratios)),
        "max_ratio": float(np.max(ratios)),
    }
    return estimated_dimension, summary


def reduce_features_via_intrinsic_dimension(
    features: np.ndarray,
    estimated_dimension: float,
) -> tuple[np.ndarray, int]:
    n_samples, n_features = features.shape
    n_components = int(np.clip(np.floor(estimated_dimension), 1, min(n_samples, n_features)))
    reducer = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    reduced = reducer.fit_transform(features)
    return reduced.astype(np.float32), n_components


def compute_knn_density_scores(features: np.ndarray, k: int = 2) -> np.ndarray:
    if k < 1:
        raise ValueError("k must be at least 1.")

    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    nbrs.fit(features)
    distances, _ = nbrs.kneighbors(features)
    return distances[:, 1:].mean(axis=1).astype(np.float32)


class IntrinsicDimensionalityTwoNNScorer(SampleScorer):
    def __init__(self) -> None:
        self._last_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "intrinsic_dimensionality_twonn"

    def build_metadata(self) -> dict[str, Any]:
        return dict(self._last_metadata)

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "features" not in metadata:
            raise ValueError("metadata['features'] is required for intrinsic-dimensionality scoring.")

        features = np.asarray(metadata["features"], dtype=np.float32)
        labels_np = np.asarray(labels, dtype=np.int64)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)

        if features.ndim != 2:
            raise ValueError(f"Expected 2D features, got shape={features.shape}")
        if features.shape[0] != sample_ids_np.shape[0]:
            raise ValueError("Feature rows must match the number of sample ids.")

        print(f"  Estimating intrinsic dimension via TwoNN ({features.shape[0]} samples)...")
        estimated_dimension, ratio_summary = estimate_intrinsic_dimension_twonn(features=features)
        print(f"  Estimated dimension: {estimated_dimension:.1f}")
        reduced_features, n_components = reduce_features_via_intrinsic_dimension(
            features=features,
            estimated_dimension=estimated_dimension,
        )
        print(f"  PCA reduced to {n_components} components, computing kNN density scores...")
        scores = compute_knn_density_scores(features=reduced_features, k=2)
        print(f"  Done. Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        self._last_metadata = {
            "estimator": "TwoNN",
            "estimated_dimension": float(estimated_dimension),
            "pca_components": int(n_components),
            "scoring_k": 2,
            "feature_shape": [int(features.shape[0]), int(features.shape[1])],
            "reduced_feature_shape": [int(reduced_features.shape[0]), int(reduced_features.shape[1])],
            **ratio_summary,
        }

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

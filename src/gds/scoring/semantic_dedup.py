from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def compute_minhash_signatures(
    binary_vectors: np.ndarray,
    num_perm: int = 64,
    seed: int = 777,
) -> np.ndarray:
    """Compute MinHash signatures for binarized vectors.

    binary_vectors shape: [n_samples, n_features]
    returns: [n_samples, num_perm]
    """
    if binary_vectors.ndim != 2:
        raise ValueError(f"Expected 2D binary_vectors, got shape={binary_vectors.shape}")

    n_samples, n_features = binary_vectors.shape
    max_hash = np.uint64((1 << 32) - 1)
    prime = np.uint64(4294967311)

    rng = np.random.default_rng(seed)
    a = rng.integers(1, int(prime - 1), size=num_perm, dtype=np.uint64)
    b = rng.integers(0, int(prime - 1), size=num_perm, dtype=np.uint64)

    signatures = np.full((n_samples, num_perm), fill_value=max_hash, dtype=np.uint64)
    feature_indices = np.arange(n_features, dtype=np.uint64)

    for i in range(n_samples):
        active = feature_indices[binary_vectors[i] > 0]
        if active.size == 0:
            continue
        hashed = (a[:, None] * active[None, :] + b[:, None]) % prime
        signatures[i] = hashed.min(axis=1)

    return signatures


def estimate_max_jaccard_from_signatures(
    signatures: np.ndarray,
    rows_per_band: int = 4,
) -> np.ndarray:
    """Estimate per-sample max Jaccard similarity using MinHash+LSH candidates."""
    if signatures.ndim != 2:
        raise ValueError(f"Expected 2D signatures, got shape={signatures.shape}")

    n_samples, num_perm = signatures.shape
    if rows_per_band <= 0:
        raise ValueError("rows_per_band must be > 0")

    num_bands = max(1, num_perm // rows_per_band)
    max_sim = np.zeros(n_samples, dtype=np.float32)
    candidates: list[set[int]] = [set() for _ in range(n_samples)]

    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = min((band_idx + 1) * rows_per_band, num_perm)
        if start >= end:
            continue

        buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        band = signatures[:, start:end]
        for i in range(n_samples):
            key = tuple(band[i].tolist())
            buckets[key].append(i)

        for bucket_indices in buckets.values():
            if len(bucket_indices) < 2:
                continue
            for i in bucket_indices:
                candidates[i].update(j for j in bucket_indices if j != i)

    for i in range(n_samples):
        if not candidates[i]:
            continue
        sig_i = signatures[i]
        sims = [np.mean(sig_i == signatures[j]) for j in candidates[i]]
        max_sim[i] = float(max(sims))

    return max_sim


class SemanticDedupScorer(SampleScorer):
    def __init__(
        self,
        seed: int = 777,
        num_perm: int = 64,
        rows_per_band: int = 4,
        binarize_threshold: float = 0.5,
    ) -> None:
        self.seed = seed
        self.num_perm = num_perm
        self.rows_per_band = rows_per_band
        self.binarize_threshold = binarize_threshold

    @property
    def name(self) -> str:
        return "semantic_dedup"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "embeddings" not in metadata:
            raise ValueError("metadata['embeddings'] is required for semantic dedup scoring.")

        embeddings = np.asarray(metadata["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")

        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)
        binary_vectors = (embeddings > self.binarize_threshold).astype(np.uint8)

        signatures = compute_minhash_signatures(
            binary_vectors=binary_vectors,
            num_perm=self.num_perm,
            seed=self.seed,
        )
        max_jaccard = estimate_max_jaccard_from_signatures(
            signatures=signatures,
            rows_per_band=self.rows_per_band,
        )

        # Low score => highly redundant sample (removed first by subset builder).
        uniqueness = 1.0 - max_jaccard
        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=uniqueness)

        df = pd.DataFrame(
            {
                "sample_id": sample_ids_np,
                "label": labels_np,
                "score": uniqueness.astype(float),
                "rank": ranks.astype(int),
                "method": self.name,
            }
        )
        return df.sort_values("rank", kind="stable").reset_index(drop=True)


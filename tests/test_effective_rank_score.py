# import numpy as np

# from gds.scoring.effective_rank import compute_effective_scores


# def test_effective_scores_correct() -> None:
#     preds = np.array(
#         [
#             [0, 1, 2, 3],
#             [0, 2, 2, 4],
#             [1, 1, 2, 3],
#         ]
#     )
#     labels = np.array([0, 1, 2, 3])
#     scores = compute_effective_scores(predictions=preds, labels=labels)
#     # Sample 0: models [0,0,1] vs label 0 → 2 correct → 2/3
#     # Sample 1: models [1,2,1] vs label 1 → 2 correct → 2/3
#     # Sample 2: models [2,2,2] vs label 2 → 3 correct → 3/3
#     # Sample 3: models [3,4,3] vs label 3 → 2 correct → 2/3
#     expected = np.array([2/3, 2/3, 1.0, 2/3], dtype=np.float32)
#     assert np.allclose(scores, expected)

from __future__ import annotations

import numpy as np
import pytest

from gds.scoring.effective_rank import EffectiveRankScorer, compute_effective_rank


# ---------------------------------------------------------------------------
# compute_effective_rank
# ---------------------------------------------------------------------------

def test_compute_effective_rank_returns_correct_shape() -> None:
    features = np.random.default_rng(0).normal(size=(40, 16)).astype(np.float32)
    scores = compute_effective_rank(features)
    assert scores.shape == (40,)


def test_compute_effective_rank_dtype_is_float32() -> None:
    features = np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32)
    scores = compute_effective_rank(features)
    assert scores.dtype == np.float32


def test_compute_effective_rank_outliers_score_higher_than_duplicates() -> None:
    """Samples that extend the representation space should score higher
    than near-duplicate (redundant) samples."""
    rng = np.random.default_rng(2)
    base     = rng.normal(scale=0.01, size=(37, 8)).astype(np.float32)
    outliers = rng.normal(scale=5.0,  size=(3,  8)).astype(np.float32)
    features = np.vstack([base, outliers])

    scores = compute_effective_rank(features)

    outlier_mean   = scores[-3:].mean()
    redundant_mean = scores[:37].mean()
    assert outlier_mean > redundant_mean, (
        f"Expected outlier mean ({outlier_mean:.4f}) > redundant mean ({redundant_mean:.4f})"
    )


def test_compute_effective_rank_raises_on_1d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        compute_effective_rank(np.ones(10, dtype=np.float32))


def test_compute_effective_rank_raises_on_too_few_samples() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        compute_effective_rank(np.ones((2, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# EffectiveRankScorer
# ---------------------------------------------------------------------------

def _make_features(n: int, d: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_effective_rank_scorer_returns_correct_columns() -> None:
    scorer = EffectiveRankScorer()
    df = scorer.score(
        sample_ids=list(range(40)),
        labels=[idx % 10 for idx in range(40)],
        metadata={"features": _make_features(40, 8)},
    )
    assert list(df.columns) == ["sample_id", "label", "score", "rank", "method"]


def test_effective_rank_scorer_rank_is_sequential() -> None:
    scorer = EffectiveRankScorer()
    df = scorer.score(
        sample_ids=list(range(40)),
        labels=[0] * 40,
        metadata={"features": _make_features(40, 8)},
    )
    assert df["rank"].tolist() == list(range(40))


def test_effective_rank_scorer_method_name() -> None:
    scorer = EffectiveRankScorer()
    df = scorer.score(
        sample_ids=list(range(20)),
        labels=[0] * 20,
        metadata={"features": _make_features(20, 6)},
    )
    assert (df["method"] == "effective_rank").all()


def test_effective_rank_scorer_metadata_keys() -> None:
    scorer = EffectiveRankScorer()
    scorer.score(
        sample_ids=list(range(30)),
        labels=[0] * 30,
        metadata={"features": _make_features(30, 10)},
    )
    meta = scorer.build_metadata()
    for key in ("method", "feature_shape", "mean_score", "min_score", "max_score"):
        assert key in meta, f"Missing key: {key}"


def test_effective_rank_scorer_raises_without_features() -> None:
    scorer = EffectiveRankScorer()
    with pytest.raises(ValueError, match="features"):
        scorer.score(sample_ids=[0, 1, 2], labels=[0, 0, 0], metadata={})


def test_effective_rank_scorer_raises_on_mismatched_lengths() -> None:
    scorer = EffectiveRankScorer()
    with pytest.raises(ValueError):
        scorer.score(
            sample_ids=list(range(5)),
            labels=[0] * 5,
            metadata={"features": _make_features(10, 4)},
        )
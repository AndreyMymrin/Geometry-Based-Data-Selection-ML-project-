from __future__ import annotations

import numpy as np
import pytest

from gds.scoring.effective_rank import EffectiveRankScorer, erank, average_erank


# ---------------------------------------------------------------------------
# erank (Roy & Vetterli, 2007)
# ---------------------------------------------------------------------------

def test_erank_identity_matrix() -> None:
    """Identity matrix should have maximal erank (= min(n, d))."""
    X = np.eye(5, dtype=np.float64)
    er = erank(X)
    assert abs(er - 5.0) < 0.01, f"erank(I_5) = {er}, expected ~5.0"


def test_erank_rank1_matrix() -> None:
    """Rank-1 matrix should have erank ~1."""
    X = np.ones((10, 5), dtype=np.float64)
    er = erank(X)
    assert abs(er - 1.0) < 0.1, f"erank(rank-1) = {er}, expected ~1.0"


def test_erank_raises_on_1d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        erank(np.ones(10, dtype=np.float64))


def test_erank_raises_on_too_few_rows() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        erank(np.ones((1, 5), dtype=np.float64))


def test_erank_in_valid_range() -> None:
    """erank should be in [1, min(n, d)]."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 8))
    er = erank(X)
    assert 1.0 <= er <= min(20, 8) + 0.01


# ---------------------------------------------------------------------------
# average_erank (per-sample across layers)
# ---------------------------------------------------------------------------

def test_average_erank_multiple_layers() -> None:
    rng = np.random.default_rng(0)
    hidden_states = [rng.normal(size=(16, 32)) for _ in range(4)]
    avg = average_erank(hidden_states)
    assert 1.0 <= avg <= 16.0


# ---------------------------------------------------------------------------
# EffectiveRankScorer
# ---------------------------------------------------------------------------

def _make_hidden_states(n: int, n_layers: int, n_tokens: int, d: int, seed: int = 0):
    """Create synthetic hidden states: list of N sample lists, each with L arrays."""
    rng = np.random.default_rng(seed)
    return [
        [rng.normal(size=(n_tokens, d)).astype(np.float64) for _ in range(n_layers)]
        for _ in range(n)
    ]


def test_scorer_returns_correct_columns() -> None:
    scorer = EffectiveRankScorer()
    hs = _make_hidden_states(20, 4, 16, 32)
    df = scorer.score(
        sample_ids=list(range(20)),
        labels=[i % 5 for i in range(20)],
        metadata={"hidden_states": hs},
    )
    assert list(df.columns) == ["sample_id", "label", "score", "rank", "method"]


def test_scorer_rank_is_sequential() -> None:
    scorer = EffectiveRankScorer()
    hs = _make_hidden_states(20, 4, 16, 32)
    df = scorer.score(
        sample_ids=list(range(20)),
        labels=[0] * 20,
        metadata={"hidden_states": hs},
    )
    assert sorted(df["rank"].tolist()) == list(range(1, 21))


def test_scorer_method_name() -> None:
    scorer = EffectiveRankScorer()
    hs = _make_hidden_states(10, 2, 8, 16)
    df = scorer.score(
        sample_ids=list(range(10)),
        labels=[0] * 10,
        metadata={"hidden_states": hs},
    )
    assert (df["method"] == "effective_rank").all()


def test_scorer_raises_without_hidden_states() -> None:
    scorer = EffectiveRankScorer()
    with pytest.raises(ValueError, match="hidden_states"):
        scorer.score(sample_ids=[0, 1], labels=[0, 0], metadata={})


def test_scorer_raises_on_mismatched_lengths() -> None:
    scorer = EffectiveRankScorer()
    hs = _make_hidden_states(5, 2, 8, 16)
    with pytest.raises(ValueError, match="entries"):
        scorer.score(sample_ids=list(range(3)), labels=[0] * 3, metadata={"hidden_states": hs})


def test_scorer_low_rank_samples_get_low_rank() -> None:
    """Samples with rank-1 hidden states should score lower (removed first)
    than samples with full-rank hidden states."""
    rng = np.random.default_rng(42)
    n = 20
    hs = []
    for i in range(n):
        if i < 10:
            # Low-rank: repeat same vector
            v = rng.normal(size=(1, 32))
            layer = np.repeat(v, 16, axis=0)
            layer += rng.normal(size=(16, 32)) * 0.001  # tiny noise for stability
        else:
            # Full-rank: random
            layer = rng.normal(size=(16, 32))
        hs.append([layer])

    scorer = EffectiveRankScorer()
    df = scorer.score(
        sample_ids=list(range(n)),
        labels=[0] * n,
        metadata={"hidden_states": hs},
    )

    low_rank_avg = df[df["sample_id"] < 10]["rank"].mean()
    full_rank_avg = df[df["sample_id"] >= 10]["rank"].mean()
    assert low_rank_avg < full_rank_avg, (
        f"Low-rank samples should have lower rank (removed first): "
        f"low_rank_avg={low_rank_avg:.1f}, full_rank_avg={full_rank_avg:.1f}"
    )

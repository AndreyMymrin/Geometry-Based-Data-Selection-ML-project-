import numpy as np

from gds.scoring.semantic_dedup import (
    SemanticDedupScorer,
    compute_minhash_signatures,
    estimate_max_jaccard_from_signatures,
)


def test_minhash_signatures_identical_rows_match() -> None:
    vectors = np.array(
        [
            [1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    sig = compute_minhash_signatures(vectors, num_perm=32, seed=7)
    assert np.array_equal(sig[0], sig[1])
    assert not np.array_equal(sig[0], sig[2])


def test_semantic_dedup_scores_duplicates_as_more_redundant() -> None:
    embeddings = np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scorer = SemanticDedupScorer(seed=7, num_perm=64, rows_per_band=4, binarize_threshold=0.5)
    df = scorer.score(
        sample_ids=[0, 1, 2, 3],
        labels=[0, 0, 1, 1],
        metadata={"embeddings": embeddings},
    )

    by_id = df.set_index("sample_id")
    # Identical vectors should be most redundant => lower uniqueness score.
    assert by_id.loc[0, "score"] < by_id.loc[2, "score"]
    assert by_id.loc[1, "score"] < by_id.loc[3, "score"]


def test_estimate_max_jaccard_is_zero_without_candidates() -> None:
    signatures = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
        ],
        dtype=np.uint64,
    )
    sims = estimate_max_jaccard_from_signatures(signatures=signatures, rows_per_band=2)
    assert np.allclose(sims, np.zeros(3, dtype=np.float32))

import numpy as np
import pytest

pytest.importorskip("sklearn")

from gds.scoring.intrinsic_dimensionality import (  # noqa: E402
    IntrinsicDimensionalityTwoNNScorer,
    estimate_intrinsic_dimension_twonn,
)


def test_estimate_intrinsic_dimension_twonn_returns_positive_dimension() -> None:
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(64, 3))
    projection = rng.normal(size=(3, 12))
    features = (latent @ projection).astype(np.float32)

    estimated_dimension, ratio_summary = estimate_intrinsic_dimension_twonn(features=features)

    assert estimated_dimension > 0
    assert ratio_summary["max_ratio"] >= ratio_summary["min_ratio"] >= 1.0


def test_intrinsic_dimensionality_twonn_scorer_returns_ranked_dataframe() -> None:
    rng = np.random.default_rng(1)
    features = rng.normal(size=(40, 8)).astype(np.float32)
    scorer = IntrinsicDimensionalityTwoNNScorer()

    df = scorer.score(
        sample_ids=list(range(40)),
        labels=[idx % 10 for idx in range(40)],
        metadata={"features": features},
    )
    metadata = scorer.build_metadata()

    assert list(df.columns) == ["sample_id", "label", "score", "rank", "method"]
    assert df["rank"].tolist() == list(range(40))
    assert (df["method"] == "intrinsic_dimensionality_twonn").all()
    assert metadata["pca_components"] >= 1
    assert metadata["estimator"] == "TwoNN"
    assert metadata["scoring_k"] == 2

from __future__ import annotations

from gds.scoring.base import SampleScorer
from gds.scoring.error_rate import ErrorRateEnsembleScorer
from gds.scoring.intrinsic_dimensionality import IntrinsicDimensionalityTwoNNScorer
from gds.scoring.random_scorer import RandomScorer


def get_scorer(method: str, random_seed: int = 777) -> SampleScorer:
    if method == "error_rate_ensemble":
        return ErrorRateEnsembleScorer()
    if method == "intrinsic_dimensionality_twonn":
        return IntrinsicDimensionalityTwoNNScorer()
    if method == "random":
        return RandomScorer(seed=random_seed)
    raise ValueError(f"Unknown scoring method: {method}")


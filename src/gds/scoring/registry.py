from __future__ import annotations

from gds.scoring.base import SampleScorer
from gds.scoring.effective_rank import EffectiveRankScorer
from gds.scoring.error_rate import ErrorRateEnsembleScorer
from gds.scoring.random_scorer import RandomScorer


def get_scorer(method: str, random_seed: int = 777) -> SampleScorer:
    if method == "error_rate_ensemble":
        return ErrorRateEnsembleScorer()
    if method == "random":
        return RandomScorer(seed=random_seed)
    if method == "effective_rank":
        return EffectiveRankScorer()
    raise ValueError(f"Unknown scoring method: {method}")


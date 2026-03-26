from __future__ import annotations

from gds.scoring.base import SampleScorer
from gds.scoring.corr_integral import CorrIntScorer
from gds.scoring.effective_rank import EffectiveRankScorer
from gds.scoring.forgetting import ForgettingEventScorer
from gds.scoring.heuristic_filter import HeuristicFilteringScorer
from gds.scoring.intrinsic_dimensionality import IntrinsicDimensionalityTwoNNScorer
from gds.scoring.llm_classifier import LLMClassifierScorer
from gds.scoring.perplexity import PerplexityFilteringScorer
from gds.scoring.random_scorer import RandomScorer
from gds.scoring.semantic_dedup import SemanticDedupScorer

# Methods that need per-sample hidden states from a pretrained model
# (ResNet-18 for images, Qwen2-0.5B for text).
_HIDDEN_STATE_METHODS = {"effective_rank", "corr_integral"}

# Methods that need flat pretrained features (N, d) — last layer of
# pretrained model, mean-pooled.
_FEATURE_METHODS = {
    "intrinsic_dimensionality_twonn",
    "semantic_dedup",
}

# Methods that need a trained text model to compute per-sample scores
# (perplexity, entropy).
_TEXT_MODEL_METHODS = {"perplexity_filtering", "llm_classifier"}

# Methods that work on raw text chunks without model training.
_TEXT_HEURISTIC_METHODS = {"heuristic_filtering"}

# Methods that are only available for text datasets.
TEXT_ONLY_METHODS = _TEXT_MODEL_METHODS | _TEXT_HEURISTIC_METHODS | {"semantic_dedup"}


def get_scorer(method: str, random_seed: int = 777) -> SampleScorer:
    if method == "forgetting_events":
        return ForgettingEventScorer()
    if method == "effective_rank":
        return EffectiveRankScorer()
    if method == "intrinsic_dimensionality_twonn":
        return IntrinsicDimensionalityTwoNNScorer()
    if method == "corr_integral":
        return CorrIntScorer()
    if method == "perplexity_filtering":
        return PerplexityFilteringScorer()
    if method == "semantic_dedup":
        return SemanticDedupScorer()
    if method == "heuristic_filtering":
        return HeuristicFilteringScorer()
    if method == "llm_classifier":
        return LLMClassifierScorer()
    if method == "random":
        return RandomScorer(seed=random_seed)
    raise ValueError(f"Unknown scoring method: {method}")


def is_forgetting_method(method: str) -> bool:
    """Return True if *method* uses forgetting-event scoring."""
    return method == "forgetting_events"


def is_hidden_state_method(method: str) -> bool:
    """Return True if *method* needs per-sample hidden states from a pretrained model."""
    return method in _HIDDEN_STATE_METHODS


def is_feature_method(method: str) -> bool:
    """Return True if *method* needs flat pretrained features."""
    return method in _FEATURE_METHODS


def is_text_model_method(method: str) -> bool:
    """Return True if *method* needs a trained LM to compute per-sample scores."""
    return method in _TEXT_MODEL_METHODS


def is_text_heuristic_method(method: str) -> bool:
    """Return True if *method* uses heuristic text quality features."""
    return method in _TEXT_HEURISTIC_METHODS

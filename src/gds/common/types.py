from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SampleRecord:
    sample_id: int
    label: int
    input_ref: Any


@dataclass(frozen=True)
class ScoreResult:
    sample_id: int
    score: float
    rank: int
    method: str


@dataclass(frozen=True)
class SubsetSpec:
    method: str
    percent_removed: int
    retained_ids: list[int]


@dataclass(frozen=True)
class RunResult:
    method: str
    percent_removed: int
    seed: int
    best_val_acc: float
    best_val_loss: float
    test_acc: float


class FeatureProvider(ABC):
    """Modality-agnostic feature extraction hook for future scorers."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ImageFeatureProvider(FeatureProvider):
    @property
    def name(self) -> str:
        return "image_feature_provider"

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        return batch


class TextFeatureProvider(FeatureProvider):
    @property
    def name(self) -> str:
        return "text_feature_provider"

    def extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Text feature extraction is not implemented in v1.")


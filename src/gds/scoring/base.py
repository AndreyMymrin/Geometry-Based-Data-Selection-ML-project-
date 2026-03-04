from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class SampleScorer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return columns: sample_id, label, score, rank, method."""
        raise NotImplementedError


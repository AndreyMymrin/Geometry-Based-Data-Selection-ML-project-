from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gds.scoring.base import SampleScorer


class RandomScorer(SampleScorer):
    def __init__(self, seed: int = 777) -> None:
        self.seed = seed

    @property
    def name(self) -> str:
        return "random"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(sample_ids_np))

        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))

        # Random score is rank-normalized to keep artifact schema consistent.
        score = rank / max(1, len(rank) - 1)

        df = pd.DataFrame(
            {
                "sample_id": sample_ids_np,
                "label": labels_np,
                "score": score.astype(float),
                "rank": rank.astype(int),
                "method": self.name,
            }
        )
        return df.sort_values("rank", kind="stable").reset_index(drop=True)


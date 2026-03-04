import numpy as np

from gds.scoring.utils import stable_rank_from_scores


def test_ranking_tie_breaks_by_sample_id() -> None:
    sample_ids = np.array([10, 5, 3, 20])
    scores = np.array([0.1, 0.1, 0.2, 0.2])
    rank = stable_rank_from_scores(sample_ids=sample_ids, scores=scores)
    ordered_ids = sample_ids[np.argsort(rank)]
    assert ordered_ids.tolist() == [5, 10, 3, 20]


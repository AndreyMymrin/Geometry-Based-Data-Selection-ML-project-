import pandas as pd

from gds.subsets.builder import build_subsets_from_ranking


def test_subset_sizes_match_percentiles() -> None:
    n = 100
    ranking_df = pd.DataFrame(
        {
            "sample_id": list(range(n)),
            "label": [0] * n,
            "score": [0.1] * n,
            "rank": list(range(n)),
            "method": ["forgetting_events"] * n,
        }
    )
    subsets = build_subsets_from_ranking(
        ranking_df=ranking_df,
        method="forgetting_events",
        percentiles=[0, 5, 10, 60],
    )
    retained_counts = [len(s.retained_ids) for s in subsets]
    assert retained_counts == [100, 95, 90, 40]


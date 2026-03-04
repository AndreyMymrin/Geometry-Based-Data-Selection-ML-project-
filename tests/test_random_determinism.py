from gds.scoring.random_scorer import RandomScorer


def test_random_scorer_is_deterministic_for_fixed_seed() -> None:
    scorer_a = RandomScorer(seed=777)
    scorer_b = RandomScorer(seed=777)

    sample_ids = list(range(50))
    labels = [i % 10 for i in sample_ids]
    df_a = scorer_a.score(sample_ids=sample_ids, labels=labels)
    df_b = scorer_b.score(sample_ids=sample_ids, labels=labels)
    assert df_a["sample_id"].tolist() == df_b["sample_id"].tolist()
    assert df_a["rank"].tolist() == df_b["rank"].tolist()


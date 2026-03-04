import numpy as np

from gds.scoring.error_rate import compute_error_rate_scores


def test_error_rate_scores_correct() -> None:
    preds = np.array(
        [
            [0, 1, 2, 3],
            [0, 2, 2, 4],
            [1, 1, 2, 3],
        ]
    )
    labels = np.array([0, 1, 2, 3])
    scores = compute_error_rate_scores(predictions=preds, labels=labels)
    assert np.allclose(scores, np.array([1 / 3, 1 / 3, 0.0, 1 / 3], dtype=np.float32))


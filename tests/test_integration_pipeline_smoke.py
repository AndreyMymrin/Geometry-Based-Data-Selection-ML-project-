from pathlib import Path

import pandas as pd

from gds.subsets.builder import build_subsets_from_ranking, save_subsets


def test_pipeline_smoke_subset_build(tmp_path: Path) -> None:
    ranking_df = pd.DataFrame(
        {
            "sample_id": list(range(2000)),
            "label": [i % 10 for i in range(2000)],
            "score": [i / 2000 for i in range(2000)],
            "rank": list(range(2000)),
            "method": ["error_rate_ensemble"] * 2000,
        }
    )
    subsets = build_subsets_from_ranking(
        ranking_df=ranking_df,
        method="error_rate_ensemble",
        percentiles=[0, 5],
    )
    out_dir = tmp_path / "subsets" / "error_rate_ensemble"
    save_subsets(subsets=subsets, output_dir=out_dir)

    assert (out_dir / "p00.json").exists()
    assert (out_dir / "p05.json").exists()
    assert (out_dir / "manifest.json").exists()


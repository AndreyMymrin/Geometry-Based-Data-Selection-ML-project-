from pathlib import Path

import pandas as pd

from gds.analysis.aggregate import aggregate_curves, collect_run_summaries, save_summary
from gds.common.io import write_json


def test_aggregation_mean_std(tmp_path: Path) -> None:
    root = tmp_path / "training"
    payloads = [
        {"method": "forgetting_events", "percent_removed": 0, "seed": 101, "best_val_acc": 0.90, "best_val_loss": 0.3, "test_acc": 0.89},
        {"method": "forgetting_events", "percent_removed": 0, "seed": 202, "best_val_acc": 0.92, "best_val_loss": 0.25, "test_acc": 0.91},
        {"method": "forgetting_events", "percent_removed": 5, "seed": 101, "best_val_acc": 0.88, "best_val_loss": 0.33, "test_acc": 0.87},
    ]
    for item in payloads:
        path = root / item["method"] / f"p{item['percent_removed']:02d}" / f"seed{item['seed']}" / "run_summary.json"
        write_json(path, item)

    run_df = collect_run_summaries(root)
    summary_df = aggregate_curves(run_df)
    assert not summary_df.empty

    row = summary_df[
        (summary_df["method"] == "forgetting_events") & (summary_df["percent_removed"] == 0)
    ].iloc[0]
    assert abs(float(row["val_acc_mean"]) - 0.91) < 1e-6

    csv_path, json_path = save_summary(summary_df=summary_df, summary_dir=tmp_path / "summary")
    assert csv_path.exists()
    assert json_path.exists()


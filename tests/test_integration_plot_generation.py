from pathlib import Path

import os
import pandas as pd
import pytest

from gds.analysis.aggregate import save_plots

pytestmark = pytest.mark.skipif(
    os.environ.get("GDS_ENABLE_PLOT_TEST", "0") != "1",
    reason="Set GDS_ENABLE_PLOT_TEST=1 to enable plot integration test.",
)


def test_plot_generation(tmp_path: Path) -> None:
    summary_df = pd.DataFrame(
        {
            "method": ["forgetting_events", "forgetting_events", "random", "random"],
            "percent_removed": [0, 5, 0, 5],
            "val_acc_mean": [0.90, 0.91, 0.89, 0.88],
            "val_acc_std": [0.01, 0.02, 0.01, 0.03],
            "test_acc_mean": [0.89, 0.90, 0.88, 0.87],
            "test_acc_std": [0.02, 0.01, 0.02, 0.03],
        }
    )
    val_plot, test_plot = save_plots(summary_df=summary_df, plot_dir=tmp_path / "plots", dpi=90)
    assert val_plot.exists()
    assert test_plot.exists()

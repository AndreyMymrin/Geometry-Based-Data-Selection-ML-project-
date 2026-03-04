from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.analysis.aggregate import aggregate_curves, collect_run_summaries, save_plots, save_summary
from gds.common.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate run metrics and plot curves.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    run_df = collect_run_summaries(artifacts_dir / "training")
    if run_df.empty:
        raise RuntimeError("No run_summary.json files found. Train first.")

    summary_df = aggregate_curves(run_df)
    summary_dir = artifacts_dir / "summary"
    plot_dir = artifacts_dir / "plots"
    csv_path, json_path = save_summary(summary_df=summary_df, summary_dir=summary_dir)
    val_plot, test_plot = save_plots(
        summary_df=summary_df,
        plot_dir=plot_dir,
        dpi=int(cfg["analysis"]["plot_dpi"]),
    )

    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")
    print(f"Saved val plot: {val_plot}")
    print(f"Saved test plot: {test_plot}")


if __name__ == "__main__":
    main()


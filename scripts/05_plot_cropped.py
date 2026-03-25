"""Re-generate all charts but limited to a subset of retention budgets.

Usage:
    python scripts/05_plot_cropped.py --config configs/experiment/cifar10.yaml \
        --percentiles 0 10 20 30 50

Outputs go to <artifacts_dir>/plots/cropped/ with the same filenames as the
full plots in <artifacts_dir>/plots/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.analysis.aggregate import aggregate_curves, collect_run_summaries, save_plots
from gds.common.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate charts for a cropped subset of retention budgets.",
    )
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument(
        "--percentiles",
        type=int,
        nargs="+",
        default=[0, 10, 20, 30, 50],
        help="Percent-removed values to include (e.g. 0 10 20 30 50).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    dataset_name = cfg["dataset"].get("name", "mnist")
    primary_metric = cfg.get("analysis", {}).get("primary_metric", "accuracy")
    dpi = int(cfg["analysis"]["plot_dpi"])

    # Collect and aggregate all runs
    run_df = collect_run_summaries(artifacts_dir / "training")
    if run_df.empty:
        raise RuntimeError("No run_summary.json files found. Train first.")

    summary_df = aggregate_curves(run_df)

    # Filter to requested percentiles only
    keep_set = set(args.percentiles)
    cropped_df = summary_df[summary_df["percent_removed"].isin(keep_set)].copy()

    if cropped_df.empty:
        available = sorted(summary_df["percent_removed"].unique())
        raise RuntimeError(
            f"No data for percentiles {sorted(keep_set)}. "
            f"Available: {available}"
        )

    kept = sorted(cropped_df["percent_removed"].unique())
    print(f"Cropped to percent_removed = {kept}")
    print(f"  → retentions: {sorted(cropped_df['retention'].unique(), reverse=True)}")

    plot_dir = artifacts_dir / "plots" / "cropped"
    generated = save_plots(
        summary_df=cropped_df,
        plot_dir=plot_dir,
        dpi=dpi,
        dataset_name=dataset_name,
        primary_metric=primary_metric,
        artifacts_dir=artifacts_dir,
    )
    print(f"\nGenerated {len(generated)} artifacts in {plot_dir}")


if __name__ == "__main__":
    main()

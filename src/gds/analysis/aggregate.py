from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

from gds.common.io import ensure_dir, read_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_run_summaries(training_root: Path) -> pd.DataFrame:
    records: list[dict] = []
    for summary_path in training_root.glob("*/p*/seed*/run_summary.json"):
        data = read_json(summary_path)
        records.append(data)
    if not records:
        return pd.DataFrame(
            columns=[
                "method",
                "percent_removed",
                "seed",
                "best_val_acc",
                "best_val_loss",
                "test_acc",
            ]
        )
    return pd.DataFrame(records)


def aggregate_curves(run_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        run_df.groupby(["method", "percent_removed"], as_index=False)
        .agg(
            val_acc_mean=("best_val_acc", "mean"),
            val_acc_std=("best_val_acc", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
        )
        .sort_values(["method", "percent_removed"], kind="stable")
    )
    return grouped


def save_summary(summary_df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    ensure_dir(summary_dir)
    csv_path = summary_dir / "curve_metrics.csv"
    json_path = summary_dir / "curve_metrics.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


def _plot_metric(
    summary_df: pd.DataFrame,
    metric_col: str,
    std_col: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in sorted(summary_df["method"].unique().tolist()):
        method_df = summary_df[summary_df["method"] == method].sort_values("percent_removed")
        x = method_df["percent_removed"].to_numpy()
        y = method_df[metric_col].to_numpy()
        std = method_df[std_col].fillna(0.0).to_numpy()
        ax.plot(x, y, marker="o", label=method)
        ax.fill_between(x, y - std, y + std, alpha=0.15)
    ax.set_xlabel("Percent Easiest Samples Removed")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plots(summary_df: pd.DataFrame, plot_dir: Path, dpi: int = 150) -> tuple[Path, Path]:
    val_plot = plot_dir / "metric_vs_percentile_val_acc.png"
    test_plot = plot_dir / "metric_vs_percentile_test_acc.png"
    _plot_metric(
        summary_df=summary_df,
        metric_col="val_acc_mean",
        std_col="val_acc_std",
        title="Validation Accuracy vs Removed Percentile",
        out_path=val_plot,
        dpi=dpi,
    )
    _plot_metric(
        summary_df=summary_df,
        metric_col="test_acc_mean",
        std_col="test_acc_std",
        title="Test Accuracy vs Removed Percentile",
        out_path=test_plot,
        dpi=dpi,
    )
    return val_plot, test_plot

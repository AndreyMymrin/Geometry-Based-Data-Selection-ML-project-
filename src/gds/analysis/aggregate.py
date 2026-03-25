from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from gds.common.io import ensure_dir, read_json, write_json
from gds.scoring.registry import is_forgetting_method

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Colour palette — one colour per method, consistent across all charts
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "forgetting_events": "#2ecc71",
    "effective_rank": "#e74c3c",
    "intrinsic_dimensionality_twonn": "#9b59b6",
    "corr_integral": "#2980b9",
    "perplexity_filtering": "#e67e22",
    "semantic_dedup": "#1abc9c",
    "heuristic_filtering": "#f1c40f",
    "llm_classifier": "#3498db",
    "random": "#95a5a6",
}
METHOD_MARKERS = {
    "forgetting_events": "o",
    "effective_rank": "s",
    "intrinsic_dimensionality_twonn": "D",
    "corr_integral": "d",
    "perplexity_filtering": "^",
    "semantic_dedup": "v",
    "heuristic_filtering": "P",
    "llm_classifier": "X",
    "random": "*",
}


def _color(method: str) -> str:
    return METHOD_COLORS.get(method, "#34495e")


def _marker(method: str) -> str:
    return METHOD_MARKERS.get(method, "o")


def _nice_name(method: str) -> str:
    return method.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Data collection & aggregation
# ---------------------------------------------------------------------------


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
            val_loss_mean=("best_val_loss", "mean"),
            val_loss_std=("best_val_loss", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            n_seeds=("seed", "count"),
        )
        .sort_values(["method", "percent_removed"], kind="stable")
    )
    # Retention fraction (1.0 = full data, 0.1 = 10 %)
    grouped["retention"] = 1.0 - grouped["percent_removed"] / 100.0
    return grouped


def save_summary(summary_df: pd.DataFrame, summary_dir: Path) -> tuple[Path, Path]:
    ensure_dir(summary_dir)
    csv_path = summary_dir / "curve_metrics.csv"
    json_path = summary_dir / "curve_metrics.json"
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    return csv_path, json_path


# ---------------------------------------------------------------------------
# Scaling-law fitting  (power law:  metric = a * N^b + c)
# ---------------------------------------------------------------------------


def _power_law(n: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.power(n, b) + c


def fit_scaling_law(
    retention: np.ndarray,
    metric: np.ndarray,
) -> dict[str, Any]:
    """Fit  metric = a * retention^b + c  and return coefficients + R^2."""
    try:
        popt, _ = curve_fit(
            _power_law,
            retention,
            metric,
            p0=[1.0, 0.5, 0.0],
            maxfev=10000,
        )
        predicted = _power_law(retention, *popt)
        ss_res = np.sum((metric - predicted) ** 2)
        ss_tot = np.sum((metric - np.mean(metric)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"a": popt[0], "b": popt[1], "c": popt[2], "r2": r2}
    except (RuntimeError, ValueError):
        return {"a": float("nan"), "b": float("nan"), "c": float("nan"), "r2": float("nan")}


# ---------------------------------------------------------------------------
# Chart 1: Accuracy / Loss vs Data Retention (all methods, with std bands)
# ---------------------------------------------------------------------------


def plot_retention_curve(
    summary_df: pd.DataFrame,
    metric_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
    higher_is_better: bool = True,
) -> None:
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in sorted(summary_df["method"].unique()):
        mdf = summary_df[summary_df["method"] == method].sort_values("retention")
        x = mdf["retention"].to_numpy() * 100  # show as percentage
        y = mdf[metric_col].to_numpy()
        std = mdf[std_col].fillna(0.0).to_numpy()
        ax.plot(
            x, y,
            marker=_marker(method), color=_color(method),
            linewidth=2, markersize=8, label=_nice_name(method),
        )
        ax.fill_between(x, y - std, y + std, color=_color(method), alpha=0.12)

    ax.set_xlabel("Data Retained (%)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, 105)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 2: Scaling-law plot (log–log with power-law fit)
# ---------------------------------------------------------------------------


def plot_scaling_law(
    summary_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
) -> dict[str, dict]:
    """Log-log scaling-law plot with fitted curves. Returns fit params per method."""
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))
    fits: dict[str, dict] = {}

    for method in sorted(summary_df["method"].unique()):
        mdf = summary_df[summary_df["method"] == method].sort_values("retention")
        retention = mdf["retention"].to_numpy()
        y = mdf[metric_col].to_numpy()

        # Skip if too few points
        if len(retention) < 2:
            continue

        # Scatter actual points
        ax.scatter(
            retention * 100, y,
            marker=_marker(method), color=_color(method),
            s=100, zorder=5, edgecolors="white", linewidths=0.5,
        )

        # Fit and draw smooth curve
        fit = fit_scaling_law(retention, y)
        fits[method] = fit

        if not np.isnan(fit["r2"]):
            x_smooth = np.linspace(retention.min(), retention.max(), 200)
            y_smooth = _power_law(x_smooth, fit["a"], fit["b"], fit["c"])
            ax.plot(
                x_smooth * 100, y_smooth,
                color=_color(method), linewidth=2, alpha=0.8,
                label=f"{_nice_name(method)}  (R²={fit['r2']:.3f})",
            )
        else:
            # Fallback: just connect the dots
            ax.plot(
                retention * 100, y,
                color=_color(method), linewidth=1.5, linestyle="--",
                label=_nice_name(method),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Data Retained (%)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25, which="both")

    # Nice tick labels for log scale
    ax.set_xticks([10, 30, 70, 100])
    ax.set_xticklabels(["10%", "30%", "70%", "100%"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fits


# ---------------------------------------------------------------------------
# Chart 3: Relative performance (normalised to 100 % baseline)
# ---------------------------------------------------------------------------


def plot_relative_performance(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
    higher_is_better: bool = True,
) -> None:
    """Bar chart: performance at each retention level relative to 100 % baseline."""
    ensure_dir(out_path.parent)
    methods = sorted(summary_df["method"].unique())
    retentions = sorted(summary_df["retention"].unique())

    # Normalise each method to its own 100 % baseline
    rel_data: dict[str, list[float]] = {}
    for method in methods:
        mdf = summary_df[summary_df["method"] == method]
        baseline_row = mdf[mdf["retention"] == 1.0]
        if baseline_row.empty:
            baseline = mdf[metric_col].max() if higher_is_better else mdf[metric_col].min()
        else:
            baseline = float(baseline_row[metric_col].values[0])
        if baseline == 0:
            baseline = 1e-8
        vals = []
        for r in retentions:
            row = mdf[mdf["retention"] == r]
            if row.empty:
                vals.append(float("nan"))
            else:
                vals.append(float(row[metric_col].values[0]) / baseline * 100)
        rel_data[method] = vals

    fig, ax = plt.subplots(figsize=(12, 6))
    n_methods = len(methods)
    bar_width = 0.8 / max(n_methods, 1)
    x_base = np.arange(len(retentions))

    for i, method in enumerate(methods):
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(
            x_base + offset, rel_data[method],
            width=bar_width, label=_nice_name(method),
            color=_color(method), edgecolor="white", linewidth=0.5,
        )
        # Value labels on bars
        for bar, val in zip(bars, rel_data[method]):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
                )

    ax.axhline(y=100, color="black", linewidth=1, linestyle="--", alpha=0.5, label="100% baseline")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{r*100:.0f}%" for r in retentions])
    ax.set_xlabel("Data Retained", fontsize=13)
    ax.set_ylabel("Relative Performance (%)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower left", framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 4: Method comparison heatmap
# ---------------------------------------------------------------------------


def plot_method_heatmap(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
    fmt: str = ".4f",
) -> None:
    """Heatmap: methods (rows) × retention budgets (columns)."""
    ensure_dir(out_path.parent)
    methods = sorted(summary_df["method"].unique())
    retentions = sorted(summary_df["retention"].unique())

    matrix = np.full((len(methods), len(retentions)), float("nan"))
    for i, method in enumerate(methods):
        for j, r in enumerate(retentions):
            row = summary_df[(summary_df["method"] == method) & (summary_df["retention"] == r)]
            if not row.empty:
                matrix[i, j] = float(row[metric_col].values[0])

    fig, ax = plt.subplots(figsize=(max(8, len(retentions) * 2.5), max(4, len(methods) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn")

    ax.set_xticks(range(len(retentions)))
    ax.set_xticklabels([f"{r*100:.0f}%" for r in retentions], fontsize=11)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([_nice_name(m) for m in methods], fontsize=11)
    ax.set_xlabel("Data Retained", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(methods)):
        for j in range(len(retentions)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val < (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.8, label=metric_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 5: Data efficiency chart (performance drop vs data saved)
# ---------------------------------------------------------------------------


def plot_efficiency(
    summary_df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: Path,
    dpi: int = 150,
    higher_is_better: bool = True,
) -> None:
    """Scatter: x = data saved (%), y = performance drop (%).

    Ideal methods cluster in the bottom-right (lots of data saved, little drop).
    """
    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 7))

    for method in sorted(summary_df["method"].unique()):
        mdf = summary_df[summary_df["method"] == method].sort_values("retention")
        baseline_row = mdf[mdf["retention"] == 1.0]
        if baseline_row.empty:
            continue
        baseline = float(baseline_row[metric_col].values[0])
        if baseline == 0:
            continue

        for _, row in mdf.iterrows():
            r = float(row["retention"])
            if r == 1.0:
                continue
            data_saved = (1 - r) * 100
            if higher_is_better:
                perf_drop = (1 - float(row[metric_col]) / baseline) * 100
            else:
                # For loss: lower is better, so "drop" = increase in loss
                perf_drop = (float(row[metric_col]) / baseline - 1) * 100

            ax.scatter(
                data_saved, perf_drop,
                marker=_marker(method), color=_color(method),
                s=180, edgecolors="white", linewidths=0.8, zorder=5,
            )

        # Invisible point for legend
        ax.scatter([], [], marker=_marker(method), color=_color(method),
                   s=100, label=_nice_name(method))

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Data Dropped (%)", fontsize=13)
    ylabel = "Performance Drop (%)" if higher_is_better else "Loss Increase (%)"
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.25)

    # Annotate the ideal quadrant
    ax.annotate(
        "ideal\n(low drop,\nhigh savings)",
        xy=(0.85, 0.05), xycoords="axes fraction",
        fontsize=9, fontstyle="italic", color="green", alpha=0.6,
        ha="center",
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Chart 6: Forgetting statistics distribution (Toneva et al. 2019, Fig. 1)
# ---------------------------------------------------------------------------


def plot_forgetting_distribution(
    artifacts_dir: Path,
    out_path: Path,
    dpi: int = 150,
) -> bool:
    """Histogram of forgetting event counts across all training samples.

    Reproduces the style of Figure 1 from Toneva et al. (2019):
    x-axis = number of forgetting events, y-axis = number of examples.
    Also annotates unforgettable and never-learned counts.

    Returns True if the plot was generated, False if no data found.
    """
    scores_path = artifacts_dir / "rankings" / "forgetting_events" / "scores.parquet"
    if not scores_path.exists():
        return False

    import pandas as pd

    df = pd.read_parquet(scores_path)
    scores = df["score"].to_numpy()

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))

    n_unforgettable = int((scores == 0).sum())
    max_score = scores.max()
    # Never-learned samples have score = max + 1 (set by ensemble code)
    # Detect them: they are the ones with score > second-highest unique
    unique_sorted = np.sort(np.unique(scores))
    n_never_learned = 0
    if len(unique_sorted) >= 2 and unique_sorted[-1] > unique_sorted[-2] + 0.5:
        n_never_learned = int((scores == unique_sorted[-1]).sum())
        # Exclude never-learned from histogram (plot them separately)
        plot_scores = scores[scores < unique_sorted[-1]]
    else:
        plot_scores = scores

    # Integer bins for forgetting counts
    max_count = int(np.ceil(plot_scores.max())) + 1
    bins = np.arange(-0.5, max_count + 0.5, 1)
    ax.hist(plot_scores, bins=bins, color="#2ecc71", edgecolor="white", linewidth=0.5, alpha=0.85)

    ax.set_xlabel("Number of Forgetting Events", fontsize=13)
    ax.set_ylabel("Number of Examples", fontsize=13)
    ax.set_title("Distribution of Forgetting Events", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")

    # Annotate key statistics
    n_total = len(scores)
    text_lines = [
        f"Total samples: {n_total:,}",
        f"Unforgettable (0 events): {n_unforgettable:,} ({100*n_unforgettable/n_total:.1f}%)",
    ]
    if n_never_learned > 0:
        text_lines.append(
            f"Never learned: {n_never_learned:,} ({100*n_never_learned/n_total:.1f}%)"
        )
    ax.text(
        0.97, 0.95, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Chart 7: Sorted forgetting counts (Toneva et al. 2019, Fig. 2 style)
# ---------------------------------------------------------------------------


def plot_forgetting_sorted(
    artifacts_dir: Path,
    out_path: Path,
    dpi: int = 150,
) -> bool:
    """Plot sorted per-sample forgetting counts (ascending).

    x-axis = example index (sorted), y-axis = forgetting count.
    Shows the transition from unforgettable to highly-forgotten samples.

    Returns True if the plot was generated, False if no data found.
    """
    scores_path = artifacts_dir / "rankings" / "forgetting_events" / "scores.parquet"
    if not scores_path.exists():
        return False

    import pandas as pd

    df = pd.read_parquet(scores_path)
    scores = np.sort(df["score"].to_numpy())

    ensure_dir(out_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(scores))
    ax.plot(x, scores, color="#2ecc71", linewidth=0.8)
    ax.fill_between(x, 0, scores, color="#2ecc71", alpha=0.2)

    ax.set_xlabel("Example Index (sorted by forgetting count)", fontsize=13)
    ax.set_ylabel("Forgetting Events", fontsize=13)
    ax.set_title("Sorted Forgetting Counts per Example", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25)

    # Mark the unforgettable boundary
    n_unforgettable = int((scores == 0).sum())
    if n_unforgettable > 0 and n_unforgettable < len(scores):
        ax.axvline(x=n_unforgettable, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(
            n_unforgettable, ax.get_ylim()[1] * 0.9,
            f"  {n_unforgettable:,} unforgettable",
            fontsize=9, color="red", va="top",
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Quantitative conclusions
# ---------------------------------------------------------------------------


def compute_conclusions(
    summary_df: pd.DataFrame,
    metric_col: str = "test_acc_mean",
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Compute quantitative conclusions about data efficiency."""
    methods = sorted(summary_df["method"].unique())
    conclusions: dict[str, Any] = {"methods": {}}

    for method in methods:
        mdf = summary_df[summary_df["method"] == method].sort_values("retention")
        baseline_row = mdf[mdf["retention"] == 1.0]
        if baseline_row.empty:
            continue
        baseline = float(baseline_row[metric_col].values[0])

        # Per-retention stats
        budget_stats = []
        for _, row in mdf.iterrows():
            r = float(row["retention"])
            val = float(row[metric_col])
            if higher_is_better:
                relative_perf = val / baseline * 100 if baseline != 0 else float("nan")
                drop = (1 - val / baseline) * 100 if baseline != 0 else float("nan")
            else:
                relative_perf = baseline / val * 100 if val != 0 else float("nan")
                drop = (val / baseline - 1) * 100 if baseline != 0 else float("nan")
            budget_stats.append({
                "retention_pct": r * 100,
                "metric_value": val,
                "relative_to_baseline_pct": round(relative_perf, 2),
                "drop_pct": round(drop, 2),
            })

        # Find the most aggressive budget with < 2% drop
        safe_budgets = [
            s for s in budget_stats
            if s["drop_pct"] < 2.0
        ]
        most_aggressive_safe = min(
            safe_budgets, key=lambda s: s["retention_pct"]
        ) if safe_budgets else None

        # Scaling law fit
        retention = mdf["retention"].to_numpy()
        metric = mdf[metric_col].to_numpy()
        scaling_fit = fit_scaling_law(retention, metric)

        conclusions["methods"][method] = {
            "baseline_value": baseline,
            "budget_analysis": budget_stats,
            "most_aggressive_safe_budget": most_aggressive_safe,
            "scaling_law": {
                "formula": f"{metric_col} = {scaling_fit['a']:.4f} * retention^{scaling_fit['b']:.4f} + {scaling_fit['c']:.4f}",
                "r_squared": round(scaling_fit["r2"], 4),
            },
        }

    # Cross-method comparison at each retention level
    retentions = sorted(summary_df["retention"].unique())
    best_at_budget: dict[str, str] = {}
    for r in retentions:
        if r == 1.0:
            continue
        sub = summary_df[summary_df["retention"] == r]
        if sub.empty:
            continue
        if higher_is_better:
            best_row = sub.loc[sub[metric_col].idxmax()]
        else:
            best_row = sub.loc[sub[metric_col].idxmin()]
        best_at_budget[f"{r*100:.0f}%"] = str(best_row["method"])

    conclusions["best_method_per_budget"] = best_at_budget
    conclusions["metric"] = metric_col
    conclusions["higher_is_better"] = higher_is_better
    return conclusions


def format_conclusions_text(conclusions: dict) -> str:
    """Format conclusions dict into human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append("QUANTITATIVE CONCLUSIONS — DATA EFFICIENCY ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Metric: {conclusions['metric']}")
    lines.append(f"Direction: {'higher is better' if conclusions['higher_is_better'] else 'lower is better'}")
    lines.append("")

    for method, info in conclusions["methods"].items():
        lines.append(f"--- {_nice_name(method)} ---")
        lines.append(f"  Baseline (100% data): {info['baseline_value']:.4f}")
        lines.append(f"  Scaling law: {info['scaling_law']['formula']}")
        lines.append(f"  Scaling law R²: {info['scaling_law']['r_squared']}")
        lines.append("  Performance at each budget:")
        for bs in info["budget_analysis"]:
            lines.append(
                f"    {bs['retention_pct']:5.0f}% data → {bs['metric_value']:.4f} "
                f"({bs['relative_to_baseline_pct']:.1f}% of baseline, "
                f"drop={bs['drop_pct']:.1f}%)"
            )
        safe = info["most_aggressive_safe_budget"]
        if safe:
            lines.append(
                f"  Most aggressive safe budget (<2% drop): "
                f"{safe['retention_pct']:.0f}% data "
                f"(drop={safe['drop_pct']:.1f}%)"
            )
        else:
            lines.append("  No budget achieves <2% performance drop.")
        lines.append("")

    lines.append("--- Best Method per Budget ---")
    for budget, method in conclusions.get("best_method_per_budget", {}).items():
        lines.append(f"  {budget} retention → {_nice_name(method)}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_plots — main entry point
# ---------------------------------------------------------------------------


def save_plots(
    summary_df: pd.DataFrame,
    plot_dir: Path,
    dpi: int = 150,
    dataset_name: str = "",
    primary_metric: str = "accuracy",
    artifacts_dir: Path | None = None,
) -> tuple[Path, ...]:
    """Generate all charts and quantitative analysis.

    Returns tuple of generated file paths.
    """
    ensure_dir(plot_dir)
    ds_label = dataset_name.replace("_", " ").title() if dataset_name else ""
    is_loss = primary_metric == "loss"

    if is_loss:
        metric_col = "val_loss_mean"
        std_col = "val_loss_std"
        ylabel = "Validation Loss"
        higher_is_better = False
        test_metric_col = "val_loss_mean"  # text tasks use loss as primary
        test_std_col = "val_loss_std"
        test_ylabel = "Validation Loss"
    else:
        metric_col = "test_acc_mean"
        std_col = "test_acc_std"
        ylabel = "Test Accuracy"
        higher_is_better = True
        test_metric_col = "test_acc_mean"
        test_std_col = "test_acc_std"
        test_ylabel = "Test Accuracy"

    generated: list[Path] = []

    # 1. Retention curve (primary metric)
    p1 = plot_dir / "retention_curve.png"
    plot_retention_curve(
        summary_df, metric_col, std_col, ylabel,
        f"Performance vs Data Budget — {ds_label}" if ds_label else "Performance vs Data Budget",
        p1, dpi, higher_is_better,
    )
    generated.append(p1)
    print(f"  Saved: {p1}")

    # 2. Scaling-law plot (log scale)
    p2 = plot_dir / "scaling_law.png"
    fits = plot_scaling_law(
        summary_df, metric_col, ylabel,
        f"Scaling Law — {ds_label}" if ds_label else "Scaling Law",
        p2, dpi,
    )
    generated.append(p2)
    print(f"  Saved: {p2}")

    # 3. Relative performance bars
    p3 = plot_dir / "relative_performance.png"
    plot_relative_performance(
        summary_df, metric_col,
        f"Relative Performance vs Baseline — {ds_label}" if ds_label else "Relative Performance vs Baseline",
        p3, dpi, higher_is_better,
    )
    generated.append(p3)
    print(f"  Saved: {p3}")

    # 4. Heatmap
    p4 = plot_dir / "method_heatmap.png"
    plot_method_heatmap(
        summary_df, metric_col,
        f"Method × Budget Heatmap — {ds_label}" if ds_label else "Method × Budget Heatmap",
        p4, dpi,
    )
    generated.append(p4)
    print(f"  Saved: {p4}")

    # 5. Data efficiency scatter
    p5 = plot_dir / "data_efficiency.png"
    plot_efficiency(
        summary_df, metric_col,
        f"Data Efficiency — {ds_label}" if ds_label else "Data Efficiency",
        p5, dpi, higher_is_better,
    )
    generated.append(p5)
    print(f"  Saved: {p5}")

    # 6. Validation accuracy curve (if not loss-primary)
    if not is_loss:
        p6 = plot_dir / "retention_curve_val_acc.png"
        plot_retention_curve(
            summary_df, "val_acc_mean", "val_acc_std", "Validation Accuracy",
            f"Validation Accuracy vs Data Budget — {ds_label}" if ds_label else "Validation Accuracy vs Data Budget",
            p6, dpi, True,
        )
        generated.append(p6)
        print(f"  Saved: {p6}")

    # 7. Quantitative conclusions
    conclusions = compute_conclusions(summary_df, metric_col, higher_is_better)
    conclusions_text = format_conclusions_text(conclusions)
    print("\n" + conclusions_text)

    conclusions_path = plot_dir / "conclusions.json"
    write_json(conclusions_path, conclusions)
    generated.append(conclusions_path)
    print(f"\n  Saved conclusions: {conclusions_path}")

    conclusions_txt_path = plot_dir / "conclusions.txt"
    conclusions_txt_path.write_text(conclusions_text)
    generated.append(conclusions_txt_path)
    print(f"  Saved conclusions: {conclusions_txt_path}")

    # 8. Scaling law parameters
    if fits:
        fits_path = plot_dir / "scaling_law_fits.json"
        write_json(fits_path, fits)
        generated.append(fits_path)
        print(f"  Saved scaling law fits: {fits_path}")

    # 9. Forgetting distribution histogram (Toneva et al. 2019, Fig. 1)
    if artifacts_dir is not None:
        p9 = plot_dir / "forgetting_distribution.png"
        if plot_forgetting_distribution(artifacts_dir, p9, dpi):
            generated.append(p9)
            print(f"  Saved: {p9}")

        # 10. Sorted forgetting counts (Toneva et al. 2019, Fig. 2 style)
        p10 = plot_dir / "forgetting_sorted.png"
        if plot_forgetting_sorted(artifacts_dir, p10, dpi):
            generated.append(p10)
            print(f"  Saved: {p10}")

    return tuple(generated)

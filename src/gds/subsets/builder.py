from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gds.common.io import ensure_dir, write_json
from gds.common.types import SubsetSpec


def generate_percentiles(min_percent: int, max_percent: int, step_percent: int) -> list[int]:
    if step_percent <= 0:
        raise ValueError("step_percent must be > 0")
    return list(range(min_percent, max_percent + 1, step_percent))


def build_subsets_from_ranking(
    ranking_df: pd.DataFrame,
    method: str,
    percentiles: list[int],
) -> list[SubsetSpec]:
    df = ranking_df.sort_values("rank", kind="stable").reset_index(drop=True)
    sample_ids = df["sample_id"].astype(int).tolist()
    n = len(sample_ids)
    subsets: list[SubsetSpec] = []
    for p in percentiles:
        remove_count = int((n * p) // 100)
        retained = sample_ids[remove_count:]
        subsets.append(SubsetSpec(method=method, percent_removed=p, retained_ids=retained))
    return subsets


def save_subsets(
    subsets: list[SubsetSpec],
    output_dir: Path,
) -> None:
    ensure_dir(output_dir)
    manifest: list[dict] = []
    for subset in subsets:
        pct = f"{subset.percent_removed:02d}"
        subset_path = output_dir / f"p{pct}.json"
        write_json(subset_path, asdict(subset))
        manifest.append({"percent_removed": subset.percent_removed, "path": str(subset_path)})
    write_json(output_dir / "manifest.json", {"subsets": manifest})


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.common.config import load_config
from gds.subsets.builder import build_subsets_from_ranking, generate_percentiles, save_subsets


def main() -> None:
    parser = argparse.ArgumentParser(description="Build retained subset IDs from ranking files.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    ranking_path = artifacts_dir / "rankings" / args.method / "scores.parquet"
    if not ranking_path.exists():
        raise FileNotFoundError(f"Ranking file not found: {ranking_path}")
    ranking_df = pd.read_parquet(ranking_path)

    subsets_cfg = cfg["subsets"]
    if "percentiles" in subsets_cfg:
        percentiles = [int(p) for p in subsets_cfg["percentiles"]]
    else:
        percentiles = generate_percentiles(
            min_percent=int(subsets_cfg["min_percent"]),
            max_percent=int(subsets_cfg["max_percent"]),
            step_percent=int(subsets_cfg["step_percent"]),
        )
    direction = str(subsets_cfg.get("direction", "remove_easy"))
    subsets = build_subsets_from_ranking(
        ranking_df=ranking_df,
        method=args.method,
        percentiles=percentiles,
        direction=direction,
    )
    output_dir = artifacts_dir / "subsets" / args.method
    save_subsets(subsets=subsets, output_dir=output_dir)

    # Verification: print subset stats so we can confirm correctness
    print(f"  Direction: {direction}")
    for s in subsets:
        retained = set(s.retained_ids)
        ret_df = ranking_df[ranking_df["sample_id"].isin(retained)]
        rem_df = ranking_df[~ranking_df["sample_id"].isin(retained)]
        ret_score = f"{ret_df['score'].mean():.3f}" if len(ret_df) > 0 else "N/A"
        rem_score = f"{rem_df['score'].mean():.3f}" if len(rem_df) > 0 else "N/A"
        print(
            f"  p{s.percent_removed:02d}: retained={len(retained):>6d}, "
            f"retained_mean_score={ret_score}, removed_mean_score={rem_score}"
        )
    print(f"Saved {len(subsets)} subsets in {output_dir}")


if __name__ == "__main__":
    main()


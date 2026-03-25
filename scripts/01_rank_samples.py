from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.common.config import load_config
from gds.common.io import ensure_dir
from gds.scoring.pipeline import run_ranking_pipeline, save_ranking_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank training samples by hardness.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    method = args.method

    dataset_cfg = cfg["dataset"]
    scoring_cfg = cfg["scoring"]

    dataset_name = dataset_cfg.get("name", "mnist")
    in_channels = int(dataset_cfg.get("in_channels", 1))
    is_text = dataset_cfg.get("type", "image") == "text"

    kwargs = dict(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        val_size=int(dataset_cfg.get("val_size", 10000)),
        split_seed=int(dataset_cfg["split_seed"]),
        method=method,
        random_seed=int(scoring_cfg["random_seed"]),
        batch_size=int(scoring_cfg["batch_size"]),
        num_workers=int(scoring_cfg["num_workers"]),
        scoring_model=str(scoring_cfg.get("model", "simple_cnn")),
        num_classes=int(dataset_cfg.get("num_classes", 10)),
        scoring_epochs=int(scoring_cfg.get("num_epochs", 10)),
        scoring_seeds=list(scoring_cfg.get("seeds", [2, 4, 8])),
        scoring_lr=float(scoring_cfg.get("lr", 0.01)),
        scoring_momentum=float(scoring_cfg.get("momentum", 0.9)),
        scoring_weight_decay=float(scoring_cfg.get("weight_decay", 5e-4)),
        scoring_nesterov=bool(scoring_cfg.get("nesterov", False)),
        scoring_scheduler=str(scoring_cfg.get("scheduler", "cosine")),
        scoring_milestones=[int(m) for m in scoring_cfg["milestones"]] if scoring_cfg.get("milestones") else None,
        scoring_gamma=float(scoring_cfg.get("gamma", 0.2)),
        dataset_name=dataset_name,
        in_channels=in_channels,
        augment=bool(dataset_cfg.get("augment", False)),
        is_text=is_text,
        block_size=int(dataset_cfg.get("block_size", 128)),
        val_fraction=float(dataset_cfg.get("val_fraction", 0.1)),
    )

    ranking_df = run_ranking_pipeline(**kwargs)

    output_dir = ensure_dir(artifacts_dir / "rankings" / method)
    metadata = {
        "method": method,
        "dataset": dataset_name,
        "num_samples": int(len(ranking_df)),
        "config_path": str(args.config),
    }
    parquet_path, _, _ = save_ranking_artifacts(
        ranking_df=ranking_df,
        output_dir=output_dir,
        metadata=metadata,
    )
    print(f"Saved ranking ({method}) to {parquet_path}")


if __name__ == "__main__":
    main()

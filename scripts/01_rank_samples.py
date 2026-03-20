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
    parser = argparse.ArgumentParser(description="Rank MNIST training samples by hardness.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    method = args.method

    ranking_df, pipeline_metadata = run_ranking_pipeline(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        val_size=int(cfg["dataset"]["val_size"]),
        split_seed=int(cfg["dataset"]["split_seed"]),
        method=method,
        random_seed=int(cfg["scoring"]["random_seed"]),
        model_names=list(cfg["scoring"]["pretrained_models"]),
        batch_size=int(cfg["scoring"]["batch_size"]),
        num_workers=int(cfg["scoring"]["num_workers"]),
        image_size=int(cfg["dataset"]["image_size"]),
        class_head_mode=str(cfg["scoring"].get("class_head_mode", "first10")),
    )

    output_dir = ensure_dir(artifacts_dir / "rankings" / method)
    metadata = {
        "method": method,
        "num_samples": int(len(ranking_df)),
        "config_path": str(args.config),
    }
    metadata.update(pipeline_metadata)
    parquet_path, _, _ = save_ranking_artifacts(
        ranking_df=ranking_df,
        output_dir=output_dir,
        metadata=metadata,
    )
    print(f"Saved ranking ({method}) to {parquet_path}")


if __name__ == "__main__":
    main()


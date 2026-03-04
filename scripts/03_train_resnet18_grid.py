from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gds.common.config import load_config
from gds.common.io import read_json, write_json
from gds.data.mnist import load_or_create_split
from gds.training.runner import run_training


def _parse_deterministic(value: object) -> bool | str:
    if isinstance(value, bool):
        return value
    if value is None:
        return "warn"
    parsed = str(value).strip().lower()
    if parsed in {"true", "1", "yes"}:
        return True
    if parsed in {"false", "0", "no"}:
        return False
    if parsed == "warn":
        return "warn"
    raise ValueError("training.deterministic must be one of: true, false, warn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train configured classifier on each retained subset.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    parser.add_argument("--method", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    dataset_cfg = cfg["dataset"]
    training_cfg = cfg["training"]

    artifacts_dir = Path(paths_cfg["artifacts_dir"])
    data_dir = Path(paths_cfg["data_dir"])
    method = args.method

    split = load_or_create_split(
        data_dir=data_dir,
        split_file=artifacts_dir / "splits" / f"mnist_split_seed{int(dataset_cfg['split_seed'])}.json",
        val_size=int(dataset_cfg["val_size"]),
        seed=int(dataset_cfg["split_seed"]),
    )

    subsets_dir = artifacts_dir / "subsets" / method
    subset_files = sorted(subsets_dir.glob("p*.json"))
    if not subset_files:
        raise FileNotFoundError(f"No subset files found in {subsets_dir}")

    run_summaries: list[dict] = []
    for subset_path in tqdm(subset_files, desc=f"Subsets ({method})"):
        subset_spec = read_json(subset_path)
        percent_removed = int(subset_spec["percent_removed"])
        train_ids = [int(x) for x in subset_spec["retained_ids"]]
        for seed in tqdm(
            training_cfg["seeds"],
            desc=f"Seeds p={percent_removed:02d}",
            leave=False,
        ):
            run_dir = artifacts_dir / "training" / method / f"p{percent_removed:02d}" / f"seed{int(seed)}"
            result = run_training(
                data_dir=data_dir,
                run_dir=run_dir,
                train_ids=train_ids,
                val_ids=split.val_ids,
                seed=int(seed),
                model_name=str(training_cfg.get("model", "resnet18")),
                max_epochs=int(training_cfg["max_epochs"]),
                patience=int(training_cfg["early_stopping_patience"]),
                batch_size=int(training_cfg["batch_size"]),
                num_workers=int(training_cfg["num_workers"]),
                image_size=int(dataset_cfg["image_size"]),
                lr=float(training_cfg["lr"]),
                weight_decay=float(training_cfg["weight_decay"]),
                accelerator=str(training_cfg["accelerator"]),
                devices=training_cfg["devices"],
                deterministic=_parse_deterministic(training_cfg.get("deterministic", "warn")),
                method=method,
                percent_removed=percent_removed,
            )
            summary_payload = result.__dict__.copy()
            write_json(run_dir / "run_summary.json", summary_payload)
            run_summaries.append(summary_payload)
            print(
                f"Completed method={method} p={percent_removed}% seed={seed}: "
                f"val_acc={result.best_val_acc:.4f} test_acc={result.test_acc:.4f}"
            )

    method_df = pd.DataFrame(run_summaries).sort_values(
        by=["percent_removed", "seed"], kind="stable"
    )
    method_summary_path = artifacts_dir / "training" / method / "runs.csv"
    method_df.to_csv(method_summary_path, index=False)
    print(f"Saved run table to {method_summary_path}")


if __name__ == "__main__":
    main()

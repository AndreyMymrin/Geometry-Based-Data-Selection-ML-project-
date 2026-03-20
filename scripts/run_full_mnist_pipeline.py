from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_step(args: list[str]) -> None:
    print(f"> {' '.join(args)}")
    subprocess.run(args, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full MNIST data-selection pipeline.")
    parser.add_argument("--config", type=str, default="configs/experiment/default.yaml")
    args = parser.parse_args()

    methods = ["error_rate_ensemble", "intrinsic_dimensionality_twonn", "random"]
    stage_plan: list[tuple[str, list[str]]] = []
    for method in methods:
        stage_plan.append(
            (
                f"{method}: rank",
                [sys.executable, "scripts/01_rank_samples.py", "--config", args.config, "--method", method],
            )
        )
        stage_plan.append(
            (
                f"{method}: subsets",
                [sys.executable, "scripts/02_build_subsets.py", "--config", args.config, "--method", method],
            )
        )
        stage_plan.append(
            (
                f"{method}: train",
                [sys.executable, "scripts/03_train_resnet18_grid.py", "--config", args.config, "--method", method],
            )
        )
    stage_plan.append(
        (
            "aggregate+plot",
            [sys.executable, "scripts/04_aggregate_and_plot.py", "--config", args.config],
        )
    )

    for stage_name, stage_cmd in tqdm(stage_plan, desc="Pipeline stages"):
        print(f"[stage] {stage_name}")
        run_step(stage_cmd)


if __name__ == "__main__":
    main()

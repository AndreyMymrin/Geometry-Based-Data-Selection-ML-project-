# Geometry-Based-Data-Selection-ML-project-

MNIST-first scaffold for data-selection experiments with sample ranking, subset construction, and classifier training.

## What is implemented

- `error_rate_ensemble` ranking using 6 pretrained torchvision models.
- `effective_rank` ranking using ensemble accuracy (complement of error rate).
- `random` ranking baseline.
- Percentile-based easiest-sample removal (`0..60`, step `5`).
- Lightning training on each subset and 3 fixed seeds (configurable model).
- Aggregation and plots:
  - `metric_vs_percentile_val_acc.png`
  - `metric_vs_percentile_test_acc.png`

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

## Run

Run all stages:

```bash
python scripts/run_full_mnist_pipeline.py --config configs/experiment/default.yaml
```

To switch the training model, set `training.model` in your config, e.g. `alexnet`.
Supported values: `resnet18`, `alexnet`, `mobilenet_v3_small`.
If you see CUDA deterministic errors (for example with AlexNet), set `training.deterministic: warn` (default) or `false`.

Run stage-by-stage:

```bash
python scripts/01_rank_samples.py --method error_rate_ensemble
python scripts/02_build_subsets.py --method error_rate_ensemble
python scripts/03_train_resnet18_grid.py --method error_rate_ensemble

python scripts/01_rank_samples.py --method effective_rank
python scripts/02_build_subsets.py --method effective_rank
python scripts/03_train_resnet18_grid.py --method effective_rank

python scripts/01_rank_samples.py --method random
python scripts/02_build_subsets.py --method random
python scripts/03_train_resnet18_grid.py --method random

python scripts/04_aggregate_and_plot.py
```

## Artifact layout

- `artifacts/rankings/{method}/scores.parquet`
- `artifacts/subsets/{method}/pXX.json`
- `artifacts/training/{method}/pXX/seedS/metrics.csv`
- `artifacts/training/{method}/pXX/seedS/run_summary.json`
- `artifacts/summary/curve_metrics.csv`
- `artifacts/plots/*.png`

## Tests

```bash
pytest -q
```

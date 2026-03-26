# Geometry-Based Data Selection

Data pruning via geometric and statistical scoring methods. Score every training sample by difficulty or redundancy, remove the easiest/most redundant samples, retrain, and measure the effect on model performance. Demonstrates that a large fraction of training data can be safely removed without hurting accuracy.

## Overview

The pipeline supports two experimental setups:

### a) Image Classification (MNIST, CIFAR-10)

**Scoring methods:**
- **Forgetting Events** (Toneva et al., 2018): count correct-to-incorrect transitions during training
- **Effective Rank** (Diff-eRank, Wei et al., NeurIPS 2024): leave-one-out contribution to covariance matrix rank
- **Intrinsic Dimensionality (TwoNN)**: estimate manifold dimension via nearest-neighbor distance ratios, then PCA + kNN density scoring
- **Random**: uniform random baseline

**Models:** SimpleCNN, ResNet-18

### b) Character-Level Language Modelling (TinyShakespeare)

**Scoring methods:**
- **Effective Rank**
- **Intrinsic Dimensionality (TwoNN)**
- **Perplexity Filtering**: ranking samples by per-sample perplexity under a trained LM
- **Semantic Deduplication**: removing redundant data via cosine similarity in embedding space
- **Heuristic Filtering**: basic data cleaning based on character composition (alphabetic ratio, diversity, whitespace/punctuation density)
- **LLM-Based Classifier**: using prediction entropy from a trained LM to classify and filter data
- **Random**

**Model:** nanoGPT (4-layer transformer, 128 embed dim)

### Supported Datasets

| Dataset | Type | Details |
|---------|------|---------|
| MNIST | Image | 1ch, 28x28, 10 classes, 50k train |
| CIFAR-10 | Image | 3ch, 32x32, 10 classes, 40k train |
| TinyShakespeare | Text | ~1M chars, character-level, chunked into 128-token sequences |

## Environment

Requires Python >= 3.10.

```bash
pip install -e .
```

### Dependencies

Core: `torch`, `torchvision`, `lightning`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `PyYAML`, `pyarrow`, `tqdm`.

## Quick Start

### Run the full pipeline (MNIST, default config)

```bash
python scripts/run_full_mnist_pipeline.py
```

### Run on CIFAR-10

```bash
python scripts/run_full_mnist_pipeline.py --config configs/experiment/cifar10.yaml
```

### Run on TinyShakespeare

```bash
python scripts/run_full_mnist_pipeline.py --config configs/experiment/tiny_shakespeare.yaml
```

## Pipeline Stages

The pipeline is composed of four stages, orchestrated by `run_full_mnist_pipeline.py`. Each stage can also be run independently.

### Stage 1: Rank Samples (`01_rank_samples.py`)

Scores every training sample using the specified method. For forgetting events, trains multiple models and counts transitions. For effective rank and intrinsic dimensionality, trains a model, extracts embeddings, and computes geometric scores. For text datasets, uses nanoGPT for embedding extraction.

```bash
python scripts/01_rank_samples.py --method <method_name> --config configs/experiment/<dataset>.yaml
```

**Output:** `artifacts/rankings/{method}/scores.parquet`

### Stage 2: Build Subsets (`02_build_subsets.py`)

Removes the easiest samples at each data-budget level. Default retention grid: **100%, 70%, 30%, 10%** of training data (i.e., removes 0%, 30%, 70%, 90%).

```bash
python scripts/02_build_subsets.py --method forgetting_events
```

**Output:** `artifacts/subsets/{method}/p00.json` through `p60.json`

### Stage 3: Train on Subsets (`03_train_resnet18_grid.py`)

For each subset x seed combination, trains a classifier with early stopping. For image datasets, monitors validation accuracy. For text datasets, monitors validation loss.

```bash
python scripts/03_train_resnet18_grid.py --method forgetting_events
```

**Output:** `artifacts/training/{method}/pXX/seedS/run_summary.json`

### Stage 4: Aggregate and Plot (`04_aggregate_and_plot.py`)

Collects all run summaries, computes mean +/- std across seeds, and generates plots. Charts & analysis can be viewed in `artifacts/` directory.

```bash
python scripts/04_aggregate_and_plot.py
```


## Configuration


### Text Dataset Config Example

```yaml
dataset:
  name: tiny_shakespeare
  type: text
  block_size: 128
  val_fraction: 0.1
  split_seed: 42

scoring:
  methods:
    - effective_rank
    - intrinsic_dimensionality_twonn
    - perplexity_filtering
    - semantic_dedup
    - heuristic_filtering
    - llm_classifier
    - random

subsets:
  percentiles: [0, 30, 70, 90]    # Retain 100%, 70%, 30%, 10%

training:
  model: nano_gpt
  seeds: [101, 202, 303]
  max_epochs: 20
```


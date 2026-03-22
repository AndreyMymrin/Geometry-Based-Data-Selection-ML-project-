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

## Installation

Requires Python >= 3.10.

```bash
pip install -e .

# With dev dependencies (pytest)
pip install -e ".[dev]"
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
python scripts/01_rank_samples.py --method forgetting_events --config configs/experiment/default.yaml
python scripts/01_rank_samples.py --method effective_rank --config configs/experiment/default.yaml
python scripts/01_rank_samples.py --method intrinsic_dimensionality_twonn --config configs/experiment/default.yaml
python scripts/01_rank_samples.py --method random --config configs/experiment/default.yaml
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

Collects all run summaries, computes mean +/- std across seeds, and generates plots.

```bash
python scripts/04_aggregate_and_plot.py
```

**Output (charts & analysis):**

| File | Description |
|------|-------------|
| `retention_curve.png` | Primary metric vs data budget (all methods, with std bands) |
| `scaling_law.png` | Log-scale scaling-law plot with power-law fits (metric = a·N^b + c) |
| `relative_performance.png` | Bar chart: performance at each budget relative to 100% baseline |
| `method_heatmap.png` | Heatmap: methods × retention budgets |
| `data_efficiency.png` | Scatter: data saved vs performance drop (ideal = bottom-right) |
| `retention_curve_val_acc.png` | Validation accuracy vs data budget (image datasets) |
| `conclusions.json` | Quantitative analysis: scaling-law fits, safe budgets, best method per budget |
| `conclusions.txt` | Human-readable summary of all quantitative conclusions |
| `scaling_law_fits.json` | Power-law coefficients and R² per method |
| `tsne_embeddings.png` | t-SNE visualization of embeddings colored by difficulty (image datasets) |

## Scoring Methods

### Forgetting Events (Image only)

From Toneva et al. (2018, arXiv:1812.05159). Trains multiple models with different seeds and counts per-sample forgetting events (correct-to-incorrect transitions). Averages across seeds. Samples with score = 0 are "unforgettable" and removed first.

### Effective Rank (Diff-eRank)

From Wei et al. (NeurIPS 2024). Extracts embeddings from a trained model, computes the covariance matrix, then calculates each sample's leave-one-out contribution to the effective rank (Shannon entropy of normalized singular values). Higher contribution = more informative sample.

### Intrinsic Dimensionality (TwoNN)

Estimates the intrinsic dimension of the feature manifold using the TwoNN estimator (ratio of 2nd to 1st nearest-neighbor distances). Reduces features to estimated dimension via PCA, then scores samples by kNN density. Lower density = harder, more unique samples.

### Perplexity Filtering (Text only)

Trains a nanoGPT language model, then computes per-sample perplexity (exponential of mean cross-entropy loss). Low perplexity = easy/predictable text, high perplexity = hard/noisy text. Samples with the lowest perplexity are removed first during pruning.

### Semantic Deduplication (Text only)

Extracts embeddings from a trained nanoGPT, then scores each sample by its uniqueness in embedding space. For each sample, computes the maximum cosine similarity to any other sample. Score = 1 - max_similarity: more redundant samples (high similarity to neighbours) get lower scores and are removed first.

### Heuristic Filtering (Text only)

Scores text chunks by character-level quality features without training a model:
- **Alphabetic ratio**: fraction of letters (higher = more natural text)
- **Character diversity**: unique chars / total chars
- **Whitespace ratio**: fraction of whitespace (penalised if too high)
- **Punctuation ratio**: fraction of punctuation (penalised if too high)

Combined into a weighted quality score. Lower quality chunks are removed first.

### LLM-Based Classifier (Text only)

Trains a nanoGPT language model, then computes per-sample mean prediction entropy (Shannon entropy of the softmax distribution over the vocabulary). Low entropy = model is confident about predictions = easy/clean data. High entropy = model is uncertain = hard/noisy data. Samples with the lowest entropy are removed first.

### Random

Uniform random scores as a baseline.

## Models

### SimpleCNN (default for images)

Lightweight CNN for small images (28x28, 32x32):

```
Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool
Conv2d -> BN -> ReLU -> Conv2d -> BN -> ReLU -> MaxPool
AdaptiveAvgPool -> FC(128, 256) -> ReLU -> FC(256, num_classes)
```

256-dim penultimate layer used for embedding extraction.

### ResNet-18

Standard architecture, adapted for single-channel input when needed.

### nanoGPT (for text)

Minimal GPT for character-level language modelling:

```
Token Embedding + Positional Embedding
4x TransformerBlock (LayerNorm -> CausalSelfAttention -> LayerNorm -> MLP)
LayerNorm -> Linear Head
```

128-dim mean-pooled embeddings used for feature extraction.

## Project Structure

```
.
├── configs/
│   ├── experiment/
│   │   ├── default.yaml              # MNIST experiment config
│   │   ├── cifar10.yaml              # CIFAR-10 config
│   │   └── tiny_shakespeare.yaml     # TinyShakespeare config
│   ├── dataset/
│   │   ├── mnist.yaml
│   │   ├── cifar10.yaml
│   │   └── tiny_shakespeare.yaml
│   ├── scoring/
│   │   ├── forgetting_events.yaml
│   │   ├── effective_rank.yaml
│   │   ├── intrinsic_dimensionality_twonn.yaml
│   │   └── random.yaml
│   └── training/
│       ├── simple_cnn.yaml
│       └── resnet18.yaml
├── scripts/
│   ├── run_full_mnist_pipeline.py        # Orchestrates all 4 stages
│   ├── 01_rank_samples.py               # Stage 1: Score & rank samples
│   ├── 02_build_subsets.py               # Stage 2: Build retained subsets
│   ├── 03_train_resnet18_grid.py         # Stage 3: Train on each subset
│   └── 04_aggregate_and_plot.py          # Stage 4: Aggregate & plot
├── src/gds/
│   ├── common/
│   │   ├── config.py                     # YAML config loading
│   │   ├── io.py                         # File I/O utilities
│   │   ├── seed.py                       # Reproducibility
│   │   └── types.py                      # Dataclasses
│   ├── data/
│   │   ├── datasets.py                   # Generic dataset registry (MNIST, CIFAR-10)
│   │   └── tiny_shakespeare.py           # TinyShakespeare dataset + vocab
│   ├── models/
│   │   ├── simple_cnn.py                 # Lightweight CNN
│   │   └── nano_gpt.py                   # nanoGPT transformer
│   ├── scoring/
│   │   ├── base.py                       # SampleScorer ABC
│   │   ├── forgetting.py                 # Forgetting-event scorer
│   │   ├── effective_rank.py             # Diff-eRank scorer
│   │   ├── intrinsic_dimensionality.py   # TwoNN scorer
│   │   ├── perplexity.py                 # Perplexity filtering (text only)
│   │   ├── semantic_dedup.py             # Semantic deduplication (text only)
│   │   ├── heuristic_filter.py           # Heuristic text filtering (text only)
│   │   ├── llm_classifier.py             # LLM-based classifier (text only)
│   │   ├── random_scorer.py              # Random baseline
│   │   ├── registry.py                   # Method name -> scorer mapping
│   │   ├── pipeline.py                   # End-to-end ranking pipeline
│   │   └── utils.py                      # Stable ranking utility
│   ├── training/
│   │   ├── lightning_module.py           # ResNet18Classifier + NanoGPTLightning
│   │   ├── datamodule.py                 # Generic + Text DataModules
│   │   └── runner.py                     # Training loop & metric collection
│   ├── subsets/
│   │   └── builder.py                    # Percentile-based subset generation
│   └── analysis/
│       ├── aggregate.py                  # Aggregation, accuracy curves, paper-style plot
│       └── tsne_viz.py                   # t-SNE embedding visualization
├── tests/
│   ├── test_effective_rank_score.py
│   └── test_intrinsic_dimensionality_scorer.py
├── pyproject.toml
└── README.md
```

## Configuration

All experiment parameters are set via YAML configs.

### Image Dataset Config Example

```yaml
dataset:
  name: cifar10
  split_seed: 42
  val_size: 10000
  num_classes: 10
  in_channels: 3
  image_size: 32

scoring:
  methods:
    - forgetting_events
    - effective_rank
    - intrinsic_dimensionality_twonn
    - random
  model: simple_cnn
  num_epochs: 10
  seeds: [42, 123, 456, 789, 1024]

subsets:
  percentiles: [0, 30, 70, 90]    # Remove 0%, 30%, 70%, 90% → retain 100%, 70%, 30%, 10%

training:
  model: simple_cnn            # simple_cnn | resnet18
  seeds: [101, 202, 303]
  max_epochs: 15
```

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

## Reproducibility

| Seed | Config Key | Purpose |
|------|-----------|---------|
| Split seed | `dataset.split_seed` | Train/val partition |
| Scoring seeds | `scoring.seeds` | Forgetting-event ensemble (averaged) |
| Training seeds | `training.seeds` | Final model training (mean +/- std) |

All seeds are set via `seed_everything()` which controls Python, NumPy, PyTorch (CPU + CUDA + cuDNN).

## Device Support

- **CPU**: Always works.
- **CUDA**: Detected automatically via `accelerator: auto`.

## Tests

```bash
pytest -q
```

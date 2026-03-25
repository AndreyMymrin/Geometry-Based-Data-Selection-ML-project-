"""Scoring pipeline: rank training samples by various methods.

Geometric methods (effective_rank, corr_integral) use hidden states
from pretrained models:
  - Images: pretrained ResNet-18 (ImageNet weights)
  - Text:   pretrained Qwen2-0.5B

Feature-based methods (intrinsic_dimensionality, semantic_dedup) use
flat features from the last layer of the same pretrained models.

Following Yusupov et al. (2025), "From Internal Representations to Text
Quality: A Geometric Approach to LLM Evaluation."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

from gds.common.io import ensure_dir, write_json
from gds.scoring.forgetting import run_forgetting_ensemble
from gds.scoring.registry import (
    get_scorer, is_feature_method, is_forgetting_method,
    is_hidden_state_method, is_text_heuristic_method, is_text_model_method,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ------------------------------------------------------------------
# Text model scoring (perplexity / entropy) — requires training
# ------------------------------------------------------------------

def _train_and_compute_text_model_scores(
    loader,
    vocab_size: int,
    block_size: int,
    scoring_epochs: int,
    scoring_lr: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Train a nanoGPT and compute per-sample perplexity and entropy."""
    from gds.models.nano_gpt import NanoGPT

    model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scoring_lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    print(f"  Training nanoGPT for {scoring_epochs} epochs (perplexity + entropy scoring)...")
    model.train()
    for epoch in range(scoring_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"    Epoch {epoch + 1}/{scoring_epochs}  loss={epoch_loss / max(n_batches, 1):.4f}")

    print("  Computing per-sample perplexity and entropy...")
    model.eval()
    all_perplexity, all_entropy, all_labels, all_sids = [], [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, V = logits.shape
            token_losses = criterion(logits.view(-1, V), y.view(-1)).view(B, T)
            sample_loss = token_losses.mean(dim=1)
            perplexity = torch.exp(sample_loss)
            all_perplexity.append(perplexity.cpu().numpy())
            probs = torch.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-10)
            token_entropy = -(probs * log_probs).sum(dim=-1)
            sample_entropy = token_entropy.mean(dim=1)
            all_entropy.append(sample_entropy.cpu().numpy())
            all_labels.extend(y[:, 0].cpu().numpy().astype(int).tolist())
            if hasattr(sid, "numpy"):
                all_sids.extend(sid.numpy().astype(int).tolist())
            else:
                all_sids.extend([int(s) for s in sid])

    return (
        np.concatenate(all_perplexity).astype(np.float32),
        np.concatenate(all_entropy).astype(np.float32),
        all_labels,
        all_sids,
    )


# ------------------------------------------------------------------
# Dataset loading helpers
# ------------------------------------------------------------------

def _build_image_loader(
    data_dir, artifacts_dir, dataset_name, val_size, split_seed,
    batch_size, num_workers, in_channels, shuffle=True, augment=False,
):
    from gds.data.datasets import (
        build_indexed_dataset, build_loader, get_dataset_info,
        load_or_create_split, make_train_transform,
    )
    if dataset_name is None:
        dataset_name = "mnist"
    info = get_dataset_info(dataset_name)
    if in_channels is None:
        in_channels = info["in_channels"]
    split_file = artifacts_dir / "splits" / f"{dataset_name}_split_seed{split_seed}.json"
    split = load_or_create_split(data_dir, split_file, dataset_name, val_size, split_seed)
    transform = make_train_transform(dataset_name, augment=augment)
    ds = build_indexed_dataset(data_dir, dataset_name, True, transform, split.train_ids)
    loader = build_loader(ds, batch_size, num_workers, shuffle)
    return loader, split, in_channels


def _build_text_loader(
    data_dir, artifacts_dir, block_size, val_fraction,
    split_seed, batch_size, num_workers, shuffle=True,
):
    from gds.data.tiny_shakespeare import (
        TinyShakespeareDataset, build_text_loader,
        load_or_create_text_split,
    )
    split_file = artifacts_dir / "splits" / f"tiny_shakespeare_split_seed{split_seed}.json"
    split, chunks, stoi, itos = load_or_create_text_split(
        data_dir, split_file, block_size, val_fraction, split_seed,
    )
    ds = TinyShakespeareDataset(chunks, split.train_ids)
    loader = build_text_loader(ds, batch_size, num_workers, shuffle)
    return loader, split, len(stoi), chunks, stoi, itos


def _collect_text_ids(loader) -> tuple[list[int], list[int]]:
    """Iterate loader to collect labels and sample IDs (no features)."""
    labels, sids = [], []
    for _, y, sid in tqdm(loader, desc="  Collecting IDs"):
        labels.extend(y[:, 0].numpy().astype(int).tolist())
        if hasattr(sid, "numpy"):
            sids.extend(sid.numpy().astype(int).tolist())
        else:
            sids.extend([int(s) for s in sid])
    return labels, sids


def _collect_image_ids(loader) -> tuple[list[int], list[int]]:
    """Iterate loader to collect labels and sample IDs (no features)."""
    labels, sids = [], []
    for _, y, sid in tqdm(loader, desc="  Collecting IDs"):
        labels.extend(y.numpy().astype(int).tolist())
        sids.extend(sid.numpy().astype(int).tolist())
    return labels, sids


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------

def run_ranking_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    val_size: int,
    split_seed: int,
    method: str,
    random_seed: int,
    batch_size: int,
    num_workers: int,
    scoring_model: str = "simple_cnn",
    num_classes: int = 10,
    scoring_epochs: int = 10,
    scoring_seeds: list[int] | None = None,
    scoring_lr: float = 0.01,
    scoring_momentum: float = 0.9,
    scoring_weight_decay: float = 5e-4,
    scoring_nesterov: bool = False,
    scoring_scheduler: str = "cosine",
    scoring_milestones: list[int] | None = None,
    scoring_gamma: float = 0.2,
    dataset_name: str | None = None,
    in_channels: int | None = None,
    augment: bool = False,
    # Text dataset support
    is_text: bool = False,
    block_size: int = 256,
    val_fraction: float = 0.1,
    vocab_size: int | None = None,
) -> pd.DataFrame:
    device = _get_device()
    scorer = get_scorer(method=method, random_seed=random_seed)
    metadata = None
    labels: list[int]
    sample_ids: list[int]

    # ------------------------------------------------------------------
    # TEXT path
    # ------------------------------------------------------------------
    if is_text:
        loader, split, vs, chunks, stoi, itos = _build_text_loader(
            data_dir, artifacts_dir, block_size, val_fraction,
            split_seed, batch_size, num_workers, shuffle=True,
        )
        if vocab_size is None:
            vocab_size = vs

        if is_hidden_state_method(method):
            # ERank / CorrInt: pretrained Qwen2-0.5B hidden states
            from gds.scoring.pretrained_features import (
                extract_text_hidden_states, set_itos,
            )
            set_itos(itos)
            hidden_states, labels, sample_ids = extract_text_hidden_states(
                loader, device,
            )
            metadata = {"hidden_states": hidden_states}

        elif is_feature_method(method):
            # Intrinsic dim / Semantic dedup: pretrained Qwen2-0.5B flat features
            from gds.scoring.pretrained_features import (
                extract_text_features, set_itos,
            )
            set_itos(itos)
            features, labels, sample_ids = extract_text_features(
                loader, device,
            )
            metadata = {"features": features}

        elif is_text_model_method(method):
            ppl, ent, labels, sample_ids = _train_and_compute_text_model_scores(
                loader, vocab_size, block_size,
                scoring_epochs, scoring_lr, device,
            )
            metadata = {"perplexity": ppl, "entropy": ent}

        elif is_text_heuristic_method(method):
            labels, sample_ids = _collect_text_ids(loader)
            metadata = {"chunks": chunks, "itos": itos}

        else:
            # random baseline
            labels, sample_ids = _collect_text_ids(loader)

        ranking_df = scorer.score(sample_ids=sample_ids, labels=labels, metadata=metadata)
        return ranking_df

    # ------------------------------------------------------------------
    # IMAGE path
    # ------------------------------------------------------------------
    loader, split, in_channels = _build_image_loader(
        data_dir, artifacts_dir, dataset_name, val_size, split_seed,
        batch_size, num_workers, in_channels,
        augment=augment if is_forgetting_method(method) else False,
    )

    if is_forgetting_method(method):
        if scoring_seeds is None:
            scoring_seeds = [2, 4, 8]
        forgetting_counts, labels_np, sids_np = run_forgetting_ensemble(
            loader=loader,
            model_name=scoring_model,
            num_classes=num_classes,
            num_epochs=scoring_epochs,
            seeds=scoring_seeds,
            lr=scoring_lr,
            momentum=scoring_momentum,
            weight_decay=scoring_weight_decay,
            device=device,
            show_progress=True,
            in_channels=in_channels,
            nesterov=scoring_nesterov,
            scheduler_name=scoring_scheduler,
            milestones=scoring_milestones,
            gamma=scoring_gamma,
        )
        metadata = {"forgetting_scores": forgetting_counts}
        labels = labels_np.astype(int).tolist()
        sample_ids = sids_np.astype(int).tolist()

    elif is_hidden_state_method(method):
        # ERank / CorrInt: pretrained ResNet-18 hidden states
        from gds.scoring.pretrained_features import extract_image_hidden_states
        hidden_states, labels, sample_ids = extract_image_hidden_states(
            loader, device,
        )
        metadata = {"hidden_states": hidden_states}

    elif is_feature_method(method):
        # Intrinsic dim / Semantic dedup: pretrained ResNet-18 flat features
        from gds.scoring.pretrained_features import extract_image_features
        features, labels, sample_ids = extract_image_features(loader, device)
        metadata = {"features": features}

    else:
        # random
        labels, sample_ids = _collect_image_ids(loader)

    ranking_df = scorer.score(sample_ids=sample_ids, labels=labels, metadata=metadata)
    return ranking_df


def save_ranking_artifacts(
    ranking_df: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> tuple[Path, Path, Path]:
    ensure_dir(output_dir)
    parquet_path = output_dir / "scores.parquet"
    csv_path = output_dir / "scores.csv"
    metadata_path = output_dir / "metadata.json"
    ranking_df.to_parquet(parquet_path, index=False)
    ranking_df.to_csv(csv_path, index=False)
    write_json(metadata_path, metadata)
    return parquet_path, csv_path, metadata_path

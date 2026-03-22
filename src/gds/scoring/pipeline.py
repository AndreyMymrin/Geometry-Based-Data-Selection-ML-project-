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
    is_text_heuristic_method, is_text_model_method,
)


# ------------------------------------------------------------------
# Helpers: train a model and extract features (images or text)
# ------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _train_and_extract_image_features(
    loader,
    scoring_model: str,
    num_classes: int,
    in_channels: int,
    scoring_epochs: int,
    scoring_lr: float,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Train a SimpleCNN/ResNet on image data and extract embeddings."""
    from gds.scoring.forgetting import _build_classifier

    model = _build_classifier(scoring_model, num_classes, in_channels)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scoring_lr)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training {scoring_model} for {scoring_epochs} epochs to extract features...")
    model.train()
    for epoch in range(scoring_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"    Epoch {epoch + 1}/{scoring_epochs}  loss={epoch_loss / max(n_batches, 1):.4f}")

    # Extract features
    print("  Extracting embeddings...")
    model.eval()
    all_features, all_labels, all_sids = [], [], []
    with torch.no_grad():
        for x, y, sid in tqdm(loader, desc="  Feature extraction"):
            x = x.to(device)
            if hasattr(model, "get_embeddings"):
                feat = model.get_embeddings(x)
            else:
                feat = x.view(x.size(0), -1)
            all_features.append(feat.cpu().numpy())
            all_labels.extend(y.numpy().astype(int).tolist())
            all_sids.extend(sid.numpy().astype(int).tolist())

    features = np.concatenate(all_features, axis=0)
    print(f"  Features shape: {features.shape}")
    return features, all_labels, all_sids


def _train_and_extract_text_features(
    loader,
    vocab_size: int,
    block_size: int,
    scoring_epochs: int,
    scoring_lr: float,
    device: torch.device,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Train a nanoGPT on text data and extract embeddings."""
    from gds.models.nano_gpt import NanoGPT

    model = NanoGPT(vocab_size=vocab_size, block_size=block_size)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scoring_lr)
    criterion = nn.CrossEntropyLoss()

    print(f"  Training nanoGPT for {scoring_epochs} epochs to extract features...")
    model.train()
    for epoch in range(scoring_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        print(f"    Epoch {epoch + 1}/{scoring_epochs}  loss={epoch_loss / max(n_batches, 1):.4f}")

    # Extract embeddings
    print("  Extracting embeddings...")
    model.eval()
    all_features, all_labels, all_sids = [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            x = x.to(device)
            emb = model.get_embeddings(x)
            all_features.append(emb.cpu().numpy())
            # For text, use first target token as "label" (class proxy)
            all_labels.extend(y[:, 0].numpy().astype(int).tolist())
            if hasattr(sid, "numpy"):
                all_sids.extend(sid.numpy().astype(int).tolist())
            else:
                all_sids.extend([int(s) for s in sid])

    features = np.concatenate(all_features, axis=0)
    return features, all_labels, all_sids


def _train_and_compute_text_model_scores(
    loader,
    vocab_size: int,
    block_size: int,
    scoring_epochs: int,
    scoring_lr: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Train a nanoGPT and compute per-sample perplexity and entropy.

    Returns (perplexity, entropy, labels, sample_ids).
    """
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

    # Compute per-sample perplexity and entropy
    print("  Computing per-sample perplexity and entropy...")
    model.eval()
    all_perplexity, all_entropy, all_labels, all_sids = [], [], [], []
    with torch.no_grad():
        for x, y, sid in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)  # (B, T, V)

            # Per-token cross-entropy loss -> per-sample mean -> perplexity
            B, T, V = logits.shape
            token_losses = criterion(logits.view(-1, V), y.view(-1)).view(B, T)
            sample_loss = token_losses.mean(dim=1)  # (B,)
            perplexity = torch.exp(sample_loss)
            all_perplexity.append(perplexity.cpu().numpy())

            # Per-token prediction entropy -> per-sample mean
            probs = torch.softmax(logits, dim=-1)  # (B, T, V)
            log_probs = torch.log(probs + 1e-10)
            token_entropy = -(probs * log_probs).sum(dim=-1)  # (B, T)
            sample_entropy = token_entropy.mean(dim=1)  # (B,)
            all_entropy.append(sample_entropy.cpu().numpy())

            all_labels.extend(y[:, 0].cpu().numpy().astype(int).tolist())
            if hasattr(sid, "numpy"):
                all_sids.extend(sid.numpy().astype(int).tolist())
            else:
                all_sids.extend([int(s) for s in sid])

    perplexity_arr = np.concatenate(all_perplexity, axis=0).astype(np.float32)
    entropy_arr = np.concatenate(all_entropy, axis=0).astype(np.float32)
    return perplexity_arr, entropy_arr, all_labels, all_sids


# ------------------------------------------------------------------
# Dataset loading helpers
# ------------------------------------------------------------------

def _build_image_loader(
    data_dir, artifacts_dir, dataset_name, val_size, split_seed,
    batch_size, num_workers, in_channels, shuffle=True,
):
    """Build a data loader for an image dataset. Returns (loader, split, in_channels)."""
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
    transform = make_train_transform(dataset_name)
    ds = build_indexed_dataset(data_dir, dataset_name, True, transform, split.train_ids)
    loader = build_loader(ds, batch_size, num_workers, shuffle)
    return loader, split, in_channels


def _build_text_loader(
    data_dir, artifacts_dir, block_size, val_fraction,
    split_seed, batch_size, num_workers, shuffle=True,
):
    """Build a data loader for TinyShakespeare. Returns (loader, split, vocab_size, chunks, stoi, itos)."""
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
    dataset_name: str | None = None,
    in_channels: int | None = None,
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

        if is_feature_method(method):
            features, labels, sample_ids = _train_and_extract_text_features(
                loader, vocab_size, block_size,
                scoring_epochs, scoring_lr, device,
            )
            metadata = {"features": features}

        elif is_text_model_method(method):
            # perplexity_filtering or llm_classifier
            ppl, ent, labels, sample_ids = _train_and_compute_text_model_scores(
                loader, vocab_size, block_size,
                scoring_epochs, scoring_lr, device,
            )
            metadata = {"perplexity": ppl, "entropy": ent}

        elif is_text_heuristic_method(method):
            # heuristic_filtering — works on raw chunks, no model needed
            labels = []
            sample_ids = []
            for _, y, sid in tqdm(loader, desc="Heuristic scoring batches"):
                labels.extend(y[:, 0].numpy().astype(int).tolist())
                if hasattr(sid, "numpy"):
                    sample_ids.extend(sid.numpy().astype(int).tolist())
                else:
                    sample_ids.extend([int(s) for s in sid])
            metadata = {"chunks": chunks, "itos": itos}

        else:
            # random baseline
            labels = []
            sample_ids = []
            for _, y, sid in tqdm(loader, desc="Text ranking batches"):
                labels.extend(y[:, 0].numpy().astype(int).tolist())
                if hasattr(sid, "numpy"):
                    sample_ids.extend(sid.numpy().astype(int).tolist())
                else:
                    sample_ids.extend([int(s) for s in sid])

        ranking_df = scorer.score(sample_ids=sample_ids, labels=labels, metadata=metadata)
        return ranking_df

    # ------------------------------------------------------------------
    # IMAGE path
    # ------------------------------------------------------------------
    loader, split, in_channels = _build_image_loader(
        data_dir, artifacts_dir, dataset_name, val_size, split_seed,
        batch_size, num_workers, in_channels,
    )

    if is_forgetting_method(method):
        if scoring_seeds is None:
            scoring_seeds = [42, 123, 456, 789, 1024]
        scores, labels_np, sids_np = run_forgetting_ensemble(
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
        )
        metadata = {"forgetting_scores": scores}
        labels = labels_np.astype(int).tolist()
        sample_ids = sids_np.astype(int).tolist()

    elif is_feature_method(method):
        features, labels, sample_ids = _train_and_extract_image_features(
            loader, scoring_model, num_classes, in_channels,
            scoring_epochs, scoring_lr, device,
        )
        metadata = {"features": features}

    else:
        # random
        labels = []
        sample_ids = []
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        for _, y, sid in tqdm(loader, desc="Random ranking batches", total=total_batches):
            labels.extend(y.numpy().astype(int).tolist())
            sample_ids.extend(sid.numpy().astype(int).tolist())

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

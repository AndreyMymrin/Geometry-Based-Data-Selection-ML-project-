from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torchvision import transforms
from tqdm.auto import tqdm

from gds.common.io import ensure_dir, write_json
from gds.data.mnist import (
    build_loader,
    build_mnist_indexed_dataset,
    load_or_create_split,
    make_imagenet_eval_transform,
)
from gds.models.pretrained import run_pretrained_predictions
from gds.scoring.registry import get_scorer


def run_ranking_pipeline(
    data_dir: Path,
    artifacts_dir: Path,
    val_size: int,
    split_seed: int,
    method: str,
    random_seed: int,
    model_names: list[str],
    batch_size: int,
    num_workers: int,
    image_size: int,
    class_head_mode: str = "first10",
) -> tuple[pd.DataFrame, dict]:
    split_file = artifacts_dir / "splits" / f"mnist_split_seed{split_seed}.json"
    split = load_or_create_split(
        data_dir=data_dir,
        split_file=split_file,
        val_size=val_size,
        seed=split_seed,
    )

    scorer = get_scorer(method=method, random_seed=random_seed)
    metadata = None
    pipeline_metadata: dict = {}
    labels: list[int]
    sample_ids: list[int]
    if method == "error_rate_ensemble":
        ranking_dataset = build_mnist_indexed_dataset(
            data_dir=data_dir,
            train=True,
            transform=make_imagenet_eval_transform(image_size=image_size),
            indices=split.train_ids,
        )
        ranking_loader = build_loader(
            dataset=ranking_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictions, labels_np, sample_ids_np = run_pretrained_predictions(
            model_names=model_names,
            loader=ranking_loader,
            device=device,
            class_head_mode=class_head_mode,
            show_progress=True,
        )
        metadata = {"predictions": predictions}
        labels = labels_np.astype(int).tolist()
        sample_ids = sample_ids_np.astype(int).tolist()
    elif method == "intrinsic_dimensionality_twonn":
        feature_dataset = build_mnist_indexed_dataset(
            data_dir=data_dir,
            train=True,
            transform=transforms.ToTensor(),
            indices=split.train_ids,
        )
        feature_loader = build_loader(
            dataset=feature_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        labels = []
        sample_ids = []
        feature_batches: list[torch.Tensor] = []
        total_batches = len(feature_loader) if hasattr(feature_loader, "__len__") else None
        for x, y, sid in tqdm(
            feature_loader,
            desc="Intrinsic-dimension feature batches",
            total=total_batches,
        ):
            feature_batches.append(x.view(x.shape[0], -1).cpu())
            labels.extend(y.numpy().astype(int).tolist())
            sample_ids.extend(sid.numpy().astype(int).tolist())

        metadata = {"features": torch.cat(feature_batches, dim=0).numpy()}
    else:
        ranking_dataset = build_mnist_indexed_dataset(
            data_dir=data_dir,
            train=True,
            transform=make_imagenet_eval_transform(image_size=image_size),
            indices=split.train_ids,
        )
        ranking_loader = build_loader(
            dataset=ranking_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )
        labels = []
        sample_ids = []
        total_batches = len(ranking_loader) if hasattr(ranking_loader, "__len__") else None
        for _, y, sid in tqdm(
            ranking_loader,
            desc="Random ranking batches",
            total=total_batches,
        ):
            labels.extend(y.numpy().astype(int).tolist())
            sample_ids.extend(sid.numpy().astype(int).tolist())

    ranking_df = scorer.score(sample_ids=sample_ids, labels=labels, metadata=metadata)
    if hasattr(scorer, "build_metadata"):
        pipeline_metadata = scorer.build_metadata()
    return ranking_df, pipeline_metadata


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

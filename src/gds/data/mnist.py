from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from gds.common.io import ensure_dir, read_json, write_json

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class MnistSplit:
    train_ids: list[int]
    val_ids: list[int]
    split_seed: int


def make_imagenet_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(image_size, padding=8),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transform(image_size: int = 224) -> transforms.Compose:
    return make_imagenet_eval_transform(image_size=image_size)


def stratified_train_val_split(
    labels: Sequence[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    labels_np = np.asarray(labels)
    n = len(labels_np)
    if val_size <= 0 or val_size >= n:
        raise ValueError(f"val_size must be in [1, {n-1}], got {val_size}")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    unique_labels = sorted(np.unique(labels_np).tolist())
    for label in unique_labels:
        class_indices = np.where(labels_np == label)[0]
        class_indices = class_indices[rng.permutation(len(class_indices))]

        raw_val_count = (len(class_indices) * val_size) / n
        class_val_count = int(round(raw_val_count))
        class_val_count = max(1, min(class_val_count, len(class_indices) - 1))

        val_indices.extend(class_indices[:class_val_count].tolist())
        train_indices.extend(class_indices[class_val_count:].tolist())

    val_indices = sorted(val_indices)
    train_indices = sorted(train_indices)

    overflow = len(val_indices) - val_size
    if overflow > 0:
        train_indices.extend(val_indices[-overflow:])
        val_indices = val_indices[:-overflow]
    elif overflow < 0:
        missing = -overflow
        val_indices.extend(train_indices[-missing:])
        train_indices = train_indices[:-missing]

    val_indices = sorted(val_indices)
    train_indices = sorted(train_indices)

    if len(val_indices) != val_size:
        raise RuntimeError("Failed to construct requested validation size deterministically.")

    return train_indices, val_indices


def load_or_create_split(
    data_dir: Path,
    split_file: Path,
    val_size: int,
    seed: int,
) -> MnistSplit:
    if split_file.exists():
        payload = read_json(split_file)
        return MnistSplit(
            train_ids=list(payload["train_ids"]),
            val_ids=list(payload["val_ids"]),
            split_seed=int(payload["split_seed"]),
        )

    ensure_dir(split_file.parent)
    train_ds = datasets.MNIST(root=str(data_dir), train=True, download=True, transform=None)
    labels = train_ds.targets.tolist()
    train_ids, val_ids = stratified_train_val_split(labels=labels, val_size=val_size, seed=seed)
    payload = {"train_ids": train_ids, "val_ids": val_ids, "split_seed": seed}
    write_json(split_file, payload)
    return MnistSplit(train_ids=train_ids, val_ids=val_ids, split_seed=seed)


class IndexedDataset(Dataset):
    """Dataset wrapper that returns (x, y, sample_id)."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sample_id = int(self.indices[idx])
        image, label = self.base_dataset[sample_id]
        if isinstance(image, Image.Image):
            raise RuntimeError("Expected tensor image after applying transform.")
        return image, int(label), sample_id


def build_mnist_indexed_dataset(
    data_dir: Path,
    train: bool,
    transform: transforms.Compose,
    indices: Sequence[int] | None = None,
) -> IndexedDataset:
    base = datasets.MNIST(root=str(data_dir), train=train, download=True, transform=transform)
    if indices is None:
        indices = list(range(len(base)))
    return IndexedDataset(base_dataset=base, indices=indices)


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )

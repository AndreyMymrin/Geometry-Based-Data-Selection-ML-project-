"""Generic dataset support for MNIST and CIFAR-10.

Uses native image sizes (28x28 for MNIST, 32x32 for CIFAR-10)
with lightweight transforms suitable for training on CPU.
"""

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


@dataclass(frozen=True)
class DatasetSplit:
    train_ids: list[int]
    val_ids: list[int]
    split_seed: int


DATASET_REGISTRY: dict[str, dict] = {
    "cifar10": {
        "class": datasets.CIFAR10,
        "in_channels": 3,
        "image_size": 32,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
    },
    "mnist": {
        "class": datasets.MNIST,
        "in_channels": 1,
        "image_size": 28,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
    },
}


def get_dataset_info(dataset_name: str) -> dict:
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported: {list(DATASET_REGISTRY.keys())}"
        )
    return DATASET_REGISTRY[dataset_name]


class Cutout:
    """Randomly mask out square patches from the image (DeVries & Taylor, 2017).

    Used by Toneva et al. (2019) for CIFAR-10 ResNet-18 training.
    """

    def __init__(self, n_holes: int = 1, length: int = 16) -> None:
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        mask = torch.ones_like(img)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[:, y1:y2, x1:x2] = 0.0
        return img * mask


def make_train_transform(dataset_name: str, augment: bool = False) -> transforms.Compose:
    """Training transform.

    When *augment* is True, applies dataset-specific augmentation following
    Toneva et al. (2019):
      - CIFAR-10: RandomCrop(32, padding=4) + RandomHorizontalFlip + Cutout(16)
      - MNIST: no augmentation (paper uses none)
    """
    info = get_dataset_info(dataset_name)
    t: list = []
    if augment and dataset_name == "cifar10":
        t.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    t.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=info["mean"], std=info["std"]),
    ])
    if augment and dataset_name == "cifar10":
        t.append(Cutout(n_holes=1, length=16))
    return transforms.Compose(t)


def make_eval_transform(dataset_name: str) -> transforms.Compose:
    info = get_dataset_info(dataset_name)
    t = [
        transforms.ToTensor(),
        transforms.Normalize(mean=info["mean"], std=info["std"]),
    ]
    return transforms.Compose(t)


def stratified_train_val_split(
    labels: Sequence[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    labels_np = np.asarray(labels)
    n = len(labels_np)
    if val_size <= 0 or val_size >= n:
        raise ValueError(f"val_size must be in [1, {n - 1}], got {val_size}")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in sorted(np.unique(labels_np).tolist()):
        class_idx = np.where(labels_np == label)[0]
        class_idx = class_idx[rng.permutation(len(class_idx))]
        raw_val_count = (len(class_idx) * val_size) / n
        class_val_count = max(1, min(int(round(raw_val_count)), len(class_idx) - 1))
        val_indices.extend(class_idx[:class_val_count].tolist())
        train_indices.extend(class_idx[class_val_count:].tolist())

    overflow = len(val_indices) - val_size
    if overflow > 0:
        train_indices.extend(val_indices[-overflow:])
        val_indices = val_indices[:-overflow]
    elif overflow < 0:
        missing = -overflow
        val_indices.extend(train_indices[-missing:])
        train_indices = train_indices[:-missing]

    return sorted(train_indices), sorted(val_indices)


def _get_labels(ds: Dataset) -> list[int]:
    """Extract labels from a torchvision dataset."""
    if hasattr(ds, "targets"):
        targets = ds.targets
        if isinstance(targets, torch.Tensor):
            return targets.tolist()
        return list(targets)
    raise ValueError("Cannot extract labels from dataset")


def load_or_create_split(
    data_dir: Path,
    split_file: Path,
    dataset_name: str,
    val_size: int,
    seed: int,
) -> DatasetSplit:
    if split_file.exists():
        payload = read_json(split_file)
        return DatasetSplit(
            train_ids=list(payload["train_ids"]),
            val_ids=list(payload["val_ids"]),
            split_seed=int(payload["split_seed"]),
        )

    ensure_dir(split_file.parent)
    info = get_dataset_info(dataset_name)
    ds = info["class"](root=str(data_dir), train=True, download=True)
    labels = _get_labels(ds)
    train_ids, val_ids = stratified_train_val_split(labels, val_size, seed)
    payload = {"train_ids": train_ids, "val_ids": val_ids, "split_seed": seed}
    write_json(split_file, payload)
    return DatasetSplit(train_ids=train_ids, val_ids=val_ids, split_seed=seed)


class IndexedDataset(Dataset):
    """Wraps a base dataset to return (image, label, sample_id) tuples."""

    def __init__(self, base_dataset: Dataset, indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        sample_id = int(self.indices[idx])
        image, label = self.base_dataset[sample_id]
        return image, int(label), sample_id


def build_indexed_dataset(
    data_dir: Path,
    dataset_name: str,
    train: bool,
    transform: transforms.Compose,
    indices: Sequence[int] | None = None,
) -> IndexedDataset:
    info = get_dataset_info(dataset_name)
    base = info["class"](root=str(data_dir), train=train, download=True, transform=transform)
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

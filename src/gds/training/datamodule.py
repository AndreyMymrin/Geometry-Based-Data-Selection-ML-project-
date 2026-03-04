from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from gds.data.mnist import (
    build_mnist_indexed_dataset,
    make_eval_transform,
    make_train_transform,
)


class MNISTSubsetDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        train_ids: list[int],
        val_ids: list[int],
        batch_size: int = 128,
        num_workers: int = 4,
        image_size: int = 224,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        train_transform = make_train_transform(image_size=self.image_size)
        eval_transform = make_eval_transform(image_size=self.image_size)

        self._train_dataset = build_mnist_indexed_dataset(
            data_dir=self.data_dir,
            train=True,
            transform=train_transform,
            indices=self.train_ids,
        )
        self._val_dataset = build_mnist_indexed_dataset(
            data_dir=self.data_dir,
            train=True,
            transform=eval_transform,
            indices=self.val_ids,
        )
        self._test_dataset = build_mnist_indexed_dataset(
            data_dir=self.data_dir,
            train=False,
            transform=eval_transform,
            indices=None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
        )

from __future__ import annotations

from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from gds.data.datasets import (
    build_indexed_dataset,
    make_eval_transform,
    make_train_transform,
)


class GenericSubsetDataModule(L.LightningDataModule):
    """Dataset-agnostic data module for MNIST, CIFAR-10, etc."""

    def __init__(
        self,
        data_dir: Path,
        dataset_name: str,
        train_ids: list[int],
        val_ids: list[int],
        batch_size: int = 128,
        num_workers: int = 2,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augment = augment

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: str | None = None) -> None:
        train_tf = make_train_transform(self.dataset_name, augment=self.augment)
        eval_tf = make_eval_transform(self.dataset_name)

        self._train_dataset = build_indexed_dataset(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            train=True,
            transform=train_tf,
            indices=self.train_ids,
        )
        self._val_dataset = build_indexed_dataset(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            train=True,
            transform=eval_tf,
            indices=self.val_ids,
        )
        self._test_dataset = build_indexed_dataset(
            data_dir=self.data_dir,
            dataset_name=self.dataset_name,
            train=False,
            transform=eval_tf,
            indices=None,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class TextSubsetDataModule(L.LightningDataModule):
    """DataModule for TinyShakespeare character-level language modelling."""

    def __init__(
        self,
        data_dir: Path,
        train_ids: list[int],
        val_ids: list[int],
        block_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 2,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.train_ids = train_ids
        self.val_ids = val_ids
        self.block_size = block_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        self._train_dataset = None
        self._val_dataset = None
        self._chunks = None
        self._vocab_size = None

    @property
    def vocab_size(self) -> int:
        if self._vocab_size is None:
            raise RuntimeError("Call setup() first")
        return self._vocab_size

    def setup(self, stage: str | None = None) -> None:
        from gds.data.tiny_shakespeare import (
            TinyShakespeareDataset,
            download_tiny_shakespeare,
            build_char_vocab,
            encode_text,
            chunk_into_samples,
        )
        text = download_tiny_shakespeare(self.data_dir)
        stoi, _ = build_char_vocab(text)
        self._vocab_size = len(stoi)
        encoded = encode_text(text, stoi)
        self._chunks = chunk_into_samples(encoded, self.block_size)

        self._train_dataset = TinyShakespeareDataset(self._chunks, self.train_ids)
        self._val_dataset = TinyShakespeareDataset(self._chunks, self.val_ids)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return self.val_dataloader()

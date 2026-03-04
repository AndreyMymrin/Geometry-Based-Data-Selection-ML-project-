from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("lightning")

from gds.training import datamodule as dm_mod
from gds.data.mnist import IndexedDataset
from gds.training.runner import run_training


class FakeMNIST:
    def __init__(self, root: str, train: bool, download: bool, transform=None) -> None:
        self.transform = transform
        self.train = train
        size = 120 if train else 40
        rng = np.random.default_rng(42 if train else 24)
        self.images = rng.integers(0, 255, size=(size, 28, 28), dtype=np.uint8)
        self.targets = [i % 10 for i in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        img = Image.fromarray(self.images[index], mode="L")
        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def test_train_loop_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dm_mod, "build_mnist_indexed_dataset", _fake_build_dataset)

    run_dir = tmp_path / "training" / "error_rate_ensemble" / "p00" / "seed101"
    train_ids = list(range(80))
    val_ids = list(range(80, 120))

    result = run_training(
        data_dir=tmp_path / "data",
        run_dir=run_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        method="error_rate_ensemble",
        percent_removed=0,
        seed=101,
        model_name="resnet18",
        max_epochs=1,
        patience=1,
        batch_size=16,
        num_workers=0,
        image_size=64,
        lr=1e-3,
        weight_decay=1e-4,
        accelerator="cpu",
        devices=1,
    )

    assert result.method == "error_rate_ensemble"
    assert result.percent_removed == 0
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "run_summary.json").exists()


def _fake_build_dataset(data_dir: Path, train: bool, transform, indices=None):
    base = FakeMNIST(root=str(data_dir), train=train, download=True, transform=transform)
    if indices is None:
        indices = list(range(len(base)))
    return IndexedDataset(base_dataset=base, indices=indices)

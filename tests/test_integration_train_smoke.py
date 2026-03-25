from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("lightning")

from gds.data.datasets import IndexedDataset
from gds.training.runner import run_training
from gds.training import datamodule as dm_mod


class FakeCIFAR:
    """Fake dataset that mimics CIFAR-10 (32x32x3)."""

    def __init__(self, root: str, train: bool, download: bool, transform=None) -> None:
        self.transform = transform
        self.train = train
        size = 120 if train else 40
        rng = np.random.default_rng(42 if train else 24)
        self.images = rng.integers(0, 255, size=(size, 32, 32, 3), dtype=np.uint8)
        self.targets = [i % 10 for i in range(size)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        img = Image.fromarray(self.images[index], mode="RGB")
        label = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _fake_build_indexed_dataset(data_dir, dataset_name, train, transform, indices=None):
    base = FakeCIFAR(root=str(data_dir), train=train, download=True, transform=transform)
    if indices is None:
        indices = list(range(len(base)))
    return IndexedDataset(base_dataset=base, indices=indices)


def test_train_loop_smoke(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(dm_mod, "build_indexed_dataset", _fake_build_indexed_dataset)

    run_dir = tmp_path / "training" / "forgetting_events" / "p00" / "seed101"
    train_ids = list(range(80))
    val_ids = list(range(80, 120))

    result = run_training(
        data_dir=tmp_path / "data",
        run_dir=run_dir,
        train_ids=train_ids,
        val_ids=val_ids,
        method="forgetting_events",
        percent_removed=0,
        seed=101,
        model_name="simple_cnn",
        max_epochs=1,
        patience=1,
        batch_size=16,
        num_workers=0,
        lr=1e-3,
        weight_decay=1e-4,
        accelerator="cpu",
        devices=1,
        dataset_name="cifar10",
        in_channels=3,
        num_classes=10,
    )

    assert result.method == "forgetting_events"
    assert result.percent_removed == 0
    assert (run_dir / "run_summary.json").exists()

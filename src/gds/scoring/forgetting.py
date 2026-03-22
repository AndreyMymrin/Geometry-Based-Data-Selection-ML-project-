"""Forgetting-event scorer following Toneva et al. (2018).

Algorithm 1 from "An Empirical Study of Example Forgetting during Deep
Neural Network Learning" (arXiv:1812.05159):

For each training run:
  - Initialise prev_acc[i] = 0 and forgetting_count[i] = 0 for all samples
  - For each mini-batch during SGD training:
      * Compute accuracy for every example in the batch
      * If prev_acc[i] > acc[i] (was correct, now wrong) → forgetting event
      * Update prev_acc[i]
  - Return forgetting_count

Multiple seeds are averaged to produce stable scores.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.auto import tqdm

from gds.models.simple_cnn import SimpleCNN
from gds.scoring.base import SampleScorer
from gds.scoring.utils import stable_rank_from_scores


def _build_classifier(
    model_name: str, num_classes: int, in_channels: int = 1
) -> nn.Module:
    """Build a randomly-initialised classifier (no pretrained weights)."""
    if model_name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(
        f"Unsupported scoring model '{model_name}'. "
        "Supported: simple_cnn, resnet18."
    )


def compute_forgetting_counts(
    model: nn.Module,
    loader: DataLoader,
    num_epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    show_progress: bool = True,
) -> dict[int, int]:
    """Train *model* for *num_epochs* and return per-sample forgetting counts.

    A forgetting event for sample *i* happens when, on consecutive
    presentations of *i* in a mini-batch, it goes from correctly classified
    to incorrectly classified (Algorithm 1 of Toneva et al.).
    """
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # State: last-known accuracy for each sample id.
    prev_acc: dict[int, int] = {}      # sample_id → 0 or 1
    forgetting: dict[int, int] = {}    # sample_id → count

    for epoch in range(num_epochs):
        epoch_iter = tqdm(
            loader,
            desc=f"  epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not show_progress,
        )
        for x, y, sample_ids in epoch_iter:
            x, y = x.to(device), y.to(device)

            # Forward + backward
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check accuracy *after* the gradient update (matches paper
            # definition: accuracy after processing the mini-batch).
            with torch.no_grad():
                preds = model(x).argmax(dim=1)
                correct = (preds == y).cpu()

            sids = sample_ids.tolist()
            for j, sid in enumerate(sids):
                c = int(correct[j].item())
                if prev_acc.get(sid, 0) > c:
                    forgetting[sid] = forgetting.get(sid, 0) + 1
                prev_acc[sid] = c
                # Ensure every sample has an entry
                forgetting.setdefault(sid, 0)

        scheduler.step()

    return forgetting


def run_forgetting_ensemble(
    loader: DataLoader,
    model_name: str,
    num_classes: int,
    num_epochs: int,
    seeds: list[int],
    lr: float,
    momentum: float,
    weight_decay: float,
    device: torch.device,
    show_progress: bool = True,
    in_channels: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train *len(seeds)* independent models and average forgetting counts.

    Returns
    -------
    scores : ndarray [n_samples]
        Mean forgetting count across seeds (float).
    labels : ndarray [n_samples]
        Ground-truth labels.
    sample_ids : ndarray [n_samples]
        Original sample indices.
    """
    # Collect all sample_ids and labels from loader (deterministic order).
    all_sids: list[int] = []
    all_labels: list[int] = []
    for _, y, sid in loader:
        all_labels.extend(y.tolist())
        all_sids.extend(sid.tolist())

    n_samples = len(all_sids)
    sid_to_idx = {sid: i for i, sid in enumerate(all_sids)}
    counts_sum = np.zeros(n_samples, dtype=np.float64)

    for seed in tqdm(seeds, desc="Forgetting seeds", disable=not show_progress):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = _build_classifier(model_name, num_classes, in_channels=in_channels)
        fc = compute_forgetting_counts(
            model=model,
            loader=loader,
            num_epochs=num_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            show_progress=show_progress,
        )
        for sid, count in fc.items():
            counts_sum[sid_to_idx[sid]] += count

    scores = (counts_sum / len(seeds)).astype(np.float32)
    sample_ids = np.array(all_sids, dtype=np.int64)
    labels = np.array(all_labels, dtype=np.int64)
    return scores, labels, sample_ids


class ForgettingEventScorer(SampleScorer):
    """Ranks training examples by forgetting-event frequency."""

    @property
    def name(self) -> str:
        return "forgetting_events"

    def score(
        self,
        sample_ids: list[int],
        labels: list[int],
        metadata: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        if metadata is None or "forgetting_scores" not in metadata:
            raise ValueError(
                "metadata['forgetting_scores'] is required.  "
                "Run forgetting-event tracking first."
            )

        scores = np.asarray(metadata["forgetting_scores"], dtype=np.float32)
        sample_ids_np = np.asarray(sample_ids, dtype=np.int64)
        labels_np = np.asarray(labels, dtype=np.int64)

        ranks = stable_rank_from_scores(sample_ids=sample_ids_np, scores=scores)

        df = pd.DataFrame(
            {
                "sample_id": sample_ids_np,
                "label": labels_np,
                "score": scores.astype(float),
                "rank": ranks.astype(int),
                "method": self.name,
            }
        )
        return df.sort_values("rank", kind="stable").reset_index(drop=True)

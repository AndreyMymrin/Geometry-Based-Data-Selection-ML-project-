"""Forgetting-event scorer following Toneva et al. (2018).

Algorithm 1 from "An Empirical Study of Example Forgetting during Deep
Neural Network Learning" (arXiv:1812.05159):

For each training run:
  - Initialise prev_acc[i] = 0 and forgetting_count[i] = 0 for all samples
  - For each mini-batch during SGD training:
      * Compute accuracy for every example in the batch
      * If prev_acc[i] > acc[i] (was correct, now wrong) -> forgetting event
      * Update prev_acc[i]
  - Return forgetting_count

Multiple seeds are averaged to produce stable scores.

Score used for ranking: mean forgetting count across seeds.
Higher count = more frequently forgotten = harder sample.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
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
    nesterov: bool = False,
    scheduler_name: str = "cosine",
    milestones: list[int] | None = None,
    gamma: float = 0.2,
) -> tuple[dict[int, int], dict[int, bool]]:
    """Train *model* for *num_epochs* and return per-sample forgetting counts.

    Following Toneva et al. (2019) Algorithm 1.  Samples that are **never
    learnt** (never correctly classified during the entire training) are
    treated separately — the paper assigns them infinite forgetting count.

    Returns
    -------
    forgetting : dict[int, int]
        Sample ID -> forgetting event count.
    ever_learnt : dict[int, bool]
        Sample ID -> True if the sample was correctly classified at least once.
    """
    model = model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay,
        nesterov=nesterov,
    )
    if scheduler_name == "multistep":
        scheduler = MultiStepLR(optimizer, milestones=milestones or [60, 120, 160], gamma=gamma)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    prev_acc: dict[int, int] = {}       # sample_id -> 0 or 1
    forgetting: dict[int, int] = {}     # sample_id -> count
    ever_learnt: dict[int, bool] = {}   # sample_id -> was ever correct?

    for epoch in range(num_epochs):
        epoch_iter = tqdm(
            loader,
            desc=f"  epoch {epoch + 1}/{num_epochs}",
            leave=False,
            disable=not show_progress,
        )
        for x, y, sample_ids in epoch_iter:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(x).argmax(dim=1)
                correct = (preds == y).cpu()

            sids = sample_ids.tolist()
            for j, sid in enumerate(sids):
                c = int(correct[j].item())
                if c == 1:
                    ever_learnt[sid] = True
                if prev_acc.get(sid, 0) > c:
                    forgetting[sid] = forgetting.get(sid, 0) + 1
                prev_acc[sid] = c
                forgetting.setdefault(sid, 0)
                ever_learnt.setdefault(sid, False)

        scheduler.step()

    return forgetting, ever_learnt


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
    nesterov: bool = False,
    scheduler_name: str = "cosine",
    milestones: list[int] | None = None,
    gamma: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train *len(seeds)* independent models and average forgetting counts.

    Returns
    -------
    forgetting_scores : ndarray [n_samples]
        Mean forgetting count across seeds.
    labels : ndarray [n_samples]
        Ground-truth labels.
    sample_ids : ndarray [n_samples]
        Original sample indices.
    """
    all_sids: list[int] = []
    all_labels: list[int] = []
    for _, y, sid in loader:
        all_labels.extend(y.tolist())
        all_sids.extend(sid.tolist())

    n_samples = len(all_sids)
    sid_to_idx = {sid: i for i, sid in enumerate(all_sids)}
    forgetting_sum = np.zeros(n_samples, dtype=np.float64)
    never_learnt_count = np.zeros(n_samples, dtype=np.int32)

    for seed in tqdm(seeds, desc="Forgetting seeds", disable=not show_progress):
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = _build_classifier(model_name, num_classes, in_channels=in_channels)
        forgetting, ever_learnt = compute_forgetting_counts(
            model=model,
            loader=loader,
            num_epochs=num_epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            device=device,
            show_progress=show_progress,
            nesterov=nesterov,
            scheduler_name=scheduler_name,
            milestones=milestones,
            gamma=gamma,
        )
        for sid, count in forgetting.items():
            forgetting_sum[sid_to_idx[sid]] += count
        for sid, learnt in ever_learnt.items():
            if not learnt:
                never_learnt_count[sid_to_idx[sid]] += 1

    n_seeds = len(seeds)
    forgetting_scores = (forgetting_sum / n_seeds).astype(np.float32)

    # Toneva et al.: "Samples that are never learnt are considered forgotten
    # an infinite number of times for sorting purposes."
    # For samples never learnt in ANY seed, assign max_forgetting + 1.
    max_forgetting = forgetting_scores.max()
    never_learnt_mask = never_learnt_count == n_seeds  # never learnt in ALL seeds
    n_never = never_learnt_mask.sum()
    if n_never > 0:
        forgetting_scores[never_learnt_mask] = max_forgetting + 1

    sample_ids = np.array(all_sids, dtype=np.int64)
    labels = np.array(all_labels, dtype=np.int64)

    n_unforgettable = (forgetting_scores == 0).sum()
    print(f"  Forgetting stats: {n_unforgettable}/{n_samples} unforgettable "
          f"({100*n_unforgettable/n_samples:.1f}%), "
          f"{n_never} never learnt ({100*n_never/n_samples:.1f}%)")
    print(f"  Forgetting counts: mean={forgetting_scores.mean():.3f}, max={forgetting_scores.max():.1f}")

    return forgetting_scores, labels, sample_ids


class ForgettingEventScorer(SampleScorer):
    """Ranks training examples by forgetting count (Toneva et al. 2018).

    Score = mean forgetting count across seeds.
    Higher count = more frequently forgotten = harder sample.
    """

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

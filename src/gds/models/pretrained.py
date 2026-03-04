from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torchvision import models


def _build_model(name: str) -> nn.Module:
    builders: dict[str, Callable[[], nn.Module]] = {
        "resnet18": lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
        "resnet34": lambda: models.resnet34(weights=models.ResNet34_Weights.DEFAULT),
        "resnet50": lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
        "densenet121": lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
        "mobilenet_v3_large": lambda: models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT
        ),
        "efficientnet_b0": lambda: models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        ),
    }
    if name not in builders:
        raise ValueError(f"Unsupported pretrained model: {name}")
    model = builders[name]()
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_pretrained_predictions(
    model_names: list[str],
    loader: DataLoader,
    device: torch.device,
    class_head_mode: str = "first10",
    show_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      predictions: [n_models, n_samples]
      labels: [n_samples]
      sample_ids: [n_samples]
    """
    labels_all: list[np.ndarray] = []
    sample_ids_all: list[np.ndarray] = []
    model_preds: list[np.ndarray] = []

    for model_name in tqdm(
        model_names,
        desc="Scoring models",
        disable=not show_progress,
    ):
        model = _build_model(model_name).to(device)
        preds_batches: list[np.ndarray] = []
        total_batches = len(loader) if hasattr(loader, "__len__") else None
        with torch.no_grad():
            for batch in tqdm(
                loader,
                desc=f"{model_name} batches",
                total=total_batches,
                leave=False,
                disable=not show_progress,
            ):
                x, y, sample_ids = batch
                x = x.to(device)
                logits = model(x)
                if class_head_mode == "first10":
                    logits = logits[:, :10]
                elif class_head_mode != "argmax_1000":
                    raise ValueError(
                        f"Unsupported class_head_mode={class_head_mode}. "
                        "Use first10 or argmax_1000."
                    )
                preds = torch.argmax(logits, dim=1)
                preds_batches.append(preds.cpu().numpy())
                if len(model_preds) == 0:
                    labels_all.append(y.numpy())
                    sample_ids_all.append(sample_ids.numpy())
        model_preds.append(np.concatenate(preds_batches, axis=0))

    labels = np.concatenate(labels_all, axis=0)
    sample_ids = np.concatenate(sample_ids_all, axis=0)
    predictions = np.stack(model_preds, axis=0)
    return predictions, labels, sample_ids

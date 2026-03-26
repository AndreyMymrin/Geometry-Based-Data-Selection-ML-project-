"""Lightweight CNN for small images (28x28 / 32x32).

Designed to be fast while providing reasonable accuracy.
Supports extracting penultimate-layer embeddings for t-SNE visualization.
"""

from __future__ import annotations

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Small CNN suitable for MNIST (28x28x1) and CIFAR-10 (32x32x3).

    Architecture:
        Conv(in, 64) -> Conv(64, 64) -> MaxPool
        Conv(64, 128) -> Conv(128, 128) -> MaxPool
        GlobalAvgPool -> FC(128, 256) -> FC(256, num_classes)

    The 256-dim vector before the final FC is the "embedding".
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_layer = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.embedding_layer(x)
        return self.head(x)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 256-dim embeddings from the penultimate layer."""
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.embedding_layer(x)

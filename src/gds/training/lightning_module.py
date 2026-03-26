from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torchvision import models


class ImageClassifier(L.LightningModule):
    """Configurable image classifier supporting multiple optimizers and schedulers.

    Optimizer choices (``optimizer`` param):
      - ``"sgd"``  — SGD with configurable momentum/nesterov (paper default)
      - ``"adamw"`` — AdamW

    Scheduler choices (``scheduler`` param):
      - ``"cosine"``    — CosineAnnealingLR(T_max=max_epochs)
      - ``"multistep"`` — MultiStepLR(milestones, gamma)
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 20,
        in_channels: int = 1,
        # Optimizer config
        optimizer: str = "adamw",
        momentum: float = 0.9,
        nesterov: bool = False,
        # Scheduler config
        scheduler: str = "cosine",
        milestones: list[int] | None = None,
        gamma: float = 0.2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.model = _build_classifier(
            model_name=model_name, num_classes=num_classes, in_channels=in_channels
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage != "train"))
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> dict:
        hp = self.hparams
        if hp.optimizer == "sgd":
            optimizer = SGD(
                self.parameters(),
                lr=hp.lr,
                momentum=hp.momentum,
                weight_decay=hp.weight_decay,
                nesterov=hp.nesterov,
            )
        elif hp.optimizer == "adamw":
            optimizer = AdamW(
                self.parameters(),
                lr=hp.lr,
                weight_decay=hp.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer '{hp.optimizer}'. Use 'sgd' or 'adamw'.")

        if hp.scheduler == "multistep":
            milestones = hp.milestones or [60, 120, 160]
            sched = MultiStepLR(optimizer, milestones=milestones, gamma=hp.gamma)
        elif hp.scheduler == "cosine":
            sched = CosineAnnealingLR(optimizer, T_max=hp.max_epochs)
        else:
            raise ValueError(f"Unknown scheduler '{hp.scheduler}'. Use 'cosine' or 'multistep'.")

        return {"optimizer": optimizer, "lr_scheduler": sched}


# Keep backward-compatible alias
ResNet18Classifier = ImageClassifier


class NanoGPTLightning(L.LightningModule):
    """Lightning wrapper for character-level nanoGPT on TinyShakespeare.

    Training setup matches Karpathy's nanoGPT train_shakespeare_char.py:
      - AdamW with beta2=0.99, weight_decay=0.1
      - Linear warmup + cosine decay to min_lr
      - Gradient clipping at 1.0 (set via Trainer)
    """

    def __init__(
        self,
        vocab_size: int = 65,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 6,
        n_embd: int = 384,
        dropout: float = 0.2,
        lr: float = 1e-3,
        min_lr: float = 1e-4,
        weight_decay: float = 0.1,
        beta1: float = 0.9,
        beta2: float = 0.99,
        warmup_fraction: float = 0.02,
        max_epochs: int = 100,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        from gds.models.nano_gpt import NanoGPT

        self.model = NanoGPT(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            dropout=dropout,
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.model(idx)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        x, y, _ = batch
        logits = self(x)
        loss = self.criterion(
            logits.view(-1, self.hparams.vocab_size), y.view(-1)
        )
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=(stage != "train"))
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> dict:
        hp = self.hparams
        optimizer = AdamW(
            self.parameters(),
            lr=hp.lr,
            weight_decay=hp.weight_decay,
            betas=(hp.beta1, hp.beta2),
        )
        # Karpathy-style: linear warmup then cosine decay to min_lr
        warmup_epochs = max(1, int(hp.max_epochs * hp.warmup_fraction))
        from torch.optim.lr_scheduler import LambdaLR
        import math

        def lr_lambda(epoch: int) -> float:
            # Linear warmup
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            # Cosine decay to min_lr / lr
            progress = (epoch - warmup_epochs) / max(hp.max_epochs - warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = hp.min_lr / hp.lr
            return min_ratio + (1.0 - min_ratio) * cosine

        scheduler = LambdaLR(optimizer, lr_lambda)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def _build_classifier(
    model_name: str, num_classes: int, in_channels: int = 1
) -> nn.Module:
    if model_name == "simple_cnn":
        from gds.models.simple_cnn import SimpleCNN
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
        f"Unsupported training model '{model_name}'. "
        "Supported: simple_cnn, resnet18."
    )

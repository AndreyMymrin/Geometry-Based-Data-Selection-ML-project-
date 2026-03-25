from __future__ import annotations

import shutil
from pathlib import Path

import lightning as L
import pandas as pd
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from gds.common.io import ensure_dir, write_json
from gds.common.seed import seed_everything
from gds.common.types import RunResult
from gds.training.datamodule import GenericSubsetDataModule, TextSubsetDataModule
from gds.training.lightning_module import ImageClassifier, NanoGPTLightning


def _load_best_val_loss(metrics_path: Path) -> float:
    """Load best validation loss for text models (no accuracy metric)."""
    df = pd.read_csv(metrics_path)
    col = "val_loss_epoch" if "val_loss_epoch" in df.columns else "val_loss"
    valid = df.dropna(subset=[col])
    if valid.empty:
        return float("nan")
    return float(valid[col].min())


def _load_best_val_metrics(metrics_path: Path) -> tuple[float, float]:
    df = pd.read_csv(metrics_path)
    if "val_acc_epoch" in df.columns:
        val_acc_col = "val_acc_epoch"
        val_loss_col = "val_loss_epoch"
    else:
        val_acc_col = "val_acc"
        val_loss_col = "val_loss"
    valid = df.dropna(subset=[val_acc_col])
    if valid.empty:
        return float("nan"), float("nan")
    best_idx = valid[val_acc_col].idxmax()
    row = valid.loc[best_idx]
    return float(row[val_acc_col]), float(row[val_loss_col])


def run_training(
    data_dir: Path,
    run_dir: Path,
    train_ids: list[int],
    val_ids: list[int],
    method: str,
    percent_removed: int,
    seed: int,
    model_name: str,
    max_epochs: int,
    patience: int,
    batch_size: int,
    num_workers: int,
    lr: float,
    weight_decay: float,
    accelerator: str = "auto",
    devices: int | str = 1,
    deterministic: bool | str = "warn",
    dataset_name: str | None = None,
    in_channels: int = 1,
    num_classes: int = 10,
    save_checkpoints: bool = False,
    # Optimizer / scheduler config
    optimizer: str = "adamw",
    momentum: float = 0.9,
    nesterov: bool = False,
    scheduler: str = "cosine",
    milestones: list[int] | None = None,
    gamma: float = 0.2,
    # Augmentation
    augment: bool = False,
    # Gradient clipping
    gradient_clip_val: float | None = None,
    # Text dataset support
    is_text: bool = False,
    block_size: int = 256,
    vocab_size: int | None = None,
    # NanoGPT-specific (Karpathy config)
    min_lr: float = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    warmup_fraction: float = 0.02,
) -> RunResult:
    ensure_dir(run_dir)
    seed_everything(seed)
    L.seed_everything(seed, workers=True)

    if is_text:
        dm = TextSubsetDataModule(
            data_dir=data_dir,
            train_ids=train_ids,
            val_ids=val_ids,
            block_size=block_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        dm.setup()
        if vocab_size is None:
            vocab_size = dm.vocab_size
        model = NanoGPTLightning(
            vocab_size=vocab_size,
            block_size=block_size,
            lr=lr,
            min_lr=min_lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            warmup_fraction=warmup_fraction,
            max_epochs=max_epochs,
        )
    else:
        if dataset_name is None:
            dataset_name = "mnist"
        dm = GenericSubsetDataModule(
            data_dir=data_dir,
            dataset_name=dataset_name,
            train_ids=train_ids,
            val_ids=val_ids,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=augment,
        )
        model = ImageClassifier(
            model_name=model_name,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            in_channels=in_channels,
            optimizer=optimizer,
            momentum=momentum,
            nesterov=nesterov,
            scheduler=scheduler,
            milestones=milestones,
            gamma=gamma,
        )

    logger = CSVLogger(save_dir=str(run_dir), name="logs")

    if is_text:
        monitor_metric, monitor_mode = "val_loss", "min"
    else:
        monitor_metric, monitor_mode = "val_acc", "max"

    callbacks = [EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=patience)]

    if save_checkpoints:
        checkpoint_dir = ensure_dir(run_dir / "checkpoints")
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename="best",
                monitor=monitor_metric,
                mode=monitor_mode,
                save_top_k=1,
            )
        )

    # Use mixed precision on GPU for ~2x throughput on A100/V100
    precision = "16-mixed" if accelerator in ("gpu", "cuda") else "32-true"

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        deterministic=deterministic,
        enable_progress_bar=True,
        enable_checkpointing=save_checkpoints,
        gradient_clip_val=gradient_clip_val,
    )

    trainer.fit(model=model, datamodule=dm)
    ckpt_path = "best" if save_checkpoints else None
    test_metrics = trainer.test(model=model, datamodule=dm, ckpt_path=ckpt_path)
    if is_text:
        test_acc = float("nan")  # text model doesn't produce accuracy
    else:
        test_acc = float(test_metrics[0].get("test_acc", float("nan")))

    lightning_metrics_path = Path(logger.log_dir) / "metrics.csv"
    run_metrics_path = run_dir / "metrics.csv"
    shutil.copy2(lightning_metrics_path, run_metrics_path)
    if is_text:
        best_val_acc = float("nan")
        best_val_loss = _load_best_val_loss(run_metrics_path)
    else:
        best_val_acc, best_val_loss = _load_best_val_metrics(run_metrics_path)

    result = RunResult(
        method=method,
        percent_removed=percent_removed,
        seed=seed,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        test_acc=test_acc,
    )
    write_json(run_dir / "run_summary.json", result.__dict__)
    return result

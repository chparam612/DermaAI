"""Core training and validation loop engine."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
    clip_grad: Optional[float] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Execute one training epoch.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device ('cpu' or 'cuda').
        clip_grad: Optional max gradient norm for clipping.
        max_batches: Optional limit on number of batches (for smoke test).

    Returns:
        Dictionary containing 'loss' and 'accuracy'.
    """
    model.train()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total_samples = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if clip_grad is not None and clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)

    return {
        "train_loss": avg_loss,
        "train_accuracy": acc,
        "train_samples": total_samples,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str = "cpu",
    max_batches: Optional[int] = None,
    class_names: Optional[list] = None,
) -> Dict[str, float]:
    """Execute evaluation over validation DataLoader.

    Args:
        model: Model to evaluate.
        dataloader: Validation DataLoader.
        criterion: Loss function.
        device: Device.
        max_batches: Optional limit on number of batches.
        class_names: List of diagnostic class names.

    Returns:
        Dictionary of validation metrics (val_loss, val_accuracy, val_balanced_acc, val_macro_f1).
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)

    if total_samples > 0:
        metrics = calculate_classification_metrics(all_targets, all_preds, class_names=class_names)
        return {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_balanced_acc": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
        }

    return {
        "val_loss": avg_loss,
        "val_accuracy": 0.0,
        "val_balanced_acc": 0.0,
        "val_macro_f1": 0.0,
        "val_weighted_f1": 0.0,
    }

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
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Execute one training epoch with optional gradient accumulation.

    Args:
        model: Model to train.
        dataloader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device ('cpu' or 'cuda').
        clip_grad: Optional max gradient norm for clipping.
        max_batches: Optional limit on number of batches (for smoke test).
        accumulation_steps: Number of forward/backward steps before optimizer step.

    Returns:
        Dictionary containing 'train_loss', 'train_accuracy', 'train_samples', and 'peak_vram_mb'.
    """
    model.train()
    model.to(device)

    is_cuda = (device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
    scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)
    accum_steps = max(1, int(accumulation_steps))

    total_loss = 0.0
    correct = 0
    total_samples = 0

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, labels, _ = batch
        images = images.to(device, non_blocking=is_cuda)
        labels = labels.to(device, non_blocking=is_cuda)

        with torch.amp.autocast("cuda", enabled=is_cuda):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_scaled = loss / accum_steps

        if is_cuda:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        is_accum_step = ((batch_idx + 1) % accum_steps == 0) or ((batch_idx + 1) == len(dataloader))
        if is_accum_step:
            if is_cuda:
                if clip_grad is not None and clip_grad > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                if clip_grad is not None and clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
            optimizer.zero_grad()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total_samples += batch_size

        if (batch_idx + 1) % 3 == 0 or (batch_idx + 1) == len(dataloader):
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if is_cuda else 0.0
            logger.info(
                f"Train Batch [{batch_idx + 1}/{len(dataloader)}] - "
                f"Loss: {loss.item():.4f} - "
                f"Running Acc: {correct / total_samples:.4f} - "
                f"Peak VRAM: {vram_mb:.1f} MB"
            )

    avg_loss = total_loss / max(total_samples, 1)
    acc = correct / max(total_samples, 1)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024 ** 2), 2) if is_cuda else 0.0

    return {
        "train_loss": avg_loss,
        "train_accuracy": acc,
        "train_samples": total_samples,
        "peak_vram_mb": peak_vram,
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

    is_cuda = (device == "cuda" or (isinstance(device, str) and device.startswith("cuda")))
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images, labels, _ = batch
            images = images.to(device, non_blocking=is_cuda)
            labels = labels.to(device, non_blocking=is_cuda)

            with torch.amp.autocast("cuda", enabled=is_cuda):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            total_samples += batch_size

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(dataloader):
                logger.info(f"Val Batch [{batch_idx + 1}/{len(dataloader)}]")

    avg_loss = total_loss / max(total_samples, 1)

    if total_samples > 0:
        metrics = calculate_classification_metrics(all_targets, all_preds, class_names=class_names)
        pred_counts = {}
        for p in all_preds:
            cls_name = class_names[p] if (class_names and p < len(class_names)) else str(p)
            pred_counts[cls_name] = pred_counts.get(cls_name, 0) + 1
        num_pred_classes = len(set(all_preds))

        return {
            "val_loss": avg_loss,
            "val_accuracy": metrics["accuracy"],
            "val_balanced_acc": metrics["balanced_accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "val_weighted_f1": metrics["weighted_f1"],
            "val_num_predicted_classes": num_pred_classes,
            "val_prediction_distribution": pred_counts,
        }

    return {
        "val_loss": avg_loss,
        "val_accuracy": 0.0,
        "val_balanced_acc": 0.0,
        "val_macro_f1": 0.0,
        "val_weighted_f1": 0.0,
        "val_num_predicted_classes": 0,
        "val_prediction_distribution": {},
    }

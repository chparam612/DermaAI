"""Loss functions for dermatology diagnostic classification."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing, matching PanDerm official implementation."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate loss.

        Args:
            x: Logits tensor of shape (batch_size, num_classes).
            target: Ground truth indices of shape (batch_size).
        """
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def compute_class_weights(
    targets_or_counts: Union[List[int], np.ndarray, Dict[int, int]],
    num_classes: int,
    norm: bool = True,
) -> torch.Tensor:
    """Compute inverse class frequency weights: w_c = N / (C * N_c).

    Args:
        targets_or_counts: List/array of integer labels, or dict mapping class_idx -> count.
        num_classes: Total number of classes.
        norm: Whether to normalize weights so their mean is 1.0.

    Returns:
        torch.FloatTensor of shape (num_classes,).
    """
    if isinstance(targets_or_counts, dict):
        counts = np.array([targets_or_counts.get(i, 0) for i in range(num_classes)], dtype=np.float32)
    elif isinstance(targets_or_counts, (list, np.ndarray, torch.Tensor)):
        arr = np.asarray(targets_or_counts)
        counts = np.bincount(arr.astype(int), minlength=num_classes).astype(np.float32)
    else:
        raise TypeError(f"Unsupported type for targets_or_counts: {type(targets_or_counts)}")

    total_samples = float(np.sum(counts))
    weights = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total_samples / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    if norm:
        weights = weights / np.mean(weights)

    return torch.from_numpy(weights).float()


def get_loss_function(
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build classification loss function.

    Args:
        loss_name: 'cross_entropy', 'ce', 'class_balanced_ce', 'weighted_ce', or 'label_smoothing'.
        label_smoothing: Smoothing factor epsilon in [0, 1).
        weight: Optional class weight tensor.

    Returns:
        PyTorch nn.Module.
    """
    name_clean = loss_name.strip().lower()
    if name_clean in ("label_smoothing", "ls_ce") or (label_smoothing > 0 and name_clean not in ("class_balanced_ce", "weighted_ce")):
        smoothing = label_smoothing if label_smoothing > 0 else 0.1
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    return nn.CrossEntropyLoss(weight=weight)


build_loss = get_loss_function

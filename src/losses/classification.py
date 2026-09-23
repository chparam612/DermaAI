"""Loss functions for dermatology diagnostic classification."""

from __future__ import annotations

from typing import Optional
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


def get_loss_function(
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    weight: Optional[torch.Tensor] = None,
) -> nn.Module:
    """Build classification loss function.

    Args:
        loss_name: 'cross_entropy', 'ce', or 'label_smoothing'.
        label_smoothing: Smoothing factor epsilon in [0, 1).
        weight: Optional class weight tensor.

    Returns:
        PyTorch nn.Module.
    """
    name_clean = loss_name.strip().lower()
    if name_clean in ("label_smoothing", "ls_ce") or label_smoothing > 0:
        smoothing = label_smoothing if label_smoothing > 0 else 0.1
        return LabelSmoothingCrossEntropy(smoothing=smoothing)

    return nn.CrossEntropyLoss(weight=weight)

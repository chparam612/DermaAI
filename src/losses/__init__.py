"""Loss functions and optimization objectives for dermatology classification.

Phase 0 provides standard cross-entropy and focal loss stubs.
Phase 1+ extends with class-weighted, subgroup-weighted, and fairness-aware loss functions.
"""

from typing import Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_loss(loss_name: str, class_weights: Optional[torch.Tensor] = None, **kwargs: Any) -> nn.Module:
    """Build criterion for training.

    Args:
        loss_name: Criterion identifier (e.g. 'cross_entropy', 'bce_with_logits', 'weighted_cross_entropy').
        class_weights: Optional class balancing tensor.

    Returns:
        torch.nn.Module loss instance.
    """
    if loss_name.lower() in ["ce", "cross_entropy"]:
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name.lower() in ["bce", "bce_with_logits"]:
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        raise NotImplementedError(f"Loss '{loss_name}' is not yet implemented.")


__all__ = ["build_loss"]

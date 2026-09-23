"""Model factory for initializing baseline CNNs and PanDerm models."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch.nn as nn

from src.models.baseline import BaselineClassifier
from src.models.panderm import PanDermClassifier


def build_model(
    model_name: str = "resnet50",
    num_classes: int = 11,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
) -> nn.Module:
    """Instantiate a classification model for Exp-1 or Exp-2.

    Args:
        model_name: Model identifier (e.g. 'resnet50', 'efficientnet_b0', 'panderm_base', 'panderm_large').
        num_classes: Number of diagnostic output classes (11).
        pretrained: Whether to load ImageNet or foundation pretrained weights.
        checkpoint_path: Path to checkpoint weights (for PanDerm).
        **kwargs: Additional model-specific hyperparameters.

    Returns:
        Instantiated nn.Module.
    """
    name_clean = model_name.strip().lower()

    if "panderm" in name_clean:
        return PanDermClassifier(
            architecture=name_clean,
            num_classes=num_classes,
            pretrained=pretrained,
            checkpoint_path=checkpoint_path,
            **kwargs,
        )

    # Standard CNN baseline
    return BaselineClassifier(
        architecture=name_clean,
        num_classes=num_classes,
        pretrained=pretrained,
        **kwargs,
    )

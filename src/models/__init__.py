"""Model definitions and foundation model adapters for SkinCancerTracker.

In Phase 1+, this package provides backbones (PanDerm, DermFM-Zero, ResNet, ViT)
and linear probing / fine-tuning heads.
"""

from typing import Any, Dict, Optional
import torch.nn as nn


def build_model(model_name: str, num_classes: int = 2, pretrained: bool = True, **kwargs: Any) -> nn.Module:
    """Factory stub for model instantiation (to be extended in Phase 1).

    Args:
        model_name: Architecture identifier (e.g. 'resnet50', 'panderm_base', 'dermfm_zero').
        num_classes: Target classification head dimensionality.
        pretrained: Whether to load pretrained foundation model weights.

    Returns:
        torch.nn.Module instance.
    """
    # Simple lightweight backbone for Phase 0 validation and testing
    import torchvision.models as models

    if model_name.lower() in ["resnet18", "baseline_resnet18"]:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    elif model_name.lower() in ["resnet50", "baseline_resnet50"]:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
    else:
        raise NotImplementedError(
            f"Model '{model_name}' is scheduled for implementation in Phase 1 (foundation model adapters)."
        )


__all__ = ["build_model"]

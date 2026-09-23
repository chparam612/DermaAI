"""Baseline CNN models for Exp-1 dermatology image classification."""

from __future__ import annotations

import logging
from typing import Optional
import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)


class BaselineClassifier(nn.Module):
    """Standard CNN baseline classifier (Exp-1) supporting ResNet and EfficientNet backbones.

    Replaces final linear projection head with an 11-class diagnostic classification layer.
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 11,
        pretrained: bool = True,
        dropout_rate: float = 0.2,
    ):
        """Initialize BaselineClassifier.

        Args:
            architecture: Backbone model name ('resnet50', 'resnet34', 'resnet18', 'efficientnet_b0').
            num_classes: Number of diagnostic target classes (11).
            pretrained: Whether to initialize with ImageNet weights if accessible.
            dropout_rate: Dropout probability before final classification head.
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained

        if self.architecture.startswith("resnet"):
            self.backbone, in_features = self._build_resnet(self.architecture, pretrained)
            # Replace ResNet fc head
            if dropout_rate > 0:
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
            else:
                self.backbone.fc = nn.Linear(in_features, num_classes)
        elif self.architecture.startswith("efficientnet"):
            self.backbone, in_features = self._build_efficientnet(self.architecture, pretrained)
            # Replace EfficientNet classifier head
            if dropout_rate > 0:
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate),
                    nn.Linear(in_features, num_classes)
                )
            else:
                self.backbone.classifier = nn.Linear(in_features, num_classes)
        else:
            raise ValueError(f"Unsupported baseline architecture: {architecture}")

    def _build_resnet(self, name: str, pretrained: bool) -> tuple[nn.Module, int]:
        weights = "DEFAULT" if pretrained else None
        try:
            if name == "resnet50":
                model = models.resnet50(weights=weights)
                in_features = model.fc.in_features
            elif name == "resnet34":
                model = models.resnet34(weights=weights)
                in_features = model.fc.in_features
            elif name == "resnet18":
                model = models.resnet18(weights=weights)
                in_features = model.fc.in_features
            else:
                raise ValueError(f"Unknown ResNet variant: {name}")
        except Exception as e:
            logger.warning(f"Could not load online weights for {name} ({e}). Falling back to uninitialized.")
            model = getattr(models, name)(weights=None)
            in_features = model.fc.in_features

        return model, in_features

    def _build_efficientnet(self, name: str, pretrained: bool) -> tuple[nn.Module, int]:
        weights = "DEFAULT" if pretrained else None
        try:
            if name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=weights)
                in_features = model.classifier[1].in_features
            else:
                raise ValueError(f"Unknown EfficientNet variant: {name}")
        except Exception as e:
            logger.warning(f"Could not load online weights for {name} ({e}). Falling back to uninitialized.")
            model = getattr(models, name)(weights=None)
            in_features = model.classifier[1].in_features

        return model, in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, height, width).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        return self.backbone(x)

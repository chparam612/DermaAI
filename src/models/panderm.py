"""PanDerm fine-tuning adapter module for Exp-2."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
from typing import Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Ensure external/panderm/classification is importable
PANDERM_CLASSIFICATION_DIR = Path(__file__).resolve().parent.parent.parent / "external" / "panderm" / "classification"
if PANDERM_CLASSIFICATION_DIR.exists() and str(PANDERM_CLASSIFICATION_DIR) not in sys.path:
    sys.path.insert(0, str(PANDERM_CLASSIFICATION_DIR))


class PanDermClassifier(nn.Module):
    """Adapter wrapping official PanDerm Vision Transformer for 11-class dermatology classification.

    Directly leverages panderm_base_patch16_224_finetune / panderm_large_patch16_224_finetune
    from external/panderm/classification/models/modeling_finetune.py.
    """

    def __init__(
        self,
        architecture: str = "panderm_base",
        num_classes: int = 11,
        pretrained: bool = True,
        checkpoint_path: Optional[str] = None,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        attn_drop_rate: float = 0.0,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        use_rel_pos_bias: bool = True,
        layer_scale_init_value: float = 0.1,
        lin_probe: bool = False,
        dropout_rate: Optional[float] = None,
        **kwargs: Any,
    ):
        """Initialize PanDermClassifier.

        Args:
            architecture: 'panderm_base' or 'panderm_large'.
            num_classes: Number of output diagnostic classes (11).
            pretrained: Whether to attempt loading pretrained checkpoint.
            checkpoint_path: Path to official PanDerm pretrained weights (.pth / .bin).
            drop_rate: Dropout rate.
            drop_path_rate: Stochastic depth drop path rate.
            attn_drop_rate: Attention dropout rate.
            use_mean_pooling: Use mean pooling instead of CLS token.
            init_scale: Classifier initialization scale.
            use_rel_pos_bias: Enable relative position bias.
            layer_scale_init_value: Layer scale parameter (0.1 for base, 1e-5 for large).
            lin_probe: Enable linear probe mode (freezes backbone).
            dropout_rate: Alias for drop_rate.
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        effective_drop_rate = drop_rate if dropout_rate is None else dropout_rate

        # Try to import from official PanDerm repo
        try:
            from models.modeling_finetune import (
                panderm_base_patch16_224_finetune,
                panderm_large_patch16_224_finetune,
            )
            is_large = "large" in self.architecture
            fn = panderm_large_patch16_224_finetune if is_large else panderm_base_patch16_224_finetune
            init_val = 1e-5 if is_large else layer_scale_init_value

            self.model = fn(
                pretrained=False,
                num_classes=num_classes,
                drop_rate=effective_drop_rate,
                drop_path_rate=drop_path_rate,
                attn_drop_rate=attn_drop_rate,
                drop_block_rate=None,
                use_mean_pooling=use_mean_pooling,
                init_scale=init_scale,
                use_rel_pos_bias=use_rel_pos_bias,
                init_values=init_val,
                lin_probe=lin_probe,
            )
            logger.info(f"Initialized official PanDerm ViT ({'Large' if is_large else 'Base'})")
        except Exception as e:
            logger.warning(
                f"Could not initialize official PanDerm ViT directly ({e}). "
                "Falling back to timm standard ViT backbone."
            )
            import timm
            self.model = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes)

        if pretrained and checkpoint_path:
            self._load_checkpoint(checkpoint_path)
        elif pretrained:
            logger.info("No checkpoint_path specified. Initializing PanDerm with random weights.")

    def _load_checkpoint(self, path_str: str) -> None:
        """Load pretrained checkpoint from disk with key remapping."""
        path = Path(path_str)
        if not path.is_file():
            logger.warning(
                f"PanDerm checkpoint not found at '{path}'. "
                "Proceeding with randomly initialized weights for smoke test."
            )
            return

        logger.info(f"Loading PanDerm pretrained weights from {path}")
        try:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Remap keys if necessary
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("module."):
                    new_k = new_k[7:]
                if new_k.startswith("encoder."):
                    new_k = new_k[8:]
                cleaned_state_dict[new_k] = v

            missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
            logger.info(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {path}: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor of shape (B, 3, 224, 224).

        Returns:
            Logits of shape (B, num_classes).
        """
        return self.model(x)

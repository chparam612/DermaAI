"""PanDerm fine-tuning adapter module for Exp-2."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Ensure external/panderm/classification is importable
env_panderm_dir = os.environ.get("PANDERM_CLASSIFICATION_DIR")
if env_panderm_dir and Path(env_panderm_dir).exists():
    PANDERM_CLASSIFICATION_DIR = Path(env_panderm_dir).resolve()
else:
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
        use_rel_pos_bias: Optional[bool] = False,
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
            use_rel_pos_bias: Enable relative position bias (default False for standard pos_embed).
            layer_scale_init_value: Layer scale parameter (0.1 for base, 1e-5 for large).
            lin_probe: Enable linear probe mode (freezes backbone).
            dropout_rate: Alias for drop_rate.
        """
        super().__init__()
        self.architecture = architecture.lower()
        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path
        effective_drop_rate = drop_rate if dropout_rate is None else dropout_rate

        # Auto-detect relative position bias from checkpoint if use_rel_pos_bias is None
        if use_rel_pos_bias is None:
            if checkpoint_path and Path(checkpoint_path).is_file():
                try:
                    ckpt_preview = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    sd_preview = ckpt_preview.get("model", ckpt_preview.get("state_dict", ckpt_preview))
                    use_rel_pos_bias = any("rel_pos_bias" in k for k in sd_preview.keys())
                except Exception:
                    use_rel_pos_bias = False
            else:
                use_rel_pos_bias = False

        # Import official PanDerm ViT architecture
        # modeling_finetune.py does not need open_clip, but models/__init__.py imports builder.py which does.
        # We support both standard package import and direct file loading, with NO silent fallback to generic timm ViT.
        try:
            from models.modeling_finetune import (
                panderm_base_patch16_224_finetune,
                panderm_large_patch16_224_finetune,
            )
        except (ImportError, ModuleNotFoundError) as err:
            logger.info(f"Standard package import of PanDerm failed ({err}); loading modeling_finetune directly from file...")
            import importlib.util
            finetune_path = PANDERM_CLASSIFICATION_DIR / "models" / "modeling_finetune.py"
            if finetune_path.is_file():
                spec = importlib.util.spec_from_file_location("models.modeling_finetune", finetune_path)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["models.modeling_finetune"] = mod
                spec.loader.exec_module(mod)
                panderm_base_patch16_224_finetune = mod.panderm_base_patch16_224_finetune
                panderm_large_patch16_224_finetune = mod.panderm_large_patch16_224_finetune
            else:
                raise RuntimeError(
                    f"CRITICAL: Failed to locate official PanDerm architecture at {finetune_path}. "
                    "Cannot initialize official PanDerm ViT model."
                ) from err

        is_large = "large" in self.architecture
        fn = panderm_large_patch16_224_finetune if is_large else panderm_base_patch16_224_finetune
        init_val = 1e-5 if is_large else layer_scale_init_value

        try:
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
            raise RuntimeError(
                f"CRITICAL: Failed to instantiate official PanDerm ViT model ({e}). "
                "Refusing to fall back to a generic model architecture."
            ) from e

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

            # Remap keys following official PanDerm run_class_finetuning.py
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith("module."):
                    new_k = new_k[7:]
                if new_k.startswith("encoder."):
                    new_k = new_k[8:]
                cleaned_state_dict[new_k] = v

            # Remove decoder and teacher keys
            for key in list(cleaned_state_dict.keys()):
                if key.startswith("decoder.") or key.startswith("teacher."):
                    cleaned_state_dict.pop(key)

            # Remap norm. to fc_norm.
            for key in list(cleaned_state_dict.keys()):
                if key.startswith("norm."):
                    new_key = key.replace("norm.", "fc_norm.")
                    cleaned_state_dict[new_key] = cleaned_state_dict.pop(key)

            # Expand shared rel_pos_bias if present in checkpoint
            if (
                getattr(self.model, "use_rel_pos_bias", False)
                and "rel_pos_bias.relative_position_bias_table" in cleaned_state_dict
            ):
                logger.info("Expanding shared rel_pos_bias to each transformer block.")
                num_layers = self.model.get_num_layers() if hasattr(self.model, "get_num_layers") else 12
                rel_pos_bias = cleaned_state_dict["rel_pos_bias.relative_position_bias_table"]
                for i in range(num_layers):
                    cleaned_state_dict[f"blocks.{i}.attn.relative_position_bias_table"] = rel_pos_bias.clone()
                cleaned_state_dict.pop("rel_pos_bias.relative_position_bias_table")

            # Remove head keys if shape mismatch (for fine-tuning on new class count)
            model_sd = self.model.state_dict()
            for k in ["head.weight", "head.bias"]:
                if k in cleaned_state_dict and (k not in model_sd or cleaned_state_dict[k].shape != model_sd[k].shape):
                    logger.info(f"Removing key {k} with shape mismatch from pretrained checkpoint")
                    cleaned_state_dict.pop(k)

            missing, unexpected = self.model.load_state_dict(cleaned_state_dict, strict=False)
            logger.info(f"Loaded checkpoint. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            backbone_missing = [k for k in missing if not k.startswith("head.")]
            if len(backbone_missing) == 0 and len(unexpected) == 0:
                logger.info("VERIFIED: Zero missing backbone keys and zero unexpected keys in loaded PanDerm state_dict.")
            else:
                if backbone_missing:
                    logger.warning(f"Backbone missing keys: {backbone_missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys: {unexpected}")
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

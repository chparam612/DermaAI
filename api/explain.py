"""Attention Rollout explainability module for PanDerm VisionTransformer."""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.panderm import PanDermClassifier


class AttentionRollout:
    """Computes Attention Rollout (Abnar & Zuidema, 2020) for PanDerm VisionTransformer.

    Captures per-layer multi-head self-attention maps non-invasively at inference time
    without modifying underlying weights or external/panderm library files.
    Accounts for residual skip connections and layer-scale blocks.
    """

    def __init__(self, model: PanDermClassifier):
        """Initialize AttentionRollout adapter.

        Args:
            model: PanDermClassifier adapter model.
        """
        self.model = model
        self.blocks = model.model.blocks
        self.num_layers = len(self.blocks)
        self.captured_attentions: List[torch.Tensor] = []
        self._original_forwards = {}

    def _patch_attention_forward(self) -> None:
        """Dynamically intercept self-attention forward pass to record post-softmax attention."""
        self.captured_attentions.clear()
        self._original_forwards.clear()

        for idx, block in enumerate(self.blocks):
            attn_mod = block.attn
            self._original_forwards[idx] = attn_mod.forward

            def make_patched(original_mod: nn.Module):
                def patched_forward(x: torch.Tensor, rel_pos_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
                    B, N, C = x.shape
                    qkv_bias = None
                    if original_mod.q_bias is not None:
                        qkv_bias = torch.cat(
                            (
                                original_mod.q_bias,
                                torch.zeros_like(original_mod.v_bias, requires_grad=False),
                                original_mod.v_bias,
                            )
                        )
                    qkv = F.linear(input=x, weight=original_mod.qkv.weight, bias=qkv_bias)
                    qkv = qkv.reshape(B, N, 3, original_mod.num_heads, -1).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]

                    q = q * original_mod.scale
                    attn = q @ k.transpose(-2, -1)

                    if original_mod.relative_position_bias_table is not None:
                        relative_position_bias = (
                            original_mod.relative_position_bias_table[
                                original_mod.relative_position_index.view(-1)
                            ]
                            .view(
                                original_mod.window_size[0] * original_mod.window_size[1] + 1,
                                original_mod.window_size[0] * original_mod.window_size[1] + 1,
                                -1,
                            )
                            .permute(2, 0, 1)
                            .contiguous()
                        )
                        attn = attn + relative_position_bias.unsqueeze(0)

                    if rel_pos_bias is not None:
                        attn = attn + rel_pos_bias

                    attn = attn.softmax(dim=-1)
                    # Record attention matrix: shape (B, num_heads, N, N)
                    self.captured_attentions.append(attn.detach().cpu())
                    attn = original_mod.attn_drop(attn)

                    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
                    x = original_mod.proj(x)
                    x = original_mod.proj_drop(x)
                    return x

                return patched_forward

            attn_mod.forward = make_patched(attn_mod)

    def _restore_attention_forward(self) -> None:
        """Restore original forward methods on all attention blocks."""
        for idx, block in enumerate(self.blocks):
            if idx in self._original_forwards:
                block.attn.forward = self._original_forwards[idx]
        self._original_forwards.clear()

    @torch.no_grad()
    def compute_raw_rollout(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute the 2D spatial attention rollout map for a single input image.

        Args:
            input_tensor: Preprocessed image tensor of shape (1, 3, 224, 224).

        Returns:
            Normalized 2D float32 numpy array of shape (14, 14) with values in [0, 1].
        """
        self._patch_attention_forward()
        try:
            self.model.eval()
            _ = self.model(input_tensor)
        finally:
            self._restore_attention_forward()

        if len(self.captured_attentions) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} attention maps, captured {len(self.captured_attentions)}"
            )

        # Average attention across heads for each layer
        # captured_attentions contains tensors of shape (1, num_heads, N, N) where N=197
        layers = [a.squeeze(0).mean(dim=0).float() for a in self.captured_attentions]
        num_tokens = layers[0].size(0)  # 197
        identity = torch.eye(num_tokens, dtype=torch.float32)

        # Abnar & Zuidema (2020): A_hat = 0.5 * A + 0.5 * I
        rollout_matrix = identity.clone()
        for layer_attn in layers:
            a_hat = 0.5 * layer_attn + 0.5 * identity
            a_hat = a_hat / a_hat.sum(dim=-1, keepdim=True)
            rollout_matrix = torch.matmul(a_hat, rollout_matrix)

        # Attention from CLS token (token 0) to all 196 spatial patch tokens (tokens 1..196)
        cls_rollout = rollout_matrix[0, 1:].numpy()  # Shape (196,)

        # Reshape to 14x14 grid
        grid_size = int(np.sqrt(len(cls_rollout)))  # 14
        heatmap_2d = cls_rollout.reshape(grid_size, grid_size).astype(np.float32)

        # Normalize to [0, 1]
        h_min, h_max = heatmap_2d.min(), heatmap_2d.max()
        if h_max > h_min:
            heatmap_norm = (heatmap_2d - h_min) / (h_max - h_min)
        else:
            heatmap_norm = np.zeros_like(heatmap_2d)

        return heatmap_norm

    def generate_overlay_data_uri(
        self,
        input_tensor: torch.Tensor,
        target_size: Tuple[int, int] = (224, 224),
        colormap_name: str = "jet",
    ) -> str:
        """Generate RGBA PNG base64 data URI overlay with transparent background.

        Args:
            input_tensor: Preprocessed tensor of shape (1, 3, 224, 224).
            target_size: (width, height) to resize the final overlay image.
            colormap_name: Matplotlib colormap name.

        Returns:
            Data URI string: 'data:image/png;base64,...'
        """
        heatmap_14 = self.compute_raw_rollout(input_tensor)

        # Upsample to target size using PIL Bicubic interpolation
        heatmap_img = Image.fromarray(heatmap_14).resize(target_size, resample=Image.BICUBIC)
        heatmap_resized = np.asarray(heatmap_img, dtype=np.float32)
        # Ensure values stay in [0, 1] after bicubic overshoot
        heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0)

        # Apply colormap to get RGB
        cmap = matplotlib.colormaps.get_cmap(colormap_name)
        rgba = cmap(heatmap_resized)  # Shape (H, W, 4), values in [0, 1]

        # Transparent background where attention is low
        # Alpha is proportional to attention intensity: low attention -> transparent
        alpha = np.clip(heatmap_resized * 0.75, 0.0, 0.75)
        rgba[..., 3] = alpha

        rgba_uint8 = (rgba * 255.0).astype(np.uint8)
        overlay_pil = Image.fromarray(rgba_uint8, mode="RGBA")

        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64_str}"


def blend_heatmap_on_image(
    original_image: Image.Image,
    heatmap_2d: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """Helper to render visual blended comparison (original image + colormapped heatmap).

    Args:
        original_image: Input PIL Image (RGB).
        heatmap_2d: 2D float array in [0, 1].
        alpha: Blending weight for heatmap.
        colormap_name: Matplotlib colormap name.

    Returns:
        Blended PIL Image in RGB format.
    """
    orig_rgb = original_image.convert("RGB")
    w, h = orig_rgb.size
    heatmap_pil = Image.fromarray(heatmap_2d).resize((w, h), resample=Image.BICUBIC)
    heatmap_norm = np.clip(np.asarray(heatmap_pil, dtype=np.float32), 0.0, 1.0)

    cmap = matplotlib.colormaps.get_cmap(colormap_name)
    heat_rgb = (cmap(heatmap_norm)[..., :3] * 255.0).astype(np.uint8)
    heat_pil = Image.fromarray(heat_rgb, mode="RGB")

    return Image.blend(orig_rgb, heat_pil, alpha=alpha)

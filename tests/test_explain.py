"""Tests for Attention Rollout explainability module."""
from __future__ import annotations

import base64
import io
from pathlib import Path
import pytest
from PIL import Image
import torch
import numpy as np

from api.explain import AttentionRollout, blend_heatmap_on_image
from src.models.panderm import PanDermClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

@pytest.fixture(scope="module")
def demo_model():
    ckpt_path = PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"
    if not ckpt_path.is_file():
        pytest.skip("demo_model.pt not present")
    model = PanDermClassifier(architecture="panderm_base", num_classes=11, pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def test_rollout_output_shape_and_bounds(demo_model):
    rollout = AttentionRollout(demo_model)
    x = torch.randn(1, 3, 224, 224)
    heatmap = rollout.compute_raw_rollout(x)
    assert heatmap.shape == (14, 14)
    assert isinstance(heatmap, np.ndarray)
    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0

def test_rollout_determinism(demo_model):
    rollout = AttentionRollout(demo_model)
    x = torch.randn(1, 3, 224, 224)
    hm1 = rollout.compute_raw_rollout(x)
    hm2 = rollout.compute_raw_rollout(x)
    assert np.allclose(hm1, hm2, atol=1e-6)

def test_rollout_data_uri_rgba(demo_model):
    rollout = AttentionRollout(demo_model)
    x = torch.randn(1, 3, 224, 224)
    target_size = (300, 200)  # (w, h)
    data_uri = rollout.generate_overlay_data_uri(x, target_size=target_size)
    assert data_uri.startswith("data:image/png;base64,")
    b64_data = data_uri.split(",", 1)[1]
    decoded = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(decoded))
    assert img.size == target_size
    assert img.mode == "RGBA"

def test_blend_helper():
    orig = Image.new("RGB", (100, 100), color=(200, 100, 50))
    heatmap = np.ones((14, 14), dtype=np.float32) * 0.5
    blended = blend_heatmap_on_image(orig, heatmap, alpha=0.5)
    assert blended.size == (100, 100)
    assert blended.mode == "RGB"

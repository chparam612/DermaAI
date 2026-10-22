"""Test demo model checkpoint loading and parity smoke test."""
from __future__ import annotations

from pathlib import Path
import pytest
import torch

from src.data.transforms import get_panderm_eval_transforms
from src.models.panderm import PanDermClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_demo_model_checkpoint_weights_only_load():
    """Verify demo_model.pt exists and loads strictly with weights_only=True on CPU."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / "final" / "demo_model.pt"
    if not ckpt_path.is_file():
        pytest.skip("checkpoints/final/demo_model.pt not found on disk")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 100

    model = PanDermClassifier(architecture="panderm_base", num_classes=11, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0
    assert len(unexpected) == 0

    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
    assert logits.shape == (1, 11)
    probs = torch.softmax(logits, dim=-1)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_eval_transforms_panderm():
    """Verify official PanDerm evaluation transform shape and type."""
    from PIL import Image
    transform = get_panderm_eval_transforms(image_size=224)
    img = Image.new("RGB", (300, 400), color=(128, 128, 128))
    tensor = transform(img)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32

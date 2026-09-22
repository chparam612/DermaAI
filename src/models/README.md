# Models Module (`src/models`)

This module houses model architectures, backbone wrappers, and foundation model adapters.

## Foundation Model Integrations

In Phase 1+, adapters will be integrated here for:
1. **PanDerm**: A multimodal vision foundation model for clinical dermatology (Nature Medicine 2025), located at `external/panderm`.
2. **DermFM-Zero**: A zero-shot vision-language foundation model for clinical dermatology (Feb 2026), located at `external/dermfm-zero`.
3. **Standard CNN / Transformer Baselines**: ResNet-18, ResNet-50, and ViT models for benchmarking.

## Phase 0 Factory Stub

```python
from src.models import build_model

# Instantiate a baseline model for pipeline testing
model = build_model("baseline_resnet18", num_classes=2, pretrained=True)
```

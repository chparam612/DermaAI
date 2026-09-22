# External Research Foundation Models (`external/`)

This directory contains external foundation model repositories integrated as Git submodules for medical dermatology experimentation.

## Git Submodule Management

To initialize or update the submodules after cloning this repository:

```bash
git submodule update --init --recursive
```

---

## 1. PanDerm

- **Repository**: [https://github.com/SiyuanYan1/PanDerm](https://github.com/SiyuanYan1/PanDerm)
- **Local Path**: `external/panderm`
- **Citation**: *Nature Medicine* (2025). "A Multimodal Vision Foundation Model for Clinical Dermatology".
- **Authors**: AIM for Health Lab (Monash University) & Collaborators.
- **License**: Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0).
- **Core Architecture**: Vision Transformer (ViT-B/16 and ViT-L/16) pretrained with self-supervised masked latent modelling on 2+ million multi-institutional clinical dermatology images across 4 modalities (clinical photos, dermoscopy, total body photography, and dermatopathology).
- **Key Dependencies**:
  - Python: `>=3.10` (tested upstream on Python 3.10)
  - PyTorch: `2.4.1` with CUDA 11.8/12.1
  - Packages: `timm==0.9.16`, `h5py`, `pandas`, `PyYAML`, `opencv-python`, `scikit-learn`, `scipy`, `tqdm`, `wandb`
- **Pretrained Checkpoint Sources**:
  - `DermLIP_PanDerm` (ViT-B/16): Hugging Face (`redlessone/DermLIP_PanDerm-base-w-PubMed-256`)
  - `PanDerm_Base` (ViT-B/16): Google Drive
  - `PanDerm` (ViT-L/16): Google Drive

---

## 2. DermFM-Zero

- **Repository**: [https://github.com/SiyuanYan1/DermFM-Zero](https://github.com/SiyuanYan1/DermFM-Zero)
- **Local Path**: `external/dermfm-zero`
- **Citation**: arXiv:2602.10624 (Feb 2026). "DermFM-Zero: A Vision-Language Foundation Model for Dermatology".
- **Authors**: AIM for Health Lab & Collaborators.
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- **Core Architecture**: Multimodal vision-language foundation model supporting zero-shot diagnosis across 400+ skin conditions, cross-modal retrieval, sparse autoencoders (SAE) for concept discovery, and native resolution inputs.
- **Key Dependencies**:
  - Python: `>=3.9`
  - PyTorch: `>=1.9.0` (compatible with PyTorch 2.x)
  - Packages: `torchvision`, `transformers`, `huggingface_hub`, `safetensors`, `timm`, `scikit-learn`, `scipy`, `statsmodels`, `webdataset`, `open_clip`, `jaxtyping`, `pydantic`, `einops`
- **Pretrained Checkpoint Sources**:
  - `DermFM-Zero` / `DermFM-Zero-Open`: Hugging Face (`redlessone/DermFM-Zero`)

---

## Compatibility Notes

1. **Python Environment**: Both submodules are fully compatible with Python 3.10 and PyTorch 2.1 - 2.4. Python 3.10 is the designated conda environment version specified in `environment.yml`.
2. **Submodule Integrity**: Upstream submodules are tracked at verified commit hashes. Never directly modify files inside `external/panderm` or `external/dermfm-zero`; wrap functionality through adapters located in `src/models/`.

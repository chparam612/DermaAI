# Official PanDerm Fine-Tuning Reference & Architectural Mapping

## 1. Overview and Source Attribution

This document records the official PanDerm fine-tuning implementation located in `external/panderm/` and details the architectural design and intentional adaptations implemented in `SkinCancerTracker` Phase 2 (`Exp-2`).

- **Official Source Repository**: PanDerm (`external/panderm`)
- **Official Classification Entrypoint**: `external/panderm/classification/run_class_finetuning.py`
- **Official Model Definitions**: `external/panderm/classification/models/modeling_finetune.py`
- **Encoder Builder**: `external/panderm/classification/panderm_model/get_encoder/get_encoder.py`
- **Furnace Training Utilities**: `external/panderm/classification/furnace/`

---

## 2. Official PanDerm Pipeline Specification

### 2.1 Architecture
The official PanDerm classification architecture is built upon a Vision Transformer (ViT) backbone:
- **Base Architecture (`PanDerm_Base_FT`)**:
  - `VisionTransformer` (`patch_size=16`, `embed_dim=768`, `depth=12`, `num_heads=12`, `mlp_ratio=4`, `qkv_bias=True`)
  - Relative Position Bias (`use_rel_pos_bias=True`)
  - Sinusoidal Position Embedding (`sin_pos_emb=True`)
  - LayerScale (`init_values=0.1` for Base, `1e-5` for Large)
  - Mean Pooling representation (`use_mean_pooling=True`)
  - Drop path rate: `0.1`
- **Large Architecture (`PanDerm_Large_FT`)**:
  - `patch_size=16`, `embed_dim=1024`, `depth=24`, `num_heads=16`, `mlp_ratio=4`
- **Classification Head**:
  - Linear projection layer: `nn.Linear(embed_dim, num_classes)`

### 2.2 Input Dimensions & Normalization
- **Resolution**: `224 x 224` pixels (`patch_size = 16`, patch grid = `14 x 14`)
- **Channel Normalization**:
  - Mean: `[0.485, 0.456, 0.406]`
  - Standard Deviation: `[0.228, 0.224, 0.225]` *(Note: green channel std is `0.228`, differing slightly from standard ImageNet `0.229`)*

### 2.3 Official Augmentation Pipeline
- **Training**:
  1. `Resize(256)`
  2. `RandomResizedCrop(224, scale=(0.75, 1.0))`
  3. `RandomHorizontalFlip()`
  4. `RandomVerticalFlip()`
  5. `RandomRotation(degrees=45)`
  6. `ColorJitter(hue=0.2)`
  7. `ToTensor()`
  8. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])`
- **Validation / Evaluation**:
  1. `Resize(256)`
  2. `CenterCrop(224)`
  3. `ToTensor()`
  4. `Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])`

### 2.4 Optimization & Loss
- **Optimizer**: `AdamW` (`lr=5e-4`, `weight_decay=0.05`, `betas=(0.9, 0.999)`, `eps=1e-8`)
- **Learning Rate Schedule**: Cosine annealing with warmup (`warmup_epochs=5`, `warmup_lr=1e-6`, `min_lr=1e-6`)
- **Layer-wise LR Decay**: `0.9` (via `LayerDecayValueAssigner`)
- **Loss**: `LabelSmoothingCrossEntropy(smoothing=0.1)` or `CrossEntropyLoss`

---

## 3. Deviations & Phase 2 Adaptations

In integrating PanDerm into the `SkinCancerTracker` benchmark framework, the following intentional adaptations are made to ensure rigorous comparative integrity with `Exp-1` (CNN Baseline):

1. **Lesion-Level Split Integrity**:
   - The official PanDerm script accepted external CSVs but did not enforce strict lesion grouping or check for image-level cross-split leakage.
   - **Adaptation**: Both `Exp-1` and `Exp-2` strictly utilize `data/splits/lesion_level_splits.csv` derived from Phase 1, guaranteeing zero lesion leakage across train/validation/test sets.

2. **Fairness / Skin-Tone Metadata Tracking**:
   - The official PanDerm evaluation reported macro metrics without demographic subpopulation disaggregation.
   - **Adaptation**: Dataset and evaluation pipelines preserve ground-truth `skin_tone_class` metadata and compute stratified metrics per skin-tone subgroup to audit algorithmic fairness.

3. **Modular Adapter Pattern (`PanDermClassifier`)**:
   - Rather than executing monolithic standalone training scripts, `src/models/panderm.py` wraps `panderm_base_patch16_224_finetune` in a standard PyTorch module conforming to the unified `SkinCancerTracker` model interface.
   - Seamlessly supports loading official `.pth` / `.bin` checkpoints when present, while offering safe initialization and descriptive warnings when running in local CPU/smoke-test environments.

4. **W&B and Experiment Tracking**:
   - Replaced ad-hoc prints with structured JSON artifacts, CSV prediction dumps with calibrated class probabilities, and automatic offline/online Weights & Biases logging via `src/eval/wandb_logger.py`.

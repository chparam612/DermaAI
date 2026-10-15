# Phase 2 — Baseline Experiments: Exp-1 & Exp-2 (v1 & v2 Close-Out)

**Project**: Multimodal Dermatology / Skin-Lesion Classification (MILK10k Dataset)  
**Authors**: Antigravity Research Team  
**Date**: October 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 2 establishes the reference baseline benchmarks for multimodal dermatology classification on the **MILK10k** dataset, strictly isolating the **dermoscopic image modality** to provide a controlled reference for subsequent multimodal fusion and imbalance-mitigation experiments:

1. **Exp-1: ResNet-50 CNN Baseline** (`exp1-resnet50-baseline`)  
   - ImageNet-pretrained standard convolutional architecture (25.5M parameters).
   - **Status**: 1 epoch, local, not comparable. Trained for only 1 epoch (91.65s) on local RTX 2050 as an engineering smoke test. Standalone Kaggle 30-epoch training script packaged under `kaggle/run_exp1_resnet50.py`. **Confound Warning**: Exp-1 differs from Exp-2 in training recipe (class-balanced CE, unweighted sampler, no layer-wise LR decay, standard AdamW schedule) as well as backbone architecture (ResNet-50 vs ViT-Base). Any performance difference between Exp-1 and Exp-2 must NOT be described as a pure "backbone effect".
2. **Exp-2 (v1): PanDerm ViT-Base Initial Fine-Tune** (`exp2-panderm-finetune`)  
   - Dermatology domain foundation model (Nature Medicine 2025, 86M parameters).
   - Standard fine-tuning recipe with unweighted sampler and step LR schedule.
   - Result: Severe majority-class collapse onto BCC and NV (Macro-F1: 0.1091).
3. **Exp-2 (v2): PanDerm ViT-Base Corrected 30-Epoch Run** (`exp2-panderm-finetune-v2`)  
   - Official PanDerm ViT architecture with layer-scale (`gamma_1`), 12 blocks, and bicubic relative position bias interpolation.
   - Five root-cause improvements: weighted random sampler, layer-wise LR decay (0.9), key remapping for official ViT blocks, `val_macro_f1` checkpointing, and 30-epoch cosine schedule with 5-epoch warmup.
   - Model selection: **Strictly selected on VALIDATION metrics only** (`val_macro_f1` at Epoch 26 = 0.4064).
   - Hardware: Trained on Kaggle GPU (Tesla T4, 16GB VRAM) in 29m 20s (1,760.5s).
   - Test Result (evaluated once on frozen model): Top-1 Accuracy 64.38%, Balanced Accuracy 39.34%, Macro-F1 0.3925, Malignant Recall 89.02%, Malignant AUROC 0.8022.

---

## 2. Dataset Partitioning & Modality Contract

- **Benchmark Dataset**: MILK10k (10,480 images across 5,240 unique lesions).
- **Partition Contract**: Enforces strict lesion-level disjointness ($0\%$ lesion leakage across splits).
- **Split File**: `data/splits/lesion_level_splits.csv`
- **Split SHA-256 Hash**: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`
- **Modality Isolation**: Strictly filtered to `image_type == 'dermoscopic'`. Each lesion in MILK10k possesses exactly 1 dermoscopic image and 1 clinical close-up image. Exactly **ONE image per lesion** (the dermoscopic image) is used in Phase 2:
  - **Train**: 3,668 lesions / images (70.0% of 5,240)
  - **Validation**: 786 lesions / images (15.0% of 5,240)
  - **Test**: 786 lesions / images (15.0% of 5,240)
  - **Total**: 5,240 dermoscopic images
- **Taxonomy (11 standardized diagnostic classes)**:  
  `["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]`
- **Clinical Grouping**:
  - **Malignant / Pre-malignant (5 classes)**: `["BCC", "MEL", "SCCKA", "AKIEC", "MAL_OTH"]` (583 / 786 test lesions = 74.17%)
  - **Benign (6 classes)**: `["NV", "BKL", "DF", "INF", "BEN_OTH", "VASC"]` (203 / 786 test lesions = 25.83%)

---

## 3. Ablation Results Table

Evaluated on the MILK10k held-out test split ($N = 786$, zero lesion leakage, identical label mapping):

| Experiment | Architecture | Modality | Top-1 Accuracy | Balanced Accuracy | Macro-F1 | Weighted F1 | Malignant Recall | Malignant AUROC | Epochs | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp-1** | ResNet-50 | Dermoscopic | 0.5191 | 0.3338 | 0.2869 | 0.5398 | 0.8868 | 0.7549 | 1 (ref) | **1 epoch, local, not comparable** (RTX 2050 91s reference; full 30-epoch Kaggle run pending) |
| **Exp-2 (v1)** | PanDerm ViT-B | Dermoscopic | 0.5344 | 0.2185 | 0.1091 | 0.4497 | 0.8165 | 0.6699 | 10 | v1 recipe (unweighted sampler, majority collapse onto BCC/NV; val-acc monitoring) |
| **Exp-2 (v2)** | PanDerm ViT-B | Dermoscopic | **0.6438** | **0.3934** | **0.3925** | **0.6469** | **0.8902** | **0.8022** | **30** | **v2 corrected** (weighted sampler, layer_decay=0.9, val_macro_f1 checkpointing; **selection on val only**) |

*Artifact file paths*:
- CSV: [`results/ablation_table.csv`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/ResearchProject/SkinCancerTracker/results/ablation_table.csv)
- Markdown: [`results/ablation_table.md`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/ResearchProject/SkinCancerTracker/results/ablation_table.md)

> [!IMPORTANT]
> **No Backbone Comparison Claim**: The Exp-1 ResNet-50 row is strictly labeled *1 epoch, local, not comparable*. It served as an initial local pipeline integration test and does not represent a fully converged CNN baseline. No superiority or architectural claim between PanDerm ViT-Base and ResNet-50 is made until the 30-epoch Kaggle Exp-1 run is executed under identical training schedules.

---

## 4. Methodological Selection: Why Exp-2 v2 Was Selected Over v1

Exp-2 (v2) was selected to replace Exp-2 (v1) based on five critical architectural and methodological corrections:

1. **Batch Sampling**: Exp-2 (v1) used uniform random shuffling, allowing the dominant BCC class (48.1% of training data) to dominate individual mini-batches and starve tail classes. Exp-2 (v2) introduced a `WeightedRandomSampler` with inverse-class-frequency sampling probabilities ($w_c \propto 1/N_c$), ensuring rare classes are consistently represented across optimization steps.
2. **Layer-Wise Learning Rate Decay**: Exp-2 (v1) applied a single global learning rate across all 12 transformer blocks. Exp-2 (v2) implemented official BEiT/PanDerm layer-wise LR decay (`layer_decay = 0.9`), scaling earlier transformer blocks with lower learning rates ($\eta_l = \eta_0 \cdot 0.9^{12 - l}$) to preserve self-supervised medical feature representations while adapting classification layers.
3. **Official Architecture & Checkpoint Alignment**: Exp-2 (v1) inadvertently fell back to generic `timm` ViT when loading `panderm_base.pth`, resulting in 16 missing keys and 50 unexpected keys. Exp-2 (v2) instantiated the official `models.modeling_finetune.VisionTransformer` with 12 layer-scale parameters (`blocks.*.gamma_1`), bicubic relative position bias interpolation, and verified zero missing backbone keys.
4. **Validation-Only Model Selection Contract**:
   - In Exp-2 (v1), checkpoints were saved by tracking validation loss or accuracy, which favored majority-class overprediction and masked tail-class collapse.
   - In Exp-2 (v2), **model selection strictly used VALIDATION metrics only**: the `CheckpointManager` monitored `val_macro_f1` on the 786-lesion validation set across 30 epochs.
   - Checkpoint selection occurred at **Epoch 26** because it attained the global peak validation Macro-F1 (`val_macro_f1 = 0.4064`, 10/11 classes predicted).
   - **The held-out test set ($N=786$) was never touched during training or hyperparameter tuning**, and was evaluated exactly **once** on the frozen Epoch 26 checkpoint.
5. **Restored Optimization Horizon**: Expanded training from 10 epochs (v1) to 30 epochs with a 5-epoch linear warmup followed by cosine annealing to $\eta_{\min} = 10^{-6}$.

---

## 5. Exp-2 (v2) Detailed Per-Class Evaluation (N = 786)

| Class | Category | Test Support | Predicted Count | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | Malignant | 54 | 55 | 0.3455 | 0.3519 | 0.3486 |
| **BCC** | Malignant | 391 | 354 | 0.8305 | 0.7519 | 0.7893 |
| **BEN_OTH** | Benign | 8 | 5 | 0.2000 | 0.1250 | 0.1538 |
| **BKL** | Benign | 80 | 66 | 0.4242 | 0.3500 | 0.3836 |
| **DF** | Benign | 5 | 2 | 0.5000 | 0.2000 | 0.2857 |
| **INF** | Benign | 5 | 7 | 0.0000 | 0.0000 | 0.0000 |
| **MAL_OTH** | Malignant | 2 | 0 | 0.0000 | 0.0000 | 0.0000 |
| **MEL** | Malignant | 67 | 93 | 0.4946 | 0.6866 | 0.5750 |
| **NV** | Benign | 99 | 93 | 0.7097 | 0.6667 | 0.6875 |
| **SCCKA** | Malignant | 69 | 106 | 0.4528 | 0.6957 | 0.5486 |
| **VASC** | Benign | 6 | 5 | 0.6000 | 0.5000 | 0.5455 |

### 11x11 Confusion Matrix (Rows = Ground Truth, Columns = Predicted)

```
True \ Pred |   AKIEC     BCC BEN_OTH     BKL      DF     INF MAL_OTH     MEL      NV   SCCKA    VASC
-----------------------------------------------------------------------------------------------------
AKIEC       |      19      12       0       8       0       1       0       2       2       9       1
BCC         |      19     294       1      15       1       5       0      16       6      34       0
BEN_OTH     |       0       1       1       2       0       0       0       3       1       0       0
BKL         |       7      19       1      28       0       0       0       6       5      14       0
DF          |       0       4       0       0       1       0       0       0       0       0       0
INF         |       0       4       0       0       0       0       0       0       1       0       0
MAL_OTH     |       0       0       0       0       0       0       0       0       1       0       1
MEL         |       1       4       0       5       0       0       0      46      11       0       0
NV          |       2       9       1       3       0       0       0      17      66       1       0
SCCKA       |       7       5       1       5       0       0       0       3       0      48       0
VASC        |       0       2       0       0       0       1       0       0       0       0       3
```

---

## 6. Binary Malignant vs. Benign Safety Audit

In clinical triage, the highest-cost failure mode is misclassifying a malignant lesion as benign (a **false negative**).

- **Total Test Lesions**: 786
- **True Malignant**: 583 ($74.17\%$)
- **True Benign**: 203 ($25.83\%$)
- **True Positives (Malignant $\rightarrow$ Malignant)**: 519
- **False Negatives (Malignant $\rightarrow$ Benign)**: **64** ($10.98\%$ Malignant False Negative Rate)
- **False Positives (Benign $\rightarrow$ Malignant)**: 89
- **True Negatives (Benign $\rightarrow$ Benign)**: 114
- **Malignant Sensitivity (Recall)**: **0.8902** ($89.02\%$)
- **Malignant Specificity**: **0.5616** ($56.16\%$)
- **Malignant Precision**: **0.8536** ($85.36\%$)
- **Malignant FNR ($1 - \text{Sens}$)**: **0.1098** ($10.98\%$)

### Breakdown of the 64 Critical False Negatives:
1. **BCC misclassified as Benign (28 cases)**: BKL (15), NV (6), INF (5), BEN_OTH (1), DF (1).
2. **MEL misclassified as Benign (16 cases)**: NV (11), BKL (5).
3. **AKIEC misclassified as Benign (12 cases)**: BKL (8), NV (2), INF (1), VASC (1).
4. **SCCKA misclassified as Benign (6 cases)**: BKL (5), BEN_OTH (1).
5. **MAL_OTH misclassified as Benign (2 cases)**: NV (1), VASC (1).

*Observation*: Benign keratosis (BKL) and melanocytic nevus (NV) act as the primary attractor states for missed malignancies, accounting for $58 / 64$ ($90.6\%$) of all critical false negatives.

---

## 7. Subgroup Fairness & Disaggregated Skin-Tone Audit

Fairness is evaluated across **3 Fitzpatrick skin-tone buckets** and all **5 individual skin-tone levels** present in the test partition. Subgroups with $n < 30$ lesions are explicitly flagged **LOW-CONFIDENCE**. Bootstrap 95% confidence intervals are computed over 2,000 resamples.

### 3-Bucket Fairness Audit

| Skin-Tone Bucket | Sample Count ($N$) | Low-Confidence Flag | Top-1 Accuracy | Balanced Accuracy | Macro-F1 (95% CI) | Malignant FNR (95% CI) | Malignant Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dark (Tones 0–1)** | 12 | **TRUE ($n < 30$)** | 0.5833 | 0.5833 | 0.4626 [0.241, 0.738] | 0.1667 [0.000, 0.500] | 0.8333 |
| **Medium (Tones 2–3)** | 558 | FALSE | 0.6254 | 0.3410 | 0.3401 [0.278, 0.415] | 0.1060 [0.076, 0.137] | 0.8940 |
| **Light (Tones 4–5)** | 216 | FALSE | 0.6944 | 0.5493 | 0.5032 [0.356, 0.591] | 0.1173 [0.071, 0.172] | 0.8827 |

- **Statistical Significance of Dark-Bucket FNR Gap**:
  While the point estimate for Dark-bucket Malignant FNR is higher ($16.67\%$) compared to Medium ($10.60\%$) and Light ($11.73\%$), **this gap is not statistically supportable**. With only $n = 12$ total test lesions in the Dark bucket, the bootstrap 95% confidence interval is extremely wide ($[0.000, 0.500]$) and heavily overlaps with both the Medium bucket ($[0.076, 0.137]$) and Light bucket ($[0.071, 0.172]$). No statistically significant disparity in Malignant FNR across skin-tone buckets can be claimed on this test split.

### 5 Individual Skin-Tones Audit

| Tone Level | Sample Count ($N$) | Low-Confidence Flag | Top-1 Accuracy | Balanced Accuracy | Macro-F1 (95% CI) | Malignant FNR (95% CI) | Malignant Sensitivity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tone 1** | 12 | **TRUE ($n < 30$)** | 0.5833 | 0.5833 | 0.4626 [0.250, 0.705] | 0.1667 [0.000, 0.500] | 0.8333 |
| **Tone 2** | 72 | FALSE | 0.6389 | 0.6517 | 0.5206 [0.323, 0.669] | 0.1351 [0.031, 0.250] | 0.8649 |
| **Tone 3** | 486 | FALSE | 0.6235 | 0.2823 | 0.2718 [0.242, 0.322] | 0.1032 [0.075, 0.135] | 0.8968 |
| **Tone 4** | 153 | FALSE | 0.7059 | 0.4914 | 0.4366 [0.341, 0.593] | 0.1161 [0.061, 0.175] | 0.8839 |
| **Tone 5** | 63 | FALSE | 0.6667 | 0.5916 | 0.4949 [0.322, 0.610] | 0.1200 [0.039, 0.213] | 0.8800 |

---

## 8. Dissecting Tone 3 (The Largest Subgroup, N = 486)

Tone 3 accounts for **61.84%** of the entire test partition (486 of 786 lesions). While Tone 3 achieves solid raw accuracy ($62.35\%$), its Balanced Accuracy is deeply depressed ($28.23\%$) and Macro-F1 is low ($0.2718$ vs $0.5206$ in Tone 2 and $0.5032$ in Light tones).

Disaggregating performance within Tone 3 vs Non-Tone 3 reveals factors **associated with** this performance profile. Bootstrap 95% confidence intervals (2,000 resamples) are provided for each recall estimate and for the pairwise difference ($\text{Recall}_{\text{Tone 3}} - \text{Recall}_{\text{Non-Tone 3}}$):

| Class | Tone 3 Supp | Tone 3 Recall [95% CI] | Non-T3 Supp | Non-T3 Recall [95% CI] | Recall Difference [95% CI] | Tone 3 Preval. | Non-T3 Preval. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | 41 | 0.3171 [0.184, 0.469] | 13 | 0.4615 [0.182, 0.750] | -0.1445 [-0.469, +0.183] | 8.44% | 4.33% |
| **BCC** | 260 | **0.7923 [0.740, 0.842]** | 131 | **0.6718 [0.589, 0.748]** | **+0.1206 [+0.034, +0.219]** | **53.50%** | **43.67%** |
| **BEN_OTH** | 2 | 0.0000 [0.000, 0.000] | 6 | 0.1667 [0.000, 0.500] | -0.1667 [-0.500, +0.000] | 0.41% | 2.00% |
| **BKL** | 52 | **0.2500 [0.140, 0.375]** | 28 | **0.5357 [0.333, 0.727]** | **-0.2857 [-0.518, -0.057]** | 10.70% | 9.33% |
| **DF** | 3 | 0.0000 [0.000, 0.000] | 2 | 0.5000 [0.000, 1.000] | -0.5000 [-1.000, +0.000] | 0.62% | 0.67% |
| **INF** | 1 | 0.0000 [0.000, 0.000] | 4 | 0.0000 [0.000, 0.000] | +0.0000 [+0.000, +0.000] | 0.21% | 1.33% |
| **MAL_OTH** | 2 | 0.0000 [0.000, 0.000] | 0 | N/A | N/A | 0.41% | 0.00% |
| **MEL** | 32 | **0.5625 [0.400, 0.737]** | 35 | **0.8000 [0.655, 0.929]** | **-0.2375 [-0.455, -0.011]** | 6.58% | 11.67% |
| **NV** | 47 | **0.5319 [0.391, 0.676]** | 52 | **0.7885 [0.673, 0.892]** | **-0.2565 [-0.430, -0.068]** | 9.67% | 17.33% |
| **SCCKA** | 43 | 0.6512 [0.500, 0.794] | 26 | 0.7692 [0.588, 0.926] | -0.1181 [-0.345, +0.114] | 8.85% | 8.67% |
| **VASC** | 3 | 0.0000 [0.000, 0.000] | 3 | 1.0000 [1.000, 1.000] | -1.0000 [-1.000, -1.000] | 0.62% | 1.00% |

### Observations:
1. **BCC Dominance**: Tone 3 is heavily concentrated in BCC ($53.50\%$ vs $43.67\%$ in Non-Tone 3). The model exhibits a statistically significant higher recall on BCC in Tone 3 ($+0.1206$ difference, 95% CI $[+0.034, +0.219]$).
2. **Statistically Significant Recall Deficits on Non-BCC Categories**:
   - **BKL**: $-0.2857$ difference (95% CI $[-0.518, -0.057]$).
   - **MEL**: $-0.2375$ difference (95% CI $[-0.455, -0.011]$).
   - **NV**: $-0.2565$ difference (95% CI $[-0.430, -0.068]$).
   All three differences exclude zero, confirming that discrimination across these prevalent non-BCC categories is significantly lower within Tone 3.
3. **Tail-Class Zero-Recall**: All 5 rare classes (`BEN_OTH`, `DF`, `INF`, `MAL_OTH`, `VASC`) had 0 correctly detected lesions in Tone 3. However, because their test support in Tone 3 is tiny ($1$ to $3$ lesions), their confidence intervals are wide or degenerate.

*Interpretation*: The depressed Macro-F1 and Balanced Accuracy in Tone 3 is **associated with** the combination of extreme local class skew towards BCC and significant recall drops on BKL, MEL, and NV, rather than direct optical interference from Fitzpatrick pigmentation alone.

---

## 9. Overfitting Evidence & Training Dynamics

Inspection of the best model checkpoint (`checkpoints/exp2_panderm_v2/best_model.pt`) and training logs reveals noticeable generalization divergence:

- **Best Checkpoint Epoch**: **Epoch 26 of 30** (selected by `val_macro_f1 = 0.4064`).
- **Train Loss**: $0.7589$ vs. **Val Loss**: $1.4394$ ($\Delta = +0.6805$).
- **Train Accuracy**: **88.69%** ($3,253 / 3,668$) vs. **Val Accuracy**: **62.09%** ($488 / 786$) ($\Delta = +26.60\%$).
- **Generalization Plateau**: Despite label smoothing ($\epsilon = 0.1$) and layer-decay ($0.9$), training accuracy climbed to 88.69% while validation loss plateaued after epoch 20 and validation Macro-F1 peaked at epoch 26 before decaying slightly in epochs 27–30.
- **Val vs. Test Alignment**:
  - Validation Accuracy: $62.09\%$ $\rightarrow$ Test Accuracy: $64.38\%$ ($+2.29\%$).
  - Validation Balanced Accuracy: $43.80\%$ $\rightarrow$ Test Balanced Accuracy: $39.34\%$ ($-4.46\%$).
  - Validation Macro-F1: $0.4064$ $\rightarrow$ Test Macro-F1: $0.3925$ ($-1.39\%$).
  - The close agreement between validation and test Macro-F1 confirms that `val_macro_f1` checkpointing reliably selected a generalizable model without overfitting the validation split.

---

## 10. Honest Limitations & Protocol Update

1. **Single Seed Reference**: Exp-2 (v2) was trained on a single random seed (`seed = 42`). Multi-seed runs (e.g. 3 seeds) are required to establish variance bounds for publication.
2. **Internal vs Official Benchmark Set**: Evaluated on MILK10k internal held-out test partition ($N = 786$, zero lesion leakage). This internal benchmark is **not directly comparable** to the published 53.6% top-1 accuracy reported in the PanDerm Nature Medicine paper, which evaluated on external public benchmarks (e.g., ISIC 2018 / 2019 / Derm7pt) under differing label taxonomies.
3. **Tail-Class Test Counts & Metric Reliability Threshold**:
   - In the 786-lesion test set, tail classes possess extremely limited sample counts: `MAL_OTH` ($n = 2$), `DF` ($n = 5$), `INF` ($n = 5$), `VASC` ($n = 6$), and `BEN_OTH` ($n = 8$).
   - A shift of just a single correct prediction in `MAL_OTH` changes its recall by $0.50$ and shifts overall Macro-F1 by $\approx 0.045$.
   - **Protocol Rule**: Macro-F1 differences below $\approx \mathbf{0.05}$ are considered statistically unreliable without confidence intervals. **From Phase 3 onward, every experiment comparison must report a paired bootstrap 95% confidence interval on the difference ($\Delta \text{Macro-F1}$ and $\Delta \text{Malignant FNR}$).**
4. **Dark Skin-Tone Scarcity**: Skin tone level 1 contains only $12$ test lesions ($1.5\%$ of the test set), and skin tone level 0 has zero samples. All metrics reported for the Dark bucket are explicitly flagged **LOW-CONFIDENCE** ($n < 30$).

---

## 11. Reproducibility & Integrity Manifest

- **Splits File**: `data/splits/lesion_level_splits.csv`
- **Splits SHA-256**: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`
- **Label Mapping**: `configs/label_mapping.json` (11 classes)
- **Exp-2 Checkpoint Path**: `checkpoints/exp2_panderm_v2/best_model.pt`
- **Checkpoint Verification**:
  - Keys: `epoch`, `model_state_dict`, `optimizer_state_dict`, `metrics`.
  - Prefix: `model.` on all parameter names.
  - Layer scale: Verified 12 `gamma_1` keys (`model.blocks.0.gamma_1` ... `model.blocks.11.gamma_1`).
  - Head keys: `model.head.weight` and `model.head.bias`.
- **Training Command (Kaggle P100 / T4 GPU)**:
  ```bash
  python scripts/train_exp2_panderm.py \
      --config configs/experiments/exp2_panderm_kaggle.yaml \
      --image-dir /kaggle/input/datasets/paramveerchaudhary/milk10k-dataset/images \
      --splits-file /kaggle/input/datasets/paramveerchaudhary/milk10k-dataset/lesion_level_splits.csv \
      --checkpoint-path /kaggle/input/datasets/paramveerchaudhary/panderm-weights/panderm_base.pth \
      --checkpoint-dir /kaggle/working/checkpoints/exp2_panderm_v2 \
      --results-dir /kaggle/working/results/exp2_panderm_v2 \
      --batch-size 64 \
      --lr 2.5e-4 \
      --grad-accum 1 \
      --num-workers 4 \
      --max-keep 2 \
      --online
  ```
- **Local Analysis Command (CPU)**:
  ```bash
  python scripts/analyze_exp2_results.py
  ```

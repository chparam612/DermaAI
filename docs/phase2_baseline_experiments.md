# Phase 2 — Baseline Experiments: Exp-1 & Exp-2 (v1 & v2 Close-Out)

**Project**: Multimodal Dermatology / Skin-Lesion Classification (MILK10k Dataset)  
**Authors**: Antigravity Research Team  
**Date**: October 2026  
**Status**: Completed & Verified  

---

## 1. Executive Summary

Phase 2 establishes the baseline benchmarks for multimodal dermatology classification on the **MILK10k** dataset, strictly isolating the **dermoscopic image modality** to provide a controlled reference for subsequent multimodal fusion and imbalance-mitigation experiments:

1. **Exp-1: ResNet-50 CNN Baseline** (`exp1-resnet50-baseline`)  
   - ImageNet-pretrained standard convolutional architecture (25.5M parameters).
   - Class-balanced Cross-Entropy loss ($w_c = \frac{N}{C \cdot N_c}$, normalized).
   - Reference local run (RTX 2050): 1 epoch sanity check (91.65s). Standalone Kaggle 30-epoch training script packaged under `kaggle/run_exp1_resnet50.py`.
2. **Exp-2 (v1): PanDerm ViT-Base Initial Fine-Tune** (`exp2-panderm-finetune`)  
   - Dermatology domain foundation model (Nature Medicine 2025, 86M parameters).
   - Standard fine-tuning recipe with unweighted sampler and step LR schedule.
   - Result: Severe majority-class collapse onto BCC and NV (Macro-F1: 0.1091).
3. **Exp-2 (v2): PanDerm ViT-Base Corrected 30-Epoch Run** (`exp2-panderm-finetune-v2`)  
   - Official PanDerm ViT architecture with layer-scale (`gamma_1`), 12 blocks, and bicubic relative position bias interpolation.
   - Five root-cause improvements: weighted random sampler, layer-wise LR decay (0.9), key remapping for official ViT blocks, `val_macro_f1` checkpointing, and 30-epoch cosine schedule with 5-epoch warmup.
   - Hardware: Trained on Kaggle GPU (Tesla T4, 16GB VRAM) in 29m 20s (1,760.5s).
   - Result: Top-1 Accuracy 64.38%, Balanced Accuracy 39.34%, Macro-F1 0.3925, Malignant Recall 89.02%, Malignant AUROC 0.8022.

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

| Experiment | Architecture | Modality | Top-1 Accuracy | Balanced Accuracy | Macro-F1 | Weighted F1 | Malignant Recall | Malignant AUROC | Epochs | Training Time / Device |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Exp-1** | ResNet-50 | Dermoscopic | 0.5191 | 0.3338 | 0.2869 | 0.5398 | 0.8868 | 0.7549 | 1 (ref) | 1m 32s (RTX 2050 4GB) |
| **Exp-2 (v1)** | PanDerm ViT-B | Dermoscopic | 0.5344 | 0.2185 | 0.1091 | 0.4497 | 0.8165 | 0.6699 | 10 | 11m 10s (RTX 2050 4GB) |
| **Exp-2 (v2)** | PanDerm ViT-B | Dermoscopic | **0.6438** | **0.3934** | **0.3925** | **0.6469** | **0.8902** | **0.8022** | **30** | **29m 20s (Tesla T4 16GB)** |

*Artifact file paths*:
- CSV: [`results/ablation_table.csv`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/ResearchProject/SkinCancerTracker/results/ablation_table.csv)
- Markdown: [`results/ablation_table.md`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/ResearchProject/SkinCancerTracker/results/ablation_table.md)

---

## 4. Exp-2 (v2) Detailed Per-Class Evaluation (N = 786)

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

## 5. Binary Malignant vs. Benign Safety Audit

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

## 6. Subgroup Fairness & Disaggregated Skin-Tone Audit

Fairness is evaluated across **3 Fitzpatrick skin-tone buckets** and all **5 individual skin-tone levels** present in the test partition. Subgroups with $n < 30$ lesions are explicitly flagged **LOW-CONFIDENCE**. Bootstrap 95% confidence intervals are computed over 2,000 resamples.

### 3-Bucket Fairness Audit

| Skin-Tone Bucket | Sample Count ($N$) | Low-Confidence Flag | Top-1 Accuracy | Balanced Accuracy | Macro-F1 (95% CI) | Malignant FNR (95% CI) | Malignant Sensitivity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dark (Tones 0–1)** | 12 | **TRUE ($n < 30$)** | 0.5833 | 0.5833 | 0.4626 [0.241, 0.738] | 0.1667 [0.000, 0.500] | 0.8333 |
| **Medium (Tones 2–3)** | 558 | FALSE | 0.6254 | 0.3410 | 0.3401 [0.278, 0.415] | 0.1060 [0.076, 0.137] | 0.8940 |
| **Light (Tones 4–5)** | 216 | FALSE | 0.6944 | 0.5493 | 0.5032 [0.356, 0.591] | 0.1173 [0.071, 0.172] | 0.8827 |

- **3-Bucket Malignant FNR Disparity ($\max - \min$)**: **6.06%** ($0.1667 - 0.1060$).
- **3-Bucket Balanced Accuracy Disparity ($\max - \min$)**: **24.23%** ($0.5833 - 0.3410$).

### 5 Individual Skin-Tones Audit

| Tone Level | Sample Count ($N$) | Low-Confidence Flag | Top-1 Accuracy | Balanced Accuracy | Macro-F1 (95% CI) | Malignant FNR (95% CI) | Malignant Sensitivity |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tone 1** | 12 | **TRUE ($n < 30$)** | 0.5833 | 0.5833 | 0.4626 [0.250, 0.705] | 0.1667 [0.000, 0.500] | 0.8333 |
| **Tone 2** | 72 | FALSE | 0.6389 | 0.6517 | 0.5206 [0.323, 0.669] | 0.1351 [0.031, 0.250] | 0.8649 |
| **Tone 3** | 486 | FALSE | 0.6235 | 0.2823 | 0.2718 [0.242, 0.322] | 0.1032 [0.075, 0.135] | 0.8968 |
| **Tone 4** | 153 | FALSE | 0.7059 | 0.4914 | 0.4366 [0.341, 0.593] | 0.1161 [0.061, 0.175] | 0.8839 |
| **Tone 5** | 63 | FALSE | 0.6667 | 0.5916 | 0.4949 [0.322, 0.610] | 0.1200 [0.039, 0.213] | 0.8800 |

- **5-Tone Malignant FNR Disparity ($\max - \min$)**: **6.35%** ($0.1667 - 0.1032$).
- **5-Tone Balanced Accuracy Disparity ($\max - \min$)**: **36.95%** ($0.6517 - 0.2823$).

---

## 7. Dissecting Tone 3 (The Largest Subgroup, N = 486)

Tone 3 accounts for **61.84%** of the entire test partition (486 of 786 lesions). While Tone 3 achieves solid raw accuracy ($62.35\%$), its Balanced Accuracy is surprisingly poor ($28.23\%$) and Macro-F1 is deeply depressed ($0.2718$ vs $0.5206$ in Tone 2 and $0.5032$ in Light tones).

Disaggregating performance within Tone 3 vs Non-Tone 3 reveals the root cause:

| Class | Tone 3 Support | Tone 3 Recall | Non-Tone 3 Support | Non-Tone 3 Recall | Recall Difference (T3 - NonT3) | Tone 3 Class Preval. | Non-Tone 3 Preval. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | 41 | 0.3171 | 13 | 0.4615 | -0.1445 | 8.44% | 4.33% |
| **BCC** | 260 | **0.7923** | 131 | 0.6718 | **+0.1206** | **53.50%** | **43.67%** |
| **BEN_OTH** | 2 | **0.0000** | 6 | 0.1667 | -0.1667 | 0.41% | 2.00% |
| **BKL** | 52 | 0.2500 | 28 | 0.5357 | **-0.2857** | 10.70% | 9.33% |
| **DF** | 3 | **0.0000** | 2 | 0.5000 | -0.5000 | 0.62% | 0.67% |
| **INF** | 1 | **0.0000** | 4 | 0.0000 | +0.0000 | 0.21% | 1.33% |
| **MAL_OTH** | 2 | **0.0000** | 0 | N/A | N/A | 0.41% | 0.00% |
| **MEL** | 32 | 0.5625 | 35 | 0.8000 | **-0.2375** | 6.58% | 11.67% |
| **NV** | 47 | 0.5319 | 52 | 0.7885 | **-0.2565** | 9.67% | 17.33% |
| **SCCKA** | 43 | 0.6512 | 26 | 0.7692 | -0.1181 | 8.85% | 8.67% |
| **VASC** | 3 | **0.0000** | 3 | 1.0000 | **-1.0000** | 0.62% | 1.00% |

### Findings:
1. **Extreme BCC Dominance**: More than half of all Tone 3 lesions are BCC ($53.50\%$, 260 / 486). The model exploits this prior, achieving $+12.06\%$ higher recall on BCC in Tone 3 ($79.23\%$) than outside Tone 3 ($67.18\%$).
2. **Total Tail-Class Extinction in Tone 3**: All five rare classes (`BEN_OTH`, `DF`, `INF`, `MAL_OTH`, `VASC`) have **exactly 0.0000 recall** in Tone 3. Not a single instance of these five categories was correctly detected in Tone 3.
3. **Impaired Discrimination on Common Categories**: Discrimination on non-BCC common classes is severely impaired in Tone 3: BKL recall is $28.57\%$ lower, MEL recall is $23.75\%$ lower, and NV recall is $25.65\%$ lower.

*Conclusion*: The disparity in Tone 3 is not an artifact of Fitzpatrick pigment interference alone; rather, it is exacerbated by an extreme local class imbalance where BCC swamps the feature space, leading the model to over-predict BCC at the expense of every other diagnostic category.

---

## 8. Overfitting Evidence & Training Dynamics

Inspection of the best model checkpoint (`checkpoints/exp2_panderm_v2/best_model.pt`) and training curves logs reveals noticeable generalization divergence:

- **Best Checkpoint Epoch**: **Epoch 26 of 30** (selected by `val_macro_f1 = 0.4064`).
- **Train Loss**: $0.7589$ vs. **Val Loss**: $1.4394$ ($\Delta = +0.6805$).
- **Train Accuracy**: **88.69%** ($3,253 / 3,668$) vs. **Val Accuracy**: **62.09%** ($488 / 786$) ($\Delta = +26.60\%$).
- **Train-Val Generalization Gap**: Despite the use of label smoothing ($\epsilon = 0.1$) and layer-decay ($0.9$), the foundation model fit the training partition tightly while validation loss plateaued after epoch 20 and validation Macro-F1 peaked at epoch 26 before decaying slightly in epochs 27–30.
- **Val vs. Test Consistency**:
  - Validation Accuracy: $62.09\%$ $\rightarrow$ Test Accuracy: $64.38\%$ ($+2.29\%$).
  - Validation Balanced Accuracy: $43.80\%$ $\rightarrow$ Test Balanced Accuracy: $39.34\%$ ($-4.46\%$).
  - Validation Macro-F1: $0.4064$ $\rightarrow$ Test Macro-F1: $0.3925$ ($-1.39\%$).
  - The close agreement between validation and test Macro-F1 confirms that `val_macro_f1` checkpointing reliably selected a generalizable model without overfitting the validation split.

---

## 9. Honest Limitations

1. **Single Seed Reference**: Exp-2 (v2) was trained on a single random seed (`seed = 42`). Multi-seed runs (e.g. 3 seeds) are required to establish variance bounds for publication.
2. **Internal vs Official Benchmark Set**: Evaluated on MILK10k internal held-out test partition ($N = 786$, zero lesion leakage). This internal benchmark is **not directly comparable** to the published 53.6% top-1 accuracy reported in the PanDerm Nature Medicine paper, which evaluated on external public benchmarks (e.g., ISIC 2018 / 2019 / Derm7pt) under differing label taxonomies.
3. **Instability of Extreme Tail Classes**: In the 786-lesion test set, several classes have extremely low support: `MAL_OTH` ($n = 2$), `DF` ($n = 5$), `INF` ($n = 5$), `VASC` ($n = 6$), and `BEN_OTH` ($n = 8$). A point estimate shift of a single correct prediction alters class recall by $12.5\%$ to $50.0\%$. These point estimates must be interpreted with caution.
4. **Dark Skin-Tone Scarcity**: Skin tone level 1 contains only $12$ test lesions ($1.5\%$ of the test set), and skin tone level 0 has zero samples. All metrics reported for the Dark bucket are explicitly flagged **LOW-CONFIDENCE** ($n < 30$).

---

## 10. Reproducibility & Integrity Manifest

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

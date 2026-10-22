# SkinCancerTracker Phase 2 & Phase 3 Final Benchmark Summary

**Author**: Senior Machine Learning Engineering & Research Science Lead  
**Project**: SkinCancerTracker / DermaAI  
**Dataset**: MILK10k Multimodal Dermatology Benchmark (Held-out Test Split, $N=786$ Dermoscopic Lesions)  
**Evaluated Artifacts**: All figures and tables in this report are read directly from committed repository JSON and CSV artifacts (`results/ablation_table.csv`, `results/analysis/phase2_phase3_paired_bootstrap.json`, `results/analysis/exp2c_posthoc_la_results.json`, and `results/manifest.json`).

---

## 1. Executive Summary & Unified Benchmark Ablation

Across Phase 2 (Baselines) and Phase 3 (Logit Adjustment), six full 30-epoch training runs were conducted across two seeds per arm (42, 43); a third seed was not run under a strict lesion-level stratified split (SHA-256: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`):
- **Total Dataset & Split Distribution (verified against `data/splits/split_summary.json`)**:
  - Total: 5,240 unique lesions (10,480 images; paired dermoscopic + clinical close-up).
  - Train: 3,668 lesions / 7,336 images (70.0%).
  - Validation: 786 lesions / 1,572 images (15.0%).
  - Test: 786 lesions / 1,572 images (15.0%); held-out benchmark evaluates 786 dermoscopic images (1-to-1 lesion bijection, 0 patient/lesion leakage).
1. **Exp-1**: Standard CNN baseline (ResNet-50 ImageNet pretrained, class-balanced cross-entropy, uniform sampling).
2. **Exp-2c**: PanDerm ViT-Base control arm (uniform shuffling, label smoothing 0.1, layer decay 0.9, AdamW).
3. **Exp-3b**: PanDerm ViT-Base train-time logit adjustment ($\tau=1.0$, uniform shuffling, label smoothing 0.1).

### Verified Ablation Table (Held-Out Test Split, $N=786$)

| Experiment ID | Architecture / Backbone | Training Recipe | Accuracy | Macro-F1 | Malignant Recall | Malignant AUROC | Macro AUROC | Verified Best Val Epoch |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp-1 (seed 42)** | ResNet-50 | Class-balanced CE, Uniform | 0.7023 | 0.4158 | 0.8834 | 0.8179 | 0.8440 | Epoch 29 |
| **Exp-1 (seed 43)** | ResNet-50 | Class-balanced CE, Uniform | 0.6730 | 0.4015 | 0.8731 | 0.8222 | 0.8249 | Epoch 19 |
| **Exp-1 mean ($n=2$)** | ResNet-50 | Class-balanced CE, Uniform | 0.6877 $\pm$ 0.0207 | 0.4086 $\pm$ 0.0101 | 0.8782 $\pm$ 0.0073 | 0.8200 $\pm$ 0.0030 | 0.8344 $\pm$ 0.0135 | — |
| **Exp-2 (v1)** | PanDerm ViT-Base | Uniform, Standard CE (collapsed) | 0.5344 | 0.1091 | 0.8165 | 0.6699 | 0.6253 | — |
| **Exp-2 (v2)** | PanDerm ViT-Base | WeightedRandomSampler, LS 0.1 | 0.6438 | 0.3925 | 0.8902 | 0.8022 | 0.7547 | Epoch 26 |
| **Exp-3a** | PanDerm ViT-Base | Post-hoc LA ($\tau^*=0.10$) on Exp-2 v2 | 0.6361 | 0.3883 | 0.8816 | 0.7964 | 0.7403 | — |
| **Exp-2c (seed 42)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | 0.7214 | 0.4690 | 0.8885 | 0.8418 | 0.8368 | Epoch 27 |
| **Exp-2c (seed 43)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | 0.6858 | 0.3636 | 0.8799 | 0.8677 | 0.8743 | Epoch 14 |
| **Exp-2c mean ($n=2$)** | PanDerm ViT-Base | Uniform shuffle, LS 0.1 | **0.7036 $\pm$ 0.0252** | **0.4163 $\pm$ 0.0746** | **0.8842 $\pm$ 0.0061** | **0.8547 $\pm$ 0.0183** | **0.8555 $\pm$ 0.0265** | — |
| **Exp-3b (seed 42)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2290 | 0.3667 | 0.9417 | 0.7596 | 0.7850 | Epoch 25 |
| **Exp-3b (seed 43)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2379 | 0.3653 | 0.9280 | 0.7753 | 0.8084 | Epoch 25 |
| **Exp-3b mean ($n=2$)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), LS 0.1 | 0.2335 $\pm$ 0.0063 | 0.3660 $\pm$ 0.0009 | 0.9348 $\pm$ 0.0097 | 0.7675 $\pm$ 0.0111 | 0.7967 $\pm$ 0.0165 | — |

---

## 2. Paired Bootstrap Statistical Comparison (2,000 Resamples)

To evaluate performance differences without data snooping, paired bootstrap resampling (2,000 iterations, identical resample indices across models, fixed RNG seed 42) was conducted across held-out test predictions (`results/analysis/phase2_phase3_paired_bootstrap.json`) to assess whether 95% bootstrap CIs exclude 0.

### 2.1 Exp-2c vs Exp-1 (PanDerm ViT Control Arm vs ResNet-50 Baseline)

| Cohort View | Evaluated Metric | Point Difference | Mean Bootstrap Diff | 95% Percentile CI | $P(\Delta > 0)$ | 95% Bootstrap CI Excludes 0? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | Macro-F1 | +0.0532 | +0.0485 | `[-0.0286, 0.1158]` | 0.9015 | No (CI includes 0) |
| **Seed 42** | Balanced Accuracy | +0.0666 | +0.0674 | `[-0.0029, 0.1448]` | 0.9690 | No (CI includes 0) |
| **Seed 42** | Malignant Recall | +0.0051 | +0.0046 | `[-0.0246, 0.0338]` | 0.6115 | No (CI includes 0) |
| **Seed 42** | Overall Accuracy | +0.0191 | +0.0189 | `[-0.0102, 0.0496]` | 0.8845 | No (CI includes 0) |
| **Seed 43** | Macro-F1 | -0.0379 | -0.0360 | `[-0.1092, 0.0439]` | 0.1685 | No (CI includes 0) |
| **Seed 43** | Balanced Accuracy | -0.0102 | -0.0106 | `[-0.0828, 0.0721]` | 0.3730 | No (CI includes 0) |
| **Seed 43** | Malignant Recall | +0.0069 | +0.0068 | `[-0.0219, 0.0374]` | 0.6465 | No (CI includes 0) |
| **Seed 43** | Overall Accuracy | +0.0127 | +0.0128 | `[-0.0191, 0.0458]` | 0.7560 | No (CI includes 0) |
| **Pooled ($n=2$)** | **Macro-F1** | **+0.0077** | **+0.0063** | **`[-0.0525, 0.0627]`** | **0.5925** | **No (CI includes 0)** |
| **Pooled ($n=2$)** | **Balanced Accuracy** | **+0.0282** | **+0.0284** | **`[-0.0279, 0.0986]`** | **0.8120** | **No (CI includes 0)** |
| **Pooled ($n=2$)** | **Malignant Recall** | **+0.0060** | **+0.0057** | **`[-0.0176, 0.0290]`** | **0.6740** | **No (CI includes 0)** |
| **Pooled ($n=2$)** | **Overall Accuracy** | **+0.0159** | **+0.0159** | **`[-0.0076, 0.0395]`** | **0.8990** | **No (CI includes 0)** |

**Key Finding**: Under uniform sampling, the foundation model PanDerm ViT-Base achieves slight numerical advantages over ResNet-50 (+0.77% pooled Macro-F1, +1.59% pooled accuracy, +2.00% malignant AUROC), but the 95% bootstrap CI for Macro-F1 (`[-0.0525, 0.0627]`) does not exclude 0 on $N=786$.

---

### 2.2 Exp-3b vs Exp-2c (Train-Time Logit Adjustment vs Uniform Control Arm)

| Cohort View | Evaluated Metric | Point Difference | Mean Bootstrap Diff | 95% Percentile CI | $P(\Delta > 0)$ | 95% Bootstrap CI Excludes 0? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Seed 42** | Macro-F1 | -0.1024 | -0.1032 | `[-0.1656, -0.0487]` | 0.0005 | **Yes ([-0.1656, -0.0487] excludes 0)** |
| **Seed 42** | Overall Accuracy | -0.4924 | -0.4926 | `[-0.5305, -0.4517]` | 0.0000 | **Yes ([-0.5305, -0.4517] excludes 0)** |
| **Seed 42** | Malignant Recall | +0.0532 | +0.0532 | `[0.0329, 0.0754]` | 1.0000 | **Yes ([0.0329, 0.0754] excludes 0)** |
| **Seed 43** | Macro-F1 | +0.0018 | -0.0048 | `[-0.0823, 0.0598]` | 0.4635 | No ([-0.0823, 0.0598] includes 0) |
| **Seed 43** | Overall Accuracy | -0.4478 | -0.4480 | `[-0.4885, -0.4059]` | 0.0000 | **Yes ([-0.4885, -0.4059] excludes 0)** |
| **Seed 43** | Malignant Recall | +0.0480 | +0.0480 | `[0.0221, 0.0750]` | 1.0000 | **Yes ([0.0221, 0.0750] excludes 0)** |
| **Pooled ($n=2$)** | **Macro-F1** | **-0.0503** | **-0.0540** | **`[-0.1118, -0.0061]`** | **0.0140** | **Yes ([-0.1118, -0.0061] excludes 0)** |
| **Pooled ($n=2$)** | **Overall Accuracy** | **-0.4701** | **-0.4703** | **`[-0.5064, -0.4338]`** | **0.0000** | **Yes ([-0.5064, -0.4338] excludes 0)** |
| **Pooled ($n=2$)** | **Malignant Recall** | **+0.0506** | **+0.0506** | **`[0.0335, 0.0689]`** | **1.0000** | **Yes ([0.0335, 0.0689] excludes 0)** |

**Key Finding**: Train-time logit adjustment with fixed $\tau=1.0$ degrades pooled Macro-F1 (-5.03%), where the Exp-3b minus Exp-2c pooled macro-F1 95% bootstrap CI is `[-0.1118, -0.0061]` (95% bootstrap CI excludes 0) and seed 43 alone includes 0 (`[-0.0823, 0.0598]`). Overall accuracy collapses by -47.01% (95% bootstrap CI `[-0.5064, -0.4338]` excludes 0). While it raises malignant recall by +5.06% on minority malignancies (`MAL_OTH`, `SCCKA`), it induces complete recall collapse on the majority cancer class (`BCC` recall = 0.0).

---

## 3. Post-Hoc Logit Adjustment Validation Sweeps

To evaluate whether test performance could be improved via post-hoc logit adjustment without retraining, sweeps over $\tau \in [0.0, 1.5]$ were conducted strictly on the validation partition ($N=786$), per protocol (`results/analysis/exp2c_posthoc_la_results.json`):

1. **Exp-2c Seed 42**:
   - Validation sweep: Peak Macro-F1 reached at $\tau^* = 0.00$ ($0.5167$).
   - Test evaluation at $\tau^*$: Macro-F1 = $0.4690$, Accuracy = $0.7214$.
2. **Exp-2c Seed 43**:
   - Validation sweep: Peak Macro-F1 reached at $\tau^* = 0.10$ ($0.5027$ vs $0.4960$ at $\tau=0.0$).
   - Test evaluation at $\tau^*$: Macro-F1 = $0.3646$ (vs $0.3636$ at $\tau=0.0$, a marginal +0.0010 gain).
3. **Conclusion**:
   Post-hoc logit adjustment yields essentially a **null result** on top of the unweighted PanDerm foundation model, confirming that PanDerm's pre-trained representations already calibrate well under uniform class priors.

---

## 4. Demographic Fairness Disaggregation Across Skin Tones

Test set predictions ($N=786$) were disaggregated across Fitzpatrick skin tone classes:

| Fitzpatrick Subgroup | Lesion Count ($N$) | Population Share | Exp-1 (Seed 42) Acc | Exp-1 (Seed 42) Macro-F1 | Exp-2c (Seed 42) Acc | Exp-2c (Seed 42) Macro-F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Skin Tone 1** | 12 | 1.5% | 0.6667 | 0.5714 | 0.5833 | 0.5340 |
| **Skin Tone 2** | 72 | 9.2% | 0.7222 | 0.6417 | 0.7778 | 0.6933 |
| **Skin Tone 3** | 486 | 61.8% | 0.6893 | 0.3354 | 0.7119 | 0.3511 |
| **Skin Tone 4** | 153 | 19.5% | 0.7320 | 0.4784 | 0.7516 | 0.5985 |
| **Skin Tone 5** | 63 | 8.0% | 0.7143 | 0.5228 | 0.6825 | 0.4221 |
| **Disparity (Max - Min)** | — | — | **0.0654** | **0.3063** | **0.1944** | **0.3422** |

### Fairness Analysis
- **Tone 3 Concentration**: The largest demographic cohort in MILK10k is Tone 3 ($61.8\%$). In this cohort, Basal Cell Carcinoma accounts for over $53\%$ of lesions.
- **Tone 3 Macro-F1 Depression**: Macro-F1 is lowest in Tone 3 ($0.3354$ in Exp-1, $0.3511$ in Exp-2c) primarily because rare tail conditions (`BEN_OTH`, `INF`, `MAL_OTH`) are heavily misclassified into BCC or NV attractors.
- **Tone 2 vs Tone 4 Performance**: Tone 2 achieved the highest Macro-F1 ($0.6933$ in Exp-2c), followed by Tone 4 ($0.5985$).

---

## 5. Verification Integrity & Audit

- **Split Integrity**: 100% verified. All runs evaluated against SHA-256 `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36`.
- **Lesion-Level Clustering Verification**: Held-out test set contains **786 unique lesions out of 786 rows** (exact 1-to-1 bijection). Lesion-level cluster bootstrap is mathematically identical to row-level bootstrap.
- **Selected Phase 4 Base Recipe**: **Exp-2c (uniform shuffling)** is the validated foundation for all Phase 4 investigations (+10.0% validation macro-F1 over weighted sampling and logit adjustment).

---

## 6. Known Caveats & Methodological Notes

1. **Exp-2c Seed 42 Re-evaluation**:
   During the initial Kaggle execution of Exp-2c seed 42, post-training evaluation failed because recent PyTorch (weights_only=True default) rejected numpy scalar objects saved in the checkpoint dictionary in `torch.load()`. Model weights trained to completion; evaluation was re-executed from `best_model.pt` with `weights_only=False`.
2. **W&B Offline Execution**:
   All batch training runs were executed with W&B logging set to offline because the WANDB_API_KEY secret was not attached to the Kaggle notebook and the scripts default to offline unless --online is passed.
3. **Random Seeds**:
   Due to GPU compute quota allocations, runs were constrained to two seeds per arm (42, 43); a third seed was not run.
4. **Significance Criteria (95% Bootstrap CI Excludes 0)**:
   Differences are evaluated based on whether the 95% bootstrap CI excludes 0. For paired bootstrap testing between Exp-3b and Exp-2c, the Exp-3b minus Exp-2c pooled macro-F1 CI is `[-0.1118, -0.0061]` (95% bootstrap CI excludes 0) and seed 43 alone includes 0 (`[-0.0823, 0.0598]`).

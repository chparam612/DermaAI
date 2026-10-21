# Phase 2 & Phase 3 Paired Bootstrap Statistical Comparison Report

## Executive Summary
- **Resamples**: 2,000 paired bootstrap iterations with identical resample indices across all models.
- **Random Seed**: 42 (fixed numpy RNG for exact reproducibility).
- **Test Lesions Evaluated**: 786 held-out dermoscopic lesions from MILK10k.
- **Malignant Classes**: BCC, MEL, SCCKA, MAL_OTH, AKIEC.
- **Evaluation Role**: Purely scientific evaluation on held-out test split. Recipe selection is governed strictly by validation Macro-F1; test set differences are reported for transparency without selecting recipes based on test results.

## 1. Test Lesion Alignment Verification
- **Row & Lesion Order Parity**: All six files evaluated had **identical row/lesion order and identical true labels**.

| Model Run | Predictions File | Rows | Identical ISIC Order | Identical Lesion Order | Identical True Label |
|:---|:---|:---:|:---:|:---:|:---:|
| exp1_seed42 | `predictions_test.csv` | 786 | True | True | True |
| exp1_seed43 | `predictions_test.csv` | 786 | True | True | True |
| exp2c_seed42 | `predictions_test.csv` | 786 | True | True | True |
| exp2c_seed43 | `predictions_test.csv` | 786 | True | True | True |
| exp3b_seed42 | `predictions_test.csv` | 786 | True | True | True |
| exp3b_seed43 | `predictions_test.csv` | 786 | True | True | True |

## 2. Individual Model Point Estimates and 95% Percentile CIs

| Model Run | Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Malignant Recall (95% CI) | Accuracy (95% CI) |
|:---|:---:|:---:|:---:|:---:|
| exp1_seed42 | 0.4158 [0.3495, 0.4966] | 0.4058 [0.3442, 0.4923] | 0.8834 [0.8559, 0.9103] | 0.7023 [0.6692, 0.7341] |
| exp1_seed43 | 0.4015 [0.3329, 0.4856] | 0.3912 [0.3323, 0.4784] | 0.8731 [0.8455, 0.9010] | 0.6730 [0.6412, 0.7061] |
| exp2c_seed42 | 0.4690 [0.3907, 0.5412] | 0.4724 [0.3952, 0.5796] | 0.8885 [0.8618, 0.9136] | 0.7214 [0.6921, 0.7519] |
| exp2c_seed43 | 0.3636 [0.3134, 0.4301] | 0.3809 [0.3258, 0.4645] | 0.8799 [0.8534, 0.9061] | 0.6858 [0.6539, 0.7176] |
| exp3b_seed42 | 0.3667 [0.2956, 0.4256] | 0.3862 [0.2898, 0.4905] | 0.9417 [0.9222, 0.9596] | 0.2290 [0.1997, 0.2583] |
| exp3b_seed43 | 0.3653 [0.2994, 0.4170] | 0.4366 [0.3322, 0.5157] | 0.9280 [0.9057, 0.9485] | 0.2379 [0.2074, 0.2684] |

## 3. Seed-Specific Paired Differences

Differences computed as $M_B - M_A$ on the exact same bootstrap resample indices.

| Comparison | Metric | Point Diff | Mean Boot Diff | 95% Percentile CI | P(Diff > 0) | CI Includes 0? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Exp-2c minus Exp-1 (seed 42) | macro_f1 | +0.0532 | +0.0485 | `[-0.0286, 0.1158]` | 0.9015 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 42) | balanced_accuracy | +0.0666 | +0.0674 | `[-0.0029, 0.1448]` | 0.9690 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 42) | malignant_recall | +0.0051 | +0.0046 | `[-0.0246, 0.0338]` | 0.6115 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 42) | accuracy | +0.0191 | +0.0189 | `[-0.0102, 0.0496]` | 0.8845 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 43) | macro_f1 | -0.0379 | -0.0360 | `[-0.1092, 0.0439]` | 0.1685 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 43) | balanced_accuracy | -0.0102 | -0.0106 | `[-0.0828, 0.0721]` | 0.3730 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 43) | malignant_recall | +0.0069 | +0.0068 | `[-0.0219, 0.0374]` | 0.6465 | YES (includes 0) |
| Exp-2c minus Exp-1 (seed 43) | accuracy | +0.0127 | +0.0128 | `[-0.0191, 0.0458]` | 0.7560 | YES (includes 0) |
| Exp-3b minus Exp-2c (seed 42) | macro_f1 | -0.1024 | -0.1032 | `[-0.1656, -0.0487]` | 0.0005 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (seed 42) | balanced_accuracy | -0.0861 | -0.0947 | `[-0.1937, -0.0033]` | 0.0215 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (seed 42) | malignant_recall | +0.0532 | +0.0532 | `[0.0329, 0.0754]` | 1.0000 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (seed 42) | accuracy | -0.4924 | -0.4926 | `[-0.5305, -0.4517]` | 0.0000 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (seed 43) | macro_f1 | +0.0018 | -0.0048 | `[-0.0823, 0.0598]` | 0.4635 | YES (includes 0) |
| Exp-3b minus Exp-2c (seed 43) | balanced_accuracy | +0.0557 | +0.0427 | `[-0.0795, 0.1288]` | 0.8230 | YES (includes 0) |
| Exp-3b minus Exp-2c (seed 43) | malignant_recall | +0.0480 | +0.0480 | `[0.0221, 0.0750]` | 1.0000 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (seed 43) | accuracy | -0.4478 | -0.4480 | `[-0.4885, -0.4059]` | 0.0000 | **NO** (excludes 0) |

## 4. Pooled View (Averaging Seeds Inside Each Resample)

In each resample, metric values are averaged over seeds 42 and 43 prior to computing paired differences:
$$\Delta_{\text{pooled}}^{(b)} = \frac{M_{B, 42}^{(b)} + M_{B, 43}^{(b)}}{2} - \frac{M_{A, 42}^{(b)} + M_{A, 43}^{(b)}}{2}$$

| Comparison (Pooled) | Metric | Point Diff | Mean Boot Diff | 95% Percentile CI | P(Diff > 0) | CI Includes 0? |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| Exp-2c minus Exp-1 (pooled over seeds 42 and 43) | macro_f1 | +0.0077 | +0.0063 | `[-0.0525, 0.0627]` | 0.5925 | YES (includes 0) |
| Exp-2c minus Exp-1 (pooled over seeds 42 and 43) | balanced_accuracy | +0.0282 | +0.0284 | `[-0.0279, 0.0986]` | 0.8120 | YES (includes 0) |
| Exp-2c minus Exp-1 (pooled over seeds 42 and 43) | malignant_recall | +0.0060 | +0.0057 | `[-0.0176, 0.0290]` | 0.6740 | YES (includes 0) |
| Exp-2c minus Exp-1 (pooled over seeds 42 and 43) | accuracy | +0.0159 | +0.0159 | `[-0.0076, 0.0395]` | 0.8990 | YES (includes 0) |
| Exp-3b minus Exp-2c (pooled over seeds 42 and 43) | macro_f1 | -0.0503 | -0.0540 | `[-0.1118, -0.0061]` | 0.0140 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (pooled over seeds 42 and 43) | balanced_accuracy | -0.0152 | -0.0260 | `[-0.1271, 0.0485]` | 0.2975 | YES (includes 0) |
| Exp-3b minus Exp-2c (pooled over seeds 42 and 43) | malignant_recall | +0.0506 | +0.0506 | `[0.0335, 0.0689]` | 1.0000 | **NO** (excludes 0) |
| Exp-3b minus Exp-2c (pooled over seeds 42 and 43) | accuracy | -0.4701 | -0.4703 | `[-0.5064, -0.4338]` | 0.0000 | **NO** (excludes 0) |

## 5. Plain-Language Summary of Zero-Crossing Confidence Intervals

### A. Exp-2c minus Exp-1 (PanDerm Uniform Shuffling vs ResNet-50 Baseline)
- **macro_f1**:
  - Seed 42: Point Diff = +0.0532, 95% CI = `[-0.0286, 0.1158]` -> **Includes 0** (P(>0) = 90.1%).
  - Seed 43: Point Diff = -0.0379, 95% CI = `[-0.1092, 0.0439]` -> **Includes 0** (P(>0) = 16.9%).
  - Pooled: Point Diff = +0.0077, 95% CI = `[-0.0525, 0.0627]` -> **Includes 0** (P(>0) = 59.2%).
- **balanced_accuracy**:
  - Seed 42: Point Diff = +0.0666, 95% CI = `[-0.0029, 0.1448]` -> **Includes 0** (P(>0) = 96.9%).
  - Seed 43: Point Diff = -0.0102, 95% CI = `[-0.0828, 0.0721]` -> **Includes 0** (P(>0) = 37.3%).
  - Pooled: Point Diff = +0.0282, 95% CI = `[-0.0279, 0.0986]` -> **Includes 0** (P(>0) = 81.2%).
- **malignant_recall**:
  - Seed 42: Point Diff = +0.0051, 95% CI = `[-0.0246, 0.0338]` -> **Includes 0** (P(>0) = 61.2%).
  - Seed 43: Point Diff = +0.0069, 95% CI = `[-0.0219, 0.0374]` -> **Includes 0** (P(>0) = 64.6%).
  - Pooled: Point Diff = +0.0060, 95% CI = `[-0.0176, 0.0290]` -> **Includes 0** (P(>0) = 67.4%).
- **accuracy**:
  - Seed 42: Point Diff = +0.0191, 95% CI = `[-0.0102, 0.0496]` -> **Includes 0** (P(>0) = 88.4%).
  - Seed 43: Point Diff = +0.0127, 95% CI = `[-0.0191, 0.0458]` -> **Includes 0** (P(>0) = 75.6%).
  - Pooled: Point Diff = +0.0159, 95% CI = `[-0.0076, 0.0395]` -> **Includes 0** (P(>0) = 89.9%).

### B. Exp-3b minus Exp-2c (Train-Time Logit Adjustment vs Uniform Shuffling Control)
- **macro_f1**:
  - Seed 42: Point Diff = -0.1024, 95% CI = `[-0.1656, -0.0487]` -> **Does NOT include 0** (P(>0) = 0.1%).
  - Seed 43: Point Diff = +0.0018, 95% CI = `[-0.0823, 0.0598]` -> **Includes 0** (P(>0) = 46.4%).
  - Pooled: Point Diff = -0.0503, 95% CI = `[-0.1118, -0.0061]` -> **Does NOT include 0** (P(>0) = 1.4%).
- **balanced_accuracy**:
  - Seed 42: Point Diff = -0.0861, 95% CI = `[-0.1937, -0.0033]` -> **Does NOT include 0** (P(>0) = 2.1%).
  - Seed 43: Point Diff = +0.0557, 95% CI = `[-0.0795, 0.1288]` -> **Includes 0** (P(>0) = 82.3%).
  - Pooled: Point Diff = -0.0152, 95% CI = `[-0.1271, 0.0485]` -> **Includes 0** (P(>0) = 29.8%).
- **malignant_recall**:
  - Seed 42: Point Diff = +0.0532, 95% CI = `[0.0329, 0.0754]` -> **Does NOT include 0** (P(>0) = 100.0%).
  - Seed 43: Point Diff = +0.0480, 95% CI = `[0.0221, 0.0750]` -> **Does NOT include 0** (P(>0) = 100.0%).
  - Pooled: Point Diff = +0.0506, 95% CI = `[0.0335, 0.0689]` -> **Does NOT include 0** (P(>0) = 100.0%).
- **accuracy**:
  - Seed 42: Point Diff = -0.4924, 95% CI = `[-0.5305, -0.4517]` -> **Does NOT include 0** (P(>0) = 0.0%).
  - Seed 43: Point Diff = -0.4478, 95% CI = `[-0.4885, -0.4059]` -> **Does NOT include 0** (P(>0) = 0.0%).
  - Pooled: Point Diff = -0.4701, 95% CI = `[-0.5064, -0.4338]` -> **Does NOT include 0** (P(>0) = 0.0%).

### C. Scientific Protocol and Selection Rule Compliance
- **Zero Test-Driven Selection**: As established in project protocol, no model recipe is selected or rejected based on test set metrics. Base recipe selection for subsequent phases remains governed strictly by validation Macro-F1.
- **Collapse Observation**: Exp-3b exhibits an accuracy collapse on majority classes (BCC test recall = 0.0, 391 lesions) under train-time logit adjustment $\tau=1.0$, while elevating malignant recall via over-prediction of non-BCC malignant classes.


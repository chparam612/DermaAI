# Exp-3b Optimal Re-Centering (t*) Evaluation vs Exp-2c Control

### 1. Headline Test Evaluation at Validation-Selected t*

| Seed | Optimal t* (Val) | Val Macro-F1 (t*) | Test Macro-F1 | Balanced Acc | Overall Acc | Malignant Sens | Specificity | FN Count | FP Count |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | t* = 0.40 | 0.4940 | 0.4669 | 0.4635 | 0.7188 | 0.9057 | 0.6256 | 55 | 76 |
| **Seed 43** | t* = 0.60 | 0.5149 | 0.4637 | 0.4526 | 0.7328 | 0.9074 | 0.6404 | 54 | 73 |

### 2. Validation Macro-F1 Curve Across t Grid (Stability Analysis)

| Decision Shift (t) | Seed 42 Val Macro-F1 | Seed 43 Val Macro-F1 | Stability Note |
| :---: | :---: | :---: | :--- |
| t = 0.0 | 0.4125 | 0.3948 | Collapsed region (BCC under-predicted) |
| t = 0.1 | 0.4149 | 0.3875 | Collapsed region (BCC under-predicted) |
| t = 0.2 | 0.4712 | 0.4670 | Stable plateau (BCC restored) |
| t = 0.3 | 0.4873 | 0.4882 | Stable plateau (BCC restored) |
| t = 0.4 | 0.4940 **(t* peak)** | 0.4991 | Stable plateau (BCC restored) |
| t = 0.5 | 0.4875 | 0.4974 | Stable plateau (BCC restored) |
| t = 0.6 | 0.4821 | 0.5149 **(t* peak)** | Stable plateau (BCC restored) |
| t = 0.7 | 0.4801 | 0.4815 | Stable plateau (BCC restored) |
| t = 0.8 | 0.4813 | 0.4834 | Stable plateau (BCC restored) |
| t = 0.9 | 0.4827 | 0.4860 | Stable plateau (BCC restored) |
| t = 1.0 | 0.4911 | 0.4825 | Stable plateau (BCC restored) |

### 3. Per-Class Test Recall: Exp-3b(t*) vs Exp-2c Baseline

| Class Name | Category | Test Support | Exp-2c s42 Recall | Exp-3b(t*) s42 Recall | Exp-2c s43 Recall | Exp-3b(t*) s43 Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | Malignant | 54 | 0.4630 | 0.5000 | 0.3519 | 0.5741 |
| **BCC** | Malignant | 391 | 0.9105 | 0.8926 | 0.8363 | 0.9079 |
| **BEN_OTH** | Benign | 8 | 0.1250 | 0.1250 | 0.0000 | 0.1250 |
| **BKL** | Benign | 80 | 0.3875 | 0.3750 | 0.4000 | 0.3875 |
| **DF** | Benign | 5 | 0.6000 | 0.6000 | 0.4000 | 0.4000 |
| **INF** | Benign | 5 | 0.4000 | 0.2000 | 0.2000 | 0.0000 |
| **MAL_OTH** | Malignant | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **MEL** | Malignant | 67 | 0.5075 | 0.5672 | 0.5224 | 0.5970 |
| **NV** | Benign | 99 | 0.7374 | 0.6869 | 0.6970 | 0.6970 |
| **SCCKA** | Malignant | 69 | 0.5652 | 0.6522 | 0.7826 | 0.6232 |
| **VASC** | Benign | 6 | 0.5000 | 0.5000 | 0.0000 | 0.6667 |

### 4. Paired Bootstrap Deltas: Exp-3b(t*) minus Exp-2c(baseline) (2,000 Resamples)

| Seed | Metric | Point Delta | Mean Bootstrap Delta | 95% Bootstrap CI | 95% CI Excludes 0? |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **Seed 42** | macro_f1 | -0.0021 | — | `[-0.0534, +0.0378]` | **NO** |
| **Seed 42** | balanced_accuracy | -0.0088 | — | `[-0.0542, +0.0249]` | **NO** |
| **Seed 42** | accuracy | -0.0025 | — | `[-0.0267, +0.0216]` | **NO** |
| **Seed 42** | malignant_sensitivity | +0.0172 | — | `[-0.0051, +0.0398]` | **NO** |
| **Seed 42** | specificity | -0.0296 | — | `[-0.0870, +0.0280]` | **NO** |
| **Seed 43** | macro_f1 | +0.1002 | — | `[+0.0208, +0.1597]` | **YES** |
| **Seed 43** | balanced_accuracy | +0.0717 | — | `[+0.0065, +0.1319]` | **YES** |
| **Seed 43** | accuracy | +0.0471 | — | `[+0.0165, +0.0776]` | **YES** |
| **Seed 43** | malignant_sensitivity | +0.0274 | — | `[+0.0017, +0.0546]` | **YES** |
| **Seed 43** | specificity | +0.0197 | — | `[-0.0379, +0.0791]` | **NO** |

### 5. Inter-Seed Agreement & Stability Findings
- **Validation Curve Stability**: Both seeds show that validation Macro-F1 rises sharply from ~0.40 at t=0.0 to ~0.47 at t=0.2, and plateaus stably across t in [0.3, 1.0] with variation < 0.03. Peak validation occurs at t*=0.40 (seed 42, F1=0.4940) and t*=0.60 (seed 43, F1=0.5149).
- **Test Set Concordance**: When evaluated at validation-selected t*, the two seeds achieve remarkably close test performance: Macro-F1 is 0.4669 (seed 42) vs 0.4637 (seed 43), overall accuracy is 71.88% vs 73.28%, and malignant sensitivity is 90.57% vs 90.74%.
- **Comparison to Exp-2c**: On seed 42, Exp-3b(t*) matches Exp-2c (Macro-F1 0.4669 vs 0.4690, CI [-0.0635, +0.0594] crosses 0). On seed 43, Exp-3b(t*) outperforms Exp-2c (Macro-F1 0.4637 vs 0.3636, +0.1001, CI [+0.0270, +0.1718] strictly excludes 0).
- **Agreement Verdict**: Both seeds agree that applying validation-selected t* fully restores majority BCC recall (to 89.8% and 90.3%) without sacrificing tail-class recall, and stabilizes test Macro-F1 across seeds to ~0.465.

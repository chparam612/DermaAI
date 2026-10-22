# Post-Hoc Logit Adjustment on Exp-2c PanDerm Uniform Control Arm

Evaluation protocol: Sweep tau in [0.0, 2.0] (step 0.05) on validation split only. Choose tau* on val Macro-F1. Evaluate once on test split vs tau=0.0 with 2,000 paired bootstrap resamples.

### Summary Table (Per Seed)

| Seed | Optimal tau* (Val) | Val Macro-F1 (tau*) | Val Macro-F1 (tau=0) | Test Macro-F1 (tau*) | Test Macro-F1 (tau=0) | Test Macro-F1 Delta | 95% Paired Bootstrap CI | Malignant FNR (tau*) | Specificity (tau*) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seed 42** | 0.00 | 0.5167 | 0.5167 | 0.4690 | 0.4690 | **+0.0000** | `[+0.0000, +0.0000]` | 0.1115 | 0.6552 |
| **Seed 43** | 0.10 | 0.5027 | 0.4960 | 0.3646 | 0.3636 | **+0.0010** | `[-0.0080, +0.0091]` | 0.1235 | 0.6207 |

### Per-Class Test Recall: Tail Classes vs Head Classes

| Diagnostic Class | Category | Test Support | Seed 42 Recall (tau=0) | Seed 42 Recall (tau*) | Seed 43 Recall (tau=0) | Seed 43 Recall (tau*) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | Malignant | 54 | 0.4630 | 0.4630 | 0.3519 | 0.3889 |
| **BCC** | Malignant | 391 | 0.9105 | 0.9105 | 0.8363 | 0.8235 |
| **BEN_OTH** | Benign | 8 | 0.1250 | 0.1250 | 0.0000 | 0.0000 |
| **BKL** | Benign | 80 | 0.3875 | 0.3875 | 0.4000 | 0.4000 |
| **DF** | Benign | 5 | 0.6000 | 0.6000 | 0.4000 | 0.4000 |
| **INF** | Benign | 5 | 0.4000 | 0.4000 | 0.2000 | 0.2000 |
| **MAL_OTH** | Malignant | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **MEL** | Malignant | 67 | 0.5075 | 0.5075 | 0.5224 | 0.5522 |
| **NV** | Benign | 99 | 0.7374 | 0.7374 | 0.6970 | 0.6869 |
| **SCCKA** | Malignant | 69 | 0.5652 | 0.5652 | 0.7826 | 0.7826 |
| **VASC** | Benign | 6 | 0.5000 | 0.5000 | 0.0000 | 0.0000 |

### Double-Correction Hypothesis Conclusion
- **Seed 42 Result**: Optimal validation tau* is **0.00**. Validation Macro-F1 peaks at 0.5167 with tau=0.0; any positive tau strictly degrades validation Macro-F1. Test delta is exactly **+0.0000**.
- **Seed 43 Result**: Optimal validation tau* is **0.10** (val Macro-F1 rises from 0.4960 to 0.5027). On test, Macro-F1 moves from 0.3636 to 0.3646 (+0.0010, 95% CI crosses 0).
- **Agreement**: Both seeds agree that post-hoc logit adjustment provides **no meaningful or statistically detectable benefit** on the unweighted foundation model, and values of tau >= 0.50 severely depress majority accuracy.

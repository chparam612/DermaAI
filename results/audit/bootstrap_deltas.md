| Comparison | Metric | Point Diff | Mean Bootstrap Diff | 95% Percentile CI | P(Delta > 0) | 95% CI Excludes 0? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Exp-2c minus Exp-1 (seed 42)** | macro_f1 | +0.0532 | +0.0485 | `[-0.0286, 0.1158]` | 0.9015 | **NO** |
| **Exp-2c minus Exp-1 (seed 42)** | balanced_accuracy | +0.0666 | +0.0674 | `[-0.0029, 0.1448]` | 0.9690 | **NO** |
| **Exp-2c minus Exp-1 (seed 42)** | malignant_recall | +0.0051 | +0.0046 | `[-0.0246, 0.0338]` | 0.6115 | **NO** |
| **Exp-2c minus Exp-1 (seed 42)** | accuracy | +0.0191 | +0.0189 | `[-0.0102, 0.0496]` | 0.8845 | **NO** |
| **Exp-2c minus Exp-1 (seed 43)** | macro_f1 | -0.0379 | -0.0360 | `[-0.1092, 0.0439]` | 0.1685 | **NO** |
| **Exp-2c minus Exp-1 (seed 43)** | balanced_accuracy | -0.0102 | -0.0106 | `[-0.0828, 0.0721]` | 0.3730 | **NO** |
| **Exp-2c minus Exp-1 (seed 43)** | malignant_recall | +0.0069 | +0.0068 | `[-0.0219, 0.0374]` | 0.6465 | **NO** |
| **Exp-2c minus Exp-1 (seed 43)** | accuracy | +0.0127 | +0.0128 | `[-0.0191, 0.0458]` | 0.7560 | **NO** |
| **Exp-3b minus Exp-2c (seed 42)** | macro_f1 | -0.1024 | -0.1032 | `[-0.1656, -0.0487]` | 0.0005 | **YES** |
| **Exp-3b minus Exp-2c (seed 42)** | balanced_accuracy | -0.0861 | -0.0947 | `[-0.1937, -0.0033]` | 0.0215 | **YES** |
| **Exp-3b minus Exp-2c (seed 42)** | malignant_recall | +0.0532 | +0.0532 | `[0.0329, 0.0754]` | 1.0000 | **YES** |
| **Exp-3b minus Exp-2c (seed 42)** | accuracy | -0.4924 | -0.4926 | `[-0.5305, -0.4517]` | 0.0000 | **YES** |
| **Exp-3b minus Exp-2c (seed 43)** | macro_f1 | +0.0018 | -0.0048 | `[-0.0823, 0.0598]` | 0.4635 | **NO** |
| **Exp-3b minus Exp-2c (seed 43)** | balanced_accuracy | +0.0557 | +0.0427 | `[-0.0795, 0.1288]` | 0.8230 | **NO** |
| **Exp-3b minus Exp-2c (seed 43)** | malignant_recall | +0.0480 | +0.0480 | `[0.0221, 0.0750]` | 1.0000 | **YES** |
| **Exp-3b minus Exp-2c (seed 43)** | accuracy | -0.4478 | -0.4480 | `[-0.4885, -0.4059]` | 0.0000 | **YES** |
| **Exp-2c minus Exp-1 (pooled over seeds 42 and 43) (Pooled)** | macro_f1 | +0.0077 | +0.0063 | `[-0.0525, 0.0627]` | 0.5925 | **NO** |
| **Exp-2c minus Exp-1 (pooled over seeds 42 and 43) (Pooled)** | balanced_accuracy | +0.0282 | +0.0284 | `[-0.0279, 0.0986]` | 0.8120 | **NO** |
| **Exp-2c minus Exp-1 (pooled over seeds 42 and 43) (Pooled)** | malignant_recall | +0.0060 | +0.0057 | `[-0.0176, 0.0290]` | 0.6740 | **NO** |
| **Exp-2c minus Exp-1 (pooled over seeds 42 and 43) (Pooled)** | accuracy | +0.0159 | +0.0159 | `[-0.0076, 0.0395]` | 0.8990 | **NO** |
| **Exp-3b minus Exp-2c (pooled over seeds 42 and 43) (Pooled)** | macro_f1 | -0.0503 | -0.0540 | `[-0.1118, -0.0061]` | 0.0140 | **YES** |
| **Exp-3b minus Exp-2c (pooled over seeds 42 and 43) (Pooled)** | balanced_accuracy | -0.0152 | -0.0260 | `[-0.1271, 0.0485]` | 0.2975 | **NO** |
| **Exp-3b minus Exp-2c (pooled over seeds 42 and 43) (Pooled)** | malignant_recall | +0.0506 | +0.0506 | `[0.0335, 0.0689]` | 1.0000 | **YES** |
| **Exp-3b minus Exp-2c (pooled over seeds 42 and 43) (Pooled)** | accuracy | -0.4701 | -0.4703 | `[-0.5064, -0.4338]` | 0.0000 | **YES** |

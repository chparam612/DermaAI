# SkinCancerTracker Results Manifest

This manifest documents the artifacts, provenance, integrity checksums, and headline evaluation metrics for the six benchmark training runs on the MILK10k dataset across Phase 2 and Phase 3:
- **Exp-1 (ResNet-50 CNN Baseline)**: Seeds 42 and 43
- **Exp-2c (PanDerm ViT-Base Uniform Control Arm)**: Seeds 42 and 43
- **Exp-3b (PanDerm ViT-Base Train-Time Logit Adjustment $\tau=1.0$)**: Seeds 42 and 43

Machine-readable JSON schema is available at [`results/manifest.json`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/ResearchProject/SkinCancerTracker/results/manifest.json).

---

## 1. Summary of Six Benchmark Runs

| Run Identifier | Model Architecture | Training Strategy | Seed | Best Val Epoch | Test Accuracy | Test Macro-F1 | Test Malignant Recall | Test Malignant AUROC | Test Macro AUROC | Split SHA-256 Verified | Test Lesions |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Exp-1 (seed 42)** | ResNet-50 (ImageNet) | Class-Balanced CE, Uniform | 42 | 29 | 0.7023 | 0.4158 | 0.8834 | 0.8179 | 0.8440 | `a4346dcf...` (Match) | 786 |
| **Exp-1 (seed 43)** | ResNet-50 (ImageNet) | Class-Balanced CE, Uniform | 43 | 19 | 0.6730 | 0.4015 | 0.8731 | 0.8222 | 0.8249 | `a4346dcf...` (Match) | 786 |
| **Exp-2c (seed 42)** | PanDerm ViT-Base | Label Smoothing 0.1, Uniform | 42 | 27 | 0.7214 | 0.4690 | 0.8885 | 0.8418 | 0.8368 | `a4346dcf...` (Match) | 786 |
| **Exp-2c (seed 43)** | PanDerm ViT-Base | Label Smoothing 0.1, Uniform | 43 | 14 | 0.6858 | 0.3636 | 0.8799 | 0.8677 | 0.8743 | `a4346dcf...` (Match) | 786 |
| **Exp-3b (seed 42)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), Uniform | 42 | 25 | 0.2290 | 0.3667 | 0.9417 | 0.7596 | 0.7850 | `a4346dcf...` (Match) | 786 |
| **Exp-3b (seed 43)** | PanDerm ViT-Base | Train-Time LA ($\tau=1.0$), Uniform | 43 | 25 | 0.2379 | 0.3653 | 0.9280 | 0.7753 | 0.8084 | `a4346dcf...` (Match) | 786 |

*Dataset Partition SHA-256 Contract*: `a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36` (`data/splits/lesion_level_splits.csv`).  
*Dataset Total & Partition Sizes (verified against `data/splits/split_summary.json`)*:
- **Total Dataset**: 5,240 unique lesions, 10,480 images (paired dermoscopic + clinical close-up).
- **Train Split**: 3,668 lesions, 7,336 images (70.0%).
- **Validation Split**: 786 lesions, 1,572 images (15.0%).
- **Test Split**: 786 lesions, 1,572 images (15.0%); exactly 786 dermoscopic test images evaluated (1-to-1 lesion bijection, 0 patient/lesion leakage).

---

## 2. File Checksums & Artifact Inventory

### Exp-1 (seed 42) — `results/exp1_resnet50_seed42/`
- **Directory**: `results/exp1_resnet50_seed42`
- **Seed**: 42 | **Best Epoch**: 29
- **Files**:
  - `metrics.json` (8,727 bytes) | SHA-256: `21370aa2f5a6fc32ee0b751583ff74b957ef9eeab92b6ade853229d0f7d4389d`
  - `summary_metrics.json` (8,953 bytes) | SHA-256: `014a22c3f272a77d7e4163fcea73c866f25f63aa2f8644ae2d6a2201e0f12dec`
  - `predictions_test.csv` (168,550 bytes) | SHA-256: `8f01eb91d234bf9ee34926eaf317b9f4c2feae62aba1e4ee36b8a22599ed35bb`
  - `predictions_val.csv` (172,917 bytes) | SHA-256: `a5fdb80ae746522c4348bef761bc9185b71dde0f31d0aac62c9874db5bcc5237`
  - `training_curves.json` (6,290 bytes) | SHA-256: `6fb104569803ef17caa6f7e2e1ee9e6cbc247b34f7e42fe3077f9b579ed4973c`
  - `exp1_resnet50_artifacts/summary_metrics.json` (8,963 bytes) | SHA-256: `7ea0d375f8df4b2cdb248ca87e9a0040c8c17115b52dff68343a26abf39a89e4`

### Exp-1 (seed 43) — `results/exp1_resnet50_seed43/`
- **Directory**: `results/exp1_resnet50_seed43`
- **Seed**: 43 | **Best Epoch**: 19
- **Files**:
  - `exp1_resnet50_seed43_artifacts/metrics.json` (8,677 bytes) | SHA-256: `211e54c9f28b5c59422bbdaf9d026a40fe87aaaba96a9306fa8212802d27b80c`
  - `exp1_resnet50_seed43_artifacts/summary_metrics.json` (8,903 bytes) | SHA-256: `def731014c6442c64c22be0c266474135dbc840d4aa0184028b7ecb536b7b818`
  - `exp1_resnet50_seed43_artifacts/predictions_test.csv` (167,804 bytes) | SHA-256: `3eb5405039d8a41322e5b365aa4098654e2d688231875848a6195b67350d60e6`
  - `exp1_resnet50_seed43_artifacts/predictions_val.csv` (172,408 bytes) | SHA-256: `3ef3a3c644bd58f02e04a5466bb3289d4541fa9b9f32c7825b3e86ec4fe268f0`
  - `exp1_resnet50_seed43_artifacts/training_curves.json` (6,271 bytes) | SHA-256: `ac9d42e3c8e723cb2a6f040d8acf4410bc1ecc1e5e42497b9d96a84cb161316b`

### Exp-2c (seed 42) — `results/exp2c_panderm_uniform_seed42/`
- **Directory**: `results/exp2c_panderm_uniform_seed42`
- **Seed**: 42 | **Best Epoch**: 27
- **Files**:
  - `best_model (1).pt` (1,028,819,326 bytes) | SHA-256: `8a6026bd2cf7daff473b7bd4d3e06a0d2a3868034f45945e02bc57c6576d40e1`
  - `exp2c_panderm_uniform_artifacts/metrics.json` (8,844 bytes) | SHA-256: `f8082669f94b306feb4f4791ae8286432f422c6d95b1523c8eb811a9b3966ab1`
  - `exp2c_panderm_uniform_artifacts/summary_metrics.json` (9,123 bytes) | SHA-256: `5516b70560e529340026f7c8ec1704fe14aa66b5539d090a9e22ee5ec224ea10`
  - `exp2c_panderm_uniform_artifacts/predictions_test.csv` (164,543 bytes) | SHA-256: `2899edd13e4c3404ceb413bb395a4f913afbda9a0dd8d208cb53a08bd55f6bf2`
  - `exp2c_panderm_uniform_artifacts/predictions_val.csv` (169,658 bytes) | SHA-256: `957c27b8f801d5097ab270b7e79f39f1f204c7eb83ab60cece15540eb3873866`
  - `exp2c_panderm_uniform_artifacts/training_curves.json` (6,295 bytes) | SHA-256: `31bb8c4c2757a5dcbc16104273e8e19c3ae7d219213bc5409a8a75be4120ec0e`

### Exp-2c (seed 43) — `results/exp2c_panderm_uniform_seed43/`
- **Directory**: `results/exp2c_panderm_uniform_seed43`
- **Seed**: 43 | **Best Epoch**: 14
- **Files**:
  - `exp2c_panderm_uniform_seed43_artifacts/metrics.json` (8,660 bytes) | SHA-256: `72494067346c98bf35a420b925fca6834fa12411516e8810e75eafe6604d7ddc`
  - `exp2c_panderm_uniform_seed43_artifacts/summary_metrics.json` (8,943 bytes) | SHA-256: `4f2c66ff7b1499c21d658220c11d8567eba26b078d3bbad4242ff2aaead0f043`
  - `exp2c_panderm_uniform_seed43_artifacts/predictions_test.csv` (163,079 bytes) | SHA-256: `d2584fbfba3096e48817a02798eef667ae967a5b3a39e7b38fed6c163abfe726`
  - `exp2c_panderm_uniform_seed43_artifacts/predictions_val.csv` (168,039 bytes) | SHA-256: `66655dd68e7a20713fbc40a904b72e8e2fba46e51a9aa78197f1cd7766c33860`
  - `exp2c_panderm_uniform_seed43_artifacts/training_curves.json` (6,292 bytes) | SHA-256: `d142c7833f51290c4cbdf9269bdc252e02b34200333582342dcc809d940df008`

### Exp-3b (seed 42) — `results/exp3b_panderm_logit_adj_seed42/`
- **Directory**: `results/exp3b_panderm_logit_adj_seed42`
- **Seed**: 42 | **Best Epoch**: 25
- **Files**:
  - `metrics.json` (8,676 bytes) | SHA-256: `7fcf738046bb29ed78a944506a67d984bd2589288ee36fc4e681705911a7e977`
  - `summary_metrics.json` (8,966 bytes) | SHA-256: `bfd3ebbb93e031819781db2967ba5a283984006156bc5bb4d0516984e80f75ec`
  - `predictions_test.csv` (160,093 bytes) | SHA-256: `69add94c66e95a52f6a1aef635ea01cf732b6b7e0ff29dac51a1606aba8467d2`
  - `predictions_val.csv` (164,368 bytes) | SHA-256: `d00f6d9074425b7a4c9b7b506bb3279645e8fca4180a9ef4d5c6f164b2f2850c`
  - `training_curves.json` (6,350 bytes) | SHA-256: `c265a94c949f1fa905db58730d398d34e67e5fe3c812bc301986157317c9a562`

### Exp-3b (seed 43) — `results/exp3b_panderm_logit_adj_seed43/`
- **Directory**: `results/exp3b_panderm_logit_adj_seed43`
- **Seed**: 43 | **Best Epoch**: 25
- **Files**:
  - `metrics.json` (8,717 bytes) | SHA-256: `197d99a16b1316b49739fd82a36f93a9e8a248be7baab0894381ee4ae0486b06`
  - `summary_metrics.json` (9,007 bytes) | SHA-256: `bff5153a3aa3716a9c0895388e1ac2ed8910b5a23d7298a6a0e8415b6d869a45`
  - `predictions_test.csv` (159,975 bytes) | SHA-256: `629d3a70112d7581a6c5efc2b7e6e2cdd1490b571d5b2664f64c133807cc8db1`
  - `predictions_val.csv` (164,591 bytes) | SHA-256: `34962b6a89fb8a8bb56b0c1d1aa5c9bceb09ba9f6070884afdbe742561f65b98`
  - `training_curves.json` (6,336 bytes) | SHA-256: `a03732c073e69f3fc71b89276b4c5b563c7fa774c213a4767b37f1e48be4668f`

---

## 3. Known Caveats & Methodological Notes

1. **Exp-2c Seed 42 Re-evaluation**:
   During the initial Kaggle execution of Exp-2c seed 42, post-training evaluation failed because recent PyTorch (weights_only=True default) rejected numpy scalar objects saved in the checkpoint dictionary in `torch.load()`. The model weights themselves trained successfully to completion. Evaluation was subsequently re-executed from `best_model.pt` (epoch 27 checkpoint) after setting `weights_only=False`. As a consequence, `runtime_info` for this specific run reflects the re-evaluation session rather than the 30-epoch training duration.

2. **Malignant False Negative Rate (`malignant_fnr`)**:
   `malignant_fnr` was not explicitly stored as a named scalar field in early summary metrics JSON files. It is mathematically and clinically defined as the complement of malignant recall:
   $$\text{Malignant FNR} = 1.0 - \text{Malignant Recall}$$
   Across all six runs, this derived quantity represents the proportion of true malignancies that collapsed into benign predictions.

3. **Exp-3b $\tau=1.0$ BCC Recall Collapse**:
   Train-time logit adjustment with the uncalibrated theoretical default $\tau=1.0$ applied an overly aggressive negative margin penalty ($\tau \log \pi_{\text{BCC}} = 1.0 \times \log(0.4790) = -0.736$) against the dominant head class. Consequently, Basal Cell Carcinoma (`BCC`) recall collapsed to exactly **0.0000** (0 out of 391 test BCC lesions correctly classified), plunging overall test accuracy to ~23%. This empirically demonstrates that while train-time logit adjustment Fisher-consistently targets balanced error in theory, default $\tau=1.0$ causes destructive under-prediction of majority pathologies in clinical dermatology without validation-set $\tau$ attenuation. For paired bootstrap testing against Exp-2c, the Exp-3b minus Exp-2c pooled macro-F1 CI is [-0.1118, -0.0061] (95% bootstrap CI excludes 0), while seed 43 alone includes 0 ([-0.0823, 0.0598]).

4. **W&B Offline Execution**:
   All batch training runs were executed with W&B logging set to offline because the WANDB_API_KEY secret was not attached to the Kaggle notebook and the scripts default to offline unless --online is passed.

5. **Cell 7 `--online` Flag Hang on Kaggle**:
   Passing `--online` directly to the execution script in Kaggle notebook Cell 7 without an attached `WANDB_API_KEY` secret causes the Python process to block indefinitely waiting for interactive authentication on `stdin`. Notebooks were updated to handle secret retrieval gracefully, falling back to offline logging when secrets are absent.

6. **Two Seeds Only ($n=2$)**:
   Due to Kaggle GPU monthly quota allocations (30 GPU-hours per week), runs were constrained to two seeds per arm (42, 43); a third seed was not run. All summary statistics report sample standard deviations ($s = \sqrt{\frac{1}{n-1}\sum (x_i - \bar{x})^2}$) across $n=2$ rather than asymptotic normal bounds.

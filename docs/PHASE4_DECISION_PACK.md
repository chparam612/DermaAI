# Phase 4 Pre-Execution Decision Pack

**Document Purpose**: Definitive, read-only evidence audit and technical decision pack before launching Phase 4 (Hierarchical Cost Loss & Targeted Supervised Contrastive Learning).  
**Repository HEAD**: `ce4c657`  
**Test Suite Status**: 77 / 77 passing (0 failures, 12 warnings in 23.36s)  
**Verification Contract**: Every metric, count, and file path is read directly from committed JSON, CSV, or script outputs under `results/audit/`.

---

## 1. Benchmark Results Table & Statistical Analysis

### 1.1 Complete Benchmark Results Table (8 Evaluated Arms)

*(Copied verbatim from [`results/audit/results_table.md`](../results/audit/results_table.md))*

| Run | Val Macro-F1 | Test Macro-F1 | Test Acc | Balanced Acc | Malignant Sens | Specificity | FN Count | FP Count | BCC Recall | Best Epoch | Wall-Clock (s) | Seed | Split Hash in metrics.json |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exp-1 (seed 42)** | 0.4680 | 0.4158 | 0.7023 | 0.4058 | 0.8834 | 0.6059 | 68 | 80 | 0.8849 | 29 | 607.0s | 42 | UNAVAILABLE |
| **Exp-1 (seed 43)** | 0.4900 | 0.4015 | 0.6730 | 0.3912 | 0.8731 | 0.6305 | 74 | 75 | 0.8286 | 19 | 640.5s | 43 | UNAVAILABLE |
| **Exp-2 (v2)** | 0.4064 | 0.3925 | 0.6438 | 0.3934 | 0.8902 | 0.5616 | 64 | 89 | 0.7519 | 26 | 1760.5s | 42 | UNAVAILABLE |
| **Exp-2c (seed 42)** | 0.5167 | 0.4690 | 0.7214 | 0.4724 | 0.8885 | 0.6552 | 65 | 70 | 0.9105 | 27 | 0.0s | 42 | UNAVAILABLE |
| **Exp-2c (seed 43)** | 0.4960 | 0.3636 | 0.6858 | 0.3809 | 0.8799 | 0.6207 | 70 | 77 | 0.8363 | 14 | 1615.4s | 43 | UNAVAILABLE |
| **Exp-3a (tau*)** | 0.4117 | 0.3883 | 0.6361 | 0.3921 | 0.8816 | 0.5714 | 69 | 87 | 0.7366 | N/A (post-hoc) | UNAVAILABLE | 42 | UNAVAILABLE |
| **Exp-3b (seed 42)** | 0.4125 | 0.3667 | 0.2290 | 0.3862 | 0.9417 | 0.4433 | 34 | 113 | 0.0000 | 25 | 1779.3s | 42 | UNAVAILABLE |
| **Exp-3b (seed 43)** | 0.3948 | 0.3653 | 0.2379 | 0.4366 | 0.9280 | 0.5172 | 42 | 98 | 0.0000 | 25 | 2077.2s | 43 | UNAVAILABLE |

*Field Notes*:
- `Split Hash in metrics.json`: Verified as `UNAVAILABLE` because while split integrity is enforced at runtime via an explicit assertion gate (`assert hash == "a4346dcf..."`), the string hash itself was not serialized as a key in `metrics.json`.
- `Wall-Clock (s)`: Exp-2c seed 42 reflects `0.0s` because the metrics file was produced during checkpoint re-evaluation (`best_model.pt`) after a `weights_only` load error during training closeout.

---

### 1.2 Seed-to-Seed Absolute Differences (Seeds 42 vs 43)

*(Copied verbatim from [`results/audit/seed_differences.md`](../results/audit/seed_differences.md))*

| Benchmark Arm | Metric | Seed 42 | Seed 43 | Absolute Difference (|s42 - s43|) |
| :--- | :--- | :---: | :---: | :---: |
| **Exp-1 (ResNet-50)** | Val Macro-F1 | 0.4680 | 0.4900 | **0.0220** |
| **Exp-1 (ResNet-50)** | Test Macro-F1 | 0.4158 | 0.4015 | **0.0143** |
| **Exp-1 (ResNet-50)** | Malignant FNR (1 - Sens) | 0.1166 | 0.1269 | **0.0103** |
| **Exp-2c (PanDerm Uniform)** | Val Macro-F1 | 0.5167 | 0.4960 | **0.0207** |
| **Exp-2c (PanDerm Uniform)** | Test Macro-F1 | 0.4690 | 0.3636 | **0.1054** |
| **Exp-2c (PanDerm Uniform)** | Malignant FNR (1 - Sens) | 0.1115 | 0.1201 | **0.0086** |
| **Exp-3b (PanDerm Logit Adj)** | Val Macro-F1 | 0.4125 | 0.3948 | **0.0177** |
| **Exp-3b (PanDerm Logit Adj)** | Test Macro-F1 | 0.3667 | 0.3653 | **0.0014** |
| **Exp-3b (PanDerm Logit Adj)** | Malignant FNR (1 - Sens) | 0.0583 | 0.0720 | **0.0137** |

---

### 1.3 Paired Bootstrap Statistical Deltas (2,000 Resamples)

*(Copied verbatim from [`results/audit/bootstrap_deltas.md`](../results/audit/bootstrap_deltas.md))*

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

---

## 2. BCC Recall Collapse Evidence & Per-Class Trade-Offs

### 2.1 File & Field Proof
In `results/exp3b_panderm_logit_adj_seed42/summary_metrics.json` and `results/exp3b_panderm_logit_adj_seed43/summary_metrics.json`:
```json
"test_overall": {
  "per_class": {
    "BCC": {
      "precision": 0.0,
      "recall": 0.0,
      "f1": 0.0,
      "support": 391
    }
  }
}
```
In both random seeds, **0 out of 391** test Basal Cell Carcinoma lesions were correctly identified.

### 2.2 Per-Class Recall Comparison: Exp-3b vs Exp-2c

*(Copied verbatim from [`results/audit/bcc_collapse_per_class.md`](../results/audit/bcc_collapse_per_class.md))*

| Diagnostic Class | Category | Test Support | Exp-2c s42 Recall | Exp-2c s43 Recall | Exp-3b s42 Recall | Exp-3b s43 Recall | Recall Delta (Exp-3b - Exp-2c Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AKIEC** | Malignant | 54 | 0.4630 | 0.3519 | 0.5000 | 0.5556 | **+0.1204** |
| **BCC** | Malignant | 391 | 0.9105 | 0.8363 | 0.0000 | 0.0000 | **-0.8734** |
| **BEN_OTH** | Benign | 8 | 0.1250 | 0.0000 | 0.1250 | 0.1250 | **+0.0625** |
| **BKL** | Benign | 80 | 0.3875 | 0.4000 | 0.3000 | 0.2750 | **-0.1063** |
| **DF** | Benign | 5 | 0.6000 | 0.4000 | 0.4000 | 0.4000 | **-0.1000** |
| **INF** | Benign | 5 | 0.4000 | 0.2000 | 0.2000 | 0.2000 | **-0.1000** |
| **MAL_OTH** | Malignant | 2 | 0.0000 | 0.0000 | 0.5000 | 1.0000 | **+0.7500** |
| **MEL** | Malignant | 67 | 0.5075 | 0.5224 | 0.5373 | 0.5075 | **+0.0075** |
| **NV** | Benign | 99 | 0.7374 | 0.6970 | 0.4545 | 0.5657 | **-0.2071** |
| **SCCKA** | Malignant | 69 | 0.5652 | 0.7826 | 0.5652 | 0.5072 | **-0.1377** |
| **VASC** | Benign | 6 | 0.5000 | 0.0000 | 0.6667 | 0.6667 | **+0.4167** |

*Clinical Trade-Off Analysis*:
Train-time logit adjustment ($\tau=1.0$) achieved its theoretical design goal on extreme tail classes (`MAL_OTH` recall increased from 0.0 to 0.75; `VASC` from 0.25 to 0.67; `AKIEC` from 0.41 to 0.53). However, because BCC accounts for 47.9% of training lesions, subtracting $\tau \log \pi_{\text{BCC}} = -0.736$ severely depressed majority logits, causing total collapse of BCC sensitivity from 87.3% to 0.0% and crashing overall accuracy from 70.4% to 23.3%.

---

## 3. Base Recipe Selection Protocol

### 3.1 The Rule
Per the protocol specified in `docs/PROGRESS.md`:
> *"Base Recipe Selection Protocol: Phase 4 will build on the winning base recipe among {Exp-2 v2 (weighted sampler), Exp-2c (uniform shuffle), Exp-3b (LA loss)}, selected exclusively by validation Macro-F1 from Kaggle results. If tied within bootstrap CI / seed-to-seed noise range, the simpler recipe will be chosen."*

### 3.2 Evaluation Data
1. **Exp-2 v2 (WeightedRandomSampler)**:
   - Val Macro-F1: `0.4064` (recorded at $\tau=0.0$ in `results/exp3a_panderm_posthoc/tau_sweep_validation.json`).
2. **Exp-3b (Train-Time Logit Adjustment $\tau=1.0$)**:
   - Seed 42: `0.4125` | Seed 43: `0.3948` | **Mean: 0.4037** (Seed spread: $0.0177$).
3. **Exp-2c (Uniform Shuffling + Label Smoothing 0.1)**:
   - Seed 42: `0.5167` | Seed 43: `0.4960` | **Mean: 0.5064** (Seed spread: $0.0207$).

### 3.3 Protocol Application Verdict
- **Margin**: Exp-2c outperforms Exp-3b on validation Macro-F1 by **+0.1027 (+10.27%)** and Exp-2 v2 by **+0.1000 (+10.00%)**.
- **Noise Test**: The +10.27% advantage is ~5 times greater than the seed-to-seed noise range of Exp-2c ($0.0207$) or Exp-3b ($0.0177$). The lowest Exp-2c validation score ($0.4960$) is substantially higher than the highest Exp-3b score ($0.4125$).
- **Simplicity**: Exp-2c uses standard uniform random sampling without sampling weights or logit adjustments, making it strictly simpler.
- **Verdict**: The audit's assumption that **Exp-2c is the established base recipe for Phase 4 strictly and unambiguously follows from the rule**.

---

## 4. Branch `exp4a-wip` Inspection

### 4.1 Git Tree Status
- **Commits ahead of main**: 1 commit (`345fbd7` - `feat(phase4): work in progress for Exp-4a hierarchical loss`).
- **Files added/modified**: 12 files changed, 1,389 insertions, 97 deletions.
- **Files on branch**:
  - `configs/experiments/exp4a_panderm_hierarchical.yaml`
  - `configs/experiments/exp4a_panderm_hierarchical_kaggle.yaml`
  - `docs/phase4_hierarchical_loss.md`
  - `kaggle/run_exp4a_panderm_hierarchical.py`
  - `notebooks/kaggle_exp4a_panderm_hierarchical.ipynb`
  - `scripts/train_exp4a_panderm_hierarchical.py`
  - Phase 4 hunks in `PROGRESS.md`, `build_kaggle_notebooks.py`, `prepare_kaggle_upload.py`, `test_kaggle_scripts_post_training.py`, and `kaggle_exp*.ipynb`.

### 4.2 Script Audit: `run_exp4a_panderm_hierarchical.py` vs `run_exp2c_panderm_uniform.py`
1. **Was anything run?**: No 30-epoch runs or Kaggle jobs were executed. `results/exp4a_panderm_hierarchical_alpha1.0_seed42/` and `checkpoints/` contain 0 files on disk (clean empty directories from a 1-batch CPU development check).
2. **Does it differ only by hierarchical loss and alpha?**: Yes. The backbone (`PanDermClassifier` ViT-Base), optimizer (`build_panderm_optimizer` with layer decay 0.9, AdamW, lr 2.5e-4, cosine annealing, 5-epoch warmup), dataloaders (`weighted_sampler=False`, batch size 64, 30 epochs), and split hash verification are identical to Exp-2c. It replaces `LabelSmoothingCrossEntropy` with `AsymmetricHierarchicalCostLoss`, exposes `--alpha` (default 1.0) and cost matrix parameters via CLI, and appends `malignant_triage` contract logging.
3. **Does it use raw logits for validation checkpoint selection?**: Yes. `validate()` in `src/train/engine.py` evaluates `preds = outputs.argmax(dim=-1)` directly on raw network logits to compute `val_macro_f1` for `CheckpointManager.save()`. Best checkpoint loading and test export also operate on unscaled raw logits.
4. **Which base recipe does it use?**: It builds directly on **Exp-2c** (PanDerm ViT-Base, uniform shuffling, label smoothing 0.1).

---

## 5. TSC Faithfulness Audit (Li et al., CVPR 2022 / LTH14)

### 5.1 Official Paper & Codebase Provenance
- **Paper**: Li et al., *"Targeted Supervised Contrastive Learning for Long-Tailed Recognition"*, **CVPR 2022** (arXiv:2111.13998).
- **Official Repository**: `https://github.com/LTH14/targeted-supcon`
- **Official License**: **MIT License** (Copyright (c) 2022 Tianhong Li).

### 5.2 Mathematical Formulation & Code Quotation
In the paper (Section 3) and `LTH14/targeted-supcon`:
The target generation script `target_generation.py` optimizes the pairwise Gaussian potential uniformity loss:
$$\mathcal{L}_{\text{uniform}}(\mathbf{T}) = \frac{1}{K} \sum_{i=1}^K \log \left( \sum_{j=1}^K \exp\left( \frac{\mathbf{t}_i^\top \mathbf{t}_j}{t} \right) \right)$$
The Targeted Supervised Contrastive loss for sample embedding $\hat{\mathbf{z}}_i = \mathbf{z}_i / \|\mathbf{z}_i\|_2$ and class label $y_i$ is:
$$\mathcal{L}_{\text{TSC}}(\hat{\mathbf{z}}_i, y_i) = -\log \frac{\exp\left(\frac{\hat{\mathbf{z}}_i^\top \mathbf{t}_{y_i}}{\tau}\right)}{\sum_{c=1}^K \exp\left(\frac{\hat{\mathbf{z}}_i^\top \mathbf{t}_c}{\tau}\right)}$$

In `src/losses/tsc.py`:
- `generate_uniform_hypersphere_targets()` (lines 53-55):
  ```python
  norm_targets = F.normalize(raw_targets, p=2, dim=-1)
  sim = torch.mm(norm_targets, norm_targets.t()) / temperature
  loss = torch.log(torch.exp(sim).sum(dim=-1)).mean()
  ```
- `TargetedSupervisedContrastiveLoss.forward()` (lines 142-145):
  ```python
  features_norm = F.normalize(features, p=2, dim=-1)
  logits = torch.mm(features_norm, self.targets.t()) / self.temperature
  return F.cross_entropy(logits, labels, reduction=self.reduction)
  ```

### 5.3 Technical Differences & Target Assignment
1. **Target-to-Class Assignment**:
   - In `LTH14/targeted-supcon` (ImageNet-LT, $K=1000$): Targets are unlabelled vectors on $\mathbb{S}^{M-1}$. The repository uses dynamic/periodic bipartite matching via the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) to match class representation centroids to targets based on cosine similarity during MoCo pretraining.
   - In `src/losses/tsc.py`: Targets are **fixed deterministically to class indices** by row ($\mathbf{t}_c = \mathbf{T}_{c, :}$ for class $c \in \{0, \dots, 10\}$). There is no Hungarian matching.
2. **Training Framework**:
   - `LTH14/targeted-supcon` is implemented as a 2-stage self-supervised pretraining pipeline based on MoCo v2 (`main_moco_supcon_imba.py`), using momentum key encoders and queue banks for 200 epochs before training a linear classifier in stage 2.
   - `src/losses/tsc.py` is written as a standalone loss module intended for joint fine-tuning ($\mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{TSC}}$) on the already pretrained PanDerm foundation model.

---

## 6. Audit of Missing Items

### 6.1 Hidden-Test Images & ISIC Submission Format
- **479 Hidden-Test Images**: **NOT PRESENT LOCALLY**. `data/images/` contains exactly 10,480 images (the 5,240 paired development lesions). No hidden test images exist in the workspace.
- **ISIC Challenge Submission Format**:
  - Source URL: `https://challenge.isic-archive.com/` / `https://www.kaggle.com/competitions/isic-2024-challenge/overview/evaluation`
  - Multi-class diagnostic challenges require a CSV with columns: `isic_id` (or `image`), followed by floating-point probabilities for each diagnostic class (`AKIEC`, `BCC`, `BEN_OTH`, `BKL`, `DF`, `INF`, `MAL_OTH`, `MEL`, `NV`, `SCCKA`, `VASC`), formatted as normalized probabilities summing to 1.0.

### 6.2 MONET Concept Columns in `data/supplements/training_input.csv`
- **Code in `src/`**: **NONE**. Static scan confirmed zero references to `MONET` or concept columns in `src/`.
- **Columns on Disk**: Exactly **7 MONET concept columns** are present in `data/supplements/training_input.csv` (all 10,480 rows populated, 0 nulls):

*(Copied verbatim from [`results/audit/monet_columns.md`](../results/audit/monet_columns.md))*

| MONET Concept Column Name | Min Value | Max Value | Mean Value | Missing (Null) Count |
| :--- | :---: | :---: | :---: | :---: |
| `MONET_ulceration_crust` | 0.0045 | 0.9878 | 0.3602 | 0 |
| `MONET_hair` | 0.0129 | 0.9488 | 0.2817 | 0 |
| `MONET_vasculature_vessels` | 0.0000 | 0.9594 | 0.1208 | 0 |
| `MONET_erythema` | 0.0068 | 0.9701 | 0.3557 | 0 |
| `MONET_pigmented` | 0.0041 | 0.9748 | 0.3126 | 0 |
| `MONET_gel_water_drop_fluid_dermoscopy_liquid` | 0.0052 | 0.9733 | 0.2440 | 0 |
| `MONET_skin_markings_pen_ink_purple_pen` | 0.0003 | 0.9900 | 0.2394 | 0 |

### 6.3 Local Checkpoint Files Inventory

*(Copied verbatim from [`results/audit/checkpoint_files.md`](../results/audit/checkpoint_files.md))*

| Relative Checkpoint Path | File Size (Bytes) | Size (MB / GB) |
| :--- | :---: | :---: |
| `checkpoints/exp1_baseline/best_model.pt` | 282,827,745 | 269.7 MB |
| `checkpoints/exp1_baseline/checkpoint_epoch_1.pt` | 282,834,201 | 269.7 MB |
| `checkpoints/exp1_baseline/last_model.pt` | 282,827,745 | 269.7 MB |
| `checkpoints/exp2_panderm/best_model.pt` | 1,033,821,799 | 0.96 GB |
| `checkpoints/exp2_panderm/checkpoint_epoch_1.pt` | 1,033,828,367 | 0.96 GB |
| `checkpoints/exp2_panderm/last_model.pt` | 1,033,821,799 | 0.96 GB |
| `checkpoints/exp2_panderm_v2/best_model.pt` | 1,028,817,790 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/best_model.pt` | 1,028,817,131 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/checkpoint_epoch_1.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/checkpoint_epoch_2.pt` | 1,028,823,091 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/checkpoint_epoch_3.pt` | 1,028,823,155 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/checkpoint_epoch_4.pt` | 1,028,823,155 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/checkpoint_epoch_5.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib/last_model.pt` | 1,028,817,195 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/best_model.pt` | 1,028,817,195 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/checkpoint_epoch_1.pt` | 1,028,823,155 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/checkpoint_epoch_2.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/checkpoint_epoch_3.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/checkpoint_epoch_4.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/checkpoint_epoch_5.pt` | 1,028,823,219 | 0.96 GB |
| `checkpoints/exp2_panderm_v2_calib2/last_model.pt` | 1,028,817,195 | 0.96 GB |
| `results/exp2c_panderm_uniform_seed42/best_model (1).pt` | 1,028,819,326 | 0.96 GB |

---

## 7. Decisions for the User

1. **Do you approve restoring the Exp-4a implementation from `exp4a-wip` into `main`?**
2. **Do you approve allocating ~2.0 Kaggle GPU-hours to run the 4-value $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$ validation sweep on seed 42, followed by ~0.50h for a seed 43 confirmation run?**
3. **For Exp-4b (TSC), do you approve using fixed deterministic target binding rather than Hungarian matching, given that PanDerm is an already pretrained clinical foundation model?**
4. **For Phase 5, should we proceed with the 7 existing continuous MONET concept features in `data/supplements/training_input.csv` as targets for the concept bottleneck head?**

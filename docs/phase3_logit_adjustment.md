# Phase 3: Logit Adjustment for Extreme Class Imbalance & Fairness

**Author**: Senior Machine Learning Engineering & Research Science Lead  
**Project**: SkinCancerTracker / DermaAI  
**Specification Version**: Phase 3 Deliverable & Comprehensive Experimental Analysis  
**Reference Paper**: Menon et al., *"Long-Tail Learning via Logit Adjustment"*, ICLR 2021 (arXiv:2007.07314)  
**Evaluated Artifacts**:
- Exp-2 (v2) Checkpoint: `checkpoints/exp2_panderm_v2/best_model.pt` (Epoch 26/30)
- Exp-3a Deliverables: `results/exp3a_panderm_posthoc/`
- Exp-3b Kaggle Pipeline: `kaggle/run_exp3b_panderm_logit_adj.py` & `configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml`

---

## 1. Scientific Motivation & Problem Statement

Phase 2 established our PanDerm ViT-Base baseline (Exp-2 v2), achieving Top-1 Accuracy of $64.38\%$, Balanced Accuracy of $39.34\%$, Macro-F1 of $0.3925$, and Malignant Recall of $89.02\%$. However, deep disaggregation revealed severe limitations:

1. **Extreme Class Imbalance Skew ($280.2 : 1$)**:
   In our training dataset of $3,668$ dermoscopic lesions, Basal Cell Carcinoma (`BCC`) constitutes $1,757$ lesions ($47.90\%$), whereas rare malignancies (`MAL_OTH`) have only $6$ lesions ($0.16\%$) and `BEN_OTH` has $27$ ($0.74\%$).
2. **Minority Class Recall Deficits**:
   Exp-2 (v2) exhibited $0.0000$ recall across all five rare tail classes (`BEN_OTH`, `DF`, `INF`, `MAL_OTH`, `VASC`) in the Tone 3 subgroup, and zero recall on `INF` and `MAL_OTH` globally. Missed malignant lesions overwhelmingly collapsed into common benign classes (`BKL` with 33 cases and `NV` with 20 cases).
3. **Tone 3 Skew Dynamics**:
   The largest demographic cohort in the test set, Tone 3 ($N=486$, $61.8\%$ of total), suffered from a depressed Macro-F1 ($0.2718$) primarily driven by local BCC concentration ($53.50\%$).

Phase 3 implements **Logit Adjustment** (Menon et al., ICLR 2021), which shifts model decision boundaries by training class priors to target balanced error directly.

---

## 2. Mathematical Formulations

### 2.1 Post-Hoc Logit Adjustment (Inference-Time)

Given raw, unnormalized diagnostic logits $z = f(x) \in \mathbb{R}^K$ produced by the neural network and empirical training class priors $\pi = [\pi_0, \dots, \pi_{K-1}]$ where:
$$\pi_k = \frac{N_{k, \text{train}}}{N_{\text{train}}}, \quad \sum_{k=0}^{K-1} \pi_k = 1$$
The post-hoc adjusted prediction is:
$$\hat{y}_{\text{post}} = \arg\max_{k \in \{0, \dots, K-1\}} \Big( z_k - \tau \cdot \log \pi_k \Big)$$
where $\tau \ge 0$ is a temperature hyperparameter controlling the strength of the adjustment.

The adjusted softmax probabilities are:
$$\tilde{p}_k = \frac{\exp(z_k - \tau \log \pi_k)}{\sum_{j=0}^{K-1} \exp(z_j - \tau \log \pi_j)} = \frac{p_k \cdot \pi_k^{-\tau}}{\sum_{j=0}^{K-1} p_j \cdot \pi_j^{-\tau}}$$

### 2.2 Train-Time Logit Adjustment (Loss-Level)

Train-time logit adjustment incorporates class margins directly into the loss function during backpropagation:
$$\mathcal{L}_{\text{LA}}(y, z) = -\log \frac{\exp(z_y + \tau \log \pi_y)}{\sum_{k=0}^{K-1} \exp(z_k + \tau \log \pi_k)}$$
Combined with label smoothing $\epsilon \in [0, 1)$:
$$\mathcal{L}_{\text{LA-LS}}(y, z) = (1 - \epsilon) \cdot \left[ -\log \frac{\exp(z_y + \tau \log \pi_y)}{\sum_k \exp(z_k + \tau \log \pi_k)} \right] + \frac{\epsilon}{K} \sum_{j=0}^{K-1} \left[ -\log \frac{\exp(z_j + \tau \log \pi_j)}{\sum_k \exp(z_k + \tau \log \pi_k)} \right]$$

**Mathematical Rationale**:
Minimizing $\mathcal{L}_{\text{LA}}$ forces the model to separate class $y$ from class $k$ by an additive log-ratio margin:
$$z_y - z_k \ge \tau \cdot \log\left(\frac{\pi_k}{\pi_y}\right)$$
When trained with uniform sampling under the natural imbalanced distribution, Menon et al. (ICLR 2021) prove that minimizing this margin-enforcing loss yields a classifier whose standard $\arg\max_k z_k$ inference on raw logits is Fisher-consistent for balanced error.

---

## 3. Training Set Class Priors ($\pi$)

Evaluated on the $N=3,668$ dermoscopic training partition (`data/splits/lesion_level_splits.csv`):

| Class Index | Pathology Name | Clinical Type | Training Count ($N_c$) | Prior ($\pi_c$) | $\log(\pi_c)$ | Imbalance vs BCC |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| 0 | **AKIEC** | Malignant | 202 | 0.05507 | -2.899 | $8.7\times$ |
| 1 | **BCC** | Malignant | 1,757 | 0.47901 | -0.736 | $1.0\times$ (dominant) |
| 2 | **BEN_OTH** | Benign | 27 | 0.00736 | -4.912 | $65.1\times$ |
| 3 | **BKL** | Benign | 385 | 0.10496 | -2.254 | $4.6\times$ |
| 4 | **DF** | Benign | 39 | 0.01063 | -4.544 | $45.1\times$ |
| 5 | **INF** | Benign | 38 | 0.01036 | -4.570 | $46.2\times$ |
| 6 | **MAL_OTH** | Malignant | 6 | 0.00164 | -6.416 | **$292.8\times$** |
| 7 | **MEL** | Malignant | 325 | 0.08860 | -2.424 | $5.4\times$ |
| 8 | **NV** | Benign | 521 | 0.14204 | -1.952 | $3.4\times$ |
| 9 | **SCCKA** | Malignant | 332 | 0.09051 | -2.402 | $5.3\times$ |
| 10 | **VASC** | Benign | 36 | 0.00981 | -4.624 | $48.8\times$ |
| **Total** | | | **3,668** | **1.00000** | | |

---

## 4. Exp-3a: Validation Tuning & Trajectory Analysis

Hyperparameter $\tau$ was tuned on the held-out validation split ($N=786$ lesions), sweeping $\tau \in [0.00, 2.00]$ in increments of $0.05$:

```
tau    Val Acc    Val Bal Acc  Val Macro-F1   Val W-F1   Val Mal Sens Val Mal FNR 
--------------------------------------------------------------------------------
0.00   0.6209     0.4380       0.4064         0.6331     0.8714       0.1286      (Baseline)
0.05   0.6132     0.4365       0.4026         0.6269     0.8678       0.1322      
0.10   0.6145     0.4508       0.4117         0.6291     0.8659       0.1341      <-- OPTIMAL (tau*)
0.15   0.6094     0.4503       0.4104         0.6254     0.8641       0.1359      
0.20   0.6018     0.4471       0.4057         0.6198     0.8605       0.1395      
0.25   0.5992     0.4466       0.4026         0.6181     0.8587       0.1413      
0.30   0.5954     0.4467       0.3969         0.6162     0.8514       0.1486      
0.40   0.5852     0.4482       0.3934         0.6100     0.8388       0.1612      
0.50   0.5687     0.4450       0.3772         0.5978     0.8297       0.1703      
0.75   0.4618     0.4169       0.3491         0.5191     0.8207       0.1793      
1.00   0.1425     0.3741       0.2450         0.1568     0.9275       0.0725      (Severe Degradation)
1.50   0.0153     0.2400       0.1145         0.0112     0.9710       0.0290      
2.00   0.0115     0.2026       0.1084         0.0106     0.9855       0.0145      
```

- Validation Macro-F1 peaked at $\tau^* = 0.100$ (`val_macro_f1 = 0.4117`, `val_balanced_acc = 0.4508`).
- For $\tau > 0.25$, accuracy drops precipitously because the empirical imbalance penalty pushes extreme offsets on tail classes (e.g. $\log \pi_{\text{MAL\_OTH}} = -6.416$), causing severe false-positive flooding.
- $\tau^* = 0.100$ was frozen for held-out test evaluation.

---

## 5. Exp-3a: Test Set Evaluation & NULL Result Assessment

The table below summarizes test set performance ($N=786$) comparing Exp-2 (v2) with Exp-3a ($\tau^* = 0.100$). Statistical significance is tested via **2,000 paired bootstrap iterations**:

| Metric | Exp-2 (v2) Baseline | Exp-3a Post-Hoc ($\tau^*=0.10$) | $\Delta$ (Exp-3a - Exp-2) | Paired 95% Bootstrap CI | Outcome Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Top-1 Accuracy** | **0.6438** | 0.6361 | $-0.0076$ | $[-0.015, +0.000]$ | Marginal decrease |
| **Balanced Accuracy** | 0.3934 | 0.3921 | $-0.0014$ | $[-0.006, +0.004]$ | **NULL (CI includes 0)** |
| **Macro-F1** | **0.3925** | 0.3883 | $-0.0042$ | $[-0.010, +0.001]$ | **NULL (CI includes 0)** |
| **Weighted F1** | 0.6469 | 0.6426 | $-0.0043$ | $[-0.011, +0.001]$ | NULL (CI includes 0) |
| **Malignant Recall** | **0.8902** | 0.8816 | $-0.0086$ | $[-0.017, -0.002]$ | Small loss (5 more false negatives) |
| **Malignant FNR** | **0.1098** | 0.1184 | $+0.0086$ | $[+0.002, +0.017]$ | Small increase |
| **Melanoma (`MEL`) Recall** | 0.6866 | **0.7015** | **$+0.0149$** | $[+0.000, +0.030]$ | **1 lesion flipped (47/67 vs 46/67)** |
| **Melanoma (`MEL`) F1** | 0.5750 | **0.5839** | $+0.0089$ | $[-0.005, +0.022]$ | NULL (CI includes 0) |
| **3-Bucket FNR Disparity** | 6.06% | 4.94% | $-1.12\%$ | $[-2.5\%, -0.1\%]$ | **Not statistically reliable (see §7)** |
| **3-Bucket Macro-F1 Gap** | 0.1908 | 0.1599 | $-0.0309$ | $[-0.060, -0.005]$ | Small narrowing |

### Critical Assessment:
1. **Exp-3a is a NULL result**: The primary metric, $\Delta \text{Macro-F1}$, has a 95% confidence interval of $[-0.010, +0.001]$, which includes zero. Balanced Accuracy delta ($[-0.006, +0.004]$) likewise spans zero.
2. **Melanoma Recall $+1.49\%$ is a Single Lesion**: There are 67 Melanoma lesions in the test split. Exp-2 v2 detected 46, while Exp-3a detected 47 ($47/67 - 46/67 = 1/67 \approx +0.0149$). An increase of one lesion is not evidence of a systematic clinical shift.
3. **FNR Disparity Change is Not Statistically Reliable**: While the numerical gap narrowed from $6.06\%$ to $4.94\%$, the Dark skin-tone bucket contains only $N=12$ lesions. A single flipped lesion in the Dark bucket shifts its FNR by $1/12 \approx 8.33$ percentage points, dwarfing the observed $1.12\%$ disparity change.

---

## 6. Diagnostic: Why Post-Hoc LA Double-Corrects Exp-2 (v2)

### 6.1 Resampling in Exp-2 v2
Inspection of `configs/experiments/exp2_panderm_kaggle.yaml` (line 19) and `src/data/loaders.py` (lines 91–108) confirms:
```python
# src/data/loaders.py: lines 97-106
weight_dict = {
    cls: total_samples / (num_classes * count)
    for cls, count in label_counts.items()
}
sample_weights = torch.tensor([weight_dict[y] for y in labels], dtype=torch.double)
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(train_dataset),
    replacement=True,
)
```
Exp-2 v2 was fine-tuned using `WeightedRandomSampler`, where samples were drawn inversely proportional to training frequency ($w_c \propto 1/N_c$). The Vision Transformer was already exposed to an approximately balanced expected class distribution ($\approx 1/11$ per class) during its 30 training epochs.

### 6.2 Predicted vs True Class Distributions
Examining the output distributions on validation and test confirms that Exp-2 v2 was already outputting balanced predictions:

#### Validation Split ($N=786$)
| Class | Val True Count | Val True % | Exp-2 v2 Pred Count | Exp-2 v2 Pred % |
| :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | 47 | 6.0% | 58 | 7.4% |
| **BCC** | 374 | 47.6% | 333 | 42.4% |
| **BEN_OTH** | 9 | 1.1% | 5 | 0.6% |
| **BKL** | 79 | 10.1% | 90 | 11.5% |
| **DF** | 8 | 1.0% | 9 | 1.1% |
| **INF** | 7 | 0.9% | 6 | 0.8% |
| **MAL_OTH** | 1 | 0.1% | 0 | 0.0% |
| **MEL** | 58 | 7.4% | 74 | 9.4% |
| **NV** | 126 | 16.0% | 104 | 13.2% |
| **SCCKA** | 72 | 9.2% | 94 | 12.0% |
| **VASC** | 5 | 0.6% | 13 | 1.7% |

#### Test Split ($N=786$)
| Class | Test True Count | Test True % | Exp-2 v2 Pred Count | Exp-2 v2 Pred % |
| :--- | :---: | :---: | :---: | :---: |
| **AKIEC** | 54 | 6.9% | 55 | 7.0% |
| **BCC** | 391 | 49.7% | 354 | 45.0% |
| **BEN_OTH** | 8 | 1.0% | 5 | 0.6% |
| **BKL** | 80 | 10.2% | 66 | 8.4% |
| **DF** | 5 | 0.6% | 2 | 0.3% |
| **INF** | 5 | 0.6% | 7 | 0.9% |
| **MAL_OTH** | 2 | 0.3% | 0 | 0.0% |
| **MEL** | 67 | 8.5% | 93 | 11.8% |
| **NV** | 99 | 12.6% | 93 | 11.8% |
| **SCCKA** | 69 | 8.8% | 106 | 13.5% |
| **VASC** | 6 | 0.8% | 5 | 0.6% |

Applying post-hoc logit adjustment $z_k - \tau \log \pi_k$ using the raw training imbalance prior $\pi_k$ assumes the model was trained with standard ERM under prior $\pi_k$. Subtracting $-\tau \log \pi_k$ on a model whose classification head was already trained on resampled balanced batches double-corrects for class imbalance, over-penalizing majority classes.

### 6.3 Mechanism Status & Scientific Downgrade
Double-correction is **supported by mechanism (WeightedRandomSampler confirmed) and BCC under-prediction; to be tested by controlled Exp-2c vs Exp-3b**. 

Whether the frozen backbone exhibits internal feature representation collapse on rare classes is **not established**, as latent feature geometries have not been directly probed via probing classifiers or linear CKA. Controlled comparison between Exp-2c (uniform shuffle baseline) and Exp-3b (uniform shuffle + train-time LA loss) will provide the definitive test of whether loss-level logit adjustment outperforms balanced resampling without confounding.

---

## 7. Subgroup Demographic Fairness Auditing

| Subgroup | Test $N$ | Low Confidence? | Exp-2 Bal Acc | Exp-3a Bal Acc | Exp-2 Macro-F1 | Exp-3a Macro-F1 (95% CI) | Exp-2 Mal FNR | Exp-3a Mal FNR (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dark (0-1)** | 12 | **True** ($n < 30$) | 0.5833 | 0.5833 | 0.4626 | 0.4626 [0.241, 0.738] | 16.67% | 16.67% [0.000, 0.500] |
| **Medium (2-3)** | 558 | False | 0.3421 | 0.3406 | 0.3397 | 0.3379 [0.274, 0.415] | 10.60% | 11.81% [0.087, 0.152] |
| **Light (4-5)** | 216 | False | 0.5443 | 0.5443 | 0.5305 | 0.4978 [0.351, 0.579] | 11.73% | 11.73% [0.071, 0.172] |
| **Disparity** | | | **0.2427** | **0.2427** | **0.1908** | **0.1599** | **6.06%** | **4.94%** |

*Methodological Note*: The Dark bucket contains only $N=12$ lesions (all Tone 1; Tone 0 has zero samples in MILK10k). A single misclassified lesion alters the Dark FNR by $\approx 8.33\%$. Therefore, claims of "statistically significant" disparity narrowing are unwarranted and flagged as low-confidence.

---

## 8. Redesigned Exp-3b: Train-Time Logit Adjustment on Kaggle

To properly test Menon et al. (ICLR 2021), Exp-3b must train with **uniform shuffling (no WeightedRandomSampler)** under the natural imbalanced class distribution, allowing `LogitAdjustedCrossEntropy` to enforce margins directly:

### 8.1 Configuration Diff: Exp-2 v2 vs Exp-3b
The ONLY differences between `exp2_panderm_kaggle.yaml` and `exp3b_panderm_logit_adj_kaggle.yaml`:
1. `dataloader.weighted_sampler`: `true` $\to$ `false` (standard uniform shuffling).
2. `loss.name`: `label_smoothing` $\to$ `logit_adjusted`.
3. `loss.tau`: added `1.0` (with class priors computed from training counts at runtime).
4. `loss.label_smoothing`: `0.1` preserved identically to Exp-2 v2.

### 8.2 Evaluation Rule
Per Menon et al. (ICLR 2021) Section 4:
- During training, loss enforces additive margins $z_y + \tau \log \pi_y$.
- During evaluation (`validate()` and `generate_predictions()`) and checkpoint selection (`val_macro_f1`), predictions and metrics **MUST use raw logits $z$ without adjustment**. Standard $\arg\max_k z_k$ on raw logits is Fisher-consistent for balanced error.

### 8.3 Kaggle Execution Steps & Resource Estimate
- **Estimated Runtime**: $\approx 29$ minutes on Tesla T4 or P100 16GB GPU ($\approx 0.5$ GPU hours).
- **Execution Command**:
  ```bash
  python kaggle/run_exp3b_panderm_logit_adj.py \
      --config configs/experiments/exp3b_panderm_logit_adj_kaggle.yaml \
      --image-dir /kaggle/input/datasets/paramveerchaudhary/milk10k-dataset/images \
      --splits-file /kaggle/input/datasets/paramveerchaudhary/milk10k-dataset/lesion_level_splits.csv \
      --checkpoint-path /kaggle/input/datasets/paramveerchaudhary/panderm-weights/panderm_base.pth \
      --checkpoint-dir /kaggle/working/checkpoints/exp3b_panderm_logit_adj \
      --results-dir /kaggle/working/results/exp3b_panderm_logit_adj \
      --tau 1.0 \
      --batch-size 64 \
      --lr 2.5e-4 \
      --online
  ```

---

## 9. Master Ablation Table

| Exp ID | Model Architecture | Modality | Top-1 Acc | Macro-F1 | Malignant Recall | AUROC (Macro / Mal) | Notes |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Exp-1** | ResNet-50 | Dermoscopic | 0.5191 | 0.2869 | 0.8868 | 0.7758 / 0.7549 | 1 epoch local reference; 30-epoch Kaggle baseline script ready |
| **Exp-2 (v1)** | PanDerm ViT-Base | Dermoscopic | 0.5344 | 0.1091 | 0.8165 | 0.6253 / 0.6699 | Unweighted sampler collapse onto BCC/NV |
| **Exp-2 (v2)** | PanDerm ViT-Base | Dermoscopic | **0.6438** | **0.3925** | **0.8902** | 0.7547 / **0.8022** | Corrected recipe (weighted sampler, layer decay 0.9, val-macro-F1 ckpt) |
| **Exp-3a** | PanDerm ViT-Base | Dermoscopic | 0.6361 | 0.3883 | 0.8816 | 0.7403 / 0.7964 | NULL result (Delta F1 CI includes 0; double-corrects resampled model; MEL +1.49% is 1 lesion) |
| **Exp-3b** | PanDerm ViT-Base | Dermoscopic | *Pending GPU* | *Pending GPU* | *Pending GPU* | *Pending GPU* | Train-time LA loss (tau=1.0, uniform shuffle, LS=0.1); script verified |

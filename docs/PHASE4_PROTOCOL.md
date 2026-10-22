# Phase 4 Stability-Aware Execution Protocol

**Document Status**: Official Phase 4 Specification & Execution Protocol  
**Pre-requisite Audit**: `docs/PHASE4_DECISION_PACK.md` & `docs/EXP3B_DIAGNOSIS.md`  
**Protocol Version**: 1.0 (Stability-Aware & Collapse-Guarded)

---

## 1. Executive Summary & Design Principles

Phase 4 introduces clinically structured inductive biases into foundation model fine-tuning on the MILK10k dataset ($N=5,240$ lesions, 11 diagnostic classes):
1. **Exp-4a**: Asymmetric Hierarchical Cost Loss ($\mathcal{L}_{\text{AHC}}$) penalizing false negatives (malignant classified as benign) $3.33\times$ higher than false positives, with intra-superclass penalty structure.
2. **Exp-4b**: Targeted Supervised Contrastive Learning ($\mathcal{L}_{\text{TSC}}$) enforcing uniform hypersphere separation across class prototypes.

### Core Principles Established by Phase 2/3 Audits:
- **Base Recipe**: Phase 4 builds strictly upon **Exp-2c** (PanDerm ViT-Base, uniform shuffling, label smoothing $\epsilon=0.1$, AdamW with layer decay 0.9, lr $2.5\times 10^{-4}$), which won the base recipe selection protocol with validation Macro-F1 $0.5064$ (+10.0% over weighted sampling).
- **Collapse Guarding**: Following the Exp-3b diagnosis where default margin penalties caused majority collapse onto rare classes ($D(\text{BCC}, \text{MAL\_OTH}) = +1.0744$), all Phase 4 recipes require runtime gating to prevent degradation of majority clinical sensitivity.
- **Dual-Track Metrics**: Model selection uses validation Macro-F1, while clinical reporting tracks stability-aware metrics (Malignant AUROC and Malignant FNR at matched operational specificities $S \in \{0.50, 0.60, 0.70\}$).

---

## 2. Exp-4a: Asymmetric Hierarchical Cost Loss Protocol

### 2.1 Mathematical Formulation
The loss function minimizes task cross-entropy plus the expected hierarchical penalty:
$$\mathcal{L}_{\text{total}}(\mathbf{x}, y) = \mathcal{L}_{\text{LS-CE}}(\mathbf{x}, y) + \alpha \sum_{k=1}^K C(y, k) \cdot p_k(\mathbf{x})$$
where $p_k(\mathbf{x}) = \operatorname{softmax}(\mathbf{x})_k$, and $C(y, k)$ is the clinical cost matrix:
- **Malignant $\to$ Benign (False Negative)**: $c_{\text{fn}} = 5.0$
- **Benign $\to$ Malignant (False Positive)**: $c_{\text{fp}} = 1.5$
- **Within Malignant**: $c_{\text{within\_mal}} = 1.0$ (e.g. BCC $\to$ MEL)
- **Within Benign**: $c_{\text{within\_ben}} = 0.5$ (e.g. NV $\to$ BKL)
- **Correct Diagnosis ($y = k$)**: $0.0$

### 2.2 Validation Hyperparameter Sweep ($\alpha$)
To prevent the cost penalty from dominating cross-entropy gradients:
- **Sweep Grid**: $\alpha \in \{0.25, 0.50, 1.00, 2.00\}$ on random seed 42.
- **Compute Budget**: 4 runs $\times$ ~1,800s wall-clock = ~2.0 Kaggle GPU-hours on Tesla T4.
- **Selection Criterion**: $\alpha^*$ is chosen to maximize validation Macro-F1 subject to the stability gate:
  $$\text{Validation BCC Recall} \ge 0.80 \quad \text{and} \quad \text{Validation Malignant FNR} \le 0.12$$
- **Confirmation Run**: Execute $\alpha^*$ on seed 43 (~0.50 GPU-hours) before testing.

---

## 3. Exp-4b: Targeted Supervised Contrastive Learning Protocol

### 3.1 Mathematical Formulation & Target Binding
Based on the CVPR 2022 paper (Li et al., arXiv:2111.13998) and codebase audit:
- Target vectors $\mathbf{T} \in \mathbb{R}^{K \times d}$ ($K=11, d=768$) are pre-computed on the unit hypersphere $\mathbb{S}^{d-1}$ by minimizing the pairwise Gaussian potential uniformity loss:
  $$\mathcal{L}_{\text{uniform}}(\mathbf{T}) = \frac{1}{K} \sum_{i=1}^K \log \left( \sum_{j=1}^K \exp\left( \frac{\mathbf{t}_i^\top \mathbf{t}_j}{t} \right) \right)$$
- **Target Assignment**: In our medical fine-tuning setting ($K=11$ on a pre-trained dermatology foundation model), targets are **fixed deterministically to class indices** by row ($\mathbf{t}_c = \mathbf{T}_{c, :}$). Dynamic Hungarian matching is explicitly avoided because re-assigning targets between epochs destabilizes pre-trained representation alignment.

### 3.2 Joint Fine-Tuning Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{TSC}}$$
$$\mathcal{L}_{\text{TSC}}(\hat{\mathbf{z}}_i, y_i) = -\log \frac{\exp\left(\frac{\hat{\mathbf{z}}_i^\top \mathbf{t}_{y_i}}{\tau}\right)}{\sum_{c=1}^K \exp\left(\frac{\hat{\mathbf{z}}_i^\top \mathbf{t}_c}{\tau}\right)}$$
where $\hat{\mathbf{z}}_i$ is the normalized feature embedding from the ViT backbone before the linear classification head, and temperature $\tau = 0.1$.
- **Sweep Grid**: $\lambda \in \{0.05, 0.10, 0.20\}$ on seed 42.
- **Compute Budget**: 3 runs $\times$ ~0.50h = ~1.5 Kaggle GPU-hours.

---

## 4. Evaluation & Statistical Verification Contract

### 4.1 Required Metrics Per Run
Every completed Phase 4 run must compute and serialize:
1. **Multi-Class Classification**: Macro-F1, Weighted-F1, Overall Accuracy, Balanced Accuracy, Per-Class Precision/Recall/F1/Support (11 classes).
2. **Binary Clinical Triage**: Malignant Sensitivity, Malignant Specificity, Malignant FNR ($1 - \text{Sens}$), FN count, FP count.
3. **Stable Clinical Metrics**:
   - Malignant AUROC ($p_{\text{mal}} = \sum_{c \in \text{malignant}} p_c$).
   - Malignant FNR at operational specificities $\{0.50, 0.60, 0.70\}$ with threshold calibrated on validation benign lesions and applied to test.
4. **Subgroup Fairness**: Macro-F1, Balanced Accuracy, and Malignant FNR disaggregated across Fitzpatrick skin tone classes (`Tone 1`, `Tone 2`, `Tone 3`, `Tone 4`).
5. **Data Provenance**: Explicit serialization of `split_sha256: "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"` inside `metrics.json`.

### 4.2 Statistical Significance Testing
- Paired cluster bootstrap resampling ($B=2,000$ iterations) against Exp-2c baseline at lesion level.
- Claim of improvement requires $95\%$ bootstrap percentile confidence interval of $\Delta \text{Metric}$ to strictly exclude $0.0$.

# Phase 4 Stability-Aware Execution Protocol

**Document Status**: Official Phase 4 Specification & Execution Protocol  
**Pre-requisite Audit**: `docs/PHASE4_DECISION_PACK.md` & `docs/EXP3B_DIAGNOSIS.md`  
**Protocol Version**: 2.0 (Stability-Aware, Multi-Seed & Calibrated Guard)

---

## 1. Executive Summary & Design Principles

Phase 4 introduces clinically structured inductive biases into foundation model fine-tuning on the MILK10k dataset ($N=5,240$ lesions, 11 diagnostic classes):
1. **Exp-4a**: Asymmetric Hierarchical Cost Loss ($\mathcal{L}_{\text{AHC}}$) penalizing false negatives (malignant classified as benign) $3.33\times$ higher than false positives, with intra-superclass penalty structure.
2. **Exp-4b**: Targeted Supervised Contrastive Learning ($\mathcal{L}_{\text{TSC}}$) enforcing uniform hypersphere separation across class prototypes.

### Core Principles Established by Phase 2/3 Audits:
- **Base Recipe**: Phase 4 builds strictly upon **Exp-2c** (PanDerm ViT-Base, uniform shuffling, label smoothing $\epsilon=0.1$, AdamW with layer decay 0.9, lr $2.5\times 10^{-4}$).  
  *Justification*: Exp-2c is the simplest recipe (standard uniform shuffling without sampling weights or prior adjustments) and its test Macro-F1 ($0.4690$ s42, $0.3636$ s43) is not worse than alternative baseline recipes. The earlier "5x noise" validation argument is explicitly rejected.
- **Seeds Protocol**: All Phase 4 arms will be evaluated across **three random seeds: 42, 43, and 44** to resolve inter-seed variance.
- **Primary Clinical Evaluation Metrics**:
  1. **Malignant FNR at Matched Specificity ($S=0.60$)**: Clinical triage sensitivity when false-alarm rates are constrained on benign lesions.
  2. **Malignant AUROC ($p_{\text{mal}} = \sum_{c \in \text{malignant}} p_c$)**: Threshold-free discriminative performance distinguishing malignant from benign lesions.
  3. **Multi-Class Macro-F1**: Monitored as standard benchmark metric, but acknowledged as sensitive to single-sample volatility in extreme tail classes.
- **Collapse Guard**: Validation threshold guard is set to **USER TO DECIDE** (see Section 2.3 for Exp-2c reference numbers).

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
- **Selection Criterion**: $\alpha^*$ is chosen to optimize validation performance subject to the collapse guard.
- **Confirmation Runs**: Execute $\alpha^*$ on seeds 43 and 44 (~1.0 GPU-hours) before test split reporting.

### 2.3 Collapse Guard & Exp-2c Reference Numbers
To prevent majority collapse (as observed in Exp-3b where BCC recall collapsed to 0.0), a validation triage guard is applied during checkpoint selection:
- **Validation Specificity Guard Threshold**: **[USER TO DECIDE]**
- **Exp-2c Reference Performance** (for setting the decision threshold):
  - **Exp-2c Seed 42**:
    - Malignant AUROC: $0.8418$ `[0.8071, 0.8735]`
    - Validation $\theta_{0.60} = 0.4446$ $\to$ Test Malignant FNR = **0.0446** `[0.0279, 0.0628]` (Test Spec: 0.4581)
    - Test BCC Recall: $0.9105$ | Test NV Recall: $0.7374$
  - **Exp-2c Seed 43**:
    - Malignant AUROC: $0.8676$ `[0.8399, 0.8949]`
    - Validation $\theta_{0.60} = 0.4969$ $\to$ Test Malignant FNR = **0.0497** `[0.0330, 0.0687]` (Test Spec: 0.4631)
    - Test BCC Recall: $0.8363$ | Test NV Recall: $0.6970$
- *Recommended Guard Default*: Assert Validation BCC Recall $\ge 0.80$ and Validation Malignant FNR $\le 0.08$ at $\theta_{0.60}$.

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
- **Confirmation Runs**: Seeds 43 and 44 on optimal $\lambda^*$.
- **Compute Budget**: 3 sweep runs + 2 confirm runs = ~2.5 Kaggle GPU-hours.

---

## 4. Evaluation & Statistical Verification Contract

### 4.1 Required Metrics Per Run
Every completed Phase 4 run must compute and serialize:
1. **Primary Clinical Metrics**:
   - Malignant FNR at matched operational specificity $S=0.60$ (with threshold calibrated on validation benign lesions and applied to test).
   - Malignant AUROC ($p_{\text{mal}} = \sum_{c \in \text{malignant}} p_c$).
2. **Multi-Class Classification**: Macro-F1, Weighted-F1, Overall Accuracy, Balanced Accuracy, Per-Class Precision/Recall/F1/Support (11 classes).
3. **Binary Clinical Triage**: Malignant Sensitivity, Malignant Specificity, FN count, FP count.
4. **Subgroup Fairness**: Macro-F1, Balanced Accuracy, and Malignant FNR disaggregated across MILK10k skin tone classes (`Tone 1`, `Tone 2`, `Tone 3`, `Tone 4`, `Tone 5`).
5. **Data Provenance**: Explicit serialization of `split_sha256: "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"` inside `metrics.json`.

### 4.2 Statistical Significance Testing
- Paired cluster bootstrap resampling ($B=2,000$ iterations) against Exp-2c baseline at lesion level.
- Claim of improvement requires $95\%$ bootstrap percentile confidence interval of $\Delta \text{Metric}$ to strictly exclude $0.0$.

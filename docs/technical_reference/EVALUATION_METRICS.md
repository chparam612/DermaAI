# Clinical Evaluation Metrics & Fairness Auditing Reference

**Author**: Senior Machine Learning Engineering & Research Science Reference  
**Project**: SkinCancerTracker / DermaAI  
**Target Audience**: B.Tech CSE / AI Researchers, Faculty Evaluators, Technical Interviewers  
**Specification Version**: Phase 1 & Phase 2 Technical Baseline  

---

## 1. The Confusion Matrix in Dermatology Diagnosis

For multi-class classification over $K=11$ diagnostic classes, the confusion matrix $C \in \mathbb{Z}_{\ge 0}^{K \times K}$ records the counts where row $i$ represents true pathology and column $j$ represents model prediction:
$$C_{i, j} = \sum_{m=1}^N \mathbb{I}(y_m = i \land \hat{y}_m = j)$$

For any specific target class $c$ (e.g. Melanoma `MEL`), the multi-class matrix can be decomposed into a binary One-vs-Rest (OvR) contingency table:

```
                      PREDICTED POSITIVE (MEL)      PREDICTED NEGATIVE (Non-MEL)
ACTUAL POSITIVE (MEL)          True Positive (TP)             False Negative (FN)
ACTUAL NEGATIVE (Non-MEL)     False Positive (FP)             True Negative (TN)
```

### Definitions and Medical Implications:
- **True Positive ($TP$)**: Malignant melanoma correctly classified as `MEL`. Patient receives urgent surgical excision and oncology staging.
- **False Negative ($FN$)**: Malignant melanoma misclassified as benign nevus (`NV`) or benign keratosis (`BKL`). **Catastrophic clinical outcome**: Patient sent home untreated; melanoma metastasizes with fatal prognosis.
- **False Positive ($FP$)**: Benign lesion misclassified as melanoma. Patient undergoes unnecessary surgical biopsy, tissue trauma, and psychological stress.
- **True Negative ($TN$)**: Benign lesion correctly identified as non-melanoma.

---

## 2. Comprehensive Metric Formulation Table

| Metric Name | Mathematical Formula | Implementation Location | Medical Interpretation in Skin Cancer |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\frac{\sum_{k=0}^{K-1} C_{k, k}}{\sum_{i=0}^{K-1} \sum_{j=0}^{K-1} C_{i, j}}$ | [`src/eval/metrics.py:47`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L47) (`accuracy_score`) | Fraction of total lesions correctly classified. **Dangerously misleading** under severe class skew. |
| **Balanced Accuracy** | $\frac{1}{K} \sum_{k=0}^{K-1} \text{Recall}_k = \frac{1}{K} \sum_{k=0}^{K-1} \frac{C_{k, k}}{\sum_j C_{k, j}}$ | [`src/eval/metrics.py:48`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L48) (`balanced_accuracy_score`) | Macro-average of recall across all 11 classes. Treats rare classes with equal weight to common classes. |
| **Sensitivity (Recall)** | $\frac{TP_k}{TP_k + FN_k} = \frac{C_{k, k}}{\sum_j C_{k, j}}$ | [`src/eval/metrics.py:56`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L56) (`recall_score`) | Clinical detection rate: Ability to identify true malignancies without missing them ($FN \to 0$). |
| **Specificity** | $\frac{TN_k}{TN_k + FP_k}$ | [`external/panderm/.../engine_for_finetuning.py:42`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/external/panderm/classification/furnace/engine_for_finetuning.py#L42) | Ability to rule out healthy or benign tissue, minimizing false alarms and unnecessary biopsies. |
| **Precision (PPV)** | $\frac{TP_k}{TP_k + FP_k} = \frac{C_{k, k}}{\sum_i C_{i, k}}$ | [`src/eval/metrics.py:57`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L57) (`precision_score`) | Positive Predictive Value: When model flags a lesion as melanoma, probability it is truly malignant. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | [`src/eval/metrics.py:55`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L55) (`f1_score`) | Harmonic mean balancing sensitivity against precision. |
| **Macro F1** | $\frac{1}{K} \sum_{k=0}^{K-1} \text{F1}_k$ | [`src/eval/metrics.py:49`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L49) (`f1_score(average='macro')`) | Unweighted arithmetic mean of per-class F1 scores. Gold-standard benchmark for imbalanced datasets. |
| **Weighted F1** | $\sum_{k=0}^{K-1} \left( \frac{N_k}{N} \right) \text{F1}_k$ | [`src/eval/metrics.py:50`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L50) (`f1_score(average='weighted')`) | Support-weighted mean of per-class F1 scores. Biased toward majority classes (BCC, NV). |
| **ROC-AUC** | $\int_0^1 \text{Sensitivity}(t) \, d(1 - \text{Specificity}(t))$ | [`src/eval/wandb_logger.py:43`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/wandb_logger.py#L43), `sklearn.metrics.roc_auc_score` | Area Under Receiver Operating Characteristic Curve. Evaluates discrimination across all decision thresholds. |
| **Accuracy Disparity** | $\max_{g} \text{Acc}(g) - \min_{g} \text{Acc}(g)$ | [`src/eval/metrics.py:141`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L141) | Maximum gap in classification accuracy between well-represented and under-represented skin tones. |
| **BalAcc Disparity** | $\max_{g} \text{BalAcc}(g) - \min_{g} \text{BalAcc}(g)$ | [`src/eval/metrics.py:142`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/metrics.py#L142) | Maximum gap in balanced accuracy across demographic skin-tone groups ($N_g \ge 10$). |

---

## 3. Why Accuracy Alone Is Insufficient in Medical Imaging

In our dataset of $10,480$ images:
- **BCC (Basal Cell Carcinoma)**: $5,044$ images ($48.13\%$)
- **MAL_OTH (Rare Malignancies)**: $18$ images ($0.17\%$)
- **DF (Dermatofibroma)**: $104$ images ($0.99\%$)

Consider a naive baseline model that outputs $\hat{y} = \text{BCC}$ for every single input:
$$\text{Accuracy} = \frac{5,044}{10,480} = 48.13\%$$
However, evaluating Balanced Accuracy reveals the true diagnostic failure:
$$\text{Recall}_{\text{BCC}} = 1.0, \quad \text{Recall}_{k \ne \text{BCC}} = 0.0$$
$$\text{Balanced Accuracy} = \frac{1.0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0 + 0}{11} = \frac{1}{11} \approx 9.09\%$$
$$\text{Macro F1} = \frac{0.6498 + 0 + \dots + 0}{11} \approx 0.0591$$

Furthermore, on deadly Melanoma (`MEL`), the model's Sensitivity is **$0.00\%$** (all 900 Melanoma images misdiagnosed). Therefore, **Accuracy must never be used in isolation for clinical AI evaluation**. Balanced Accuracy and Macro F1 are mathematically required.

---

## 4. Key Metric Distinctions for Clinical Evaluation

### 4.1 Precision vs Recall
- **Precision**: $\frac{TP}{TP + FP}$ — Answers: *"When the algorithm alerts the doctor that a lesion is melanoma, what is the probability the patient truly has melanoma?"*
- **Recall (Sensitivity)**: $\frac{TP}{TP + FN}$ — Answers: *"Out of all patients walking into the clinic who truly have melanoma, what fraction did the algorithm successfully catch?"*
- **Clinical Tradeoff**: In oncology screening, missing a malignancy ($FN$) is fatal, while a false alarm ($FP$) only causes a biopsy. Therefore, medical triage prioritizes **Recall over Precision**.

### 4.2 Sensitivity vs Specificity
- **Sensitivity**: True Positive Rate ($TPR = TP / P$). High sensitivity rules disease *OUT* when the test is negative (clinical mnemonic: **SnNOut**).
- **Specificity**: True Negative Rate ($TNR = TN / N$). High specificity rules disease *IN* when the test is positive (clinical mnemonic: **SpPIn**).

### 4.3 Macro F1 vs Weighted F1
- **Macro F1**:
  $$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^K \text{F1}_k$$
  Ascribes equal importance ($1/11$) to `MAL_OTH` (9 lesions) and `BCC` (2,522 lesions). An algorithm cannot achieve a high Macro F1 without correctly classifying rare malignancies.
- **Weighted F1**:
  $$\text{Weighted F1} = \sum_{k=1}^K \left( \frac{N_k}{N} \right) \text{F1}_k$$
  Weights each class F1 by its prevalence ($48.1\%$ for BCC vs $0.17\%$ for MAL_OTH). A model can completely fail on rare malignancies and still report a deceptively high Weighted F1 of $>80\%$.

### 4.4 ROC-AUC vs PR-AUC
- **ROC-AUC**: Plots True Positive Rate ($y$-axis) against False Positive Rate ($x$-axis). Because the number of true negatives ($TN$) in medical datasets is enormous, $FPR = \frac{FP}{FP + TN}$ can remain deceptively small even when hundreds of false positives occur.
- **PR-AUC**: Plots Precision against Recall. Does not include $TN$ in its denominator. For rare disease detection (prevalence $<10\%$), PR-AUC provides a far more sensitive indicator of false positive degradation than ROC-AUC.

---

## 5. Weights & Biases Structured Medical Evaluation Tables

[`src/eval/wandb_logger.py`](file:///C:/Users/prtv1/OneDrive/Attachments/Desktop/SkinCancerTracker/src/eval/wandb_logger.py) formats and logs three specialized evaluation artifacts:

### Table 1: Per-Class Diagnostic Performance (`eval/per_class_metrics`)
Logs disaggregated diagnostic statistics across all 11 classes to detect minority degradation:
- Columns: `class_name`, `support`, `precision`, `recall`, `f1_score`, `specificity`, `balanced_accuracy`.

### Table 2: Per-Skin-Tone Fairness Auditing (`eval/per_skin_tone_metrics`)
Logs disaggregated metrics across demographic skin tone subgroups (`skin_tone_class` 0 through 5) to evaluate demographic equity:
- Columns: `skin_tone_group`, `support`, `accuracy`, `balanced_accuracy`, `precision`, `recall`, `f1_score`, `auc`.
- **Denominator Safety Rule**: Distinguishes unannotated data (`Unannotated / Unknown`) without fabricating zeros or skewing denominators.

### Table 3: Subgroup Error Analysis (`eval/subgroup_error_analysis`)
Logs sample-level error diagnosis for case-by-case clinical audits:
- Columns: `sample_id`, `true_label`, `predicted_label`, `confidence`, `skin_tone_group`, `error_type`.

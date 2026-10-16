"""
Phase 2.3: Detailed Local Analysis of Exp-2 PanDerm v2 Saved Predictions.
Computes top-1, balanced accuracy, Macro-F1, weighted F1, per-class P/R/F1,
confusion matrix, malignant-vs-benign metrics, skin-tone bucket fairness
with bootstrap 95% CIs, and tone 3 per-class breakdown.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

PRED_FILE = Path("results/exp2_panderm_v2/predictions.csv")
LABEL_MAPPING_FILE = Path("configs/label_mapping.json")
SUMMARY_FILE = Path("results/exp2_panderm_v2/summary_metrics.json")
CKPT_FILE = Path("checkpoints/exp2_panderm_v2/best_model.pt")

with open(LABEL_MAPPING_FILE) as f:
    label_map = json.load(f)
classes = label_map["classes"]
idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}
class_to_idx = label_map["class_to_idx"]

malignant_classes = {"BCC", "MEL", "SCCKA", "AKIEC", "MAL_OTH"}
benign_classes = {"NV", "BKL", "DF", "INF", "BEN_OTH", "VASC"}

df = pd.read_csv(PRED_FILE)
y_true = df["true_label"].values
y_pred = df["pred_label"].values

print("=" * 80)
print("1. OVERALL TEST METRICS (N = 786)")
print("=" * 80)
acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
macro_p = precision_score(y_true, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_true, y_pred, average="macro", zero_division=0)

print(f"Top-1 Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
print(f"Balanced Accuracy:  {bal_acc:.4f} ({bal_acc*100:.2f}%)")
print(f"Macro-F1:           {macro_f1:.4f}")
print(f"Weighted F1:        {weighted_f1:.4f}")
print(f"Macro Precision:    {macro_p:.4f}")
print(f"Macro Recall:       {macro_r:.4f}")

print("\n" + "=" * 80)
print("2. PER-CLASS METRICS (11 CLASSES)")
print("=" * 80)
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
p_per = precision_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
r_per = recall_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
f1_per = f1_score(y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0)
support_per = np.bincount(y_true, minlength=len(classes))
pred_counts = np.bincount(y_pred, minlength=len(classes))

print(f"{'Class':<10} {'Type':<10} {'Support':<8} {'Predicted':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-" * 75)
for idx, cls in enumerate(classes):
    ctype = "Malignant" if cls in malignant_classes else "Benign"
    print(f"{cls:<10} {ctype:<10} {support_per[idx]:<8} {pred_counts[idx]:<10} {p_per[idx]:<10.4f} {r_per[idx]:<10.4f} {f1_per[idx]:<10.4f}")

print("\n" + "=" * 80)
print("3. 11x11 CONFUSION MATRIX (Row=True, Col=Pred)")
print("=" * 80)
header = "True \\ Pred | " + " ".join(f"{c:>7}" for c in classes)
print(header)
print("-" * len(header))
for idx, cls in enumerate(classes):
    row_str = f"{cls:<11} | " + " ".join(f"{cm[idx, j]:>7}" for j in range(len(classes)))
    print(row_str)

print("\n" + "=" * 80)
print("4. BINARY MALIGNANT VS BENIGN ANALYSIS")
print("=" * 80)
# Binary definitions:
# True Malignant vs Predicted Malignant
true_is_mal = df["true_class"].isin(malignant_classes).values
pred_is_mal = df["pred_class"].isin(malignant_classes).values

# Binary confusion matrix:
# Rows: True Benign (0), True Malignant (1)
# Cols: Pred Benign (0), Pred Malignant (1)
tp = np.sum(true_is_mal & pred_is_mal)      # True Malignant predicted Malignant
fn = np.sum(true_is_mal & (~pred_is_mal))   # True Malignant predicted Benign (CRITICAL ERROR)
fp = np.sum((~true_is_mal) & pred_is_mal)   # True Benign predicted Malignant
tn = np.sum((~true_is_mal) & (~pred_is_mal))# True Benign predicted Benign

mal_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0 # Malignant Recall
mal_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0 # Benign Recall
mal_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
mal_fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0          # Malignant False Negative Rate
benign_fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0

print(f"Total Lesions:                  {len(df)}")
print(f"Total True Malignant:           {np.sum(true_is_mal)} ({np.mean(true_is_mal)*100:.1f}%)")
print(f"Total True Benign:              {np.sum(~true_is_mal)} ({np.mean(~true_is_mal)*100:.1f}%)")
print(f"True Positives (Mal -> Mal):    {tp}")
print(f"False Negatives (Mal -> Benign):{fn} <-- CRITICAL MISSED MALIGNANCIES")
print(f"False Positives (Benign -> Mal):{fp}")
print(f"True Negatives (Benign -> Ben): {tn}")
print(f"Malignant Sensitivity (Recall): {mal_sensitivity:.4f} ({mal_sensitivity*100:.2f}%)")
print(f"Malignant Specificity:          {mal_specificity:.4f} ({mal_specificity*100:.2f}%)")
print(f"Malignant Precision:            {mal_precision:.4f} ({mal_precision*100:.2f}%)")
print(f"Malignant FNR (1 - Sens):       {mal_fnr:.4f} ({mal_fnr*100:.2f}%)")

print("\nCritical False Negatives (Malignant lesions misclassified as Benign):")
fn_df = df[true_is_mal & (~pred_is_mal)]
print(fn_df[["isic_id", "true_class", "pred_class", "skin_tone_class"]].groupby(["true_class", "pred_class"]).size())

print("\n" + "=" * 80)
print("5. VALIDATION VS TEST GAP")
print("=" * 80)
import torch
# Checkpoints are produced by this repo and are trusted local files; never use this for untrusted files.
ckpt = torch.load(CKPT_FILE, map_location="cpu", weights_only=False)
val_m = ckpt.get("metrics", {})
print(f"Best Checkpoint Epoch:   {ckpt.get('epoch')}")
print(f"Train Loss:              {val_m.get('train_loss'):.4f}")
print(f"Train Accuracy:          {val_m.get('train_accuracy'):.4f} ({val_m.get('train_accuracy')*100:.2f}%)")
print(f"Val Loss:                {val_m.get('val_loss'):.4f} | Train-Val Loss Gap: {val_m.get('val_loss') - val_m.get('train_loss'):.4f}")
print(f"Val Accuracy:            {val_m.get('val_accuracy'):.4f} | Test Accuracy: {acc:.4f} (Gap: {acc - val_m.get('val_accuracy'):+.4f})")
print(f"Val Balanced Acc:        {val_m.get('val_balanced_acc'):.4f} | Test Balanced Acc: {bal_acc:.4f} (Gap: {bal_acc - val_m.get('val_balanced_acc'):+.4f})")
print(f"Val Macro-F1:            {val_m.get('val_macro_f1'):.4f} | Test Macro-F1: {macro_f1:.4f} (Gap: {macro_f1 - val_m.get('val_macro_f1'):+.4f})")
print(f"Val Weighted F1:         {val_m.get('val_weighted_f1'):.4f} | Test Weighted F1: {weighted_f1:.4f}")
print(f"Val Predicted Classes:   {val_m.get('val_num_predicted_classes')}/11 (Missing: MAL_OTH)")
print(f"Test Predicted Classes:  {np.count_nonzero(pred_counts)}/11 (Missing: INF, MAL_OTH)")

print("\n" + "=" * 80)
print("6. FAIRNESS & SKIN TONE AUDIT (3 BUCKETS & 5 TONES)")
print("=" * 80)
# Group into 3 buckets:
# Dark: tones 0, 1 (tone 0 has 0 samples, tone 1 has 12)
# Medium: tones 2, 3 (tone 2 has 72, tone 3 has 486)
# Light: tones 4, 5 (tone 4 has 153, tone 5 has 63)

def bucket_name(t):
    if t in [0, 1]:
        return "Dark (0-1)"
    elif t in [2, 3]:
        return "Medium (2-3)"
    elif t in [4, 5]:
        return "Light (4-5)"
    return "Unknown"

df["skin_bucket"] = df["skin_tone_class"].apply(bucket_name)

np.random.seed(42)
def compute_subgroup_metrics(sub_df, n_boot=2000):
    n = len(sub_df)
    if n == 0:
        return None
    yt = sub_df["true_label"].values
    yp = sub_df["pred_label"].values
    sub_acc = accuracy_score(yt, yp)
    sub_bal = balanced_accuracy_score(yt, yp)
    sub_f1 = f1_score(yt, yp, average="macro", zero_division=0)
    
    sub_true_mal = sub_df["true_class"].isin(malignant_classes).values
    sub_pred_mal = sub_df["pred_class"].isin(malignant_classes).values
    sub_tp = np.sum(sub_true_mal & sub_pred_mal)
    sub_fn = np.sum(sub_true_mal & (~sub_pred_mal))
    sub_tn = np.sum((~sub_true_mal) & (~sub_pred_mal))
    sub_fp = np.sum((~sub_true_mal) & sub_pred_mal)
    
    sub_sens = sub_tp / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else np.nan
    sub_spec = sub_tn / (sub_tn + sub_fp) if (sub_tn + sub_fp) > 0 else np.nan
    sub_fnr = sub_fn / (sub_tp + sub_fn) if (sub_tp + sub_fn) > 0 else np.nan

    # Bootstrap 95% CIs
    boot_f1s = []
    boot_fnrs = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        b_yt = yt[idx]
        b_yp = yp[idx]
        boot_f1s.append(f1_score(b_yt, b_yp, average="macro", zero_division=0))
        
        b_tmal = sub_true_mal[idx]
        b_pmal = sub_pred_mal[idx]
        b_tp = np.sum(b_tmal & b_pmal)
        b_fn = np.sum(b_tmal & (~b_pmal))
        if (b_tp + b_fn) > 0:
            boot_fnrs.append(b_fn / (b_tp + b_fn))
        else:
            boot_fnrs.append(np.nan)
            
    f1_ci = (np.percentile(boot_f1s, 2.5), np.percentile(boot_f1s, 97.5))
    valid_fnrs = [x for x in boot_fnrs if not np.isnan(x)]
    fnr_ci = (np.percentile(valid_fnrs, 2.5), np.percentile(valid_fnrs, 97.5)) if valid_fnrs else (np.nan, np.nan)
    
    return {
        "n": n,
        "acc": sub_acc,
        "bal_acc": sub_bal,
        "macro_f1": sub_f1,
        "macro_f1_ci": f1_ci,
        "mal_sensitivity": sub_sens,
        "mal_specificity": sub_spec,
        "mal_fnr": sub_fnr,
        "mal_fnr_ci": fnr_ci,
        "low_conf": n < 30
    }

print("\n--- 3-BUCKET FAIRNESS AUDIT ---")
print(f"{'Bucket':<14} {'N':<6} {'LowConf':<8} {'Accuracy':<10} {'Bal Acc':<10} {'Macro-F1 (95% CI)':<26} {'Mal FNR (95% CI)':<24} {'Mal Sens':<10}")
print("-" * 110)
bucket_stats = {}
for b in ["Dark (0-1)", "Medium (2-3)", "Light (4-5)"]:
    sub = df[df["skin_bucket"] == b]
    res = compute_subgroup_metrics(sub)
    bucket_stats[b] = res
    f1_ci_str = f"{res['macro_f1']:.4f} [{res['macro_f1_ci'][0]:.3f}, {res['macro_f1_ci'][1]:.3f}]"
    fnr_ci_str = f"{res['mal_fnr']:.4f} [{res['mal_fnr_ci'][0]:.3f}, {res['mal_fnr_ci'][1]:.3f}]"
    print(f"{b:<14} {res['n']:<6} {str(res['low_conf']):<8} {res['acc']:<10.4f} {res['bal_acc']:<10.4f} {f1_ci_str:<26} {fnr_ci_str:<24} {res['mal_sensitivity']:<10.4f}")

# Compute FNR and Macro-F1 gap across buckets
max_fnr_b = max(bucket_stats[b]["mal_fnr"] for b in bucket_stats)
min_fnr_b = min(bucket_stats[b]["mal_fnr"] for b in bucket_stats)
fnr_gap_3b = max_fnr_b - min_fnr_b
print(f"\n3-Bucket Malignant FNR Gap (Max - Min): {fnr_gap_3b:.4f} ({fnr_gap_3b*100:.2f}%)")

print("\n--- 5 INDIVIDUAL SKIN TONES AUDIT (1 to 5) ---")
print(f"{'Tone':<6} {'N':<6} {'LowConf':<8} {'Accuracy':<10} {'Bal Acc':<10} {'Macro-F1 (95% CI)':<26} {'Mal FNR (95% CI)':<24} {'Mal Sens':<10}")
print("-" * 110)
tone_stats = {}
for t in range(1, 6):
    sub = df[df["skin_tone_class"] == t]
    res = compute_subgroup_metrics(sub)
    tone_stats[t] = res
    f1_ci_str = f"{res['macro_f1']:.4f} [{res['macro_f1_ci'][0]:.3f}, {res['macro_f1_ci'][1]:.3f}]"
    fnr_ci_str = f"{res['mal_fnr']:.4f} [{res['mal_fnr_ci'][0]:.3f}, {res['mal_fnr_ci'][1]:.3f}]"
    print(f"{t:<6} {res['n']:<6} {str(res['low_conf']):<8} {res['acc']:<10.4f} {res['bal_acc']:<10.4f} {f1_ci_str:<26} {fnr_ci_str:<24} {res['mal_sensitivity']:<10.4f}")

max_fnr_t = max(tone_stats[t]["mal_fnr"] for t in tone_stats)
min_fnr_t = min(tone_stats[t]["mal_fnr"] for t in tone_stats)
fnr_gap_5t = max_fnr_t - min_fnr_t
print(f"\n5-Tone Malignant FNR Gap (Max - Min): {fnr_gap_5t:.4f} ({fnr_gap_5t*100:.2f}%)")

print("\n" + "=" * 80)
print("7. DISSECTING TONE 3 (LARGEST BUCKET, N=486)")
print("=" * 80)
df_tone3 = df[df["skin_tone_class"] == 3]
df_not_tone3 = df[df["skin_tone_class"] != 3]

print(f"Tone 3 N={len(df_tone3)} (61.8% of test set)")
print(f"Non-Tone 3 N={len(df_not_tone3)} (38.2% of test set)")

print("\nPer-class Support & Recall in Tone 3 vs Non-Tone 3:")
print(f"{'Class':<10} {'Tone3 Supp':<12} {'Tone3 Rec':<12} {'Non-T3 Supp':<12} {'Non-T3 Rec':<12} {'Rec Diff (T3 - NonT3)':<20}")
print("-" * 80)
for idx, cls in enumerate(classes):
    t3_mask = df_tone3["true_label"] == idx
    t3_supp = np.sum(t3_mask)
    t3_rec = recall_score(df_tone3["true_label"] == idx, df_tone3["pred_label"] == idx, pos_label=True, zero_division=0) if t3_supp > 0 else np.nan
    
    nt3_mask = df_not_tone3["true_label"] == idx
    nt3_supp = np.sum(nt3_mask)
    nt3_rec = recall_score(df_not_tone3["true_label"] == idx, df_not_tone3["pred_label"] == idx, pos_label=True, zero_division=0) if nt3_supp > 0 else np.nan
    
    diff_str = f"{t3_rec - nt3_rec:+.4f}" if (not np.isnan(t3_rec) and not np.isnan(nt3_rec)) else "N/A"
    print(f"{cls:<10} {t3_supp:<12} {t3_rec:<12.4f} {nt3_supp:<12} {nt3_rec:<12.4f} {diff_str:<20}")

print("\nClass distribution comparison:")
print(f"{'Class':<10} {'Tone 3 %':<12} {'Non-Tone 3 %':<12}")
for idx, cls in enumerate(classes):
    t3_pct = np.mean(df_tone3["true_label"] == idx) * 100
    nt3_pct = np.mean(df_not_tone3["true_label"] == idx) * 100
    print(f"{cls:<10} {t3_pct:<12.2f} {nt3_pct:<12.2f}")

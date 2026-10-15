"""
Phase 3.1: Exp-3a Post-Hoc Logit Adjustment (Menon et al., NeurIPS 2020).

Protocol:
1. Extract training class priors pi_k from train split (dermoscopic modality, N=3,668).
2. Tune adjustment strength tau on VALIDATION split (N=786) sweeping tau in [0.0, 2.0].
   Select tau* maximizing val_macro_f1.
3. Apply frozen tau* to TEST split (N=786) logits:
   z_tilde_k = z_k - tau* * log(pi_k)
   pred = argmax(z_tilde)
4. Compute full classification, per-class, binary malignant vs benign,
   and subgroup fairness audits (3-bucket & 5-tone) with 95% bootstrap CIs.
5. Compute paired bootstrap 95% CIs on differences (Delta Macro-F1, Delta Malignant FNR,
   Delta Balanced Acc, Delta Accuracy) against baseline Exp-2 (v2).
6. Save structured JSON and CSV deliverables in results/exp3a_panderm_posthoc/.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import save_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("exp3a_post_hoc")

MALIGNANT_CLASSES = ["BCC", "MEL", "SCCKA", "MAL_OTH", "AKIEC"]
BENIGN_CLASSES = ["NV", "BKL", "DF", "INF", "BEN_OTH", "VASC"]


def compute_training_priors(splits_file: Path, label_mapping: dict) -> tuple[np.ndarray, dict]:
    """Calculate empirical class priors pi_k from dermoscopic training split."""
    df_splits = pd.read_csv(splits_file)
    train_df = df_splits[(df_splits["split"] == "train") & (df_splits["image_type"] == "dermoscopic")]
    
    classes = label_mapping["classes"]
    counts = train_df["diagnosis"].value_counts().to_dict()
    total_samples = len(train_df)

    priors = np.zeros(len(classes), dtype=np.float64)
    counts_dict = {}
    for i, cls in enumerate(classes):
        cnt = counts.get(cls, 0)
        priors[i] = cnt / total_samples
        counts_dict[cls] = cnt

    return priors, counts_dict


def apply_logit_adjustment(logits: np.ndarray, priors: np.ndarray, tau: float) -> np.ndarray:
    """Apply post-hoc logit adjustment: z_tilde_k = z_k - tau * log(pi_k)."""
    # Avoid log(0)
    eps = 1e-12
    safe_priors = np.clip(priors, eps, 1.0)
    adjusted_logits = logits - tau * np.log(safe_priors)
    return adjusted_logits


def sweep_tau_validation(
    df_val: pd.DataFrame,
    class_names: list[str],
    priors: np.ndarray,
    tau_values: np.ndarray,
) -> tuple[float, list[dict]]:
    """Sweep tau on validation split and select tau* maximizing macro-F1."""
    logit_cols = [f"logit_{c}" for c in class_names]
    val_logits = df_val[logit_cols].values
    y_val_true = df_val["true_label"].values

    mal_indices = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    y_val_true_mal = np.isin(y_val_true, mal_indices)

    sweep_results = []
    best_tau = 0.0
    best_f1 = -1.0
    best_bal_acc = -1.0

    for tau in tau_values:
        adj_logits = apply_logit_adjustment(val_logits, priors, tau)
        y_val_pred = np.argmax(adj_logits, axis=1)

        acc = float(accuracy_score(y_val_true, y_val_pred))
        bal_acc = float(balanced_accuracy_score(y_val_true, y_val_pred))
        macro_f1 = float(f1_score(y_val_true, y_val_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_val_true, y_val_pred, average="weighted", zero_division=0))

        y_val_pred_mal = np.isin(y_val_pred, mal_indices)
        tp = np.sum(y_val_true_mal & y_val_pred_mal)
        fn = np.sum(y_val_true_mal & (~y_val_pred_mal))
        mal_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        mal_fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

        res = {
            "tau": round(float(tau), 3),
            "val_accuracy": round(acc, 4),
            "val_balanced_acc": round(bal_acc, 4),
            "val_macro_f1": round(macro_f1, 4),
            "val_weighted_f1": round(weighted_f1, 4),
            "val_malignant_recall": round(mal_recall, 4),
            "val_malignant_fnr": round(mal_fnr, 4),
        }
        sweep_results.append(res)

        # Selection criterion: primary val_macro_f1, secondary val_balanced_acc
        if (macro_f1 > best_f1) or (np.isclose(macro_f1, best_f1, atol=1e-4) and bal_acc > best_bal_acc):
            best_f1 = macro_f1
            best_bal_acc = bal_acc
            best_tau = float(tau)

    return best_tau, sweep_results


def compute_fairness_buckets(df: pd.DataFrame, class_names: list[str], n_boot: int = 2000) -> dict:
    """Compute 3-bucket and 5-tone fairness metrics with bootstrap 95% CIs."""
    def bucket_fn(t):
        if t in [0, 1]:
            return "Dark (0-1)"
        elif t in [2, 3]:
            return "Medium (2-3)"
        elif t in [4, 5]:
            return "Light (4-5)"
        return "Unknown"

    df_copy = df.copy()
    df_copy["skin_bucket"] = df_copy["skin_tone_class"].apply(bucket_fn)

    mal_indices = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    np.random.seed(42)

    def eval_subgroup(sub_df):
        n = len(sub_df)
        if n == 0:
            return None
        yt = sub_df["true_label"].values
        yp = sub_df["pred_label"].values
        acc = accuracy_score(yt, yp)
        try:
            bal_acc = balanced_accuracy_score(yt, yp)
        except Exception:
            bal_acc = acc
        f1 = f1_score(yt, yp, average="macro", zero_division=0)

        yt_mal = np.isin(yt, mal_indices)
        yp_mal = np.isin(yp, mal_indices)
        tp = np.sum(yt_mal & yp_mal)
        fn = np.sum(yt_mal & (~yp_mal))
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fnr = fn / (tp + fn) if (tp + fn) > 0 else np.nan

        # Bootstrap CIs
        boot_f1s = []
        boot_fnrs = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            b_yt = yt[idx]
            b_yp = yp[idx]
            boot_f1s.append(f1_score(b_yt, b_yp, average="macro", zero_division=0))

            b_yt_mal = yt_mal[idx]
            b_yp_mal = yp_mal[idx]
            b_tp = np.sum(b_yt_mal & b_yp_mal)
            b_fn = np.sum(b_yt_mal & (~b_yp_mal))
            if (b_tp + b_fn) > 0:
                boot_fnrs.append(b_fn / (b_tp + b_fn))

        f1_ci = (float(np.percentile(boot_f1s, 2.5)), float(np.percentile(boot_f1s, 97.5)))
        fnr_ci = (float(np.percentile(boot_fnrs, 2.5)), float(np.percentile(boot_fnrs, 97.5))) if boot_fnrs else (np.nan, np.nan)

        return {
            "n": int(n),
            "low_confidence": bool(n < 30),
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(f1),
            "macro_f1_ci_95": f1_ci,
            "malignant_sensitivity": float(sens),
            "malignant_fnr": float(fnr),
            "malignant_fnr_ci_95": fnr_ci,
        }

    bucket_results = {}
    for b in ["Dark (0-1)", "Medium (2-3)", "Light (4-5)"]:
        sub = df_copy[df_copy["skin_bucket"] == b]
        bucket_results[b] = eval_subgroup(sub)

    tone_results = {}
    for t in range(1, 6):
        sub = df_copy[df_copy["skin_tone_class"] == t]
        tone_results[f"Tone_{t}"] = eval_subgroup(sub)

    # Disparities across 3 buckets
    fnrs_b = [v["malignant_fnr"] for v in bucket_results.values() if v and not np.isnan(v["malignant_fnr"])]
    f1s_b = [v["macro_f1"] for v in bucket_results.values() if v]
    fnr_gap_3b = max(fnrs_b) - min(fnrs_b) if len(fnrs_b) > 1 else 0.0
    f1_gap_3b = max(f1s_b) - min(f1s_b) if len(f1s_b) > 1 else 0.0

    return {
        "buckets_3": bucket_results,
        "tones_5": tone_results,
        "disparities": {
            "fnr_gap_3_buckets": float(fnr_gap_3b),
            "macro_f1_gap_3_buckets": float(f1_gap_3b),
        },
    }


def compute_paired_bootstrap_comparison(
    df_exp2: pd.DataFrame,
    df_exp3a: pd.DataFrame,
    class_names: list[str],
    n_boot: int = 2000,
) -> dict:
    """Compute paired bootstrap difference distributions and 95% CIs against Exp-2."""
    n = len(df_exp2)
    assert len(df_exp3a) == n, "DataFrames must have identical lengths"

    y_true = df_exp2["true_label"].values
    y_pred_exp2 = df_exp2["pred_label"].values
    y_pred_exp3a = df_exp3a["pred_label"].values

    mal_indices = [i for i, c in enumerate(class_names) if c in MALIGNANT_CLASSES]
    y_true_mal = np.isin(y_true, mal_indices)
    y_pred_mal_exp2 = np.isin(y_pred_exp2, mal_indices)
    y_pred_mal_exp3a = np.isin(y_pred_exp3a, mal_indices)

    # Point estimates
    f1_exp2 = f1_score(y_true, y_pred_exp2, average="macro", zero_division=0)
    f1_exp3a = f1_score(y_true, y_pred_exp3a, average="macro", zero_division=0)
    delta_f1 = f1_exp3a - f1_exp2

    bal_exp2 = balanced_accuracy_score(y_true, y_pred_exp2)
    bal_exp3a = balanced_accuracy_score(y_true, y_pred_exp3a)
    delta_bal = bal_exp3a - bal_exp2

    acc_exp2 = accuracy_score(y_true, y_pred_exp2)
    acc_exp3a = accuracy_score(y_true, y_pred_exp3a)
    delta_acc = acc_exp3a - acc_exp2

    fn_exp2 = np.sum(y_true_mal & (~y_pred_mal_exp2))
    fn_exp3a = np.sum(y_true_mal & (~y_pred_mal_exp3a))
    p_mal = np.sum(y_true_mal)
    fnr_exp2 = fn_exp2 / p_mal
    fnr_exp3a = fn_exp3a / p_mal
    delta_fnr = fnr_exp3a - fnr_exp2
    delta_sens = (1 - fnr_exp3a) - (1 - fnr_exp2)

    # Bootstrap
    np.random.seed(42)
    boot_delta_f1 = []
    boot_delta_bal = []
    boot_delta_acc = []
    boot_delta_fnr = []

    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        b_yt = y_true[idx]
        b_yp2 = y_pred_exp2[idx]
        b_yp3 = y_pred_exp3a[idx]

        b_f1_2 = f1_score(b_yt, b_yp2, average="macro", zero_division=0)
        b_f1_3 = f1_score(b_yt, b_yp3, average="macro", zero_division=0)
        boot_delta_f1.append(b_f1_3 - b_f1_2)

        try:
            b_bal_2 = balanced_accuracy_score(b_yt, b_yp2)
            b_bal_3 = balanced_accuracy_score(b_yt, b_yp3)
            boot_delta_bal.append(b_bal_3 - b_bal_2)
        except Exception:
            pass

        boot_delta_acc.append(accuracy_score(b_yt, b_yp3) - accuracy_score(b_yt, b_yp2))

        b_tmal = y_true_mal[idx]
        b_pmal_2 = y_pred_mal_exp2[idx]
        b_pmal_3 = y_pred_mal_exp3a[idx]
        b_pm = np.sum(b_tmal)
        if b_pm > 0:
            b_fnr_2 = np.sum(b_tmal & (~b_pmal_2)) / b_pm
            b_fnr_3 = np.sum(b_tmal & (~b_pmal_3)) / b_pm
            boot_delta_fnr.append(b_fnr_3 - b_fnr_2)

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "n_samples": int(n),
        "n_boot": int(n_boot),
        "macro_f1": {
            "exp2": float(f1_exp2),
            "exp3a": float(f1_exp3a),
            "delta": float(delta_f1),
            "delta_ci_95": ci(boot_delta_f1),
            "significant_at_95": bool(ci(boot_delta_f1)[0] > 0 or ci(boot_delta_f1)[1] < 0),
        },
        "balanced_accuracy": {
            "exp2": float(bal_exp2),
            "exp3a": float(bal_exp3a),
            "delta": float(delta_bal),
            "delta_ci_95": ci(boot_delta_bal),
            "significant_at_95": bool(ci(boot_delta_bal)[0] > 0 or ci(boot_delta_bal)[1] < 0),
        },
        "accuracy": {
            "exp2": float(acc_exp2),
            "exp3a": float(acc_exp3a),
            "delta": float(delta_acc),
            "delta_ci_95": ci(boot_delta_acc),
            "significant_at_95": bool(ci(boot_delta_acc)[0] > 0 or ci(boot_delta_acc)[1] < 0),
        },
        "malignant_fnr": {
            "exp2": float(fnr_exp2),
            "exp3a": float(fnr_exp3a),
            "delta": float(delta_fnr),
            "delta_ci_95": ci(boot_delta_fnr),
            "significant_at_95": bool(ci(boot_delta_fnr)[0] > 0 or ci(boot_delta_fnr)[1] < 0),
        },
        "malignant_sensitivity": {
            "exp2": float(1 - fnr_exp2),
            "exp3a": float(1 - fnr_exp3a),
            "delta": float(delta_sens),
            "delta_ci_95": [-ci(boot_delta_fnr)[1], -ci(boot_delta_fnr)[0]],
            "significant_at_95": bool(ci(boot_delta_fnr)[0] > 0 or ci(boot_delta_fnr)[1] < 0),
        },
    }


def main():
    t_start = time.time()
    logger.info("=" * 80)
    logger.info("PHASE 3: EXP-3A POST-HOC LOGIT ADJUSTMENT (Menon et al., NeurIPS 2020)")
    logger.info("=" * 80)

    # 1. Load label mapping & priors
    label_map_file = PROJECT_ROOT / "configs" / "label_mapping.json"
    with open(label_map_file, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    classes = label_map["classes"]
    idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}

    splits_file = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"
    priors, train_counts = compute_training_priors(splits_file, label_map)

    logger.info("Training Class Frequencies & Priors (Dermoscopic Modality, N=3,668):")
    for idx, cls in enumerate(classes):
        logger.info(f"  [{idx:2d}] {cls:<10}: N = {train_counts[cls]:4d} | pi = {priors[idx]:.5f} | log(pi) = {np.log(priors[idx]):.3f}")

    # 2. Validation Split Sweep
    val_pred_file = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_val.csv"
    if not val_pred_file.is_file():
        raise FileNotFoundError(f"Validation predictions not found: {val_pred_file}")
    df_val = pd.read_csv(val_pred_file)
    logger.info(f"Loaded validation predictions: {len(df_val)} lesions.")

    tau_sweep = np.linspace(0.0, 2.0, 41)  # step = 0.05
    best_tau, sweep_data = sweep_tau_validation(df_val, classes, priors, tau_sweep)
    logger.info(f"\n--- VALIDATION TAU SWEEP RESULTS (Optimal tau* = {best_tau:.3f}) ---")
    logger.info(f"{'tau':<6} {'Val Acc':<10} {'Val Bal Acc':<12} {'Val Macro-F1':<14} {'Val W-F1':<10} {'Val Mal Sens':<12} {'Val Mal FNR':<12}")
    logger.info("-" * 80)
    for row in sweep_data:
        mark = " <-- OPTIMAL (tau*)" if np.isclose(row["tau"], best_tau) else ""
        logger.info(f"{row['tau']:<6.2f} {row['val_accuracy']:<10.4f} {row['val_balanced_acc']:<12.4f} {row['val_macro_f1']:<14.4f} {row['val_weighted_f1']:<10.4f} {row['val_malignant_recall']:<12.4f} {row['val_malignant_fnr']:<12.4f}{mark}")

    # 3. Apply best_tau to Validation Split and save
    logit_cols = [f"logit_{c}" for c in classes]
    val_logits = df_val[logit_cols].values
    val_adj_logits = apply_logit_adjustment(val_logits, priors, best_tau)
    val_adj_probs = softmax(val_adj_logits, axis=1)
    val_adj_preds = np.argmax(val_adj_logits, axis=1)

    df_val_exp3a = df_val.copy()
    df_val_exp3a["orig_pred_label"] = df_val["pred_label"]
    df_val_exp3a["orig_pred_class"] = df_val["pred_class"]
    df_val_exp3a["pred_label"] = val_adj_preds
    df_val_exp3a["pred_class"] = [classes[i] for i in val_adj_preds]
    for i, cls in enumerate(classes):
        df_val_exp3a[f"prob_{cls}"] = val_adj_probs[:, i]
        df_val_exp3a[f"logit_adj_{cls}"] = val_adj_logits[:, i]

    # 4. Apply best_tau to Test Split
    test_pred_file = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_test.csv"
    if not test_pred_file.is_file():
        raise FileNotFoundError(f"Test predictions not found: {test_pred_file}")
    df_test = pd.read_csv(test_pred_file)
    logger.info(f"\nLoaded test predictions: {len(df_test)} lesions.")

    test_logits = df_test[logit_cols].values
    test_adj_logits = apply_logit_adjustment(test_logits, priors, best_tau)
    test_adj_probs = softmax(test_adj_logits, axis=1)
    test_adj_preds = np.argmax(test_adj_logits, axis=1)

    df_test_exp3a = df_test.copy()
    df_test_exp3a["orig_pred_label"] = df_test["pred_label"]
    df_test_exp3a["orig_pred_class"] = df_test["pred_class"]
    df_test_exp3a["pred_label"] = test_adj_preds
    df_test_exp3a["pred_class"] = [classes[i] for i in test_adj_preds]
    for i, cls in enumerate(classes):
        df_test_exp3a[f"prob_{cls}"] = test_adj_probs[:, i]
        df_test_exp3a[f"logit_adj_{cls}"] = test_adj_logits[:, i]

    # 5. Compute full metrics on Test Split
    test_prob_cols = [f"prob_{c}" for c in classes]
    metrics_exp3a = calculate_classification_metrics(
        y_true=df_test_exp3a["true_label"].values,
        y_pred=df_test_exp3a["pred_label"].values,
        y_prob=df_test_exp3a[test_prob_cols].values,
        class_names=classes,
    )
    subgroup_metrics_exp3a = calculate_subgroup_metrics(
        df_predictions=df_test_exp3a,
        subgroup_col="skin_tone_class",
    )
    fairness_buckets_exp3a = compute_fairness_buckets(df_test_exp3a, classes)

    # 6. Paired Bootstrap Comparison
    logger.info("\nComputing Paired Bootstrap 95% Confidence Intervals against Exp-2 (B = 2,000)...")
    comparison = compute_paired_bootstrap_comparison(df_test, df_test_exp3a, classes, n_boot=2000)

    # 7. Print Comprehensive Summary
    logger.info("=" * 80)
    logger.info("EXP-3A TEST PERFORMANCE EVALUATION (tau* = %.3f)" % best_tau)
    logger.info("=" * 80)
    logger.info(f"Top-1 Accuracy:     {metrics_exp3a['accuracy']:.4f} (Exp-2: {comparison['accuracy']['exp2']:.4f}, Delta: {comparison['accuracy']['delta']:+.4f} [{comparison['accuracy']['delta_ci_95'][0]:+.3f}, {comparison['accuracy']['delta_ci_95'][1]:+.3f}])")
    logger.info(f"Balanced Accuracy:  {metrics_exp3a['balanced_accuracy']:.4f} (Exp-2: {comparison['balanced_accuracy']['exp2']:.4f}, Delta: {comparison['balanced_accuracy']['delta']:+.4f} [{comparison['balanced_accuracy']['delta_ci_95'][0]:+.3f}, {comparison['balanced_accuracy']['delta_ci_95'][1]:+.3f}])")
    logger.info(f"Macro-F1:           {metrics_exp3a['macro_f1']:.4f} (Exp-2: {comparison['macro_f1']['exp2']:.4f}, Delta: {comparison['macro_f1']['delta']:+.4f} [{comparison['macro_f1']['delta_ci_95'][0]:+.3f}, {comparison['macro_f1']['delta_ci_95'][1]:+.3f}])")
    logger.info(f"Weighted F1:        {metrics_exp3a['weighted_f1']:.4f}")
    logger.info(f"Malignant Recall:   {metrics_exp3a['malignant_recall']:.4f} ({metrics_exp3a['malignant_recall']*100:.2f}%)")
    logger.info(f"Malignant FNR:      {comparison['malignant_fnr']['exp3a']:.4f} (Exp-2: {comparison['malignant_fnr']['exp2']:.4f}, Delta: {comparison['malignant_fnr']['delta']:+.4f} [{comparison['malignant_fnr']['delta_ci_95'][0]:+.3f}, {comparison['malignant_fnr']['delta_ci_95'][1]:+.3f}])")
    logger.info(f"Malignant AUROC:    {metrics_exp3a['malignant_auroc']:.4f}")
    logger.info(f"Macro AUROC:        {metrics_exp3a['macro_auroc']:.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("PER-CLASS PERFORMANCE COMPARISON (Exp-2 vs Exp-3a)")
    logger.info("=" * 80)
    cm_orig = confusion_matrix(df_test["true_label"], df_test["pred_label"], labels=list(range(len(classes))))
    rec_orig = recall_score(df_test["true_label"], df_test["pred_label"], average=None, labels=list(range(len(classes))), zero_division=0)
    f1_orig = f1_score(df_test["true_label"], df_test["pred_label"], average=None, labels=list(range(len(classes))), zero_division=0)

    logger.info(f"{'Class':<10} {'Support':<8} {'Exp-2 Rec':<12} {'Exp-3a Rec':<12} {'Delta Rec':<12} {'Exp-2 F1':<12} {'Exp-3a F1':<12} {'Delta F1':<12}")
    logger.info("-" * 90)
    for idx, cls in enumerate(classes):
        rec_3a = metrics_exp3a["per_class"][cls]["recall"]
        f1_3a = metrics_exp3a["per_class"][cls]["f1"]
        supp = metrics_exp3a["per_class"][cls]["support"]
        d_rec = rec_3a - rec_orig[idx]
        d_f1 = f1_3a - f1_orig[idx]
        logger.info(f"{cls:<10} {supp:<8d} {rec_orig[idx]:<12.4f} {rec_3a:<12.4f} {d_rec:<+12.4f} {f1_orig[idx]:<12.4f} {f1_3a:<12.4f} {d_f1:<+12.4f}")

    logger.info("\n" + "=" * 80)
    logger.info("3-BUCKET FAIRNESS AUDIT (Dark, Medium, Light)")
    logger.info("=" * 80)
    logger.info(f"{'Bucket':<14} {'N':<6} {'LowConf':<8} {'Accuracy':<10} {'Bal Acc':<10} {'Macro-F1 (95% CI)':<26} {'Mal FNR (95% CI)':<24} {'Mal Sens':<10}")
    logger.info("-" * 110)
    for b_name, b_res in fairness_buckets_exp3a["buckets_3"].items():
        if b_res:
            f1_str = f"{b_res['macro_f1']:.4f} [{b_res['macro_f1_ci_95'][0]:.3f}, {b_res['macro_f1_ci_95'][1]:.3f}]"
            fnr_str = f"{b_res['malignant_fnr']:.4f} [{b_res['malignant_fnr_ci_95'][0]:.3f}, {b_res['malignant_fnr_ci_95'][1]:.3f}]"
            logger.info(f"{b_name:<14} {b_res['n']:<6d} {str(b_res['low_confidence']):<8} {b_res['accuracy']:<10.4f} {b_res['balanced_accuracy']:<10.4f} {f1_str:<26} {fnr_str:<24} {b_res['malignant_sensitivity']:<10.4f}")

    logger.info(f"\n3-Bucket Malignant FNR Disparity: {fairness_buckets_exp3a['disparities']['fnr_gap_3_buckets']:.4f}")
    logger.info(f"3-Bucket Macro-F1 Disparity:        {fairness_buckets_exp3a['disparities']['macro_f1_gap_3_buckets']:.4f}")

    # 8. Tone 3 Dissection
    logger.info("\n" + "=" * 80)
    logger.info("TONE 3 DISSECTION UNDER EXP-3A (N=486 vs Non-Tone 3 N=300)")
    logger.info("=" * 80)
    df_t3 = df_test_exp3a[df_test_exp3a["skin_tone_class"] == 3]
    df_nt3 = df_test_exp3a[df_test_exp3a["skin_tone_class"] != 3]
    logger.info(f"{'Class':<10} {'T3 Supp':<10} {'T3 Rec':<12} {'Non-T3 Supp':<12} {'Non-T3 Rec':<12} {'Rec Diff (T3 - NonT3)':<22}")
    logger.info("-" * 80)
    for idx, cls in enumerate(classes):
        t3_m = df_t3["true_label"] == idx
        nt3_m = df_nt3["true_label"] == idx
        t3_s = np.sum(t3_m)
        nt3_s = np.sum(nt3_m)
        t3_r = recall_score(df_t3["true_label"] == idx, df_t3["pred_label"] == idx, pos_label=True, zero_division=0) if t3_s > 0 else np.nan
        nt3_r = recall_score(df_nt3["true_label"] == idx, df_nt3["pred_label"] == idx, pos_label=True, zero_division=0) if nt3_s > 0 else np.nan
        diff_s = f"{t3_r - nt3_r:+.4f}" if (not np.isnan(t3_r) and not np.isnan(nt3_r)) else "N/A"
        logger.info(f"{cls:<10} {t3_s:<10d} {t3_r:<12.4f} {nt3_s:<12d} {nt3_r:<12.4f} {diff_s:<22}")

    # 9. Save Deliverables
    out_dir = PROJECT_ROOT / "results" / "exp3a_panderm_posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(df_val_exp3a, out_dir / "predictions_val.csv")
    save_predictions(df_test_exp3a, out_dir / "predictions_test.csv")
    save_predictions(df_test_exp3a, out_dir / "predictions.csv")

    with open(out_dir / "tau_sweep_validation.json", "w", encoding="utf-8") as f:
        json.dump({"optimal_tau": best_tau, "sweep": sweep_data}, f, indent=2)

    with open(out_dir / "paired_bootstrap_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    metrics_deliverable = {
        "val_overall": {
            "optimal_tau": best_tau,
            "accuracy": float(accuracy_score(df_val_exp3a["true_label"], df_val_exp3a["pred_label"])),
            "balanced_accuracy": float(balanced_accuracy_score(df_val_exp3a["true_label"], df_val_exp3a["pred_label"])),
            "macro_f1": float(f1_score(df_val_exp3a["true_label"], df_val_exp3a["pred_label"], average="macro", zero_division=0)),
        },
        "test_overall": metrics_exp3a,
        "subgroup_fairness": subgroup_metrics_exp3a,
        "fairness_buckets": fairness_buckets_exp3a,
        "paired_bootstrap_vs_exp2": comparison,
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_deliverable, f, indent=2)

    summary_deliverable = {
        "experiment_name": "exp3a-panderm-posthoc",
        "description": "PanDerm ViT-Base Post-Hoc Logit Adjustment (Menon et al., NeurIPS 2020)",
        "method": "Post-Hoc Logit Adjustment",
        "optimal_tau": best_tau,
        "selection_split": "validation",
        "evaluation_split": "test",
        "test_overall": metrics_exp3a,
        "paired_bootstrap_vs_exp2": comparison,
        "fairness_buckets": fairness_buckets_exp3a,
        "disparities": fairness_buckets_exp3a["disparities"],
    }
    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_deliverable, f, indent=2)

    logger.info(f"\nAll Exp-3a deliverables successfully persisted to {out_dir}")
    logger.info(f"Execution finished in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    main()

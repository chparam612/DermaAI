"""Exp-3b Diagnosis & Hypothesis Test Script.

Analyzes why Exp-3b BCC recall collapsed to 0.0000 under train-time logit adjustment (tau=1.0).
Computes:
1. Training split priors and asserts alignment with configs/label_mapping.json.
2. 11x11 ideal raw-logit gap matrix D(c, k) = ln((eps/K)/(1-eps+eps/K)) - tau*(ln pi_k - ln pi_c).
3. Predicted-class counts and true-BCC confusion rows for Exp-3b and Exp-2c (seeds 42 & 43, val & test).
4. Diagnostic t-sweep: argmax(z + t * ln pi) for t in {0.0, 0.25, 0.5, 0.75, 1.0} on Exp-3b saved raw logits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MALIGNANT_CLASSES = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]


def load_label_mapping() -> List[str]:
    mapping_path = PROJECT_ROOT / "configs" / "label_mapping.json"
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return list(data["classes"])


def compute_training_priors(classes: List[str]) -> Tuple[np.ndarray, Dict[str, int], int]:
    splits_path = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"
    df = pd.read_csv(splits_path)
    train_df = df[(df["split"] == "train") & (df["image_type"] == "dermoscopic")]
    total = len(train_df)
    counts = train_df["diagnosis"].value_counts().to_dict()

    priors = np.zeros(len(classes), dtype=np.float64)
    counts_dict = {}
    for idx, cls in enumerate(classes):
        cnt = counts.get(cls, 0)
        priors[idx] = cnt / total
        counts_dict[cls] = cnt

    return priors, counts_dict, total


def compute_d_matrix(priors: np.ndarray, classes: List[str], tau: float = 1.0, eps: float = 0.1) -> np.ndarray:
    """Compute 11x11 ideal raw-logit gap matrix D(c, k).

    D(c, k) = ln((eps/K) / (1 - eps + eps/K)) - tau * (ln pi_k - ln pi_c)
    For true class c and wrong class k.
    """
    K = len(classes)
    p_true = 1.0 - eps + (eps / K)
    p_wrong = eps / K
    target_log_ratio = np.log(p_wrong / p_true)  # ln(0.01) = -4.605170

    log_priors = np.log(priors)
    D = np.zeros((K, K), dtype=np.float64)

    for c in range(K):
        for k in range(K):
            if c == k:
                D[c, k] = 0.0  # diagonal (not a wrong class)
            else:
                D[c, k] = target_log_ratio - tau * (log_priors[k] - log_priors[c])
    return D


def format_d_matrix_markdown(D: np.ndarray, classes: List[str]) -> str:
    lines = [
        "| True Class (c) \\ Comp Class (k) | " + " | ".join(classes) + " |",
        "| :--- | " + " | ".join([":---:"] * len(classes)) + " |",
    ]
    for c_idx, c_name in enumerate(classes):
        row_vals = []
        for k_idx, k_name in enumerate(classes):
            if c_idx == k_idx:
                row_vals.append("—")
            else:
                val = D[c_idx, k_idx]
                if val > 0:
                    row_vals.append(f"**+{val:.4f}**")
                else:
                    row_vals.append(f"{val:.4f}")
        lines.append(f"| **{c_name}** | " + " | ".join(row_vals) + " |")
    return "\n".join(lines)


def get_confusion_and_predictions(classes: List[str]):
    exp3b_runs = {
        "Exp-3b (seed 42, test)": PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed42" / "predictions_test.csv",
        "Exp-3b (seed 42, val)": PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed42" / "predictions_val.csv",
        "Exp-3b (seed 43, test)": PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed43" / "predictions_test.csv",
        "Exp-3b (seed 43, val)": PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed43" / "predictions_val.csv",
        "Exp-2c (seed 42, test)": PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "predictions_test.csv",
        "Exp-2c (seed 43, test)": PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed43" / "exp2c_panderm_uniform_seed43_artifacts" / "predictions_test.csv",
    }

    bcc_rows = {}
    pred_counts = {}

    for name, path in exp3b_runs.items():
        df = pd.read_csv(path)
        # 1. Overall predicted class distribution
        pred_vc = df["pred_class"].value_counts().to_dict()
        pred_counts[name] = [pred_vc.get(c, 0) for c in classes]

        # 2. Confusion row for true BCC
        bcc_df = df[df["true_class"] == "BCC"]
        bcc_vc = bcc_df["pred_class"].value_counts().to_dict()
        bcc_rows[name] = {
            "total": len(bcc_df),
            "counts": [bcc_vc.get(c, 0) for c in classes],
        }

    return bcc_rows, pred_counts


def run_t_sweep(classes: List[str], priors: np.ndarray) -> List[Dict[str, float]]:
    mal_indices = [classes.index(c) for c in MALIGNANT_CLASSES]
    bcc_idx = classes.index("BCC")
    log_priors = np.log(priors)

    runs = [
        ("Exp-3b (seed 42)", "val", PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed42" / "predictions_val.csv"),
        ("Exp-3b (seed 42)", "test", PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed42" / "predictions_test.csv"),
        ("Exp-3b (seed 43)", "val", PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed43" / "predictions_val.csv"),
        ("Exp-3b (seed 43)", "test", PROJECT_ROOT / "results" / "exp3b_panderm_logit_adj_seed43" / "predictions_test.csv"),
    ]

    t_values = [0.0, 0.25, 0.50, 0.75, 1.00]
    records = []

    for run_name, split, path in runs:
        df = pd.read_csv(path)
        logit_cols = [f"logit_{c}" for c in classes]
        logits = df[logit_cols].values
        y_true = df["true_label"].values
        y_true_mal = np.isin(y_true, mal_indices).astype(int)

        for t in t_values:
            adj_logits = logits + t * log_priors
            preds = np.argmax(adj_logits, axis=1)

            macro_f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            bcc_rec = recall_score(y_true == bcc_idx, preds == bcc_idx, zero_division=0)

            preds_mal = np.isin(preds, mal_indices).astype(int)
            mal_sens = recall_score(y_true_mal, preds_mal, zero_division=0)
            mal_fnr = 1.0 - mal_sens
            specificity = recall_score(y_true_mal == 0, preds_mal == 0, zero_division=0)

            records.append({
                "run": run_name,
                "split": split,
                "t": t,
                "macro_f1": float(macro_f1),
                "bcc_recall": float(bcc_rec),
                "malignant_fnr": float(mal_fnr),
                "specificity": float(specificity),
            })

    return records


def main():
    classes = load_label_mapping()
    priors, counts_dict, total_train = compute_training_priors(classes)

    audit_dir = PROJECT_ROOT / "results" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # 1. Training Priors Output
    prior_lines = [
        "| Index | Class Name | Training Count | Class Prior (pi_c) | Log Prior ln(pi_c) |",
        "| :---: | :--- | :---: | :---: | :---: |",
    ]
    for idx, cls in enumerate(classes):
        prior_lines.append(f"| {idx} | **{cls}** | {counts_dict[cls]} | {priors[idx]:.6f} | {np.log(priors[idx]):.4f} |")
    with open(audit_dir / "training_priors.md", "w", encoding="utf-8") as f:
        f.write("\n".join(prior_lines) + "\n")

    # 2. 11x11 D(c, k) Table
    D = compute_d_matrix(priors, classes, tau=1.0, eps=0.1)
    d_md = format_d_matrix_markdown(D, classes)
    with open(audit_dir / "d_matrix_11x11.md", "w", encoding="utf-8") as f:
        f.write(d_md + "\n")

    # 3. True-BCC Confusion Rows & Predicted Counts
    bcc_rows, pred_counts = get_confusion_and_predictions(classes)

    bcc_lines = [
        "| Run & Split | Total True BCC | " + " | ".join(classes) + " | Absorbing Class |",
        "| :--- | :---: | " + " | ".join([":---:"] * len(classes)) + " | :--- |",
    ]
    for name, data in bcc_rows.items():
        c_list = data["counts"]
        total = data["total"]
        # Find max non-BCC prediction
        bcc_idx = classes.index("BCC")
        max_idx = int(np.argmax([c_list[i] if i != bcc_idx else -1 for i in range(len(classes))]))
        absorbing = f"{classes[max_idx]} ({c_list[max_idx]}/{total} = {c_list[max_idx]/total*100:.1f}%)" if c_list[bcc_idx] == 0 else f"BCC retained ({c_list[bcc_idx]}/{total})"
        row_str = f"| **{name}** | {total} | " + " | ".join([str(c) for c in c_list]) + f" | **{absorbing}** |"
        bcc_lines.append(row_str)

    with open(audit_dir / "bcc_confusion_rows.md", "w", encoding="utf-8") as f:
        f.write("\n".join(bcc_lines) + "\n")

    # Overall predicted counts
    pred_lines = [
        "| Run & Split | Total Lesions | " + " | ".join(classes) + " |",
        "| :--- | :---: | " + " | ".join([":---:"] * len(classes)) + " |",
    ]
    for name, counts in pred_counts.items():
        total = sum(counts)
        pred_lines.append(f"| **{name}** | {total} | " + " | ".join([str(c) for c in counts]) + " |")
    with open(audit_dir / "predicted_class_counts.md", "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines) + "\n")

    # 4. Diagnostic t-sweep table
    t_records = run_t_sweep(classes, priors)
    t_df = pd.DataFrame(t_records)

    t_lines = [
        "| Run | Split | Decision Rule Shift (t) | Macro-F1 | BCC Recall | Malignant FNR | Specificity |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in t_records:
        t_lines.append(
            f"| **{r['run']}** | {r['split']} | t = {r['t']:.2f} | {r['macro_f1']:.4f} | {r['bcc_recall']:.4f} | {r['malignant_fnr']:.4f} | {r['specificity']:.4f} |"
        )
    with open(audit_dir / "t_sweep_table.md", "w", encoding="utf-8") as f:
        f.write("\n".join(t_lines) + "\n")

    print("[SUCCESS] Exp-3b diagnosis script executed successfully.")


if __name__ == "__main__":
    main()

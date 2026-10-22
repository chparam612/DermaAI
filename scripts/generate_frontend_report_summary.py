"""Generate results/frontend_report_summary.json for the frontend F4 panel from audited files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def compute_sha256(p: Path) -> str:
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=== Generating results/frontend_report_summary.json ===")
    summary_path = PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "summary_metrics.json"
    preds_path = PROJECT_ROOT / "results" / "exp2c_panderm_uniform_seed42" / "exp2c_panderm_uniform_artifacts" / "predictions_test.csv"
    splits_path = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"

    with open(summary_path, "r", encoding="utf-8") as f:
        summary_raw = json.load(f)

    test_overall = summary_raw["test_overall"]
    per_class_raw = test_overall["per_class"]
    subgroups_raw = summary_raw["subgroup_fairness"]["subgroups"]

    # Compute exact binary malignant metrics from predictions_test.csv
    preds_df = pd.read_csv(preds_path)
    mal_set = set(test_overall["malignant_classes"])
    is_true_mal = preds_df["true_class"].isin(mal_set)
    is_pred_mal = preds_df["pred_class"].isin(mal_set)

    tp = int((is_true_mal & is_pred_mal).sum())
    fn = int((is_true_mal & (~is_pred_mal)).sum())
    fp = int(((~is_true_mal) & is_pred_mal).sum())
    tn = int(((~is_true_mal) & (~is_pred_mal)).sum())

    sens = float(tp / (tp + fn))
    spec = float(tn / (tn + fp))

    split_sha256 = compute_sha256(splits_path)

    # Compile Per-Class Table
    per_class_data = {}
    tail_supports = {}
    for cls_name, pdata in per_class_raw.items():
        supp = int(pdata["support"])
        per_class_data[cls_name] = {
            "recall": round(float(pdata["recall"]), 4),
            "f1_score": round(float(pdata["f1"]), 4),
            "precision": round(float(pdata["precision"]), 4),
            "support": supp,
            "is_tail_class": bool(supp < 20),
        }
        if supp < 20:
            tail_supports[cls_name] = supp

    # Compile Skin Tone Subgroups
    skin_tone_data = {}
    for grp_key, grp_data in subgroups_raw.items():
        n = int(grp_data["sample_count"])
        skin_tone_data[grp_key] = {
            "sample_count": n,
            "accuracy": round(float(grp_data["accuracy"]), 4),
            "balanced_accuracy": round(float(grp_data["balanced_accuracy"]), 4),
            "macro_f1": round(float(grp_data["macro_f1"]), 4),
            "low_confidence": bool(n < 30),
        }

    report = {
        "status": "audited_and_verified",
        "provenance": {
            "source": "results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/summary_metrics.json",
            "model_architecture": "PanDerm VisionTransformer (panderm_base)",
            "experiment_name": summary_raw["experiment_name"],
            "seed": int(summary_raw["seed"]),
            "split_file": "data/splits/lesion_level_splits.csv",
            "split_sha256": split_sha256,
            "modality_evaluated": "dermoscopic (single image per lesion)",
            "total_test_samples": int(test_overall["total_samples"]),
        },
        "overall_test_metrics": {
            "source": "results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/summary_metrics.json:test_overall",
            "macro_f1": round(float(test_overall["macro_f1"]), 4),
            "accuracy": round(float(test_overall["accuracy"]), 4),
            "balanced_accuracy": round(float(test_overall["balanced_accuracy"]), 4),
            "weighted_f1": round(float(test_overall["weighted_f1"]), 4),
            "macro_auroc": round(float(test_overall["macro_auroc"]), 4),
        },
        "malignant_vs_benign_metrics": {
            "source": "results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/predictions_test.csv",
            "malignant_classes": list(test_overall["malignant_classes"]),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "false_negative_count": fn,
            "false_positive_count": fp,
            "true_positive_count": tp,
            "true_negative_count": tn,
            "total_malignant": tp + fn,
            "total_benign": tn + fp,
        },
        "per_class_metrics": {
            "source": "results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/summary_metrics.json:test_overall.per_class",
            "classes": per_class_data,
        },
        "skin_tone_subgroup_metrics": {
            "source": "results/exp2c_panderm_uniform_seed42/exp2c_panderm_uniform_artifacts/summary_metrics.json:subgroup_fairness",
            "subgroups": skin_tone_data,
            "accuracy_disparity": round(float(summary_raw["subgroup_fairness"]["accuracy_disparity"]), 4),
            "balanced_accuracy_disparity": round(float(summary_raw["subgroup_fairness"]["balanced_accuracy_disparity"]), 4),
            "disparity_caveat": "High sample concentration in Tone 3 (n=486, 61.8%); Tone 1 has low sample support (n=12 < 30).",
        },
        "caveats": {
            "source": "docs/STATUS_AUDIT.md and docs/PHASE4_DECISION_PACK.md",
            "single_run": "All reported primary metrics originate from a single training run (seed 42).",
            "seed_variance": "A replicate run with the identical recipe at seed 43 yielded test Macro-F1 = 0.3636, demonstrating substantial seed-to-seed variance (|Delta F1| = 0.1054).",
            "internal_test_split": "Evaluated strictly on the held-out partition of MILK10k (N=786 dermoscopic lesions); zero external hospital validation.",
            "tail_class_supports": tail_supports,
            "published_benchmark_incomparability": "Not comparable with the published MILK10k benchmark top-1 accuracy (53.6%), which used a different multi-institutional partition, different modality combinations, and different pre-training adapters.",
            "clinical_decision_support_only": "Research prototype for decision-support auditing; strictly not approved for autonomous primary diagnostic triage."
        }
    }

    out_file = PROJECT_ROOT / "results" / "frontend_report_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved frontend report summary to {out_file}")

if __name__ == "__main__":
    main()

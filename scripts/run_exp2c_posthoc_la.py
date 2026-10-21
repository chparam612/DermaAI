"""Post-Hoc Logit Adjustment on Exp-2c PanDerm Uniform Control Arm.

Applies Menon et al. (NeurIPS 2020 / ICLR 2021) post-hoc logit adjustment:
    pred = argmax_k(log p_k - tau * log pi_k)
where pi_k are empirical class priors from the dermoscopic training split.
Sweeps tau in [0.0, 1.0] with step 0.05 on VALIDATION ONLY.
Selects tau* maximizing validation macro-F1 per seed, then evaluates once on test split.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

logger = logging.getLogger("run_exp2c_posthoc_la")

MALIGNANT_CLASSES = ["BCC", "MEL", "SCCKA", "MAL_OTH", "AKIEC"]


def load_priors(
    splits_file: Path,
    label_mapping_file: Path,
) -> Tuple[np.ndarray, Dict[str, int], List[str]]:
    """Compute training set priors from lesion_level_splits.csv."""
    with open(label_mapping_file, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    classes = mapping["classes"]

    splits_df = pd.read_csv(splits_file)
    train_dermo = splits_df[(splits_df["split"] == "train") & (splits_df["image_type"] == "dermoscopic")]
    counts = train_dermo["diagnosis"].value_counts().to_dict()
    total = len(train_dermo)

    priors = np.array([counts.get(c, 0) / total for c in classes], dtype=np.float64)
    counts_dict = {c: counts.get(c, 0) for c in classes}
    return priors, counts_dict, classes


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
) -> Dict[str, float]:
    """Compute macro_f1, accuracy, balanced_accuracy, malignant_recall, and bcc_recall."""
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    mal_indices = [i for i, c in enumerate(classes) if c in MALIGNANT_CLASSES]
    y_true_mal = np.isin(y_true, mal_indices).astype(int)
    y_pred_mal = np.isin(y_pred, mal_indices).astype(int)
    mal_rec = float(recall_score(y_true_mal, y_pred_mal, zero_division=0))

    bcc_idx = classes.index("BCC")
    per_class_rec = recall_score(
        y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0
    )
    bcc_rec = float(per_class_rec[bcc_idx])

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "malignant_recall": mal_rec,
        "bcc_recall": bcc_rec,
    }


def find_artifact_csv(base_dir: Path, filename: str) -> Path:
    """Find CSV file directly or inside nested artifacts directory."""
    direct = base_dir / filename
    if direct.is_file():
        return direct
    matches = list(base_dir.rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find {filename} under {base_dir}")


def run_sweep_for_seed(
    seed: int,
    base_dir: Path,
    classes: List[str],
    priors: np.ndarray,
    tau_values: np.ndarray,
) -> Dict[str, Any]:
    """Run validation sweep and single test evaluation for a seed."""
    val_csv = find_artifact_csv(base_dir, "predictions_val.csv")
    test_csv = find_artifact_csv(base_dir, "predictions_test.csv")

    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    prob_cols = [f"prob_{c}" for c in classes]
    p_val = np.clip(df_val[prob_cols].values, 1e-12, 1.0)
    log_p_val = np.log(p_val)
    y_val = df_val["true_label"].values

    p_test = np.clip(df_test[prob_cols].values, 1e-12, 1.0)
    log_p_test = np.log(p_test)
    y_test = df_test["true_label"].values

    log_priors = np.log(np.clip(priors, 1e-12, 1.0))

    # 1. Validation Sweep
    sweep_records = []
    best_tau = 0.0
    best_val_f1 = -1.0
    best_val_bal_acc = -1.0

    for tau in tau_values:
        tau_val = round(float(tau), 4)
        adj_val = log_p_val - tau_val * log_priors
        pred_val = np.argmax(adj_val, axis=1)
        metrics = compute_metrics(y_val, pred_val, classes)
        metrics["tau"] = tau_val
        sweep_records.append(metrics)

        # Primary selection: val macro-F1, secondary: val balanced accuracy
        if (metrics["macro_f1"] > best_val_f1) or (
            np.isclose(metrics["macro_f1"], best_val_f1, atol=1e-5)
            and metrics["balanced_accuracy"] > best_val_bal_acc
        ):
            best_val_f1 = metrics["macro_f1"]
            best_val_bal_acc = metrics["balanced_accuracy"]
            best_tau = tau_val

    # 2. Reference evaluation (tau = 0.0)
    ref_val = [r for r in sweep_records if np.isclose(r["tau"], 0.0)][0]
    pred_test_ref = np.argmax(log_p_test - 0.0 * log_priors, axis=1)
    ref_test = compute_metrics(y_test, pred_test_ref, classes)
    ref_test["tau"] = 0.0

    # 3. Optimal evaluation on test with chosen tau*
    opt_val = [r for r in sweep_records if np.isclose(r["tau"], best_tau)][0]
    pred_test_opt = np.argmax(log_p_test - best_tau * log_priors, axis=1)
    opt_test = compute_metrics(y_test, pred_test_opt, classes)
    opt_test["tau"] = best_tau

    return {
        "seed": seed,
        "best_tau": best_tau,
        "val_sweep": sweep_records,
        "ref_val": ref_val,
        "ref_test": ref_test,
        "opt_val": opt_val,
        "opt_test": opt_test,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-hoc logit adjustment on Exp-2c results.")
    parser.add_argument(
        "--splits-file",
        type=str,
        default="data/splits/lesion_level_splits.csv",
        help="Path to lesion_level_splits.csv",
    )
    parser.add_argument(
        "--label-mapping",
        type=str,
        default="configs/label_mapping.json",
        help="Path to label_mapping.json",
    )
    parser.add_argument(
        "--exp2c-seed42-dir",
        type=str,
        default="results/exp2c_panderm_uniform_seed42",
        help="Directory containing Exp-2c seed 42 predictions",
    )
    parser.add_argument(
        "--exp2c-seed43-dir",
        type=str,
        default="results/exp2c_panderm_uniform_seed43",
        help="Directory containing Exp-2c seed 43 predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/analysis",
        help="Output directory for analysis results",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load priors
    priors, counts_dict, classes = load_priors(
        splits_file=Path(args.splits_file),
        label_mapping_file=Path(args.label_mapping),
    )

    print("=" * 80)
    print("TRAINING SPLIT CLASS COUNTS AND PRIORS (Dermoscopic Modality)")
    print("=" * 80)
    total_train = sum(counts_dict.values())
    for c in classes:
        p_c = priors[classes.index(c)]
        lp_c = np.log(p_c)
        cnt = counts_dict[c]
        print(f"  {c:<10}: count={cnt:4d} | prior={p_c:.5f} | log(pi)={lp_c:.4f}")
    print(f"Total Dermoscopic Training Lesions: {total_train}")
    print("=" * 80)

    # 2. Sweep tau in [0.0, 1.0] step 0.05
    tau_values = np.round(np.arange(0.0, 1.01, 0.05), 2)

    results = {}
    seeds = [42, 43]
    seed_dirs = {
        42: Path(args.exp2c_seed42_dir),
        43: Path(args.exp2c_seed43_dir),
    }

    for seed in seeds:
        print(f"\n" + "=" * 80)
        print(f"EXP-2C SEED {seed}: VALIDATION SWEEP (tau in [0.0, 1.0], step 0.05)")
        print("=" * 80)
        seed_res = run_sweep_for_seed(
            seed=seed,
            base_dir=seed_dirs[seed],
            classes=classes,
            priors=priors,
            tau_values=tau_values,
        )
        results[seed] = seed_res

        # Print validation sweep table
        print(f"{'tau':>6} | {'Val Macro-F1':>12} | {'Val Acc':>9} | {'Val Bal Acc':>11} | {'Val Mal Rec':>11} | {'Val BCC Rec':>11}")
        print("-" * 75)
        for row in seed_res["val_sweep"]:
            mark = " <-- tau*" if np.isclose(row["tau"], seed_res["best_tau"]) else ""
            print(
                f"{row['tau']:6.2f} | {row['macro_f1']:12.4f} | {row['accuracy']:9.4f} | "
                f"{row['balanced_accuracy']:11.4f} | {row['malignant_recall']:11.4f} | "
                f"{row['bcc_recall']:11.4f}{mark}"
            )

        print("\n" + f"SELECTED tau* FOR SEED {seed}: {seed_res['best_tau']:.2f} (Val Macro-F1: {seed_res['opt_val']['macro_f1']:.4f})")
        print("-" * 80)
        print("SEED RESULTS (Reference tau=0.0 vs Optimal tau*):")
        print(f"  Reference (tau=0.0):")
        print(f"    Validation : Macro-F1={seed_res['ref_val']['macro_f1']:.4f} | Acc={seed_res['ref_val']['accuracy']:.4f} | Bal Acc={seed_res['ref_val']['balanced_accuracy']:.4f} | Mal Rec={seed_res['ref_val']['malignant_recall']:.4f} | BCC Rec={seed_res['ref_val']['bcc_recall']:.4f}")
        print(f"    Test       : Macro-F1={seed_res['ref_test']['macro_f1']:.4f} | Acc={seed_res['ref_test']['accuracy']:.4f} | Bal Acc={seed_res['ref_test']['balanced_accuracy']:.4f} | Mal Rec={seed_res['ref_test']['malignant_recall']:.4f} | BCC Rec={seed_res['ref_test']['bcc_recall']:.4f}")
        print(f"  Optimal (tau*={seed_res['best_tau']:.2f}):")
        print(f"    Validation : Macro-F1={seed_res['opt_val']['macro_f1']:.4f} | Acc={seed_res['opt_val']['accuracy']:.4f} | Bal Acc={seed_res['opt_val']['balanced_accuracy']:.4f} | Mal Rec={seed_res['opt_val']['malignant_recall']:.4f} | BCC Rec={seed_res['opt_val']['bcc_recall']:.4f}")
        print(f"    Test       : Macro-F1={seed_res['opt_test']['macro_f1']:.4f} | Acc={seed_res['opt_test']['accuracy']:.4f} | Bal Acc={seed_res['opt_test']['balanced_accuracy']:.4f} | Mal Rec={seed_res['opt_test']['malignant_recall']:.4f} | BCC Rec={seed_res['opt_test']['bcc_recall']:.4f}")

    # 3. Save JSON deliverable
    json_path = out_dir / "exp2c_posthoc_la_results.json"
    deliverable_data = {
        "description": "Post-hoc logit adjustment sweep on Exp-2c PanDerm uniform control arm.",
        "formula": "pred = argmax_k(log p_k - tau * log pi_k)",
        "training_counts_dermoscopic": counts_dict,
        "training_total_samples": total_train,
        "classes": classes,
        "seeds": {
            str(s): {
                "best_tau": results[s]["best_tau"],
                "ref_val": results[s]["ref_val"],
                "ref_test": results[s]["ref_test"],
                "opt_val": results[s]["opt_val"],
                "opt_test": results[s]["opt_test"],
                "val_sweep": results[s]["val_sweep"],
            }
            for s in seeds
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(deliverable_data, f, indent=2)
    logger.info(f"Saved results JSON to {json_path}")

    # 4. Save Markdown report
    md_path = out_dir / "exp2c_posthoc_la_report.md"
    md_lines = [
        "# Post-Hoc Logit Adjustment on Exp-2c (PanDerm Uniform Control Arm)\n",
        "## 1. Methodology\n",
        "- **Adjustment Formula**: $\\hat{y} = \\arg\\max_k (\\log p_k - \\tau \\cdot \\log \\pi_k)$",
        "- **Priors**: Computed on MILK10k dermoscopic training split ($N = 3,668$):",
    ]
    for c in classes:
        p_c = priors[classes.index(c)]
        lp_c = np.log(p_c)
        cnt = counts_dict[c]
        md_lines.append(f"  - `{c:<10}`: N = {cnt:4d} (pi = {p_c:.5f}, log(pi) = {lp_c:.4f})")
    md_lines.extend([
        "\n- **Hyperparameter Tuning**: $\\tau \\in [0.00, 1.00]$ in increments of $0.05$ evaluated **strictly on the validation split** ($N = 786$).",
        "- **Selection Rule**: $\\tau^*$ selected per seed solely by validation Macro-F1.",
        "- **Test Evaluation**: Chosen $\\tau^*$ applied exactly once to the held-out test split ($N = 786$). No test labels used during selection.\n",
        "## 2. Validation Sweeps\n",
    ])

    for seed in seeds:
        md_lines.append(f"### Seed {seed} Validation Sweep Table\n")
        md_lines.append("| tau | Val Macro-F1 | Val Accuracy | Val Balanced Acc | Val Malignant Recall | Val BCC Recall | Note |")
        md_lines.append("|:---:|:---:|:---:|:---:|:---:|:---:|:---|")
        for row in results[seed]["val_sweep"]:
            note = f"**Optimal (tau* = {results[seed]['best_tau']:.2f})**" if np.isclose(row["tau"], results[seed]["best_tau"]) else ("Reference (Baseline)" if np.isclose(row["tau"], 0.0) else "")
            md_lines.append(
                f"| {row['tau']:.2f} | {row['macro_f1']:12.4f} | {row['accuracy']:9.4f} | "
                f"{row['balanced_accuracy']:11.4f} | {row['malignant_recall']:11.4f} | "
                f"{row['bcc_recall']:11.4f} | {note} |"
            )
        md_lines.append("")

    md_lines.append("## 3. Summary Performance Comparison (Reference tau=0 vs Optimal tau*)\n")
    md_lines.append("| Arm / Seed | tau | Split | Macro-F1 | Accuracy | Balanced Acc | Malignant Recall | BCC Recall |")
    md_lines.append("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for seed in seeds:
        res = results[seed]
        bt = res["best_tau"]
        md_lines.append(f"| Exp-2c (seed {seed}) Ref | 0.00 | Val | {res['ref_val']['macro_f1']:.4f} | {res['ref_val']['accuracy']:.4f} | {res['ref_val']['balanced_accuracy']:.4f} | {res['ref_val']['malignant_recall']:.4f} | {res['ref_val']['bcc_recall']:.4f} |")
        md_lines.append(f"| Exp-2c (seed {seed}) Ref | 0.00 | Test | {res['ref_test']['macro_f1']:.4f} | {res['ref_test']['accuracy']:.4f} | {res['ref_test']['balanced_accuracy']:.4f} | {res['ref_test']['malignant_recall']:.4f} | {res['ref_test']['bcc_recall']:.4f} |")
        md_lines.append(f"| Exp-2c (seed {seed}) Post-Hoc LA | {bt:.2f} | Val | {res['opt_val']['macro_f1']:.4f} | {res['opt_val']['accuracy']:.4f} | {res['opt_val']['balanced_accuracy']:.4f} | {res['opt_val']['malignant_recall']:.4f} | {res['opt_val']['bcc_recall']:.4f} |")
        md_lines.append(f"| Exp-2c (seed {seed}) Post-Hoc LA | {bt:.2f} | Test | {res['opt_test']['macro_f1']:.4f} | {res['opt_test']['accuracy']:.4f} | {res['opt_test']['balanced_accuracy']:.4f} | {res['opt_test']['malignant_recall']:.4f} | {res['opt_test']['bcc_recall']:.4f} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    logger.info(f"Saved Markdown report to {md_path}")


if __name__ == "__main__":
    main()

"""Generate ablation table from experiment summary metrics.

Pulls metrics from Exp-1 and Exp-2, compiles them into a structured ablation table,
and saves results as both CSV and GitHub-flavored Markdown for paper inclusion.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("generate_ablation_table")

ABLATION_COLUMNS = [
    "exp_id",
    "model",
    "modality",
    "accuracy",
    "macro_f1",
    "malignant_recall",
    "auroc",
    "notes",
]


def _fmt4(val: Any) -> str:
    """Format numeric value to exactly 4 decimal places."""
    return f"{float(val):.4f}"


def load_metrics_from_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from summary_metrics.json or metrics.json, resolving nested dirs if needed."""
    resolved_path: Optional[Path] = None
    if summary_path.is_file():
        resolved_path = summary_path
    else:
        dir_to_search = summary_path if summary_path.is_dir() else summary_path.parent
        direct = dir_to_search / "summary_metrics.json"
        if direct.is_file():
            resolved_path = direct
        else:
            for sub in dir_to_search.rglob("summary_metrics.json"):
                resolved_path = sub
                break
            if resolved_path is None:
                direct_m = dir_to_search / "metrics.json"
                if direct_m.is_file():
                    resolved_path = direct_m
                else:
                    for sub_m in dir_to_search.rglob("metrics.json"):
                        resolved_path = sub_m
                        break

    if resolved_path is None or not resolved_path.is_file():
        return None

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if "test_overall" in data:
            return data
        if "overall_metrics" in data:
            return {"test_overall": data.get("overall_metrics", {})}
        return data


def _get_best_epoch(summary_path: Optional[Path]) -> Optional[int]:
    """Attempt to find best epoch from training_curves.json in the summary directory."""
    if summary_path is None:
        return None
    dir_path = summary_path if summary_path.is_dir() else summary_path.parent
    for tc_path in [dir_path / "training_curves.json", *dir_path.rglob("training_curves.json")]:
        if tc_path.is_file():
            try:
                with open(tc_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                val_f1 = data.get("val_macro_f1", [])
                epochs = data.get("epochs", [])
                if val_f1 and epochs and len(val_f1) == len(epochs):
                    best_idx = max(range(len(val_f1)), key=lambda i: val_f1[i])
                    return int(epochs[best_idx])
            except Exception:
                pass
    return None


def build_ablation_table(
    exp1_summary_path: Path,
    exp2_summary_path: Path,
    output_dir: Path,
    exp2_v2_summary_path: Optional[Path] = None,
    exp3a_summary_path: Optional[Path] = None,
    exp3b_summary_path: Optional[Path] = None,
    exp2c_seed42_summary_path: Optional[Path] = None,
    exp2c_seed43_summary_path: Optional[Path] = None,
    exp1_seed43_summary_path: Optional[Path] = None,
    exp3b_seed43_summary_path: Optional[Path] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Compile ablation table from Exp-1, Exp-2, Exp-3a, Exp-2c, and Exp-3b metrics."""
    records = []

    # Exp-1: ResNet-50 (Seeds 42 and 43)
    exp1_records = []

    # Exp-1 seed 42
    if exp1_summary_path:
        exp1_data = load_metrics_from_summary(exp1_summary_path)
        if exp1_data:
            m1 = exp1_data.get("test_overall", {})
            acc42 = float(m1.get("accuracy", 0.0))
            f1_42 = float(m1.get("macro_f1", 0.0))
            rec42 = float(m1.get("malignant_recall", 0.0))
            auroc42 = float(m1.get("macro_auroc", 0.0))
            mal_auroc42 = float(m1.get("malignant_auroc", 0.0))
            auroc_str42 = f"{auroc42:.4f} (mal: {mal_auroc42:.4f})" if mal_auroc42 > 0 else f"{auroc42:.4f}"
            records.append({
                "exp_id": "Exp-1 (seed 42)",
                "model": "ResNet-50",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc42),
                "macro_f1": _fmt4(f1_42),
                "malignant_recall": _fmt4(rec42),
                "auroc": auroc_str42,
                "notes": "30-epoch run, seed 42, Kaggle T4 (ImageNet ResNet-50, class-balanced CE, unweighted sampler; differs from Exp-2 in training recipe as well as backbone; not a pure backbone effect)",
            })
            exp1_records.append({
                "acc": acc42,
                "f1": f1_42,
                "rec": rec42,
                "auroc": auroc42,
                "mal_auroc": mal_auroc42,
            })
        else:
            msg = f"Could not find Exp-1 (seed 42) metrics at {exp1_summary_path}"
            if strict:
                raise FileNotFoundError(msg)
            logger.warning(msg)

    # Exp-1 seed 43
    if exp1_seed43_summary_path:
        exp1_43_data = load_metrics_from_summary(exp1_seed43_summary_path)
        if exp1_43_data:
            m1_43 = exp1_43_data.get("test_overall", {})
            acc43 = float(m1_43.get("accuracy", 0.0))
            f1_43 = float(m1_43.get("macro_f1", 0.0))
            rec43 = float(m1_43.get("malignant_recall", 0.0))
            auroc43 = float(m1_43.get("macro_auroc", 0.0))
            mal_auroc43 = float(m1_43.get("malignant_auroc", 0.0))
            auroc_str43 = f"{auroc43:.4f} (mal: {mal_auroc43:.4f})" if mal_auroc43 > 0 else f"{auroc43:.4f}"
            ep1_43 = _get_best_epoch(exp1_seed43_summary_path)
            ep1_43_str = f", best epoch {ep1_43}" if ep1_43 is not None else ""
            notes1_43 = f"30-epoch run, seed 43, Kaggle T4{ep1_43_str}"
            records.append({
                "exp_id": "Exp-1 (seed 43)",
                "model": "ResNet-50",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc43),
                "macro_f1": _fmt4(f1_43),
                "malignant_recall": _fmt4(rec43),
                "auroc": auroc_str43,
                "notes": notes1_43,
            })
            exp1_records.append({
                "acc": acc43,
                "f1": f1_43,
                "rec": rec43,
                "auroc": auroc43,
                "mal_auroc": mal_auroc43,
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-1 (seed 43) metrics at {exp1_seed43_summary_path}")

    # Exp-1 mean (n=2)
    if len(exp1_records) >= 2:
        acc_vals = [r["acc"] for r in exp1_records]
        f1_vals = [r["f1"] for r in exp1_records]
        rec_vals = [r["rec"] for r in exp1_records]
        auroc_vals = [r["auroc"] for r in exp1_records]
        mal_auroc_vals = [r["mal_auroc"] for r in exp1_records]

        m_acc, s_acc = np.mean(acc_vals), np.std(acc_vals, ddof=1)
        m_f1, s_f1 = np.mean(f1_vals), np.std(f1_vals, ddof=1)
        m_rec, s_rec = np.mean(rec_vals), np.std(rec_vals, ddof=1)
        m_auroc, s_auroc = np.mean(auroc_vals), np.std(auroc_vals, ddof=1)
        m_mal_auroc, s_mal_auroc = np.mean(mal_auroc_vals), np.std(mal_auroc_vals, ddof=1)

        mean_auroc_str = f"{m_auroc:.4f} \u00b1 {s_auroc:.4f} (mal: {m_mal_auroc:.4f} \u00b1 {s_mal_auroc:.4f})"
        notes_mean = "mean and sample std over seeds 42 and 43"

        records.append({
            "exp_id": "Exp-1 mean (n=2)",
            "model": "ResNet-50",
            "modality": "Dermoscopic",
            "accuracy": f"{m_acc:.4f} \u00b1 {s_acc:.4f}",
            "macro_f1": f"{m_f1:.4f} \u00b1 {s_f1:.4f}",
            "malignant_recall": f"{m_rec:.4f} \u00b1 {s_rec:.4f}",
            "auroc": mean_auroc_str,
            "notes": notes_mean,
        })

    # Exp-2 v1 (original unweighted collapse run)
    exp2_data = load_metrics_from_summary(exp2_summary_path)
    has_v2 = exp2_v2_summary_path is not None and (exp2_v2_summary_path.is_file() or (exp2_v2_summary_path.parent / "metrics.json").is_file())
    if exp2_data:
        m2 = exp2_data.get("test_overall", {})
        acc = m2.get("accuracy", 0.0)
        f1 = m2.get("macro_f1", 0.0)
        mal_rec = m2.get("malignant_recall", 0.0)
        auroc = m2.get("macro_auroc", 0.0)
        mal_auroc = m2.get("malignant_auroc", 0.0)
        auroc_str = f"{auroc:.4f} (mal: {mal_auroc:.4f})" if mal_auroc > 0 else f"{auroc:.4f}"
        label = "Exp-2 (v1)" if has_v2 else "Exp-2"
        notes = "v1 recipe (unweighted sampler, majority-class collapse onto BCC/NV)" if has_v2 else "Dermatology foundation model fine-tuned via official recipe"
        records.append({
            "exp_id": label,
            "model": "PanDerm ViT-Base",
            "modality": "Dermoscopic",
            "accuracy": _fmt4(acc),
            "macro_f1": _fmt4(f1),
            "malignant_recall": _fmt4(mal_rec),
            "auroc": auroc_str,
            "notes": notes,
        })
    else:
        msg = f"Could not find Exp-2 metrics at {exp2_summary_path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)

    # Exp-2 v2 (corrected recipe)
    if has_v2 and exp2_v2_summary_path is not None:
        v2_data = load_metrics_from_summary(exp2_v2_summary_path)
        if v2_data:
            m2_v2 = v2_data.get("test_overall", {})
            acc = m2_v2.get("accuracy", 0.0)
            f1 = m2_v2.get("macro_f1", 0.0)
            mal_rec = m2_v2.get("malignant_recall", 0.0)
            auroc = m2_v2.get("macro_auroc", 0.0)
            mal_auroc = m2_v2.get("malignant_auroc", 0.0)
            auroc_str = f"{auroc:.4f} (mal: {mal_auroc:.4f})" if mal_auroc > 0 else f"{auroc:.4f}"
            records.append({
                "exp_id": "Exp-2 (v2)",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc),
                "macro_f1": _fmt4(f1),
                "malignant_recall": _fmt4(mal_rec),
                "auroc": auroc_str,
                "notes": "v2 corrected (weighted sampler, layer_decay=0.9, val_macro_f1 checkpointing; selection on val only)",
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-2 (v2) metrics at {exp2_v2_summary_path}")

    # Exp-3a: Post-Hoc Logit Adjustment (Menon et al., NeurIPS 2020)
    if exp3a_summary_path and (exp3a_summary_path.is_file() or (exp3a_summary_path.parent / "metrics.json").is_file()):
        d3a = load_metrics_from_summary(exp3a_summary_path)
        if d3a:
            m3a = d3a.get("test_overall", {})
            acc = m3a.get("accuracy", 0.0)
            f1 = m3a.get("macro_f1", 0.0)
            mal_rec = m3a.get("malignant_recall", 0.0)
            auroc = m3a.get("macro_auroc", 0.0)
            mal_auroc = m3a.get("malignant_auroc", 0.0)
            auroc_str = f"{auroc:.4f} (mal: {mal_auroc:.4f})" if mal_auroc > 0 else f"{auroc:.4f}"
            tau_val = d3a.get("optimal_tau", 0.10)
            records.append({
                "exp_id": "Exp-3a",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc),
                "macro_f1": _fmt4(f1),
                "malignant_recall": _fmt4(mal_rec),
                "auroc": auroc_str,
                "notes": f"NULL result (Delta F1 CI includes 0; double-correction supported by mechanism; MEL +1.49% is 1 lesion; tau*={tau_val:.2f})",
            })

    # Exp-2c: PanDerm ViT-Base Control Arm (Uniform Shuffling, Seeds 42 and 43)
    exp2c_records = []

    # Exp-2c seed 42
    if exp2c_seed42_summary_path:
        d2c_42 = load_metrics_from_summary(exp2c_seed42_summary_path)
        if d2c_42:
            m42 = d2c_42.get("test_overall", {})
            acc42 = float(m42.get("accuracy", 0.0))
            f1_42 = float(m42.get("macro_f1", 0.0))
            rec42 = float(m42.get("malignant_recall", 0.0))
            auroc42 = float(m42.get("macro_auroc", 0.0))
            mal_auroc42 = float(m42.get("malignant_auroc", 0.0))
            auroc_str42 = f"{auroc42:.4f} (mal: {mal_auroc42:.4f})" if mal_auroc42 > 0 else f"{auroc42:.4f}"
            ep42 = _get_best_epoch(exp2c_seed42_summary_path)
            ep42_str = f" (epoch {ep42})" if ep42 is not None else ""
            notes42 = f"uniform shuffling, no weighted sampler; evaluation re-run from best_model.pt{ep42_str} after a torch.load error; runtime_info not valid"
            records.append({
                "exp_id": "Exp-2c (seed 42)",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc42),
                "macro_f1": _fmt4(f1_42),
                "malignant_recall": _fmt4(rec42),
                "auroc": auroc_str42,
                "notes": notes42,
            })
            exp2c_records.append({
                "acc": acc42,
                "f1": f1_42,
                "rec": rec42,
                "auroc": auroc42,
                "mal_auroc": mal_auroc42,
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-2c (seed 42) metrics at {exp2c_seed42_summary_path}")

    # Exp-2c seed 43
    if exp2c_seed43_summary_path:
        d2c_43 = load_metrics_from_summary(exp2c_seed43_summary_path)
        if d2c_43:
            m43 = d2c_43.get("test_overall", {})
            acc43 = float(m43.get("accuracy", 0.0))
            f1_43 = float(m43.get("macro_f1", 0.0))
            rec43 = float(m43.get("malignant_recall", 0.0))
            auroc43 = float(m43.get("macro_auroc", 0.0))
            mal_auroc43 = float(m43.get("malignant_auroc", 0.0))
            auroc_str43 = f"{auroc43:.4f} (mal: {mal_auroc43:.4f})" if mal_auroc43 > 0 else f"{auroc43:.4f}"
            ep43 = _get_best_epoch(exp2c_seed43_summary_path)
            ep43_str = f", best epoch {ep43}" if ep43 is not None else ""
            notes43 = f"uniform shuffling, no weighted sampler; ran end to end with checkpoint fix{ep43_str}"
            records.append({
                "exp_id": "Exp-2c (seed 43)",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc43),
                "macro_f1": _fmt4(f1_43),
                "malignant_recall": _fmt4(rec43),
                "auroc": auroc_str43,
                "notes": notes43,
            })
            exp2c_records.append({
                "acc": acc43,
                "f1": f1_43,
                "rec": rec43,
                "auroc": auroc43,
                "mal_auroc": mal_auroc43,
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-2c (seed 43) metrics at {exp2c_seed43_summary_path}")

    # Exp-2c mean (n=2)
    if len(exp2c_records) >= 2:
        acc_vals = [r["acc"] for r in exp2c_records]
        f1_vals = [r["f1"] for r in exp2c_records]
        rec_vals = [r["rec"] for r in exp2c_records]
        auroc_vals = [r["auroc"] for r in exp2c_records]
        mal_auroc_vals = [r["mal_auroc"] for r in exp2c_records]

        m_acc, s_acc = np.mean(acc_vals), np.std(acc_vals, ddof=1)
        m_f1, s_f1 = np.mean(f1_vals), np.std(f1_vals, ddof=1)
        m_rec, s_rec = np.mean(rec_vals), np.std(rec_vals, ddof=1)
        m_auroc, s_auroc = np.mean(auroc_vals), np.std(auroc_vals, ddof=1)
        m_mal_auroc, s_mal_auroc = np.mean(mal_auroc_vals), np.std(mal_auroc_vals, ddof=1)

        mean_auroc_str = f"{m_auroc:.4f} \u00b1 {s_auroc:.4f} (mal: {m_mal_auroc:.4f} \u00b1 {s_mal_auroc:.4f})"
        notes_mean = (
            "mean and sample std over seeds 42 and 43; "
            "seed 42 evaluated via re-run from best checkpoint; "
            "means shown are tau=0; post-hoc LA with val-selected tau (42: 0.00, 43: 0.10) gave no meaningful change"
        )

        records.append({
            "exp_id": "Exp-2c mean (n=2)",
            "model": "PanDerm ViT-Base",
            "modality": "Dermoscopic",
            "accuracy": f"{m_acc:.4f} \u00b1 {s_acc:.4f}",
            "macro_f1": f"{m_f1:.4f} \u00b1 {s_f1:.4f}",
            "malignant_recall": f"{m_rec:.4f} \u00b1 {s_rec:.4f}",
            "auroc": mean_auroc_str,
            "notes": notes_mean,
        })

    # Exp-3b: Train-Time Logit Adjustment (Menon et al., ICLR 2021)
    exp3b_records = []
    notes3b = "train-time logit adjustment, planned tau=1.0; BCC recall 0.0 (391 test lesions), accuracy collapse"

    # Exp-3b seed 42
    if exp3b_summary_path and (exp3b_summary_path.is_file() or (exp3b_summary_path.parent / "metrics.json").is_file() or exp3b_summary_path.is_dir()):
        d3b_42 = load_metrics_from_summary(exp3b_summary_path)
        if d3b_42:
            m3b_42 = d3b_42.get("test_overall", {})
            acc42 = float(m3b_42.get("accuracy", 0.0))
            f1_42 = float(m3b_42.get("macro_f1", 0.0))
            rec42 = float(m3b_42.get("malignant_recall", 0.0))
            auroc42 = float(m3b_42.get("macro_auroc", 0.0))
            mal_auroc42 = float(m3b_42.get("malignant_auroc", 0.0))
            auroc_str42 = f"{auroc42:.4f} (mal: {mal_auroc42:.4f})" if mal_auroc42 > 0 else f"{auroc42:.4f}"
            records.append({
                "exp_id": "Exp-3b (seed 42)",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc42),
                "macro_f1": _fmt4(f1_42),
                "malignant_recall": _fmt4(rec42),
                "auroc": auroc_str42,
                "notes": notes3b,
            })
            exp3b_records.append({
                "acc": acc42,
                "f1": f1_42,
                "rec": rec42,
                "auroc": auroc42,
                "mal_auroc": mal_auroc42,
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-3b (seed 42) metrics at {exp3b_summary_path}")

    # Exp-3b seed 43
    if exp3b_seed43_summary_path and exp3b_seed43_summary_path.exists():
        d3b_43 = load_metrics_from_summary(exp3b_seed43_summary_path)
        if d3b_43:
            m3b_43 = d3b_43.get("test_overall", {})
            acc43 = float(m3b_43.get("accuracy", 0.0))
            f1_43 = float(m3b_43.get("macro_f1", 0.0))
            rec43 = float(m3b_43.get("malignant_recall", 0.0))
            auroc43 = float(m3b_43.get("macro_auroc", 0.0))
            mal_auroc43 = float(m3b_43.get("malignant_auroc", 0.0))
            auroc_str43 = f"{auroc43:.4f} (mal: {mal_auroc43:.4f})" if mal_auroc43 > 0 else f"{auroc43:.4f}"
            records.append({
                "exp_id": "Exp-3b (seed 43)",
                "model": "PanDerm ViT-Base",
                "modality": "Dermoscopic",
                "accuracy": _fmt4(acc43),
                "macro_f1": _fmt4(f1_43),
                "malignant_recall": _fmt4(rec43),
                "auroc": auroc_str43,
                "notes": notes3b,
            })
            exp3b_records.append({
                "acc": acc43,
                "f1": f1_43,
                "rec": rec43,
                "auroc": auroc43,
                "mal_auroc": mal_auroc43,
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-3b (seed 43) metrics at {exp3b_seed43_summary_path}")

    # Exp-3b mean (n=2)
    if len(exp3b_records) >= 2:
        acc_vals = [r["acc"] for r in exp3b_records]
        f1_vals = [r["f1"] for r in exp3b_records]
        rec_vals = [r["rec"] for r in exp3b_records]
        auroc_vals = [r["auroc"] for r in exp3b_records]
        mal_auroc_vals = [r["mal_auroc"] for r in exp3b_records]

        m_acc, s_acc = np.mean(acc_vals), np.std(acc_vals, ddof=1)
        m_f1, s_f1 = np.mean(f1_vals), np.std(f1_vals, ddof=1)
        m_rec, s_rec = np.mean(rec_vals), np.std(rec_vals, ddof=1)
        m_auroc, s_auroc = np.mean(auroc_vals), np.std(auroc_vals, ddof=1)
        m_mal_auroc, s_mal_auroc = np.mean(mal_auroc_vals), np.std(mal_auroc_vals, ddof=1)

        mean_auroc_str = f"{m_auroc:.4f} \u00b1 {s_auroc:.4f} (mal: {m_mal_auroc:.4f} \u00b1 {s_mal_auroc:.4f})"
        records.append({
            "exp_id": "Exp-3b mean (n=2)",
            "model": "PanDerm ViT-Base",
            "modality": "Dermoscopic",
            "accuracy": f"{m_acc:.4f} \u00b1 {s_acc:.4f}",
            "macro_f1": f"{m_f1:.4f} \u00b1 {s_f1:.4f}",
            "malignant_recall": f"{m_rec:.4f} \u00b1 {s_rec:.4f}",
            "auroc": mean_auroc_str,
            "notes": "mean and sample std over seeds 42 and 43",
        })

    df = pd.DataFrame(records, columns=ABLATION_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "ablation_table.csv"
    md_path = output_dir / "ablation_table.md"

    df.to_csv(csv_path, index=False)
    logger.info(f"Saved ablation table CSV to {csv_path}")

    # Format Markdown
    md_content = "# Phase 2 Baseline Experiments Ablation Table\n\n"
    try:
        md_content += df.to_markdown(index=False)
    except Exception:
        # Manual fallback
        headers = list(df.columns)
        md_content += "| " + " | ".join(headers) + " |\n"
        md_content += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for _, row in df.iterrows():
            md_content += "| " + " | ".join(str(row[h]) for h in headers) + " |\n"
    md_content += "\n\n*Evaluation performed on MILK10k held-out test split (786 dermoscopic images, zero lesion leakage).*\n"
    md_content += "*Malignant classes evaluated: BCC, MEL, SCCKA, MAL_OTH, AKIEC.*\n"
    md_content += "*Exp-2c post-hoc sweep: means shown are tau=0; post-hoc LA with val-selected tau (42: 0.00, 43: 0.10) gave no meaningful change.*\n"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved ablation table Markdown to {md_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate ablation table for Phase 2 experiments.")
    parser.add_argument(
        "--exp1-dir",
        "--exp1-seed42-dir",
        dest="exp1_dir",
        type=str,
        default="results/exp1_resnet50_seed42",
        help="Directory containing Exp-1 seed 42 results (default: results/exp1_resnet50_seed42)",
    )
    parser.add_argument(
        "--exp1-seed43-dir",
        type=str,
        default="results/exp1_resnet50_seed43",
        help="Directory containing Exp-1 seed 43 results (default: results/exp1_resnet50_seed43)",
    )
    parser.add_argument(
        "--exp2-dir",
        type=str,
        default="results/exp2_panderm",
        help="Directory containing Exp-2 (v1) results (default: results/exp2_panderm)",
    )
    parser.add_argument(
        "--exp2-v2-dir",
        type=str,
        default="results/exp2_panderm_v2",
        help="Directory containing Exp-2 (v2) results (default: results/exp2_panderm_v2)",
    )
    parser.add_argument(
        "--exp3a-dir",
        type=str,
        default="results/exp3a_panderm_posthoc",
        help="Directory containing Exp-3a results (default: results/exp3a_panderm_posthoc)",
    )
    parser.add_argument(
        "--exp3b-dir",
        "--exp3b-seed42-dir",
        dest="exp3b_dir",
        type=str,
        default="results/exp3b_panderm_logit_adj_seed42",
        help="Directory containing Exp-3b seed 42 results (default: results/exp3b_panderm_logit_adj_seed42)",
    )
    parser.add_argument(
        "--exp3b-seed43-dir",
        type=str,
        default="results/exp3b_panderm_logit_adj_seed43",
        help="Directory containing Exp-3b seed 43 results (default: results/exp3b_panderm_logit_adj_seed43)",
    )
    parser.add_argument(
        "--exp2c-seed42-dir",
        type=str,
        default="results/exp2c_panderm_uniform_seed42",
        help="Directory containing Exp-2c seed 42 results (default: results/exp2c_panderm_uniform_seed42)",
    )
    parser.add_argument(
        "--exp2c-seed43-dir",
        type=str,
        default="results/exp2c_panderm_uniform_seed43",
        help="Directory containing Exp-2c seed 43 results (default: results/exp2c_panderm_uniform_seed43)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for ablation table (default: results)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail explicitly if any expected baseline metrics summary file is missing (disables silent skipping)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    exp1_summary = Path(args.exp1_dir) if args.exp1_dir else None
    exp1_s43_summary = Path(args.exp1_seed43_dir) if args.exp1_seed43_dir else None
    exp2_summary = Path(args.exp2_dir) / "summary_metrics.json"
    exp2_v2_summary = (Path(args.exp2_v2_dir) / "summary_metrics.json") if args.exp2_v2_dir else None
    exp3a_summary = (Path(args.exp3a_dir) / "summary_metrics.json") if args.exp3a_dir else None
    exp3b_s42_summary = (Path(args.exp3b_dir)) if args.exp3b_dir else None
    exp3b_s43_summary = (Path(args.exp3b_seed43_dir)) if args.exp3b_seed43_dir else None
    exp2c_s42_summary = (Path(args.exp2c_seed42_dir)) if args.exp2c_seed42_dir else None
    exp2c_s43_summary = (Path(args.exp2c_seed43_dir)) if args.exp2c_seed43_dir else None
    output_dir = Path(args.output_dir)

    df = build_ablation_table(
        exp1_summary,
        exp2_summary,
        output_dir,
        exp2_v2_summary_path=exp2_v2_summary,
        exp3a_summary_path=exp3a_summary,
        exp3b_summary_path=exp3b_s42_summary,
        exp2c_seed42_summary_path=exp2c_s42_summary,
        exp2c_seed43_summary_path=exp2c_s43_summary,
        exp1_seed43_summary_path=exp1_s43_summary,
        exp3b_seed43_summary_path=exp3b_s43_summary,
        strict=args.strict,
    )
    print("\n" + "=" * 70)
    print("SKINCANCERTRACKER ABLATION TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

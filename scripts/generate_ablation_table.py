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


def load_metrics_from_summary(summary_path: Path) -> Optional[Dict[str, Any]]:
    """Load evaluation metrics from summary_metrics.json or metrics.json."""
    if not summary_path.is_file():
        # Fallback to metrics.json
        alt_path = summary_path.parent / "metrics.json"
        if alt_path.is_file():
            with open(alt_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"test_overall": data.get("overall_metrics", {})}
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ablation_table(
    exp1_summary_path: Path,
    exp2_summary_path: Path,
    output_dir: Path,
    exp2_v2_summary_path: Optional[Path] = None,
    strict: bool = False,
) -> pd.DataFrame:
    """Compile ablation table from Exp-1 and Exp-2 metrics."""
    records = []

    # Exp-1
    exp1_data = load_metrics_from_summary(exp1_summary_path)
    if exp1_data:
        m1 = exp1_data.get("test_overall", {})
        acc = m1.get("accuracy", 0.0)
        f1 = m1.get("macro_f1", 0.0)
        mal_rec = m1.get("malignant_recall", 0.0)
        auroc = m1.get("macro_auroc", 0.0)
        mal_auroc = m1.get("malignant_auroc", 0.0)
        auroc_str = f"{auroc:.4f} (mal: {mal_auroc:.4f})" if mal_auroc > 0 else f"{auroc:.4f}"
        records.append({
            "exp_id": "Exp-1",
            "model": "ResNet-50",
            "modality": "Dermoscopic",
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "malignant_recall": round(float(mal_rec), 4),
            "auroc": auroc_str,
            "notes": "Standard CNN baseline, ImageNet weights, class-balanced CE",
        })
    else:
        msg = f"Could not find Exp-1 metrics at {exp1_summary_path}"
        if strict:
            raise FileNotFoundError(msg)
        logger.warning(msg)

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
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(f1), 4),
            "malignant_recall": round(float(mal_rec), 4),
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
                "accuracy": round(float(acc), 4),
                "macro_f1": round(float(f1), 4),
                "malignant_recall": round(float(mal_rec), 4),
                "auroc": auroc_str,
                "notes": "v2 corrected (weighted sampler, layer_decay=0.9, val_macro_f1 checkpointing)",
            })
        elif strict:
            raise FileNotFoundError(f"Could not find Exp-2 (v2) metrics at {exp2_v2_summary_path}")

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

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Saved ablation table Markdown to {md_path}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate ablation table for Phase 2 experiments.")
    parser.add_argument(
        "--exp1-dir",
        type=str,
        default="results/exp1_baseline",
        help="Directory containing Exp-1 results (default: results/exp1_baseline)",
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
        default=None,
        help="Directory containing Exp-2 (v2) results (e.g. results/exp2_panderm_v2 or /kaggle/working/results/exp2_panderm_v2)",
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

    exp1_summary = Path(args.exp1_dir) / "summary_metrics.json"
    exp2_summary = Path(args.exp2_dir) / "summary_metrics.json"
    exp2_v2_summary = (Path(args.exp2_v2_dir) / "summary_metrics.json") if args.exp2_v2_dir else None
    output_dir = Path(args.output_dir)

    df = build_ablation_table(exp1_summary, exp2_summary, output_dir, exp2_v2_summary_path=exp2_v2_summary, strict=args.strict)
    print("\n" + "=" * 70)
    print("PHASE 2 ABLATION TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

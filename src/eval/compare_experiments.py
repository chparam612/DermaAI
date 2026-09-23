"""Experiment comparison and ablation table generator."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def generate_ablation_table(
    exp_dirs: Dict[str, Path],
    output_csv: Path = Path("results/ablation_table.csv"),
) -> pd.DataFrame:
    """Read metrics.json from multiple experiments and construct a unified ablation table.

    Args:
        exp_dirs: Dict mapping experiment name to its results directory.
        output_csv: Output path for ablation CSV.

    Returns:
        pd.DataFrame containing comparative metrics.
    """
    rows = []

    for exp_name, res_dir in exp_dirs.items():
        metrics_file = res_dir / "metrics.json"
        if not metrics_file.is_file():
            logger.warning(f"No metrics.json found at {metrics_file}. Skipping {exp_name}.")
            rows.append({
                "Experiment": exp_name,
                "Status": "Pending / Incomplete",
                "Accuracy": None,
                "Balanced Accuracy": None,
                "Macro F1": None,
                "Weighted F1": None,
                "Skin-Tone Disparity": None,
            })
            continue

        with open(metrics_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        overall = data.get("overall_metrics", {})
        subgroups = data.get("subgroup_fairness", {})

        row = {
            "Experiment": exp_name,
            "Status": "Completed",
            "Accuracy": round(overall.get("accuracy", 0.0), 4),
            "Balanced Accuracy": round(overall.get("balanced_accuracy", 0.0), 4),
            "Macro F1": round(overall.get("macro_f1", 0.0), 4),
            "Weighted F1": round(overall.get("weighted_f1", 0.0), 4),
            "Skin-Tone Disparity": round(subgroups.get("balanced_accuracy_disparity", 0.0), 4),
        }

        # Add specific skin tone accuracies if available
        sub_dict = subgroups.get("subgroups", {})
        for st_key in ["skin_tone_1", "skin_tone_2", "skin_tone_3", "skin_tone_4", "skin_tone_5"]:
            if st_key in sub_dict:
                row[f"{st_key}_acc"] = round(sub_dict[st_key].get("accuracy", 0.0), 4)

        rows.append(row)

    df_ablation = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_ablation.to_csv(output_csv, index=False)
    logger.info(f"Saved ablation table to {output_csv}")
    return df_ablation


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparative ablation table.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing experiment subfolders")
    parser.add_argument("--output", type=str, default="results/ablation_table.csv", help="Output path for ablation CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    base_dir = Path(args.results_dir).resolve()
    exp_dirs = {
        "Exp-1 (ResNet-50 Baseline)": base_dir / "exp1_baseline",
        "Exp-2 (PanDerm Fine-Tuning)": base_dir / "exp2_panderm",
    }

    df = generate_ablation_table(exp_dirs, output_csv=Path(args.output).resolve())
    print("\n" + "=" * 70)
    print("PHASE 2 ABLATION TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)


if __name__ == "__main__":
    main()

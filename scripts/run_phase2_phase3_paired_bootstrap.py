"""Paired bootstrap comparison across Exp-1, Exp-2c, and Exp-3b test predictions.

Performs 2000 paired bootstrap resamples of test lesions with identical resample
indices across models to evaluate statistical differences between:
- Exp-2c (PanDerm ViT-Base uniform shuffle) vs Exp-1 (ResNet-50 baseline)
- Exp-3b (PanDerm ViT-Base train-time logit adjustment) vs Exp-2c (uniform shuffle)
both per seed (42, 43) and under a pooled view (averaging seeds within each resample).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score

# Suppress sklearn UserWarning when rare classes are omitted in a bootstrap resample
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger("paired_bootstrap")

MALIGNANT_CLASSES = ["BCC", "MEL", "SCCKA", "MAL_OTH", "AKIEC"]
DEFAULT_METRICS = ["macro_f1", "balanced_accuracy", "malignant_recall", "accuracy"]


def find_predictions_test(exp_dir: Path) -> Path:
    """Find predictions_test.csv in the directory or any subfolder."""
    direct = exp_dir / "predictions_test.csv"
    if direct.is_file():
        return direct
    for p in exp_dir.rglob("predictions_test.csv"):
        if p.is_file():
            return p
    raise FileNotFoundError(f"Could not find predictions_test.csv in {exp_dir}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_true_mal: np.ndarray,
    y_pred_mal: np.ndarray,
) -> Dict[str, float]:
    """Compute macro_f1, balanced_accuracy, malignant_recall, and accuracy."""
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "malignant_recall": float(recall_score(y_true_mal, y_pred_mal, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def compute_percentile_ci(arr: List[float] | np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Compute percentile bootstrap confidence interval."""
    low = float(np.percentile(arr, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def run_paired_bootstrap(
    exp_dirs: Dict[str, Path],
    n_boot: int = 2000,
    seed: int = 42,
    cluster_col: Optional[str] = None,
    output_dir: Optional[Path] = None,
    output_suffix: str = "",
) -> Dict[str, Any]:
    """Execute paired bootstrap evaluation across Exp-1, Exp-2c, and Exp-3b."""
    logger.info("Loading test predictions for 6 experimental runs...")
    csv_paths = {k: find_predictions_test(v) for k, v in exp_dirs.items()}
    dfs = {k: pd.read_csv(v) for k, v in csv_paths.items()}

    # 1. Alignment check across all six files
    ref_key = "exp1_seed42"
    ref_df = dfs[ref_key]
    n_samples = len(ref_df)

    # Check cluster / lesion uniqueness
    cluster_info = {}
    if cluster_col is not None and cluster_col in ref_df.columns:
        n_unique_clusters = int(ref_df[cluster_col].nunique())
        cluster_info = {
            "cluster_col": cluster_col,
            "n_unique_clusters": n_unique_clusters,
            "n_samples": n_samples,
            "has_duplicate_clusters": bool(n_unique_clusters < n_samples),
            "duplicates_count": n_samples - n_unique_clusters,
        }
        logger.info(
            f"Cluster analysis on '{cluster_col}': {n_unique_clusters} unique clusters out of {n_samples} rows "
            f"(duplicates={cluster_info['duplicates_count']})."
        )
    else:
        cluster_info = {
            "cluster_col": None,
            "n_unique_clusters": n_samples,
            "n_samples": n_samples,
            "has_duplicate_clusters": False,
            "duplicates_count": 0,
        }

    alignment_report = {
        "reference_model": ref_key,
        "n_samples": n_samples,
        "cluster_info": cluster_info,
        "files_checked": {},
        "all_pre_aligned": True,
    }

    for k, df in dfs.items():
        same_len = len(df) == n_samples
        same_isic = (df["isic_id"].values == ref_df["isic_id"].values).all() if same_len else False
        same_lesion = (df["lesion_id"].values == ref_df["lesion_id"].values).all() if same_len else False
        same_label = (df["true_label"].values == ref_df["true_label"].values).all() if same_len else False

        alignment_report["files_checked"][k] = {
            "path": str(csv_paths[k]),
            "rows": len(df),
            "identical_isic_order": bool(same_isic),
            "identical_lesion_order": bool(same_lesion),
            "identical_true_label": bool(same_label),
        }

        if not (same_isic and same_label):
            alignment_report["all_pre_aligned"] = False
            logger.warning(f"Aligning {k} by isic_id against reference {ref_key}")
            df_aligned = df.set_index("isic_id").reindex(ref_df["isic_id"]).reset_index()
            if not (df_aligned["true_label"].values == ref_df["true_label"].values).all():
                raise ValueError(f"Mismatch in true_label after alignment for {k}")
            dfs[k] = df_aligned

    logger.info(f"Alignment verification completed. Pre-aligned: {alignment_report['all_pre_aligned']}")

    # Ground truth arrays
    y_true = ref_df["true_label"].values
    true_classes = ref_df["true_class"].values
    y_true_mal = np.isin(true_classes, MALIGNANT_CLASSES).astype(int)

    # Model predictions
    y_preds = {k: dfs[k]["pred_label"].values for k in exp_dirs}
    y_preds_mal = {k: np.isin(dfs[k]["pred_class"].values, MALIGNANT_CLASSES).astype(int) for k in exp_dirs}

    # 2. Point estimates
    point_estimates = {}
    for k in exp_dirs:
        point_estimates[k] = compute_metrics(y_true, y_preds[k], y_true_mal, y_preds_mal[k])

    # 3. Paired bootstrap resampling (SAME resample indices for all models)
    np.random.seed(seed)
    if cluster_col is not None and cluster_col in ref_df.columns:
        logger.info(f"Generating {n_boot} paired cluster-bootstrap resamples by '{cluster_col}' (seed={seed})...")
        unique_clusters = ref_df[cluster_col].unique()
        n_clusters = len(unique_clusters)
        cluster_to_indices = ref_df.groupby(cluster_col).indices
        boot_indices = []
        for _ in range(n_boot):
            sampled_clusters = np.random.choice(unique_clusters, size=n_clusters, replace=True)
            # All rows of a lesion move together
            b_idx = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
            boot_indices.append(b_idx)
    else:
        logger.info(f"Generating {n_boot} paired bootstrap resamples (seed={seed})...")
        boot_indices = [np.random.randint(0, n_samples, size=n_samples) for _ in range(n_boot)]

    boot_model_metrics: Dict[str, Dict[str, List[float]]] = {
        k: {m: [] for m in DEFAULT_METRICS} for k in exp_dirs
    }

    for b_idx in boot_indices:
        b_yt = y_true[b_idx]
        b_yt_mal = y_true_mal[b_idx]

        for k in exp_dirs:
            b_yp = y_preds[k][b_idx]
            b_yp_mal = y_preds_mal[k][b_idx]
            m = compute_metrics(b_yt, b_yp, b_yt_mal, b_yp_mal)
            for metric_name in DEFAULT_METRICS:
                boot_model_metrics[k][metric_name].append(m[metric_name])

    # Model CIs
    model_summaries = {}
    for k in exp_dirs:
        model_summaries[k] = {}
        for m in DEFAULT_METRICS:
            pt = point_estimates[k][m]
            dist = np.array(boot_model_metrics[k][m])
            ci_low, ci_high = compute_percentile_ci(dist)
            model_summaries[k][m] = {
                "point_estimate": float(pt),
                "bootstrap_mean": float(np.mean(dist)),
                "bootstrap_std": float(np.std(dist, ddof=1)),
                "ci_95": [float(ci_low), float(ci_high)],
                "ci_95_str": f"[{ci_low:.4f}, {ci_high:.4f}]",
            }

    # 4. Paired Differences per seed:
    # (a) Exp-2c minus Exp-1 (seed 42)
    # (b) Exp-2c minus Exp-1 (seed 43)
    # (c) Exp-3b minus Exp-2c (seed 42)
    # (d) Exp-3b minus Exp-2c (seed 43)
    comparisons = {
        "exp2c_minus_exp1_seed42": ("exp2c_seed42", "exp1_seed42", "Exp-2c minus Exp-1 (seed 42)"),
        "exp2c_minus_exp1_seed43": ("exp2c_seed43", "exp1_seed43", "Exp-2c minus Exp-1 (seed 43)"),
        "exp3b_minus_exp2c_seed42": ("exp3b_seed42", "exp2c_seed42", "Exp-3b minus Exp-2c (seed 42)"),
        "exp3b_minus_exp2c_seed43": ("exp3b_seed43", "exp2c_seed43", "Exp-3b minus Exp-2c (seed 43)"),
    }

    paired_differences = {}
    for comp_key, (m_b, m_a, comp_name) in comparisons.items():
        paired_differences[comp_key] = {
            "name": comp_name,
            "model_b": m_b,
            "model_a": m_a,
            "metrics": {},
        }
        for m in DEFAULT_METRICS:
            pt_diff = point_estimates[m_b][m] - point_estimates[m_a][m]
            dist_b = np.array(boot_model_metrics[m_b][m])
            dist_a = np.array(boot_model_metrics[m_a][m])
            diff_dist = dist_b - dist_a
            ci_low, ci_high = compute_percentile_ci(diff_dist)
            p_gt_zero = float(np.mean(diff_dist > 0))
            includes_zero = bool(ci_low <= 0.0 <= ci_high)

            paired_differences[comp_key]["metrics"][m] = {
                "point_difference": float(pt_diff),
                "mean_difference": float(np.mean(diff_dist)),
                "std_difference": float(np.std(diff_dist, ddof=1)),
                "ci_95": [float(ci_low), float(ci_high)],
                "ci_95_str": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "fraction_gt_zero": p_gt_zero,
                "ci_includes_zero": includes_zero,
            }

    # 5. Pooled view: average the metric over the two seeds inside each resample
    pooled_boot_metrics = {
        "exp1": {m: (np.array(boot_model_metrics["exp1_seed42"][m]) + np.array(boot_model_metrics["exp1_seed43"][m])) / 2.0 for m in DEFAULT_METRICS},
        "exp2c": {m: (np.array(boot_model_metrics["exp2c_seed42"][m]) + np.array(boot_model_metrics["exp2c_seed43"][m])) / 2.0 for m in DEFAULT_METRICS},
        "exp3b": {m: (np.array(boot_model_metrics["exp3b_seed42"][m]) + np.array(boot_model_metrics["exp3b_seed43"][m])) / 2.0 for m in DEFAULT_METRICS},
    }

    pooled_point_estimates = {
        "exp1": {m: (point_estimates["exp1_seed42"][m] + point_estimates["exp1_seed43"][m]) / 2.0 for m in DEFAULT_METRICS},
        "exp2c": {m: (point_estimates["exp2c_seed42"][m] + point_estimates["exp2c_seed43"][m]) / 2.0 for m in DEFAULT_METRICS},
        "exp3b": {m: (point_estimates["exp3b_seed42"][m] + point_estimates["exp3b_seed43"][m]) / 2.0 for m in DEFAULT_METRICS},
    }

    pooled_comparisons = {
        "exp2c_minus_exp1_pooled": ("exp2c", "exp1", "Exp-2c minus Exp-1 (pooled over seeds 42 and 43)"),
        "exp3b_minus_exp2c_pooled": ("exp3b", "exp2c", "Exp-3b minus Exp-2c (pooled over seeds 42 and 43)"),
    }

    pooled_differences = {}
    for comp_key, (m_b, m_a, comp_name) in pooled_comparisons.items():
        pooled_differences[comp_key] = {
            "name": comp_name,
            "model_b": m_b,
            "model_a": m_a,
            "metrics": {},
        }
        for m in DEFAULT_METRICS:
            pt_diff = pooled_point_estimates[m_b][m] - pooled_point_estimates[m_a][m]
            diff_dist = pooled_boot_metrics[m_b][m] - pooled_boot_metrics[m_a][m]
            ci_low, ci_high = compute_percentile_ci(diff_dist)
            p_gt_zero = float(np.mean(diff_dist > 0))
            includes_zero = bool(ci_low <= 0.0 <= ci_high)

            pooled_differences[comp_key]["metrics"][m] = {
                "point_difference": float(pt_diff),
                "mean_difference": float(np.mean(diff_dist)),
                "std_difference": float(np.std(diff_dist, ddof=1)),
                "ci_95": [float(ci_low), float(ci_high)],
                "ci_95_str": f"[{ci_low:.4f}, {ci_high:.4f}]",
                "fraction_gt_zero": p_gt_zero,
                "ci_includes_zero": includes_zero,
            }

    results = {
        "meta": {
            "n_boot": n_boot,
            "seed": seed,
            "n_test_samples": n_samples,
            "cluster_col": cluster_col,
            "n_unique_clusters": cluster_info["n_unique_clusters"],
            "has_duplicate_clusters": cluster_info["has_duplicate_clusters"],
            "malignant_classes": MALIGNANT_CLASSES,
            "metrics": DEFAULT_METRICS,
            "output_suffix": output_suffix,
        },
        "alignment_report": alignment_report,
        "model_point_estimates_and_cis": model_summaries,
        "paired_differences_per_seed": paired_differences,
        "pooled_differences": pooled_differences,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"phase2_phase3_paired_bootstrap{output_suffix}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved paired bootstrap JSON to {json_path}")

        md_path = output_dir / f"phase2_phase3_paired_bootstrap_report{output_suffix}.md"
        generate_markdown_report(results, md_path)
        logger.info(f"Saved paired bootstrap Markdown report to {md_path}")
        if output_suffix:
            alt_md_path = output_dir / f"phase2_phase3_paired_bootstrap{output_suffix}_report.md"
            generate_markdown_report(results, alt_md_path)

    return results


def generate_markdown_report(results: Dict[str, Any], output_path: Path) -> None:
    """Generate Markdown report detailing paired bootstrap comparison and zero-crossing analysis."""
    meta = results["meta"]
    align = results["alignment_report"]
    models = results["model_point_estimates_and_cis"]
    paired = results["paired_differences_per_seed"]
    pooled = results["pooled_differences"]

    resample_desc = (
        f"Cluster bootstrap by '{meta['cluster_col']}' ({meta['n_unique_clusters']} unique clusters across {meta['n_test_samples']} test rows)"
        if meta.get("cluster_col")
        else "Row-level paired bootstrap (independent test lesions)"
    )

    lines = [
        "# Phase 2 & Phase 3 Paired Bootstrap Statistical Comparison Report",
        "",
        "## Executive Summary",
        f"- **Resamples**: {meta['n_boot']:,} paired bootstrap iterations with identical resample indices across all models.",
        f"- **Resampling Scheme**: {resample_desc}.",
        f"- **Random Seed**: {meta['seed']} (fixed numpy RNG for exact reproducibility).",
        f"- **Test Lesions Evaluated**: {meta['n_test_samples']} held-out dermoscopic lesions from MILK10k.",
        f"- **Malignant Classes**: {', '.join(meta['malignant_classes'])}.",
        "- **Evaluation Role**: Purely scientific evaluation on held-out test split. Recipe selection is governed strictly by validation Macro-F1; test set differences are reported for transparency without selecting recipes based on test results.",
        "",
        "## 1. Test Lesion Alignment Verification",
        f"- **Row & Lesion Order Parity**: All six files evaluated had **{'identical row/lesion order and identical true labels' if align['all_pre_aligned'] else 'been aligned by lesion identifier'}**.",
    ]

    cinfo = align.get("cluster_info", {})
    if cinfo.get("cluster_col"):
        lines.append(
            f"- **Lesion Uniqueness ({cinfo['cluster_col']})**: {cinfo['n_unique_clusters']} unique lesions among {cinfo['n_samples']} rows "
            f"({cinfo['duplicates_count']} duplicate lesions)."
        )
        if not cinfo["has_duplicate_clusters"]:
            lines.append(
                "  - *Note*: Every test row represents an independent unique lesion (1.000 rows/lesion). "
                "Consequently, cluster-level resampling (where all rows of a lesion move together) is mathematically "
                "identical to row-level resampling under the same seed."
            )

    lines.extend([
        "",
        "| Model Run | Predictions File | Rows | Identical ISIC Order | Identical Lesion Order | Identical True Label |",
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ])

    for k, v in align["files_checked"].items():
        fname = Path(v["path"]).name
        lines.append(f"| {k} | `{fname}` | {v['rows']} | {v['identical_isic_order']} | {v['identical_lesion_order']} | {v['identical_true_label']} |")

    lines.extend([
        "",
        "## 2. Individual Model Point Estimates and 95% Percentile CIs",
        "",
        "| Model Run | Macro-F1 (95% CI) | Balanced Accuracy (95% CI) | Malignant Recall (95% CI) | Accuracy (95% CI) |",
        "|:---|:---:|:---:|:---:|:---:|",
    ])

    for k, metrics in models.items():
        f1_str = f"{metrics['macro_f1']['point_estimate']:.4f} {metrics['macro_f1']['ci_95_str']}"
        bal_str = f"{metrics['balanced_accuracy']['point_estimate']:.4f} {metrics['balanced_accuracy']['ci_95_str']}"
        rec_str = f"{metrics['malignant_recall']['point_estimate']:.4f} {metrics['malignant_recall']['ci_95_str']}"
        acc_str = f"{metrics['accuracy']['point_estimate']:.4f} {metrics['accuracy']['ci_95_str']}"
        lines.append(f"| {k} | {f1_str} | {bal_str} | {rec_str} | {acc_str} |")

    lines.extend([
        "",
        "## 3. Seed-Specific Paired Differences",
        "",
        "Differences computed as $M_B - M_A$ on the exact same bootstrap resample indices.",
        "",
        "| Comparison | Metric | Point Diff | Mean Boot Diff | 95% Percentile CI | P(Diff > 0) | CI Includes 0? |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for comp_key, comp_data in paired.items():
        cname = comp_data["name"]
        for m, mdata in comp_data["metrics"].items():
            inc_zero = "YES (includes 0)" if mdata["ci_includes_zero"] else "**NO** (excludes 0)"
            lines.append(
                f"| {cname} | {m} | {mdata['point_difference']:+.4f} | {mdata['mean_difference']:+.4f} | "
                f"`{mdata['ci_95_str']}` | {mdata['fraction_gt_zero']:.4f} | {inc_zero} |"
            )

    lines.extend([
        "",
        "## 4. Pooled View (Averaging Seeds Inside Each Resample)",
        "",
        "In each resample, metric values are averaged over seeds 42 and 43 prior to computing paired differences:",
        r"$$\Delta_{\text{pooled}}^{(b)} = \frac{M_{B, 42}^{(b)} + M_{B, 43}^{(b)}}{2} - \frac{M_{A, 42}^{(b)} + M_{A, 43}^{(b)}}{2}$$",
        "",
        "| Comparison (Pooled) | Metric | Point Diff | Mean Boot Diff | 95% Percentile CI | P(Diff > 0) | CI Includes 0? |",
        "|:---|:---|:---:|:---:|:---:|:---:|:---:|",
    ])

    for comp_key, comp_data in pooled.items():
        cname = comp_data["name"]
        for m, mdata in comp_data["metrics"].items():
            inc_zero = "YES (includes 0)" if mdata["ci_includes_zero"] else "**NO** (excludes 0)"
            lines.append(
                f"| {cname} | {m} | {mdata['point_difference']:+.4f} | {mdata['mean_difference']:+.4f} | "
                f"`{mdata['ci_95_str']}` | {mdata['fraction_gt_zero']:.4f} | {inc_zero} |"
            )

    lines.extend([
        "",
        "## 5. Plain-Language Summary of Zero-Crossing Confidence Intervals",
        "",
        "### A. Exp-2c minus Exp-1 (PanDerm Uniform Shuffling vs ResNet-50 Baseline)",
    ])

    for m in DEFAULT_METRICS:
        p42 = paired["exp2c_minus_exp1_seed42"]["metrics"][m]
        p43 = paired["exp2c_minus_exp1_seed43"]["metrics"][m]
        ppool = pooled["exp2c_minus_exp1_pooled"]["metrics"][m]

        lines.append(f"- **{m}**:")
        lines.append(f"  - Seed 42: Point Diff = {p42['point_difference']:+.4f}, 95% CI = `{p42['ci_95_str']}` -> **{'Includes 0' if p42['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {p42['fraction_gt_zero']:.1%}).")
        lines.append(f"  - Seed 43: Point Diff = {p43['point_difference']:+.4f}, 95% CI = `{p43['ci_95_str']}` -> **{'Includes 0' if p43['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {p43['fraction_gt_zero']:.1%}).")
        lines.append(f"  - Pooled: Point Diff = {ppool['point_difference']:+.4f}, 95% CI = `{ppool['ci_95_str']}` -> **{'Includes 0' if ppool['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {ppool['fraction_gt_zero']:.1%}).")

    lines.extend([
        "",
        "### B. Exp-3b minus Exp-2c (Train-Time Logit Adjustment vs Uniform Shuffling Control)",
    ])

    for m in DEFAULT_METRICS:
        p42 = paired["exp3b_minus_exp2c_seed42"]["metrics"][m]
        p43 = paired["exp3b_minus_exp2c_seed43"]["metrics"][m]
        ppool = pooled["exp3b_minus_exp2c_pooled"]["metrics"][m]

        lines.append(f"- **{m}**:")
        lines.append(f"  - Seed 42: Point Diff = {p42['point_difference']:+.4f}, 95% CI = `{p42['ci_95_str']}` -> **{'Includes 0' if p42['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {p42['fraction_gt_zero']:.1%}).")
        lines.append(f"  - Seed 43: Point Diff = {p43['point_difference']:+.4f}, 95% CI = `{p43['ci_95_str']}` -> **{'Includes 0' if p43['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {p43['fraction_gt_zero']:.1%}).")
        lines.append(f"  - Pooled: Point Diff = {ppool['point_difference']:+.4f}, 95% CI = `{ppool['ci_95_str']}` -> **{'Includes 0' if ppool['ci_includes_zero'] else 'Does NOT include 0'}** (P(>0) = {ppool['fraction_gt_zero']:.1%}).")

    lines.extend([
        "",
        "### C. Scientific Protocol and Selection Rule Compliance",
        "- **Zero Test-Driven Selection**: As established in project protocol, no model recipe is selected or rejected based on test set metrics. Base recipe selection for subsequent phases remains governed strictly by validation Macro-F1.",
        "- **Collapse Observation**: Exp-3b exhibits an accuracy collapse on majority classes (BCC test recall = 0.0, 391 lesions) under train-time logit adjustment $\\tau=1.0$, while elevating malignant recall via over-prediction of non-BCC malignant classes.",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired bootstrap evaluation on Exp-1, Exp-2c, Exp-3b test predictions.")
    parser.add_argument("--exp1-seed42-dir", type=str, default="results/exp1_resnet50_seed42")
    parser.add_argument("--exp1-seed43-dir", type=str, default="results/exp1_resnet50_seed43")
    parser.add_argument("--exp2c-seed42-dir", type=str, default="results/exp2c_panderm_uniform_seed42")
    parser.add_argument("--exp2c-seed43-dir", type=str, default="results/exp2c_panderm_uniform_seed43")
    parser.add_argument("--exp3b-seed42-dir", type=str, default="results/exp3b_panderm_logit_adj_seed42")
    parser.add_argument("--exp3b-seed43-dir", type=str, default="results/exp3b_panderm_logit_adj_seed43")
    parser.add_argument("--n-boot", type=int, default=2000, help="Number of bootstrap resamples (default: 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap resampling (default: 42)")
    parser.add_argument("--cluster-col", type=str, default=None, help="Column to group rows by for cluster bootstrap (e.g. lesion_id)")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix to append to output filenames (e.g. _lesion_level)")
    parser.add_argument("--output-dir", type=str, default="results/analysis", help="Output directory (default: results/analysis)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    exp_dirs = {
        "exp1_seed42": Path(args.exp1_seed42_dir),
        "exp1_seed43": Path(args.exp1_seed43_dir),
        "exp2c_seed42": Path(args.exp2c_seed42_dir),
        "exp2c_seed43": Path(args.exp2c_seed43_dir),
        "exp3b_seed42": Path(args.exp3b_seed42_dir),
        "exp3b_seed43": Path(args.exp3b_seed43_dir),
    }

    run_paired_bootstrap(
        exp_dirs=exp_dirs,
        n_boot=args.n_boot,
        seed=args.seed,
        cluster_col=args.cluster_col,
        output_dir=Path(args.output_dir),
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()

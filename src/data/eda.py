"""Reproducible Exploratory Data Analysis (EDA) module for dermatology datasets.

Computes comprehensive dataset overview, 11-class diagnostic distribution,
skin-tone fairness distributions (using real column skin_tone_class without fabrication),
demographic analyses, cross-tabulations, publication-ready plots, and serialized summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

logger = logging.getLogger("eda")

DIAGNOSTIC_CLASSES_11 = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]


def load_dataset_tables(
    data_root: Union[str, Path],
    input_csv: str = "training_input.csv",
    gt_csv: str = "training_gt.csv",
    metadata_csv: str = "metadata.csv",
    supp_csv: str = "training_supp.csv",
) -> Dict[str, pd.DataFrame]:
    """Load all core dataset CSV tables from data root."""
    root = Path(data_root).resolve()
    tables: Dict[str, pd.DataFrame] = {}

    table_candidates = {
        "training_input": [root / input_csv, root / "supplements" / input_csv],
        "training_gt": [root / gt_csv, root / "supplements" / gt_csv],
        "metadata": [root / metadata_csv, root / "supplements" / metadata_csv],
        "training_supp": [root / supp_csv, root / "supplements" / supp_csv],
    }

    for key, paths in table_candidates.items():
        for p in paths:
            if p.is_file():
                tables[key] = pd.read_csv(p, low_memory=False)
                break

    return tables


def get_dataset_overview(
    tables: Dict[str, pd.DataFrame],
    image_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Calculate comprehensive dataset overview statistics directly from CSVs."""
    overview: Dict[str, Any] = {
        "row_counts": {k: len(v) for k, v in tables.items()},
        "duplicate_rows": {k: int(v.duplicated().sum()) for k, v in tables.items()},
        "missing_values_per_column": {
            k: v.isna().sum().to_dict() for k, v in tables.items()
        },
    }

    overview["images"] = {}
    for k, df in tables.items():
        if "isic_id" in df.columns:
            overview["images"][k] = {
                "unique_isic_ids": int(df["isic_id"].nunique()),
                "duplicate_isic_ids": int(df["isic_id"].duplicated().sum()),
                "null_isic_ids": int(df["isic_id"].isna().sum()),
            }

    overview["lesions"] = {}
    for k, df in tables.items():
        if "lesion_id" in df.columns:
            overview["lesions"][k] = {
                "unique_lesions": int(df["lesion_id"].nunique()),
                "duplicate_lesion_ids": int(df["lesion_id"].duplicated().sum()),
                "null_lesion_ids": int(df["lesion_id"].isna().sum()),
            }

    primary_df = tables["training_input"] if "training_input" in tables else tables.get("metadata")
    if primary_df is not None and "lesion_id" in primary_df.columns:
        counts = primary_df["lesion_id"].dropna().value_counts()
        overview["images_per_lesion"] = {
            "min": int(counts.min()) if not counts.empty else 0,
            "max": int(counts.max()) if not counts.empty else 0,
            "mean": round(float(counts.mean()), 3) if not counts.empty else 0.0,
            "median": float(counts.median()) if not counts.empty else 0.0,
            "std": round(float(counts.std()), 3) if len(counts) > 1 else 0.0,
            "distribution_summary": {
                "1_image": int((counts == 1).sum()),
                "2_to_5_images": int(((counts >= 2) & (counts <= 5)).sum()),
                "6_to_10_images": int(((counts >= 6) & (counts <= 10)).sum()),
                "more_than_10_images": int((counts > 10).sum()),
            },
        }

    if image_dir is not None:
        img_path = Path(image_dir).resolve()
        if img_path.is_dir():
            available_files = {f.name for f in img_path.iterdir() if f.is_file()}
            overview["disk_images"] = {
                "image_dir": str(img_path),
                "total_files_on_disk": len(available_files),
            }

            sample_ids = None
            if primary_df is not None and "isic_id" in primary_df.columns:
                sample_ids = list(primary_df["isic_id"].dropna().astype(str))

            if sample_ids:
                found = 0
                missing = []
                for isic_id in sample_ids:
                    matched = False
                    for ext in ["", ".jpg", ".png", ".jpeg"]:
                        if f"{isic_id}{ext}" in available_files:
                            matched = True
                            found += 1
                            break
                    if not matched:
                        missing.append(isic_id)

                overview["disk_images"]["resolved_count"] = found
                overview["disk_images"]["missing_count"] = len(missing)
                overview["disk_images"]["missing_samples"] = missing[:10]

    return overview


def get_class_distribution(
    gt_df: pd.DataFrame,
    class_columns: Optional[List[str]] = None,
    lesion_id_col: Optional[str] = "lesion_id",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Calculate diagnostic class distribution across the 11 one-hot classes.

    Returns:
        Tuple of (summary_dataframe, imbalance_and_consistency_metrics).
    """
    classes = class_columns or [c for c in DIAGNOSTIC_CLASSES_11 if c in gt_df.columns]
    total_images = len(gt_df)

    stats = []
    for cls in classes:
        if cls not in gt_df.columns:
            continue
        mask = gt_df[cls] == 1
        count = int(mask.sum())
        pct = round((count / total_images * 100), 2) if total_images > 0 else 0.0

        unique_lesions = None
        if lesion_id_col and lesion_id_col in gt_df.columns:
            unique_lesions = int(gt_df.loc[mask, lesion_id_col].nunique())

        stats.append({
            "class_name": cls,
            "image_count": count,
            "percentage": pct,
            "unique_lesions": unique_lesions,
        })

    summary_df = pd.DataFrame(stats)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by="image_count", ascending=False).reset_index(drop=True)

    # Class imbalance ratio & multi-label consistency checks
    available_classes = [c for c in classes if c in gt_df.columns]
    metrics: Dict[str, Any] = {
        "total_records": total_images,
        "classes_evaluated": len(available_classes),
    }

    if not summary_df.empty and len(summary_df) > 1:
        max_count = int(summary_df["image_count"].max())
        min_count = int(summary_df["image_count"].min())
        imbalance_ratio = round(max_count / max(1, min_count), 2)
        metrics["majority_class"] = summary_df.iloc[0]["class_name"]
        metrics["majority_count"] = max_count
        metrics["minority_class"] = summary_df.iloc[-1]["class_name"]
        metrics["minority_count"] = min_count
        metrics["imbalance_ratio_max_to_min"] = imbalance_ratio

    if available_classes:
        row_sums = gt_df[available_classes].sum(axis=1)
        metrics["zero_label_count"] = int((row_sums == 0).sum())
        metrics["single_label_count"] = int((row_sums == 1).sum())
        metrics["multi_label_count"] = int((row_sums > 1).sum())
        metrics["is_strictly_mutually_exclusive"] = bool((row_sums == 1).all())

    return summary_df, metrics


def get_skin_tone_distribution(
    input_df: pd.DataFrame,
    skin_tone_col: str = "skin_tone_class",
    lesion_id_col: str = "lesion_id",
) -> pd.DataFrame:
    """Analyze real skin tone column without fabricating or inferring labels."""
    if skin_tone_col not in input_df.columns:
        return pd.DataFrame()

    total_images = len(input_df)
    s = input_df[skin_tone_col].fillna("Unknown / Missing")

    stats = []
    for val, count in s.value_counts(dropna=False).items():
        val_str = str(val)
        mask = s == val
        pct = round(count / total_images * 100, 2)
        lesion_count = int(input_df.loc[mask, lesion_id_col].nunique()) if lesion_id_col in input_df.columns else None

        stats.append({
            "skin_tone_class": val_str,
            "image_count": int(count),
            "percentage": pct,
            "unique_lesions": lesion_count,
        })

    df = pd.DataFrame(stats)
    return df.sort_values(by="image_count", ascending=False).reset_index(drop=True)


def get_demographic_distributions(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Calculate frequency distributions for demographic and clinical metadata."""
    target_cols = columns or ["age_approx", "sex", "site", "anatom_site_general", "image_type", "image_manipulation", "diagnosis_confirm_type"]
    results: Dict[str, pd.DataFrame] = {}

    for col in target_cols:
        if col in df.columns:
            val_counts = df[col].value_counts(dropna=False)
            total = len(df)
            res_df = pd.DataFrame({
                col: [str(x) if pd.notna(x) else "Unknown / Missing" for x in val_counts.index],
                "count": val_counts.values,
                "percentage": [round(c / total * 100, 2) for c in val_counts.values],
            })
            results[col] = res_df

    return results


def generate_cross_tabulations(
    merged_df: pd.DataFrame,
    class_col: str = "diagnostic_class",
) -> Dict[str, pd.DataFrame]:
    """Generate demographic and diagnostic cross-tabulations."""
    crosstabs: Dict[str, pd.DataFrame] = {}

    pairs = [
        ("class_by_skin_tone", class_col, "skin_tone_class"),
        ("class_by_sex", class_col, "sex"),
        ("class_by_site", class_col, "site" if "site" in merged_df.columns else "anatom_site_general"),
        ("crosstab_skin_tone_vs_sex", "skin_tone_class", "sex"),
        ("crosstab_image_type_vs_class", "image_type", class_col),
    ]

    for name, row_col, col_col in pairs:
        if row_col in merged_df.columns and col_col in merged_df.columns:
            ct = pd.crosstab(
                merged_df[row_col].fillna("Unknown"),
                merged_df[col_col].fillna("Unknown"),
                margins=True,
                margins_name="Total",
            )
            crosstabs[name] = ct

    return crosstabs


# ------------------------------------------------------------------------------
# Publication-Ready Visualization Routines
# ------------------------------------------------------------------------------

def plot_class_distribution(
    class_summary_df: pd.DataFrame,
    title: str = "Dermatology 11-Class Diagnostic Distribution",
    log_scale: bool = False,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Generate bar chart for diagnostic classes."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(class_summary_df))))
    bars = ax.bar(
        class_summary_df["class_name"],
        class_summary_df["image_count"],
        color=palette,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Total Image Count (Log Scale)", fontsize=12)
        title += " (Log Scale)"
    else:
        ax.set_ylabel("Total Image Count", fontsize=12)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Diagnostic Class", fontsize=12, labelpad=10)
    ax.tick_params(axis="x", rotation=45)

    for bar, pct in zip(bars, class_summary_df["percentage"]):
        h = bar.get_height()
        ax.annotate(
            f"{int(h)}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_skin_tone_distribution(
    skin_tone_df: pd.DataFrame,
    title: str = "Skin-Tone Class Distribution",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Plot skin-tone representation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

    palette = plt.cm.Set2(np.linspace(0, 1, max(1, len(skin_tone_df))))
    bars = ax.bar(
        skin_tone_df["skin_tone_class"],
        skin_tone_df["image_count"],
        color=palette,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Skin-Tone Class", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Image Count", fontsize=11)
    ax.tick_params(axis="x", rotation=30)

    for bar, pct in zip(bars, skin_tone_df["percentage"]):
        h = bar.get_height()
        ax.annotate(
            f"{int(h)} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_images_per_lesion_distribution(
    images_per_lesion_summary: Dict[str, Any],
    title: str = "Images per Lesion Distribution",
    figsize: Tuple[int, int] = (8, 5),
) -> plt.Figure:
    """Generate bar chart of images-per-lesion groupings."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

    dist = images_per_lesion_summary.get("distribution_summary", {})
    categories = list(dist.keys())
    counts = list(dist.values())
    labels = [c.replace("_", " ").title() for c in categories]

    bars = ax.bar(labels, counts, color="steelblue", edgecolor="black", linewidth=0.8, zorder=3)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Unique Lesions", fontsize=11)
    ax.set_xlabel("Lesion Cluster Size", fontsize=11)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(
            str(int(h)),
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# Complete EDA Pipeline Execution
# ------------------------------------------------------------------------------

def run_eda_pipeline(
    data_root: Union[str, Path] = "data",
    output_dir: Union[str, Path] = "data/processed/eda",
) -> Dict[str, Any]:
    """Execute complete end-to-end EDA pipeline and save all CSVs and plots."""
    out_path = Path(output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    tables = load_dataset_tables(data_root)
    if not tables:
        raise FileNotFoundError(f"No valid dataset tables found in {data_root}")

    # 1. Dataset Overview
    image_dir = Path(data_root) / "images"
    overview = get_dataset_overview(tables, image_dir=image_dir if image_dir.exists() else None)

    # Missingness Report CSV
    missing_records = []
    for table_name, col_dict in overview.get("missing_values_per_column", {}).items():
        total_rows = overview["row_counts"].get(table_name, 0)
        for col_name, null_cnt in col_dict.items():
            missing_records.append({
                "table": table_name,
                "column": col_name,
                "missing_count": null_cnt,
                "total_rows": total_rows,
                "missing_percentage": round(null_cnt / max(1, total_rows) * 100, 2),
            })
    pd.DataFrame(missing_records).to_csv(out_path / "missingness_report.csv", index=False)

    # 2. Class Distribution
    gt_df = tables.get("training_gt")
    class_summary_df = pd.DataFrame()
    imbalance_metrics: Dict[str, Any] = {}
    if gt_df is not None:
        class_summary_df, imbalance_metrics = get_class_distribution(gt_df, class_columns=DIAGNOSTIC_CLASSES_11)
        class_summary_df.to_csv(out_path / "class_distribution.csv", index=False)

        # Plots
        fig1 = plot_class_distribution(class_summary_df, log_scale=False)
        fig1.savefig(out_path / "class_count_bar.png", dpi=200)
        plt.close(fig1)

        fig1_log = plot_class_distribution(class_summary_df, log_scale=True)
        fig1_log.savefig(out_path / "class_log_count_bar.png", dpi=200)
        plt.close(fig1_log)

    # 3. Skin-Tone Distribution
    input_df = tables.get("training_input")
    skin_tone_df = pd.DataFrame()
    if input_df is not None:
        skin_tone_df = get_skin_tone_distribution(input_df, skin_tone_col="skin_tone_class")
        if not skin_tone_df.empty:
            skin_tone_df.to_csv(out_path / "skin_tone_distribution.csv", index=False)
            fig_st = plot_skin_tone_distribution(skin_tone_df)
            fig_st.savefig(out_path / "skin_tone_distribution.png", dpi=200)
            plt.close(fig_st)

    # 4. Demographic Distributions
    demo_dfs = {}
    if input_df is not None:
        demo_dfs = get_demographic_distributions(input_df)
        if "age_approx" in demo_dfs:
            demo_dfs["age_approx"].to_csv(out_path / "age_distribution.csv", index=False)
        if "sex" in demo_dfs:
            demo_dfs["sex"].to_csv(out_path / "sex_distribution.csv", index=False)
        if "site" in demo_dfs:
            demo_dfs["site"].to_csv(out_path / "anatomical_site_distribution.csv", index=False)
        elif "anatom_site_general" in demo_dfs:
            demo_dfs["anatom_site_general"].to_csv(out_path / "anatomical_site_distribution.csv", index=False)

    # 5. Cross-Tabulations
    # Merge input with diagnostic class
    if input_df is not None and gt_df is not None:
        # Determine dominant diagnostic class per lesion
        avail_classes = [c for c in DIAGNOSTIC_CLASSES_11 if c in gt_df.columns]
        gt_with_class = gt_df.copy()
        if avail_classes:
            gt_with_class["diagnostic_class"] = gt_with_class[avail_classes].idxmax(axis=1)
            # If all zeros, label as unclassified
            all_zero = gt_with_class[avail_classes].sum(axis=1) == 0
            gt_with_class.loc[all_zero, "diagnostic_class"] = "Unclassified"

        merged = pd.merge(input_df, gt_with_class[["lesion_id", "diagnostic_class"]], on="lesion_id", how="left")
        crosstabs = generate_cross_tabulations(merged)

        for ct_name, ct_df in crosstabs.items():
            ct_df.to_csv(out_path / f"{ct_name}.csv")

    # 6. Images per Lesion Plot
    if "images_per_lesion" in overview:
        fig_ipl = plot_images_per_lesion_distribution(overview["images_per_lesion"])
        fig_ipl.savefig(out_path / "images_per_lesion_distribution.png", dpi=200)
        plt.close(fig_ipl)

    # 7. Comprehensive Summary JSON
    summary = {
        "dataset_overview": overview,
        "class_imbalance": imbalance_metrics,
        "class_distribution": class_summary_df.to_dict(orient="records") if not class_summary_df.empty else [],
        "skin_tone_distribution": skin_tone_df.to_dict(orient="records") if not skin_tone_df.empty else [],
        "artifacts_generated": [p.name for p in out_path.iterdir() if p.is_file()],
    }
    with open(out_path / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"EDA pipeline completed. Artifacts saved to: {out_path}")
    return summary


def main() -> None:
    """CLI runner for EDA pipeline."""
    parser = argparse.ArgumentParser(description="SkinCancerTracker Reproducible EDA CLI")
    parser.add_argument("--data-root", type=str, default="data", help="Dataset directory containing CSVs")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Configuration YAML path")
    parser.add_argument("--output-dir", type=str, default="data/processed/eda", help="Output directory for EDA tables and plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(f"Running EDA Pipeline on data root: {args.data_root}")
    summary = run_eda_pipeline(data_root=args.data_root, output_dir=args.output_dir)
    print(f"EDA Pipeline finished successfully. Generated {len(summary['artifacts_generated'])} artifacts under {args.output_dir}:")
    for f in summary["artifacts_generated"]:
        print(f"  • {f}")


if __name__ == "__main__":
    main()

"""Lesion-Level Data Splitting Pipeline for SkinCancerTracker.

Implements leakage-free partitioning using GroupShuffleSplit with lesion_id
as the grouping variable, guaranteeing zero lesion overlap between train,
validation, and test partitions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import yaml

logger = logging.getLogger("splitting")

VALID_SPLIT_NAMES = {"train", "validation", "test"}


class DataLeakageError(Exception):
    """Raised when lesion overlap or partition leakage is detected."""
    pass


def create_lesion_level_splits(
    data_root: Union[str, Path] = "data",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    input_csv: str = "training_input.csv",
    gt_csv: str = "training_gt.csv",
    metadata_csv: str = "metadata.csv",
    supp_csv: str = "training_supp.csv",
    group_col: str = "lesion_id",
    primary_id_col: str = "isic_id",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute two-stage GroupShuffleSplit ensuring complete lesion isolation.

    Args:
        data_root: Base dataset directory.
        train_ratio: Proportion of unique lesions in training partition.
        val_ratio: Proportion of unique lesions in validation partition.
        test_ratio: Proportion of unique lesions in test partition.
        seed: Random state for deterministic reproducibility.
        group_col: Column name defining patient lesion clusters (strictly lesion_id).
        primary_id_col: Unique image record key (isic_id).

    Returns:
        Tuple of (split_dataframe, split_statistics_dict).
    """
    root = Path(data_root).resolve()

    # Verify ratio sum
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-5):
        raise ValueError(f"Split ratios must sum to 1.0, got: {total_ratio}")

    # Step 1: Load image-level records
    input_path = root / input_csv if (root / input_csv).exists() else root / "supplements" / input_csv
    if not input_path.exists():
        raise FileNotFoundError(f"Input records file not found at: {input_path}")
    df_input = pd.read_csv(input_path, low_memory=False)

    # Step 2: Attach diagnostic labels from training_gt.csv
    gt_path = root / gt_csv if (root / gt_csv).exists() else root / "supplements" / gt_csv
    df_gt = pd.read_csv(gt_path, low_memory=False) if gt_path.exists() else None

    merged_df = df_input.copy()

    if df_gt is not None and group_col in df_gt.columns:
        # Determine dominant diagnostic class for stratification reference
        diagnostic_cols = [
            c for c in ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]
            if c in df_gt.columns
        ]
        if diagnostic_cols:
            gt_labels = df_gt[[group_col] + diagnostic_cols].copy()
            gt_labels["diagnosis"] = gt_labels[diagnostic_cols].idxmax(axis=1)
            # Merge diagnosis into input records
            merged_df = pd.merge(merged_df, gt_labels[[group_col, "diagnosis"]], on=group_col, how="left")

    # Step 3: Attach supplementary diagnosis or metadata if available
    supp_path = root / supp_csv if (root / supp_csv).exists() else root / "supplements" / supp_csv
    if supp_path.exists():
        df_supp = pd.read_csv(supp_path, low_memory=False)
        if primary_id_col in df_supp.columns and "diagnosis_full" in df_supp.columns:
            merged_df = pd.merge(
                merged_df,
                df_supp[[primary_id_col, "diagnosis_full"]],
                on=primary_id_col,
                how="left",
            )

    # Step 4: Extract grouping variable
    if group_col not in merged_df.columns:
        raise KeyError(f"Grouping column '{group_col}' missing from records.")

    groups = merged_df[group_col].values

    # Step 5: Perform Two-Stage GroupShuffleSplit
    # Stage 1: Split into Train+Val vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_val_idx, test_idx = next(gss_test.split(merged_df, groups=groups))

    df_train_val = merged_df.iloc[train_val_idx].copy()
    train_val_groups = df_train_val[group_col].values

    # Stage 2: Split Train+Val into Train vs Validation
    val_rel_ratio = val_ratio / (train_ratio + val_ratio)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_ratio, random_state=seed)
    train_sub_idx, val_sub_idx = next(gss_val.split(df_train_val, groups=train_val_groups))

    # Map back to original dataframe indices
    train_idx = train_val_idx[train_sub_idx]
    val_idx = train_val_idx[val_sub_idx]

    # Step 6: Assign each image record to exactly one split
    merged_df["split"] = "unassigned"
    merged_df.loc[train_idx, "split"] = "train"
    merged_df.loc[val_idx, "split"] = "validation"
    merged_df.loc[test_idx, "split"] = "test"

    # Step 7: Rigorous Lesion-Level Isolation Verification
    verify_split_leakage(merged_df, group_col=group_col, id_col=primary_id_col)

    # Compile statistics
    stats = compute_split_statistics(merged_df, group_col=group_col, id_col=primary_id_col)

    # Reorder columns to ensure required columns are prominent
    priority_cols = [group_col, primary_id_col, "split"]
    optional_cols = ["diagnosis", "diagnosis_full", "skin_tone_class", "image_type", "age_approx", "sex", "image_manipulation"]
    final_cols = [c for c in priority_cols if c in merged_df.columns] + [
        c for c in optional_cols if c in merged_df.columns and c not in priority_cols
    ]
    # Add any remaining columns
    remaining = [c for c in merged_df.columns if c not in final_cols]
    final_df = merged_df[final_cols + remaining].copy()

    return final_df, stats


def verify_split_leakage(
    df: pd.DataFrame,
    group_col: str = "lesion_id",
    id_col: str = "isic_id",
) -> None:
    """Verify complete isolation between train, validation, and test splits.

    Raises:
        DataLeakageError: If any lesion appears in more than one partition,
                          or if an image is unassigned, duplicated, or missing.
    """
    # Check 1: Valid split values
    actual_splits = set(df["split"].unique())
    invalid_splits = actual_splits - VALID_SPLIT_NAMES
    if invalid_splits:
        raise DataLeakageError(f"Found invalid split values: {invalid_splits}. Allowed: {VALID_SPLIT_NAMES}")

    # Check 2: Image-level integrity
    if df[id_col].isna().any():
        raise DataLeakageError(f"Found null values in primary ID column '{id_col}'.")
    if df[id_col].duplicated().any():
        dupes = df[df[id_col].duplicated()][id_col].tolist()[:5]
        raise DataLeakageError(f"Found duplicate records in ID column '{id_col}': {dupes}")

    # Check 3: Extract lesion sets
    train_lesions: Set[Any] = set(df.loc[df["split"] == "train", group_col].dropna().unique())
    val_lesions: Set[Any] = set(df.loc[df["split"] == "validation", group_col].dropna().unique())
    test_lesions: Set[Any] = set(df.loc[df["split"] == "test", group_col].dropna().unique())

    # Check 4: Strict Disjointness
    leakage_train_val = train_lesions.intersection(val_lesions)
    leakage_train_test = train_lesions.intersection(test_lesions)
    leakage_val_test = val_lesions.intersection(test_lesions)

    errors = []
    if leakage_train_val:
        errors.append(f"Leakage between TRAIN and VALIDATION: {len(leakage_train_val)} shared lesions (samples: {list(leakage_train_val)[:3]})")
    if leakage_train_test:
        errors.append(f"Leakage between TRAIN and TEST: {len(leakage_train_test)} shared lesions (samples: {list(leakage_train_test)[:3]})")
    if leakage_val_test:
        errors.append(f"Leakage between VALIDATION and TEST: {len(leakage_val_test)} shared lesions (samples: {list(leakage_val_test)[:3]})")

    if errors:
        raise DataLeakageError("Lesion data leakage detected:\n  - " + "\n  - ".join(errors))


def compute_split_statistics(
    df: pd.DataFrame,
    group_col: str = "lesion_id",
    id_col: str = "isic_id",
) -> Dict[str, Any]:
    """Calculate summary metrics for images and lesions across splits."""
    total_images = len(df)
    total_lesions = df[group_col].nunique()

    stats: Dict[str, Any] = {
        "total_images": total_images,
        "total_lesions": total_lesions,
        "partitions": {},
    }

    for split_name in ["train", "validation", "test"]:
        sub = df[df["split"] == split_name]
        n_imgs = len(sub)
        n_les = sub[group_col].nunique()
        stats["partitions"][split_name] = {
            "image_count": n_imgs,
            "image_pct": round(n_imgs / max(1, total_images) * 100, 2),
            "lesion_count": n_les,
            "lesion_pct": round(n_les / max(1, total_lesions) * 100, 2),
        }

    return stats


def save_splits(
    splits_df: pd.DataFrame,
    output_path: Union[str, Path] = "data/splits/lesion_level_splits.csv",
    stats: Optional[Dict[str, Any]] = None,
) -> Path:
    """Serialize the split DataFrame and statistics summary to disk."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(out, index=False)
    logger.info(f"Saved lesion-level splits ({len(splits_df)} rows) to: {out}")

    if stats is not None:
        stats_path = out.parent / "split_summary.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    return out


def main() -> None:
    """CLI runner for lesion-level data splitting."""
    parser = argparse.ArgumentParser(description="SkinCancerTracker Lesion-Level Data Splitting CLI")
    parser.add_argument("--data-root", type=str, default="data", help="Directory containing dataset CSVs")
    parser.add_argument("--config", type=str, default="configs/data.yaml", help="Configuration YAML path")
    parser.add_argument("--output-dir", type=str, default="data/splits", help="Output directory for split CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    out_file = Path(args.output_dir) / "lesion_level_splits.csv"
    print(f"Executing Lesion-Level GroupShuffleSplit (Seed: {args.seed})...")

    splits_df, stats = create_lesion_level_splits(
        data_root=args.data_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_splits(splits_df, output_path=out_file, stats=stats)

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY (Zero Lesion Leakage Verified)")
    print("=" * 60)
    for p_name, p_meta in stats["partitions"].items():
        print(f"  • {p_name.upper():12}: {p_meta['lesion_count']:5} lesions ({p_meta['lesion_pct']:5.2f}%) | {p_meta['image_count']:6} images ({p_meta['image_pct']:5.2f}%)")
    print(f"  • TOTAL       : {stats['total_lesions']:5} lesions | {stats['total_images']:6} images")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

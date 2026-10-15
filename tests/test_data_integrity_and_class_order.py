"""Regression test asserting dataset partition integrity, sum of supports, and consistent class order."""

import json
from pathlib import Path
import pandas as pd
import pytest
import torch

from src.eval.metrics import DEFAULT_CLASSES as METRICS_DEFAULT_CLASSES
from src.eval.predictions import DEFAULT_CLASSES as PREDICTIONS_DEFAULT_CLASSES
from src.losses.hierarchical import DEFAULT_CLASSES as HIERARCHICAL_DEFAULT_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_canonical_label_mapping_order():
    """Verify that configs/label_mapping.json is consistent, sorted, and contiguous."""
    map_path = PROJECT_ROOT / "configs" / "label_mapping.json"
    assert map_path.is_file(), f"label_mapping.json not found at {map_path}"

    with open(map_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    classes = mapping["classes"]
    class_to_idx = mapping["class_to_idx"]
    idx_to_class = mapping["idx_to_class"]

    assert len(classes) == 11
    # Verify canonical alphabetical order
    assert classes == sorted(classes), "Classes list in label_mapping.json must be alphabetically sorted"

    for idx, cls_name in enumerate(classes):
        assert class_to_idx[cls_name] == idx
        assert idx_to_class[str(idx)] == cls_name

    # Assert all repository constants match this exact ordering
    assert METRICS_DEFAULT_CLASSES == classes, "DEFAULT_CLASSES in src/eval/metrics.py does not match label_mapping.json"
    assert PREDICTIONS_DEFAULT_CLASSES == classes, "DEFAULT_CLASSES in src/eval/predictions.py does not match label_mapping.json"
    assert HIERARCHICAL_DEFAULT_CLASSES == classes, "DEFAULT_CLASSES in src/losses/hierarchical.py does not match label_mapping.json"


def test_lesion_level_splits_support_sum():
    """Verify train + val + test == total for every class, and no leakages."""
    splits_path = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"
    assert splits_path.is_file(), f"Splits file not found at {splits_path}"

    df = pd.read_csv(splits_path)
    # Check dermoscopic modality
    df_dermo = df[df["image_type"] == "dermoscopic"].copy()
    assert len(df_dermo) == 5240, f"Expected 5,240 dermoscopic lesions, got {len(df_dermo)}"

    train_df = df_dermo[df_dermo["split"] == "train"]
    val_df = df_dermo[df_dermo["split"].isin(["val", "validation"])]
    test_df = df_dermo[df_dermo["split"] == "test"]

    assert len(train_df) == 3668
    assert len(val_df) == 786
    assert len(test_df) == 786
    assert len(train_df) + len(val_df) + len(test_df) == 5240

    # Zero lesion leakage
    train_lesions = set(train_df["lesion_id"])
    val_lesions = set(val_df["lesion_id"])
    test_lesions = set(test_df["lesion_id"])

    assert train_lesions.isdisjoint(val_lesions), "Lesion leakage detected between train and val!"
    assert train_lesions.isdisjoint(test_lesions), "Lesion leakage detected between train and test!"
    assert val_lesions.isdisjoint(test_lesions), "Lesion leakage detected between val and test!"

    # Verify per-class counts: train + val + test == total, and val + test <= total
    with open(PROJECT_ROOT / "configs" / "label_mapping.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    expected_test_counts = {
        "AKIEC": 54,
        "BCC": 391,
        "BEN_OTH": 8,
        "BKL": 80,
        "DF": 5,
        "INF": 5,
        "MAL_OTH": 2,
        "MEL": 67,
        "NV": 99,
        "SCCKA": 69,
        "VASC": 6,
    }

    expected_val_counts = {
        "AKIEC": 47,
        "BCC": 374,
        "BEN_OTH": 9,
        "BKL": 79,
        "DF": 8,
        "INF": 7,
        "MAL_OTH": 1,
        "MEL": 58,
        "NV": 126,
        "SCCKA": 72,
        "VASC": 5,
    }

    for cls in classes:
        n_tr = int((train_df["diagnosis"] == cls).sum())
        n_va = int((val_df["diagnosis"] == cls).sum())
        n_te = int((test_df["diagnosis"] == cls).sum())
        n_tot = int((df_dermo["diagnosis"] == cls).sum())

        assert n_tr + n_va + n_te == n_tot, f"Sum mismatch for {cls}: {n_tr}+{n_va}+{n_te} != {n_tot}"
        assert n_va + n_te <= n_tot, f"Val+Test exceeds total for {cls}: {n_va+n_te} > {n_tot}"
        assert n_te == expected_test_counts[cls], f"Test count for {cls} changed: {n_te} != {expected_test_counts[cls]}"
        assert n_va == expected_val_counts[cls], f"Val count for {cls} changed: {n_va} != {expected_val_counts[cls]}"


def test_prediction_files_alignment():
    """Verify that predictions_val.csv and predictions_test.csv match ground truth and checkpoint."""
    with open(PROJECT_ROOT / "configs" / "label_mapping.json", "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    # Test predictions
    preds_test_path = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_test.csv"
    if preds_test_path.is_file():
        df_test = pd.read_csv(preds_test_path)
        assert len(df_test) == 786
        for idx, cls in enumerate(classes):
            # Check true label matches class name
            cls_mask = df_test["true_class"] == cls
            lbl_mask = df_test["true_label"] == idx
            assert (cls_mask == lbl_mask).all(), f"Mismatch between true_class and true_label for {cls}"

    # Val predictions and checkpoint alignment
    preds_val_path = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_val.csv"
    ckpt_path = PROJECT_ROOT / "checkpoints" / "exp2_panderm_v2" / "best_model.pt"

    if preds_val_path.is_file() and ckpt_path.is_file():
        df_val = pd.read_csv(preds_val_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt_val_dist = ckpt.get("metrics", {}).get("val_prediction_distribution", {})

        for idx, cls in enumerate(classes):
            csv_pred_count = int((df_val["pred_label"] == idx).sum())
            ckpt_pred_count = ckpt_val_dist.get(cls, 0)
            assert csv_pred_count == ckpt_pred_count, (
                f"Validation prediction count mismatch for {cls}: "
                f"predictions_val.csv={csv_pred_count} vs best_model.pt={ckpt_pred_count}"
            )

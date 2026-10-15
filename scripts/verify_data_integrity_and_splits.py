"""Complete verification of dataset partitions, per-class supports, and prediction distributions."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 1. Load canonical label mapping
with open(PROJECT_ROOT / "configs" / "label_mapping.json", "r", encoding="utf-8") as f:
    label_map = json.load(f)

classes = label_map["classes"]
class_to_idx = label_map["class_to_idx"]
idx_to_class = {int(k): v for k, v in label_map["idx_to_class"].items()}

print("=" * 80)
print("1. CANONICAL CLASS MAPPING ORDER (configs/label_mapping.json)")
print("=" * 80)
for idx, name in enumerate(classes):
    print(f"  Index {idx:2d}: {name:<10} (class_to_idx={class_to_idx[name]})")

# 2. Load splits CSV
splits_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv")
print("\n" + "=" * 80)
print("2. DATASET SPLITS INTEGRITY (lesion_level_splits.csv)")
print("=" * 80)
print(f"Total rows in CSV: {len(splits_df)} (dermoscopic + clinical close-up)")

# Filter to dermoscopic modality (1 image per lesion, exactly 5,240 lesions)
dermo_df = splits_df[splits_df["image_type"] == "dermoscopic"].copy()
print(f"Dermoscopic lesions: {len(dermo_df)}")
split_counts = dermo_df["split"].value_counts().to_dict()
print(f"Split counts (dermoscopic): {split_counts}")
assert split_counts["train"] == 3668, f"Expected 3668 train, got {split_counts['train']}"
assert split_counts["validation"] == 786, f"Expected 786 val, got {split_counts['validation']}"
assert split_counts["test"] == 786, f"Expected 786 test, got {split_counts['test']}"

# Compute true counts per split using canonical class order
train_df = dermo_df[dermo_df["split"] == "train"]
val_df = dermo_df[dermo_df["split"].isin(["val", "validation"])]
test_df = dermo_df[dermo_df["split"] == "test"]

print("\n" + "=" * 80)
print("3. PER-CLASS TRUE SUPPORTS ACROSS PARTITIONS (DERMOSCOPIC MODALITY, N=5,240)")
print("=" * 80)
print(f"{'Idx':<4} {'Class':<10} {'Train':<8} {'Val':<8} {'Test':<8} {'Sum(T+V+T)':<12} {'Total':<8} {'Val+Test <= Total?':<20}")
print("-" * 80)

for idx, cls in enumerate(classes):
    n_train = int((train_df["diagnosis"] == cls).sum())
    n_val = int((val_df["diagnosis"] == cls).sum())
    n_test = int((test_df["diagnosis"] == cls).sum())
    n_sum = n_train + n_val + n_test
    n_total = int((dermo_df["diagnosis"] == cls).sum())

    assert n_sum == n_total, f"Sum mismatch for {cls}: {n_sum} != {n_total}"
    assert (n_val + n_test) <= n_total, f"Val+Test exceeds total for {cls}: {n_val+n_test} > {n_total}"

    valid_check = "PASS" if (n_val + n_test <= n_total) else "FAIL"
    print(f"{idx:<4} {cls:<10} {n_train:<8} {n_val:<8} {n_test:<8} {n_sum:<12} {n_total:<8} {valid_check:<20}")

print("-" * 80)
print(f"{'ALL':<4} {'TOTAL':<10} {len(train_df):<8} {len(val_df):<8} {len(test_df):<8} {len(dermo_df):<12} {len(dermo_df):<8} PASS")

# 4. Compare Predictions Against Checkpoint
print("\n" + "=" * 80)
print("4. PREDICTION DISTRIBUTIONS & CHECKPOINT COMPARISON (Exp-2 v2)")
print("=" * 80)

ckpt_path = PROJECT_ROOT / "checkpoints" / "exp2_panderm_v2" / "best_model.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt_metrics = ckpt.get("metrics", {})
ckpt_val_dist = ckpt_metrics.get("val_prediction_distribution", {})
print(f"Checkpoint Best Epoch: {ckpt.get('epoch')}")
print(f"Checkpoint val_macro_f1: {ckpt_metrics.get('val_macro_f1'):.4f}")
print(f"Checkpoint val_prediction_distribution stored in best_model.pt:\n  {ckpt_val_dist}")

preds_val_path = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_val.csv"
preds_test_path = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions_test.csv"
preds_orig_path = PROJECT_ROOT / "results" / "exp2_panderm_v2" / "predictions.csv"

df_p_val = pd.read_csv(preds_val_path)
df_p_test = pd.read_csv(preds_test_path)
df_p_orig = pd.read_csv(preds_orig_path)

val_pred_counts_from_csv = {cls: int((df_p_val["pred_label"] == i).sum()) for i, cls in enumerate(classes)}
test_pred_counts_from_csv = {cls: int((df_p_test["pred_label"] == i).sum()) for i, cls in enumerate(classes)}
orig_pred_counts_from_csv = {cls: int((df_p_orig["pred_label"] == i).sum()) for i, cls in enumerate(classes)}

print("\nValidation Preds comparison:")
print(f"{'Idx':<4} {'Class':<10} {'Val Pred CSV':<14} {'Checkpoint Dist':<16} {'Exact Match?':<12}")
print("-" * 60)
for idx, cls in enumerate(classes):
    c_val = val_pred_counts_from_csv[cls]
    c_ckpt = ckpt_val_dist.get(cls, 0)
    match_str = "MATCH" if c_val == c_ckpt else "MISMATCH"
    assert c_val == c_ckpt, f"Mismatch for {cls}: CSV {c_val} != Ckpt {c_ckpt}"
    print(f"{idx:<4} {cls:<10} {c_val:<14} {c_ckpt:<16} {match_str:<12}")

print("\nTest Preds comparison (predictions_test.csv vs predictions.csv):")
print(f"{'Idx':<4} {'Class':<10} {'Test True':<10} {'Test Pred':<10} {'Orig Pred':<10} {'Match?':<10}")
print("-" * 60)
for idx, cls in enumerate(classes):
    t_true = int((test_df["diagnosis"] == cls).sum())
    p_test = test_pred_counts_from_csv[cls]
    p_orig = orig_pred_counts_from_csv[cls]
    assert p_test == p_orig, f"Test pred mismatch between CSVs for {cls}: {p_test} != {p_orig}"
    print(f"{idx:<4} {cls:<10} {t_true:<10} {p_test:<10} {p_orig:<10} MATCH")

print("\n[ALL INTEGRITY CHECKS PASSED PERFECTLY]")

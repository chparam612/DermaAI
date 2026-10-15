"""
Generate complete validation and test predictions with raw logits and probabilities
for the fine-tuned PanDerm v2 checkpoint (checkpoints/exp2_panderm_v2/best_model.pt).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import load_label_mapping
from src.data.loaders import create_dataloaders
from src.eval.predictions import generate_predictions, save_predictions
from src.models.panderm import PanDermClassifier
from src.train.checkpointing import CheckpointManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("generate_exp2_predictions")


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    label_map_path = PROJECT_ROOT / "configs" / "label_mapping.json"
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map_data = json.load(f)
    class_names = label_map_data["classes"]
    num_classes = len(class_names)

    ckpt_path = PROJECT_ROOT / "checkpoints" / "exp2_panderm_v2" / "best_model.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    logger.info(f"Initializing PanDermClassifier (num_classes={num_classes})...")
    model = PanDermClassifier(
        architecture="panderm_base",
        num_classes=num_classes,
        checkpoint_path=None,
    )
    CheckpointManager.load(str(ckpt_path), model)
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully in {time.time() - t0:.2f}s")

    splits_file = PROJECT_ROOT / "data" / "splits" / "lesion_level_splits.csv"
    image_dir = PROJECT_ROOT / "data" / "images"

    logger.info("Creating DataLoaders (batch_size=32, num_workers=0, dermoscopic only)...")
    _, val_loader, test_loader = create_dataloaders(
        metadata_path=splits_file,
        image_dir=image_dir,
        pipeline_type="panderm",
        batch_size=32,
        eval_batch_size=32,
        num_workers=0,
        image_size=224,
        label_mapping=label_map_data["class_to_idx"],
        modality="dermoscopic",
    )
    logger.info(f"Validation batches: {len(val_loader)} (N={len(val_loader.dataset)})")
    logger.info(f"Test batches: {len(test_loader)} (N={len(test_loader.dataset)})")

    results_dir = PROJECT_ROOT / "results" / "exp2_panderm_v2"
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validation Split Inference
    logger.info("Generating predictions on VALIDATION split...")
    t_val = time.time()
    df_val = generate_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
    )
    val_path = results_dir / "predictions_val.csv"
    save_predictions(df_val, val_path)
    val_acc = accuracy_score(df_val["true_label"], df_val["pred_label"])
    val_bal_acc = balanced_accuracy_score(df_val["true_label"], df_val["pred_label"])
    val_macro_f1 = f1_score(df_val["true_label"], df_val["pred_label"], average="macro", zero_division=0)
    logger.info(f"Validation inference took {time.time() - t_val:.2f}s")
    logger.info(f"Validation Acc: {val_acc:.4f} | Bal Acc: {val_bal_acc:.4f} | Macro-F1: {val_macro_f1:.4f}")

    # 2. Test Split Inference
    logger.info("Generating predictions on TEST split...")
    t_test = time.time()
    df_test = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
    )
    test_path = results_dir / "predictions_test.csv"
    save_predictions(df_test, test_path)
    test_acc = accuracy_score(df_test["true_label"], df_test["pred_label"])
    test_bal_acc = balanced_accuracy_score(df_test["true_label"], df_test["pred_label"])
    test_macro_f1 = f1_score(df_test["true_label"], df_test["pred_label"], average="macro", zero_division=0)
    logger.info(f"Test inference took {time.time() - t_test:.2f}s")
    logger.info(f"Test Acc: {test_acc:.4f} | Bal Acc: {test_bal_acc:.4f} | Macro-F1: {test_macro_f1:.4f}")

    # Check alignment with existing test predictions
    orig_test_path = results_dir / "predictions.csv"
    if orig_test_path.is_file():
        df_orig = pd.read_csv(orig_test_path)
        orig_acc = accuracy_score(df_orig["true_label"], df_orig["pred_label"])
        orig_f1 = f1_score(df_orig["true_label"], df_orig["pred_label"], average="macro", zero_division=0)
        logger.info(f"Original Test Acc: {orig_acc:.4f} | Original Test Macro-F1: {orig_f1:.4f}")
        assert np.isclose(test_acc, orig_acc, atol=1e-3), f"Test accuracy mismatch: {test_acc} vs {orig_acc}"
        assert np.isclose(test_macro_f1, orig_f1, atol=1e-3), f"Test Macro-F1 mismatch: {test_macro_f1} vs {orig_f1}"
        logger.info("[VERIFIED] Newly generated test predictions perfectly match original Exp-2 test predictions!")

    logger.info(f"Total time elapsed: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()

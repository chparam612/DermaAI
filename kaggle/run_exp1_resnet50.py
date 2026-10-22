"""
Exp-1 Kaggle Execution Script: ResNet-50 ImageNet-Pretrained Baseline (30 Epochs).

Workflow compliance:
- Reads the split file, computes SHA-256 hash, and strictly verifies it matches the repo hash.
- Config-driven (configs/experiments/exp1_resnet50_kaggle.yaml).
- Evaluates best checkpoint on BOTH validation and test splits.
- Saves predictions CSVs (with per-class LOGITS and PROBABILITIES for val AND test).
- Saves metrics JSON, summary metrics JSON, training curves JSON, and best_model.pt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import datetime
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import load_label_mapping
from src.data.loaders import create_dataloaders
from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import generate_predictions, save_predictions
from src.models.factory import build_model
from src.train.checkpointing import CheckpointManager
from src.losses import get_loss_function
from src.losses.classification import compute_class_weights
from src.train.engine import train_one_epoch, validate
from src.eval.wandb_logger import WandbLogger

EXPECTED_SPLIT_HASH = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"
logger = logging.getLogger("kaggle_exp1_resnet50")


def verify_split_integrity(splits_path: Path) -> str:
    """Compute and verify SHA-256 hash of splits CSV against repository contract."""
    if not splits_path.is_file():
        raise FileNotFoundError(f"Splits file not found at: {splits_path}")
    hasher = hashlib.sha256()
    with open(splits_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    computed_hash = hasher.hexdigest()
    logger.info(f"Splits file SHA-256: {computed_hash}")
    if computed_hash != EXPECTED_SPLIT_HASH:
        raise ValueError(
            f"SPLIT INTEGRITY ERROR: Expected SHA-256 {EXPECTED_SPLIT_HASH}, got {computed_hash}"
        )
    logger.info("[VERIFIED] Split integrity confirmed: identical to repository partition.")
    return computed_hash


def parse_args():
    parser = argparse.ArgumentParser(description="Run Exp-1 ResNet-50 30-Epoch Baseline on Kaggle GPU.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_resnet50_kaggle.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--image-dir", type=str, default=None, help="Override image directory")
    parser.add_argument("--splits-file", type=str, default=None, help="Override splits file")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results output directory")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader workers")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--online", action="store_true", help="Enable online W&B logging")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking completely")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch smoke test")
    return parser.parse_args()


def run_exp1_kaggle(config: dict, smoke_test: bool = False, no_wandb: bool = False, online: bool = False, seed: int = 42):
    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "exp1-resnet50-baseline-kaggle")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = config.get("data", {})
    splits_file = Path(data_cfg["splits_file"]).resolve()
    image_dir = Path(data_cfg["image_dir"]).resolve()
    label_mapping_file = Path(data_cfg.get("label_mapping", "configs/label_mapping.json")).resolve()

    # 1. Verify split integrity
    verify_split_integrity(splits_file)

    # 2. Label mapping
    label_map = load_label_mapping(label_mapping_file)
    class_names = list(label_map.keys())
    num_classes = len(class_names)

    # 3. Directories (include seed suffix when seed != 42)
    out_cfg = config.get("output", {})
    suffix = f"_seed{seed}" if seed != 42 else ""
    ckpt_dir_default = f"/kaggle/working/checkpoints/exp1_resnet50{suffix}"
    res_dir_default = f"/kaggle/working/results/exp1_resnet50{suffix}"
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", ckpt_dir_default)).resolve()
    results_dir = Path(out_cfg.get("results_dir", res_dir_default)).resolve()
    if seed != 42 and out_cfg.get("checkpoint_dir") and not str(checkpoint_dir).endswith(suffix):
        checkpoint_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}{suffix}"
    if seed != 42 and out_cfg.get("results_dir") and not str(results_dir).endswith(suffix):
        results_dir = results_dir.parent / f"{results_dir.name}{suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 4. Device
    device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running Exp-1 on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith('cuda') else 'CPU'})")

    # 5. Dataloaders
    train_cfg = config.get("training", {})
    batch_size = int(train_cfg.get("batch_size", 64))
    dl_cfg = config.get("dataloader", {})
    num_workers = int(dl_cfg.get("num_workers", 4))

    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=splits_file,
        image_dir=image_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        modality=data_cfg.get("modality", "dermoscopic"),
        image_size=data_cfg.get("image_size", 224),
        label_mapping=label_map,
        use_weighted_sampler=dl_cfg.get("weighted_sampler", False),
    )

    # 6. Model
    model_cfg = config.get("model", {})
    model = build_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        dropout_rate=float(model_cfg.get("dropout_rate", 0.2)),
    ).to(device)

    # 7. Criterion
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "class_balanced_ce")
    weight = None
    if loss_name in ("class_balanced_ce", "weighted_ce"):
        train_labels = train_loader.dataset.get_labels()
        weight = compute_class_weights(train_labels, num_classes=num_classes).to(device)
        logger.info(f"Using class-balanced weights: {weight.tolist()}")

    criterion = get_loss_function(
        loss_name=loss_name,
        label_smoothing=float(loss_cfg.get("label_smoothing", 0.0)),
        weight=weight,
    )

    # 8. Optimizer & Scheduler
    opt_cfg = config.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1.0e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 1 if smoke_test else int(train_cfg.get("epochs", 30))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-6)

    # 9. Logger
    wandb_mode = "online" if online else ("offline" if not no_wandb else "disabled")
    wb_logger = WandbLogger(
        project="SkinCancerTracker",
        experiment_name=f"{exp_name}_seed{seed}",
        config=config,
        enabled=(not no_wandb),
        mode=wandb_mode,
    )

    # 10. Checkpoint Manager
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        monitor="val_macro_f1",
        mode="max",
        max_keep=int(out_cfg.get("max_keep", 2)),
    )

    # 11. Training loop with curve tracking
    training_curves = {
        "epochs": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_acc": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "lr": [],
    }

    start_time = time.time()
    logger.info(f"Starting Exp-1 ResNet-50 training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad=float(train_cfg.get("clip_grad", 1.0)),
            max_batches=2 if smoke_test else None,
            accumulation_steps=1,
        )

        val_m = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=2 if smoke_test else None,
            class_names=class_names,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        combined_m = {**train_m, **val_m, "lr": current_lr}
        wb_logger.log_metrics(combined_m, step=epoch)

        # Record curves
        training_curves["epochs"].append(epoch)
        training_curves["train_loss"].append(float(train_m["train_loss"]))
        training_curves["train_accuracy"].append(float(train_m["train_accuracy"]))
        training_curves["val_loss"].append(float(val_m["val_loss"]))
        training_curves["val_accuracy"].append(float(val_m["val_accuracy"]))
        training_curves["val_balanced_acc"].append(float(val_m["val_balanced_acc"]))
        training_curves["val_macro_f1"].append(float(val_m["val_macro_f1"]))
        training_curves["val_weighted_f1"].append(float(val_m["val_weighted_f1"]))
        training_curves["lr"].append(float(current_lr))

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_m['train_loss']:.4f} | "
            f"Val Loss: {val_m['val_loss']:.4f} | "
            f"Val Macro-F1: {val_m['val_macro_f1']:.4f} | "
            f"Val Bal Acc: {val_m['val_balanced_acc']:.4f}"
        )

        ckpt_mgr.save(model, optimizer, epoch, combined_m)

    total_training_time = time.time() - start_time
    runtime_info = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() and device.startswith("cuda") else "CPU",
        "total_training_time_seconds": round(total_training_time, 2),
        "total_training_time_formatted": str(datetime.timedelta(seconds=int(total_training_time))),
    }

    # Save training curves JSON
    curves_path = results_dir / "training_curves.json"
    with open(curves_path, "w", encoding="utf-8") as f:
        json.dump(training_curves, f, indent=2)
    logger.info(f"Saved training curves to {curves_path}")

    # Load best checkpoint for evaluation
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.is_file():
        ckpt_mgr.load(best_ckpt, model)
        logger.info(f"Loaded best checkpoint from {best_ckpt}")

    # 12. Evaluate on BOTH validation and test splits
    logger.info("Generating predictions and metrics on VALIDATION split...")
    df_val = generate_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
        max_batches=2 if smoke_test else None,
    )
    save_predictions(df_val, results_dir / "predictions_val.csv")

    logger.info("Generating predictions and metrics on TEST split...")
    df_test = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
        max_batches=2 if smoke_test else None,
    )
    save_predictions(df_test, results_dir / "predictions_test.csv")

    # Unified predictions.csv
    df_unified = pd.concat([df_val, df_test], ignore_index=True)
    save_predictions(df_unified, results_dir / "predictions.csv")

    # Metrics computation
    val_prob_cols = [f"prob_{c}" for c in class_names]
    val_metrics = calculate_classification_metrics(
        y_true=df_val["true_label"].values,
        y_pred=df_val["pred_label"].values,
        y_prob=df_val[val_prob_cols].values,
        class_names=class_names,
    )

    test_prob_cols = [f"prob_{c}" for c in class_names]
    test_metrics = calculate_classification_metrics(
        y_true=df_test["true_label"].values,
        y_pred=df_test["pred_label"].values,
        y_prob=df_test[test_prob_cols].values,
        class_names=class_names,
    )

    subgroup_metrics = calculate_subgroup_metrics(
        df_predictions=df_test,
        subgroup_col="skin_tone_class",
    )

    # Save metrics JSONs
    metrics_data = {
        "seed": seed,
        "split_sha256": EXPECTED_SPLIT_HASH,
        "val_overall": val_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    with open(results_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)

    summary_data = {
        "experiment_name": exp_name,
        "seed": seed,
        "runtime_info": runtime_info,
        "validation_overall": val_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    with open(results_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    wb_logger.finish()
    logger.info(f"[COMPLETED] Exp-1 Kaggle run (seed={seed}) finished in {runtime_info['total_training_time_formatted']}. Artifacts saved to {results_dir}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    args = parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.image_dir:
        config.setdefault("data", {})["image_dir"] = args.image_dir
    if args.splits_file:
        config.setdefault("data", {})["splits_file"] = args.splits_file
    if args.checkpoint_dir:
        config.setdefault("output", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        config.setdefault("output", {})["results_dir"] = args.results_dir
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr:
        config.setdefault("optimizer", {})["lr"] = args.lr
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.num_workers is not None:
        config.setdefault("dataloader", {})["num_workers"] = args.num_workers
    if args.device:
        config["device"] = args.device

    run_exp1_kaggle(config, smoke_test=args.smoke_test, no_wandb=args.no_wandb, online=args.online, seed=args.seed)


if __name__ == "__main__":
    main()

"""
Exp-2c Kaggle Execution Script: PanDerm ViT-Base Control Arm (Uniform Shuffling, 30 Epochs).

Workflow compliance:
- Reads the split file, computes SHA-256 hash, and strictly verifies it matches the repo hash.
- Config-driven (configs/experiments/exp2c_panderm_uniform_kaggle.yaml).
- Control arm: uses UNIFORM SHUFFLING (weighted_sampler: false) with Label Smoothing (0.1).
- Direct scientific ablation: Exp-2 v2 (weighted sampler) vs Exp-2c (uniform shuffle) vs Exp-3b (LA loss).
- Layer-wise LR decay (0.9), AdamW, cosine annealing with 5-epoch warmup.
- Evaluates best checkpoint on BOTH validation and test splits using RAW logits.
- Saves predictions CSVs (with per-class RAW LOGITS and PROBABILITIES for val AND test).
- Saves metrics JSON, summary metrics JSON, training curves JSON, and best_model.pt.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
import yaml

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import load_label_mapping
from src.data.loaders import create_dataloaders
from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import generate_predictions, save_predictions
from src.eval.wandb_logger import WandbLogger
from src.losses.classification import LabelSmoothingCrossEntropy
from src.models.panderm import PanDermClassifier
from src.train.checkpointing import CheckpointManager
from src.train.engine import train_one_epoch, validate
from src.utils.reproducibility import set_seed

EXPECTED_SPLIT_HASH = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"
logger = logging.getLogger("kaggle_exp2c_panderm_uniform")


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


def build_panderm_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    layer_decay: float = 0.9,
) -> torch.optim.Optimizer:
    """Build AdamW optimizer with layer-wise learning rate decay for Vision Transformer."""
    num_layers = 12
    scales = [layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)]

    def get_layer_id_for_vit(name: str) -> int:
        clean_name = name.replace("model.", "")
        if clean_name in ("cls_token", "mask_token", "pos_embed") or clean_name.startswith("patch_embed"):
            return 0
        elif clean_name.startswith("blocks."):
            return int(clean_name.split(".")[1]) + 1
        else:
            return num_layers + 1

    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lid = get_layer_id_for_vit(name)
        scale = scales[lid]
        no_decay = (len(param.shape) == 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name)
        wd = 0.0 if no_decay else weight_decay
        key = (lid, wd)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr * scale,
                "weight_decay": wd,
                "lr_scale": scale,
            }
        groups[key]["params"].append(param)

    optimizer = torch.optim.AdamW(list(groups.values()), eps=1e-8, betas=(0.9, 0.999))
    logger.info(f"Initialized AdamW with layer-wise LR decay (decay={layer_decay}, {len(groups)} parameter groups).")
    return optimizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Exp-2c PanDerm Control Arm (Uniform Shuffling) on Kaggle GPU.")
    parser.add_argument("--config", type=str, default="configs/experiments/exp2c_panderm_uniform_kaggle.yaml")
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--splits-file", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--online", action="store_true", help="Enable online W&B logging")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch verification test")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 1. Load configuration
    config_path = Path(args.config).resolve()
    logger.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Apply CLI overrides
    if args.image_dir:
        cfg.setdefault("data", {})["image_dir"] = args.image_dir
    if args.splits_file:
        cfg.setdefault("data", {})["splits_file"] = args.splits_file
    if args.checkpoint_path:
        cfg.setdefault("model", {})["checkpoint_path"] = args.checkpoint_path
    if args.checkpoint_dir:
        cfg.setdefault("output", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        cfg.setdefault("output", {})["results_dir"] = args.results_dir
    if args.epochs is not None:
        cfg.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("optimizer", {})["lr"] = args.lr

    smoke_test = args.smoke_test
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "exp2c-panderm-uniform")
    seed = args.seed if args.seed is not None else int(exp_cfg.get("seed", 42))
    set_seed(seed)

    data_cfg = cfg.get("data", {})
    splits_path = Path(data_cfg["splits_file"]).resolve()
    image_dir = Path(data_cfg["image_dir"]).resolve()
    label_map_path = Path(data_cfg.get("label_mapping", "configs/label_mapping.json")).resolve()

    train_cfg = cfg.get("training", {})
    epochs = 1 if smoke_test else int(train_cfg.get("epochs", 30))
    batch_size = 4 if smoke_test else int(train_cfg.get("batch_size", 64))
    num_workers = 0 if smoke_test else int(cfg.get("dataloader", {}).get("num_workers", 4))
    monitor = train_cfg.get("monitor", "val_macro_f1")

    opt_cfg = cfg.get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 2.5e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    layer_decay = float(opt_cfg.get("layer_decay", 0.9))
    warmup_epochs = int(opt_cfg.get("warmup_epochs", 5))

    loss_cfg = cfg.get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))

    out_cfg = cfg.get("output", {})
    suffix = f"_seed{seed}" if seed != 42 else ""
    default_ckpt = f"/kaggle/working/checkpoints/exp2c_panderm_uniform{suffix}"
    default_res = f"/kaggle/working/results/exp2c_panderm_uniform{suffix}"
    checkpoint_dir = Path(out_cfg.get("checkpoint_dir", default_ckpt)).resolve()
    results_dir = Path(out_cfg.get("results_dir", default_res)).resolve()
    if seed != 42 and out_cfg.get("checkpoint_dir") and not str(checkpoint_dir).endswith(suffix):
        checkpoint_dir = checkpoint_dir.parent / f"{checkpoint_dir.name}{suffix}"
    if seed != 42 and out_cfg.get("results_dir") and not str(results_dir).endswith(suffix):
        results_dir = results_dir.parent / f"{results_dir.name}{suffix}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Verify splits integrity
    verify_split_integrity(splits_path)

    # 4. Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    # 5. Label mapping
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    class_names = label_data["classes"]
    num_classes = len(class_names)
    class_to_idx = label_data["class_to_idx"]

    # 6. DataLoaders (UNIFORM SHUFFLING: weighted_sampler=False)
    logger.info("Instantiating DataLoaders (dermoscopic modality, weighted_sampler=False)...")
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=splits_path,
        image_dir=image_dir,
        pipeline_type="panderm",
        batch_size=batch_size,
        eval_batch_size=batch_size,
        num_workers=num_workers,
        image_size=int(data_cfg.get("image_size", 224)),
        label_mapping=class_to_idx,
        modality="dermoscopic",
        use_weighted_sampler=False,
    )

    # 7. Model Initialization
    model_cfg = cfg.get("model", {})
    ckpt_path = model_cfg.get("checkpoint_path", None)
    logger.info(f"Initializing PanDermClassifier (architecture='panderm_base', num_classes={num_classes})...")
    model = PanDermClassifier(
        architecture="panderm_base",
        num_classes=num_classes,
        checkpoint_path=ckpt_path,
    )
    model.to(device)

    # 8. Loss Function: LabelSmoothingCrossEntropy (matches Exp-2 v2 exactly)
    logger.info(f"Initializing LabelSmoothingCrossEntropy (label_smoothing={label_smoothing})...")
    criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing).to(device)

    # 9. Smoke Test Gate
    if smoke_test:
        logger.info("[SMOKE TEST] Running 1-batch verification...")
        batch = next(iter(train_loader))
        images, labels, _ = batch
        images = images.to(device)
        labels = labels.to(device)
        model.train()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        logger.info(f"[SMOKE TEST] Forward pass, loss calculation ({loss.item():.4f}), and backward pass succeeded.")
        print("[SMOKE TEST] Exp-2c smoke test passed", flush=True)
        return

    # 10. Optimizer & LR Scheduler
    optimizer = build_panderm_optimizer(model, lr=base_lr, weight_decay=weight_decay, layer_decay=layer_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(1e-6 / base_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 11. CheckpointManager & W&B Logger
    ckpt_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor,
        mode="max",
        max_keep=int(out_cfg.get("max_keep", 2)),
    )
    wandb_mode = "online" if args.online else "offline"
    wb_logger = WandbLogger(
        project=cfg.get("wandb", {}).get("project", "SkinCancerTracker"),
        experiment_name=f"{exp_name}_seed{seed}",
        config=cfg,
        enabled=True,
        mode=wandb_mode,
    )

    # 12. Training Loop
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
    logger.info(f"Starting Exp-2c training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad=float(train_cfg.get("clip_grad", 1.0)),
            max_batches=None,
            accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        )

        val_m = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=None,
            class_names=class_names,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        combined_m = {**train_m, **val_m, "lr": current_lr}
        wb_logger.log_metrics(combined_m, step=epoch)

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
            f"Val Bal Acc: {val_m['val_balanced_acc']:.4f} | "
            f"Val Macro-F1: {val_m['val_macro_f1']:.4f} | "
            f"Pred Classes: {val_m.get('val_num_predicted_classes', 0)}/11"
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
    with open(results_dir / "training_curves.json", "w", encoding="utf-8") as f:
        json.dump(training_curves, f, indent=2)

    # 13. Load best checkpoint for final evaluation
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.is_file():
        ckpt_mgr.load(best_ckpt, model)
        logger.info(f"Loaded best checkpoint from {best_ckpt}")

    # 14. Generate predictions on BOTH validation and test splits using RAW logits
    logger.info("Generating predictions and metrics on VALIDATION split...")
    df_val = generate_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
    )
    save_predictions(df_val, results_dir / "predictions_val.csv")

    logger.info("Generating predictions and metrics on TEST split...")
    df_test = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
    )
    save_predictions(df_test, results_dir / "predictions_test.csv")
    save_predictions(df_test, results_dir / "predictions.csv")

    # 15. Compute final metrics
    prob_cols = [f"prob_{c}" for c in class_names]
    val_metrics = calculate_classification_metrics(
        y_true=df_val["true_label"].values,
        y_pred=df_val["pred_label"].values,
        y_prob=df_val[prob_cols].values,
        class_names=class_names,
    )

    test_metrics = calculate_classification_metrics(
        y_true=df_test["true_label"].values,
        y_pred=df_test["pred_label"].values,
        y_prob=df_test[prob_cols].values,
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
        "method": "Uniform Shuffling Control Arm (PanDerm ViT-Base)",
        "runtime_info": runtime_info,
        "validation_overall": val_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    with open(results_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"EXP-2C EXECUTION COMPLETE (seed={seed}): Test Top-1 Acc: {test_metrics['accuracy']:.4f} | Bal Acc: {test_metrics['balanced_accuracy']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info("=" * 80)
    wb_logger.finish()


if __name__ == "__main__":
    main()

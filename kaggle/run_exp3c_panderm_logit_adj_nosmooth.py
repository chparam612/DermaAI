"""
Exp-3c Kaggle Execution Script: PanDerm ViT-Base with Train-Time Logit Adjustment without Label Smoothing (30 Epochs).

Workflow compliance:
- Reads the split file, computes SHA-256 hash, and strictly verifies it matches the repo hash.
- Config-driven (configs/experiments/exp3c_panderm_logit_adj_nosmooth_kaggle.yaml).
- Implements Train-Time Logit Adjustment (Menon et al., ICLR 2021) without label smoothing (tau=1.0, label_smoothing=0.0).
- Layer-wise LR decay (0.9), AdamW, cosine annealing with 5-epoch warmup.
- Evaluates best checkpoint on BOTH validation and test splits using RAW logits.
- Saves predictions CSVs (with per-class RAW LOGITS and PROBABILITIES for val AND test).
- Explicitly exports val_raw_logits.npy and test_raw_logits.npy for post-hoc analysis.
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
from src.losses.classification import LogitAdjustedCrossEntropy
from src.models.panderm import PanDermClassifier
from src.train.checkpointing import CheckpointManager
from src.train.engine import train_one_epoch, validate
from src.utils.reproducibility import set_seed

EXPECTED_SPLIT_HASH = "a4346dcf4d91a8bf6382a26c42618be327975de7f17596ce1fddfbcf1e160c36"
logger = logging.getLogger("kaggle_exp3c_panderm_logit_adj_nosmooth")


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
            f"DATASET INTEGRITY FAILED: SHA-256 of splits file ({computed_hash}) "
            f"does not match expected contract ({EXPECTED_SPLIT_HASH})!"
        )
    logger.info("[PASSED] Split file SHA-256 hash verified: exact lesion-level partition confirmed.")
    return computed_hash


def build_panderm_optimizer(
    model: nn.Module,
    lr: float = 2.5e-4,
    weight_decay: float = 0.05,
    layer_decay: float = 0.9,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with layer-wise learning rate decay for PanDerm ViT-Base."""
    num_layers = 12
    layer_scales = [layer_decay ** (num_layers - i) for i in range(num_layers + 2)]

    param_groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        cur_decay = 0.0 if (param.ndim <= 1 or name.endswith(".bias")) else weight_decay

        layer_id = 0
        if "patch_embed" in name or "pos_embed" in name or "cls_token" in name:
            layer_id = 0
        elif "blocks." in name or "encoder.layers." in name:
            for part in name.split("."):
                if part.isdigit():
                    layer_id = int(part) + 1
                    break
        elif "head" in name or "fc" in name or "classifier" in name or "norm" in name:
            layer_id = num_layers + 1

        scale = layer_scales[min(layer_id, len(layer_scales) - 1)]
        group_lr = lr * scale

        group_key = (group_lr, cur_decay)
        if group_key not in param_groups:
            param_groups[group_key] = {"params": [], "lr": group_lr, "weight_decay": cur_decay}
        param_groups[group_key]["params"].append(param)

    optimizer = torch.optim.AdamW(list(param_groups.values()))
    logger.info(f"Initialized AdamW optimizer with {len(param_groups)} layer-decay parameter groups (base_lr={lr}, layer_decay={layer_decay})")
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Exp-3c: PanDerm Fine-Tuning with Train-Time Logit Adjustment (No Label Smoothing)")
    parser.add_argument("--config", type=str, default="configs/experiments/exp3c_panderm_logit_adj_nosmooth_kaggle.yaml", help="Path to config YAML")
    parser.add_argument("--image-dir", type=str, default=None, help="Override image directory")
    parser.add_argument("--splits-file", type=str, default=None, help="Override splits file")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Override pretrained checkpoint path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override checkpoint output directory")
    parser.add_argument("--results-dir", type=str, default=None, help="Override results output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--tau", type=float, default=None, help="Logit adjustment strength tau")
    parser.add_argument("--label-smoothing", type=float, default=None, help="Label smoothing epsilon (default 0.0 for Exp-3c)")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch smoke test and exit")
    parser.add_argument("--online", action="store_true", help="Enable online W&B logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # 1. Load Configuration
    config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. Extract Hyperparameters & CLI Overrides
    exp_cfg = cfg.get("experiment", {})
    exp_name = exp_cfg.get("name", "exp3c-panderm-logit-adj-nosmooth")
    seed = args.seed if args.seed is not None else exp_cfg.get("seed", 42)

    data_cfg = cfg.get("data", {})
    image_dir = args.image_dir or data_cfg.get("image_dir")
    splits_file = args.splits_file or data_cfg.get("splits_file")
    label_mapping_file = data_cfg.get("label_mapping", "configs/label_mapping.json")

    train_cfg = cfg.get("training", {})
    epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 30)
    batch_size = args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 64)
    smoke_test = args.smoke_test or train_cfg.get("smoke_test", False)
    monitor = train_cfg.get("monitor", "val_macro_f1")
    warmup_epochs = int(train_cfg.get("warmup_epochs", 5))

    opt_cfg = cfg.get("optimizer", {})
    base_lr = float(opt_cfg.get("lr", 2.5e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.05))
    layer_decay = float(opt_cfg.get("layer_decay", 0.9))

    loss_cfg = cfg.get("loss", {})
    tau = float(args.tau if args.tau is not None else loss_cfg.get("tau", 1.0))
    label_smoothing = float(args.label_smoothing if args.label_smoothing is not None else loss_cfg.get("label_smoothing", 0.0))

    out_cfg = cfg.get("output", {})
    suffix = f"_seed{seed}" if seed != 42 else ""
    default_ckpt_dir = f"/kaggle/working/checkpoints/{exp_name.replace('-', '_')}{suffix}"
    default_results_dir = f"/kaggle/working/results/{exp_name.replace('-', '_')}{suffix}"
    checkpoint_dir = Path(args.checkpoint_dir or out_cfg.get("checkpoint_dir", default_ckpt_dir))
    results_dir = Path(args.results_dir or out_cfg.get("results_dir", default_results_dir))

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Environment & Seed Setup
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Execution environment: Device={device}, Experiment={exp_name}, Seed={seed}, tau={tau}, label_smoothing={label_smoothing}")

    # 4. Strict Split File SHA-256 Gate
    splits_path = Path(splits_file)
    verify_split_integrity(splits_path)

    # 5. Label Mapping Verification
    label_mapping_path = PROJECT_ROOT / label_mapping_file if not Path(label_mapping_file).is_absolute() else Path(label_mapping_file)
    class_to_idx = load_label_mapping(label_mapping_path)
    class_names = [cls for cls, _ in sorted(class_to_idx.items(), key=lambda x: x[1])]
    num_classes = len(class_names)
    assert num_classes == 11, f"Expected 11 diagnostic classes, got {num_classes}: {class_names}"

    # 6. Dataloaders Setup (Strictly Uniform Sampler)
    num_workers = int(cfg.get("dataloader", {}).get("num_workers", 4))
    logger.info(f"Creating dataloaders (batch_size={batch_size}, num_workers={num_workers}, uniform_sampler=True)...")
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
        use_weighted_sampler=cfg.get("dataloader", {}).get("weighted_sampler", False),
    )

    # 7. Compute class priors from training split
    train_labels = train_loader.dataset.get_labels()
    counts = np.bincount(train_labels, minlength=num_classes)
    total_samples = int(np.sum(counts))
    priors = (counts / total_samples).astype(np.float32)
    logger.info(f"Computed training class priors across {total_samples} samples:")
    for idx, cls in enumerate(class_names):
        logger.info(f"  [{idx:2d}] {cls:<10}: count={counts[idx]:4d} | prior={priors[idx]:.5f}")

    # 8. Model Initialization
    model_cfg = cfg.get("model", {})
    ckpt_path = model_cfg.get("checkpoint_path", None)
    logger.info(f"Initializing PanDermClassifier (architecture='panderm_base', num_classes={num_classes})...")
    model = PanDermClassifier(
        architecture="panderm_base",
        num_classes=num_classes,
        checkpoint_path=ckpt_path,
    )
    model.to(device)

    # 9. Loss Function: LogitAdjustedCrossEntropy without label smoothing
    logger.info(f"Initializing LogitAdjustedCrossEntropy (tau={tau}, label_smoothing={label_smoothing})...")
    criterion = LogitAdjustedCrossEntropy(
        priors=priors,
        tau=tau,
        smoothing=label_smoothing,
    ).to(device)

    # 10. Smoke Test Gate
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
        print("[SMOKE TEST] Exp-3c smoke test passed", flush=True)
        return

    # 11. Optimizer & LR Scheduler
    optimizer = build_panderm_optimizer(model, lr=base_lr, weight_decay=weight_decay, layer_decay=layer_decay)

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        return max(1e-6 / base_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 12. CheckpointManager & W&B Logger
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

    # 13. Training Loop
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
    logger.info(f"Starting Exp-3c training for {epochs} epochs on {device}...")

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
        current_lr = scheduler.get_last_lr()[0]

        combined_m = {**train_m, **val_m, "lr": current_lr, "epoch": epoch}
        wb_logger.log_metrics(combined_m, step=epoch)

        training_curves["epochs"].append(epoch)
        training_curves["train_loss"].append(train_m["train_loss"])
        training_curves["train_accuracy"].append(train_m["train_accuracy"])
        training_curves["val_loss"].append(val_m["val_loss"])
        training_curves["val_accuracy"].append(val_m["val_accuracy"])
        training_curves["val_balanced_acc"].append(val_m["val_balanced_acc"])
        training_curves["val_macro_f1"].append(val_m["val_macro_f1"])
        training_curves["val_weighted_f1"].append(val_m["val_weighted_f1"])
        training_curves["lr"].append(current_lr)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_m['train_loss']:.4f} | "
            f"Val Loss: {val_m['val_loss']:.4f} | "
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

    # 14. Load best checkpoint for final evaluation
    best_ckpt = checkpoint_dir / "best_model.pt"
    if best_ckpt.is_file():
        ckpt_mgr.load(best_ckpt, model)
        logger.info(f"Loaded best checkpoint from {best_ckpt}")

    # 15. Generate predictions on BOTH validation and test splits
    logger.info("Generating predictions and metrics on VALIDATION split...")
    df_val = generate_predictions(
        model=model,
        dataloader=val_loader,
        device=device,
        class_names=class_names,
    )
    save_predictions(df_val, results_dir / "predictions_val.csv")

    # Explicitly save raw logits array for post-hoc t-sweep
    logit_cols = [f"logit_{c}" for c in class_names]
    np.save(results_dir / "val_raw_logits.npy", df_val[logit_cols].values)

    logger.info("Generating predictions and metrics on TEST split...")
    df_test = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
    )
    save_predictions(df_test, results_dir / "predictions_test.csv")
    save_predictions(df_test, results_dir / "predictions.csv")
    np.save(results_dir / "test_raw_logits.npy", df_test[logit_cols].values)

    # 16. Compute final metrics
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
        "method": "Train-Time Logit Adjustment without Label Smoothing (Menon et al., ICLR 2021; tau=1.0, eps=0.0)",
        "runtime_info": runtime_info,
        "validation_overall": val_metrics,
        "test_overall": test_metrics,
        "subgroup_fairness": subgroup_metrics,
    }
    with open(results_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"EXP-3C EXECUTION COMPLETE (seed={seed}): Test Top-1 Acc: {test_metrics['accuracy']:.4f} | Bal Acc: {test_metrics['balanced_accuracy']:.4f} | Macro-F1: {test_metrics['macro_f1']:.4f}")
    logger.info("=" * 80)
    wb_logger.finish()


if __name__ == "__main__":
    main()

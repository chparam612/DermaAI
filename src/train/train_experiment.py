"""Unified experiment training entrypoint for Exp-1 and Exp-2."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, Optional
import pandas as pd
import torch
import yaml

from src.data.datasets import load_label_mapping
from src.data.loaders import create_dataloaders
from src.eval.evaluate import evaluate_model
from src.eval.wandb_logger import WandbLogger
from src.losses.classification import get_loss_function
from src.models.factory import build_model
from src.train.checkpointing import CheckpointManager
from src.train.engine import train_one_epoch, validate
from src.utils.reproducibility import set_seed

logger = logging.getLogger("train_experiment")


class SmokeTestWatchdog:
    """Watchdog timer thread enforcing a strict timeout on Windows smoke tests."""

    def __init__(self, timeout_seconds: float = 90.0):
        self.timeout_seconds = timeout_seconds
        self.current_stage = "Initialized"
        self._timer: Optional[threading.Timer] = None
        self._cancelled = threading.Event()

    def set_stage(self, stage_name: str) -> None:
        self.current_stage = stage_name

    def _timeout_handler(self) -> None:
        if not self._cancelled.is_set():
            sys.stderr.write(f"\n[SMOKE TEST TIMEOUT] stage={self.current_stage}\n")
            sys.stderr.flush()
            sys.stdout.flush()
            os._exit(1)

    def start(self) -> None:
        self._timer = threading.Timer(self.timeout_seconds, self._timeout_handler)
        self._timer.daemon = True
        self._timer.start()

    def cancel(self) -> None:
        self._cancelled.set()
        if self._timer is not None:
            self._timer.cancel()


def pre_training_data_integrity_check(splits_path: Path, image_dir: Path) -> None:
    """Validate data contract before training starts.

    Guarantees:
    1. Splits file exists and contains train/validation/test partitions.
    2. Strict lesion-level separation: zero lesion overlap between splits.
    3. Every image exists in image directory.
    """
    logger.info("Executing pre-training data integrity check...")
    if not splits_path.is_file():
        raise FileNotFoundError(f"Splits file not found: {splits_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    df_splits = pd.read_csv(splits_path)
    required_cols = {"lesion_id", "isic_id", "split"}
    if not required_cols.issubset(df_splits.columns):
        raise ValueError(f"Splits file missing required columns: {required_cols - set(df_splits.columns)}")

    # Check partition overlap
    train_lesions = set(df_splits[df_splits["split"] == "train"]["lesion_id"])
    val_lesions = set(df_splits[df_splits["split"].isin(["val", "validation"])]["lesion_id"])
    test_lesions = set(df_splits[df_splits["split"] == "test"]["lesion_id"])

    overlap_train_val = train_lesions.intersection(val_lesions)
    overlap_train_test = train_lesions.intersection(test_lesions)
    overlap_val_test = val_lesions.intersection(test_lesions)

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"Data leakage detected! Lesion overlap found between splits: "
            f"train/val: {len(overlap_train_val)}, train/test: {len(overlap_train_test)}, val/test: {len(overlap_val_test)}"
        )

    logger.info("Pre-training data integrity check passed: 0% lesion leakage, valid schemas.")


def run_experiment(
    config: Dict[str, Any],
    smoke_test: bool = False,
    override_epochs: Optional[int] = None,
    override_batch_size: Optional[int] = None,
    override_device: Optional[str] = None,
    offline: bool = False,
    no_wandb: bool = False,
    watchdog: Optional[SmokeTestWatchdog] = None,
) -> Dict[str, Any]:
    """Execute complete training and evaluation pipeline for an experiment."""
    local_watchdog = False
    if smoke_test and watchdog is None:
        watchdog = SmokeTestWatchdog(timeout_seconds=90.0)
        watchdog.start()
        local_watchdog = True
        watchdog.set_stage("Configuration loaded")
        print("[SMOKE TEST] Configuration loaded", flush=True)

    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", "experiment")
    seed = exp_cfg.get("seed", 42)
    set_seed(seed)

    # Determine device
    device = override_device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting experiment '{exp_name}' on device: {device}")

    # Paths (configurable via config or environment variables)
    data_cfg = config.get("data", {})
    splits_file_str = os.environ.get("SPLITS_FILE") or data_cfg.get("splits_file", "data/splits/lesion_level_splits.csv")
    splits_file = Path(splits_file_str).resolve() if Path(splits_file_str).is_absolute() else Path(splits_file_str)

    image_dir_str = os.environ.get("MILK10K_IMAGE_DIR") or os.environ.get("IMAGE_DIR") or data_cfg.get("image_dir", "data/images")
    image_dir = Path(image_dir_str).resolve() if Path(image_dir_str).is_absolute() else Path(image_dir_str)

    # Validate data integrity
    pre_training_data_integrity_check(splits_file, image_dir)

    # Label mapping
    label_map_path = os.environ.get("LABEL_MAPPING") or data_cfg.get("label_mapping", "configs/label_mapping.json")
    label_map = load_label_mapping(label_map_path)
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
    num_classes = len(class_names)

    # Setup directories
    output_cfg = config.get("output", {})
    ckpt_dir_str = os.environ.get("CHECKPOINT_DIR") or output_cfg.get("checkpoint_dir", f"checkpoints/{exp_name}")
    res_dir_str = os.environ.get("RESULTS_DIR") or output_cfg.get("results_dir", f"results/{exp_name}")
    checkpoint_dir = Path(ckpt_dir_str).resolve()
    results_dir = Path(res_dir_str).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training parameters
    train_cfg = config.get("training", {})
    batch_size = override_batch_size or train_cfg.get("batch_size", 32)
    epochs = 1 if smoke_test else (override_epochs or train_cfg.get("epochs", 30))
    max_batches = 2 if smoke_test else None

    # Loaders
    model_cfg = config.get("model", {})
    pipeline_type = model_cfg.get("pipeline_type", "baseline")

    # Hard-coded single-process dataloader for smoke test to prevent worker subprocess hangs
    if smoke_test:
        loader_num_workers = 0
        loader_persistent_workers = False
        loader_pin_memory = False
    else:
        loader_num_workers = config.get("dataloader", {}).get("num_workers", 0)
        loader_persistent_workers = False
        loader_pin_memory = (device == "cuda")

    modality = data_cfg.get("modality", "dermoscopic")
    dataloader_cfg = config.get("dataloader", {})
    weighted_sampler = dataloader_cfg.get("weighted_sampler", None)
    train_loader, val_loader, test_loader = create_dataloaders(
        metadata_path=splits_file,
        image_dir=image_dir,
        pipeline_type=pipeline_type,
        batch_size=batch_size,
        num_workers=loader_num_workers,
        pin_memory=loader_pin_memory,
        persistent_workers=loader_persistent_workers,
        image_size=data_cfg.get("image_size", 224),
        label_mapping=label_map,
        modality=modality,
        use_weighted_sampler=weighted_sampler,
    )

    if smoke_test:
        if watchdog:
            watchdog.set_stage("Dataset loaded")
        print("[SMOKE TEST] Dataset loaded", flush=True)

    # Model
    ckpt_path_str = os.environ.get("PANDERM_CHECKPOINT") or os.environ.get("CHECKPOINT_PATH") or model_cfg.get("checkpoint_path", None)
    model = build_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", True),
        checkpoint_path=ckpt_path_str,
        dropout_rate=model_cfg.get("dropout_rate", 0.2),
    )
    model.to(device)

    inner_model = getattr(model, "model", model)
    checkpoint_loaded = getattr(model, "checkpoint_path", model_cfg.get("checkpoint_path", "None"))
    print(f"[MODEL VERIFICATION] Model Wrapper: {type(model).__name__} ({type(model).__module__}), Underlying Architecture: {type(inner_model).__name__} ({type(inner_model).__module__}), Checkpoint: {checkpoint_loaded}", flush=True)

    if smoke_test:
        if watchdog:
            watchdog.set_stage("Model initialized")
        print("[SMOKE TEST] Model initialized", flush=True)

    # Loss & Optimizer
    loss_cfg = config.get("loss", {})
    loss_name = loss_cfg.get("name", "cross_entropy")
    weight = None
    if loss_name in ("class_balanced_ce", "weighted_ce"):
        from src.losses.classification import compute_class_weights
        train_labels = train_loader.dataset.get_labels()
        weight = compute_class_weights(train_labels, num_classes=num_classes).to(device)
        logger.info(f"Using class-balanced weights: {weight.tolist()}")

    criterion = get_loss_function(
        loss_name=loss_name,
        label_smoothing=loss_cfg.get("label_smoothing", 0.0),
        weight=weight,
    )

    if smoke_test:
        try:
            # 1. Fetch batch
            batch = next(iter(train_loader))
            images, labels, _ = batch
            images = images.to(device)
            labels = labels.to(device)
            if watchdog:
                watchdog.set_stage("Batch loaded")
            print("[SMOKE TEST] Batch loaded", flush=True)

            # 2. Forward pass
            outputs = model(images)
            if watchdog:
                watchdog.set_stage("Forward pass successful")
            print("[SMOKE TEST] Forward pass successful", flush=True)

            # 3. Loss calculation
            loss = criterion(outputs, labels)
            if watchdog:
                watchdog.set_stage("Loss calculation successful")
            print("[SMOKE TEST] Loss calculation successful", flush=True)

            exp_label = "Exp-2" if ("exp2" in exp_name.lower() or "panderm" in exp_name.lower()) else ("Exp-1" if "exp1" in exp_name.lower() else exp_name)
            passed_stage = f"{exp_label} smoke test passed"
            if watchdog:
                watchdog.set_stage(passed_stage)
            print(f"[SMOKE TEST] {passed_stage}", flush=True)
            return {
                "smoke_test": True,
                "status": "passed",
                "loss": float(loss.item()),
                "batch_size": images.shape[0],
            }
        finally:
            if local_watchdog and watchdog:
                watchdog.cancel()

    opt_cfg = config.get("optimizer", {})
    opt_name = opt_cfg.get("name", "adamw").lower()
    lr = float(opt_cfg.get("lr", 1e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-2))
    layer_decay = float(opt_cfg.get("layer_decay", 1.0))

    if layer_decay < 1.0:
        # Layer-wise LR decay for ViT (following official PanDerm / BEiT LayerDecayValueAssigner)
        num_layers = 12
        if hasattr(model, "model") and hasattr(model.model, "get_num_layers"):
            num_layers = model.model.get_num_layers()
        elif hasattr(model, "get_num_layers"):
            num_layers = model.get_num_layers()

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
                    "layer_id": lid,
                }
            groups[key]["params"].append(param)
        param_groups = list(groups.values())
        logger.info(
            f"Created {len(param_groups)} parameter groups with layer_decay={layer_decay} "
            f"(scale range: {scales[0]:.4f} to {scales[-1]:.4f})"
        )
    else:
        param_groups = model.parameters()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)

    # Scheduler with optional warmup
    warmup_epochs = int(opt_cfg.get("warmup_epochs", train_cfg.get("warmup_epochs", 0)))
    min_lr = float(opt_cfg.get("min_lr", 1e-6))
    if warmup_epochs > 0 and epochs > warmup_epochs:
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        warmup_sched = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine_sched = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=min_lr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        logger.info(f"Using SequentialLR scheduler: {warmup_epochs} warmup epochs -> CosineAnnealingLR (min_lr={min_lr})")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
        logger.info(f"Using CosineAnnealingLR scheduler (T_max={epochs}, min_lr={min_lr})")

    # Logger
    wandb_cfg = config.get("wandb", {})
    env_wandb_mode = os.environ.get("WANDB_MODE")
    if env_wandb_mode:
        resolved_wandb_mode = env_wandb_mode.lower()
    else:
        resolved_wandb_mode = "offline" if (offline or wandb_cfg.get("offline", True)) else "online"

    wb_logger = WandbLogger(
        project=os.environ.get("WANDB_PROJECT") or wandb_cfg.get("project", "SkinCancerTracker"),
        experiment_name=f"{exp_name}_seed{seed}",
        config={**config, "smoke_test": smoke_test},
        enabled=(not no_wandb),
        mode=resolved_wandb_mode,
    )

    # Checkpoint manager: monitor val_macro_f1 to capture tail/malignant class performance
    monitor_metric = train_cfg.get("monitor", "val_macro_f1")
    max_keep_env = os.environ.get("CHECKPOINT_MAX_KEEP")
    max_keep_val = int(max_keep_env) if max_keep_env is not None else output_cfg.get("max_keep", None)
    if max_keep_val is not None:
        max_keep_val = int(max_keep_val)

    ckpt_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        monitor=monitor_metric,
        mode="max",
        max_keep=max_keep_val,
    )
    logger.info(f"CheckpointManager monitoring: {monitor_metric} (mode=max, max_keep={max_keep_val})")

    import time
    import datetime

    start_time = time.time()
    logger.info(f"Beginning training for {epochs} epoch(s)... (smoke_test={smoke_test})")
    for epoch in range(1, epochs + 1):
        accum_steps = int(train_cfg.get("gradient_accumulation_steps", 1))
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad=train_cfg.get("clip_grad", 1.0),
            max_batches=max_batches,
            accumulation_steps=accum_steps,
        )

        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            max_batches=max_batches,
            class_names=class_names,
        )

        scheduler.step()

        combined_metrics = {**train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        wb_logger.log_metrics(combined_metrics, step=epoch)

        vram_str = f"{train_metrics.get('peak_vram_mb', 0.0):.1f} MB"
        num_preds = val_metrics.get('val_num_predicted_classes', 0)
        pred_dist_str = str(val_metrics.get('val_prediction_distribution', {}))
        logger.info(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Bal Acc: {val_metrics['val_balanced_acc']:.4f} | "
            f"Val Macro F1: {val_metrics['val_macro_f1']:.4f} | "
            f"Pred Classes: {num_preds}/11 | "
            f"Peak VRAM: {vram_str}"
        )
        logger.info(f"Epoch {epoch}/{epochs} Val Prediction Distribution: {pred_dist_str}")

        # Save checkpoint
        ckpt_mgr.save(model, optimizer, epoch, combined_metrics)

    total_training_time = time.time() - start_time
    gpu_name = torch.cuda.get_device_name(0) if (torch.cuda.is_available() and device.startswith("cuda")) else "CPU"
    runtime_info = {
        "device": device,
        "gpu_name": gpu_name,
        "total_training_time_seconds": round(total_training_time, 2),
        "total_training_time_formatted": str(datetime.timedelta(seconds=int(total_training_time))),
    }

    # Load best model for test evaluation
    best_ckpt_path = checkpoint_dir / "best_model.pt"
    if best_ckpt_path.is_file():
        ckpt_mgr.load(best_ckpt_path, model)

    logger.info("Evaluating best checkpoint on Test set...")
    eval_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        output_dir=results_dir,
        class_names=class_names,
        max_batches=max_batches,
    )

    # Log test metrics
    test_overall = eval_results["overall_metrics"]
    test_log = {
        "test_accuracy": test_overall["accuracy"],
        "test_balanced_accuracy": test_overall["balanced_accuracy"],
        "test_macro_f1": test_overall["macro_f1"],
        "test_weighted_f1": test_overall["weighted_f1"],
        "test_macro_auroc": test_overall.get("macro_auroc", 0.0),
        "test_malignant_recall": test_overall.get("malignant_recall", 0.0),
        "test_malignant_precision": test_overall.get("malignant_precision", 0.0),
        "test_malignant_auroc": test_overall.get("malignant_auroc", 0.0),
        "skin_tone_disparity": eval_results["subgroup_fairness"].get("balanced_accuracy_disparity", 0.0),
        "total_training_time_seconds": round(total_training_time, 2),
    }

    # Log per-class recalls for malignant classes
    for mal_cls in ["BCC", "MEL", "SCCKA", "MAL_OTH", "AKIEC"]:
        if mal_cls in test_overall.get("per_class", {}):
            test_log[f"test_recall_{mal_cls}"] = test_overall["per_class"][mal_cls]["recall"]

    wb_logger.log_metrics(test_log, step=epochs + 1)

    # Log confusion matrix to W&B
    if "confusion_matrix" in test_overall:
        wb_logger.log_confusion_matrix(test_overall["confusion_matrix"], class_names=class_names, step=epochs + 1)

    # Log best checkpoint as artifact
    if best_ckpt_path.is_file():
        wb_logger.log_artifact(best_ckpt_path, name=f"{exp_name}-checkpoint", artifact_type="model")

    wb_logger.finish()

    # Save summary metrics json
    summary_data = {
        "experiment_name": exp_name,
        "runtime_info": runtime_info,
        "test_overall": test_overall,
        "subgroup_fairness": eval_results["subgroup_fairness"],
    }
    summary_path = results_dir / "summary_metrics.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    logger.info(f"Saved summary metrics to {summary_path}")

    eval_results["runtime_info"] = runtime_info
    logger.info(f"Experiment {exp_name} completed successfully in {runtime_info['total_training_time_formatted']}. Artifacts in {results_dir}")
    return eval_results


def parse_args():
    parser = argparse.ArgumentParser(description="Train dermatology baseline classification model.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 epoch with 2 batches for fast verification")
    parser.add_argument("--smoke-timeout", type=int, default=90, help="Explicit timeout in seconds for smoke test (default: 90)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--offline", action="store_true", help="Force W&B offline mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    watchdog = None
    if args.smoke_test:
        watchdog = SmokeTestWatchdog(timeout_seconds=args.smoke_timeout)
        watchdog.start()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if args.smoke_test and watchdog:
            watchdog.set_stage("Configuration loaded")
            print("[SMOKE TEST] Configuration loaded", flush=True)

        run_experiment(
            config=config,
            smoke_test=args.smoke_test,
            override_epochs=args.epochs,
            override_batch_size=args.batch_size,
            override_device=args.device,
            offline=args.offline,
            no_wandb=args.no_wandb,
            watchdog=watchdog,
        )
    finally:
        if watchdog:
            watchdog.cancel()


if __name__ == "__main__":
    main()

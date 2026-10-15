"""Exp-2: PanDerm Fine-Tuning Script.

Fine-tuning PanDerm dermatology foundation model on MILK10k dermoscopic modality.
References official run_class_finetuning.py implementation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.train.train_experiment import run_experiment

logger = logging.getLogger("train_exp2_panderm")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Exp-2: PanDerm Fine-Tuning on Dermoscopic Skin Lesions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp2_panderm.yaml",
        help="Path to experiment config YAML (default: configs/experiments/exp2_panderm.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--offline", action="store_true", help="Force W&B offline mode")
    parser.add_argument("--online", action="store_true", help="Force W&B online mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking completely")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch smoke test for rapid pipeline verification")
    parser.add_argument("--image-dir", type=str, default=None, help="Override data.image_dir")
    parser.add_argument("--splits-file", type=str, default=None, help="Override data.splits_file")
    parser.add_argument("--label-mapping", type=str, default=None, help="Override data.label_mapping")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Override model.checkpoint_path")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Override output.checkpoint_dir")
    parser.add_argument("--results-dir", type=str, default=None, help="Override output.results_dir")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override training.gradient_accumulation_steps")
    parser.add_argument("--num-workers", type=int, default=None, help="Override dataloader.num_workers")
    parser.add_argument("--max-keep", type=int, default=None, help="Max epoch checkpoints to retain on disk")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config_path = Path(args.config).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI Overrides
    if args.lr is not None:
        config.setdefault("optimizer", {})["lr"] = args.lr
    if args.image_dir is not None:
        config.setdefault("data", {})["image_dir"] = args.image_dir
    if args.splits_file is not None:
        config.setdefault("data", {})["splits_file"] = args.splits_file
    if args.label_mapping is not None:
        config.setdefault("data", {})["label_mapping"] = args.label_mapping
    if args.checkpoint_path is not None:
        config.setdefault("model", {})["checkpoint_path"] = args.checkpoint_path
    if args.checkpoint_dir is not None:
        config.setdefault("output", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir is not None:
        config.setdefault("output", {})["results_dir"] = args.results_dir
    if args.grad_accum is not None:
        config.setdefault("training", {})["gradient_accumulation_steps"] = args.grad_accum
    if args.num_workers is not None:
        config.setdefault("dataloader", {})["num_workers"] = args.num_workers
    if args.max_keep is not None:
        config.setdefault("output", {})["max_keep"] = args.max_keep
    if args.online:
        config.setdefault("wandb", {})["offline"] = False

    logger.info("=" * 70)
    logger.info("STARTING EXP-2: PANDERM FINE-TUNING (Dermoscopic Modality)")
    logger.info("=" * 70)

    results = run_experiment(
        config=config,
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_device=args.device,
        offline=False if args.online else args.offline,
        no_wandb=args.no_wandb,
    )

    logger.info("Exp-2 execution finished successfully.")
    return results


if __name__ == "__main__":
    main()

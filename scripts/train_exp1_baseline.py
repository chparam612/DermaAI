"""Exp-1: ResNet-50 ImageNet-Pretrained Baseline Training Script.

Single-modality dermoscopic skin lesion classification on MILK10k dataset.
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

logger = logging.getLogger("train_exp1_baseline")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Exp-1: ResNet-50 Baseline on Dermoscopic Skin Lesions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp1_baseline.yaml",
        help="Path to experiment config YAML (default: configs/experiments/exp1_baseline.yaml)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cpu' or 'cuda')")
    parser.add_argument("--offline", action="store_true", help="Force W&B offline mode")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking completely")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1-batch smoke test for rapid pipeline verification")
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

    if args.lr is not None:
        config.setdefault("optimizer", {})["lr"] = args.lr

    logger.info("=" * 70)
    logger.info("STARTING EXP-1: RESNET-50 BASELINE (Dermoscopic Modality)")
    logger.info("=" * 70)

    results = run_experiment(
        config=config,
        smoke_test=args.smoke_test,
        override_epochs=args.epochs,
        override_batch_size=args.batch_size,
        override_device=args.device,
        offline=args.offline,
        no_wandb=args.no_wandb,
    )

    logger.info("Exp-1 execution finished successfully.")
    return results


if __name__ == "__main__":
    main()

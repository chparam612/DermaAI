"""Evaluation pipeline for generating test metrics and fairness reports."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import yaml

from src.data.datasets import load_label_mapping
from src.data.loaders import create_dataloaders
from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import generate_predictions, save_predictions
from src.models.factory import build_model

logger = logging.getLogger(__name__)


def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    output_dir: Optional[Union[str, Path]] = None,
    class_names: Optional[list] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute complete evaluation on DataLoader and optionally persist outputs.

    Args:
        model: Evaluated PyTorch model.
        test_loader: DataLoader for test partition.
        device: Device ('cpu' or 'cuda').
        output_dir: Directory to save predictions.csv and metrics.json.
        class_names: List of diagnostic class names.
        max_batches: Optional max batch limit for quick testing.

    Returns:
        Dictionary containing aggregate metrics and subgroup fairness audit.
    """
    df_preds = generate_predictions(
        model, test_loader, device=device, class_names=class_names, max_batches=max_batches
    )

    # Compute metrics
    metrics = calculate_classification_metrics(
        y_true=df_preds["true_label"].values,
        y_pred=df_preds["pred_label"].values,
        class_names=class_names,
    )

    subgroup_metrics = calculate_subgroup_metrics(
        df_predictions=df_preds,
        subgroup_col="skin_tone_class",
        true_col="true_label",
        pred_col="pred_label",
    )

    full_results = {
        "overall_metrics": metrics,
        "subgroup_fairness": subgroup_metrics,
    }

    if output_dir is not None:
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        # Save predictions CSV
        preds_file = out_path / "predictions.csv"
        save_predictions(df_preds, preds_file)

        # Save metrics JSON
        metrics_file = out_path / "metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        logger.info(f"Saved evaluation metrics to {metrics_file}")

    return full_results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained dermatology classification model.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment config YAML")
    parser.add_argument("--output-dir", type=str, default="results/evaluation", help="Output directory for reports")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate ('test' or 'validation')")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Label mapping
    label_map = load_label_mapping(config.get("label_mapping", "configs/label_mapping.json"))
    class_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

    # Build model
    model_cfg = config.get("model", {})
    model = build_model(
        model_name=model_cfg.get("name", "resnet50"),
        num_classes=len(class_names),
        pretrained=False,
    )

    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    # Build data loader
    data_cfg = config.get("data", {})
    _, val_loader, test_loader = create_dataloaders(
        metadata_path=data_cfg.get("splits_file", "data/splits/lesion_level_splits.csv"),
        image_dir=data_cfg.get("image_dir", "data/images"),
        pipeline_type=model_cfg.get("pipeline_type", "baseline"),
        batch_size=config.get("training", {}).get("batch_size", 32),
        label_mapping=label_map,
    )

    eval_loader = test_loader if args.split == "test" else val_loader
    results = evaluate_model(
        model=model,
        test_loader=eval_loader,
        device=args.device,
        output_dir=args.output_dir,
        class_names=class_names,
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    overall = results["overall_metrics"]
    print(f"Accuracy:          {overall['accuracy']:.4f}")
    print(f"Balanced Accuracy: {overall['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {overall['macro_f1']:.4f}")
    print(f"Weighted F1:       {overall['weighted_f1']:.4f}")
    print(f"Skin-tone Disparity: {results['subgroup_fairness'].get('balanced_accuracy_disparity', 0.0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

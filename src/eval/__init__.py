"""Evaluation subpackage for SkinCancerTracker."""

from src.eval.metrics import calculate_classification_metrics, calculate_subgroup_metrics
from src.eval.predictions import generate_predictions, save_predictions
from src.eval.evaluate import evaluate_model
from src.eval.compare_experiments import generate_ablation_table
from src.eval.wandb_logger import WandbLogger

__all__ = [
    "calculate_classification_metrics",
    "calculate_subgroup_metrics",
    "generate_predictions",
    "save_predictions",
    "evaluate_model",
    "generate_ablation_table",
    "WandbLogger",
]

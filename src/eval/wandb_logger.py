"""Weights & Biases Experiment Logger and Custom Medical Table Utility.

Provides structured logging for training scalars, per-class dermatology metrics,
and demographic / skin-tone subgroup fairness evaluation. Fully offline-safe
and functional even when W&B is not installed or running in headless CI environments.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

logger = logging.getLogger(__name__)

# Attempt to import wandb safely
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore
    WANDB_AVAILABLE = False


PER_CLASS_TABLE_COLUMNS = [
    "class_name",
    "support",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "balanced_accuracy",
]

PER_SKIN_TONE_TABLE_COLUMNS = [
    "skin_tone_group",
    "support",
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_score",
    "auc",
]

SUBGROUP_ERROR_TABLE_COLUMNS = [
    "sample_id",
    "true_label",
    "predicted_label",
    "confidence",
    "skin_tone_group",
    "error_type",
]


class WandbLogger:
    """Robust, offline-safe Weights & Biases experiment logger for medical CV."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        experiment_name: Optional[str] = None,
        enabled: bool = True,
        mode: Optional[str] = None,
        log_model: bool = False,
    ):
        """Initialize the WandbLogger.

        Args:
            config: Experiment hyperparameters and metadata dictionary.
            project: W&B project name.
            entity: W&B entity / team username.
            experiment_name: Descriptive name for this experiment run.
            enabled: Master switch. If False, logger operates entirely in offline dummy mode.
            mode: Run mode: 'online', 'offline', or 'disabled'.
            log_model: Whether to upload model checkpoints to W&B artifacts.
        """
        self.config = config or {}
        self.project = project or self.config.get("wandb", {}).get("project", "SkinCancerTracker")
        self.entity = entity or self.config.get("wandb", {}).get("entity", None)
        self.experiment_name = experiment_name or self.config.get("wandb", {}).get("experiment_name", None)
        self.log_model = log_model or self.config.get("wandb", {}).get("log_model", False)

        configured_enabled = self.config.get("wandb", {}).get("enabled", True)
        self.enabled = enabled and configured_enabled and WANDB_AVAILABLE

        self.mode = mode or self.config.get("wandb", {}).get("mode", "online" if self.enabled else "disabled")
        if not self.enabled:
            self.mode = "disabled"

        self.run = None
        # Cache logged tables and metrics for verification and offline testing
        self.history: List[Dict[str, Any]] = []
        self.logged_tables: Dict[str, Dict[str, Any]] = {}

        if self.enabled and WANDB_AVAILABLE:
            try:
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=self.experiment_name,
                    config=self.config,
                    mode=self.mode,
                    reinit=True,
                )
                logger.info(f"Initialized W&B run: {self.run.name} (id: {self.run.id}) in mode: {self.mode}")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B online ({e}). Falling back to disabled offline mode.")
                self.enabled = False
                self.run = None
        else:
            logger.info("WandbLogger running in offline/mock mode (W&B not initialized).")

    def log_metrics(self, metrics: Dict[str, Union[float, int, None]], step: Optional[int] = None) -> None:
        """Log scalar metrics (loss, accuracy, AUC, etc.).

        Args:
            metrics: Key-value pairs of metric names and numeric values.
            step: Optional training step / epoch index.
        """
        clean_metrics = {k: v for k, v in metrics.items() if v is not None and not np.isnan(v) if isinstance(v, (int, float, np.number))}
        entry = dict(clean_metrics)
        if step is not None:
            entry["step"] = step
        self.history.append(entry)

        if self.enabled and self.run is not None:
            self.run.log(clean_metrics, step=step)

    def log_per_class_table(
        self,
        class_metrics: List[Dict[str, Any]],
        table_name: str = "eval/per_class_metrics",
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create and log a structured W&B Table for per-class metrics.

        Columns: [class_name, support, precision, recall, f1_score, specificity, balanced_accuracy]

        Args:
            class_metrics: List of dicts, each containing metrics for one disease class.
            table_name: Artifact/table name in W&B UI.
            step: Optional epoch or evaluation step.

        Returns:
            Dictionary containing table structure and formatted rows.
        """
        rows: List[List[Any]] = []
        for item in class_metrics:
            row = [
                str(item.get("class_name", "Unknown")),
                int(item.get("support", 0)),
                _clean_metric(item.get("precision")),
                _clean_metric(item.get("recall")),
                _clean_metric(item.get("f1_score")),
                _clean_metric(item.get("specificity")),
                _clean_metric(item.get("balanced_accuracy")),
            ]
            rows.append(row)

        table_dict = {
            "columns": PER_CLASS_TABLE_COLUMNS,
            "data": rows,
            "step": step,
        }
        self.logged_tables[table_name] = table_dict

        if self.enabled and self.run is not None and WANDB_AVAILABLE:
            table = wandb.Table(columns=PER_CLASS_TABLE_COLUMNS, data=rows)
            self.run.log({table_name: table}, step=step)

        return table_dict

    def log_per_skin_tone_table(
        self,
        skin_tone_metrics: List[Dict[str, Any]],
        table_name: str = "eval/per_skin_tone_metrics",
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Create and log a structured W&B Table for fairness & skin tone evaluation.

        Columns: [skin_tone_group, support, accuracy, balanced_accuracy, precision, recall, f1_score, auc]

        Gracefully handles missing or unannotated subgroup metadata without fabricating values.

        Args:
            skin_tone_metrics: List of dicts with subgroup evaluation metrics.
            table_name: Name of the table artifact.
            step: Optional evaluation step.

        Returns:
            Dictionary containing table structure and formatted rows.
        """
        rows: List[List[Any]] = []
        for item in skin_tone_metrics:
            group = item.get("skin_tone_group")
            if group is None or str(group).strip() == "":
                group = "Unannotated / Unknown"

            row = [
                str(group),
                int(item.get("support", 0)),
                _clean_metric(item.get("accuracy")),
                _clean_metric(item.get("balanced_accuracy")),
                _clean_metric(item.get("precision")),
                _clean_metric(item.get("recall")),
                _clean_metric(item.get("f1_score")),
                _clean_metric(item.get("auc")),
            ]
            rows.append(row)

        table_dict = {
            "columns": PER_SKIN_TONE_TABLE_COLUMNS,
            "data": rows,
            "step": step,
        }
        self.logged_tables[table_name] = table_dict

        if self.enabled and self.run is not None and WANDB_AVAILABLE:
            table = wandb.Table(columns=PER_SKIN_TONE_TABLE_COLUMNS, data=rows)
            self.run.log({table_name: table}, step=step)

        return table_dict

    def log_subgroup_error_table(
        self,
        errors: List[Dict[str, Any]],
        table_name: str = "eval/subgroup_error_analysis",
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Log sample-level subgroup error analysis table for fairness diagnostics.

        Columns: [sample_id, true_label, predicted_label, confidence, skin_tone_group, error_type]

        Args:
            errors: List of error dictionaries.
            table_name: Name of table.
            step: Optional step index.

        Returns:
            Dictionary containing table structure and rows.
        """
        rows: List[List[Any]] = []
        for err in errors:
            skin_tone = err.get("skin_tone_group")
            if skin_tone is None or str(skin_tone).strip() == "":
                skin_tone = "Unannotated"

            row = [
                str(err.get("sample_id", "N/A")),
                str(err.get("true_label", "N/A")),
                str(err.get("predicted_label", "N/A")),
                _clean_metric(err.get("confidence")),
                str(skin_tone),
                str(err.get("error_type", "misclassification")),
            ]
            rows.append(row)

        table_dict = {
            "columns": SUBGROUP_ERROR_TABLE_COLUMNS,
            "data": rows,
            "step": step,
        }
        self.logged_tables[table_name] = table_dict

        if self.enabled and self.run is not None and WANDB_AVAILABLE:
            table = wandb.Table(columns=SUBGROUP_ERROR_TABLE_COLUMNS, data=rows)
            self.run.log({table_name: table}, step=step)

        return table_dict

    def finish(self) -> None:
        """Safely conclude the W&B run."""
        if self.enabled and self.run is not None and WANDB_AVAILABLE:
            try:
                self.run.finish()
            except Exception as e:
                logger.warning(f"Error finishing W&B run: {e}")
        self.run = None


def _clean_metric(val: Any) -> Optional[float]:
    """Helper to distinguish valid floats from None / NaN, preventing fake zeros."""
    if val is None:
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except (ValueError, TypeError):
        return None

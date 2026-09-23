"""Model checkpointing utilities for experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading model checkpoints during training."""

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        monitor: str = "val_balanced_acc",
        mode: str = "max",
    ):
        """Initialize CheckpointManager.

        Args:
            checkpoint_dir: Output directory to store checkpoint weights.
            monitor: Metric key to determine improvement.
            mode: 'max' (higher is better) or 'min' (lower is better).
        """
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric_value = -float("inf") if mode == "max" else float("inf")

    def is_better(self, current: float) -> bool:
        """Check if current metric value exceeds the current best."""
        if self.mode == "max":
            return current > self.best_metric_value
        return current < self.best_metric_value

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: PyTorch model.
            optimizer: Optimizer state.
            epoch: Epoch index.
            metrics: Dictionary of evaluated metrics.
            filename: Specific file name (defaults to 'checkpoint_epoch_{epoch}.pt').

        Returns:
            Path to saved checkpoint.
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        save_path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
        }
        torch.save(state, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

        # Check if this is the best so far
        metric_val = metrics.get(self.monitor)
        if metric_val is not None and self.is_better(metric_val):
            self.best_metric_value = metric_val
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(state, best_path)
            logger.info(f"Updated best model at {best_path} ({self.monitor}={metric_val:.4f})")

        # Also maintain last_model.pt
        last_path = self.checkpoint_dir / "last_model.pt"
        torch.save(state, last_path)

        return save_path

    @staticmethod
    def load(
        checkpoint_path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load state dict into model and optimizer.

        Args:
            checkpoint_path: Path to checkpoint file.
            model: Model to load weights into.
            optimizer: Optional optimizer to restore state into.

        Returns:
            Loaded state dict.
        """
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        if optimizer is not None and state.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        logger.info(f"Loaded checkpoint from {path} (epoch {state.get('epoch', 'N/A')})")
        return state

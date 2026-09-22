"""Training loops, optimization routines, and validation logic for SkinCancerTracker.

This module coordinates epochs, backpropagation, metric calculation, and checkpointing.
"""

from typing import Any, Dict


def train_one_epoch(*args: Any, **kwargs: Any) -> Dict[str, float]:
    """Stub for single training epoch (to be implemented in Phase 1)."""
    raise NotImplementedError("Training execution is scheduled for Phase 1.")


__all__ = ["train_one_epoch"]

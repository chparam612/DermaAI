"""Loss functions subpackage for SkinCancerTracker."""

from src.losses.classification import LabelSmoothingCrossEntropy, get_loss_function

__all__ = ["LabelSmoothingCrossEntropy", "get_loss_function"]

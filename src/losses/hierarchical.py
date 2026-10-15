"""Asymmetric Hierarchical Cost Loss for dermatology diagnostic triage.

Implements clinical error-cost penalization where false negatives (misclassifying
malignant lesions as benign) incur significantly higher penalties than false
positives or intra-superclass misclassifications.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.classification import LabelSmoothingCrossEntropy

# Default 11 diagnostic classes in MILK10k
DEFAULT_CLASSES = [
    "AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF",
    "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"
]

# Clinical malignancy partition (5 malignant classes, 6 benign classes)
DEFAULT_MALIGNANT_CLASSES = ["AKIEC", "BCC", "MAL_OTH", "MEL", "SCCKA"]
DEFAULT_BENIGN_CLASSES = ["BEN_OTH", "BKL", "DF", "INF", "NV", "VASC"]


def build_asymmetric_cost_matrix(
    num_classes: int = 11,
    malignant_indices: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
    malignant_names: Optional[Sequence[str]] = None,
    c_fn: float = 5.0,
    c_fp: float = 1.5,
    c_within_mal: float = 1.0,
    c_within_ben: float = 0.5,
) -> torch.Tensor:
    """Construct clinical asymmetric cost matrix C where C[i, j] is cost of predicting j when true is i.

    Cost Hierarchy:
    - C[i, i] = 0.0 (correct prediction)
    - True Malignant, Pred Benign (False Negative - missed malignancy): C[i, j] = c_fn (default 5.0)
    - True Benign, Pred Malignant (False Positive - unnecessary biopsy): C[i, j] = c_fp (default 1.5)
    - True Malignant, Pred Other Malignant (Intra-Malignant Triage): C[i, j] = c_within_mal (default 1.0)
    - True Benign, Pred Other Benign (Intra-Benign Triage): C[i, j] = c_within_ben (default 0.5)

    Args:
        num_classes: Total number of diagnostic classes K.
        malignant_indices: Sequence of integer indices representing malignant classes.
        class_names: Optional class names list to infer indices if malignant_indices is not provided.
        malignant_names: Optional malignant class names list.
        c_fn: Cost of false negative error (malignant -> benign).
        c_fp: Cost of false positive error (benign -> malignant).
        c_within_mal: Cost of error within malignant superclass.
        c_within_ben: Cost of error within benign superclass.

    Returns:
        torch.Tensor of shape (num_classes, num_classes) with dtype float32.
    """
    if malignant_indices is None:
        if class_names is not None and malignant_names is not None:
            mal_set = set(malignant_names)
            malignant_indices = [i for i, c in enumerate(class_names) if c in mal_set]
        elif class_names is not None:
            mal_set = set(DEFAULT_MALIGNANT_CLASSES)
            malignant_indices = [i for i, c in enumerate(class_names) if c in mal_set]
        else:
            # Fall back to indices matching DEFAULT_CLASSES
            c_names = DEFAULT_CLASSES[:num_classes]
            mal_set = set(DEFAULT_MALIGNANT_CLASSES)
            malignant_indices = [i for i, c in enumerate(c_names) if c in mal_set]

    mal_set = set(malignant_indices)
    cost = torch.zeros((num_classes, num_classes), dtype=torch.float32)

    for i in range(num_classes):
        is_i_mal = i in mal_set
        for j in range(num_classes):
            if i == j:
                cost[i, j] = 0.0
            else:
                is_j_mal = j in mal_set
                if is_i_mal and not is_j_mal:
                    # Malignant misclassified as Benign (False Negative)
                    cost[i, j] = float(c_fn)
                elif not is_i_mal and is_j_mal:
                    # Benign misclassified as Malignant (False Positive)
                    cost[i, j] = float(c_fp)
                elif is_i_mal and is_j_mal:
                    # Intra-malignant error (e.g. BCC as SCCKA)
                    cost[i, j] = float(c_within_mal)
                else:
                    # Intra-benign error (e.g. NV as BKL)
                    cost[i, j] = float(c_within_ben)

    return cost


class AsymmetricHierarchicalCostLoss(nn.Module):
    """Asymmetric Hierarchical Cost Loss for Dermatology Classification.

    Combines standard classification cross-entropy (or label-smoothed cross-entropy)
    with an expected clinical misclassification risk penalty:
        L(z, y) = L_CE(z, y) + alpha * sum_{j=1}^K C_{y, j} * p_j
    where:
        p = softmax(z)
        C_{y, j} is the asymmetric cost matrix entry for true class y and predicted class j
        alpha >= 0 is the risk weight factor

    When alpha = 0, this simplifies to standard Cross-Entropy (or Label-Smoothing CE).
    """

    def __init__(
        self,
        cost_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        num_classes: int = 11,
        alpha: float = 1.0,
        c_fn: float = 5.0,
        c_fp: float = 1.5,
        c_within_mal: float = 1.0,
        c_within_ben: float = 0.5,
        malignant_indices: Optional[Sequence[int]] = None,
        class_names: Optional[Sequence[str]] = None,
        label_smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'. Must be 'mean', 'sum', or 'none'.")

        self.alpha = float(alpha)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)

        if cost_matrix is not None:
            if isinstance(cost_matrix, np.ndarray):
                cost_tensor = torch.from_numpy(cost_matrix).float()
            elif isinstance(cost_matrix, torch.Tensor):
                cost_tensor = cost_matrix.clone().float()
            else:
                cost_tensor = torch.tensor(cost_matrix, dtype=torch.float32)

            if cost_tensor.ndim != 2 or cost_tensor.shape[0] != cost_tensor.shape[1]:
                raise ValueError(f"cost_matrix must be square 2D tensor, got shape {cost_tensor.shape}")
        else:
            cost_tensor = build_asymmetric_cost_matrix(
                num_classes=num_classes,
                malignant_indices=malignant_indices,
                class_names=class_names,
                c_fn=c_fn,
                c_fp=c_fp,
                c_within_mal=c_within_mal,
                c_within_ben=c_within_ben,
            )

        self.register_buffer("cost_matrix", cost_tensor)

        # Base cross entropy criterion
        if self.label_smoothing > 0.0:
            self.base_criterion = LabelSmoothingCrossEntropy(smoothing=self.label_smoothing)
        else:
            self.base_criterion = None

        if weight is not None:
            self.register_buffer("class_weights", weight.clone().float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass computing combined CE + Expected Cost Loss.

        Args:
            logits: Predicted class raw logits of shape (B, K).
            target: Ground truth integer class indices of shape (B,).

        Returns:
            Scalar loss tensor if reduction is 'mean' or 'sum', else tensor of shape (B,).
        """
        if logits.ndim != 2:
            raise ValueError(f"Expected 2D logits tensor of shape (batch_size, num_classes), got {logits.shape}")
        if target.ndim != 1:
            raise ValueError(f"Expected 1D target tensor of shape (batch_size,), got {target.shape}")

        batch_size, num_classes = logits.shape
        if num_classes != self.cost_matrix.shape[0]:
            raise ValueError(
                f"Mismatch: logits num_classes={num_classes} != cost_matrix shape={self.cost_matrix.shape}"
            )

        # 1. Base classification loss
        if self.base_criterion is not None:
            # LabelSmoothingCrossEntropy returns per-batch scalar
            base_loss = self.base_criterion(logits, target)
            if self.reduction == "none":
                # For per-sample breakdown when smoothing > 0:
                log_probs = F.log_softmax(logits, dim=-1)
                nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
                smooth_loss = -log_probs.mean(dim=-1)
                base_loss_per_sample = (1.0 - self.label_smoothing) * nll_loss + self.label_smoothing * smooth_loss
            else:
                base_loss_per_sample = None
        else:
            weight = self.class_weights if hasattr(self, "class_weights") and self.class_weights is not None else None
            if self.reduction == "none":
                base_loss_per_sample = F.cross_entropy(logits, target, weight=weight, reduction="none")
                base_loss = None
            else:
                base_loss = F.cross_entropy(logits, target, weight=weight, reduction=self.reduction)
                base_loss_per_sample = None

        # 2. Expected cost calculation: R_cost = sum_j C[y, j] * p_j
        if self.alpha > 0.0:
            probs = F.softmax(logits, dim=-1)
            # Select cost row for each sample's true class y: shape (B, K)
            sample_costs = self.cost_matrix[target]
            # Expected risk for each sample: shape (B,)
            expected_cost_per_sample = torch.sum(sample_costs * probs, dim=-1)
        else:
            expected_cost_per_sample = torch.zeros(batch_size, device=logits.device, dtype=logits.dtype)

        # 3. Combine losses per reduction mode
        if self.reduction == "none":
            total_loss_per_sample = base_loss_per_sample + self.alpha * expected_cost_per_sample
            return total_loss_per_sample
        elif self.reduction == "mean":
            cost_loss = expected_cost_per_sample.mean()
            return base_loss + self.alpha * cost_loss
        elif self.reduction == "sum":
            cost_loss = expected_cost_per_sample.sum()
            return base_loss + self.alpha * cost_loss
        else:
            raise ValueError(f"Unsupported reduction: {self.reduction}")
